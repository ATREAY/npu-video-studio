"""Pipeline stages. Each wraps one compiled ONNX model with numpy/cv2 pre/post.

Stage order in the pipeline: face -> segment -> composite -> lowlight -> superres.
All tensors are NCHW float32, RGB, value range [0, 1] unless noted.
"""
from __future__ import annotations

import pathlib
import numpy as np

from . import imgops
from .session import Session

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "export_assets"


def _to_nchw(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """BGR uint8 HxWx3 -> 1x3xHxW float32 RGB [0,1] at `size` (w, h)."""
    rgb = imgops.bgr_to_rgb(imgops.resize(bgr, size))
    return rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0


def _to_bgr(nchw: np.ndarray) -> np.ndarray:
    """1x3xHxW float32 RGB [0,1] -> BGR uint8 HxWx3."""
    rgb = np.clip(nchw[0].transpose(1, 2, 0), 0, 1) * 255.0
    return imgops.rgb_to_bgr(rgb.astype(np.uint8))


# --------------------------------------------------------------------------- #
class SegmentStage:
    """Person segmentation -> soft alpha matte (float 0..1, frame-sized)."""
    name = "segment"

    def __init__(self, prefer: str = "auto"):
        self.s = Session(ASSETS / "mediapipe_selfie-onnx-float" / "mediapipe_selfie.onnx", prefer)
        _, _, self.h, self.w = self.s.input_shapes["image"]

    def run(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = _to_nchw(frame_bgr, (self.w, self.h))
        mask = self.s.run(x)[0][0, 0]                     # HxW
        if mask.min() < 0 or mask.max() > 1:             # logits -> prob
            mask = 1.0 / (1.0 + np.exp(-mask))
        fh, fw = frame_bgr.shape[:2]
        return imgops.resize(mask, (fw, fh))

    @property
    def last_ms(self) -> float:
        return self.s.last_ms


# --------------------------------------------------------------------------- #
def _find_onnx(asset: str) -> pathlib.Path:
    """Locate the .onnx inside an export_assets subdir (flat or job_*/model.onnx)."""
    d = ASSETS / asset
    hits = sorted(d.rglob("*.onnx"))
    if not hits:
        raise FileNotFoundError(f"no .onnx under {d}")
    # prefer a top-level file, else the nested optimized one
    return next((h for h in hits if h.parent == d), hits[0])


class SuperResStage:
    """Upscale a frame. Fixed input size + scale come from the compiled asset.

    asset "quicksrnetmedium-onnx-float"       -> 128->512  (4x, toy)
    asset "quicksrnetmedium-onnx-float-2x360" -> 640x360 -> 1280x720 (2x, realistic)
    """
    name = "superres"

    def __init__(self, asset: str = "quicksrnetmedium-onnx-float", prefer: str = "auto",
                 every_n: int = 1):
        self.s = Session(_find_onnx(asset), prefer)
        in_name = self.s.input_names[0]
        _, _, self.h, self.w = self.s.input_shapes[in_name]
        oshape = self.s.sess.get_outputs()[0].shape
        oh = oshape[2] if isinstance(oshape[2], int) else 0
        self.scale = (oh // self.h) if oh else 4
        self.asset = asset
        self.every_n = max(1, every_n)
        self._i = 0
        self._cache: np.ndarray | None = None

    def run(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._i += 1
        if self.every_n > 1 and self._i % self.every_n and self._cache is not None:
            return self._cache
        x = _to_nchw(frame_bgr, (self.w, self.h))
        y = self.s.run(x)[0]
        out = _to_bgr(y)
        self._cache = out
        return out

    @property
    def last_ms(self) -> float:
        return self.s.last_ms


# --------------------------------------------------------------------------- #
class LowLightStage:
    """Low-light lift. CLAHE fallback now; swap for Zero-DCE (byo_lowlight.py) later.

    `model_onnx` (optional): path to a compiled Zero-DCE .onnx (in [0,1] RGB NCHW,
    out same). If given and loadable, it is used instead of CLAHE.
    """
    name = "lowlight"

    # prefer the AI-Hub-compiled asset, then the plain torch export
    _DEFAULTS = [
        ASSETS / "zero_dce-onnx-compiled",
        ASSETS / "zero_dce" / "zero_dce.onnx",
    ]

    def __init__(self, model_onnx: str | None = None, prefer: str = "auto",
                 luma_gate: float = 110.0):
        self.luma_gate = luma_gate
        self.s: Session | None = None
        path = model_onnx
        if path is None:
            for d in self._DEFAULTS:
                if d.is_dir():
                    hits = sorted(d.rglob("*.onnx"))
                    if hits:
                        path = str(hits[0]); break
                elif d.exists():
                    path = str(d); break
        if path and pathlib.Path(path).exists():
            self.s = Session(path, prefer)
            _, _, self.h, self.w = self.s.input_shapes[self.s.input_names[0]]
        self.backend = "zero-dce" if self.s is not None else "clahe"
        self._clahe = imgops.Clahe(clip_limit=2.0, tile=(8, 8))
        self.last_ms = 0.0

    def run(self, frame_bgr: np.ndarray) -> np.ndarray:
        import time
        if frame_bgr.mean() > self.luma_gate:            # already bright: skip
            self.last_ms = 0.0
            return frame_bgr
        t0 = time.perf_counter()
        if self.s is not None:
            x = _to_nchw(frame_bgr, (self.w, self.h))
            y = self.s.run(x)[0]
            fh, fw = frame_bgr.shape[:2]
            out = imgops.resize(_to_bgr(y), (fw, fh))
            self.last_ms = self.s.last_ms
            return out
        out = self._clahe.apply_bgr(frame_bgr)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return out


# --------------------------------------------------------------------------- #
# MediaPipe BlazeFace back-camera anchor grid, normalised (x_c, y_c). 16x16x2 + 8x8x6.
_ANCHOR_FILES = [
    pathlib.Path.home() / ".qaihm/external_repos/shared/mediapipe/mediapipe",
    pathlib.Path.home() / ".qaihm/qai-hub-models/models/mediapipe_pytorch/v1/zmurez_MediaPipePyTorch_git",
]
# box expansion so the crop encloses the whole face (DETECT_DSCALE in the reference)
_DETECT_DSCALE = 1.1


def _load_face_anchors() -> np.ndarray:
    for base in _ANCHOR_FILES:
        for f in base.rglob("anchors_face_back.npy"):
            return np.load(f)[:, :2]                       # (896, 2) centres
    # fallback: reconstruct the grid (verified identical to the file)
    out = []
    for g, k in ((16, 2), (8, 6)):
        cy, cx = np.mgrid[0:g, 0:g]
        c = np.stack([(cx + 0.5) / g, (cy + 0.5) / g], -1).reshape(-1, 2)
        out.append(np.repeat(c, k, axis=0))
    return np.concatenate(out, 0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.3) -> list[int]:
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes.T
    area = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class FaceStage:
    """Face detection (BlazeFace-back) + 468 landmarks. Drives framing / eye-contact.

    Decode validated bit-exact against qai_hub_models' reference. Returns the
    highest-scoring face as a dict {box (x0,y0,x1,y1 px), landmarks (468,2 px)|None,
    eyes ((lx,ly),(rx,ry) px), score}, or None. `all_boxes` also kept for multi-face.
    """
    name = "face"

    def __init__(self, prefer: str = "auto", min_score: float = 0.6, nms_iou: float = 0.3):
        base = ASSETS / "mediapipe_face-onnx-float"
        self.det = Session(base / "face_detector.onnx", prefer)
        self.lm = Session(base / "face_landmark_detector.onnx", prefer)
        _, _, self.dh, self.dw = self.det.input_shapes["image"]
        _, _, self.lh, self.lw = self.lm.input_shapes["image"]
        self.anchors = _load_face_anchors()               # (896, 2), normalised
        self.min_score = min_score
        self.nms_iou = nms_iou
        self.last_ms = 0.0

    def _decode(self, coords: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """coords (N,16) raw px-offsets; keep indices -> (M,16) xyxy box + 6 keypts px, normalised 0..1."""
        a = self.anchors[keep]                            # (M, 2)
        cx = coords[keep, 0] / self.dw + a[:, 0]
        cy = coords[keep, 1] / self.dh + a[:, 1]
        w = (coords[keep, 2] / self.dw) * _DETECT_DSCALE
        h = (coords[keep, 3] / self.dh) * _DETECT_DSCALE
        out = np.zeros((len(keep), 16), np.float32)
        out[:, 0], out[:, 1] = cx - w / 2, cy - h / 2
        out[:, 2], out[:, 3] = cx + w / 2, cy + h / 2
        for j in range(6):                                # keypoints -> normalised px
            out[:, 4 + 2 * j] = coords[keep, 4 + 2 * j] / self.dw + a[:, 0]
            out[:, 5 + 2 * j] = coords[keep, 5 + 2 * j] / self.dh + a[:, 1]
        return out

    def run(self, frame_bgr: np.ndarray) -> dict | None:
        import time
        t0 = time.perf_counter()
        fh, fw = frame_bgr.shape[:2]
        x = _to_nchw(frame_bgr, (self.dw, self.dh))
        c1, c2, s1, s2 = self.det.run(x)
        coords = np.concatenate([c1[0], c2[0]], 0)        # (896, 16)
        raw = np.clip(np.concatenate([s1[0, :, 0], s2[0, :, 0]], 0), -30.0, 30.0)
        scores = 1.0 / (1.0 + np.exp(-raw))
        cand = np.where(scores >= self.min_score)[0]
        if cand.size == 0:
            self.last_ms = (time.perf_counter() - t0) * 1e3
            return None
        dec = self._decode(coords, cand)                  # normalised
        px = dec.copy()
        px[:, 0::2] *= fw
        px[:, 1::2] *= fh
        keep = _nms(px[:, :4], scores[cand], self.nms_iou)
        all_boxes = np.clip(px[keep, :4], [0, 0, 0, 0], [fw, fh, fw, fh]).astype(int)
        best = keep[0]
        box = all_boxes[0]
        eyes = ((float(px[best, 4]), float(px[best, 5])),
                (float(px[best, 6]), float(px[best, 7])))

        # landmark model on the square face crop
        cx0, cy0, cx1, cy1 = box
        side = int(max(cx1 - cx0, cy1 - cy0, 8))
        mx, my = (cx0 + cx1) // 2, (cy0 + cy1) // 2
        sx0 = int(np.clip(mx - side // 2, 0, max(fw - side, 0)))
        sy0 = int(np.clip(my - side // 2, 0, max(fh - side, 0)))
        crop = frame_bgr[sy0:sy0 + side, sx0:sx0 + side]
        lms = None
        if crop.size and min(crop.shape[:2]) > 4:
            lx = _to_nchw(crop, (self.lw, self.lh))
            lscore, lcoords = self.lm.run(lx)
            if float(np.ravel(lscore)[0]) > 0.5:
                # landmark model emits coords normalised to [0,1] over the crop
                pts = lcoords[0][:, :2].astype(np.float32).copy()
                pts[:, 0] = pts[:, 0] * crop.shape[1] + sx0
                pts[:, 1] = pts[:, 1] * crop.shape[0] + sy0
                lms = pts
            # NOTE: axis-aligned square crop; the reference uses a rotation-aware
            # affine crop from the eye keypoints. Fine for an upright call subject.
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return {"box": box, "all_boxes": all_boxes, "landmarks": lms,
                "eyes": eyes, "score": float(scores[cand][keep[0]])}
