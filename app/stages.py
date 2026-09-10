"""Pipeline stages. Each wraps one compiled ONNX model with numpy/cv2 pre/post.

Stage order in the pipeline: face -> segment -> composite -> lowlight -> superres.
All tensors are NCHW float32, RGB, value range [0, 1] unless noted.
"""
from __future__ import annotations

import pathlib
import numpy as np
import cv2

from .session import Session

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "export_assets"


def _to_nchw(bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """BGR uint8 HxWx3 -> 1x3xHxW float32 RGB [0,1] at `size` (w, h)."""
    rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR),
                       cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0


def _to_bgr(nchw: np.ndarray) -> np.ndarray:
    """1x3xHxW float32 RGB [0,1] -> BGR uint8 HxWx3."""
    rgb = np.clip(nchw[0].transpose(1, 2, 0), 0, 1) * 255.0
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


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
        return cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_LINEAR)

    @property
    def last_ms(self) -> float:
        return self.s.last_ms


# --------------------------------------------------------------------------- #
class SuperResStage:
    """Upscale a low-res frame. Model is fixed-scale (from the compiled asset)."""
    name = "superres"

    def __init__(self, asset: str = "quicksrnetmedium-onnx-float", prefer: str = "auto",
                 every_n: int = 1):
        self.s = Session(ASSETS / asset / "quicksrnetmedium.onnx", prefer)
        _, _, self.h, self.w = self.s.input_shapes["image"]
        _, _, oh, ow = [d if isinstance(d, int) else 0
                        for d in self.s.sess.get_outputs()[0].shape]
        self.scale = (oh // self.h) if oh else 4
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

    def __init__(self, model_onnx: str | None = None, prefer: str = "auto",
                 luma_gate: float = 110.0):
        self.luma_gate = luma_gate
        self.s: Session | None = None
        if model_onnx and pathlib.Path(model_onnx).exists():
            self.s = Session(model_onnx, prefer)
            _, _, self.h, self.w = self.s.input_shapes[self.s.input_names[0]]
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
            out = cv2.resize(_to_bgr(y), (fw, fh))
            self.last_ms = self.s.last_ms
            return out
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return out


# --------------------------------------------------------------------------- #
def _ssd_anchors(input_size: int = 256,
                 strides: tuple[int, ...] = (16, 32),
                 counts: tuple[int, ...] = (2, 6)) -> np.ndarray:
    """MediaPipe BlazeFace back-camera anchor grid -> (N, 2) normalised centres."""
    out = []
    for stride, k in zip(strides, counts):
        g = input_size // stride
        cy, cx = np.mgrid[0:g, 0:g]
        cx = (cx + 0.5) / g
        cy = (cy + 0.5) / g
        centres = np.stack([cx, cy], -1).reshape(-1, 2)
        out.append(np.repeat(centres, k, axis=0))
    return np.concatenate(out, 0)                         # (896, 2)


class FaceStage:
    """Face detection + 468 landmarks. Drives auto-framing / eye-contact.

    Single-face (largest foreground subject) — matches the video-call use case.
    Multi-face + full NMS: TODO. Returns dict with 'box' (x0,y0,x1,y1 px),
    'landmarks' (468,2 px) or None.
    """
    name = "face"

    def __init__(self, prefer: str = "auto", min_score: float = 0.6):
        base = ASSETS / "mediapipe_face-onnx-float"
        self.det = Session(base / "face_detector.onnx", prefer)
        self.lm = Session(base / "face_landmark_detector.onnx", prefer)
        _, _, self.dh, self.dw = self.det.input_shapes["image"]
        _, _, self.lh, self.lw = self.lm.input_shapes["image"]
        self.anchors = _ssd_anchors(self.dw)
        self.min_score = min_score
        self.last_ms = 0.0

    def run(self, frame_bgr: np.ndarray) -> dict | None:
        import time
        t0 = time.perf_counter()
        fh, fw = frame_bgr.shape[:2]
        x = _to_nchw(frame_bgr, (self.dw, self.dh))
        c1, c2, s1, s2 = self.det.run(x)
        coords = np.concatenate([c1[0], c2[0]], 0)        # (896, 16)
        raw = np.clip(np.concatenate([s1[0, :, 0], s2[0, :, 0]], 0), -30.0, 30.0)
        scores = 1.0 / (1.0 + np.exp(-raw))
        i = int(scores.argmax())
        if scores[i] < self.min_score:
            self.last_ms = (time.perf_counter() - t0) * 1e3
            return None
        ax, ay = self.anchors[i]
        # raw coords are in input-pixel space, centred on the anchor
        cxp = coords[i, 0] / self.dw + ax
        cyp = coords[i, 1] / self.dh + ay
        wp = abs(coords[i, 2]) / self.dw
        hp = abs(coords[i, 3]) / self.dh
        x0, y0 = (cxp - wp / 2) * fw, (cyp - hp / 2) * fh
        x1, y1 = (cxp + wp / 2) * fw, (cyp + hp / 2) * fh
        box = np.clip([x0, y0, x1, y1], [0, 0, 0, 0], [fw, fh, fw, fh]).astype(int)

        # landmark model on the (square-padded) face crop
        cx0, cy0, cx1, cy1 = box
        side = max(cx1 - cx0, cy1 - cy0, 8)
        mx, my = (cx0 + cx1) // 2, (cy0 + cy1) // 2
        sx0, sy0 = max(mx - side // 2, 0), max(my - side // 2, 0)
        crop = frame_bgr[sy0:sy0 + side, sx0:sx0 + side]
        lms = None
        if crop.size:
            lx = _to_nchw(crop, (self.lw, self.lh))
            lscore, lcoords = self.lm.run(lx)
            if lscore.reshape(-1)[0] > 0:
                pts = lcoords[0][:, :2].copy()
                pts[:, 0] = pts[:, 0] / self.lw * crop.shape[1] + sx0
                pts[:, 1] = pts[:, 1] / self.lh * crop.shape[0] + sy0
                lms = pts
        self.last_ms = (time.perf_counter() - t0) * 1e3
        return {"box": box, "landmarks": lms, "score": float(scores[i])}
