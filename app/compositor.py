"""Background compositing + auto-framing. CPU/GPU work, not NPU."""
from __future__ import annotations

import numpy as np

from . import imgops


def blur_background(frame_bgr: np.ndarray, alpha: np.ndarray, ksize: int = 35) -> np.ndarray:
    """alpha in [0,1], 1 = foreground person. Gaussian-blur the background."""
    a = np.clip(alpha, 0, 1)[..., None]
    bg = imgops.gaussian_blur(frame_bgr, ksize)
    return (frame_bgr * a + bg * (1 - a)).astype(np.uint8)


def replace_background(frame_bgr: np.ndarray, alpha: np.ndarray,
                       bg_bgr: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0, 1)[..., None]
    bg = imgops.resize(bg_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return (frame_bgr * a + bg * (1 - a)).astype(np.uint8)


class AutoFramer:
    """Smoothly recenters/zooms on the detected face. EMA to avoid jitter."""

    def __init__(self, zoom: float = 1.6, smooth: float = 0.12):
        self.zoom = zoom
        self.smooth = smooth
        self._c: np.ndarray | None = None

    def __call__(self, frame_bgr: np.ndarray, face: dict | None) -> np.ndarray:
        fh, fw = frame_bgr.shape[:2]
        if face is None:
            target = np.array([fw / 2, fh / 2], np.float32)
        else:
            x0, y0, x1, y1 = face["box"]
            target = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        self._c = target if self._c is None else \
            (1 - self.smooth) * self._c + self.smooth * target
        cw, ch = fw / self.zoom, fh / self.zoom
        cx = np.clip(self._c[0], cw / 2, fw - cw / 2)
        cy = np.clip(self._c[1], ch / 2, fh - ch / 2)
        x0 = int(cx - cw / 2); y0 = int(cy - ch / 2)
        crop = frame_bgr[y0:y0 + int(ch), x0:x0 + int(cw)]
        return imgops.resize(crop, (fw, fh))


def draw_debug(frame_bgr: np.ndarray, face: dict | None) -> np.ndarray:
    out = frame_bgr.copy()
    if face is not None:
        x0, y0, x1, y1 = face["box"]
        imgops.draw_rect(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        imgops.put_text(out, f"{face['score']:.2f}", (x0, max(y0 - 6, 12)), (0, 255, 0))
        if face.get("landmarks") is not None:
            for px, py in face["landmarks"][::8].astype(int):
                imgops.draw_circle(out, (px, py), 1, (255, 128, 0), filled=True)
    return out
