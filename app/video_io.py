"""Frame sources and sinks: webcam / video file / image loop  ->  mp4 / virtual cam / frames dir."""
from __future__ import annotations

import pathlib
import numpy as np
import cv2


def open_source(spec: str, size: tuple[int, int] | None = None):
    """`spec`: integer webcam index, a video path, or an image path (looped).
    Yields BGR uint8 frames."""
    p = pathlib.Path(spec)
    if spec.isdigit():
        cap = cv2.VideoCapture(int(spec))
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr
        cap.release()
    elif p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        if size:
            img = cv2.resize(img, size)
        while True:
            yield img.copy()
    else:
        cap = cv2.VideoCapture(str(p))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield (cv2.resize(fr, size) if size else fr)
        cap.release()


class Sink:
    """`spec`: 'virtualcam', a .mp4 path, or a directory (writes frame_00001.png)."""

    def __init__(self, spec: str, fps: float = 30.0):
        self.spec = spec
        self.fps = fps
        self._w = None
        self._vc = None
        self._n = 0
        self._dir = None
        if spec == "virtualcam":
            import pyvirtualcam  # optional; raises here if not installed
            self._pvc_mod = pyvirtualcam
        elif spec.endswith(".mp4") or spec.endswith(".avi"):
            self._path = spec
        else:
            self._dir = pathlib.Path(spec)
            self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self.spec == "virtualcam":
            if self._vc is None:
                self._vc = self._pvc_mod.Camera(width=w, height=h, fps=self.fps)
            self._vc.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            self._vc.sleep_until_next_frame()
        elif self._dir is not None:
            self._n += 1
            cv2.imwrite(str(self._dir / f"frame_{self._n:05d}.png"), frame_bgr)
        else:
            if self._w is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._w = cv2.VideoWriter(self._path, fourcc, self.fps, (w, h))
            self._w.write(frame_bgr)

    def close(self) -> None:
        if self._w is not None:
            self._w.release()
        if self._vc is not None:
            self._vc.close()
