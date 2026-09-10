"""Per-frame orchestration + timing."""
from __future__ import annotations

import time
import numpy as np

from .stages import FaceStage, SegmentStage, LowLightStage, SuperResStage
from .compositor import blur_background, replace_background, AutoFramer, draw_debug

FRAME_BUDGET_MS = 33.0


class Pipeline:
    def __init__(self, prefer: str = "auto", *, bg_mode: str = "blur",
                 bg_image: np.ndarray | None = None, sr_asset: str | None = None,
                 sr_every_n: int = 1, lowlight_onnx: str | None = None,
                 enable=("face", "segment", "lowlight", "superres"),
                 autoframe: bool = True, debug: bool = False):
        self.enable = set(enable)
        self.bg_mode = bg_mode
        self.bg_image = bg_image
        self.debug = debug
        self.face = FaceStage(prefer) if "face" in self.enable else None
        self.segment = SegmentStage(prefer) if "segment" in self.enable else None
        self.lowlight = LowLightStage(lowlight_onnx, prefer) if "lowlight" in self.enable else None
        self.superres = (SuperResStage(sr_asset, prefer, sr_every_n)
                         if ("superres" in self.enable and sr_asset) else
                         SuperResStage(prefer=prefer, every_n=sr_every_n)
                         if "superres" in self.enable else None)
        self.framer = AutoFramer() if autoframe else None
        self.timings: dict[str, float] = {}
        self.providers: dict[str, str] = {}
        for st in (self.face, self.segment, self.lowlight, self.superres):
            if st is None:
                continue
            s = getattr(st, "s", None) or getattr(st, "det", None)
            if s is not None:
                self.providers[st.name] = s.active_provider

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        t = {}
        f = None
        if self.face is not None:
            t0 = time.perf_counter(); f = self.face.run(frame_bgr)
            t["face"] = (time.perf_counter() - t0) * 1e3
        if self.segment is not None:
            t0 = time.perf_counter()
            alpha = self.segment.run(frame_bgr)
            if self.bg_mode == "replace" and self.bg_image is not None:
                frame_bgr = replace_background(frame_bgr, alpha, self.bg_image)
            else:
                frame_bgr = blur_background(frame_bgr, alpha)
            t["segment"] = (time.perf_counter() - t0) * 1e3
        if self.lowlight is not None:
            t0 = time.perf_counter(); frame_bgr = self.lowlight.run(frame_bgr)
            t["lowlight"] = (time.perf_counter() - t0) * 1e3
        if self.framer is not None:
            t0 = time.perf_counter(); frame_bgr = self.framer(frame_bgr, f)
            t["autoframe"] = (time.perf_counter() - t0) * 1e3
        if self.superres is not None:
            t0 = time.perf_counter(); frame_bgr = self.superres.run(frame_bgr)
            t["superres"] = (time.perf_counter() - t0) * 1e3
        if self.debug:
            frame_bgr = draw_debug(frame_bgr, f)
        self.timings = t
        return frame_bgr

    def report(self) -> str:
        tot = sum(self.timings.values())
        parts = "  ".join(f"{k}={v:.1f}" for k, v in self.timings.items())
        fps = 1000.0 / tot if tot else 0.0
        verdict = "OK" if tot <= FRAME_BUDGET_MS else "OVER"
        return f"{parts}  | total={tot:.1f}ms {fps:.0f}fps [{verdict} vs {FRAME_BUDGET_MS:.0f}ms]"
