"""Per-frame enhancement pipeline. Device-side (Snapdragon X, Windows on ARM).

This is the runtime skeleton. Each stage wraps one compiled model from
`aihub/export_all.py`. Pre/post-processing details depend on the exact models
chosen after `python -m qai_hub_models.models` — see docs/pipeline.md.

Wiring order: face -> segmentation -> composite -> low-light -> super-res -> out.
"""
from __future__ import annotations

import time
import pathlib
import numpy as np

from ort_qnn import NpuModel

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


class Stage:
    """One NPU model + its pre/post processing."""
    name = "stage"

    def __init__(self, onnx_name: str):
        self.model = NpuModel(MODELS_DIR / onnx_name)

    def process(self, frame: np.ndarray, ctx: dict) -> np.ndarray:
        raise NotImplementedError


class FaceStage(Stage):
    name = "face"
    # detection + landmarks -> ctx["face_box"], ctx["landmarks"]; frame unchanged
    def process(self, frame, ctx):
        # TODO: preprocess -> self.model(...) -> decode boxes/landmarks
        ctx["face_box"] = None
        return frame


class SegmentStage(Stage):
    name = "segment"
    # person segmentation -> ctx["alpha"] (HxW float 0..1)
    def process(self, frame, ctx):
        # TODO: preprocess -> self.model(...) -> alpha matte, resize to frame
        ctx["alpha"] = None
        return frame


class CompositeStage:
    name = "composite"  # CPU/GPU, not NPU
    def __init__(self, mode: str = "blur"):
        self.mode = mode
    def process(self, frame, ctx):
        alpha = ctx.get("alpha")
        if alpha is None:
            return frame
        # TODO: blur or replace background using alpha
        return frame


class LowLightStage(Stage):
    name = "lowlight"
    def process(self, frame, ctx):
        # luma gate: skip if already bright
        if frame.mean() > 110:
            return frame
        # TODO: preprocess -> self.model(...) -> enhanced frame
        return frame


class SuperResStage(Stage):
    name = "superres"
    def __init__(self, onnx_name: str, every_n: int = 1):
        super().__init__(onnx_name)
        self.every_n = every_n
        self._i = 0
        self._last = None
    def process(self, frame, ctx):
        self._i += 1
        if self.every_n > 1 and self._i % self.every_n and self._last is not None:
            return self._last
        # TODO: preprocess -> self.model(...) -> upscaled frame
        self._last = frame
        return frame


class Pipeline:
    def __init__(self, stages: list):
        self.stages = stages
        self.timings: dict[str, float] = {}

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        ctx: dict = {}
        for st in self.stages:
            t0 = time.perf_counter()
            frame = st.process(frame, ctx)
            self.timings[st.name] = (time.perf_counter() - t0) * 1e3
        return frame

    def report(self) -> str:
        total = sum(self.timings.values())
        rows = "  ".join(f"{k}={v:.1f}ms" for k, v in self.timings.items())
        return f"{rows}  | total={total:.1f}ms ({1000/total:.0f} fps)" if total else "no frames"


def build_default() -> Pipeline:
    return Pipeline([
        FaceStage("face.onnx"),
        SegmentStage("segment.onnx"),
        CompositeStage("blur"),
        LowLightStage("lowlight.onnx"),
        SuperResStage("superres.onnx", every_n=1),
    ])


if __name__ == "__main__":
    # smoke test with a synthetic frame once models are downloaded
    pipe = build_default()
    dummy = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    for _ in range(30):
        pipe(dummy)
    print(pipe.report())
