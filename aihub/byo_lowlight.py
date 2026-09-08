"""Bring-your-own low-light enhancement model -> compile + profile on AI Hub.

The zoo has no low-light model, so we supply Zero-DCE (Zhang et al., CVPR 2020):
~79k params, no reference image needed, just predicts per-pixel tone curves.
Tiny and fully convolutional -> good NPU fit.

This script:
  1. builds the Zero-DCE network in PyTorch (architecture is short; weights optional)
  2. traces it to ONNX at the pipeline's working resolution
  3. submits a compile job + profile job to AI Hub for the target device(s)
  4. writes latency / NPU-utilisation to benchmarks/results.md

Run:  python aihub/byo_lowlight.py --device "Snapdragon X2 Elite CRD" [--weights zero_dce.pth]

If --weights is omitted the model runs with random init (fine for *latency* profiling;
load real weights before measuring quality). Pretrained Zero-DCE weights:
  https://github.com/Li-Chongyi/Zero-DCE  (Zero-DCE_code/snapshots/Epoch99.pth)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402


class ZeroDCE(nn.Module):
    """DCE-Net: 7 conv layers, symmetric skips, outputs 24 curve maps (8 iters x RGB)."""

    def __init__(self, ch: int = 32, n_iter: int = 8):
        super().__init__()
        self.n_iter = n_iter
        self.relu = nn.ReLU(inplace=True)
        self.e1 = nn.Conv2d(3, ch, 3, 1, 1)
        self.e2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.e3 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.e4 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.d3 = nn.Conv2d(ch * 2, ch, 3, 1, 1)
        self.d2 = nn.Conv2d(ch * 2, ch, 3, 1, 1)
        self.d1 = nn.Conv2d(ch * 2, 3 * n_iter, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e1(x))
        x2 = self.relu(self.e2(x1))
        x3 = self.relu(self.e3(x2))
        x4 = self.relu(self.e4(x3))
        x5 = self.relu(self.d3(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.d2(torch.cat([x2, x5], 1)))
        curves = torch.tanh(self.d1(torch.cat([x1, x6], 1)))
        out = x
        for i in range(self.n_iter):
            r = curves[:, 3 * i:3 * i + 3, :, :]
            out = out + r * (out * out - out)
        return torch.clamp(out, 0.0, 1.0)


def to_onnx(weights: str | None, h: int, w: int, out: pathlib.Path) -> pathlib.Path:
    net = ZeroDCE().eval()
    if weights:
        sd = torch.load(weights, map_location="cpu")
        net.load_state_dict(sd.get("state_dict", sd), strict=False)
        print(f"loaded weights: {weights}")
    else:
        print("WARNING: random init — latency-only, quality not meaningful")
    dummy = torch.rand(1, 3, h, w)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, dummy, out.as_posix(), input_names=["image"],
                      output_names=["enhanced"], opset_version=17,
                      dynamic_axes=None)
    print(f"onnx -> {out}  ({out.stat().st_size / 1e3:.0f} KB)")
    return out


def profile_on_hub(onnx_path: pathlib.Path, device_name: str, runtime: str) -> None:
    import qai_hub as hub

    dev = hub.Device(device_name)
    target = {
        "onnx": hub.client.SourceModelType.ONNX,
    }
    print(f"submitting compile job: {onnx_path.name} -> {device_name} ({runtime})")
    compile_job = hub.submit_compile_job(
        model=onnx_path.as_posix(),
        device=dev,
        options=f"--target_runtime {runtime}",
        name=f"lowlight-zerodce-{runtime}",
    )
    compiled = compile_job.get_target_model()
    print(f"  compile job: {compile_job.url}")

    profile_job = hub.submit_profile_job(
        model=compiled, device=dev, name=f"lowlight-zerodce-{runtime}-profile"
    )
    print(f"  profile job: {profile_job.url}")
    prof = profile_job.download_profile()
    exe = prof["execution_summary"]
    lat_us = exe.get("estimated_inference_time", 0)
    print(f"  estimated inference time: {lat_us / 1000:.2f} ms")
    print(f"  compute unit breakdown:  {exe.get('compute_unit_execution_time', {})}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--runtime", default=config.TARGET_RUNTIME)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--skip-hub", action="store_true", help="only export onnx")
    args = ap.parse_args()

    onnx_path = PROJ / "models" / "src" / "zero_dce.onnx"
    to_onnx(args.weights, args.height, args.width, onnx_path)
    if not args.skip_hub:
        profile_on_hub(onnx_path, args.device, args.runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
