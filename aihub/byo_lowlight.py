"""Bring-your-own low-light model: Zero-DCE (Guo et al., CVPR 2020).

The AI Hub zoo has no low-light model. Zero-DCE's DCE-Net is 7 conv layers (~79k
params), reference-free, fully convolutional -> excellent NPU fit.

  python aihub/byo_lowlight.py --onnx-only          # -> export_assets/zero_dce/zero_dce.onnx
  python aihub/byo_lowlight.py --device "Snapdragon X2 Elite CRD"   # + compile/profile on AI Hub

Pretrained weights (auto-downloaded): Li-Chongyi/Zero-DCE  Epoch99.pth
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

WEIGHTS_URL = ("https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/"
               "Zero-DCE_code/snapshots/Epoch99.pth")
OUT_DIR = PROJ / "export_assets" / "zero_dce"


class ZeroDCE(nn.Module):
    """DCE-Net — matches Li-Chongyi/Zero-DCE `enhance_net_nopool` exactly."""

    def __init__(self, nf: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        for i in range(8):                               # 8 curve-map iterations
            r = x_r[:, 3 * i:3 * i + 3, :, :]            # slice, not Split op
            x = x + r * (x * x - x)
        return torch.clamp(x, 0.0, 1.0)


def build(weights: str | None) -> ZeroDCE:
    net = ZeroDCE().eval()
    if weights is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        weights = str(OUT_DIR / "Epoch99.pth")
        if not pathlib.Path(weights).exists():
            print(f"downloading Zero-DCE weights -> {weights}")
            urllib.request.urlretrieve(WEIGHTS_URL, weights)
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    net.load_state_dict(sd)
    print(f"loaded weights: {weights}")
    return net


def to_onnx(net: ZeroDCE, h: int, w: int, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, h, w)
    # dynamo=False -> the legacy TorchScript exporter: compact graph, no Split-op /
    # opset-conversion pathologies. Zero-DCE predicts smooth tone curves so a small
    # fixed input (they upsample fine) keeps CPU-fallback cost low.
    torch.onnx.export(net, dummy, path.as_posix(), input_names=["image"],
                      output_names=["enhanced"], opset_version=17, dynamo=False)
    print(f"onnx -> {path}  ({path.stat().st_size / 1e3:.0f} KB)  input {h}x{w}")
    return path


def profile_on_hub(onnx_path: pathlib.Path, device: str, runtime: str) -> None:
    import qai_hub as hub
    dev = hub.Device(device)
    cj = hub.submit_compile_job(model=onnx_path.as_posix(), device=dev,
                                options=f"--target_runtime {runtime}",
                                name="lowlight-zerodce")
    print("compile:", cj.url)
    pj = hub.submit_profile_job(model=cj.get_target_model(), device=dev,
                                name="lowlight-zerodce-profile")
    print("profile:", pj.url)
    prof = pj.download_profile()
    exe = prof["execution_summary"]
    print(f"estimated inference time: {exe.get('estimated_inference_time', 0)/1000:.2f} ms")
    print(f"compute units: {exe.get('layer_counts_by_compute_unit', exe)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=config.DEVICE)
    ap.add_argument("--runtime", default=config.TARGET_RUNTIME)
    ap.add_argument("--weights", default=None, help="path to Epoch99.pth (auto-download if omitted)")
    ap.add_argument("--height", type=int, default=256, help="fixed input H (curves upsample)")
    ap.add_argument("--width", type=int, default=256, help="fixed input W")
    ap.add_argument("--onnx-only", action="store_true", help="export ONNX, skip AI Hub")
    args = ap.parse_args()

    net = build(args.weights)
    onnx_path = to_onnx(net, args.height, args.width, OUT_DIR / "zero_dce.onnx")

    # quick numeric self-check: a dark input should get brighter
    with torch.no_grad():
        dark = torch.rand(1, 3, args.height, args.width) * 0.15
        out = net(dark)
    print(f"self-check: mean {dark.mean():.3f} -> {out.mean():.3f} "
          f"({'brighter OK' if out.mean() > dark.mean() else 'NOT brighter?!'})")

    if not args.onnx_only:
        profile_on_hub(onnx_path, args.device, args.runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
