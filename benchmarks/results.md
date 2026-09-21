# Benchmarks — Snapdragon X2 Elite CRD (Qualcomm AI Hub device farm)

Runtime target: `onnx` → ONNX Runtime + QNN Execution Provider (Hexagon NPU / HTP).
Toolchain reported by AI Hub: `onnx_runtime 1.27.1`, `qairt 2.45.0`.
Budget: sum of stage latencies **< 33 ms** for 30 fps, **100% NPU** per model (no CPU fallback).

## Summary (2026-09-08 / 09-10)

### At each model's default resolution

| Stage | Model | float (ms) | w8a8 / INT8 (ms) | NPU | CPU ops |
|-------|-------|-----------:|-----------------:|-----|--------:|
| 1 face detect + landmarks | `mediapipe_face` (2 comp, 256²+192²) | **0.6** | pending¹ | 100% | 0 |
| 2 person segmentation     | `mediapipe_selfie` (256²)            | **0.4** | **0.2** | 100% | 0 |
| 3 low-light enhancement   | `zero_dce` (BYO, 256²)               | **1.4** | — | 100% (56 layers) | 0 |
| 4 super-resolution        | `quicksrnetmedium` (128²→4×→512²)    | **0.5** | pending² | 100% | 0 |
| **Pipeline total**        |                                     | **2.9** | — | **100%** | **0** |

### At realistic capture resolution (video-call scenario)

| Stage | Config | float (ms) | NPU |
|-------|--------|-----------:|-----|
| face | fixed 256²/192² input | 0.6 | 100% |
| segment | 256² (matte upsampled to frame) — already representative | 0.4 | 100% |
| low-light | Zero-DCE (BYO), 256², 56 layers, ~4 MB inference mem | **1.4** | 100% |
| super-resolution | **640×360 → 2× → 1280×720** (upscale a 360p stream) | **3.4** | 100% (19 ops) |
| super-resolution | 320×180 → 4× → 1280×720 (upscale a 180p stream) | **1.5** | 100% (19 ops), 33 MB peak |
| **realistic pipeline** | face + segment + low-light + SR@720p(2×) | **≈5.8** | **100%** |
| **realistic pipeline** | face + segment + low-light + SR@720p(4×) | **≈3.9** | **100%** |

**Sum of separately profiled models ≈ 3.9–5.8 ms vs a 33 ms budget (30 fps).** Every stage
runs 100% on the NPU with zero CPU ops. This sum excludes pre/post-processing, compositing
and capture and is **not** a measured end-to-end frame time. Landmark on-device-vs-CPU
output PSNR ≈ 75 dB (float).

Profile jobs for the realistic-resolution super-resolution runs:
2× @ 640×360: [j57erdeqp](https://workbench.aihub.qualcomm.com/jobs/j57erdeqp/) ·
4× @ 320×180: [jgk2xy0og](https://workbench.aihub.qualcomm.com/jobs/jgk2xy0og/).

### NPU vs CPU: not claimed

An earlier draft compared the NPU against CPU timings taken on a shared, CPU-throttled
cluster node (the same node took ~2 s to run the 79k-parameter Zero-DCE at 256²).
That comparison is not meaningful for a Snapdragon and has been removed rather than
quoted. A valid comparison needs the CPU and NPU measured **on the same device**.

Notes:
- The default-resolution table above uses each model's default input size; the
  realistic-resolution table re-exports super-resolution at 720p output (segmentation
  and face are fixed-size networks, so their defaults are already representative).
- Stage 3 (low-light) is Zero-DCE, bring-your-own — see `aihub/byo_lowlight.py`; profiled
  at 1.40 ms (job jpxlo71lp). Its weights are CC BY-NC 4.0 and are not redistributed here.
- ¹ `mediapipe_face` w8a8 needs the gated Kaggle `human-face` calibration set:
  `kaggle datasets download ashwingupta3012/human-face` then
  `python -m qai_hub_models.scripts.configure_dataset --class qai_hub_models.models.mediapipe_face.dataset.HumanFacesDataset --files <zip>`, then re-run `--precision w8a8 --only face`.
- ² `quicksrnetmedium` w8a8 calibration downloads BSDS300 from `www2.eecs.berkeley.edu`
  which now returns 403 (dead upstream). Needs a BSDS300 mirror + `configure_dataset`,
  or switch the SR calibration dataset. float SR (0.5 ms) is already well within budget.
- ³ Zero-DCE (bring-your-own, `aihub/byo_lowlight.py`) — AI Hub compile+profile in flight;
  number lands here when the job completes.

---

## Run 2026-09-08 22:50 — float — all stages

| Stage | Model | Latency (ms) | Comp | NPU % | CPU ops | Status |
|-------|-------|-------------:|-----:|------:|--------:|--------|
| face | `mediapipe_face` | 0.6 | 2 | 100.0 | 0 | ok |
| segment | `mediapipe_selfie` | 0.4 | 1 | 100.0 | 0 | ok |
| superres | `quicksrnetmedium` | 0.5 | 1 | 100.0 | 0 | ok |

Sum: **1.50 ms** vs 33 ms → PASS (+31.5 ms headroom)

- face jobs: https://workbench.aihub.qualcomm.com/jobs/jgj7mx7xg/ https://workbench.aihub.qualcomm.com/jobs/jpez19z1p/ https://workbench.aihub.qualcomm.com/jobs/j5w7vol6g/ https://workbench.aihub.qualcomm.com/jobs/jg9m1vzl5/ https://workbench.aihub.qualcomm.com/jobs/jgd39wdep/ https://workbench.aihub.qualcomm.com/jobs/j574wzel5/
- segment jobs: https://workbench.aihub.qualcomm.com/jobs/jp2w68drp/ https://workbench.aihub.qualcomm.com/jobs/jp0jqy99g/ https://workbench.aihub.qualcomm.com/jobs/jp8x9orkg/
- superres jobs: https://workbench.aihub.qualcomm.com/jobs/jgd39wyrp/ https://workbench.aihub.qualcomm.com/jobs/j574wz1v5/ https://workbench.aihub.qualcomm.com/jobs/jp41oq68p/

## Run 2026-09-08 23:01 — w8a8 — partial (segment only)

| Stage | Model | Latency (ms) | Comp | NPU % | CPU ops | Status |
|-------|-------|-------------:|-----:|------:|--------:|--------|
| face | `mediapipe_face` | — | — | — | — | FAIL (calibration dataset) |
| segment | `mediapipe_selfie` | 0.2 | 1 | 100.0 | 0 | ok |
| superres | `quicksrnetmedium` | — | — | — | — | FAIL (BSDS300 403) |

segment INT8 = 0.2 ms → **2× faster than float** (0.4 ms), still 100% NPU.

- segment jobs: https://workbench.aihub.qualcomm.com/jobs/jp41oqz1p/ https://workbench.aihub.qualcomm.com/jobs/jglxznnmg/ https://workbench.aihub.qualcomm.com/jobs/j567j66yp/
