# Benchmarks — Snapdragon X2 Elite CRD (Qualcomm AI Hub device farm)

Runtime target: `onnx` → ONNX Runtime + QNN Execution Provider (Hexagon NPU / HTP).
Toolchain reported by AI Hub: `onnx_runtime 1.27.1`, `qairt 2.45.0`.
Budget: sum of stage latencies **< 33 ms** for 30 fps, **100% NPU** per model (no CPU fallback).

## Summary (2026-09-08)

| Stage | Model | float (ms) | w8a8 / INT8 (ms) | NPU | CPU ops |
|-------|-------|-----------:|-----------------:|-----|--------:|
| 1 face detect + landmarks | `mediapipe_face` (2 comp) | **0.6** | pending¹ | 100% | 0 |
| 2 person segmentation     | `mediapipe_selfie`        | **0.4** | **0.2** | 100% | 0 |
| 4 super-resolution        | `quicksrnetmedium`        | **0.5** | pending² | 100% | 0 |
| **Pipeline total**        |                           | **1.5** | — | **100%** | **0** |

**float pipeline: 1.5 ms → PASS with +31.5 ms headroom (≈22× under the 30 fps budget).**
Every op runs on the NPU; zero CPU fallback. On-device vs local-CPU accuracy: landmark
PSNR ≈ 75 dB (float, effectively lossless).

Notes:
- These use each model's **default input resolution**. The real pipeline must re-export
  with explicit `--height/--width` at the target capture resolution (e.g. 720p); that
  raises stage 2 and 4. Headroom is large enough that 30 fps is safe and 60 fps likely.
- Stage 3 (low-light, Zero-DCE) is bring-your-own — see `aihub/byo_lowlight.py`. Not yet profiled.
- ¹ `mediapipe_face` w8a8 needs the gated Kaggle `human-face` calibration set:
  `kaggle datasets download ashwingupta3012/human-face` then
  `python -m qai_hub_models.scripts.configure_dataset --class qai_hub_models.models.mediapipe_face.dataset.HumanFacesDataset --files <zip>`, then re-run `--precision w8a8 --only face`.
- ² `quicksrnetmedium` w8a8 calibration downloads BSDS300 from `www2.eecs.berkeley.edu`
  which now returns 403 (dead upstream). Needs a BSDS300 mirror + `configure_dataset`,
  or switch the SR calibration dataset. float SR (0.5 ms) is already well within budget.

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
