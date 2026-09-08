# Benchmarks

Populated by `aihub/export_all.py` (compile + on-device profiling jobs on AI Hub).
Each run appends a section below.

Target: sum of the four stage latencies **< 33 ms** for 30 fps, with **NPU utilisation
~100%** per model (i.e. no silent CPU fallback).

For the submission also record, per model:
- NPU latency vs **CPU** latency (re-run with `--runtime onnx` on a CPU-only device or
  force CPU EP) — the headline "Nx faster on NPU" number.
- Peak memory, model load time.
- Quantised (INT8) vs FP16 latency + any quality delta.

<!-- runs appended below -->

## Run 2026-09-08 20:39 — device: `Snapdragon X2 Elite CRD` — runtime: `onnx`

| Stage | Model | On-device latency (ms) | NPU util (%) | Status | Log |
|-------|-------|------------------------|--------------|--------|-----|
| face | `mediapipe_face` | ? | ? | FAIL | benchmarks/raw/face.Snapdragon_X2_Elite_CRD.float.log |

**Sum of stage latencies: nan ms** vs 33 ms budget (30 fps) -> **OVER BUDGET**


## Run 2026-09-08 21:25 — device: `Snapdragon X2 Elite CRD` — runtime: `onnx`

| Stage | Model | On-device latency (ms) | NPU util (%) | Status | Log |
|-------|-------|------------------------|--------------|--------|-----|
| face | `mediapipe_face` | ? | 70 | FAIL | benchmarks/raw/face.Snapdragon_X2_Elite_CRD.float.log |

**Sum of stage latencies: nan ms** vs 33 ms budget (30 fps) -> **OVER BUDGET**


## Run 2026-09-08 22:25 — device: `Snapdragon X2 Elite CRD` — runtime: `onnx`

| Stage | Model | On-device latency (ms) | NPU util (%) | Status | Log |
|-------|-------|------------------------|--------------|--------|-----|
| face | `mediapipe_face` | 0.4 | 70 | ok | benchmarks/raw/face.Snapdragon_X2_Elite_CRD.float.log |

**Sum of stage latencies: 0.4 ms** vs 33 ms budget (30 fps) -> **PASS**


## Run 2026-09-08 22:26 — `Snapdragon X2 Elite CRD` — runtime `onnx` — precision `float`

| Stage | Model | Latency (ms) | Comp | NPU % | CPU ops | Status |
|-------|-------|-------------:|-----:|------:|--------:|--------|
| face | `mediapipe_face` | 0.6 | 2 | 100.0 | 0 | ok |

Sum of stage latencies: **0.60 ms** vs 33 ms budget → PASS (+32.4 ms headroom)

- face jobs: https://workbench.aihub.qualcomm.com/jobs/j5mm2rdq5/ https://workbench.aihub.qualcomm.com/jobs/jgnny27mg/ https://workbench.aihub.qualcomm.com/jobs/jpr0qknep/ https://workbench.aihub.qualcomm.com/jobs/jp2w68vmp/ https://workbench.aihub.qualcomm.com/jobs/jp8x9o48g/ https://workbench.aihub.qualcomm.com/jobs/j5qyk8mmg/
