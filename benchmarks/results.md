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
