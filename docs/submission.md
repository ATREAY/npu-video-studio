# Submission write-up — mapped to judging criteria

> Fill this in as the build progresses. Headers match the four scored criteria so
> a judge can score each in one pass.

## 1. Technical Implementation
- Models from the Qualcomm AI Hub zoo (`qai-hub-models 0.61.0`): `mediapipe_face`,
  `mediapipe_selfie`, `quicksrnetmedium`. Low-light (stage 3) is a bring-your-own
  Zero-DCE model compiled directly via `qai_hub.submit_compile_job` (`aihub/byo_lowlight.py`).
- Compiled + profiled on a real **Snapdragon X2 Elite CRD** (Windows 11 on ARM) via the
  AI Hub device farm. Target runtime `onnx` → **ONNX Runtime 1.27.1 + QNN EP** on the
  Hexagon NPU (HTP). QAIRT 2.45.
- **Measured on-device (float), 2026-09-08** — see `benchmarks/results.md`:

  | Stage | Model | Latency | NPU | CPU ops |
  |---|---|---:|---|---:|
  | face detect + landmarks | `mediapipe_face` | 0.6 ms | 100% | 0 |
  | person segmentation | `mediapipe_selfie` | 0.4 ms | 100% | 0 |
  | super-resolution | `quicksrnetmedium` | 0.5 ms | 100% | 0 |
  | **pipeline** | | **1.5 ms** | **100%** | **0** |

  → **22× under the 33 ms / 30 fps budget**; every op on the NPU, zero CPU fallback.
- Accuracy: on-device vs local-CPU landmark PSNR ≈ 75 dB (float, effectively lossless).
- Quantisation: INT8 (w8a8) for segmentation → 0.2 ms, **2× faster than float**, still
  100% NPU. face + SR INT8 pending calibration-dataset access (see results.md notes).
- TODO before submission: re-export at real capture resolution (720p) with `--height/--width`;
  measured concurrent per-frame wall-time in the app; NPU-vs-CPU-EP delta on the same device;
  native ARM64 build with `session.get_providers()` + ORT trace showing no fallback.

## 2. Application Use Case & Innovation
- Problem: video-call quality on thin-and-light PCs and poor connectivity in India.
- On-device is essential: CPU/GPU stay free for the call; frames never leave the
  device; incoming low-res stream upscaled locally.
- Beyond Windows Studio Effects: adds real-time super-resolution + low-light rescue
  + stacked effects, all on NPU, open and configurable.
- Side-by-side comparison vs Studio Effects: <screenshots / metrics>.

## 3. Deployment & Accessibility
- Installer: ARM64 MSIX / packaged exe. One double-click, runs offline.
- Output as a standard virtual camera -> works in Zoom / Meet / Teams unmodified.
- Fallback path for judges without a Snapdragon device (CPU EP), clearly documented.
- Public repo, MIT licence, 2-minute quickstart.
- Accessibility of the app itself: keyboard nav, captions, high-contrast toggle.

## 4. Presentation & Documentation
- Demo video 3–5 min: problem (30s) -> live toggle of each effect (90s) ->
  architecture diagram (30s) -> NPU-vs-CPU benchmark chart (30s) -> impact (30s).
- Architecture diagram: `docs/pipeline.md`.
- Benchmark table: `benchmarks/results.md`.
- This document as the written submission.
- All materials in English. Work solely my own.

## Logistics
- One submission only; cannot edit after submitting. Submit ~29 Sep, not on deadline day.
- Double-check every intake-form field before submitting.
