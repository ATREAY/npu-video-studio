# Submission write-up — mapped to judging criteria

> Fill this in as the build progresses. Headers match the four scored criteria so
> a judge can score each in one pass.

## 1. Technical Implementation
- Models sourced from Qualcomm AI Hub zoo: <list + versions>.
- Compiled for `<device>` via AI Hub; target runtime ONNX Runtime + QNN EP (HTP/NPU).
- Quantisation: INT8 for <models>; quality delta vs FP16: <numbers>.
- On-device profiling (AI Hub jobs): see `benchmarks/results.md`. Per model:
  NPU latency, NPU utilisation %, load time, peak memory.
- **NPU vs CPU:** <Nx> faster, <mW> lower power (headline chart).
- Concurrency: measured wall-time for the 4-model pipeline per frame = <ms> (≤ 33).
- Native ARM64 build; verified no CPU fallback (`session.get_providers()` + ORT trace).

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
