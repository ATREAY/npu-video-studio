# Submission write-up — DRAFT

> Headers match the four scored criteria. **Everything not yet done is marked `TODO`.**
> Nothing here should claim more than what exists in the repo today.

## 1. Technical Implementation

**Pipeline.** Four neural stages per frame, all executed through ONNX Runtime with the
QNN execution provider (Hexagon NPU) when available, CPU otherwise:

| Stage | Model | Source |
|---|---|---|
| Face detect + 468 landmarks | `mediapipe_face` | Qualcomm AI Hub zoo |
| Person segmentation (background blur) | `mediapipe_selfie` | Qualcomm AI Hub zoo |
| Low-light enhancement | Zero-DCE | **bring-your-own** (open source), compiled with `qai_hub.submit_compile_job` |
| Super-resolution | `quicksrnetmedium` | Qualcomm AI Hub zoo |

**Measured on a real Snapdragon X2 Elite** (AI Hub hosted device, `qai-hub-models 0.61.0`,
QAIRT 2.45), float precision. Full table with job links: [benchmarks/results.md](../benchmarks/results.md).

| Stage | Latency | Compute |
|---|---:|---|
| face detect + landmarks | 0.6 ms | 100% NPU, 0 CPU ops |
| person segmentation | 0.4 ms | 100% NPU, 0 CPU ops |
| low-light (Zero-DCE, 256²) | 1.4 ms | 100% NPU, 0 CPU ops |
| super-resolution 640×360→720p (2×) | 3.4 ms | 100% NPU, 0 CPU ops |
| super-resolution 320×180→720p (4×) | 1.5 ms | 100% NPU, 0 CPU ops |
| **Sum of profiled models** | **≈ 3.9–5.8 ms** | budget 33 ms for 30 fps |

The sum is of separately profiled models. It excludes pre/post-processing, compositing
and capture, and is **not** a measured end-to-end frame time.

- Accuracy: on-device vs CPU landmark output PSNR ≈ 75 dB (float).
- Quantization: INT8 (w8a8) done for segmentation only (0.2 ms, 2× faster than float).
  Face and super-resolution INT8 are blocked on calibration datasets (gated Kaggle set;
  dead upstream URL) — documented in results.md.
- Engineering findings worth noting: the face-detector decode was validated bit-exact
  against Qualcomm's reference implementation; OpenCV has no Windows-ARM64 wheel, so the
  app carries a second image-op backend (Pillow/SciPy/scikit-image/PyAV) that was tested
  end to end with OpenCV blocked.
- TODO: measure concurrent per-frame wall time in the app **on a Snapdragon device**;
  same-device NPU-vs-CPU comparison; confirm `QNNExecutionProvider` actually loads via
  `onnxruntime-qnn` on Windows-on-ARM (currently unvalidated).

## 2. Application Use Case & Innovation

- **Problem.** Video calls on thin laptops with weak lighting and unreliable bandwidth.
- **Why on-device.** Frames never leave the machine, no cloud round-trip, and running
  the networks on the NPU leaves CPU/GPU free for the call itself.
- **What is combined.** Face tracking, background blur, low-light enhancement and
  super-resolution in one NPU pipeline, with a bring-your-own low-light model from
  outside the AI Hub zoo.
- TODO: real before/after screenshots or GIFs (low-light, background blur, upscaled
  low-res frame). Generate them from `scripts/smoke_test.sh` output and a dark test image.
- TODO: decide whether to include a comparison against Windows Studio Effects. **No such
  comparison exists yet, so none is claimed.**

## 3. Deployment & Accessibility

What exists today:

- The pipeline runs from a clone on any machine (CPU fallback): `bash scripts/smoke_test.sh`.
- Windows-on-ARM64 install scripts ([packaging/](../packaging/)): pinned dependency list,
  an offline wheel bundle (verified to resolve with no network), `install.ps1`, `run.ps1`.
  **Written and dependency-checked but never executed on Windows-on-ARM.**
- Virtual camera path: preview window captured by OBS Window Capture, exposed via OBS
  Virtual Camera. **Untested.**

What does **not** exist: a packaged installer (MSIX/exe), GUI controls, or any
accessibility features in the app itself.

- TODO: if hardware access is obtained, run the "first things to test" list in
  [packaging/README.md](../packaging/README.md) and update this section with results.

## 4. Presentation & Documentation

- Repository: https://github.com/ATREAY/npu-video-studio
- Architecture: [docs/pipeline.md](pipeline.md). TODO: render a diagram image.
- Benchmarks: [benchmarks/results.md](../benchmarks/results.md), every number linked to its AI Hub job.
- TODO: demo video (3–5 min). Planned structure: problem → app running live on CPU →
  screen-recorded AI Hub job pages showing the Snapdragon X2 Elite device and NPU compute
  breakdown → architecture → limitations stated plainly.
- All materials in English.

## Ownership and third-party material

The project code is owned by the participant and released under the MIT License. Third-party models and weights are used
under their upstream terms (see the README's third-party section). Zero-DCE is
CC BY-NC 4.0 and is therefore not redistributed in the repository.

## Logistics

- One submission per participant; it cannot be edited after submitting. Target ~29 Sep.
- TODO: check every field of the submission form before submitting.
