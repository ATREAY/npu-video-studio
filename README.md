# NPU Video Studio

A four-stage, on-device video-call enhancement pipeline built for the Qualcomm
Hexagon NPU of Snapdragon X PCs. Each frame goes through face tracking, background
blur, low-light lift and super-resolution, with every neural network profiled on real
Snapdragon X2 Elite silicon through Qualcomm AI Hub.

Built for the **Snapdragon AI Lab - Build & Present Challenge**.

## Status - read this first

| | State |
|---|---|
| Model performance on Snapdragon X2 Elite | **Measured** on AI Hub's hosted device, job links below. All four stages 100% NPU, 0 CPU ops. |
| The app pipeline end to end | **Runs and is tested** on CPU (Linux) with two image-op backends. |
| The app running on a Snapdragon PC | **Not done.** The author has no Snapdragon hardware, so the NPU numbers come from AI Hub profiling, not from this app running on a device. |
| Windows-on-ARM installer | Scripts and an offline dependency bundle are written and dependency-checked, **but never executed on Windows-on-ARM**. See [packaging/README.md](packaging/README.md). |
| Virtual camera | Via OBS "Window Capture" + "Start Virtual Camera". **Untested.** |

## Results (Snapdragon X2 Elite CRD, ONNX Runtime + QNN, float)

Latency is AI Hub's per-model profile. The pipeline total is the **sum of separately
profiled models**, not a measured end-to-end frame time (pre/post-processing,
compositing and capture are not included).

| Stage | Model | Input | Latency | Compute | Profile job |
|---|---|---|---:|---|---|
| Face detect + 468 landmarks | `mediapipe_face` | 256² / 192² | 0.6 ms | 100% NPU | [detector](https://workbench.aihub.qualcomm.com/jobs/j5w7vol6g/), [landmarks](https://workbench.aihub.qualcomm.com/jobs/jg9m1vzl5/) |
| Person segmentation | `mediapipe_selfie` | 256² | 0.4 ms | 100% NPU | [job](https://workbench.aihub.qualcomm.com/jobs/jp0jqy99g/) |
| Low-light enhancement | Zero-DCE (bring-your-own) | 256² | 1.4 ms | 100% NPU | [job](https://workbench.aihub.qualcomm.com/jobs/jpxlo71lp/) |
| Super-resolution | `quicksrnetmedium` | 640×360 → 1280×720 (2×) | 3.4 ms | 100% NPU | [job](https://workbench.aihub.qualcomm.com/jobs/j57erdeqp/) |
| Super-resolution | `quicksrnetmedium` | 320×180 → 1280×720 (4×) | 1.5 ms | 100% NPU | [job](https://workbench.aihub.qualcomm.com/jobs/jgk2xy0og/) |
| **Sum** | | | **≈ 3.9–5.8 ms** | | vs a 33 ms budget for 30 fps |

- INT8 (w8a8): only segmentation is quantized so far, at **0.2 ms** (2× faster than
  float). INT8 for face and super-resolution is blocked on calibration datasets (one is
  gated on Kaggle, one upstream link is dead). Details in [benchmarks/results.md](benchmarks/results.md).
- Accuracy: on-device vs CPU landmark output PSNR ≈ 75 dB (float).
- Every number links to its AI Hub job. [aihub/export_all.py](aihub/export_all.py)
  reproduces the default-resolution runs; the 720p super-resolution runs used the
  model's own export command with `--height/--width/--scale-factor`.

## Try it

```bash
source scripts/activate          # puts the project venv on PATH
bash scripts/smoke_test.sh       # full pipeline via cv2, then via the cv2-free backend, and diffs them
bash scripts/smoke_test.sh photo.jpg   # or on your own image
```

```bash
python -m app.main --source <webcam idx | video | image> --out <window | out.mp4 | dir/> --provider cpu
```

The provider order is QNN (NPU) → CPU, so the same code uses the NPU on a Snapdragon
PC and falls back to CPU everywhere else. See [app/README.md](app/README.md).

## Why two image backends

OpenCV publishes **no Windows-ARM64 wheel** (checked against PyPI, 2026-09-11), so
`pip install opencv-python` fails on a Snapdragon PC. The app runs on OpenCV where
available and otherwise on Pillow + SciPy + scikit-image + PyAV, all of which ship
win_arm64 wheels. The smoke test runs the whole pipeline both ways and compares.

## Layout

```
app/         the runnable pipeline (session, stages, compositor, video I/O)
aihub/       Qualcomm AI Hub scripts: compile + profile models, bring-your-own Zero-DCE
benchmarks/  results.md with every measured number and its job link
packaging/   Windows-on-ARM64 installer scripts + offline wheel bundle (unvalidated on HW)
scripts/     env activation, smoke test
docs/        pipeline architecture, submission write-up draft
export_assets/  AI-Hub-compiled ONNX models for face, segmentation, super-resolution
```

## Low-light model and its licence

Zero-DCE is licensed **CC BY-NC 4.0 (non-commercial, academic use)** by its authors
([Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)), so its weights are
**not redistributed in this repository**. Without them the low-light stage falls back to
CLAHE. To build the ONNX model locally (downloads the weights from the original repo):

```bash
python aihub/byo_lowlight.py --onnx-only     # writes export_assets/zero_dce/zero_dce.onnx
```

## License

This project's own code is released under the [MIT License](LICENSE). Third-party
models, weights and libraries keep their own licences (next section) and are **not**
relicensed by it. In particular, Zero-DCE is CC BY-NC 4.0 and is not covered by, or
distributed under, the MIT licence.

## Third-party components

- Models from the [Qualcomm AI Hub Models](https://github.com/quic/ai-hub-models) zoo:
  `mediapipe_face`, `mediapipe_selfie`, `quicksrnetmedium` (code BSD-3-Clause). The
  compiled ONNX files here are AI Hub outputs of those models; **the upstream weight
  licences apply** check each model's page on AI Hub.
- Zero-DCE: see above.
- ONNX Runtime, `onnxruntime-qnn`, NumPy, Pillow, SciPy, scikit-image, PyAV, OpenCV: their own licences.

## Author

Atreay Kukanur (IIT Hyderabad).
