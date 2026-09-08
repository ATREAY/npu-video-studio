# NPU Video Studio

Real-time webcam enhancement that runs entirely on the **Qualcomm Hexagon NPU** of a
Snapdragon X-series PC. Four models run concurrently on each frame and the result is
exposed as a **virtual camera** usable in any meeting app:

| Stage | Job | Model (Qualcomm AI Hub zoo) |
|-------|-----|-----------------------------|
| 1. Auto-framing / eye-contact | face detection + landmarks | `mediapipe_face` |
| 2. Background blur / replace  | person segmentation        | TBD (`mediapipe_selfie` / `ffnet_*`) |
| 3. Low-light rescue           | low-light enhancement      | TBD (`zero_dce` / alt) |
| 4. Upscale (bad-bandwidth)    | real-time super-resolution | `quicksrnet_medium` |

**Frame budget: < 33 ms end-to-end for 30 fps.**

## Why on-device / why Snapdragon
- All effects run on the NPU, leaving CPU + GPU free for the video call itself.
- Works fully offline; no frames leave the device (privacy).
- Locally upscales a low-res incoming stream on poor connectivity.

## Repo layout
```
aihub/        Qualcomm AI Hub scripts: list devices, compile + profile all models
src/          Runtime pipeline (ONNX Runtime + QNN EP) and virtual-camera output
models/       Compiled model artifacts (gitignored)
benchmarks/   NPU-vs-CPU latency / utilization tables from AI Hub profile jobs
docs/         Architecture notes, submission write-up mapped to judging criteria
scripts/      Env setup
```

## Setup
```bash
bash scripts/setup_env.sh          # one-time: creates ./.venv, installs deps
source scripts/activate            # activates the env for a shell
qai-hub configure --api_token <YOUR_FRESHLY_REGENERATED_TOKEN>
qai-hub list-devices | grep -i snapdragon
```

## Workflow
1. `python aihub/list_devices.py` — pick the newest Snapdragon X device string.
2. `python aihub/export_all.py --device "<device string>"` — compile + profile every
   model on real hardware; writes latency/utilization to `benchmarks/results.md`.
3. Wire compiled models into `src/pipeline.py`, validate the 33 ms budget.
4. Package as an ARM64 app with virtual-camera output.

## Challenge
Snapdragon AI Lab – Build & Present Challenge. Solution submission closes **30 Sep 2026, 23:59 IST**.
