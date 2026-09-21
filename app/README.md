# app/ — runnable pipeline

```
python -m app.main --source <cam idx | video | image> --out <window | virtualcam | out.mp4 | dir/>
                   [--provider auto|npu|cpu] [--bg blur|replace|none] [--bg-image bg.jpg]
                   [--sr-asset <export_assets subdir>] [--sr-every-n N]
                   [--lowlight-onnx zero_dce.onnx] [--stages face,segment,lowlight,superres]
                   [--no-autoframe] [--debug] [--size WxH] [--frames N] [--fps F]
```

- `session.py` — ORT session, provider order `[QNNExecutionProvider, CPUExecutionProvider]`.
  On a Snapdragon X PC it runs on the Hexagon NPU; anywhere else it falls back to CPU
  automatically (same code path — this is how a judge without Snapdragon hardware runs it).
- `stages.py` — Face (detect + 468 landmarks), Segment (selfie matte), LowLight
  (Zero-DCE if `--lowlight-onnx` given, else CLAHE), SuperRes (QuickSRNet, fixed scale).
- `compositor.py` — background blur/replace from the alpha matte, EMA auto-framer.
- `pipeline.py` — orchestration + per-stage timing + 33 ms / 30 fps budget check.
- `video_io.py` — webcam / file / image source; mp4 / window / virtualcam / frames-dir sink.

Models are the AI-Hub-compiled ONNX in `../export_assets/`. Re-export with
`aihub/export_all.py` (or `qai-hub-models export ...`) at the target resolution first.

Known gaps: BlazeFace decode is single-face / no NMS (fine for a single call subject);
SR is fixed-scale per compiled asset; virtualcam path needs `pyvirtualcam` + a loopback
device (Windows/macOS/Linux-v4l2loopback).

## Notes added 2026-09-10

- **Face decode** validated bit-exact vs `qai_hub_models` reference (box, score, anchor).
  Adds NMS + DSCALE(1.1) box expansion + 6 keypoints (eyes exposed for eye-contact).
  Landmark crop is axis-aligned (reference uses a rotation-aware affine crop) — fine for
  an upright call subject.
- **Real-res SR**: `--sr-asset quicksrnetmedium-onnx-float-2x360` = 640×360 → 1280×720 @2×
  (compiled asset downloaded from AI Hub job j5wl69lzp). `_find_onnx` handles both flat and
  `job_*/model.onnx` layouts; scale is read from the model I/O.
- **Low-light**: real Zero-DCE (Li-Chongyi Epoch99 weights) at `export_assets/zero_dce/zero_dce.onnx` (not committed — CC BY-NC; build it with `python aihub/byo_lowlight.py --onnx-only`)
  (256², curves upsample). Auto-used if present; `--lowlight-onnx` overrides; CLAHE if neither.
- **CPU-fallback latency is machine-dependent.** On this cgroup-throttled cluster node the
  CPU EP is ~30–100× slower than a normal laptop; ignore absolute cluster CPU ms. The NPU
  numbers in `../benchmarks/results.md` are the real ones.
- `--out virtualcam` needs `pip install pyvirtualcam` + a loopback device (OBS on Win/mac,
  `v4l2loopback` on Linux). Use `--out out.mp4` for headless / the demo recording.
