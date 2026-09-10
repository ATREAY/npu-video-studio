# app/ — runnable pipeline

```
python -m app.main --source <cam idx | video | image> --out <virtualcam | out.mp4 | dir/>
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
- `video_io.py` — webcam / file / image source; mp4 / virtualcam / frames-dir sink.

Models are the AI-Hub-compiled ONNX in `../export_assets/`. Re-export with
`aihub/export_all.py` (or `qai-hub-models export ...`) at the target resolution first.

Known gaps: BlazeFace decode is single-face / no NMS (fine for a single call subject);
SR is fixed-scale per compiled asset; virtualcam path needs `pyvirtualcam` + a loopback
device (Windows/macOS/Linux-v4l2loopback).
