# Pipeline architecture

```
                 webcam frame (BGR, e.g. 1280x720 @ 30fps)
                              |
                    [downscale to model res]
                              |
        +---------------------+---------------------+
        |                     |                     |
   stage 1: face         stage 2: person      (stage 1+2 can run
   detection +           segmentation          in parallel on NPU)
   landmarks             (alpha matte)
        |                     |
   crop / recenter      composite: blur or
   (auto-framing,       replace background
   eye-contact warp)    using alpha matte
        +----------+----------+
                   |
            stage 3: low-light enhancement (Zero-DCE style)
                   |
            stage 4: super-resolution (QuickSRNet)  <- only when
                   |                                    incoming stream
                   |                                    is low-res
            [upscale to output res]
                   |
          preview window -> OBS Virtual Camera -> meeting app (untested)
```

## 30 fps frame budget (33.3 ms)

| Stage | Target ms | Notes |
|-------|-----------|-------|
| face detection + landmarks | ≤ 3 | tiny model, runs every frame |
| person segmentation        | ≤ 8 | 256x256 or 512x512 alpha matte, upsample on CPU/GPU |
| low-light enhancement      | ≤ 6 | skip if frame already bright (luma gate) |
| super-resolution           | ≤ 12 | 2x or 4x; run every other frame if needed |
| compositing / color / IO   | ≤ 4 | CPU/GPU, not NPU |
| **total**                  | **≤ 33** | |

Fallback levers if over budget: QuickSRNet-Small instead of Medium, lower seg
resolution, run SR at 15 fps and hold, INT8 everywhere.

## Runtime

- **On the Snapdragon device:** ONNX Runtime with the **QNN Execution Provider**
  (`QNNExecutionProvider`), one `InferenceSession` per model, HTP (NPU) backend.
- Verify with ORT profiling / `session.get_providers()` that ops land on QNN and
  do not fall back to CPU.
- Target: native **ARM64** Python (see [packaging/](../packaging/)). Not yet run on Windows-on-ARM hardware.

## Concurrency

The four models are independent per frame except stage 4 depends on 1-3 output.
Run face + segmentation as concurrent ORT sessions; the QNN backend serialises on
the single HTP but overlaps DMA/prepost. Measure actual wall-time, don't assume.

## Models

| Role | First choice | Alternatives |
|------|--------------|--------------|
| face | `mediapipe_face` | `face_det_lite` |
| segmentation | `mediapipe_selfie` | `ffnet_40s`, `ffnet_54s`, `deeplabv3_plus_mobilenet` |
| low-light | Zero-DCE (bring-your-own, **not** in the AI Hub zoo) | CLAHE on CPU |
| super-res | `quicksrnetmedium` | `quicksrnetsmall`, `xlsr`, `sesr_m5`, `real_esrgan_x4plus` |
