"""NPU Video Studio — CLI.

Examples
--------
# webcam -> virtual camera, all effects on the NPU (on a Snapdragon PC)
python -m app.main --source 0 --out virtualcam

# a clip -> annotated mp4, CPU fallback, for headless testing / the demo video
python -m app.main --source clip.mp4 --out out.mp4 --provider cpu --debug

# one image, 60 frames -> frames/ , print per-stage timing
python -m app.main --source face.jpg --out frames --frames 60
"""
from __future__ import annotations

import argparse
import sys
import time
import cv2

from .pipeline import Pipeline
from .video_io import open_source, Sink
from .session import available_providers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="npu-video-studio")
    ap.add_argument("--source", required=True, help="webcam index | video path | image path")
    ap.add_argument("--out", default="out.mp4", help="'virtualcam' | *.mp4 | directory")
    ap.add_argument("--provider", default="auto", choices=["auto", "npu", "cpu"])
    ap.add_argument("--bg", default="blur", choices=["blur", "replace", "none"])
    ap.add_argument("--bg-image", default=None)
    ap.add_argument("--sr-asset", default=None,
                    help="export_assets subdir for a resolution-specific SR model")
    ap.add_argument("--sr-every-n", type=int, default=1)
    ap.add_argument("--lowlight-onnx", default=None, help="compiled Zero-DCE .onnx")
    ap.add_argument("--stages", default="face,segment,lowlight,superres")
    ap.add_argument("--no-autoframe", action="store_true")
    ap.add_argument("--debug", action="store_true", help="draw face box / landmarks")
    ap.add_argument("--size", default=None, help="WxH capture/resize, e.g. 1280x720")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)

    size = None
    if args.size:
        w, h = args.size.lower().split("x")
        size = (int(w), int(h))
    bg_img = cv2.imread(args.bg_image) if args.bg_image else None

    pipe = Pipeline(
        prefer=args.provider,
        bg_mode="blur" if args.bg == "none" else args.bg,
        bg_image=bg_img,
        sr_asset=args.sr_asset,
        sr_every_n=args.sr_every_n,
        lowlight_onnx=args.lowlight_onnx,
        enable=[s for s in args.stages.split(",") if s] if args.bg != "none"
        else [s for s in args.stages.split(",") if s and s != "segment"],
        autoframe=not args.no_autoframe,
        debug=args.debug,
    )
    print(f"ORT providers available : {available_providers()}", file=sys.stderr)
    print(f"per-stage provider       : {pipe.providers}", file=sys.stderr)

    sink = Sink(args.out, fps=args.fps)
    n, t_wall, agg = 0, time.perf_counter(), {}
    try:
        for frame in open_source(args.source, size):
            out = pipe(frame)
            sink.write(out)
            n += 1
            for k, v in pipe.timings.items():
                agg[k] = agg.get(k, 0.0) + v
            if n <= 3 or n % 30 == 0:
                print(f"[{n:5d}] {pipe.report()}", file=sys.stderr)
            if args.frames and n >= args.frames:
                break
    finally:
        sink.close()

    dt = time.perf_counter() - t_wall
    print(f"\n{n} frames in {dt:.1f}s  ({n/dt:.1f} fps wall)", file=sys.stderr)
    if n:
        print("mean per-stage ms: " +
              "  ".join(f"{k}={v/n:.2f}" for k, v in agg.items()) +
              f"  | total={sum(agg.values())/n:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
