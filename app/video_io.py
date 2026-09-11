"""Frame sources and sinks: webcam / video file / image loop  ->  mp4 / window / virtual cam.

Two backends, same call signatures (see app/imgops.py for why):
  - cv2   : Linux / macOS / Windows-x64 dev & CPU-test path (proven, used on the cluster).
  - PyAV  : the Snapdragon (win_arm64) path — OpenCV ships no win_arm64 wheel, PyAV does.
            Webcam capture uses PyAV's `dshow` input on Windows. **Unvalidated** — written
            against PyAV's documented API but never run on real Windows-on-ARM hardware
            (this dev box is headless Linux). First thing to confirm the moment hardware
            is available; see packaging/README.md.
"""
from __future__ import annotations

import pathlib
import sys
import numpy as np

from . import imgops

HAS_CV2 = imgops.HAS_CV2
if HAS_CV2:
    import cv2
else:
    import av  # PyAV; see module docstring


def open_source(spec: str, size: tuple[int, int] | None = None):
    """`spec`: integer webcam index, a video path, or an image path (looped).
    Yields BGR uint8 frames."""
    p = pathlib.Path(spec)
    is_image = p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        img = imgops.imread(str(p))
        if size:
            img = imgops.resize(img, size)
        while True:
            yield img.copy()
        return

    if HAS_CV2:
        cap = cv2.VideoCapture(int(spec) if spec.isdigit() else str(spec))
        if size and spec.isdigit():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield (imgops.resize(fr, size) if size and not spec.isdigit() else fr)
        cap.release()
        return

    # --- PyAV path (win_arm64 / no cv2) ---
    if spec.isdigit():
        if sys.platform != "win32":
            raise RuntimeError("webcam capture without cv2 is only implemented for "
                              "Windows (PyAV dshow). Install cv2 on this platform.")
        # dshow device name lookup: ffmpeg/PyAV needs "video=<Device Name>", not an
        # index, on Windows. `spec` "0"/"1" selects the Nth video device.
        container = av.open("video=" + _dshow_device_name(int(spec)), format="dshow")
    else:
        container = av.open(str(spec))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="bgr24")
        yield (imgops.resize(arr, size) if size else arr)
    container.close()


def _dshow_device_name(index: int) -> str:
    """List DirectShow video devices via ffmpeg and return the Nth name.
    Unvalidated (no Windows to test against) — dshow device enumeration is a known
    ffmpeg quirk (it logs to stderr, not an API call), so this may need adjusting on
    first real run."""
    import subprocess
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, text=True).stderr
    names = [l.split('"')[1] for l in out.splitlines() if '"' in l and "Alternative name" not in l]
    if not names:
        raise RuntimeError("no DirectShow video devices found (ffmpeg -list_devices)")
    return names[index]


class Sink:
    """`spec`: 'virtualcam' | 'window' | a .mp4 path | a directory (frame_00001.png).

    'virtualcam' needs `pyvirtualcam` — **no win_arm64 wheel exists for it**, so on
    Snapdragon use 'window' instead: it opens a plain preview window that OBS Studio's
    "Window Capture" source can pick up, then OBS's own "Start Virtual Camera" exposes
    it to any meeting app. This sidesteps the missing wheel entirely and is the
    recommended path on Windows-on-ARM.
    """

    def __init__(self, spec: str, fps: float = 30.0):
        self.spec = spec
        self.fps = fps
        self._w = None          # cv2.VideoWriter | av output container
        self._av_stream = None
        self._vc = None
        self._win = None
        self._n = 0
        self._dir = None
        if spec == "virtualcam":
            try:
                import pyvirtualcam
            except ImportError as e:
                raise SystemExit(
                    "virtualcam needs `pip install pyvirtualcam` — note: it has NO "
                    "win_arm64 wheel as of 2026-09-11, so it will not install on a "
                    "Snapdragon PC. Use --out window instead (see docstring), which "
                    "works with OBS Studio's Window Capture + Start Virtual Camera.\n"
                    "Other platforms also need a loopback backend:\n"
                    "  Windows(x64) - OBS Studio virtual camera or Unity Capture\n"
                    "  macOS        - OBS Studio virtual camera\n"
                    "  Linux        - `sudo modprobe v4l2loopback`"
                ) from e
            self._pvc_mod = pyvirtualcam
        elif spec == "window":
            self._init_window()
        elif spec.endswith(".mp4") or spec.endswith(".avi"):
            self._path = spec
        else:
            self._dir = pathlib.Path(spec)
            self._dir.mkdir(parents=True, exist_ok=True)

    def _init_window(self) -> None:
        import tkinter as tk
        from PIL import ImageTk
        self._tk = tk
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.title("NPU Video Studio — capture this window in OBS")
        self._label = tk.Label(self._root)
        self._label.pack()

    def write(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self.spec == "virtualcam":
            if self._vc is None:
                self._vc = self._pvc_mod.Camera(width=w, height=h, fps=self.fps)
            self._vc.send(imgops.bgr_to_rgb(frame_bgr))
            self._vc.sleep_until_next_frame()
        elif self.spec == "window":
            from PIL import Image
            img = Image.fromarray(imgops.bgr_to_rgb(frame_bgr))
            photo = self._ImageTk.PhotoImage(img)
            self._label.configure(image=photo)
            self._label.image = photo             # keep a reference
            self._root.update_idletasks()
            self._root.update()
        elif self._dir is not None:
            self._n += 1
            imgops.imwrite(str(self._dir / f"frame_{self._n:05d}.png"), frame_bgr)
        elif HAS_CV2:
            if self._w is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._w = cv2.VideoWriter(self._path, fourcc, self.fps, (w, h))
            self._w.write(frame_bgr)
        else:
            if self._w is None:
                self._w = av.open(self._path, mode="w")
                self._av_stream = self._w.add_stream("h264", rate=int(self.fps))
                self._av_stream.width, self._av_stream.height = w, h
                self._av_stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(imgops.bgr_to_rgb(frame_bgr), format="rgb24")
            for packet in self._av_stream.encode(frame):
                self._w.mux(packet)

    def close(self) -> None:
        if self._w is not None and HAS_CV2:
            self._w.release()
        elif self._w is not None:                  # PyAV
            for packet in self._av_stream.encode():
                self._w.mux(packet)
            self._w.close()
        if self._vc is not None:
            self._vc.close()
        if self._win is not None or self.spec == "window":
            try:
                self._root.destroy()
            except Exception:
                pass
