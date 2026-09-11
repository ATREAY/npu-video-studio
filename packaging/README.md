# Packaging for Snapdragon X (Windows on ARM64)

Built and dependency-verified from a **headless Linux dev box with no ARM64 Windows
access**. Read this before assuming anything here "just works" — it's organized by
confidence level, not alphabetically.

## The one hard blocker this found

**OpenCV publishes no `win_arm64` wheel, for any variant** (`opencv-python`,
`-headless`, `-contrib`) — confirmed against PyPI on 2026-09-11. A packaging script
that just did `pip install -r requirements.txt` on a real Snapdragon PC would have
failed at `cv2`, silently, on demo day. That's why `app/imgops.py` and
`app/video_io.py` have two backends: `cv2` (proven, used for all dev/CPU testing on
this box) and a `Pillow` + `scipy` + `scikit-image` + `PyAV` fallback that carries zero
OpenCV dependency at all — **that fallback is the one that actually ships to the
Snapdragon target**, selected automatically at import time (`imgops.HAS_CV2`).

## What's verified vs. what's written-but-untested

| Piece | Status |
|---|---|
| Every dependency has a win_arm64 wheel | ✅ Verified — see `requirements-win-arm64.txt`, wheels sit in `wheelhouse/` |
| `wheelhouse/` installs fully offline, no PyPI | ✅ Verified — `pip install --dry-run --no-index` resolves clean, run `build_wheelhouse.sh` yourself to reproduce |
| Deployed app has **no** torch / qai-hub-models dependency | ✅ Verified (`grep` over `app/*.py`) — only `aihub/` dev scripts need those |
| Pillow/scipy/scikit-image image-op fallback (`imgops.py`) | ✅ Verified — ran the **entire pipeline** end-to-end with `cv2` forcibly blocked; output pixel-equivalent to the cv2 path (face box, landmarks, background blur, super-res all correct) |
| PyAV-based video-file/image source & mp4 sink (no cv2) | ✅ Verified — works on this Linux box (PyAV is cross-platform) |
| **QNN EP DLL resolution** (`session.py: _resolve_qnn_backend`) | ⚠️ **Unvalidated.** `onnxruntime-qnn`'s DLLs live in their own package dir, not next to `onnxruntime`; wired via `os.add_dll_directory` + `onnxruntime_qnn.get_qnn_htp_path()`, following Microsoft's documented plugin-EP pattern (same shape as `onnxruntime-directml`/`-openvino`). Never executed against a real QNN EP — no ARM64 Windows to run it on. **Confirm this first** the moment hardware is available; failure mode is a clean fallback to CPU EP, not a crash. |
| **PyAV webcam capture on Windows** (`video_io.py: _dshow_device_name`) | ⚠️ **Unvalidated.** ffmpeg's dshow device enumeration is a documented quirk (parsed from stderr text, not a real API) — likely needs adjustment on first run. |
| **`window` output + OBS Window Capture workflow** | ⚠️ **Unvalidated.** `tkinter`+`Pillow.ImageTk` is standard and ships with any official Python install, but never rendered on a real display from this box. |
| `pyvirtualcam` (`--out virtualcam`) | ❌ **No win_arm64 wheel exists.** Don't use it on Snapdragon — use `--out window` + OBS instead (see below). Kept only for dev-machine testing (Linux/macOS/Windows-x64). |

## Two ways to expose this as a system webcam

1. **Recommended on Snapdragon:** `run.ps1` with `-Out window` opens a plain preview
   window. In OBS Studio: **Sources → + → Window Capture → "NPU Video Studio"**, then
   **Start Virtual Camera**. Any meeting app can now select "OBS Virtual Camera."
   No extra Python wheel needed, no missing-package risk.
2. Dev machines only (has a real wheel there): `--out virtualcam` via `pyvirtualcam` +
   OBS/Unity Capture (Windows-x64) or `v4l2loopback` (Linux) or OBS (macOS).

## Layout

```
packaging/
  requirements-win-arm64.txt   pinned, every entry wheel-verified for win_arm64
  wheelhouse/                  the actual wheels (~110 MB) — NOT in git (see below)
  build_wheelhouse.sh          regenerates wheelhouse/ (run from any OS, needs internet)
  install.ps1                  creates .venv-win-arm64, installs offline from wheelhouse
  run.ps1                      launcher with sensible flags
```

`wheelhouse/*.whl` is gitignored — 110 MB of binaries doesn't belong in git history.
Before running `install.ps1` for the first time, either run `bash build_wheelhouse.sh`
(needs internet, ~1 min) or grab the pre-built wheelhouse from the repo's GitHub
Release assets if one's attached there.

## First things to test, in order, the moment real hardware is available

1. `.\install.ps1` — does the offline wheelhouse install cleanly? Does
   `ort.get_available_providers()` list `QNNExecutionProvider`?
2. `Session(...)` on one small model (e.g. the compiled Zero-DCE in
   `export_assets/zero_dce-onnx-compiled/`) — does `session.active_provider` come back
   `QNNExecutionProvider`, or does it silently fall back to CPU? If CPU, check
   `_resolve_qnn_backend()` in `app/session.py` — most likely fix is the exact
   `add_provider_for_devices`/registration call ONNX Runtime 1.30 expects (the
   `backend_path` provider-option approach here is the most portable guess but ORT's
   plugin-EP API has moved across versions).
3. `run.ps1 -Source 0 -Out window` — does the webcam open via PyAV/dshow? If not,
   `_dshow_device_name()` in `app/video_io.py` is the first place to fix — print the
   raw `ffmpeg -list_devices` output and adjust the parsing.
4. OBS Window Capture + Start Virtual Camera → confirm it shows up in Zoom/Meet/Teams.
