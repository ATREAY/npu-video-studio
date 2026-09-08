"""Central config for AI Hub jobs.

`DEVICE` is filled after running `python aihub/list_devices.py` — pick the newest
Snapdragon X entry (X2 Plus / X Elite). The prize hardware is Snapdragon X2 Plus,
so target the newest X2 device that AI Hub exposes; fall back to "Snapdragon X Elite CRD".
"""

# --- target device -----------------------------------------------------------
DEVICE = "Snapdragon X Elite CRD"  # override via --device or edit after list_devices.py

# --- pipeline models --------------------------------------------------------
# module name under qai_hub_models.models  ->  role in the pipeline.
# Confirm exact module names with:  python -m qai_hub_models.models
MODELS = {
    "face":       "mediapipe_face",       # stage 1: detection + landmarks (framing / eye-contact)
    "segment":    "mediapipe_selfie",     # stage 2: person segmentation (bg blur/replace)
    "lowlight":   "zero_dce",             # stage 3: low-light enhancement  (verify availability)
    "superres":   "quicksrnet_medium",    # stage 4: real-time super-resolution
}

# Runtime target for the compiled asset. Options: "tflite", "onnx", "qnn_context_binary".
# ONNX Runtime + QNN EP on Windows-on-ARM -> use "onnx".
TARGET_RUNTIME = "onnx"

# 30 fps budget
FRAME_BUDGET_MS = 33.0
