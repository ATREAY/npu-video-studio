"""Central config for AI Hub jobs.

`DEVICE` is filled after running `python aihub/list_devices.py` — pick the newest
Snapdragon X entry (X2 Plus / X Elite). The prize hardware is Snapdragon X2 Plus,
so target the newest X2 device that AI Hub exposes; fall back to "Snapdragon X Elite CRD".
"""

# --- target device ----------------------------------------------------------
# AI Hub Windows-on-ARM compute devices (from `qai-hub list-devices`, 2026-09-08):
#   "Snapdragon X Elite CRD"        qualcomm-snapdragon-x-elite      (sc8380xp)
#   "Snapdragon X Plus 8-Core CRD"  qualcomm-snapdragon-x-plus-8-core(sc8340xp)
#   "Snapdragon X2 Elite CRD"       qualcomm-snapdragon-x2-elite     (sc8480xp)  <- newest
# Prize laptop is "Snapdragon X2 Plus" (X2 gen) -> profile primarily on X2 Elite CRD,
# and also on X Elite CRD for a broader "runs across the X-series" claim.
DEVICE = "Snapdragon X2 Elite CRD"
DEVICES_ALL = ["Snapdragon X2 Elite CRD", "Snapdragon X Elite CRD"]

# --- pipeline models -------------------------------------------------------
# Verified present in qai_hub_models 0.48.0 zoo unless marked BYO.
MODELS = {
    "face":     "mediapipe_face",     # stage 1: face detect + landmarks (framing / eye-contact)
    "segment":  "mediapipe_selfie",   # stage 2: selfie/person segmentation (bg blur/replace)
    "superres": "quicksrnetmedium",   # stage 4: real-time super-resolution (note: no underscores)
    # "lowlight": zero_dce is NOT in the zoo. Plan: export Zero-DCE (~10k params) from a
    # public PyTorch impl and compile it directly via qai_hub.submit_compile_job
    # (see aihub/byo_lowlight.py). Classical CLAHE/gamma on CPU is the fallback.
}
# alternatives if a stage needs swapping:
#   segment : segformer_base | unet_segmentation | ffnet_40s | yolov8_seg
#   superres: quicksrnetsmall | xlsr | sesr_m5 | real_esrgan_general_x4v3
#   face    : face_det_lite

# Runtime target for the compiled asset. Options: "tflite", "onnx", "qnn_context_binary".
# ONNX Runtime + QNN EP on Windows-on-ARM -> use "onnx".
TARGET_RUNTIME = "onnx"

# 30 fps budget
FRAME_BUDGET_MS = 33.0
