#!/usr/bin/env bash
# One-time environment setup for NPU Video Studio (cluster: login node is fine, no GPU needed).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

if [ ! -x "$PROJ/.venv/bin/python" ]; then
  echo "[setup] creating conda env at $PROJ/.venv (python 3.11)"
  module load anaconda3/2025.06 2>/dev/null || true
  conda create -y -p "$PROJ/.venv" python=3.11 pip
fi

echo "[setup] installing python deps"
"$PROJ/.venv/bin/python" -m pip install --upgrade pip
"$PROJ/.venv/bin/python" -m pip install qai-hub qai-hub-models onnx onnxruntime opencv-python-headless numpy

echo "[setup] versions:"
"$PROJ/.venv/bin/python" - <<'PY'
import qai_hub, importlib.metadata as m
print("qai-hub        ", qai_hub.__version__)
for p in ("qai-hub-models", "onnxruntime", "onnx", "numpy"):
    try: print(f"{p:15}", m.version(p))
    except Exception as e: print(f"{p:15} MISSING ({e})")
PY

echo
echo "[setup] next:  source scripts/activate  &&  qai-hub configure --api_token <TOKEN>"
