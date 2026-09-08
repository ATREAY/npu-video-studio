#!/usr/bin/env bash
# One-time environment setup for NPU Video Studio (cluster: login node is fine, no GPU needed).
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

if [ ! -x "$PROJ/.venv/bin/python" ]; then
  echo "[setup] creating conda env at $PROJ/.venv (python 3.11)"
  module load anaconda3/2025.06 2>/dev/null || true
  conda create -y -p "$PROJ/.venv" python=3.11 pip
  # isolate from ~/.local thesis packages (mamba-ssm, torch-geometric, ...)
  conda env config vars set PYTHONNOUSERSITE=1 -p "$PROJ/.venv"
fi

echo "[setup] installing python deps (isolated: PYTHONNOUSERSITE=1, PIP_USER=0)"
export PYTHONNOUSERSITE=1 PIP_USER=0
"$PROJ/.venv/bin/python" -m pip install --upgrade pip
if [ -f "$PROJ/requirements.lock.txt" ]; then
  "$PROJ/.venv/bin/python" -m pip install -r "$PROJ/requirements.lock.txt"
else
  "$PROJ/.venv/bin/python" -m pip install \
    qai-hub qai-hub-models onnx onnxruntime numpy \
    "opencv-python-headless<4.11" \
    sympy networkx filelock psutil pyparsing joblib threadpoolctl cloudpickle mpmath
  # qai-hub-models pulls plain opencv-python (needs libGL, absent on headless nodes);
  # force the headless build:
  "$PROJ/.venv/bin/python" -m pip uninstall -y opencv-python 2>/dev/null || true
fi

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
