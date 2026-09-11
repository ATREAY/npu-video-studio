#!/usr/bin/env bash
# Rebuild packaging/wheelhouse/ — the offline win_arm64 dependency bundle.
# Run from any machine with pip (does NOT need to be Windows or ARM64: pip can
# target-download wheels for a different platform/abi).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=${PYTHON:-../.venv/bin/python}
rm -rf wheelhouse && mkdir -p wheelhouse

"$PY" -m pip download \
  --platform win_arm64 --python-version 311 --implementation cp --abi cp311 \
  --only-binary=:all: -d wheelhouse \
  -r requirements-win-arm64.txt

echo
echo "=== verifying the bundle resolves fully offline (no PyPI) ==="
"$PY" -m pip install --dry-run --no-index --find-links wheelhouse \
  --platform win_arm64 --python-version 311 --implementation cp --abi cp311 \
  --only-binary=:all: --target /tmp/wheelhouse-dryrun-$$ \
  -r requirements-win-arm64.txt
rm -rf /tmp/wheelhouse-dryrun-$$

du -sh wheelhouse
echo "wheelhouse OK: $(ls wheelhouse | wc -l) wheels, fully self-contained."
