#!/usr/bin/env bash
# Self-contained smoke test for the app/ pipeline. Run this any time to check nothing
# is broken. Exercises BOTH image-op backends (cv2, and the cv2-free Pillow/scipy path
# that's what actually ships to Snapdragon — see packaging/README.md) and diffs their
# output to prove they agree.
#
# Usage:  bash scripts/smoke_test.sh [path/to/a/real/face.jpg]
#   With no argument it fetches the same public demo face photo AI Hub's own
#   mediapipe_face model uses (needs qai_hub_models + network the first time, cached
#   after). Pass your own image to skip that.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/activate
OUT="smoke_test_output"
rm -rf "$OUT"; mkdir -p "$OUT"

IMG="${1:-}"
if [ -z "$IMG" ]; then
  echo "[smoke] no image given — fetching the AI Hub demo face photo (cached after first run)"
  python - <<'PY'
from qai_hub_models.models.mediapipe_face.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.utils.asset_loaders import load_image
load_image(INPUT_IMAGE_ADDRESS).save("smoke_test_output/input.jpg")
PY
  IMG="$OUT/input.jpg"
fi
echo "[smoke] using image: $IMG"

echo
echo "=== 1/3  cv2 backend: full pipeline (face+segment+lowlight+superres) ==="
python -m app.main --source "$IMG" --out "$OUT/cv2_frames" --frames 1 --provider cpu --debug \
  --size 1280x720 --sr-asset quicksrnetmedium-onnx-float-2x360 2>&1 | grep -vE "pthread_setaffinity|E:onnxruntime"

echo
echo "=== 2/3  cv2-free backend (Pillow+scipy+scikit-image+PyAV): same pipeline ==="
python - "$IMG" "$OUT" <<'PY'
import sys, builtins
img_path, out_dir = sys.argv[1], sys.argv[2]
_real_import = builtins.__import__
def blocked(name, *a, **k):
    if name == "cv2" or name.startswith("cv2."):
        raise ImportError("cv2 blocked for fallback-backend smoke test")
    return _real_import(name, *a, **k)
builtins.__import__ = blocked
for m in list(sys.modules):
    if m == "cv2" or m.startswith("cv2."):
        del sys.modules[m]

from app.pipeline import Pipeline
from app import imgops
assert not imgops.HAS_CV2, "expected cv2 to be blocked"
im = imgops.resize(imgops.imread(img_path), (1280, 720))
pipe = Pipeline(prefer="cpu", bg_mode="blur", sr_asset="quicksrnetmedium-onnx-float-2x360",
                enable=["face", "segment", "superres"], debug=True)
out = pipe(im)
imgops.imwrite(f"{out_dir}/nocv2_output.png", out)
print("HAS_CV2:", imgops.HAS_CV2, "| output:", out.shape)
print(pipe.report())
PY

echo
echo "=== 3/3  compare the two backends' output (should be near-identical) ==="
python - "$OUT" <<'PY'
import sys, numpy as np
from app import imgops
out_dir = sys.argv[1]
a = imgops.imread(f"{out_dir}/cv2_frames/frame_00001.png")
b = imgops.imread(f"{out_dir}/nocv2_output.png")
diff = np.abs(a.astype(int) - b.astype(int))
print(f"mean abs diff: {diff.mean():.2f}/255   max: {diff.max()}   (small = backends agree)")
PY

echo
echo "[smoke] PASS — outputs are in $OUT/  (open frame_00001.png and nocv2_output.png to look)"
