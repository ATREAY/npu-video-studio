"""Image-op backend: cv2 where available, Pillow/scipy/numpy where it isn't.

Why: OpenCV publishes NO Windows-ARM64 wheel (verified against PyPI, 2026-09-11) —
`pip install opencv-python[-headless]` fails outright on a Snapdragon X PC. Pillow,
scipy and scikit-image all ship win_arm64 wheels, so that's the real deployment path;
cv2 stays the dev/CPU-test backend (Linux/macOS/Windows-x64) since it's already proven
there. Every function here has the same signature regardless of backend, so stages.py /
compositor.py / video_io.py never need to know which one is active.

`HAS_CV2` tells the caller (and the packaging docs) which backend is live.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

if not HAS_CV2:
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
    try:
        from skimage import color as _skcolor, exposure as _skexposure
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False


# --------------------------------------------------------------------------- #
def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (w, h). Accepts HxWx3 uint8 or a single-channel (mask) array of any dtype."""
    if HAS_CV2:
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    w, h = size
    if img.ndim == 2:  # single-channel mask (e.g. alpha), possibly float
        from scipy.ndimage import zoom
        zy, zx = h / img.shape[0], w / img.shape[1]
        return zoom(img, (zy, zx), order=1)
    pil = Image.fromarray(img)
    return np.array(pil.resize((w, h), Image.BILINEAR))


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if HAS_CV2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[..., ::-1]


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(img)  # channel-swap is its own inverse


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize | 1
    if HAS_CV2:
        return cv2.GaussianBlur(img, (k, k), 0)
    sigma = k / 6.0
    return gaussian_filter(img, sigma=(sigma, sigma, 0)).astype(img.dtype)


class Clahe:
    """Contrast-limited local histogram equalisation on the luma channel.

    cv2 backend: real CLAHE (LAB L-channel). Fallback: skimage adaptive
    histogram equalisation on luma via YCbCr, which is visually equivalent for
    our purposes (low-light lift), else a cheap global-stretch as a last resort.
    """

    def __init__(self, clip_limit: float = 2.0, tile: tuple[int, int] = (8, 8)):
        if HAS_CV2:
            self._c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)

    def apply_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        if HAS_CV2:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._c.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb = img_bgr[..., ::-1].astype(np.float64) / 255.0
        if HAS_SKIMAGE:
            lab = _skcolor.rgb2lab(rgb)
            l = lab[..., 0] / 100.0
            l_eq = _skexposure.equalize_adapthist(l, clip_limit=0.02)
            lab[..., 0] = l_eq * 100.0
            out = np.clip(_skcolor.lab2rgb(lab), 0, 1)
        else:  # last-resort global stretch, no extra deps
            lo, hi = np.percentile(rgb, (1, 99))
            out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        return (out[..., ::-1] * 255).astype(np.uint8)


def draw_rect(img: np.ndarray, p0: tuple[int, int], p1: tuple[int, int],
             color_bgr: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    if HAS_CV2:
        cv2.rectangle(img, p0, p1, color_bgr, thickness)
        return img
    pil = Image.fromarray(img[..., ::-1])  # BGR -> RGB for PIL
    ImageDraw.Draw(pil).rectangle([p0, p1], outline=tuple(color_bgr[::-1]), width=thickness)
    img[...] = np.array(pil)[..., ::-1]
    return img


def draw_circle(img: np.ndarray, c: tuple[int, int], r: int,
                color_bgr: tuple[int, int, int], filled: bool = True) -> np.ndarray:
    if HAS_CV2:
        cv2.circle(img, c, r, color_bgr, -1 if filled else 1)
        return img
    pil = Image.fromarray(img[..., ::-1])
    x, y = c
    ImageDraw.Draw(pil).ellipse([x - r, y - r, x + r, y + r],
                               fill=tuple(color_bgr[::-1]) if filled else None,
                               outline=tuple(color_bgr[::-1]))
    img[...] = np.array(pil)[..., ::-1]
    return img


def put_text(img: np.ndarray, text: str, org: tuple[int, int],
             color_bgr: tuple[int, int, int]) -> np.ndarray:
    if HAS_CV2:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
        return img
    pil = Image.fromarray(img[..., ::-1])
    ImageDraw.Draw(pil).text(org, text, fill=tuple(color_bgr[::-1]))
    img[...] = np.array(pil)[..., ::-1]
    return img


def imread(path: str) -> np.ndarray:
    """Returns BGR uint8, matching cv2.imread's convention regardless of backend."""
    if HAS_CV2:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img
    return np.array(Image.open(path).convert("RGB"))[..., ::-1].copy()


def imwrite(path: str, img_bgr: np.ndarray) -> None:
    if HAS_CV2:
        cv2.imwrite(path, img_bgr)
        return
    Image.fromarray(img_bgr[..., ::-1]).save(path)
