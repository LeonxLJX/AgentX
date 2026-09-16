# ============================================================================
# AgentX.geometry.raster - raster image processing
# ============================================================================

"""Image processing built on Pillow + NumPy (OpenCV optional).

Provides the operations most frequently requested in small image jobs:
grayscale conversion, resizing, rotation, blur, brightness, histograms and
edge detection. All operations return ``self`` so transforms can be chained.

When OpenCV is installed, convolution-style ops transparently use it; the
module degrades gracefully to NumPy otherwise - no hard dependency.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

from agentx.core import get_logger

logger = get_logger("agentx.geometry.raster")

# Detect OpenCV lazily.
try:  # pragma: no cover - environment dependent
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


class RasterImage:
    """A chainable image-processing wrapper.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Internal image object (use the classmethods to construct).
    """

    def __init__(self, pil_image: Image.Image) -> None:
        self._img: Image.Image = pil_image

    # ------------------------------------------------------------ construct
    @classmethod
    def from_file(cls, path: str) -> "RasterImage":
        """Load an image from disk."""
        return cls(Image.open(path))

    @classmethod
    def from_array(cls, arr: Any) -> "RasterImage":
        """Build from a NumPy array (HxW or HxWxC, uint8 preferred)."""
        array = np.asarray(arr)
        if array.ndim == 2:
            return cls(Image.fromarray(array.astype(np.uint8), mode="L"))
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return cls(Image.fromarray(array.astype(np.uint8)))
        raise ValueError("array must be HxW or HxWxC (C in {3, 4})")

    # ------------------------------------------------------------ accessors
    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)`` of the current image."""
        return self._img.size

    @property
    def mode(self) -> str:
        """Pillow mode, e.g. ``RGB``, ``L``, ``RGBA``."""
        return self._img.mode

    def to_array(self) -> np.ndarray:
        """Return the image as a NumPy array."""
        return np.asarray(self._img)

    def copy(self) -> "RasterImage":
        """Deep copy for non-destructive pipelines."""
        return RasterImage(self._img.copy())

    # ------------------------------------------------------------ transforms
    def grayscale(self) -> "RasterImage":
        """Convert to 8-bit grayscale (mode ``L``)."""
        self._img = self._img.convert("L")
        return self

    def resize(self, width: int, height: int) -> "RasterImage":
        """Resize to an exact ``(width, height)`` using LANCZOS resampling."""
        self._img = self._img.resize((int(width), int(height)), Image.LANCZOS)
        return self

    def rotate(self, degrees: float, expand: bool = True) -> "RasterImage":
        """Rotate by ``degrees`` counter-clockwise (canvas expands by default)."""
        self._img = self._img.rotate(degrees, expand=expand, resample=Image.BICUBIC)
        return self

    def gaussian_blur(self, radius: float = 2.0) -> "RasterImage":
        """Apply a Gaussian blur with the given pixel radius."""
        self._img = self._img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
        return self

    def brightness(self, factor: float = 1.2) -> "RasterImage":
        """Scale brightness; ``factor > 1`` brightens, ``< 1`` darkens."""
        import PIL.ImageEnhance

        self._img = PIL.ImageEnhance.Brightness(self._img).enhance(float(factor))
        return self

    def contrast(self, factor: float = 1.2) -> "RasterImage":
        """Scale contrast; ``factor > 1`` increases contrast."""
        import PIL.ImageEnhance

        self._img = PIL.ImageEnhance.Contrast(self._img).enhance(float(factor))
        return self

    # ------------------------------------------------------------ analysis
    def histogram(self, bins: int = 16) -> Dict[str, List[int]]:
        """Per-channel intensity histogram.

        Returns
        -------
        dict
            ``{"channels": [{"name": "R", "counts": [...]}, ...]}`` or a single
            ``"L"`` channel for grayscale images.
        """
        arr = self.to_array()
        if arr.ndim == 2:
            hist, _ = np.histogram(arr, bins=bins, range=(0, 256))
            return {"channels": [{"name": "L", "counts": hist.tolist()}]}
        names = ("R", "G", "B", "A")[: arr.shape[2]]
        return {
            "channels": [
                {"name": name, "counts": np.histogram(arr[:, :, i], bins=bins, range=(0, 256))[0].tolist()}
                for i, name in enumerate(names)
            ]
        }

    def sobel_edges(self) -> np.ndarray:
        """Sobel edge magnitude as a float array in [0, 1].

        Uses OpenCV when available, NumPy convolution otherwise. Input is
        converted to grayscale internally; the original image is untouched.
        """
        gray = self._img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        if _HAS_CV2:
            sobel_x = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        else:
            kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            ky = kx.T
            sobel_x = _convolve2d(arr, kx)
            sobel_y = _convolve2d(arr, ky)
        mag = np.hypot(sobel_x, sobel_y)
        return mag / (mag.max() + 1e-9)

    # ------------------------------------------------------------ output
    def save(self, path: str) -> str:
        """Write the current image to ``path`` (format inferred from suffix)."""
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._img.save(path)
        logger.info("image saved to %s (size=%s)", path, self.size)
        return path

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RasterImage size={self.size} mode={self.mode}>"


def _convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Minimal 2D convolution with zero padding (used when OpenCV is absent)."""
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(k):
        for j in range(k):
            out += kernel[i, j] * padded[i : i + arr.shape[0], j : j + arr.shape[1]]
    return out
