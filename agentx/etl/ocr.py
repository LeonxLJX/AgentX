# ============================================================================
# AgentX.etl.ocr - OcrEngine (optional pytesseract dependency)
# ============================================================================

"""OCR engine for scan-to-structured-text outsourcing.

Built on ``pytesseract`` (which requires the Tesseract system binary). The
engine detects availability at runtime and raises a clear, actionable error
when Tesseract is missing, so the rest of AgentX keeps working regardless.

Installation
------------
::

    pip install -r requirements-optional.txt      # pytesseract
    # plus the Tesseract binary:
    #   Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #   macOS:   brew install tesseract tesseract-lang
    #   Debian:  apt-get install tesseract-ocr tesseract-ocr-chi-sim
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from agentx.core import get_logger
from agentx.core.exceptions import DependencyError, ETLError, ValidationError

logger = get_logger("agentx.etl.ocr")


class OcrEngine:
    """Extract text from images / scans.

    Parameters
    ----------
    lang : str
        Tesseract language pack(s), e.g. ``"eng"`` or ``"eng+chi_sim"``.
    tesseract_cmd : str, optional
        Explicit path to the ``tesseract`` binary.
    """

    def __init__(self, lang: str = "eng+chi_sim", tesseract_cmd: Optional[str] = None) -> None:
        self.lang = lang
        self.tesseract_cmd = tesseract_cmd
        self._pytesseract: Any = None
        self._available: Optional[bool] = None

    # ------------------------------------------------------------ capability
    def is_available(self) -> bool:
        """Return True when pytesseract is importable.

        A binary probe is skipped for speed; the first :meth:`extract_text`
        call surfaces binary-level problems with guidance.
        """
        if self._available is None:
            try:
                import pytesseract  # type: ignore

                if self.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                self._pytesseract = pytesseract
                self._available = True
            except Exception:
                self._available = False
        return self._available

    # ------------------------------------------------------------- extract
    def extract_text(self, source: Any) -> str:
        """Extract text from an image path, a PIL image or a NumPy array.

        Returns
        -------
        str
            Recognized text (newline separated per detected line).

        Raises
        ------
        DependencyError
            With installation guidance when OCR is unavailable.
        ETLError
            When the source image cannot be parsed.
        """
        if not self.is_available():
            raise DependencyError(
                "OCR is not available. Install pytesseract (requirements-optional.txt) "
                "and the Tesseract binary (see module docstring), then retry."
            )
        image = self._to_pil(source)
        text = self._pytesseract.image_to_string(image, lang=self.lang)
        logger.info("OCR extracted %d characters", len(text))
        return text.strip()

    def extract_lines(self, source: Any) -> List[str]:
        """Extract text and split into non-empty lines (row-like output)."""
        text = self.extract_text(source)
        return [line.strip() for line in text.splitlines() if line.strip()]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _to_pil(source: Any):
        from PIL import Image

        if isinstance(source, str):
            try:
                return Image.open(source)
            except OSError as exc:
                raise ETLError(f"cannot open image at {source}: {exc}") from exc
        if isinstance(source, np.ndarray):
            return Image.fromarray(source)
        if hasattr(source, "convert"):  # already a PIL image
            return source
        raise ValidationError("source must be a path, PIL.Image or numpy array")
