# ============================================================================
# AgentX.etl - Module 7: Data Cleaning & OCR Pipeline
# ============================================================================

"""ETL engine for data-cleaning and document-digitization outsourcing.

- :class:`DataCleaner` : deduplication, missing-value handling, string
  normalization, type inference, outlier detection/capping and a one-shot
  ``clean()`` pipeline with a full report.
- :class:`OcrEngine`   : OCR wrapper (pytesseract, optional) turning scans
  and images into structured text rows.

Quick start
-----------
>>> import pandas as pd
>>> from agentx.etl import DataCleaner
>>> df = pd.DataFrame({"name": ["A", "A", "b"], "age": [30, 30, None]})
>>> cleaner = DataCleaner(df).clean()
>>> cleaner.to_dataframe().shape
(2, 2)
"""

from agentx.etl.cleaner import DataCleaner
from agentx.etl.ocr import OcrEngine

__all__ = ["DataCleaner", "OcrEngine"]
