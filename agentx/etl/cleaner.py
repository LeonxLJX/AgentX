# ============================================================================
# AgentX.etl.cleaner - DataCleaner implementation
# ============================================================================

"""Pandas-powered data cleaning with a reproducible report.

Pipeline steps (each also callable individually):

1. ``drop_duplicates`` - exact row / subset duplicates.
2. ``fill_missing``    - numeric columns: median; categorical: mode.
3. ``trim_strings`` / ``normalize_case`` - text hygiene.
4. ``infer_dtypes``    - bool / integer / float / datetime detection.
5. ``detect_outliers`` - IQR fences; ``cap_outliers`` winsorizes them.
6. ``clean()``         - runs the full default pipeline and returns ``self``.

The :meth:`report` gives clients an audit of exactly what changed, which is
the artifact freelance clients usually want alongside the cleaned file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from agentx.core import get_logger
from agentx.core.exceptions import ValidationError
from agentx.core.schema import CleanReport

logger = get_logger("agentx.etl.cleaner")


class DataCleaner:
    """Wrap a DataFrame with cleaning operations.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data. The cleaner mutates an internal copy so the caller's
        frame is never touched.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("DataCleaner expects a pandas.DataFrame")
        self.df = df.copy()
        self._dup_removed = 0
        self._missing_filled: Dict[str, int] = {}
        self._outliers_capped: Dict[str, int] = {}
        self._dtypes_fixed: Dict[str, str] = {}

    # ------------------------------------------------------------- pipeline
    def clean(
        self,
        drop_duplicates: bool = True,
        fill_missing: bool = True,
        trim_strings: bool = True,
        infer_dtypes: bool = True,
        cap_outliers: bool = False,
        outlier_cols: Optional[Sequence[str]] = None,
    ) -> "DataCleaner":
        """Run the default cleaning pipeline and return ``self``."""
        if drop_duplicates:
            self.drop_duplicates()
        if fill_missing:
            self.fill_missing()
        if trim_strings:
            self.trim_strings()
        if infer_dtypes:
            self.infer_dtypes()
        if cap_outliers:
            self.cap_outliers(cols=outlier_cols)
        logger.info("cleaning pipeline complete: %d -> %d rows", self._rows_before(), len(self.df))
        return self

    # ------------------------------------------------------------ cleaning
    def drop_duplicates(
        self, subset: Optional[Sequence[str]] = None, keep: str = "first"
    ) -> "DataCleaner":
        """Remove duplicate rows (optionally on a column subset)."""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        self._dup_removed += before - len(self.df)
        return self

    def fill_missing(
        self, strategy: str = "auto", fill_value: Any = None
    ) -> "DataCleaner":
        """Fill missing values.

        ``strategy="auto"``: median for numeric columns, mode for object/bool
        columns; ``"constant"`` uses ``fill_value``; ``"drop"`` drops rows
        with any missing value.
        """
        if strategy == "drop":
            self.df = self.df.dropna()
            return self
        for col in self.df.columns:
            n_missing = int(self.df[col].isna().sum())
            if n_missing == 0:
                continue
            if strategy == "constant":
                self.df[col] = self.df[col].fillna(fill_value)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                mode = self.df[col].mode()
                if len(mode):
                    self.df[col] = self.df[col].fillna(mode.iloc[0])
                else:
                    self.df[col] = self.df[col].fillna("")
            self._missing_filled[col] = n_missing
        return self

    def trim_strings(self, cols: Optional[Sequence[str]] = None) -> "DataCleaner":
        """Strip whitespace from text columns (optionally a subset)."""
        targets = list(cols) if cols is not None else list(self.df.columns)
        for col in targets:
            if col in self.df.columns and pd.api.types.is_object_dtype(self.df[col]):
                self.df[col] = self.df[col].astype(str).str.strip()
        return self

    def normalize_case(self, cols: Optional[Sequence[str]] = None, case: str = "lower") -> "DataCleaner":
        """Normalize text casing (``lower`` / ``upper`` / ``title``)."""
        if case not in {"lower", "upper", "title"}:
            raise ValidationError("case must be lower/upper/title")
        targets = list(cols) if cols is not None else list(self.df.columns)
        for col in targets:
            if col in self.df.columns and pd.api.types.is_object_dtype(self.df[col]):
                self.df[col] = getattr(self.df[col].astype(str), case)()
        return self

    def infer_dtypes(self) -> "DataCleaner":
        """Heuristically fix column types (bool / int / float / datetime)."""
        for col in self.df.columns:
            if self.df[col].dtype != object:
                continue
            series = self.df[col].dropna().astype(str).str.strip()
            if series.empty:
                continue
            unique = set(series.unique())
            if unique <= {"True", "False", "true", "false", "TRUE", "FALSE", "1", "0"}:
                self.df[col] = self.df[col].map(
                    lambda v: str(v).strip().lower() in {"true", "1"} if pd.notna(v) else v
                )
                self._dtypes_fixed[col] = "bool"
            else:
                try:
                    converted = pd.to_datetime(self.df[col], errors="raise")
                    self.df[col] = converted
                    self._dtypes_fixed[col] = "datetime"
                except (ValueError, TypeError):
                    pass
        # Numeric coercion for whole-column numeric-looking data.
        for col in self.df.columns:
            if self.df[col].dtype == object:
                coerced = pd.to_numeric(self.df[col], errors="coerce")
                if coerced.notna().sum() / max(len(coerced), 1) >= 0.95:
                    self.df[col] = coerced
                    self._dtypes_fixed[col] = str(coerced.dtype)
        return self

    # ------------------------------------------------------------- outliers
    def detect_outliers(
        self,
        cols: Optional[Sequence[str]] = None,
        factor: float = 1.5,
    ) -> Dict[str, List[int]]:
        """IQR-based outlier detection per numeric column.

        Returns
        -------
        dict[str, list[int]]
            Column name -> indices of outlier rows.
        """
        targets = self._numeric_cols(cols)
        result: Dict[str, List[int]] = {}
        for col in targets:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
            mask = (self.df[col] < lo) | (self.df[col] > hi)
            result[col] = self.df.index[mask].tolist()
        return result

    def cap_outliers(
        self,
        cols: Optional[Sequence[str]] = None,
        factor: float = 1.5,
    ) -> "DataCleaner":
        """Winsorize outliers to the IQR fences (in place)."""
        targets = self._numeric_cols(cols)
        for col in targets:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
            mask = (self.df[col] < lo) | (self.df[col] > hi)
            self._outliers_capped[col] = int(mask.sum())
            self.df[col] = self.df[col].clip(lo, hi)
        return self

    # -------------------------------------------------------------- report
    def report(self) -> CleanReport:
        """Audit summary of everything the cleaner changed."""
        return CleanReport(
            rows_before=self._rows_before(),
            rows_after=len(self.df),
            duplicates_removed=self._dup_removed,
            missing_filled=self._missing_filled,
            outliers_capped=self._outliers_capped,
            dtypes_fixed=self._dtypes_fixed,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the cleaned DataFrame (a copy)."""
        return self.df.copy()

    # ------------------------------------------------------------- helpers
    def _rows_before(self) -> int:
        return len(self.df) + self._dup_removed

    def _numeric_cols(self, cols: Optional[Sequence[str]]) -> List[str]:
        if cols is not None:
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise ValidationError(f"unknown columns: {missing}")
            return [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
        return [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
