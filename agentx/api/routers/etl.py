# ============================================================================
# AgentX.api.routers.etl - ETLEngine endpoints
# ============================================================================

"""Endpoint
---------
- ``POST /api/etl/clean`` : upload rows as a JSON list, run the cleaning
  pipeline, and get the cleaned rows plus an audit report.

This is the "Excel/CSV data cleaning" endpoint clients hit most: dedupe,
fill missing, normalize text, infer dtypes and cap outliers.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from agentx.core import get_logger
from agentx.etl import DataCleaner

logger = get_logger("agentx.api.etl")
router = APIRouter()


class CleanOptions(BaseModel):
    """Which pipeline steps to run."""

    drop_duplicates: bool = True
    fill_missing: bool = True
    trim_strings: bool = True
    infer_dtypes: bool = True
    cap_outliers: bool = False


class CleanRequest(BaseModel):
    """Tabular data to clean (list of records)."""

    rows: List[Dict[str, Any]] = Field(..., min_length=1, description="[{'col': value}, ...]")
    options: CleanOptions = Field(default_factory=CleanOptions)


@router.post("/clean", summary="Clean tabular data")
def clean(payload: CleanRequest) -> Dict[str, Any]:
    """Run the cleaning pipeline and return cleaned rows + audit report."""
    df = pd.DataFrame(payload.rows)
    cleaner = DataCleaner(df)
    cleaner.clean(
        drop_duplicates=payload.options.drop_duplicates,
        fill_missing=payload.options.fill_missing,
        trim_strings=payload.options.trim_strings,
        infer_dtypes=payload.options.infer_dtypes,
        cap_outliers=payload.options.cap_outliers,
    )
    cleaned = cleaner.to_dataframe()
    # JSON-safe serialization (dates -> ISO strings, NaN -> None).
    cleaned = cleaned.astype(object).where(pd.notna(cleaned), None)
    cleaned = cleaned.apply(
        lambda col: col.map(lambda v: v.isoformat() if hasattr(v, "isoformat") else v)
    )
    report = cleaner.report()
    logger.info(
        "etl clean complete: rows_in=%d rows_out=%d deduped=%d",
        len(payload.rows),
        len(cleaned),
        report.duplicates_removed,
    )
    return {
        "report": asdict(report),
        "columns": [str(c) for c in cleaned.columns],
        "rows": cleaned.to_dict(orient="records"),
    }
