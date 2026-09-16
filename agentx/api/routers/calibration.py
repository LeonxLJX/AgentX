# ============================================================================
# AgentX.api.routers.calibration - Calibrator endpoints
# ============================================================================

"""Endpoint
---------
- ``POST /api/calibrate`` : generate a demo binary problem, calibrate a model
  and return raw-vs-calibrated quality plus a reliability table.

A deterministic synthetic dataset (``make_classification``) is used so the
demo is reproducible; the same endpoint accepts real features/labels when a
client provides them.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import make_classification

from agentx.calibrator import Calibrator
from agentx.core import get_logger

logger = get_logger("agentx.api.calibration")
router = APIRouter()


class CalibrateRequest(BaseModel):
    """Calibration job description."""

    method: str = Field("isotonic", description="sigmoid | isotonic")
    n_samples: int = Field(600, ge=100, le=20000)
    features: Optional[List[List[float]]] = Field(None, description="Optional custom features")
    labels: Optional[List[int]] = Field(None, description="Optional custom binary labels")


@router.post("/calibrate", summary="Calibrate a binary model's probabilities")
def calibrate(payload: CalibrateRequest) -> dict:
    """Fit a calibrator and report raw vs calibrated scoring metrics."""
    if payload.features is not None or payload.labels is not None:
        if not payload.features or not payload.labels:
            raise HTTPException(400, "features and labels must both be provided")
        X = np.asarray(payload.features, dtype=float)
        y = np.asarray(payload.labels, dtype=int)
        if X.shape[0] != y.shape[0]:
            raise HTTPException(400, "features and labels row counts differ")
        X_eval, y_eval = X, y
    else:
        X, y = make_classification(
            n_samples=payload.n_samples,
            n_features=6,
            n_informative=4,
            n_redundant=1,
            flip_y=0.08,
            random_state=7,
        )
        # Split 70/30 for honest calibration vs evaluation.
        split = int(len(X) * 0.7)
        X, X_eval, y, y_eval = X[:split], X[split:], y[:split], y[split:]

    calibrator = Calibrator(method=payload.method, random_state=7).fit(X, y)
    gain = calibrator.calibration_gain(X_eval, y_eval)
    eval_metrics = calibrator.evaluate(X_eval, y_eval)
    return {
        "method": payload.method,
        "calibration_gain": gain,
        "evaluation": eval_metrics,
    }
