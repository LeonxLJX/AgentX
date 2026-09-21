# ============================================================================
# AgentX.api.routers.text - TextClassifier endpoints
# ============================================================================

"""Endpoints
----------
- ``POST /api/text/train``    : train on provided or built-in data
- ``POST /api/text/classify`` : classify one or more documents

The classifier is cached in-memory; ``/classify`` auto-trains on the built-in
Chinese ticket dataset on first use so the demo works with zero setup.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentx.core import get_logger
from agentx.core.exceptions import ValidationError
from agentx.textclassifier import TextClassifier
from agentx.textclassifier.dataset import load_ticket

logger = get_logger("agentx.api.text")
router = APIRouter()

# In-memory model cache (demo-grade; production would use a model registry).
_classifier: Optional[TextClassifier] = None


class TrainRequest(BaseModel):
    """Optional training payload; omit to use the built-in demo dataset."""

    texts: Optional[List[str]] = Field(None, description="Training documents")
    labels: Optional[List[str]] = Field(None, description="Per-document labels")
    model: str = Field("logistic", description="logistic | svm")


class ClassifyRequest(BaseModel):
    """Documents to classify."""

    texts: List[str] = Field(..., min_length=1, description="Documents to classify")


class ClassifyResponse(BaseModel):
    """Per-document classification result."""

    results: List[Dict[str, Any]]


class TrainResponse(BaseModel):
    """Training outcome summary."""

    classes: List[str]
    n_samples: int
    model: str


def _get_or_train() -> TextClassifier:
    global _classifier
    if _classifier is None:
        texts, labels = load_ticket()
        _classifier = TextClassifier().fit(texts, labels)
    return _classifier


@router.post("/train", summary="Train the text classifier", response_model=TrainResponse)
def train(payload: Optional[TrainRequest] = None) -> Dict[str, Any]:
    """Train on supplied data or fall back to the built-in dataset."""
    global _classifier
    if payload is not None and payload.texts:
        if not payload.labels or len(payload.texts) != len(payload.labels):
            raise HTTPException(
                status_code=400,
                detail="texts and labels must have the same length",
            )
        try:
            _classifier = TextClassifier(model=payload.model).fit(
                payload.texts, payload.labels
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        texts, labels = load_ticket()
        _classifier = TextClassifier(
            model=payload.model if payload else "logistic"
        ).fit(texts, labels)
    logger.info("text classifier trained: %d classes", len(_classifier.labels_))
    return {
        "classes": _classifier.labels_,
        "n_samples": len(_classifier.labels_) * 9,
        "model": type(_classifier.estimator).__name__,
    }


@router.post("/classify", summary="Classify documents", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest) -> Dict[str, Any]:
    """Return label, confidence and full score distribution per document."""
    clf = _get_or_train()
    results = clf.predict(payload.texts)
    return {"results": [asdict(r) for r in results]}
