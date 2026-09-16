# ============================================================================
# AgentX.api.routers.nlp - PhaseDetect endpoints
# ============================================================================

"""Endpoints
----------
- ``POST /api/nlp/analyze`` : sentiment, keywords and entities for one text
- ``POST /api/nlp/topics``  : fit topic clusters over a document set
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agentx.core import get_logger
from agentx.phasedetect import PhaseDetector

logger = get_logger("agentx.api.nlp")
router = APIRouter()

_detector = PhaseDetector(language="auto")


class AnalyzeRequest(BaseModel):
    """Text to analyze."""

    text: str = Field(..., min_length=1, description="Document to analyze")
    language: str = Field("auto", description="auto | en | zh")


class TopicsRequest(BaseModel):
    """Document set to cluster into topics."""

    texts: List[str] = Field(..., min_length=2)
    n_topics: int = Field(3, ge=1, le=20)


@router.post("/analyze", summary="Detect sentiment / keywords / entities")
def analyze(payload: AnalyzeRequest) -> dict:
    """Full phase-detection pass over a single document."""
    detector = PhaseDetector(language=payload.language)
    result = detector.analyze(payload.text)
    return asdict(result)


@router.post("/topics", summary="Cluster documents into topics")
def topics(payload: TopicsRequest) -> dict:
    """Fit TF-IDF + K-means topics and return per-document assignments."""
    detector = PhaseDetector(language="auto")
    detector.fit_topics(payload.texts, n_topics=payload.n_topics)
    assignments = []
    for text in payload.texts:
        cid, terms = detector.predict_topic(text)
        assignments.append({"text": text, "topic_id": cid, "topic_terms": terms})
    return {
        "n_topics": detector.topic_model_["n_topics"],
        "clusters": [
            {"topic_id": i, "terms": terms}
            for i, terms in enumerate(detector.topic_terms_)
        ],
        "assignments": assignments,
    }
