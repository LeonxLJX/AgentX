# ============================================================================
# AgentX.core.schema - lightweight, dependency-free data models
# ============================================================================

"""Lightweight data models shared across modules.

These are plain ``dataclasses`` (standard library only) so every module stays
portable. The REST API layer maps them to/from Pydantic request/response models
without coupling the algorithm modules to the web stack.

Included models
---------------
- ``ClassificationResult``  : single text-classification prediction
- ``NlpPhase``              : one sentiment / topic / entity detection result
- ``Job``                   : one schedulable job
- ``ScheduleEntry``         : one scheduled slot (for Gantt rendering)
- ``Route``                 : one vehicle route (for map rendering)
- ``CleanReport``           : summary of an ETL cleaning pass
- ``ForecastPoint``         : one point of a financial forecast
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClassificationResult:
    """Prediction of the text classifier for one document."""

    text: str
    label: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class NlpPhase:
    """Result of one phase-detection pass over a document."""

    text: str
    sentiment: str
    sentiment_score: float
    topics: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)


@dataclass
class Job:
    """A schedulable unit of work."""

    job_id: str
    duration: float
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    resources: float = 1.0
    deadline: Optional[float] = None
    weight: float = 1.0


@dataclass
class ScheduleEntry:
    """One job placed on the timeline."""

    job_id: str
    start: float
    end: float
    resource: float = 1.0
    machine: int = 0


@dataclass
class Route:
    """One vehicle route: an ordered list of stops."""

    vehicle_id: str
    stops: List[Any] = field(default_factory=list)  # stop identifiers / coords
    total_distance: float = 0.0
    total_load: float = 0.0


@dataclass
class CleanReport:
    """Statistics produced by an ETL cleaning pass."""

    rows_before: int
    rows_after: int
    duplicates_removed: int
    missing_filled: Dict[str, int] = field(default_factory=dict)
    outliers_capped: Dict[str, int] = field(default_factory=dict)
    dtypes_fixed: Dict[str, str] = field(default_factory=dict)


@dataclass
class ForecastPoint:
    """One point of a time-series forecast."""

    index: str
    actual: Optional[float] = None
    predicted: Optional[float] = None
