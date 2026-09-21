# ============================================================================
# AgentX.core.exceptions - project-wide exception hierarchy
# ============================================================================

"""Custom exception types for AgentX.

Each module raises a specific subclass of :class:`AgentXError` so callers can
catch failures with the right granularity and the REST API layer can map them
to appropriate HTTP status codes without inspecting message strings.

Hierarchy
---------
- :class:`AgentXError`                    - base class for everything
    - :class:`ConfigError`               - misconfiguration / missing env vars
    - :class:`ValidationError`           - invalid input values
    - :class:`ModelNotFittedError`       - estimator used before fit
    - :class:`PersistenceError`          - save / load failures
    - :class:`DependencyError`           - missing optional dependency
    - :class:`SchedulingError`           - cycle / unknown dep / deadlock
    - :class:`OptimizationError`         - infeasible optimization inputs
    - :class:`GeometryError`             - degenerate / invalid geometry
    - :class:`ETLError`                  - data cleaning / OCR failure
    - :class:`FinanceError`              - short series / unsupported model
    - :class:`CalibrationError`          - calibration specific failures
    - :class:`NLPError`                  - NLP / phase-detection failures
    - :class:`TextClassificationError`   - classifier specific failures

Example
-------
>>> from agentx.core.exceptions import ModelNotFittedError
>>> raise ModelNotFittedError("calibrator is not fitted - call .fit(X, y) first")
"""

from __future__ import annotations


class AgentXError(Exception):
    """Base class for every AgentX-specific exception.

    Catching :class:`AgentXError` is the safe way to detect any failure that
    originates from this project (as opposed to a library or the standard
    library).
    """


# ---------------------------------------------------------------- config
class ConfigError(AgentXError):
    """Raised when required configuration / environment is missing or invalid."""


# -------------------------------------------------------------- validation
class ValidationError(AgentXError):
    """Raised when user-supplied input values fail validation.

    This is the project-level analogue of :class:`pydantic.ValidationError`
    for code paths that are not at the API boundary.
    """


# --------------------------------------------------------------- lifecycle
class ModelNotFittedError(AgentXError):
    """Raised when an estimator is used before being fitted / trained."""


class PersistenceError(AgentXError):
    """Raised when an artifact cannot be saved or loaded."""


# --------------------------------------------------------------- dependency
class DependencyError(AgentXError):
    """Raised when an optional dependency is required but not installed."""


# --------------------------------------------------------------- modules
class TextClassificationError(AgentXError):
    """Text-classifier specific failure (unknown model, empty corpus, ...)."""


class NLPError(AgentXError):
    """Phase-detection / NLP specific failure."""


class CalibrationError(AgentXError):
    """Probability-calibration specific failure."""


class SchedulingError(AgentXError):
    """Scheduling failure (cycles, unknown deps, deadlocks)."""


class OptimizationError(AgentXError):
    """Optimization failure (infeasible inputs, bad distance matrices)."""


class GeometryError(AgentXError):
    """Computational-geometry failure (degenerate inputs, etc.)."""


class ETLError(AgentXError):
    """ETL / data-cleaning / OCR failure."""


class FinanceError(AgentXError):
    """Financial-modeling failure (short series, unsupported model, ...)."""


__all__ = [
    "AgentXError",
    "ConfigError",
    "ValidationError",
    "ModelNotFittedError",
    "PersistenceError",
    "DependencyError",
    "TextClassificationError",
    "NLPError",
    "CalibrationError",
    "SchedulingError",
    "OptimizationError",
    "GeometryError",
    "ETLError",
    "FinanceError",
]
