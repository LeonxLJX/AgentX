# ============================================================================
# AgentX.api.routers - module API endpoints
# ============================================================================

"""Router package: each module ships one router module.

Importing ``agentx.api.routers`` triggers no side effects beyond defining the
``APIRouter`` instances consumed by :mod:`agentx.api.main`.
"""

from agentx.api.routers import (
    calibration,
    etl,
    finance,
    geometry,
    nlp,
    optimization,
    scheduling,
    text,
)

__all__ = [
    "text",
    "nlp",
    "calibration",
    "scheduling",
    "optimization",
    "geometry",
    "etl",
    "finance",
]
