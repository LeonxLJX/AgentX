# ============================================================================
# AgentX.core - shared infrastructure for all nine modules
# ============================================================================

"""Shared core utilities for every AgentX module.

Public surface
--------------
- :mod:`agentx.core.logging`   - unified structured logging (text or JSON)
- :mod:`agentx.core.exceptions`- project-wide exception hierarchy
- :mod:`agentx.core.config`    - typed settings (env vars / feature flags)
- :mod:`agentx.core.schema`    - lightweight shared data models
"""

from agentx.core.config import get_settings, reload_settings, settings
from agentx.core.exceptions import (
    AgentXError,
    CalibrationError,
    ConfigError,
    DependencyError,
    ETLError,
    FinanceError,
    GeometryError,
    ModelNotFittedError,
    NLPError,
    OptimizationError,
    PersistenceError,
    SchedulingError,
    TextClassificationError,
    ValidationError,
)
from agentx.core.logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    configure_root_logger,
    get_logger,
)

__all__ = [
    # logging
    "get_logger",
    "configure_root_logger",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    # config
    "settings",
    "get_settings",
    "reload_settings",
    # exceptions
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
