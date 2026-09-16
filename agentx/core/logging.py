# ============================================================================
# AgentX.core.logging - unified structured logging
# ============================================================================

"""Unified logging for every AgentX module.

All modules log through :func:`get_logger` so that the whole project shares
one timestamped, leveled format - clean output for demos, scripts and the API
server alike.

Example
-------
>>> from agentx.core import get_logger
>>> log = get_logger("agentx.textclassifier")
>>> log.info("pipeline trained on %d samples", 1200)
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "agentx", level: int = logging.INFO) -> logging.Logger:
    """Return a project-wide logger with a consistent console handler.

    Parameters
    ----------
    name : str
        Logger name, conventionally ``"agentx.<module>"``.
    level : int
        Minimum logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        A configured logger instance (handlers are attached only once).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
