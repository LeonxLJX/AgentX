# ============================================================================
# AgentX.core.logging - unified structured logging
# ============================================================================

"""Unified logging for every AgentX module.

All modules log through :func:`get_logger` so that the whole project shares
one timestamped, leveled format - clean output for demos, scripts and the API
server alike. For production / log-shipping setups, switch
``AGENTX_LOG_FORMAT=json`` to emit one log record per line as JSON (Loki,
CloudWatch, Datadog friendly).

Example
-------
>>> from agentx.core import get_logger
>>> log = get_logger("agentx.textclassifier")
>>> log.info("pipeline trained on %d samples", 1200)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Constants - expose log levels under stable names so callers do not need to
# import :mod:`logging` just to set a level.
# ---------------------------------------------------------------------------
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False  # guards against re-attaching handlers on re-import


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON for log shippers.

    Reserved fields: ``timestamp``, ``level``, ``logger``, ``message``.
    Keyword args passed via ``logger.info(..., extra={...})`` are merged in,
    and ``exc_info`` is rendered under ``exception``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge user-supplied extras (skipping stdlib/internal attributes).
        reserved = set(vars(record).keys()) - {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
        }
        for key in reserved:
            payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def _build_handler(json_format: bool) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def configure_root_logger(
    level: int = INFO,
    json_format: bool = False,
    force: bool = False,
) -> None:
    """Configure the ``agentx`` root logger once.

    Parameters
    ----------
    level : int
        Minimum log level (default :data:`logging.INFO`).
    json_format : bool
        Emit records as JSON when ``True`` (default text format).
    force : bool
        Re-attach handlers even if the logger was already configured.
    """
    global _CONFIGURED
    root = logging.getLogger("agentx")
    if _CONFIGURED and not force:
        root.setLevel(level)
        return

    # Replace any existing handlers so re-configurations take effect cleanly.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(_build_handler(json_format))
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(
    name: str = "agentx",
    level: Optional[int] = None,
) -> logging.Logger:
    """Return a project-wide logger with a consistent console handler.

    Parameters
    ----------
    name : str
        Logger name, conventionally ``"agentx.<module>"``.
    level : int, optional
        Minimum logging level. Defaults to whatever the root ``agentx``
        logger is configured with (controlled by :mod:`agentx.core.config`).

    Returns
    -------
    logging.Logger
        A configured logger instance. Handlers are attached to the root
        ``agentx`` logger only, so children inherit them automatically.
    """
    # Ensure the root agentx logger is configured at least once.
    if not _CONFIGURED:
        try:
            from agentx.core.config import settings  # local import to avoid cycle
            configure_root_logger(
                level=settings.numeric_log_level,
                json_format=(settings.log_format.lower() == "json"),
            )
        except Exception:  # pragma: no cover - settings not ready
            configure_root_logger()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    logger.propagate = True  # children bubble up to the root agentx logger
    return logger


__all__ = [
    "get_logger",
    "configure_root_logger",
    "JsonFormatter",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]
