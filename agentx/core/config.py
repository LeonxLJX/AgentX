# ============================================================================
# AgentX.core.config - centralized configuration management
# ============================================================================

"""Application configuration via environment variables.

Configuration is loaded from environment variables (or ``.env``) into a
singleton :class:`Settings` instance. Every module reads configuration through
:data:`settings`, so ops teams can flip feature flags and tune knobs without
code changes.

Supported variables
-------------------
- ``AGENTX_ENV``              : ``dev`` | ``staging`` | ``prod`` (default ``dev``)
- ``AGENTX_DEBUG``            : truthy value enables debug-level logging
- ``AGENTX_LOG_LEVEL``        : ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR``
- ``AGENTX_LOG_FORMAT``       : ``text`` (default) or ``json`` (for log
  shippers like Loki / CloudWatch).
- ``AGENTX_HOST``             : API bind host (default ``0.0.0.0``).
- ``AGENTX_PORT``             : API bind port (default ``8000``).
- ``AGENTX_CORS_ORIGINS``     : comma-separated allowed origins (default ``*``).
- ``AGENTX_ENABLE_METRICS``   : expose ``/api/metrics`` (default ``true``).
- ``AGENTX_ENABLE_REQUEST_LOG``: log every request through middleware
  (default ``true``).
- ``AGENTX_MODEL_CACHE_SIZE`` : in-memory classifier cache size (default ``8``).

Example
-------
>>> from agentx.core.config import settings
>>> settings.api_port
8000
>>> settings.is_production
False
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

# Pydantic v2 split the settings layer into a dedicated package. We prefer
# ``pydantic_settings`` and fall back to ``pydantic`` for older installs so
# the project remains importable without the optional dependency.
try:  # pragma: no cover - exercised in environments with/without the package
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = dict  # type: ignore
    _HAS_PYDANTIC_SETTINGS = False


def _truthy(value: object) -> bool:
    """Interpret env-style strings as booleans."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseSettings):  # type: ignore[misc]
    """Typed application configuration.

    All fields have sensible defaults so the project runs out of the box in a
    dev environment. Production deployments override values via environment
    variables (or a ``.env`` file at the repo root).
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ env
    env: str = "dev"
    debug: bool = False

    # --------------------------------------------------------------- logging
    log_level: str = "INFO"
    log_format: str = "text"  # "text" | "json"

    # ------------------------------------------------------------------ api
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    # ------------------------------------------------------------- features
    enable_metrics: bool = True
    enable_request_log: bool = True
    model_cache_size: int = 8

    # ----------------------------------------------------------------- OCR
    ocr_default_lang: str = "eng+chi_sim"

    # -------------------------------------------------------- derived props
    @property
    def is_production(self) -> bool:
        return self.env.lower() == "prod"

    @property
    def is_development(self) -> bool:
        return self.env.lower() in {"dev", "development", "local"}

    @property
    def numeric_log_level(self) -> int:
        """Resolve :attr:`log_level` to a :mod:`logging` level integer."""
        if self.debug:
            return logging.DEBUG
        return getattr(logging, self.log_level.upper(), logging.INFO)

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse :attr:`cors_origins` into a list of allowed origins."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def apply_to_logging(self) -> None:
        """Reconfigure the root AgentX logger to honor current settings.

        Called by :mod:`agentx.core.logging` on first import; safe to call
        again at runtime after env vars change.
        """
        from agentx.core.logging import configure_root_logger

        configure_root_logger(
            level=self.numeric_log_level,
            json_format=(self.log_format.lower() == "json"),
        )


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Return the cached :class:`Settings` singleton.

    The cache means reads are free; call :func:`reload_settings` after
    changing env vars at runtime if you need a refresh.
    """
    if _HAS_PYDANTIC_SETTINGS:
        return Settings()
    # Minimal fallback when pydantic-settings is not installed: build a
    # Settings-like object straight from environment variables.
    return _FallbackSettings()  # type: ignore[return-value]


def reload_settings() -> "Settings":
    """Force re-reading the environment (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()


class _FallbackSettings(Settings):  # type: ignore[misc]
    """Plain-object fallback used when pydantic-settings is unavailable.

    Reads ``AGENTX_*`` environment variables directly. The shape mirrors the
    pydantic-backed :class:`Settings` so downstream code does not need to
    branch on the implementation.
    """

    def __init__(self) -> None:  # noqa: D401 - simple init
        self.env = os.environ.get("AGENTX_ENV", "dev")
        self.debug = _truthy(os.environ.get("AGENTX_DEBUG", "false"))
        self.log_level = os.environ.get("AGENTX_LOG_LEVEL", "INFO")
        self.log_format = os.environ.get("AGENTX_LOG_FORMAT", "text")
        self.host = os.environ.get("AGENTX_HOST", "0.0.0.0")
        self.port = int(os.environ.get("AGENTX_PORT", "8000"))
        self.cors_origins = os.environ.get("AGENTX_CORS_ORIGINS", "*")
        self.enable_metrics = _truthy(os.environ.get("AGENTX_ENABLE_METRICS", "true"))
        self.enable_request_log = _truthy(
            os.environ.get("AGENTX_ENABLE_REQUEST_LOG", "true")
        )
        self.model_cache_size = int(os.environ.get("AGENTX_MODEL_CACHE_SIZE", "8"))
        self.ocr_default_lang = os.environ.get("AGENTX_OCR_DEFAULT_LANG", "eng+chi_sim")


# Module-level singleton importable as ``from agentx.core.config import settings``.
settings = get_settings()


__all__ = ["Settings", "settings", "get_settings", "reload_settings"]
