# ============================================================================
# AgentX.api.main - FastAPI application entry point
# ============================================================================

"""Application factory: wires all module routers, CORS, observability and the
static demo UI.

Route map
---------
- ``GET  /api/health``                     - liveness + module status probe
- ``GET  /api/metrics``                    - basic usage metrics
- ``POST /api/text/train|classify``        - TextClassifier
- ``POST /api/nlp/analyze|topics``         - PhaseDetect
- ``POST /api/calibrate``                  - Calibrator
- ``POST /api/schedule``                   - Scheduler
- ``POST /api/optimize/tsp|vrp|knapsack``  - Optimizer
- ``POST /api/geometry/hull|collision``    - Geometry
- ``POST /api/etl/clean``                  - ETLEngine
- ``POST /api/finance/forecast|factors``   - FinanceModels
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

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
from agentx.core import get_logger
from agentx.core.config import settings
from agentx.core.exceptions import AgentXError, ValidationError

logger = get_logger("agentx.api")

STATIC_DIR: Path = __file__.resolve().parent / "static"


# ---------------------------------------------------------------------------
# Response models - shared across endpoints for consistent error reporting.
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Standard error envelope returned for any non-2xx response."""

    error: str = Field(..., description="Short error code / category")
    detail: str = Field(..., description="Human-readable explanation")
    path: str = Field(..., description="Request path that produced the error")


class HealthResponse(BaseModel):
    """Liveness / readiness response with per-module status."""

    status: str = Field(..., description="Aggregate health: 'ok' or 'degraded'")
    service: str = Field("agentx", description="Service name")
    version: str = Field(..., description="Service version")
    env: str = Field(..., description="Effective environment (dev/staging/prod)")
    modules: Dict[str, str] = Field(
        ..., description="Module name -> 'ok' | 'error: <reason>'"
    )


class MetricsResponse(BaseModel):
    """Basic usage metrics for monitoring dashboards."""

    totals: Dict[str, int] = Field(..., description="Total requests per route")
    status_codes: Dict[str, int] = Field(..., description="Response code counts")
    avg_latency_ms: Dict[str, float] = Field(..., description="Avg latency per route")
    uptime_seconds: float = Field(..., description="Process uptime in seconds")


# ---------------------------------------------------------------------------
# In-process metrics store (demo-grade). A production deployment would swap
# this for a Prometheus registry or OpenTelemetry exporter.
# ---------------------------------------------------------------------------
class _MetricsStore:
    """Lightweight in-memory metrics store (process-local, demo-grade)."""

    def __init__(self) -> None:
        self.start_time: float = time.time()
        self.totals: Counter = Counter()
        self.status_codes: Counter = Counter()
        self.latency_sum: Dict[str, float] = {}
        self.latency_count: Dict[str, int] = {}

    def record(self, path: str, status_code: int, latency_ms: float) -> None:
        self.totals[path] += 1
        self.status_codes[str(status_code)] += 1
        self.latency_sum[path] = self.latency_sum.get(path, 0.0) + latency_ms
        self.latency_count[path] = self.latency_count.get(path, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        avg_latency = {
            path: round(self.latency_sum[path] / max(self.latency_count[path], 1), 4)
            for path in self.latency_sum
        }
        return {
            "totals": dict(self.totals),
            "status_codes": dict(self.status_codes),
            "avg_latency_ms": avg_latency,
            "uptime_seconds": round(time.time() - self.start_time, 4),
        }


metrics_store = _MetricsStore()


# ---------------------------------------------------------------------------
# Request / response logging middleware.
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request and response with method, path, status and latency.

    Honors ``AGENTX_ENABLE_REQUEST_LOG``: set to ``false`` to silence in
    high-throughput production deployments.
    """

    async def dispatch(self, request: Request, call_next):
        if not settings.enable_request_log and not settings.enable_metrics:
            return await call_next(request)

        start = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000.0

        # Normalize path so dynamic-looking prefixes still aggregate cleanly.
        path = request.url.path
        metrics_store.record(path, response.status_code, latency_ms)

        if settings.enable_request_log:
            logger.info(
                "%s %s -> %d in %.2fms",
                request.method,
                path,
                response.status_code,
                latency_ms,
            )
        return response


# ---------------------------------------------------------------------------
# Application factory.
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build and return the configured FastAPI application instance."""
    application = FastAPI(
        title="AgentX - Full-Stack AI Outsourcing Toolkit",
        description=(
            "Nine production-ready modules: text classification, NLP phase "
            "detection, probability calibration, job scheduling, path "
            "optimization, computational geometry, ETL/OCR, financial modeling "
            "and this REST API + frontend."
        ),
        version="1.0.0",
        contact={"name": "AgentX Project Contributors"},
        license_info={"name": "MIT"},
    )

    # CORS - configurable via AGENTX_CORS_ORIGINS.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestLoggingMiddleware)

    # Register API routers (explicit routes take precedence over the static mount).
    application.include_router(text.router, prefix="/api/text", tags=["text-classification"])
    application.include_router(nlp.router, prefix="/api/nlp", tags=["nlp-phase-detection"])
    application.include_router(calibration.router, prefix="/api", tags=["calibration"])
    application.include_router(scheduling.router, prefix="/api", tags=["scheduling"])
    application.include_router(optimization.router, prefix="/api/optimize", tags=["optimization"])
    application.include_router(geometry.router, prefix="/api/geometry", tags=["geometry"])
    application.include_router(etl.router, prefix="/api/etl", tags=["etl"])
    application.include_router(finance.router, prefix="/api/finance", tags=["finance"])

    _register_system_routes(application)
    _register_exception_handlers(application)

    # Serve the single-page demo UI (must be mounted last).
    if STATIC_DIR.exists():
        application.mount(
            "/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static"
        )
    else:
        logger.warning("static frontend directory not found: %s", STATIC_DIR)

    return application


def _register_system_routes(application: FastAPI) -> None:
    """Attach /api/health and /api/metrics."""

    @application.get(
        "/api/health",
        tags=["system"],
        summary="Liveness probe with module status checks",
        response_model=HealthResponse,
    )
    def health() -> Dict[str, Any]:
        """Return service status with per-module availability checks.

        Used by Docker healthcheck and load balancers: an ``ok`` aggregate
        status means every algorithm module imports cleanly.
        """
        module_status: Dict[str, str] = {}
        for name, import_path in _MODULE_PROBES:
            try:
                __import__(import_path)
                module_status[name] = "ok"
            except Exception as exc:  # pragma: no cover - defensive
                module_status[name] = f"error: {exc}"
        aggregate = (
            "ok"
            if all(v == "ok" for v in module_status.values())
            else "degraded"
        )
        return {
            "status": aggregate,
            "service": "agentx",
            "version": "1.0.0",
            "env": settings.env,
            "modules": module_status,
        }

    @application.get(
        "/api/metrics",
        tags=["system"],
        summary="Basic usage metrics",
        response_model=MetricsResponse,
    )
    def metrics() -> Dict[str, Any]:
        """Return request counts, status code distribution and average latency.

        Disabled when ``AGENTX_ENABLE_METRICS=false`` - returns 404 in that
        case so monitoring systems detect the change.
        """
        if not settings.enable_metrics:
            return JSONResponse(
                status_code=404,
                content={"error": "metrics_disabled", "detail": "AGENTX_ENABLE_METRICS=false"},
            )
        return metrics_store.snapshot()


def _register_exception_handlers(application: FastAPI) -> None:
    """Map project-level exceptions to consistent JSON error envelopes."""

    @application.exception_handler(ValidationError)
    async def _validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        logger.warning("validation error at %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation_error",
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    @application.exception_handler(AgentXError)
    async def _agentx_handler(request: Request, exc: AgentXError) -> JSONResponse:
        logger.warning("agentx error at %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.__class__.__name__,
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    @application.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error at %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "An unexpected error occurred.",
                "path": request.url.path,
            },
        )


# Module-level probes for the health endpoint - kept here so adding a new
# module only requires a one-line edit.
_MODULE_PROBES = (
    ("textclassifier", "agentx.textclassifier"),
    ("phasedetect", "agentx.phasedetect"),
    ("calibrator", "agentx.calibrator"),
    ("scheduler", "agentx.scheduler"),
    ("optimizer", "agentx.optimizer"),
    ("geometry", "agentx.geometry"),
    ("etl", "agentx.etl"),
    ("finance", "agentx.finance"),
)

# Backwards-compatible module-level app instance (used by uvicorn and tests).
app = create_app()


def run() -> None:
    """Console entry point: ``agentx-api``."""
    import uvicorn

    uvicorn.run(
        "agentx.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )


if __name__ == "__main__":
    run()
