# ============================================================================
# AgentX.api.main - FastAPI application entry point
# ============================================================================

"""Application factory: wires all module routers, CORS and the static demo UI.

Route map
---------
- ``GET  /api/health``                     - liveness probe
- ``POST /api/text/train|classify``        - TextClassifier
- ``POST /api/nlp/analyze``                - PhaseDetect
- ``POST /api/calibrate``                  - Calibrator
- ``POST /api/schedule``                   - Scheduler
- ``POST /api/optimize/tsp|vrp|knapsack``  - Optimizer
- ``POST /api/geometry/hull|collision``    - Geometry
- ``POST /api/etl/clean``                  - ETLEngine
- ``POST /api/finance/forecast|factors``   - FinanceModels
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

logger = get_logger("agentx.api")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers (explicit routes take precedence over the static mount).
app.include_router(text.router, prefix="/api/text", tags=["text-classification"])
app.include_router(nlp.router, prefix="/api/nlp", tags=["nlp-phase-detection"])
app.include_router(calibration.router, prefix="/api", tags=["calibration"])
app.include_router(scheduling.router, prefix="/api", tags=["scheduling"])
app.include_router(optimization.router, prefix="/api/optimize", tags=["optimization"])
app.include_router(geometry.router, prefix="/api/geometry", tags=["geometry"])
app.include_router(etl.router, prefix="/api/etl", tags=["etl"])
app.include_router(finance.router, prefix="/api/finance", tags=["finance"])


@app.get("/api/health", tags=["system"], summary="Liveness probe")
def health() -> dict:
    """Return service status - used by Docker healthcheck."""
    return {"status": "ok", "service": "agentx", "version": "1.0.0"}


# Serve the single-page demo UI (must be mounted last).
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    logger.warning("static frontend directory not found: %s", STATIC_DIR)


def run() -> None:
    """Console entry point: ``agentx-api``."""
    port = int(os.environ.get("AGENTX_PORT", "8000"))
    uvicorn.run("agentx.api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
