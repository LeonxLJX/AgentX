# ============================================================================
# AgentX.api - Module 9: REST API + Web Frontend
# ============================================================================

"""FastAPI backend exposing all eight modules as JSON endpoints.

Interactive documentation is available at ``/docs`` (Swagger UI) and
``/redoc``. A single-page demo frontend is served at ``/``.

System endpoints
----------------
- ``GET /api/health``  - liveness probe with per-module status checks
- ``GET /api/metrics`` - per-route request counts, status codes and latency

Configuration
-------------
All runtime behavior is controlled via ``AGENTX_*`` environment variables
(see :mod:`agentx.core.config` for the full list).

Run locally
-----------
::

    uvicorn agentx.api.main:app --reload --port 8000
    # or:  python -m agentx.api.main
"""

from agentx.api.main import app, create_app, run

__all__ = ["app", "create_app", "run"]
