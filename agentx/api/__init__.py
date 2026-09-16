# ============================================================================
# AgentX.api - Module 9: REST API + Web Frontend
# ============================================================================

"""FastAPI backend exposing all eight modules as JSON endpoints.

Interactive documentation is available at ``/docs`` (Swagger UI) and
``/redoc``. A single-page demo frontend is served at ``/``.

Run locally
-----------
::

    uvicorn agentx.api.main:app --reload --port 8000
    # or:  python -m agentx.api.main
"""

from agentx.api.main import app, run

__all__ = ["app", "run"]
