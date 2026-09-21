# ============================================================================
# AgentX.api.routers.geometry - Geometry endpoints
# ============================================================================

"""Endpoints
----------
- ``POST /api/geometry/hull``      : convex hull + area + perimeter
- ``POST /api/geometry/collision`` : SAT collision test + MTV
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agentx.core import get_logger
from agentx.geometry import convex_hull, hull_area, hull_perimeter, separation_vector

logger = get_logger("agentx.api.geometry")
router = APIRouter()


class HullRequest(BaseModel):
    """Point cloud to hull."""

    points: List[List[float]] = Field(..., min_length=3, description="[[x, y], ...]")


class CollisionRequest(BaseModel):
    """Two convex polygons to test."""

    poly_a: List[List[float]] = Field(..., min_length=3)
    poly_b: List[List[float]] = Field(..., min_length=3)


@router.post("/hull", summary="Compute the convex hull")
def hull(payload: HullRequest) -> Dict[str, Any]:
    """Return hull vertices, area and perimeter."""
    points = [tuple(p) for p in payload.points]
    hull_pts = convex_hull(points)
    logger.info("hull computed: %d input -> %d vertices", len(points), len(hull_pts))
    return {
        "hull": [list(p) for p in hull_pts],
        "area": round(hull_area(hull_pts), 4),
        "perimeter": round(hull_perimeter(hull_pts), 4),
        "n_points": len(points),
    }


@router.post("/collision", summary="Test polygon collision (SAT)")
def collision(payload: CollisionRequest) -> Dict[str, Any]:
    """Return whether the polygons collide and the minimum translation vector."""
    poly_a = [tuple(p) for p in payload.poly_a]
    poly_b = [tuple(p) for p in payload.poly_b]
    hit, mtv = separation_vector(poly_a, poly_b)
    logger.info("collision test: hit=%s", hit)
    return {
        "collide": hit,
        "mtv": list(mtv) if mtv is not None else None,
    }
