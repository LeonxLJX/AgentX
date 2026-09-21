# ============================================================================
# AgentX.api.routers.optimization - Optimizer endpoints
# ============================================================================

"""Endpoints
----------
- ``POST /api/optimize/tsp``       : travelling-salesman tour
- ``POST /api/optimize/vrp``       : capacitated vehicle routing
- ``POST /api/optimize/knapsack``  : 0/1 knapsack (dp | greedy)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentx.core import get_logger
from agentx.optimizer import CVRP, TSP, knapsack

logger = get_logger("agentx.api.optimization")
router = APIRouter()


class TspRequest(BaseModel):
    """Coordinates of the nodes to visit."""

    coords: List[List[float]] = Field(..., min_length=2, description="[[x, y], ...]")
    start: Optional[int] = Field(None, description="Optional depot index")


class VrpRequest(BaseModel):
    """Capacitated vehicle routing problem."""

    coords: List[List[float]] = Field(..., min_length=2, description="Node coords (index 0 = depot)")
    demands: List[float] = Field(..., min_length=2, description="Per-node demand")
    capacity: float = Field(..., gt=0)


class KnapsackRequest(BaseModel):
    """0/1 knapsack problem."""

    weights: List[float] = Field(..., min_length=1)
    values: List[float] = Field(..., min_length=1)
    capacity: float = Field(..., gt=0)
    method: str = Field("dp", description="dp | greedy")


@router.post("/tsp", summary="Solve a travelling salesman problem")
def solve_tsp(payload: TspRequest) -> Dict[str, Any]:
    """Return the optimized tour (depot at both ends) and its distance."""
    solver = TSP(coords=payload.coords)
    route, distance = solver.solve(start=payload.start)
    logger.info("tsp solved: %d nodes, distance=%.4f", len(payload.coords), distance)
    return {"route": route, "distance": round(distance, 4), "n_nodes": len(payload.coords)}


@router.post("/vrp", summary="Solve a capacitated vehicle routing problem")
def solve_vrp(payload: VrpRequest) -> Dict[str, Any]:
    """Return per-vehicle routes with distance and load."""
    if len(payload.demands) != len(payload.coords):
        raise HTTPException(400, "demands and coords must have the same length")
    solver = CVRP(
        demands=payload.demands,
        capacity=payload.capacity,
        coords=payload.coords,
    )
    routes = solver.solve(depot=0)
    total_distance = sum(r.total_distance for r in routes)
    logger.info(
        "vrp solved: %d nodes, %d vehicles, distance=%.4f",
        len(payload.coords),
        len(routes),
        total_distance,
    )
    return {
        "routes": [asdict(r) for r in routes],
        "total_distance": round(total_distance, 4),
        "n_vehicles": len(routes),
    }


@router.post("/knapsack", summary="Solve a 0/1 knapsack problem")
def solve_knapsack(payload: KnapsackRequest) -> Dict[str, Any]:
    """Return optimal (or greedy) value and the chosen item indices."""
    if len(payload.weights) != len(payload.values):
        raise HTTPException(400, "weights and values must have the same length")
    value, items = knapsack(payload.weights, payload.values, payload.capacity, method=payload.method)
    logger.info(
        "knapsack solved: %d items, value=%.4f method=%s",
        len(payload.weights),
        value,
        payload.method,
    )
    return {
        "method": payload.method,
        "value": round(value, 4),
        "items": items,
        "n_items": len(payload.weights),
    }
