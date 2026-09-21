# ============================================================================
# AgentX.optimizer.vrp - Capacitated Vehicle Routing Problem (CVRP)
# ============================================================================

"""Capacitated VRP solver using the Clarke-Wright savings algorithm.

The savings heuristic merges routes greedily while respecting vehicle
capacity: each merge saves ``d(depot,a) + d(depot,b) - d(a,b)``. Routes are
then polished with 2-opt. This is the classic workhorse for delivery
route-planning outsourcing (last-mile logistics, pickup & delivery).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from agentx.core import get_logger
from agentx.core.exceptions import ValidationError
from agentx.core.schema import Route

logger = get_logger("agentx.optimizer.vrp")


class CVRP:
    """Solve a capacitated vehicle routing problem.

    Parameters
    ----------
    demands : sequence[float]
        Demand of each node; ``demands[0]`` is the depot demand (usually 0).
    capacity : float
        Capacity of each identical vehicle.
    coords : sequence[(float, float)], optional
        Node coordinates (Euclidean distances). Mutually exclusive with
        ``distance_matrix``.
    distance_matrix : 2D array-like, optional
        Precomputed distance matrix.
    """

    def __init__(
        self,
        demands: Sequence[float],
        capacity: float,
        coords: Optional[Sequence[Sequence[float]]] = None,
        distance_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.demands = np.asarray(demands, dtype=float)
        self.capacity = float(capacity)
        if self.capacity <= 0:
            raise ValidationError("capacity must be positive")

        if coords is None and distance_matrix is None:
            raise ValidationError("provide either coords or distance_matrix")
        if coords is not None and distance_matrix is not None:
            raise ValidationError("provide coords or distance_matrix, not both")

        if distance_matrix is not None:
            self.dist = np.asarray(distance_matrix, dtype=float)
        else:
            pts = np.asarray(coords, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValidationError("coords must be a list of [x, y] pairs")
            delta = pts[:, None, :] - pts[None, :, :]
            self.dist = np.sqrt((delta ** 2).sum(axis=-1))

        if self.demands.shape[0] != self.dist.shape[0]:
            raise ValidationError("demands and distance matrix sizes must match")

    # --------------------------------------------------------------- public
    def solve(self, depot: int = 0) -> List[Route]:
        """Return a list of vehicle routes.

        Returns
        -------
        list[Route]
            Each route contains the depot at both ends, its total distance
            and total load.
        """
        n = self.dist.shape[0]
        nodes = [i for i in range(n) if i != depot]
        # Separate routes, one per customer.
        routes: List[List[int]] = [[depot, node, depot] for node in nodes]
        loads = {node: float(self.demands[node]) for node in nodes}

        # Savings list: merge two routes saves d(depot,a)+d(depot,b)-d(a,b).
        savings: List[Tuple[float, int, int]] = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                s = self.dist[depot, a] + self.dist[depot, b] - self.dist[a, b]
                savings.append((s, a, b))
        savings.sort(reverse=True, key=lambda x: x[0])

        # Map node -> index of its route.
        position = {node: k for k, route in enumerate(routes) for node in route[1:-1]}

        for s, a, b in savings:
            if s <= 0:
                break
            ra, rb = position.get(a), position.get(b)
            if ra is None or rb is None or ra == rb:
                continue
            combined_load = loads.get(a, 0.0) + loads.get(b, 0.0)
            if combined_load > self.capacity:
                continue
            route_a, route_b = routes[ra], routes[rb]
            if route_a[1] == a and route_b[1] == b:
                merged = route_a[:-1] + route_b[1:]
            elif route_a[-2] == a and route_b[-2] == b:
                merged = route_a[:-1] + route_b[1:]
            elif route_a[1] == a and route_b[-2] == b:
                merged = route_b[:-1] + route_a[1:]
            elif route_a[-2] == a and route_b[1] == b:
                merged = route_a[:-1] + route_b[1:]
            else:
                continue
            # Replace both routes.
            routes = [r for k, r in enumerate(routes) if k not in (ra, rb)]
            routes.append(merged)
            # Rebuild position map for the merged route.
            new_load = 0.0
            for node in merged[1:-1]:
                position[node] = len(routes) - 1
                new_load += self.demands[node]
            loads[a] = new_load
            loads[b] = new_load
            # Repair positions that shifted.
            for k, route in enumerate(routes):
                for node in route[1:-1]:
                    position[node] = k

        # Polish each route with 2-opt and emit Route objects.
        result: List[Route] = []
        for k, route in enumerate(routes):
            if len(route) == 2:  # empty route depot-only
                continue
            polished = self._two_opt(route, depot)
            distance = self._route_distance(polished)
            load = sum(self.demands[node] for node in polished[1:-1])
            result.append(
                Route(
                    vehicle_id=f"vehicle_{k + 1}",
                    stops=polished,
                    total_distance=round(float(distance), 4),
                    total_load=round(float(load), 4),
                )
            )

        logger.info("CVRP solved: %d routes, %d nodes, cap=%.1f", len(result), n - 1, self.capacity)
        return result

    # ------------------------------------------------------------- helpers
    def _route_distance(self, route: Sequence[int]) -> float:
        return float(sum(self.dist[route[i], route[i + 1]] for i in range(len(route) - 1)))

    def _two_opt(self, route: List[int], depot: int, max_passes: int = 20) -> List[int]:
        """2-opt on the customer segment (depot fixed at both ends)."""
        if len(route) <= 4:
            return route
        best = route[:]
        best_d = self._route_distance(best)
        inner = best[1:-1]
        n = len(inner)
        for _ in range(max_passes):
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    seg = inner[: i + 1] + inner[i + 1 : j + 1][::-1] + inner[j + 1 :]
                    candidate = [depot] + seg + [depot]
                    d = self._route_distance(candidate)
                    if d < best_d - 1e-12:
                        best = candidate
                        inner = seg
                        best_d = d
                        improved = True
            if not improved:
                break
        return best
