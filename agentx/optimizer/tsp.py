# ============================================================================
# AgentX.optimizer.tsp - Travelling Salesman Problem solver
# ============================================================================

"""Heuristic TSP solver: nearest-neighbour construction + 2-opt improvement.

The nearest-neighbour tour is fast and produces a decent starting point; the
2-opt local search repeatedly removes two crossing edges and reconnects the
path until no improvement is found. For typical delivery-scale instances
(< 200 stops) this yields tours within a few percent of optimal in a few
milliseconds - ideal for route-planning freelance jobs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from agentx.core import get_logger

logger = get_logger("agentx.optimizer.tsp")


class TSP:
    """Solve a symmetric TSP instance.

    Parameters
    ----------
    coords : sequence[(float, float)], optional
        Node coordinates; the Euclidean distance matrix is computed
        internally. Mutually exclusive with ``distance_matrix``.
    distance_matrix : 2D array-like, optional
        Precomputed symmetric distance matrix (e.g. road-network distances).
    """

    def __init__(
        self,
        coords: Optional[Sequence[Sequence[float]]] = None,
        distance_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if coords is None and distance_matrix is None:
            raise ValueError("provide either coords or distance_matrix")
        if coords is not None and distance_matrix is not None:
            raise ValueError("provide coords or distance_matrix, not both")

        if distance_matrix is not None:
            self.dist = np.asarray(distance_matrix, dtype=float)
        else:
            pts = np.asarray(coords, dtype=float)
            delta = pts[:, None, :] - pts[None, :, :]
            self.dist = np.sqrt((delta ** 2).sum(axis=-1))

        n = self.dist.shape[0]
        if self.dist.shape != (n, n):
            raise ValueError("distance_matrix must be square")
        self.n = n

    # --------------------------------------------------------------- public
    def solve(self, start: Optional[int] = None, improve: bool = True) -> Tuple[List[int], float]:
        """Return a ``(route, total_distance)`` tour.

        Parameters
        ----------
        start : int, optional
            Index of the depot / start node (defaults to the node nearest to
            the others, which usually shortens the final tour).
        improve : bool
            Whether to run the 2-opt improvement pass (default True).
        """
        if self.n == 1:
            return [0], 0.0

        if start is None:
            start = int(np.argmin(self.dist.sum(axis=1)))
        tour = self._nearest_neighbour(int(start))
        if improve:
            tour = self._two_opt(tour)
        # Normalize: start the reported tour at the depot.
        idx = tour.index(int(start))
        tour = tour[idx:] + tour[:idx]
        tour.append(int(start))
        distance = self._tour_distance(tour)
        logger.info("TSP solved: n=%d distance=%.2f", self.n, distance)
        return tour, float(distance)

    # ------------------------------------------------------------- helpers
    def _nearest_neighbour(self, start: int) -> List[int]:
        visited = {start}
        tour = [start]
        current = start
        while len(visited) < self.n:
            best = None
            best_d = float("inf")
            for nxt in range(self.n):
                if nxt not in visited and self.dist[current, nxt] < best_d:
                    best_d = self.dist[current, nxt]
                    best = nxt
            visited.add(best)
            tour.append(best)
            current = best
        return tour

    def _tour_distance(self, tour: Sequence[int]) -> float:
        return float(sum(self.dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))

    def _two_opt(self, tour: List[int], max_passes: int = 50) -> List[int]:
        best = tour[:]
        best_d = self._tour_distance(best)
        n = len(best)
        for _ in range(max_passes):
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    a, b, c, d = best[i], best[i + 1], best[j], best[(j + 1) % n]
                    delta = self.dist[a, c] + self.dist[b, d] - self.dist[a, b] - self.dist[c, d]
                    if delta < -1e-12:
                        best = best[: i + 1] + best[i + 1 : j + 1][::-1] + best[j + 1 :]
                        best_d += delta
                        improved = True
            if not improved:
                break
        return best
