# ============================================================================
# AgentX.optimizer - Module 5: Path & Combinatorial Optimization
# ============================================================================

"""Optimization solvers for delivery / routing / resource-allocation jobs.

- :class:`TSP`      - travelling salesman problem (nearest-neighbour + 2-opt)
- :class:`CVRP`     - capacitated vehicle routing (Clarke-Wright savings)
- :func:`knapsack`  - 0/1 knapsack (exact DP + greedy approximation)

All solvers are deterministic, dependency-light and return JSON-friendly
results for the REST API.

Quick start
-----------
>>> from agentx.optimizer import TSP
>>> coords = [(0, 0), (1, 5), (4, 3), (6, 0), (3, 6)]
>>> tsp = TSP(coords=coords)
>>> route, distance = tsp.solve()
>>> route
[0, 3, 1, 4, 2, 0]
"""

from agentx.optimizer.knapsack import knapsack, knapsack_dp, knapsack_greedy
from agentx.optimizer.tsp import TSP
from agentx.optimizer.vrp import CVRP

__all__ = ["TSP", "CVRP", "knapsack", "knapsack_dp", "knapsack_greedy"]
