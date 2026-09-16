# ============================================================================
# Example: Optimizer - TSP, CVRP and knapsack
# Run:  python examples/demo_optimizer.py
# ============================================================================

"""Solves a small delivery TSP, a capacitated VRP and a knapsack instance."""

from agentx.core.schema import Route
from agentx.optimizer import CVRP, TSP, knapsack

if __name__ == "__main__":
    # --- TSP ----------------------------------------------------------------
    coords = [(0, 0), (1, 5), (4, 3), (6, 0), (3, 6), (7, 4), (2, 7)]
    route, distance = TSP(coords=coords).solve()
    print(f"TSP route: {route}  distance={distance:.2f}")

    # --- CVRP ---------------------------------------------------------------
    vrp_coords = [(0, 0), (2, 3), (5, 2), (6, 6), (8, 3), (3, 8), (9, 7), (1, 6)]
    demands = [0, 4, 6, 5, 3, 7, 4, 5]
    routes: list[Route] = CVRP(demands=demands, capacity=12, coords=vrp_coords).solve()
    total = sum(r.total_distance for r in routes)
    print(f"\nCVRP: {len(routes)} vehicles, total distance={total:.2f}")
    for r in routes:
        print(f"  {r.vehicle_id}: stops={r.stops} load={r.total_load} dist={r.total_distance}")

    # --- Knapsack -----------------------------------------------------------
    value, items = knapsack([2, 3, 4, 5, 9], [3, 4, 5, 8, 10], capacity=10)
    print(f"\nKnapsack: value={value} items={items}")
