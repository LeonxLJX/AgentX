# ============================================================================
# Tests: Optimizer (TSP, CVRP, knapsack)
# ============================================================================

import numpy as np
import pytest

from agentx.optimizer import CVRP, TSP, knapsack, knapsack_dp, knapsack_greedy


def test_tsp_route_is_closed_cycle():
    coords = [(0, 0), (1, 5), (4, 3), (6, 0), (3, 6), (7, 4)]
    route, distance = TSP(coords=coords).solve()
    assert route[0] == route[-1]  # closed tour
    assert set(route[:-1]) == set(range(len(coords)))  # visits all nodes
    assert distance > 0


def test_tsp_distance_matrix_path():
    dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    route, distance = TSP(distance_matrix=dist).solve()
    assert distance == pytest.approx(4.0)


def test_tsp_requires_input():
    with pytest.raises(ValueError):
        TSP()


def test_cvrp_respects_capacity():
    coords = [(0, 0), (2, 3), (5, 2), (6, 6), (8, 3), (3, 8)]
    demands = [0, 4, 6, 5, 3, 7]
    routes = CVRP(demands=demands, capacity=10, coords=coords).solve()
    for route in routes:
        assert route.total_load <= 10.0
        assert route.stops[0] == 0 and route.stops[-1] == 0  # depot at both ends
    # Every customer is covered exactly once.
    visited = [s for r in routes for s in r.stops[1:-1]]
    assert sorted(visited) == list(range(1, len(coords)))


def test_knapsack_dp_optimal():
    weights = [2, 3, 4, 5, 9]
    values = [3, 4, 5, 8, 10]
    value, items = knapsack_dp(weights, values, capacity=10)
    # Known optimal: items [0, 1, 3] -> 3+4+8 = 15 (w = 2+3+5 = 10).
    assert value == pytest.approx(15.0)
    assert sorted(items) == [0, 1, 3]


def test_knapsack_greedy_runs():
    weights = [2, 3, 4, 5, 9]
    values = [3, 4, 5, 8, 10]
    value, items = knapsack_greedy(weights, values, capacity=10)
    assert value <= 15.0
    assert value > 0


def test_knapsack_entry_point():
    value, _ = knapsack([1, 2, 3], [6, 10, 12], capacity=5, method="dp")
    assert value == pytest.approx(22.0)
    with pytest.raises(ValueError):
        knapsack([1], [1], 1, method="nope")
