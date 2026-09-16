# ============================================================================
# AgentX.optimizer.knapsack - 0/1 knapsack solver
# ============================================================================

"""0/1 knapsack: exact dynamic programming + greedy approximation.

The exact DP is optimal for item counts up to a few thousand; the greedy
value/weight heuristic gives a fast upper-bound-quality answer for huge
instances. Both are typical "resource allocation" building blocks clients
ask for (budget selection, cargo loading, campaign mix).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from agentx.core import get_logger

logger = get_logger("agentx.optimizer.knapsack")


def knapsack_dp(
    weights: Sequence[float],
    values: Sequence[float],
    capacity: float,
) -> Tuple[float, List[int]]:
    """Exact 0/1 knapsack via dynamic programming.

    Integer weights are required for the classic DP table; float weights are
    scaled to integers automatically.

    Returns
    -------
    tuple[float, list[int]]
        Optimal total value and the list of chosen item indices.
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(w) != len(v):
        raise ValueError("weights and values must have the same length")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")

    # Scale float weights to integers.
    scale = 1.0
    if w.size and (w % 1).any():
        decimals = max(0, -int(np.floor(np.log10(np.min(w[w > 0])))) + 2) if (w > 0).any() else 0
        scale = 10 ** decimals
    int_w = np.round(w * scale).astype(np.int64)
    cap = int(round(capacity * scale))

    n = len(int_w)
    # 2D table: dp[i, c] = best value using the first i items and capacity c.
    dp = np.zeros((n + 1, cap + 1), dtype=np.float64)
    for i in range(1, n + 1):
        wi, vi = int(int_w[i - 1]), float(v[i - 1])
        dp[i] = dp[i - 1]  # default: skip item i
        if wi <= cap:
            take = dp[i - 1, : cap + 1 - wi] + vi
            better = take > dp[i - 1, wi:]
            dp[i, wi:] = np.where(better, take, dp[i, wi:])

    # Backtrack: an item was taken iff the row value changed at its weight.
    remaining = cap
    items: List[int] = []
    for i in range(n, 0, -1):
        wi = int(int_w[i - 1])
        if dp[i, remaining] != dp[i - 1, remaining]:
            items.append(i - 1)
            remaining -= wi
    items.reverse()

    logger.info(
        "knapsack DP: n=%d cap=%d value=%.2f items=%d",
        n, cap, float(dp[n, cap]), len(items),
    )
    return float(dp[n, cap]), items


def knapsack_greedy(
    weights: Sequence[float],
    values: Sequence[float],
    capacity: float,
) -> Tuple[float, List[int]]:
    """Greedy approximation by value/weight ratio (fast, near-optimal).

    Returns
    -------
    tuple[float, list[int]]
        Achieved value and chosen item indices.
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(w) != len(v):
        raise ValueError("weights and values must have the same length")

    order = sorted(
        range(len(w)),
        key=lambda i: (v[i] / w[i]) if w[i] > 0 else float("inf"),
        reverse=True,
    )
    total_w = 0.0
    total_v = 0.0
    items: List[int] = []
    for i in order:
        if total_w + w[i] <= capacity:
            total_w += w[i]
            total_v += v[i]
            items.append(i)
    items.sort()
    return float(total_v), items


def knapsack(
    weights: Sequence[float],
    values: Sequence[float],
    capacity: float,
    method: str = "dp",
) -> Tuple[float, List[int]]:
    """Unified entry point.

    Parameters
    ----------
    method : {"dp", "greedy"}
        ``"dp"`` (default, exact) or ``"greedy"`` (approximation).
    """
    if method == "dp":
        return knapsack_dp(weights, values, capacity)
    if method == "greedy":
        return knapsack_greedy(weights, values, capacity)
    raise ValueError("method must be 'dp' or 'greedy'")
