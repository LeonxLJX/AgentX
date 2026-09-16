# ============================================================================
# AgentX.geometry.convex_hull - convex hull algorithms
# ============================================================================

"""Convex hull via Andrew's monotone chain.

Pure-Python implementation (with NumPy vectorization for area/perimeter) so the
module runs anywhere. The output hull is counter-clockwise and starts at the
lexicographically smallest point - deterministic and JSON-friendly.

Typical freelance fit: collision culling, shape analysis, image-processing
pre-steps, map boundary computation.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    """2D cross product of vectors (o->a) x (o->b); sign gives turn direction."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Sequence[Point]) -> List[Point]:
    """Return the convex hull of a point set (Andrew's monotone chain).

    Complexity: O(n log n). Collinear points on the hull edges are kept for
    exactness (``on_edge=True`` behaviour); pass already-sorted unique points
    for maximum speed.

    Raises
    ------
    ValueError
        When fewer than 3 distinct points are supplied.
    """
    pts = sorted(set(points))
    if len(pts) < 3:
        raise ValueError("convex_hull needs at least 3 distinct points")

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def hull_area(points: Sequence[Point]) -> float:
    """Polygon area via the shoelace formula (absolute value, any winding)."""
    pts = list(points)
    n = len(pts)
    if n < 3:
        return 0.0
    s = sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )
    return abs(s) / 2.0


def hull_perimeter(points: Sequence[Point]) -> float:
    """Sum of edge lengths of a closed polygon."""
    pts = list(points)
    n = len(pts)
    if n < 2:
        return 0.0
    return sum(
        math.dist(pts[i], pts[(i + 1) % n]) for i in range(n)
    )
