# ============================================================================
# AgentX.geometry.collision - collision detection primitives
# ============================================================================

"""Collision detection: AABB, circles, point-in-polygon and SAT.

The Separating Axis Theorem (SAT) implementation handles convex polygons and
also returns the minimum translation vector (MTV) so a caller can *resolve*
a collision, not just detect it - the pattern game/physics clients expect.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

Point = Tuple[float, float]
Polygon = Sequence[Point]


def aabb_overlap(
    a_min: Point, a_max: Point, b_min: Point, b_max: Point
) -> bool:
    """Axis-aligned bounding box intersection test.

    ``a_min`` / ``a_max`` are the (min_x, min_y) and (max_x, max_y) corners.
    """
    return not (
        a_max[0] < b_min[0] or b_max[0] < a_min[0]
        or a_max[1] < b_min[1] or b_max[1] < a_min[1]
    )


def circles_overlap(c1: Point, r1: float, c2: Point, r2: float) -> bool:
    """Circle intersection test (touching counts as overlap)."""
    return math.dist(c1, c2) <= r1 + r2


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test (works for concave polygons).

    Boundary points count as inside.
    """
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Boundary check first.
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-12 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def _edge_axes(polygon: Polygon):
    """Normalized normals of every polygon edge (candidate SAT axes)."""
    axes = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        edge = (x2 - x1, y2 - y1)
        # Normal = (-edge_y, edge_x), normalized.
        length = math.hypot(edge[0], edge[1])
        if length < 1e-12:
            continue
        axes.append((-edge[1] / length, edge[0] / length))
    return axes


def _project(polygon: Polygon, axis: Point) -> Tuple[float, float]:
    """Project a polygon onto an axis, returning (min, max)."""
    dots = [p[0] * axis[0] + p[1] * axis[1] for p in polygon]
    return min(dots), max(dots)


def separation_vector(
    poly_a: Polygon, poly_b: Polygon
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """SAT collision test with minimum translation vector.

    Returns
    -------
    (bool, MTV or None)
        ``(True, (dx, dy))`` when the polygons collide, with the minimal
        translation needed to separate them; ``(False, None)`` otherwise.
    """
    best_overlap = float("inf")
    best_axis: Optional[Tuple[float, float]] = None
    for axis in _edge_axes(poly_a) + _edge_axes(poly_b):
        lo_a, hi_a = _project(poly_a, axis)
        lo_b, hi_b = _project(poly_b, axis)
        overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
        if overlap < 0:
            return False, None  # separating axis found -> no collision
        if overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis
    # Push along the axis pointing from A's center to B's center.
    if best_axis is not None:
        cx_a = sum(p[0] for p in poly_a) / len(poly_a)
        cy_a = sum(p[1] for p in poly_a) / len(poly_a)
        cx_b = sum(p[0] for p in poly_b) / len(poly_b)
        cy_b = sum(p[1] for p in poly_b) / len(poly_b)
        dx, dy = cx_b - cx_a, cy_b - cy_a
        if dx * best_axis[0] + dy * best_axis[1] < 0:
            best_axis = (-best_axis[0], -best_axis[1])
        return True, (best_axis[0] * best_overlap, best_axis[1] * best_overlap)
    return False, None


def polygons_collide(poly_a: Polygon, poly_b: Polygon) -> bool:
    """Boolean SAT collision test for two convex polygons."""
    hit, _ = separation_vector(poly_a, poly_b)
    return hit
