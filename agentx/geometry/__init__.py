# ============================================================================
# AgentX.geometry - Module 6: Computational Geometry & Image Processing
# ============================================================================

"""Geometry toolkit: convex hull, collision detection and raster processing.

- :func:`convex_hull`      - Andrew's monotone chain (O(n log n), pure Python)
- :func:`hull_area`        - polygon area via the shoelace formula
- :func:`hull_perimeter`   - polygon perimeter
- :func:`point_in_polygon` - ray-casting point location test
- :func:`aabb_overlap`     - axis-aligned bounding box intersection
- :func:`circles_overlap`  - circle intersection
- :func:`polygons_collide` - Separating Axis Theorem (SAT) collision test
- :class:`RasterImage`     - Pillow/NumPy image ops (gray, resize, blur, edges)

Quick start
-----------
>>> from agentx.geometry import convex_hull
>>> hull = convex_hull([(0, 0), (1, 1), (2, 0), (1, -1), (0.5, 0.5)])
>>> hull
[(0, 0), (1, -1), (2, 0), (1, 1)]
"""

from agentx.geometry.collision import (
    aabb_overlap,
    circles_overlap,
    point_in_polygon,
    polygons_collide,
    separation_vector,
)
from agentx.geometry.convex_hull import convex_hull, hull_area, hull_perimeter
from agentx.geometry.raster import RasterImage

__all__ = [
    "convex_hull",
    "hull_area",
    "hull_perimeter",
    "point_in_polygon",
    "aabb_overlap",
    "circles_overlap",
    "polygons_collide",
    "separation_vector",
    "RasterImage",
]
