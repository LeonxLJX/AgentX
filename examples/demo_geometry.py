# ============================================================================
# Example: Geometry - convex hull, collision detection, image processing
# Run:  python examples/demo_geometry.py
# ============================================================================

"""Runs the convex hull, SAT collision and raster image pipeline."""

import numpy as np

from agentx.geometry import (
    RasterImage,
    convex_hull,
    hull_area,
    hull_perimeter,
    point_in_polygon,
    polygons_collide,
    separation_vector,
)

if __name__ == "__main__":
    # --- Convex hull ---------------------------------------------------------
    points = [(0, 0), (1, 1), (2, 0), (1, -1), (0.5, 0.5), (3, 2), (2, 3), (-1, 1)]
    hull = convex_hull(points)
    print(f"hull = {hull}")
    print(f"area = {hull_area(hull):.2f}  perimeter = {hull_perimeter(hull):.2f}")
    print(f"point (1, 0) inside hull? {point_in_polygon((1, 0), hull)}")

    # --- Collision detection (SAT) ------------------------------------------
    square_a = [(0, 0), (4, 0), (4, 4), (0, 4)]
    square_b = [(3, 1), (7, 1), (7, 5), (3, 5)]  # overlapping square
    hit, mtv = separation_vector(square_a, square_b)
    print(f"\ncollision={hit} mtv={mtv}")
    print(f"polygons_collide={polygons_collide(square_a, square_b)}")

    # --- Raster image pipeline ----------------------------------------------
    rng = np.random.default_rng(1)
    arr = (rng.normal(128, 40, size=(60, 80, 3))).clip(0, 255).astype(np.uint8)
    img = (
        RasterImage.from_array(arr)
        .grayscale()
        .gaussian_blur(1.5)
        .brightness(1.2)
        .resize(40, 30)
    )
    print(f"\nimage size={img.size} mode={img.mode}")
    hist = img.histogram()
    print(f"histogram channels={[c['name'] for c in hist['channels']]}")
    edges = img.sobel_edges()
    print(f"sobel edges shape={edges.shape} range=[{edges.min():.3f}, {edges.max():.3f}]")
    img.save("outputs/processed_demo.png")
    print("saved outputs/processed_demo.png")
