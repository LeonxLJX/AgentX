# ============================================================================
# Tests: Geometry
# ============================================================================

import numpy as np
import pytest

from agentx.geometry import (
    RasterImage,
    aabb_overlap,
    circles_overlap,
    convex_hull,
    hull_area,
    hull_perimeter,
    point_in_polygon,
    polygons_collide,
    separation_vector,
)


def test_convex_hull_square():
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
    hull = convex_hull(pts)
    assert len(hull) == 4
    assert hull_area(hull) == pytest.approx(1.0)
    assert hull_perimeter(hull) == pytest.approx(4.0)


def test_convex_hull_needs_three_points():
    with pytest.raises(ValueError):
        convex_hull([(0, 0), (1, 1)])


def test_collinear_points_collapse():
    hull = convex_hull([(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)])
    assert len(hull) == 3


def test_point_in_polygon():
    square = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert point_in_polygon((2, 2), square)
    assert not point_in_polygon((5, 5), square)
    assert point_in_polygon((4, 2), square)  # boundary counts


def test_aabb_overlap():
    assert aabb_overlap((0, 0), (2, 2), (1, 1), (3, 3))
    assert not aabb_overlap((0, 0), (1, 1), (2, 2), (3, 3))


def test_circles_overlap():
    assert circles_overlap((0, 0), 1.0, (1.5, 0), 1.0)
    assert not circles_overlap((0, 0), 1.0, (5, 0), 1.0)


def test_sat_collision_and_mtv():
    square_a = [(0, 0), (4, 0), (4, 4), (0, 4)]
    square_b = [(3, 1), (7, 1), (7, 5), (3, 5)]
    hit, mtv = separation_vector(square_a, square_b)
    assert hit
    assert mtv is not None
    # Moving exactly along the MTV leaves the polygons touching (SAT counts
    # touching as overlap); a 1% overshoot must fully separate them.
    dx, dy = mtv
    moved_b = [(x + dx * 1.01, y + dy * 1.01) for x, y in square_b]
    assert not polygons_collide(square_a, moved_b)


def test_sat_no_collision():
    square_a = [(0, 0), (2, 0), (2, 2), (0, 2)]
    square_b = [(5, 5), (7, 5), (7, 7), (5, 7)]
    assert not polygons_collide(square_a, square_b)


def test_raster_pipeline():
    rng = np.random.default_rng(0)
    arr = (rng.normal(128, 40, size=(40, 50, 3))).clip(0, 255).astype(np.uint8)
    img = RasterImage.from_array(arr).grayscale().resize(25, 20).gaussian_blur(1.0)
    assert img.size == (25, 20)
    assert img.mode == "L"
    edges = img.sobel_edges()
    assert edges.shape == (20, 25)
    assert edges.min() >= 0.0 and edges.max() <= 1.0
    hist = img.histogram()
    assert hist["channels"][0]["name"] == "L"


def test_raster_save(tmp_path):
    rng = np.random.default_rng(1)
    arr = (rng.normal(100, 30, size=(20, 20, 3))).clip(0, 255).astype(np.uint8)
    out = tmp_path / "img.png"
    RasterImage.from_array(arr).save(str(out))
    assert out.exists() and out.stat().st_size > 0
