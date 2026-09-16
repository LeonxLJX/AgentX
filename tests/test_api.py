# ============================================================================
# Tests: REST API smoke tests (requires: pip install -e ".[dev]")
# ============================================================================

"""End-to-end smoke tests through FastAPI's TestClient.

These exercise every router end to end with minimal payloads. They are
skipped automatically when ``httpx`` (a dev dependency) is missing.
"""

import pytest

httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from agentx.api.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_text_classify():
    res = client.post("/api/text/classify", json={"texts": ["When will my package arrive?"]})
    assert res.status_code == 200
    body = res.json()
    assert body["results"][0]["label"] in {"refund", "logistics", "account", "tech"}


def test_nlp_analyze():
    res = client.post(
        "/api/nlp/analyze",
        json={"text": "The service is extremely good, email me at a@b.com", "language": "en"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["sentiment"] == "positive"
    assert "EMAIL" in body["entities"]


def test_calibrate():
    res = client.post("/api/calibrate", json={"method": "sigmoid", "n_samples": 300})
    assert res.status_code == 200
    body = res.json()
    assert "calibration_gain" in body and "evaluation" in body


def test_schedule():
    jobs = [
        {"job_id": "A", "duration": 2, "priority": 2},
        {"job_id": "B", "duration": 3, "priority": 1, "dependencies": ["A"]},
    ]
    res = client.post("/api/schedule", json={"machines": 2, "jobs": jobs})
    assert res.status_code == 200
    body = res.json()
    assert body["makespan"] == pytest.approx(3.0)


def test_optimize_tsp():
    res = client.post(
        "/api/optimize/tsp", json={"coords": [[0, 0], [1, 5], [4, 3], [6, 0]]}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["route"][0] == body["route"][-1]


def test_optimize_vrp():
    res = client.post(
        "/api/optimize/vrp",
        json={
            "coords": [[0, 0], [2, 3], [5, 2], [6, 6]],
            "demands": [0, 4, 5, 3],
            "capacity": 10,
        },
    )
    assert res.status_code == 200
    assert res.json()["n_vehicles"] >= 1


def test_optimize_knapsack():
    res = client.post(
        "/api/optimize/knapsack",
        json={"weights": [2, 3, 4], "values": [3, 4, 5], "capacity": 5, "method": "dp"},
    )
    assert res.status_code == 200
    assert res.json()["value"] == pytest.approx(7.0)


def test_geometry_hull():
    res = client.post(
        "/api/geometry/hull",
        json={"points": [[0, 0], [1, 1], [2, 0], [1, -1]]},
    )
    assert res.status_code == 200
    assert len(res.json()["hull"]) >= 3


def test_geometry_collision():
    res = client.post(
        "/api/geometry/collision",
        json={
            "poly_a": [[0, 0], [4, 0], [4, 4], [0, 4]],
            "poly_b": [[3, 1], [7, 1], [7, 5], [3, 5]],
        },
    )
    assert res.status_code == 200
    assert res.json()["collide"] is True


def test_etl_clean():
    rows = [
        {"name": "A", "age": 30},
        {"name": "A", "age": 30},
        {"name": "B", "age": None},
    ]
    res = client.post("/api/etl/clean", json={"rows": rows, "options": {}})
    assert res.status_code == 200
    body = res.json()
    assert body["report"]["duplicates_removed"] == 1
    assert len(body["rows"]) == 2


def test_finance_forecast():
    import numpy as np

    prices = list(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 40))))
    res = client.post(
        "/api/finance/forecast",
        json={"prices": prices, "model": "ar", "horizon": 5, "lags": 3},
    )
    assert res.status_code == 200
    assert len(res.json()["points"]) == 5


def test_finance_factors():
    import numpy as np

    prices = list(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 60))))
    res = client.post("/api/finance/factors", json={"prices": prices})
    assert res.status_code == 200
    assert "rsi_14" in res.json()
