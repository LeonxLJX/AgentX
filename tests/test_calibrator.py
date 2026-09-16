# ============================================================================
# Tests: Calibrator & metrics
# ============================================================================

import numpy as np
import pytest
from sklearn.datasets import make_classification

from agentx.calibrator import Calibrator
from agentx.calibrator.metrics import brier_score, plot_reliability, reliability_curve


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=800, n_features=6, n_informative=4,
        flip_y=0.1, random_state=3,
    )
    split = int(len(X) * 0.7)
    return X[:split], X[split:], y[:split], y[split:]


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_improves_or_keeps_brier(method, binary_data):
    X_tr, X_te, y_tr, y_te = binary_data
    cal = Calibrator(method=method, random_state=3).fit(X_tr, y_tr)
    proba = cal.predict_proba(X_te)[:, 1]
    gain = cal.calibration_gain(X_te, y_te)
    assert gain["calibrated"]["brier"] <= gain["raw"]["brier"] + 0.01
    assert 0.0 <= float(np.nanmean(proba)) <= 1.0


def test_predict_with_threshold(binary_data):
    X_tr, X_te, y_tr, y_te = binary_data
    cal = Calibrator().fit(X_tr, y_tr)
    preds = cal.predict(X_te, threshold=0.5)
    assert set(np.unique(preds)) <= {0, 1}


def test_brier_score_definition():
    y = np.array([0, 1, 1])
    p = np.array([0.0, 1.0, 1.0])
    assert brier_score(y, p) == pytest.approx(0.0)
    assert brier_score(y, np.array([0.5, 0.5, 0.5])) == pytest.approx(0.25)


def test_reliability_curve_bins():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9])
    curve = reliability_curve(y, p, n_bins=4)
    assert curve
    for row in curve:
        assert "bin_center" in row and "fraction_positive" in row


def test_plot_reliability_creates_png(tmp_path):
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    out = tmp_path / "rel.png"
    path = plot_reliability(y, p, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    assert path == str(out)
