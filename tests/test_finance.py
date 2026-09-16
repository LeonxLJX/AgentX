# ============================================================================
# Tests: FinanceModels
# ============================================================================

import numpy as np
import pytest

from agentx.finance.factors import (
    bollinger_bands,
    max_drawdown,
    momentum,
    rsi,
    sharpe_ratio,
)
from agentx.finance.models import ForecastModel
from agentx.finance.timeseries import ema, returns, sma, zscore


@pytest.fixture(scope="module")
def prices():
    # Stationary AR(1) process around 100 - the regime Bollinger bands
    # describe; a strong trend would legitimately break the band.
    rng = np.random.default_rng(7)
    p = np.empty(150)
    p[0] = 100.0
    for t in range(1, 150):
        p[t] = 0.9 * (p[t - 1] - 100.0) + 100.0 + rng.normal(0, 1.5)
    return p


def test_returns_shape_and_first_nan(prices):
    r = returns(prices)
    assert len(r) == len(prices)
    assert np.isnan(r[0])
    assert np.all(np.isfinite(r[1:]))


def test_sma_and_ema(prices):
    s = sma(prices, 20)
    assert np.isnan(s[:19]).all() and np.isfinite(s[19:]).all()
    e = ema(prices, 20)
    assert np.all(np.isfinite(e))


def test_rsi_in_range(prices):
    r = rsi(prices, 14)
    valid = r[~np.isnan(r)]
    assert valid.min() >= 0.0 and valid.max() <= 100.0


def test_bollinger_contains_price(prices):
    # A 2-sigma band holds ~95% of values in theory; with a short tail window
    # and autocorrelation, require at least 80% to stay robust.
    mid, up, lo = bollinger_bands(prices, window=20, num_std=2)
    tail = slice(-30, None)
    coverage = float(((lo[tail] <= prices[tail]) & (prices[tail] <= up[tail])).mean())
    assert coverage >= 0.8


def test_momentum(prices):
    m = momentum(prices, 20)
    assert np.isnan(m[:19]).all()
    assert np.isfinite(m[20:]).all()


def test_max_drawdown_bounds(prices):
    dd = max_drawdown(prices)
    assert 0.0 <= dd <= 1.0


def test_sharpe_ratio_finite(prices):
    r = returns(prices)
    assert np.isfinite(sharpe_ratio(r))


def test_forecast_ar_predicts(prices):
    model = ForecastModel(model="ar", lags=5).fit(prices, horizon=10)
    points = model.predict()
    assert len(points) == 10
    assert all(p.predicted is not None for p in points)
    assert all(p.index == f"t+{i + 1}" for i, p in enumerate(points))


def test_forecast_evaluate_metrics(prices):
    model = ForecastModel(model="linear")
    metrics = model.evaluate(prices, test_size=10, horizon=10)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["n_test"] == 10


def test_forecast_backtest(prices):
    model = ForecastModel(model="ar", lags=3)
    bt = model.backtest(prices, horizon=5)
    assert bt["rmse"] >= 0.0 and bt["windows"] >= 1


def test_forecast_before_fit_raises():
    model = ForecastModel()
    with pytest.raises(RuntimeError):
        model.predict()


def test_invalid_model_rejected():
    with pytest.raises(ValueError):
        ForecastModel(model="nope")
