# ============================================================================
# AgentX.finance.timeseries - time-series primitives (NumPy only)
# ============================================================================

"""Core financial time-series transforms.

All functions are pure NumPy, vectorized and NaN-aware, so they work on
real-world price series that contain gaps. Every windowed function returns an
array of the same length (window edges are NaN) for easy alignment.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from agentx.core import get_logger
from agentx.core.exceptions import ValidationError

logger = get_logger("agentx.finance.timeseries")


def _to_array(prices: np.ndarray, *, name: str = "prices") -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array, got shape {arr.shape}")
    return arr


def returns(prices: np.ndarray) -> np.ndarray:
    """Simple period-over-period returns: ``p[t]/p[t-1] - 1``."""
    arr = _to_array(prices)
    out = np.full_like(arr, np.nan)
    if len(arr) >= 2:
        out[1:] = arr[1:] / arr[:-1] - 1.0
    return out


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns: ``log(p[t]/p[t-1])`` (additive across time)."""
    arr = _to_array(prices)
    out = np.full_like(arr, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        if len(arr) >= 2:
            out[1:] = np.log(arr[1:] / arr[:-1])
    return out


def sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average (centered alignment preserved via same length)."""
    arr = _to_array(prices)
    out = np.full_like(arr, np.nan)
    if window < 1 or len(arr) < window:
        return out
    cum = np.cumsum(np.insert(arr, 0, 0.0))
    out[window - 1 :] = (cum[window:] - cum[:-window]) / window
    return out


def ema(prices: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (``span``-based smoothing factor)."""
    arr = _to_array(prices)
    out = np.full_like(arr, np.nan)
    if len(arr) == 0 or span < 1:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = arr[0]
    for t in range(1, len(arr)):
        out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
    return out


def rolling_volatility(returns_arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling sample standard deviation of returns (NaN-aware)."""
    arr = _to_array(returns_arr, name="returns")
    out = np.full_like(arr, np.nan)
    if window < 2:
        return out
    for t in range(window - 1, len(arr)):
        out[t] = np.nanstd(arr[t - window + 1 : t + 1])
    return out


def ewma_volatility(returns_arr: np.ndarray, span: int = 20) -> np.ndarray:
    """RiskMetrics-style exponentially weighted volatility.

    ``sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2``
    with ``lambda = 1 - 2/(span+1)``.
    """
    arr = _to_array(returns_arr, name="returns")
    out = np.full_like(arr, np.nan)
    if len(arr) < 2:
        return out
    lam = 1.0 - 2.0 / (span + 1.0)
    var = arr[1] ** 2
    out[1] = np.sqrt(var)
    for t in range(2, len(arr)):
        var = lam * var + (1 - lam) * arr[t] ** 2
        out[t] = np.sqrt(var)
    return out


def zscore(series: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: ``(x - rolling_mean) / rolling_std``."""
    arr = _to_array(series, name="series")
    out = np.full_like(arr, np.nan)
    if window < 2:
        return out
    for t in range(window - 1, len(arr)):
        seg = arr[t - window + 1 : t + 1]
        m, s = np.nanmean(seg), np.nanstd(seg)
        if s > 0:
            out[t] = (arr[t] - m) / s
    return out


def linear_trend(series: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares linear trend.

    Returns
    -------
    tuple[float, float, float]
        ``(slope, intercept, r_squared)`` over the time index 0..n-1.
    """
    arr = _to_array(series, name="series")
    x = np.arange(len(arr), dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return 0.0, float(np.nanmean(arr)) if mask.sum() else 0.0, 0.0
    coeffs = np.polyfit(x[mask], arr[mask], 1)
    slope, intercept = coeffs
    pred = slope * x + intercept
    ss_res = np.nansum((arr - pred) ** 2)
    ss_tot = np.nansum((arr - np.nanmean(arr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def autocorrelation(series: np.ndarray, lag: int = 1) -> float:
    """Pearson autocorrelation of a series at a given lag (NaN-aware)."""
    arr = _to_array(series, name="series")
    if lag < 0:
        raise ValidationError(f"lag must be >= 0, got {lag}")
    if len(arr) <= lag:
        return float("nan")
    x = arr[:-lag] if lag > 0 else arr
    y = arr[lag:] if lag > 0 else arr
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return float("nan")
    sx, sy = np.std(x[mask]), np.std(y[mask])
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x[mask], y[mask])[0, 1])
