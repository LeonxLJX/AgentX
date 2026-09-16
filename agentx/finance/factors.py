# ============================================================================
# AgentX.finance.factors - alpha-factor library
# ============================================================================

"""Classic cross-sectional / single-asset factors used in quant pipelines.

- :func:`momentum`             - past-window total return
- :func:`mean_reversion_zscore`- distance from the moving average
- :func:`rsi`                  - Relative Strength Index
- :func:`bollinger_bands`      - mean +/- k*std envelope
- :func:`sharpe_ratio`         - risk-adjusted return
- :func:`max_drawdown`         - worst peak-to-trough decline
- :func:`volatility_ratio`     - short/long volatility regime indicator

Everything returns plain NumPy arrays / floats for easy pandas integration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from agentx.finance.timeseries import returns, rolling_volatility, sma, zscore


def momentum(prices: np.ndarray, lookback: int) -> np.ndarray:
    """Trailing total return over ``lookback`` periods (NaN-aware)."""
    arr = np.asarray(prices, dtype=float)
    out = np.full_like(arr, np.nan)
    if lookback < 1:
        return out
    out[lookback:] = arr[lookback:] / arr[:-lookback] - 1.0
    return out


def mean_reversion_zscore(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score of price vs its moving average (fade-the-extreme)."""
    return zscore(prices, window)


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index in [0, 100].

    Overbought > 70, oversold < 30 (classic thresholds).
    """
    arr = np.asarray(prices, dtype=float)
    out = np.full_like(arr, np.nan)
    if period < 1 or len(arr) < period + 1:
        return out
    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for t in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[t]) / period
        avg_loss = (avg_loss * (period - 1) + losses[t]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
        out[t + 1] = 100.0 - 100.0 / (1.0 + rs) if np.isfinite(rs) else 100.0
    return out


def bollinger_bands(
    prices: np.ndarray, window: int = 20, num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger envelope: ``(middle, upper, lower)`` arrays."""
    arr = np.asarray(prices, dtype=float)
    middle = sma(arr, window)
    upper = np.full_like(arr, np.nan)
    lower = np.full_like(arr, np.nan)
    for t in range(window - 1, len(arr)):
        std = np.std(arr[t - window + 1 : t + 1])
        upper[t] = middle[t] + num_std * std
        lower[t] = middle[t] - num_std * std
    return middle, upper, lower


def sharpe_ratio(
    returns_arr: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252
) -> float:
    """Annualized Sharpe ratio of a returns series."""
    r = np.asarray(returns_arr, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(prices: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0..1)."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / peak
    return float(np.nanmax(drawdown))


def volatility_ratio(
    returns_arr: np.ndarray, short: int = 10, long: int = 60
) -> np.ndarray:
    """Short-term / long-term realized volatility ratio (regime indicator).

    Values > 1 signal rising volatility (stress regime).
    """
    arr = np.asarray(returns_arr, dtype=float)
    short_vol = rolling_volatility(arr, short)
    long_vol = rolling_volatility(arr, long)
    out = np.full_like(arr, np.nan)
    mask = long_vol > 0
    out[mask] = short_vol[mask] / long_vol[mask]
    return out
