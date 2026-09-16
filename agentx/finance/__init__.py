# ============================================================================
# AgentX.finance - Module 8: Financial Time-Series Modeling
# ============================================================================

"""Financial modeling toolkit: features, factors and forecasting.

- :mod:`agentx.finance.timeseries` - returns, SMA/EMA, volatility, z-score,
  trend regression and autocorrelation (NumPy only).
- :mod:`agentx.finance.factors`    - momentum, mean reversion, RSI, Bollinger
  bands, Sharpe ratio and max drawdown.
- :mod:`agentx.finance.models`     - linear / ridge / AR(p) forecasting with
  rolling-window backtesting and RMSE/MAE/MAPE evaluation.

Quick start
-----------
>>> import numpy as np
>>> from agentx.finance.models import ForecastModel
>>> prices = np.cumsum(np.random.default_rng(0).normal(0.1, 1.0, 120)) + 100
>>> model = ForecastModel(model="ar").fit(prices, horizon=5)
>>> points = model.predict()
"""

from agentx.finance import factors, models, timeseries

__all__ = ["timeseries", "factors", "models"]
