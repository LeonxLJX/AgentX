# ============================================================================
# Example: FinanceModels - time-series factors, forecasting, backtesting
# Run:  python examples/demo_finance.py
# ============================================================================

"""Fits AR/linear/ridge forecasts, backtests them and computes factors."""

import numpy as np

from agentx.finance.factors import (
    bollinger_bands,
    max_drawdown,
    momentum,
    rsi,
    sharpe_ratio,
)
from agentx.finance.models import ForecastModel
from agentx.finance.timeseries import ema, returns

if __name__ == "__main__":
    # Seeded geometric random walk as a demo price series.
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, 150)))

    # --- Factors -------------------------------------------------------------
    rets = returns(prices)
    mid, up, lo = bollinger_bands(prices)
    print("FACTORS (tail snapshot)")
    print(f"  SMA20          : {mid[-1]:.2f}")
    print(f"  Bollinger upper: {up[-1]:.2f}  lower: {lo[-1]:.2f}")
    print(f"  RSI(14)        : {rsi(prices)[-1]:.1f}")
    print(f"  Momentum(20)   : {momentum(prices, 20)[-1]:.2%}")
    print(f"  Sharpe (annual): {sharpe_ratio(rets):.2f}")
    print(f"  Max drawdown   : {max_drawdown(prices):.2%}")

    # --- Forecasting with honest evaluation -----------------------------------
    print("\nFORECAST COMPARISON (hold-out)")
    for model_name in ("naive", "linear", "ridge", "ar"):
        model = ForecastModel(model=model_name, lags=5)
        metrics = model.evaluate(prices, horizon=10)
        print(f"  {model_name:<7} rmse={metrics['rmse']:.3f} "
              f"mae={metrics['mae']:.3f} mape={metrics['mape']:.2f}%")

    # --- Multi-step forecast ---------------------------------------------------
    model = ForecastModel(model="ar", lags=5).fit(prices, horizon=10)
    print("\nAR(5) 10-step forecast:")
    for point in model.predict():
        print(f"  {point.index}: {point.predicted:.2f}")
