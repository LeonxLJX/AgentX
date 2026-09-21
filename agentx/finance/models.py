# ============================================================================
# AgentX.finance.models - forecasting models & backtesting
# ============================================================================

"""Lightweight time-series forecasting with honest evaluation.

Models (all closed-form least squares - fast, transparent, auditable):

- ``"linear"`` : linear trend regression ``p_t = a + b*t``
- ``"ridge"``  : trend regression with L2 shrinkage
- ``"ar"``     : autoregressive AR(p) using lag features, recursively
  iterated for multi-step forecasts
- ``"naive"``  : random-walk baseline (last value) as a benchmark

:meth:`evaluate` and :meth:`backtest` compute RMSE / MAE / MAPE on held-out
windows so a client report can quote real out-of-sample accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agentx.core import get_logger
from agentx.core.exceptions import FinanceError, ModelNotFittedError, ValidationError
from agentx.core.schema import ForecastPoint

logger = get_logger("agentx.finance.models")


class ForecastModel:
    """Fit and forecast a univariate time series.

    Parameters
    ----------
    model : {"linear", "ridge", "ar", "naive"}
        Model family (see module docstring).
    lags : int
        Number of lag features for ``"ar"``.
    ridge_alpha : float
        L2 penalty for ``"ridge"``.
    """

    def __init__(
        self,
        model: str = "linear",
        lags: int = 5,
        ridge_alpha: float = 1.0,
    ) -> None:
        if model not in {"linear", "ridge", "ar", "naive"}:
            raise ValidationError("model must be linear/ridge/ar/naive")
        self.model = model
        self.lags = max(1, int(lags))
        self.ridge_alpha = float(ridge_alpha)
        self.series_: Optional[np.ndarray] = None
        self.horizon_: int = 1
        self.coefs_: Optional[np.ndarray] = None
        self._fit_kind: Optional[str] = None

    # ------------------------------------------------------------------ fit
    def fit(self, series: Sequence[float], horizon: int = 5) -> "ForecastModel":
        """Fit on historical values.

        Parameters
        ----------
        series : sequence[float]
            Ordered observations (oldest first).
        horizon : int
            Number of steps :meth:`predict` should produce.
        """
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < max(3, self.lags + 1):
            raise FinanceError("series too short for the chosen model")
        if horizon < 1:
            raise ValidationError("horizon must be >= 1")

        self.series_ = arr
        self.horizon_ = int(horizon)
        self.coefs_, self._fit_kind = self._fit_model(arr)
        logger.info("forecast model '%s' fitted on %d points (horizon=%d)", self.model, len(arr), self.horizon_)
        return self

    # -------------------------------------------------------------- predict
    def predict(self, index: Optional[Sequence[str]] = None) -> List[ForecastPoint]:
        """Produce the multi-step forecast.

        Parameters
        ----------
        index : sequence[str], optional
            Labels for the forecast points (e.g. dates). Defaults to integer
            offsets starting at the end of the history.
        """
        if self.series_ is None:
            raise ModelNotFittedError("call .fit(series, horizon) before predict()")

        y = self.series_
        h = self.horizon_
        values: List[float] = []
        if self._fit_kind == "trend":
            b, a = self.coefs_[0], self.coefs_[1]
            for step in range(1, h + 1):
                t = len(y) - 1 + step
                values.append(b * t + a)
        elif self._fit_kind == "ar":
            window = list(y[-(self.lags):])
            for _ in range(h):
                x = np.array([1.0] + window[-self.lags:])
                pred = float(self.coefs_ @ x)
                values.append(pred)
                window.append(pred)
        else:  # naive
            values = [float(y[-1])] * h

        labels = (
            list(index) if index is not None
            else [f"t+{i + 1}" for i in range(h)]
        )
        return [
            ForecastPoint(index=labels[i], predicted=values[i])
            for i in range(h)
        ]

    # ------------------------------------------------------------ evaluate
    def evaluate(
        self,
        series: Sequence[float],
        test_size: Optional[int] = None,
        horizon: int = 5,
    ) -> Dict[str, Any]:
        """Hold-out evaluation: fit on the train slice, score on the test slice.

        Returns
        -------
        dict
            ``rmse``, ``mae``, ``mape``, ``n_test`` and per-step errors.
        """
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        if test_size is None:
            test_size = max(horizon, int(len(arr) * 0.2))
        test_size = min(int(test_size), len(arr) - self.lags - 1)
        if test_size < 1:
            raise FinanceError("series too short for evaluation")

        train, test = arr[:-test_size], arr[-test_size:]
        model = ForecastModel(model=self.model, lags=self.lags, ridge_alpha=self.ridge_alpha)
        model.fit(train, horizon=len(test))
        points = model.predict()
        pred = np.array([p.predicted for p in points])
        actual = test[: len(pred)]

        err = actual - pred
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        denom = np.where(actual != 0, actual, np.nan)
        mape = float(np.nanmean(np.abs(err / denom)) * 100) if np.any(np.isfinite(denom)) else float("nan")
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "n_test": int(len(actual)),
        }

    def backtest(
        self,
        series: Sequence[float],
        train_size: Optional[int] = None,
        horizon: int = 5,
        step: int = 1,
    ) -> Dict[str, Any]:
        """Rolling-window backtest over the whole tail of the series.

        Returns
        -------
        dict
            Aggregate ``rmse`` / ``mae`` / ``mape`` across windows plus the
            number of windows evaluated.
        """
        arr = np.asarray(series, dtype=float)
        arr = arr[~np.isnan(arr)]
        min_train = max(10, self.lags + 3)
        if train_size is None:
            train_size = max(min_train, len(arr) - 2 * horizon)
        train_size = int(train_size)
        if train_size < min_train or train_size + horizon > len(arr):
            raise FinanceError("series too short for this backtest configuration")

        errors = []
        actuals = []
        for start in range(train_size, len(arr) - horizon + 1, step):
            train = arr[:start]
            test = arr[start : start + horizon]
            model = ForecastModel(model=self.model, lags=self.lags, ridge_alpha=self.ridge_alpha)
            model.fit(train, horizon=len(test))
            pred = np.array([p.predicted for p in model.predict()])
            errors.append(test - pred)
            actuals.append(test)

        err = np.concatenate(errors) if errors else np.array([])
        if err.size == 0:
            raise FinanceError("no backtest windows produced")
        actual = np.concatenate(actuals)
        denom = np.where(actual != 0, actual, np.nan)
        mape = float(np.nanmean(np.abs(err / denom)) * 100) if np.any(np.isfinite(denom)) else float("nan")
        return {
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "mae": round(float(np.mean(np.abs(err))), 4),
            "mape": round(mape, 4),
            "windows": int(len(errors)),
        }

    # ------------------------------------------------------------- helpers
    def _fit_model(self, arr: np.ndarray):
        n = len(arr)
        if self.model == "naive":
            return None, "naive"

        if self.model in {"linear", "ridge"}:
            X = np.column_stack([np.arange(n, dtype=float), np.ones(n)])
            y = arr
            if self.model == "ridge":
                # Closed-form ridge: (X'X + alpha*I)^{-1} X'y (intercept not shrunk).
                xtx = X.T @ X
                reg = np.eye(xtx.shape[0]) * self.ridge_alpha
                reg[-1, -1] = 0.0
                coefs = np.linalg.solve(xtx + reg, X.T @ y)
            else:
                coefs = np.linalg.lstsq(X, y, rcond=None)[0]
            return coefs, "trend"

        # AR(p): features are lagged values plus an intercept.
        p = self.lags
        rows = n - p
        X = np.empty((rows, p + 1))
        y = np.empty(rows)
        for t in range(p, n):
            X[t - p, 0] = 1.0
            X[t - p, 1:] = arr[t - p : t]
            y[t - p] = arr[t]
        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        return coefs, "ar"
