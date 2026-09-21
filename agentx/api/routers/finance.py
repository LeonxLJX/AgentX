# ============================================================================
# AgentX.api.routers.finance - FinanceModels endpoints
# ============================================================================

"""Endpoints
----------
- ``POST /api/finance/forecast`` : fit linear / ridge / AR(p) and forecast
- ``POST /api/finance/factors``  : compute technical factor snapshots
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agentx.core import get_logger
from agentx.finance.factors import (
    bollinger_bands,
    max_drawdown,
    momentum,
    rsi,
    sharpe_ratio,
    volatility_ratio,
)
from agentx.finance.models import ForecastModel
from agentx.finance.timeseries import returns

logger = get_logger("agentx.api.finance")
router = APIRouter()


class ForecastRequest(BaseModel):
    """Time series to model."""

    prices: List[float] = Field(..., min_length=20, description="Ordered prices, oldest first")
    model: str = Field("ar", description="linear | ridge | ar | naive")
    horizon: int = Field(5, ge=1, le=60)
    lags: int = Field(5, ge=1, le=20)


class FactorsRequest(BaseModel):
    """Time series for factor computation."""

    prices: List[float] = Field(..., min_length=30)


@router.post("/forecast", summary="Forecast a time series")
def forecast(payload: ForecastRequest) -> Dict[str, Any]:
    """Return the forecast points plus in-sample evaluation metrics."""
    if len(payload.prices) < max(10, payload.lags + 3):
        raise HTTPException(400, "series too short for the requested lags")
    model = ForecastModel(model=payload.model, lags=payload.lags)
    model.fit(payload.prices, horizon=payload.horizon)
    points = [p.__dict__ for p in model.predict()]
    metrics = model.evaluate(payload.prices, horizon=payload.horizon)
    logger.info(
        "forecast complete: model=%s horizon=%d n=%d",
        payload.model,
        payload.horizon,
        len(payload.prices),
    )
    return {
        "model": payload.model,
        "horizon": payload.horizon,
        "points": points,
        "evaluation": metrics,
    }


@router.post("/factors", summary="Compute technical factors")
def factors(payload: FactorsRequest) -> Dict[str, Any]:
    """Return a snapshot of classic factors at the series tail."""
    prices = payload.prices
    rets = returns(prices)
    middle, upper, lower = bollinger_bands(prices)
    logger.info("factors computed: n=%d", len(prices))
    return {
        "last_price": prices[-1],
        "sma_20": round(float(middle[-1]), 4),
        "bollinger_upper": round(float(upper[-1]), 4),
        "bollinger_lower": round(float(lower[-1]), 4),
        "rsi_14": round(float(rsi(prices)[-1]), 4),
        "momentum_20": round(float(momentum(prices, 20)[-1] or 0.0), 4),
        "volatility_ratio_10_60": round(float(volatility_ratio(rets, 10, 60)[-1] or 0.0), 4),
        "sharpe_ratio": round(sharpe_ratio(rets), 4),
        "max_drawdown": round(max_drawdown(prices), 4),
    }
