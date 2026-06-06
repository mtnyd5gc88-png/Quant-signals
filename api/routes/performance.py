from __future__ import annotations

import json
from typing import Optional

import numpy as np
from fastapi import APIRouter, Query

from api.auth import CurrentUser
from api.config import settings
from api.schemas.performance import (
    DrawdownPoint,
    EquityCurvePoint,
    EquityCurveResponse,
    MonthlyReturnPoint,
    PerformanceMetrics,
    RollingPoint,
)

router = APIRouter(prefix="/performance", tags=["performance"])


def _read(name: str):
    path = settings.quant_data_dir / name
    if not path.exists():
        return []
    return json.loads(path.read_text())


@router.get("", response_model=PerformanceMetrics)
async def get_performance(_user: CurrentUser) -> PerformanceMetrics:
    m = _read("metrics.json")
    diag = _read("diagnostics.json")
    run_time = diag.get("run_config", {}).get("run_time", "unknown") if isinstance(diag, dict) else "unknown"
    return PerformanceMetrics(
        total_return=m.get("total_return", 0),
        annualized_return=m.get("annualized_return", 0),
        sharpe_ratio=m.get("sharpe_ratio", 0),
        sortino_ratio=m.get("sortino_ratio", 0),
        calmar_ratio=m.get("calmar_ratio", 0),
        max_drawdown=m.get("max_drawdown", 0),
        benchmark_total=m.get("benchmark_total", 0),
        benchmark_ann=m.get("benchmark_ann", 0),
        alpha_annualized=m.get("alpha_annualized", 0),
        beta=m.get("beta", 0),
        win_rate=m.get("win_rate", 0),
        profit_factor=m.get("profit_factor", 0),
        n_trading_days=int(m.get("n_trading_days", 0)),
        n_years=m.get("n_years", 0),
        last_run=run_time,
    )


@router.get("/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(_user: CurrentUser) -> EquityCurveResponse:
    points = [EquityCurvePoint(**p) for p in _read("equity_curve.json")]
    m = _read("metrics.json")
    return EquityCurveResponse(points=points, period_years=m.get("n_years", 0))


@router.get("/drawdown", response_model=list[DrawdownPoint])
async def get_drawdown(_user: CurrentUser) -> list[DrawdownPoint]:
    return [DrawdownPoint(**p) for p in _read("drawdown.json")]


@router.get("/monthly", response_model=list[MonthlyReturnPoint])
async def get_monthly_returns(_user: CurrentUser) -> list[MonthlyReturnPoint]:
    return [
        MonthlyReturnPoint(year=p["year"], month=p["month"], ret=p["return"])
        for p in _read("monthly_returns.json")
    ]


@router.get("/rolling", response_model=list[RollingPoint])
async def get_rolling_metric(
    _user: CurrentUser,
    metric: str = Query("sharpe", pattern="^(sharpe|volatility)$"),
    window: int = Query(60, ge=20, le=252),
) -> list[RollingPoint]:
    raw = _read("equity_curve.json")
    if not raw:
        return []

    dates    = [p["date"] for p in raw]
    strategy = np.array([p["strategy"] for p in raw])
    rets     = np.diff(strategy) / strategy[:-1]

    result = []
    for i in range(window, len(rets)):
        window_rets = rets[i - window: i]
        if metric == "sharpe":
            rf_daily = 0.04 / 252
            excess   = window_rets - rf_daily
            val = float(excess.mean() / (excess.std() + 1e-10) * np.sqrt(252))
        else:
            val = float(window_rets.std() * np.sqrt(252))
        result.append(RollingPoint(date=dates[i + 1], value=round(val, 4)))

    return result
