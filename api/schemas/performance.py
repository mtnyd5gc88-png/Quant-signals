from __future__ import annotations

from pydantic import BaseModel


class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    benchmark_total: float
    benchmark_ann: float
    alpha_annualized: float
    beta: float
    win_rate: float
    profit_factor: float
    n_trading_days: int
    n_years: float
    last_run: str


class EquityCurvePoint(BaseModel):
    date: str
    strategy: float
    spy: float


class EquityCurveResponse(BaseModel):
    points: list[EquityCurvePoint]
    period_years: float


class DrawdownPoint(BaseModel):
    date: str
    drawdown: float


class MonthlyReturnPoint(BaseModel):
    year: int
    month: int
    ret: float


class RollingPoint(BaseModel):
    date: str
    value: float
