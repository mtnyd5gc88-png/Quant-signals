from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    reason: str  # "rebalance" | "stop_loss" | "take_profit"

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return float("nan")
        return (self.exit_price / self.entry_price) - 1.0


@dataclass(frozen=True)
class PerformanceReport:
    cumulative_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    buy_and_hold_return: float
    win_rate: float
    number_of_trades: int


def compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd


def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    std = r.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(np.sqrt(periods_per_year) * (r.mean() / std))


def annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float:
    eq = equity.dropna()
    if len(eq) < 2:
        return float("nan")
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = (len(eq) - 1) / periods_per_year
    if years <= 0:
        return float("nan")
    return float((1 + total) ** (1 / years) - 1)


def summarize_performance(
    equity: pd.Series,
    benchmark_equity: pd.Series,
    trades: list[Trade],
) -> PerformanceReport:
    equity = equity.dropna()
    benchmark_equity = benchmark_equity.dropna()

    cum_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 else float("nan")
    bh_ret = (
        float(benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1.0)
        if len(benchmark_equity) >= 2
        else float("nan")
    )

    daily_ret = equity.pct_change()
    dd = compute_drawdown(equity)
    max_dd = float(dd.min()) if len(dd) else float("nan")

    wins = sum(1 for tr in trades if tr.pnl > 0)
    win_rate = float(wins / len(trades)) if trades else float("nan")

    return PerformanceReport(
        cumulative_return=cum_ret,
        annualized_return=annualized_return(equity),
        sharpe_ratio=sharpe_ratio(daily_ret),
        max_drawdown=max_dd,
        buy_and_hold_return=bh_ret,
        win_rate=win_rate,
        number_of_trades=len(trades),
    )

