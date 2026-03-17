from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from evaluation import Trade
from portfolio import PortfolioConfig, equal_weight_targets, select_top_stocks
from risk_management import RiskConfig, check_exit


@dataclass
class Position:
    ticker: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float  # average cost


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    drawdown: pd.Series
    trades: list[Trade]


def run_portfolio_backtest(
    price_by_ticker: Dict[str, pd.Series],
    prob_by_ticker: Dict[str, pd.Series],
    cfg: PortfolioConfig = PortfolioConfig(),
    risk: RiskConfig = RiskConfig(),
) -> BacktestResult:
    """
    Daily simulation:
      - Mark-to-market using close prices
      - Apply stop-loss / take-profit using close
      - Select top-N by predicted prob, allocate equally with max exposure
      - Rebalance at close (no transaction costs/slippage)
    """
    tickers = sorted(price_by_ticker.keys())
    if not tickers:
        raise ValueError("No tickers provided")

    # Determine the backtest calendar from the intersection of all probability series.
    common_dates = None
    for t in tickers:
        s = prob_by_ticker.get(t)
        if s is None or s.empty:
            continue
        common_dates = s.index if common_dates is None else common_dates.intersection(s.index)
    if common_dates is None or len(common_dates) < 5:
        raise ValueError("Not enough probability dates to backtest")

    # Also require prices for the same dates.
    for t in tickers:
        common_dates = common_dates.intersection(price_by_ticker[t].index)
    common_dates = common_dates.sort_values()

    cash = float(cfg.initial_capital)
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity = pd.Series(index=common_dates, dtype=float)

    def mark_to_market(date: pd.Timestamp) -> float:
        total = cash
        for pos in positions.values():
            px = float(price_by_ticker[pos.ticker].loc[date])
            total += pos.shares * px
        return float(total)

    for i, date in enumerate(common_dates):
        rebalance = (i % 5 == 0)

        # 1) Mark to market
        eq = mark_to_market(date)

        # 2) Risk exits
        for t in list(positions.keys()):
            pos = positions[t]
            px = float(price_by_ticker[t].loc[date])
            reason = check_exit(pos.entry_price, px, risk)
            if reason is None:
                continue
            cash += pos.shares * px
            trades.append(
                Trade(
                    ticker=t,
                    entry_date=pos.entry_date,
                    exit_date=pd.Timestamp(date),
                    entry_price=float(pos.entry_price),
                    exit_price=float(px),
                    shares=float(pos.shares),
                    reason=reason,
                )
            )
            del positions[t]

        if rebalance:
            # 3) Select top stocks by probability (for signals at this date)
            prob_today: Dict[str, float] = {}
            for t in tickers:
                s = prob_by_ticker.get(t)
                if s is None or date not in s.index:
                    continue
                prob_today[t] = float(s.loc[date])

            selected = select_top_stocks(prob_today, cfg.top_n)
            targets_w = equal_weight_targets(selected, cfg.max_exposure)

            # 4) Sell positions not in target set (rebalance exit)
            for t in list(positions.keys()):
                if t in targets_w:
                    continue
                pos = positions[t]
                px = float(price_by_ticker[t].loc[date])
                cash += pos.shares * px
                trades.append(
                    Trade(
                        ticker=t,
                        entry_date=pos.entry_date,
                        exit_date=pd.Timestamp(date),
                        entry_price=float(pos.entry_price),
                        exit_price=float(px),
                        shares=float(pos.shares),
                        reason="rebalance",
                    )
                )
                del positions[t]

            # 5) Buy/adjust target positions
            eq = mark_to_market(date)
            for t, w in targets_w.items():
                px = float(price_by_ticker[t].loc[date])
                if px <= 0:
                    continue
                target_value = eq * float(w)
                target_shares = target_value / px

                if t not in positions:
                    # Open new position
                    cost = target_shares * px
                    if cost > cash:
                        # Scale down to available cash (should rarely happen).
                        target_shares = cash / px
                        cost = target_shares * px
                    cash -= cost
                    positions[t] = Position(
                        ticker=t,
                        shares=float(target_shares),
                        entry_date=pd.Timestamp(date),
                        entry_price=float(px),
                    )
                else:
                    # Rebalance existing position; update average cost when adding.
                    pos = positions[t]
                    delta = target_shares - pos.shares
                    if abs(delta) < 1e-8:
                        continue
                    if delta > 0:
                        # Buy more
                        cost = delta * px
                        if cost > cash:
                            delta = cash / px
                            cost = delta * px
                        new_shares = pos.shares + delta
                        if new_shares > 0:
                            pos.entry_price = float((pos.entry_price * pos.shares + px * delta) / new_shares)
                        pos.shares = float(new_shares)
                        cash -= cost
                    else:
                        # Sell some
                        sell_shares = -delta
                        proceeds = sell_shares * px
                        pos.shares = float(max(0.0, pos.shares - sell_shares))
                        cash += proceeds
                        if pos.shares <= 1e-10:
                            # Fully closed by rounding; record a trade.
                            trades.append(
                                Trade(
                                    ticker=t,
                                    entry_date=pos.entry_date,
                                    exit_date=pd.Timestamp(date),
                                    entry_price=float(pos.entry_price),
                                    exit_price=float(px),
                                    shares=float(pos.shares + sell_shares),
                                    reason="rebalance",
                                )
                            )
                            del positions[t]

        equity.loc[date] = mark_to_market(date)

    dd = equity / equity.cummax() - 1.0
    return BacktestResult(equity_curve=equity, drawdown=dd, trades=trades)

