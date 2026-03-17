from __future__ import annotations

from typing import Literal

import pandas as pd


Regime = Literal["bull", "neutral", "risk_off"]


def compute_market_regime(
    spy_prices: pd.Series,
    vix_series: pd.Series,
    vix_threshold: float = 25.0,
) -> pd.Series:
    """
    Compute a simple daily market regime classification using SPY and VIX.

    Regime rules:
      - "risk_off" if VIX > vix_threshold
      - "bull" if VIX <= vix_threshold and SPY 20-day return > 0
      - "neutral" otherwise
    """
    if spy_prices.empty or vix_series.empty:
        return pd.Series(dtype="object")

    # Ensure aligned, sorted daily indices and forward-fill small gaps.
    spy = spy_prices.sort_index()
    vix = vix_series.sort_index()
    idx = spy.index.union(vix.index)
    spy = spy.reindex(idx).ffill()
    vix = vix.reindex(idx).ffill()

    spy_ret_20 = spy.pct_change(20)

    regime = pd.Series(index=idx, dtype="object")
    regime[vix > vix_threshold] = "risk_off"
    regime[(vix <= vix_threshold) & (spy_ret_20 > 0)] = "bull"
    regime[(vix <= vix_threshold) & (spy_ret_20 <= 0)] = "neutral"

    return regime.dropna()

