from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 100_000.0
    top_n: int = 3
    max_exposure: float = 0.90  # fraction of capital allowed invested


def select_top_stocks(prob_by_ticker: Dict[str, float], top_n: int) -> list[str]:
    items = sorted(prob_by_ticker.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in items[:top_n]]


def equal_weight_targets(selected: list[str], max_exposure: float) -> Dict[str, float]:
    """
    Return target portfolio weights per ticker. Total invested weight <= max_exposure.
    """
    if not selected:
        return {}
    w = max_exposure / len(selected)
    return {t: w for t in selected}

