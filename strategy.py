from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationThresholds:
    buy: float = 0.60
    hold_low: float = 0.45
    hold_high: float = 0.60


def recommendation_from_probability(p: float, th: RecommendationThresholds = RecommendationThresholds()) -> str:
    if p > th.buy:
        return "BUY"
    if th.hold_low <= p <= th.hold_high:
        return "HOLD"
    return "STAY IN CASH"

