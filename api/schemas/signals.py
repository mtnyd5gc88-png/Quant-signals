from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, computed_field


class SignalItem(BaseModel):
    ticker: str
    signal: Literal["BUY", "HOLD", "SELL", "CASH", "STAY IN CASH"]
    prob_up: float
    price: Optional[float] = None
    target_return: Optional[float] = None
    alpha_score: Optional[float] = None

    @computed_field
    @property
    def expected_return_pct(self) -> Optional[float]:
        return round(self.target_return * 100, 2) if self.target_return is not None else None

    @computed_field
    @property
    def risk_reward(self) -> Optional[float]:
        if self.target_return is None or self.target_return <= 0:
            return None
        prob_down = 1 - self.prob_up
        est_downside = prob_down * 0.10
        return round(self.target_return / est_downside, 2) if est_downside > 0 else None

    @computed_field
    @property
    def confidence(self) -> Literal["HIGH", "MED", "LOW"]:
        if self.prob_up >= 0.70:
            return "HIGH"
        if self.prob_up >= 0.55:
            return "MED"
        return "LOW"

    @computed_field
    @property
    def risk_score(self) -> Literal["HIGH", "MED", "LOW"]:
        certainty = abs(self.prob_up - 0.5)
        if certainty >= 0.20:
            return "LOW"
        if certainty >= 0.10:
            return "MED"
        return "HIGH"


class SignalHistoryPoint(BaseModel):
    run_at: str
    prob_up: float
    signal: str


class SignalHistoryEntry(BaseModel):
    date: str
    prob_up: float
    signal: str


class SignalDetail(SignalItem):
    history: list[SignalHistoryPoint] = []


class SignalsResponse(BaseModel):
    items: list[SignalItem]
    total: int
    buy_count: int
    hold_count: int
    sell_count: int
    cash_count: int
    last_updated: str
