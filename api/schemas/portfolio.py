from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PositionItem(BaseModel):
    ticker: str
    weight: float
    signal: Optional[str] = None
    prob_up: Optional[float] = None
    expected_return_pct: Optional[float] = None


class PortfolioResponse(BaseModel):
    positions: list[PositionItem]
    n_positions: int
    last_rebal_date: Optional[str] = None
    last_updated: str
    method: str
    rebal_freq: str
