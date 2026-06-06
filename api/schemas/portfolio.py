from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PortfolioHolding(BaseModel):
    ticker: str
    company: Optional[str] = None
    sector: Optional[str] = None
    weight: float
    signal: Optional[str] = None
    prob_up: Optional[float] = None
    price: Optional[float] = None
    target_return: Optional[float] = None


class PortfolioResponse(BaseModel):
    holdings: list[PortfolioHolding]
    total_weight: float
    n_positions: int
    expected_return: float
    last_updated: str
