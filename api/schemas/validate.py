from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from api.schemas.signals import SignalHistoryPoint


class PortfolioImpact(BaseModel):
    expected_return_impact: float       # weighted expected return contribution delta
    volatility_impact: float            # estimated annualized vol contribution
    diversification_change: float       # positive = improves diversification
    sector_concentration_change: float  # portfolio sector weight delta
    max_drawdown_impact: float          # estimated worst-case additional drawdown (negative)
    portfolio_fit_score: int            # 0-100
    portfolio_data_available: bool


class ValidationScorecard(BaseModel):
    ticker: str
    signal: str
    prob_up: float
    target_return: Optional[float] = None
    regime: str

    # Core scorecard metrics
    idea_score: int               # 0-100 composite
    evidence_strength: int        # 0-100
    portfolio_fit: int            # 0-100
    regret_risk: int              # 0-100
    trust_score: int              # 0-100 (from ROC-AUC)

    conviction: Literal["VERY HIGH", "HIGH", "MEDIUM", "LOW"]
    suggested_action: str

    # Second opinion
    verdict: Literal["AGREE", "PARTIALLY AGREE", "DISAGREE", "NEUTRAL"]
    verdict_reasons: list[str]

    # Signal history — ascending chronological order, from DB (empty if json-only mode)
    signal_trend: Literal["IMPROVING", "STABLE", "DETERIORATING", "INSUFFICIENT DATA"]
    history_count: int
    history: list[SignalHistoryPoint] = []

    # Portfolio impact analysis
    portfolio_impact: Optional[PortfolioImpact] = None
