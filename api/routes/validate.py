from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import CurrentUser
from api.config import settings
from api.db import get_db
from api.models.run_log import RunLog
from api.models.signal_snapshot import SignalSnapshot
from api.schemas.signals import SignalHistoryPoint
from api.schemas.validate import PortfolioImpact, ValidationScorecard

router = APIRouter(prefix="/validate", tags=["validate"])

MAX_POSITION_WEIGHT = 0.15


def _load_json(name: str) -> dict | list:
    path = settings.quant_data_dir / name
    if not path.exists():
        return {} if name.endswith(".json") else []
    return json.loads(path.read_text())


def _regime_alignment(regime: str, signal: str) -> float:
    bullish = signal == "BUY"
    bearish = signal in ("SELL", "CASH", "STAY IN CASH")
    if regime == "risk-on":
        return 1.0 if bullish else (0.15 if bearish else 0.65)
    if regime == "risk-off":
        return 0.15 if bullish else (1.0 if bearish else 0.55)
    return 0.65  # neutral or unknown


def _regret_risk(
    prob_up: float,
    signal: str,
    regime: str,
    target_return: Optional[float],
    roc_auc: float,
    trend: str,
) -> int:
    risk = 10
    if signal == "BUY" and regime == "risk-off":
        risk += 30
    elif signal in ("SELL", "CASH", "STAY IN CASH") and regime == "risk-on":
        risk += 20
    if 0.55 <= prob_up < 0.63:
        risk += 20
    elif prob_up < 0.55 and signal == "BUY":
        risk += 15
    if roc_auc < 0.58:
        risk += 20
    elif roc_auc < 0.62:
        risk += 10
    if target_return is not None and abs(target_return) > 0.12:
        risk += 15
    if trend == "DETERIORATING":
        risk += 15
    elif trend == "IMPROVING":
        risk -= 5
    return max(0, min(100, risk))


def _suggested_action(conviction: str, signal: str, current_weight: float) -> str:
    if signal == "BUY":
        if conviction == "VERY HIGH":
            return "Core Position Candidate"
        if conviction == "HIGH":
            return "Consider Building Position"
        if conviction == "MEDIUM":
            return "Speculative Position Only"
        return "Monitor Only"
    if signal in ("SELL", "CASH", "STAY IN CASH"):
        return "Consider Reducing Exposure" if current_weight > 0.05 else "Avoid New Entry"
    return "Hold — Insufficient Directional Edge"


def _verdict_and_reasons(
    signal: str,
    idea_score: int,
    evidence_strength: int,
    regret_risk: int,
    portfolio_fit: int,
    regime: str,
    prob_up: float,
    roc_auc: float,
    trend: str,
    current_weight: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    pct = round(prob_up * 100)

    if signal == "BUY":
        verdict = "AGREE" if idea_score >= 72 else ("PARTIALLY AGREE" if idea_score >= 52 else "DISAGREE")

        if prob_up >= 0.65:
            reasons.append(f"Strong upward probability: {pct}% model confidence")
        elif prob_up >= 0.60:
            reasons.append(f"Moderate upward probability at {pct}% — above threshold")
        else:
            reasons.append(f"Weak probability at {pct}% — signal near decision boundary")

        if regime == "risk-on":
            reasons.append("Favorable market regime (risk-on) supports bullish thesis")
        elif regime == "risk-off":
            reasons.append("Risk-off regime conflicts with bullish thesis — caution warranted")
        else:
            reasons.append("Neutral market regime — no regime tailwind or headwind")

        if roc_auc >= 0.65:
            reasons.append(f"Above-average model reliability (ROC-AUC {roc_auc:.2f})")
        elif roc_auc < 0.58:
            reasons.append(f"Below-average model reliability (ROC-AUC {roc_auc:.2f}) — verify with other sources")

        if current_weight >= MAX_POSITION_WEIGHT:
            reasons.append(f"Position at concentration cap ({current_weight * 100:.1f}%) — no additional room")
        elif current_weight > 0:
            capacity_pct = round((1 - current_weight / MAX_POSITION_WEIGHT) * 100)
            reasons.append(f"Currently holding {current_weight * 100:.1f}% — {capacity_pct}% capacity remaining")
        else:
            reasons.append("No current holding — full allocation capacity available")

        if trend == "IMPROVING":
            reasons.append("Signal has been strengthening over recent evaluation runs")
        elif trend == "DETERIORATING":
            reasons.append("Warning: signal confidence has been weakening across recent runs")

    elif signal in ("SELL", "CASH", "STAY IN CASH"):
        # idea_score measures bearish thesis strength; >= 65 = strong bearish = agree with avoiding.
        verdict = "AGREE" if idea_score >= 65 else "DISAGREE"

        downside_pct = round((1 - prob_up) * 100)
        if prob_up <= 0.40:
            reasons.append(f"Strong bearish evidence: {downside_pct}% implied downside probability")
        elif prob_up <= 0.48:
            reasons.append(f"Moderate bearish signal: {downside_pct}% implied downside probability")
        else:
            reasons.append(f"Weak bearish signal at {downside_pct}% — near decision boundary")

        if regime == "risk-off":
            reasons.append("Risk-off market regime reinforces defensive positioning")
        elif regime == "risk-on":
            reasons.append("Risk-on regime partly contradicts the bearish thesis — caution on timing")
        else:
            reasons.append("Neutral market regime — no strong regime reinforcement")

        if roc_auc >= 0.65:
            reasons.append(f"Above-average model reliability (ROC-AUC {roc_auc:.2f})")
        elif roc_auc < 0.58:
            reasons.append(f"Below-average model reliability (ROC-AUC {roc_auc:.2f}) — verify with other sources")

        if current_weight > 0:
            reasons.append(f"Currently holding {current_weight * 100:.1f}% — exit is directly actionable")
        else:
            reasons.append("No current holding — avoid initiating any long position")

        if trend == "DETERIORATING":
            reasons.append("Bearish signal has been strengthening over recent runs")
        elif trend == "IMPROVING":
            reasons.append("Warning: bearish signal has been weakening — possible reversal")
        elif trend == "STABLE":
            reasons.append("Signal conviction has been stable across recent runs")

    else:  # HOLD
        verdict = "NEUTRAL"
        reasons.append(f"No strong directional edge at {pct}% — insufficient for conviction")
        reasons.append("Monitor for improving signal conditions before committing capital")
        if regime == "risk-off":
            reasons.append("Defensive regime favors patience over action")

    return verdict, reasons[:4]


def _portfolio_impact(
    ticker: str,
    signal: str,
    prob_up: float,
    target_return: Optional[float],
    predictions: list[dict],
    portfolio_data: dict,
    regret_risk: int,
    portfolio_fit: int,
) -> PortfolioImpact:
    positions = portfolio_data.get("positions", [])
    portfolio_data_available = len(positions) > 0

    current_pos = next((p for p in positions if p.get("ticker") == ticker), None)
    current_weight = current_pos.get("weight", 0.0) if current_pos else 0.0

    preds_by_ticker = {r.get("ticker", ""): r for r in predictions}
    ticker_sector = preds_by_ticker.get(ticker, {}).get("sector")

    n_sector_positions = sum(
        1 for p in positions
        if preds_by_ticker.get(p.get("ticker", ""), {}).get("sector") == ticker_sector
    ) if ticker_sector else 0
    n_positions = max(len(positions), 1)

    if signal == "BUY":
        weight_delta = max(0.0, min(0.05, MAX_POSITION_WEIGHT - current_weight))
    elif signal in ("SELL", "CASH", "STAY IN CASH"):
        weight_delta = -current_weight
    else:
        weight_delta = 0.0

    if abs(weight_delta) < 0.0001:
        return PortfolioImpact(
            expected_return_impact=0.0,
            volatility_impact=0.0,
            diversification_change=0.0,
            sector_concentration_change=0.0,
            max_drawdown_impact=0.0,
            portfolio_fit_score=portfolio_fit,
            portfolio_data_available=portfolio_data_available,
        )

    tr = target_return if target_return is not None else (prob_up - 0.5) * 0.4
    expected_return_impact = round(tr * weight_delta, 4)

    prob_certainty = abs(prob_up - 0.5) * 2
    volatility_impact = round(0.30 * abs(weight_delta) * (1.0 - prob_certainty * 0.25), 4)

    sector_ratio = n_sector_positions / n_positions
    if signal == "BUY":
        diversification_change = round((0.25 - sector_ratio) * abs(weight_delta) * 2, 3)
    else:
        diversification_change = round(sector_ratio * abs(weight_delta) * 1.5, 3)

    sector_concentration_change = round(weight_delta, 4)
    max_drawdown_impact = round(-abs(weight_delta) * (regret_risk / 100) * 0.35, 4)

    return PortfolioImpact(
        expected_return_impact=expected_return_impact,
        volatility_impact=volatility_impact,
        diversification_change=diversification_change,
        sector_concentration_change=sector_concentration_change,
        max_drawdown_impact=max_drawdown_impact,
        portfolio_fit_score=portfolio_fit,
        portfolio_data_available=portfolio_data_available,
    )


@router.get("/{ticker}", response_model=ValidationScorecard)
async def validate_ticker(
    ticker: str,
    _user: CurrentUser,
    db: Optional[AsyncSession] = Depends(get_db),
) -> ValidationScorecard:
    predictions: list[dict] = _load_json("predictions.json")  # type: ignore[assignment]
    match = next((r for r in predictions if r["ticker"] == ticker.upper()), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    diagnostics: dict = _load_json("diagnostics.json")  # type: ignore[assignment]
    regime_data: dict = _load_json("regime.json")  # type: ignore[assignment]
    portfolio_data: dict = _load_json("portfolio.json")  # type: ignore[assignment]

    prob_up: float = match.get("prob_up", 0.5)
    signal: str = match.get("signal", "HOLD")
    target_return: Optional[float] = match.get("target_return")

    roc_auc: float = diagnostics.get("model_quality", {}).get("roc_auc_mean", 0.60)
    trust_score = max(0, min(100, round(roc_auc * 100)))

    regime: str = regime_data.get("regime", "neutral").lower()

    positions = {p["ticker"]: p["weight"] for p in portfolio_data.get("positions", [])}
    current_weight: float = positions.get(ticker.upper(), 0.0)

    # Signal history from DB — descending rows, kept for scoring; reversed for chart output
    history_rows: list[tuple[SignalSnapshot, object]] = []
    if db is not None:
        stmt = (
            select(SignalSnapshot, RunLog.run_at)
            .join(RunLog, SignalSnapshot.run_id == RunLog.id)
            .where(SignalSnapshot.ticker == ticker.upper())
            .order_by(RunLog.run_at.desc())
            .limit(20)
        )
        history_rows = list((await db.execute(stmt)).all())

    history_snaps = [snap for snap, _ in history_rows]

    signal_consistency = 0.5
    signal_trend = "INSUFFICIENT DATA"
    if len(history_snaps) >= 3:
        recent = history_snaps[:10]
        signal_consistency = sum(1 for h in recent if h.signal == signal) / len(recent)
        p0 = history_snaps[0].prob_up or 0.5  # newest
        p1 = history_snaps[1].prob_up or 0.5
        p2 = history_snaps[2].prob_up or 0.5  # oldest of the three
        if p2 < p1 < p0:
            signal_trend = "IMPROVING"
        elif p2 > p1 > p0:
            signal_trend = "DETERIORATING"
        else:
            signal_trend = "STABLE"

    # Build ascending history list for the chart (oldest → newest)
    history_points = [
        SignalHistoryPoint(
            run_at=run_at.isoformat(),
            prob_up=snap.prob_up or 0.0,
            signal=snap.signal or "HOLD",
        )
        for snap, run_at in reversed(history_rows)
    ]

    # Evidence strength
    prob_certainty = abs(prob_up - 0.5) * 2
    regime_align = _regime_alignment(regime, signal)
    model_quality_norm = max(0.0, min(1.0, (roc_auc - 0.5) * 2))
    evidence_strength = max(0, min(100, round((
        0.40 * prob_certainty +
        0.30 * regime_align +
        0.20 * model_quality_norm +
        0.10 * signal_consistency
    ) * 100)))

    # Conviction (trust_score is roc_auc*100, so 70 means 0.70 AUC)
    if evidence_strength >= 75 and trust_score >= 70:
        conviction = "VERY HIGH"
    elif evidence_strength >= 60 and trust_score >= 62:
        conviction = "HIGH"
    elif evidence_strength >= 40:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    # Portfolio fit
    if signal == "BUY":
        if current_weight >= MAX_POSITION_WEIGHT:
            portfolio_fit = 15
        else:
            capacity = 1 - current_weight / MAX_POSITION_WEIGHT
            portfolio_fit = round(capacity * 85 + 15)
    elif signal in ("SELL", "CASH", "STAY IN CASH"):
        if current_weight > 0:
            portfolio_fit = round(min(100, current_weight / MAX_POSITION_WEIGHT * 90))
        else:
            portfolio_fit = 25
    else:
        portfolio_fit = 50

    regret = _regret_risk(prob_up, signal, regime, target_return, roc_auc, signal_trend)

    # Idea score
    if signal == "BUY":
        idea_score = round(0.40 * evidence_strength + 0.30 * (100 - regret) + 0.30 * portfolio_fit)
    elif signal in ("SELL", "CASH", "STAY IN CASH"):
        if current_weight > 0:
            idea_score = round(0.50 * evidence_strength + 0.30 * (100 - regret) + 0.20 * portfolio_fit)
        else:
            idea_score = round(0.50 * evidence_strength + 0.50 * (100 - regret))
    else:
        idea_score = max(0, min(100, 50 + round((evidence_strength - 50) * 0.3)))
    idea_score = max(0, min(100, idea_score))

    suggested = _suggested_action(conviction, signal, current_weight)
    verdict, reasons = _verdict_and_reasons(
        signal, idea_score, evidence_strength, regret, portfolio_fit,
        regime, prob_up, roc_auc, signal_trend, current_weight,
    )

    impact = _portfolio_impact(
        ticker.upper(), signal, prob_up, target_return,
        predictions, portfolio_data, regret, portfolio_fit,
    )

    return ValidationScorecard(
        ticker=ticker.upper(),
        signal=signal,
        prob_up=prob_up,
        target_return=target_return,
        regime=regime,
        idea_score=idea_score,
        evidence_strength=evidence_strength,
        portfolio_fit=portfolio_fit,
        regret_risk=regret,
        trust_score=trust_score,
        conviction=conviction,
        suggested_action=suggested,
        verdict=verdict,
        verdict_reasons=reasons,
        signal_trend=signal_trend,
        history_count=len(history_snaps),
        history=history_points,
        portfolio_impact=impact,
    )
