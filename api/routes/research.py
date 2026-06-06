from __future__ import annotations
import json
import statistics
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import CurrentUser
from api.config import settings
from api.db import get_db
from api.models.run_log import RunLog
from api.models.signal_snapshot import SignalSnapshot

router = APIRouter(prefix="/research", tags=["research"])


def _load_predictions() -> list[dict]:
    path = settings.quant_data_dir / "predictions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_diag() -> dict:
    path = settings.quant_data_dir / "diagnostics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


class UniverseAnalytics(BaseModel):
    n_tickers: int
    buy_count: int
    hold_count: int
    sell_count: int
    cash_count: int
    buy_pct: float
    mean_prob_up: float
    median_prob_up: float


class PredictionDriftPoint(BaseModel):
    run_at: str
    buy_pct: float
    mean_prob_up: float
    n_tickers: int


@router.get("/universe", response_model=UniverseAnalytics)
async def get_universe_analytics(_user: CurrentUser) -> UniverseAnalytics:
    preds = _load_predictions()
    if not preds:
        return UniverseAnalytics(n_tickers=0, buy_count=0, hold_count=0, sell_count=0, cash_count=0, buy_pct=0, mean_prob_up=0, median_prob_up=0)

    probs = [p.get("prob_up", 0.5) for p in preds]
    signals = [p.get("signal", "HOLD") for p in preds]

    return UniverseAnalytics(
        n_tickers=len(preds),
        buy_count=signals.count("BUY"),
        hold_count=signals.count("HOLD"),
        sell_count=signals.count("SELL"),
        cash_count=signals.count("CASH"),
        buy_pct=round(signals.count("BUY") / len(signals), 4),
        mean_prob_up=round(statistics.mean(probs), 4),
        median_prob_up=round(statistics.median(probs), 4),
    )


@router.get("/feature-importance", response_model=list[dict])
async def get_feature_importance(_user: CurrentUser) -> list[dict]:
    return _load_diag().get("feature_importance", [])


@router.get("/prediction-drift", response_model=list[PredictionDriftPoint])
async def get_prediction_drift(
    _user: CurrentUser,
    db: Optional[AsyncSession] = Depends(get_db),
    limit: int = 90,
) -> list[PredictionDriftPoint]:
    """Signal distribution history from PostgreSQL for prediction drift analysis. Empty in JSON-only mode."""
    if db is None:
        return []
    stmt = (
        select(RunLog.id, RunLog.run_at)
        .where(RunLog.status == "success")
        .order_by(RunLog.run_at.asc())
        .limit(limit)
    )
    run_rows = (await db.execute(stmt)).all()

    result = []
    for run_id, run_at in run_rows:
        snap_stmt = select(SignalSnapshot).where(SignalSnapshot.run_id == run_id)
        snaps = (await db.execute(snap_stmt)).scalars().all()
        if not snaps:
            continue
        probs   = [s.prob_up or 0.5 for s in snaps]
        signals = [s.signal or "HOLD" for s in snaps]
        result.append(PredictionDriftPoint(
            run_at=run_at.isoformat(),
            buy_pct=round(signals.count("BUY") / len(signals), 4),
            mean_prob_up=round(statistics.mean(probs), 4),
            n_tickers=len(snaps),
        ))
    return result
