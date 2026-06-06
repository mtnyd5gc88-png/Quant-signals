from __future__ import annotations

import json
import statistics
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import CurrentUser
from api.config import settings
from api.db import get_db
from api.models.run_log import RunLog
from api.models.signal_snapshot import SignalSnapshot
from api.schemas.signals import SignalDetail, SignalHistoryPoint, SignalItem, SignalsResponse

router = APIRouter(prefix="/signals", tags=["signals"])


def _load_predictions() -> list[dict]:
    path = settings.quant_data_dir / "predictions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _load_regime() -> dict:
    path = settings.quant_data_dir / "regime.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _enrich_with_alpha_scores(raw: list[dict]) -> list[dict]:
    probs = [r.get("prob_up", 0.5) for r in raw]
    if len(probs) < 2:
        return [{**r, "alpha_score": 0.0} for r in raw]
    mean_p = statistics.mean(probs)
    std_p  = statistics.stdev(probs) or 1e-6
    return [
        {**r, "alpha_score": round((r.get("prob_up", 0.5) - mean_p) / std_p, 4)}
        for r in raw
    ]


@router.get("", response_model=SignalsResponse)
async def get_signals(
    _user: CurrentUser,
    signal_filter: Literal["ALL", "BUY", "HOLD", "SELL", "CASH"] = Query("ALL"),
    sort_by: Literal["prob_up", "ticker", "target_return"] = Query("prob_up"),
    order: Literal["asc", "desc"] = Query("desc"),
    search: Optional[str] = Query(None, max_length=10),
) -> SignalsResponse:
    raw = _enrich_with_alpha_scores(_load_predictions())

    if search:
        raw = [r for r in raw if search.upper() in r["ticker"]]
    if signal_filter != "ALL":
        raw = [r for r in raw if r.get("signal") == signal_filter]

    reverse = (order == "desc")
    if sort_by == "ticker":
        raw.sort(key=lambda r: r["ticker"], reverse=reverse)
    elif sort_by == "target_return":
        raw.sort(key=lambda r: r.get("target_return") or 0, reverse=reverse)
    else:
        raw.sort(key=lambda r: r.get("prob_up", 0), reverse=reverse)

    items = [SignalItem(**r) for r in raw]
    regime_data = _load_regime()
    last_updated = regime_data.get("last_updated", "unknown")

    return SignalsResponse(
        items=items,
        total=len(items),
        buy_count=sum(1 for i in items if i.signal == "BUY"),
        hold_count=sum(1 for i in items if i.signal == "HOLD"),
        sell_count=sum(1 for i in items if i.signal == "SELL"),
        cash_count=sum(1 for i in items if i.signal == "CASH"),
        last_updated=last_updated,
    )


@router.get("/{ticker}", response_model=SignalDetail)
async def get_signal_detail(
    ticker: str,
    _user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> SignalDetail:
    raw = _enrich_with_alpha_scores(_load_predictions())
    match = next((r for r in raw if r["ticker"] == ticker.upper()), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    stmt = (
        select(SignalSnapshot, RunLog.run_at)
        .join(RunLog, SignalSnapshot.run_id == RunLog.id)
        .where(SignalSnapshot.ticker == ticker.upper())
        .order_by(RunLog.run_at.desc())
        .limit(90)
    )
    result = await db.execute(stmt)
    rows = result.all()
    history = [
        SignalHistoryPoint(
            run_at=run_at.isoformat(),
            prob_up=snap.prob_up or 0.0,
            signal=snap.signal or "HOLD",
        )
        for snap, run_at in rows
    ]

    return SignalDetail(**match, history=history)
