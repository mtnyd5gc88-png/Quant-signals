from __future__ import annotations
import json
import statistics
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import CurrentUser
from api.config import settings
from api.db import get_db
from api.models.diagnostics_snapshot import DiagnosticsSnapshot
from api.models.signal_snapshot import FeatureImportanceSnapshot
from api.models.run_log import RunLog
from api.schemas.diagnostics import (
    AlphaAttribution,
    CalibrationBucket,
    DiagnosticsResponse,
    FeatureImportanceDriftPoint,
    FeatureImportanceItem,
    ModelDriftPoint,
    ModelQuality,
    TurnoverStats,
)

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


def _load_diag() -> dict:
    path = settings.quant_data_dir / "diagnostics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_predictions() -> list[dict]:
    path = settings.quant_data_dir / "predictions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@router.get("", response_model=DiagnosticsResponse)
async def get_diagnostics(_user: CurrentUser) -> DiagnosticsResponse:
    d = _load_diag()
    mq  = d.get("model_quality", {})
    t   = d.get("turnover",       {})
    aa  = d.get("alpha_attribution", {})
    rc  = d.get("run_config",     {})

    return DiagnosticsResponse(
        model_quality=ModelQuality(
            roc_auc_mean=mq.get("roc_auc_mean"),
            accuracy_mean=mq.get("accuracy_mean"),
            precision_mean=mq.get("precision_mean"),
            recall_mean=mq.get("recall_mean"),
            n_tickers=mq.get("n_tickers", 0),
        ),
        feature_importance=[
            FeatureImportanceItem(**fi) for fi in d.get("feature_importance", [])
        ],
        turnover=TurnoverStats(
            avg_daily=t.get("avg_daily", 0),
            annual_multiple=t.get("annual_multiple", 0),
            cost_drag_pct=t.get("cost_drag_pct", 0),
            gross_cagr=t.get("gross_cagr", 0),
            net_cagr=t.get("net_cagr", 0),
        ),
        alpha_attribution=AlphaAttribution(
            ensemble_cagr=aa.get("ensemble_cagr", 0),
            alpha_ann=aa.get("alpha_ann", 0),
            beta=aa.get("beta", 0),
        ),
        last_run=rc.get("run_time", "unknown"),
    )


@router.get("/calibration", response_model=list[CalibrationBucket])
async def get_calibration(_user: CurrentUser) -> list[CalibrationBucket]:
    """
    Compute probability calibration buckets from the current signal distribution.
    Groups tickers by their prob_up bucket and calculates the BUY rate within each.
    """
    preds = _load_predictions()
    if not preds:
        return []

    buckets: dict[tuple, list[str]] = {}
    bucket_size = 0.1
    for p in preds:
        prob = p.get("prob_up", 0.5)
        lo = round(int(prob / bucket_size) * bucket_size, 1)
        hi = round(lo + bucket_size, 1)
        key = (lo, hi)
        buckets.setdefault(key, [])
        buckets[key].append(p.get("signal", "HOLD"))

    result = []
    for (lo, hi), signals in sorted(buckets.items()):
        buy_rate = sum(1 for s in signals if s == "BUY") / len(signals)
        result.append(CalibrationBucket(
            prob_min=lo,
            prob_max=hi,
            actual_rate=round(buy_rate, 4),
            count=len(signals),
        ))
    return result


@router.get("/model-drift", response_model=list[ModelDriftPoint])
async def get_model_drift(
    _user: CurrentUser,
    db: Optional[AsyncSession] = Depends(get_db),
    limit: int = 90,
) -> list[ModelDriftPoint]:
    """Historical model quality metrics from PostgreSQL (for drift analysis). Empty in JSON-only mode."""
    if db is None:
        return []
    stmt = (
        select(DiagnosticsSnapshot, RunLog.run_at)
        .join(RunLog, DiagnosticsSnapshot.run_id == RunLog.id)
        .order_by(RunLog.run_at.asc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [
        ModelDriftPoint(
            run_at=run_at.isoformat(),
            roc_auc_mean=snap.roc_auc_mean,
            accuracy_mean=snap.accuracy_mean,
        )
        for snap, run_at in rows
    ]


@router.get("/feature-importance-history", response_model=list[FeatureImportanceDriftPoint])
async def get_feature_importance_history(
    _user: CurrentUser,
    db: Optional[AsyncSession] = Depends(get_db),
    limit: int = 30,
) -> list[FeatureImportanceDriftPoint]:
    """Feature importance evolution over time (from PostgreSQL). Empty in JSON-only mode."""
    if db is None:
        return []
    stmt = (
        select(RunLog.id, RunLog.run_at)
        .order_by(RunLog.run_at.desc())
        .limit(limit)
    )
    run_rows = (await db.execute(stmt)).all()

    result = []
    for run_id, run_at in run_rows:
        fi_stmt = (
            select(FeatureImportanceSnapshot)
            .where(FeatureImportanceSnapshot.run_id == run_id)
            .order_by(FeatureImportanceSnapshot.importance.desc())
        )
        fi_rows = (await db.execute(fi_stmt)).scalars().all()
        result.append(FeatureImportanceDriftPoint(
            run_at=run_at.isoformat(),
            importances=[FeatureImportanceItem(feature=fi.feature, importance=fi.importance) for fi in fi_rows],
        ))

    return list(reversed(result))
