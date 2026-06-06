from __future__ import annotations
"""
Reads JSON files written by main.py and persists them to PostgreSQL.
Called after each successful main.py run.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.models.diagnostics_snapshot import DiagnosticsSnapshot
from api.models.portfolio_snapshot import PortfolioSnapshot
from api.models.run_log import RunLog
from api.models.signal_snapshot import FeatureImportanceSnapshot, SignalSnapshot

log = logging.getLogger(__name__)


def _read_json(name: str) -> dict | list | None:
    path = settings.quant_data_dir / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        log.warning("Failed to read %s: %s", name, exc)
        return None


async def ingest_latest_run(db: AsyncSession) -> RunLog:
    """
    Read all JSON outputs from the latest main.py run and write to PostgreSQL.
    Returns the created RunLog row.
    """
    metrics = _read_json("metrics.json") or {}
    regime  = _read_json("regime.json") or {}
    diag    = _read_json("diagnostics.json") or {}
    port    = _read_json("portfolio.json") or {}
    signals = _read_json("predictions.json") or []

    run_time_str = diag.get("run_config", {}).get("run_time")
    run_at = (
        datetime.fromisoformat(run_time_str).replace(tzinfo=timezone.utc)
        if run_time_str
        else datetime.now(timezone.utc)
    )

    run = RunLog(
        run_at=run_at,
        n_tickers=diag.get("run_config", {}).get("n_tickers_trained"),
        regime=regime.get("regime"),
        cagr=metrics.get("annualized_return"),
        sharpe=metrics.get("sharpe_ratio"),
        max_drawdown=metrics.get("max_drawdown"),
        status="success",
    )
    db.add(run)
    await db.flush()   # get run.id

    # Signal snapshots
    run_date = run_at.date()
    # Compute alpha scores as z-scores of prob_up across universe
    probs = [s.get("prob_up", 0.5) for s in signals]
    if len(probs) > 1:
        import statistics
        mean_p = statistics.mean(probs)
        std_p  = statistics.stdev(probs) or 1e-6
        z_probs = [(p - mean_p) / std_p for p in probs]
    else:
        z_probs = [0.0] * len(probs)

    for sig, z in zip(signals, z_probs):
        db.add(SignalSnapshot(
            run_id=run.id,
            date=run_date,
            ticker=sig["ticker"],
            prob_up=sig.get("prob_up"),
            signal=sig.get("signal"),
            price=sig.get("price"),
            target_return=sig.get("target_return"),
            alpha_score=round(z, 4),
        ))

    # Portfolio snapshots
    last_rebal = port.get("last_rebal_date")
    rebal_date = datetime.strptime(last_rebal, "%Y-%m-%d").date() if last_rebal else run_date
    for pos in port.get("positions", []):
        db.add(PortfolioSnapshot(
            run_id=run.id,
            date=rebal_date,
            ticker=pos["ticker"],
            weight=pos["weight"],
        ))

    # Diagnostics snapshot
    t = diag.get("turnover", {})
    mq = diag.get("model_quality", {})
    db.add(DiagnosticsSnapshot(
        run_id=run.id,
        roc_auc_mean=mq.get("roc_auc_mean"),
        accuracy_mean=mq.get("accuracy_mean"),
        precision_mean=mq.get("precision_mean"),
        recall_mean=mq.get("recall_mean"),
        avg_daily_turnover=t.get("avg_daily"),
        annual_turnover_multiple=t.get("annual_multiple"),
        cost_drag_pct=t.get("cost_drag_pct"),
        gross_cagr=t.get("gross_cagr"),
        net_cagr=t.get("net_cagr"),
    ))

    # Feature importance snapshots
    for fi in diag.get("feature_importance", []):
        db.add(FeatureImportanceSnapshot(
            run_id=run.id,
            feature=fi["feature"],
            importance=fi["importance"],
        ))

    await db.commit()
    log.info("Ingested run %s (run_id=%d)", run_at.isoformat(), run.id)
    return run
