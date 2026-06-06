from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from api.config import settings
from api.db import AsyncSessionLocal
from api.ingest import ingest_latest_run

log = logging.getLogger(__name__)

_scheduler: Optional[AsyncIOScheduler] = None
_run_status: dict = {
    "status": "idle",       # "idle" | "running" | "error"
    "last_run": None,
    "next_run": None,
    "last_error": None,
}


def get_run_status() -> dict:
    status = dict(_run_status)
    if _scheduler and _scheduler.running:
        job = _scheduler.get_job("daily_quant_run")
        if job and job.next_run_time:
            status["next_run"] = job.next_run_time.isoformat()
    return status


async def _run_quant_engine() -> None:
    global _run_status
    if _run_status["status"] == "running":
        log.warning("Quant run already in progress; skipping")
        return

    _run_status["status"] = "running"
    _run_status["last_error"] = None
    log.info("Starting scheduled quant run")

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "main.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"main.py exited with code {proc.returncode}")

        # Ingest JSON outputs into PostgreSQL (skipped in JSON-only mode)
        from api.db import DB_AVAILABLE
        if DB_AVAILABLE:
            async with AsyncSessionLocal() as db:
                await ingest_latest_run(db)
        else:
            log.info("JSON-only mode — skipping PostgreSQL ingest")

        _run_status["last_run"] = datetime.now(timezone.utc).isoformat()
        _run_status["status"] = "idle"
        log.info("Quant run completed successfully")

    except Exception as exc:
        _run_status["status"] = "error"
        _run_status["last_error"] = str(exc)
        log.error("Quant run failed: %s", exc)


def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    _scheduler = AsyncIOScheduler()

    cron_parts = settings.scheduler_cron.split()
    trigger = CronTrigger(
        minute=cron_parts[0],
        hour=cron_parts[1],
        day=cron_parts[2],
        month=cron_parts[3],
        day_of_week=cron_parts[4],
        timezone="UTC",
    )

    _scheduler.add_job(
        _run_quant_engine,
        trigger=trigger,
        id="daily_quant_run",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _scheduler.start()
    log.info("Scheduler started (cron=%s)", settings.scheduler_cron)
    return _scheduler


def stop_scheduler() -> None:
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
