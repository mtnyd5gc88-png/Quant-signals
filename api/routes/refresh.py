from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from api.auth import CurrentUser
from api.scheduler import _run_quant_engine, get_run_status

router = APIRouter(prefix="/refresh", tags=["refresh"])


class RefreshTriggerResponse(BaseModel):
    status: str
    message: str


class RefreshStatusResponse(BaseModel):
    status: str
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_error: Optional[str] = None


@router.post("", response_model=RefreshTriggerResponse)
async def trigger_refresh(
    _user: CurrentUser,
    background_tasks: BackgroundTasks,
) -> RefreshTriggerResponse:
    status = get_run_status()
    if status["status"] == "running":
        return RefreshTriggerResponse(status="already_running", message="A quant run is already in progress")
    background_tasks.add_task(_run_quant_engine)
    return RefreshTriggerResponse(status="queued", message="Quant run queued — this takes ~15 minutes")


@router.get("/status", response_model=RefreshStatusResponse)
async def get_refresh_status(_user: CurrentUser) -> RefreshStatusResponse:
    s = get_run_status()
    return RefreshStatusResponse(
        status=s["status"],
        last_run=s.get("last_run"),
        next_run=s.get("next_run"),
        last_error=s.get("last_error"),
    )
