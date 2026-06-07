from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from api.auth import CurrentUser
from api.config import settings

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
async def search_ticker(
    q: str = Query(..., max_length=10),
    _user: CurrentUser = ...,
) -> dict:
    path = settings.quant_data_dir / "predictions.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Data not available")
    raw: list[dict] = json.loads(path.read_text())
    q_upper = q.strip().upper()
    match = next((p for p in raw if p.get("ticker", "").upper() == q_upper), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No signal data for {q_upper}")
    return match
