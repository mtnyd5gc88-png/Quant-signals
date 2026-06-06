from __future__ import annotations
import json

from fastapi import APIRouter
from pydantic import BaseModel

from api.auth import CurrentUser
from api.config import settings

router = APIRouter(prefix="/regime", tags=["regime"])


class RegimeResponse(BaseModel):
    regime: str
    last_updated: str


@router.get("", response_model=RegimeResponse)
async def get_regime(_user: CurrentUser) -> RegimeResponse:
    path = settings.quant_data_dir / "regime.json"
    if not path.exists():
        return RegimeResponse(regime="unknown", last_updated="unknown")
    data = json.loads(path.read_text())
    return RegimeResponse(regime=data.get("regime", "unknown"), last_updated=data.get("last_updated", "unknown"))
