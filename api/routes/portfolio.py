from __future__ import annotations
import json

from fastapi import APIRouter

from api.auth import CurrentUser
from api.config import settings
from api.schemas.portfolio import PortfolioResponse, PositionItem

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _load_portfolio() -> dict:
    path = settings.quant_data_dir / "portfolio.json"
    if not path.exists():
        return {"positions": [], "n_positions": 0, "last_updated": "unknown", "method": "unknown", "rebal_freq": "ME"}
    return json.loads(path.read_text())


def _load_predictions() -> dict[str, dict]:
    path = settings.quant_data_dir / "predictions.json"
    if not path.exists():
        return {}
    return {r["ticker"]: r for r in json.loads(path.read_text())}


@router.get("", response_model=PortfolioResponse)
async def get_portfolio(_user: CurrentUser) -> PortfolioResponse:
    port = _load_portfolio()
    preds = _load_predictions()

    positions = []
    for p in port.get("positions", []):
        sig = preds.get(p["ticker"], {})
        tr = sig.get("target_return")
        positions.append(PositionItem(
            ticker=p["ticker"],
            weight=p["weight"],
            signal=sig.get("signal"),
            prob_up=sig.get("prob_up"),
            expected_return_pct=round(tr * 100, 2) if tr is not None else None,
        ))

    return PortfolioResponse(
        positions=positions,
        n_positions=port.get("n_positions", len(positions)),
        last_rebal_date=port.get("last_rebal_date"),
        last_updated=port.get("last_updated", "unknown"),
        method=port.get("method", "equal_weight"),
        rebal_freq=port.get("rebal_freq", "ME"),
    )
