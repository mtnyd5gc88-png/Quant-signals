from __future__ import annotations
import json

from fastapi import APIRouter

from api.auth import CurrentUser
from api.config import settings
from api.schemas.portfolio import PortfolioHolding, PortfolioResponse

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _load_portfolio() -> dict:
    path = settings.quant_data_dir / "portfolio.json"
    if not path.exists():
        return {"positions": [], "n_positions": 0, "last_updated": "unknown"}
    return json.loads(path.read_text())


def _load_predictions() -> dict[str, dict]:
    path = settings.quant_data_dir / "predictions.json"
    if not path.exists():
        return {}
    return {r["ticker"]: r for r in json.loads(path.read_text())}


@router.get("", response_model=PortfolioResponse)
async def get_portfolio(_user: CurrentUser) -> PortfolioResponse:
    port  = _load_portfolio()
    preds = _load_predictions()

    holdings = []
    for p in port.get("positions", []):
        sig = preds.get(p["ticker"], {})
        holdings.append(PortfolioHolding(
            ticker=p["ticker"],
            weight=p["weight"],
            signal=sig.get("signal"),
            prob_up=sig.get("prob_up"),
            price=sig.get("price"),
            target_return=sig.get("target_return"),
        ))

    total_weight = sum(h.weight for h in holdings)
    buy_probs    = [h.prob_up for h in holdings if h.prob_up is not None]
    expected_ret = (sum(buy_probs) / len(buy_probs) - 0.5) * 0.2 if buy_probs else 0.0

    return PortfolioResponse(
        holdings=holdings,
        total_weight=round(total_weight, 4),
        n_positions=port.get("n_positions", len(holdings)),
        expected_return=round(expected_ret, 4),
        last_updated=port.get("last_updated", "unknown"),
    )
