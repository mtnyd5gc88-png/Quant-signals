from __future__ import annotations

import json

from fastapi import APIRouter

from api.auth import CurrentUser
from api.config import settings
from api.scheduler import get_run_status
from api.schemas.settings import RunConfig, SettingsResponse, SystemStatus

router = APIRouter(prefix="/settings", tags=["settings"])


def _load_run_config() -> dict:
    path = settings.quant_data_dir / "diagnostics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("run_config", {})


@router.get("", response_model=SettingsResponse)
async def get_settings(_user: CurrentUser) -> SettingsResponse:
    rc = _load_run_config()
    status = get_run_status()

    return SettingsResponse(
        run_config=RunConfig(
            top_n_alpha=rc.get("top_n_alpha", 5),
            equal_weight=rc.get("equal_weight", True),
            rebal_freq=rc.get("rebal_freq", "ME"),
            transaction_cost_rate=rc.get("transaction_cost_rate", 0.001),
            slippage_rate=rc.get("slippage_rate", 0.0005),
            risk_free_rate=rc.get("risk_free_rate", 0.04),
            max_position_w=rc.get("max_position_w", 0.15),
            min_position_w=rc.get("min_position_w", 0.01),
            walk_forward_train_years=rc.get("walk_forward_train_years", 2),
            walk_forward_step_years=rc.get("walk_forward_step_years", 1),
            n_tickers_universe=rc.get("n_tickers_universe", 0),
            n_tickers_trained=rc.get("n_tickers_trained", 0),
            run_time=rc.get("run_time", "unknown"),
            n_trades=rc.get("n_trades", 0),
        ),
        system=SystemStatus(
            last_run=status.get("last_run"),
            next_run=status.get("next_run"),
            status=status.get("status", "idle"),
            last_error=status.get("last_error"),
        ),
    )
