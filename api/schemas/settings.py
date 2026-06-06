from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class RunConfig(BaseModel):
    top_n_alpha: int
    equal_weight: bool
    rebal_freq: str
    transaction_cost_rate: float
    slippage_rate: float
    risk_free_rate: float
    max_position_w: float
    min_position_w: float
    walk_forward_train_years: int
    walk_forward_step_years: int
    n_tickers_universe: int
    n_tickers_trained: int
    run_time: str
    n_trades: int


class SystemStatus(BaseModel):
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    status: str
    last_error: Optional[str] = None


class SettingsResponse(BaseModel):
    run_config: RunConfig
    system: SystemStatus
