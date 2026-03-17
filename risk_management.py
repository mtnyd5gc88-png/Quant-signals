from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


def check_exit(entry_price: float, current_price: float, cfg: RiskConfig = RiskConfig()) -> str | None:
    """
    Returns:
      - "stop_loss" if price <= entry*(1-stop_loss_pct)
      - "take_profit" if price >= entry*(1+take_profit_pct)
      - None otherwise
    """
    if entry_price <= 0 or current_price <= 0:
        return None
    if current_price <= entry_price * (1 - cfg.stop_loss_pct):
        return "stop_loss"
    if current_price >= entry_price * (1 + cfg.take_profit_pct):
        return "take_profit"
    return None

