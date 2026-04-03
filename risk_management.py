from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct:   float = 0.08   # FIX: 5% → 8% (노이즈에 안 잘리게)
    take_profit_pct: float = 0.25   # FIX: 10% → 25% (복리 수익 보존)
    # 10%는 NVDA/TSLA 급 변동성 종목을 3주만에 강제청산 → 수익 차단


def check_exit(
    entry_price:   float,
    current_price: float,
    cfg:           RiskConfig = RiskConfig(),
) -> str | None:
    """
    Returns:
      "stop_loss"   — current_price <= entry * (1 - stop_loss_pct)
      "take_profit" — current_price >= entry * (1 + take_profit_pct)
      None          — 청산 조건 미달
    """
    if entry_price <= 0 or current_price <= 0:
        return None
    if current_price <= entry_price * (1.0 - cfg.stop_loss_pct):
        return "stop_loss"
    if current_price >= entry_price * (1.0 + cfg.take_profit_pct):
        return "take_profit"
    return None
