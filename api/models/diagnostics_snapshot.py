from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.base import Base


class DiagnosticsSnapshot(Base):
    __tablename__ = "diagnostics_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("run_log.id", ondelete="CASCADE"))
    roc_auc_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accuracy_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_daily_turnover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    annual_turnover_multiple: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost_drag_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gross_cagr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_cagr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    run: Mapped["RunLog"] = relationship(back_populates="diagnostics")
