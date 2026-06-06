from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.base import Base


class RunLog(Base):
    __tablename__ = "run_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    n_tickers: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    cagr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="success")
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    signals: Mapped[List["SignalSnapshot"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    positions: Mapped[List["PortfolioSnapshot"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    diagnostics: Mapped[List["DiagnosticsSnapshot"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    feature_importances: Mapped[List["FeatureImportanceSnapshot"]] = relationship(back_populates="run", cascade="all, delete-orphan")
