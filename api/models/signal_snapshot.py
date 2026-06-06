from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import Date, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.base import Base


class SignalSnapshot(Base):
    __tablename__ = "signal_snapshot"
    __table_args__ = (UniqueConstraint("run_id", "ticker", name="uq_signal_run_ticker"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("run_log.id", ondelete="CASCADE"))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    prob_up: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    signal: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    target_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    alpha_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    run: Mapped["RunLog"] = relationship(back_populates="signals")


class FeatureImportanceSnapshot(Base):
    __tablename__ = "feature_importance_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("run_log.id", ondelete="CASCADE"))
    feature: Mapped[str] = mapped_column(String(50), nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped["RunLog"] = relationship(back_populates="feature_importances")
