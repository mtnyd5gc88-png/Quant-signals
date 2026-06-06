from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ModelQuality(BaseModel):
    roc_auc_mean: Optional[float] = None
    accuracy_mean: Optional[float] = None
    precision_mean: Optional[float] = None
    recall_mean: Optional[float] = None
    n_tickers: int


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class TurnoverStats(BaseModel):
    avg_daily: float
    annual_multiple: float
    cost_drag_pct: float
    gross_cagr: float
    net_cagr: float


class AlphaAttribution(BaseModel):
    ensemble_cagr: float
    alpha_ann: float
    beta: float


class DiagnosticsResponse(BaseModel):
    model_quality: ModelQuality
    feature_importance: list[FeatureImportanceItem]
    turnover: TurnoverStats
    alpha_attribution: AlphaAttribution
    last_run: str


class CalibrationBucket(BaseModel):
    prob_min: float
    prob_max: float
    actual_rate: float
    count: int


class ModelDriftPoint(BaseModel):
    run_at: str
    roc_auc_mean: Optional[float] = None
    accuracy_mean: Optional[float] = None


class FeatureImportanceDriftPoint(BaseModel):
    run_at: str
    importances: list[FeatureImportanceItem]
