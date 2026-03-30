from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import TrainedModel


@dataclass(frozen=True)
class StockPrediction:
    ticker: str
    asof: pd.Timestamp
    expected_return: float              # PRIMARY SIGNAL ← 변경됨
    prob_up: float                      # AUXILIARY ONLY
    confidence: float                   # NEW ← 추가됨
    current_price: float | None = None


def predict_latest(
    ticker: str,
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    trained: TrainedModel,
    regressor=None,
    compute_target: bool = False,
) -> StockPrediction:

    if feature_df.empty:
        raise ValueError(f"No feature rows for {ticker}")

    row = feature_df.iloc[[-1]][trained.feature_names]
    asof = pd.Timestamp(feature_df.index[-1])

    # Classification (AUXILIARY only)
    prob_up = float(trained.pipeline.predict_proba(row)[:, 1][0])
    
    # Confidence metric
    confidence = abs(prob_up - 0.5)

    current_price = float(price_df["Close"].iloc[-1])

    # Regression output (PRIMARY signal)
    expected_return = None
    if compute_target and regressor is not None:
        expected_return = float(regressor.predict(row)[0])

    return StockPrediction(
        ticker=ticker,
        asof=asof,
        expected_return=expected_return,
        prob_up=prob_up,
        confidence=confidence,
        current_price=current_price,
    )


def predict_proba_series(feature_df: pd.DataFrame, trained: TrainedModel) -> pd.Series:
    if feature_df.empty:
        return pd.Series(dtype=float)
    X = feature_df[trained.feature_names]
    prob = trained.pipeline.predict_proba(X)[:, 1]
    
    # Center + scale to directional signal: [-1, 1] range
    signal = (prob - 0.5) * 2
    
    return pd.Series(signal, index=feature_df.index, name="directional_signal")
