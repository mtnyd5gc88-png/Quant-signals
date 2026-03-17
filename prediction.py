from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import TrainedModel


@dataclass(frozen=True)
class StockPrediction:
    ticker: str
    asof: pd.Timestamp
    prob_up: float
    pred_label: int
    current_price: float | None = None
    target_price: float | None = None


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

    prob_up = float(trained.pipeline.predict_proba(row)[:, 1][0])
    pred_label = int(prob_up >= 0.5)

    current_price = float(price_df["Close"].iloc[-1])

    target_price = None
    if compute_target and regressor is not None:
        target_price = float(regressor.predict(row)[0])

    return StockPrediction(
        ticker=ticker,
        asof=asof,
        prob_up=prob_up,
        pred_label=pred_label,
        current_price=current_price,
        target_price=target_price,
    )


def predict_proba_series(feature_df: pd.DataFrame, trained: TrainedModel) -> pd.Series:
    if feature_df.empty:
        return pd.Series(dtype=float)
    X = feature_df[trained.feature_names]
    prob = trained.pipeline.predict_proba(X)[:, 1]
    return pd.Series(prob, index=feature_df.index, name="prob_up")

