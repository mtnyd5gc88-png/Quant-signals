from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float


@dataclass(frozen=True)
class TrainedModel:
    name: str
    pipeline: Pipeline
    feature_names: list[str]
    metrics: ModelMetrics


def _make_log_reg() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def _make_random_forest(random_state: int = 42, n_jobs: int = 2) -> Pipeline:
    """
    n_jobs=2 by default (not -1) so that when we parallelise across tickers
    with ThreadPoolExecutor we don't over-subscribe the CPU.
    n_estimators reduced from 120 → 80 for ~35% speed gain with minimal
    accuracy loss on typical financial datasets.
    """
    base_rf = RandomForestClassifier(
        n_estimators=80,          # was 120; 80 is sufficient for most datasets
        max_depth=None,
        min_samples_leaf=3,       # slightly more regularization
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_rf,
        method="sigmoid",
        cv=3,                     # 3-fold gives better calibration than 2
    )
    return Pipeline(steps=[("clf", calibrated_clf)])


def _make_random_forest_regressor(random_state: int = 42, n_jobs: int = 2) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=None,
                    min_samples_leaf=3,
                    n_jobs=n_jobs,
                    random_state=random_state,
                ),
            )
        ]
    )


def chronological_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    n = len(df)
    split_idx = int(round(n * (1 - test_size)))
    split_idx = max(1, min(split_idx, n - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ModelMetrics:
    unique = np.unique(y_true)
    roc = float("nan") if unique.size < 2 else float(roc_auc_score(y_true, y_prob))
    return ModelMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
    )


def train_and_select_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, TrainedModel]:
    """
    Train LR and RF using an 80/20 chronological split. Returns both with metrics.
    """
    train_df, test_df = chronological_split(df, test_size=test_size)
    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    models: Dict[str, Pipeline] = {
        "logistic_regression": _make_log_reg(),
        "random_forest": _make_random_forest(random_state=random_state),
    }

    out: Dict[str, TrainedModel] = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = evaluate_classifier(y_test, pred, prob)
        out[name] = TrainedModel(name=name, pipeline=pipe, feature_names=feature_cols, metrics=metrics)
    return out


def select_best_model(models: Dict[str, TrainedModel]) -> TrainedModel:
    """
    Pick best model primarily by ROC-AUC, then by accuracy.
    NaN ROC-AUC (single-class period) is treated as -1 so LR wins as fallback.
    """
    def key(m: TrainedModel) -> tuple:
        roc = m.metrics.roc_auc
        roc_key = -1.0 if np.isnan(roc) else roc
        return (roc_key, m.metrics.accuracy)

    return sorted(models.values(), key=key, reverse=True)[0]


@dataclass(frozen=True)
class WalkForwardFoldResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: ModelMetrics


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    model_name: str = "random_forest",
    initial_train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    random_state: int = 42,
) -> list[WalkForwardFoldResult]:
    """Walk-forward validation for diagnostic purposes (not used in the main pipeline)."""
    if df.empty:
        return []

    if model_name not in {"logistic_regression", "random_forest"}:
        raise ValueError("model_name must be 'logistic_regression' or 'random_forest'")

    # FIX: use functools.partial instead of lambda for pickling compatibility
    # (needed if this function is ever called from a multiprocessing context)
    make_model = (
        _make_log_reg
        if model_name == "logistic_regression"
        else functools.partial(_make_random_forest, random_state=random_state)
    )

    start = df.index.min()
    end = df.index.max()

    folds: list[WalkForwardFoldResult] = []
    train_start = start
    train_end = train_start + pd.DateOffset(years=initial_train_years)

    while True:
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_start >= end:
            break

        train_slice = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test_slice = df.loc[(df.index >= test_start) & (df.index < test_end)]
        if len(train_slice) < 200 or len(test_slice) < 50:
            break

        X_train, y_train = train_slice[feature_cols], train_slice[target_col].values
        X_test, y_test = test_slice[feature_cols], test_slice[target_col].values

        model = make_model()
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = evaluate_classifier(y_test, pred, prob)
        folds.append(
            WalkForwardFoldResult(
                train_start=pd.Timestamp(train_slice.index.min()),
                train_end=pd.Timestamp(train_slice.index.max()),
                test_start=pd.Timestamp(test_slice.index.min()),
                test_end=pd.Timestamp(test_slice.index.max()),
                metrics=metrics,
            )
        )

        train_end = train_end + pd.DateOffset(years=step_years)
        if train_end >= end:
            break

    return folds


def walk_forward_predict_proba(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    model_name: str = "random_forest",
    train_years: int = 5,
    step_years: int = 1,
    random_state: int = 42,
    start_test_date: pd.Timestamp | None = None,
) -> pd.Series:
    """
    Generate out-of-sample probabilities using walk-forward retraining.

    MATH NOTE:
    - Each test window is trained on the preceding `train_years` of data.
    - No test observation ever appears in its own training window → strict OOS.
    - start_test_date should be set to (data_start + train_years) in main.py
      so the full available history is used for backtesting (more stable Sharpe).
    - Default fallback (last 20%) is kept only for backward compat.
    """
    if df.empty:
        return pd.Series(dtype=float)

    if model_name not in {"logistic_regression", "random_forest"}:
        raise ValueError("model_name must be 'logistic_regression' or 'random_forest'")

    # FIX: functools.partial instead of lambda for pickling safety
    make_model = (
        _make_log_reg
        if model_name == "logistic_regression"
        else functools.partial(_make_random_forest, random_state=random_state)
    )

    idx = df.index
    end = idx.max()

    if start_test_date is None:
        # Legacy fallback: last 20% of dates
        split_idx = int(round(len(df) * 0.8))
        start_test_date = df.index[split_idx]

    probs = pd.Series(index=df.loc[df.index >= start_test_date].index, dtype=float)

    anchor = start_test_date
    while anchor < end:
        train_end = anchor
        train_start = train_end - pd.DateOffset(years=train_years)
        test_end = anchor + pd.DateOffset(years=step_years)

        train_slice = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test_slice = df.loc[(df.index >= anchor) & (df.index < test_end)]
        if len(train_slice) < 200 or len(test_slice) == 0:
            anchor = test_end
            continue

        model = make_model()
        model.fit(train_slice[feature_cols], train_slice[target_col].values)
        prob = model.predict_proba(test_slice[feature_cols])[:, 1]
        probs.loc[test_slice.index] = prob

        anchor = test_end

    return probs.dropna()
