from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
import shutil
import datetime

from backtest import run_portfolio_backtest
from data_loader import DataConfig, align_on_common_dates, ensure_min_history, load_data
from evaluation import summarize_performance
from feature_engineering import FeatureConfig, add_features, feature_columns
from model import (
    select_best_model,
    train_and_select_model,
    walk_forward_predict_proba,
    walk_forward_validate,
    _make_random_forest_regressor,
)
from portfolio import PortfolioConfig
from market_regime import compute_market_regime
from prediction import predict_latest, predict_proba_series
from strategy import RecommendationThresholds, recommendation_from_probability
from visualization import (
    plot_drawdown,
    plot_equity_curve,
    plot_feature_importance,
    plot_strategy_vs_benchmark,
)


def main() -> None:

    import shutil
    shutil.rmtree("data", ignore_errors=True)

    print("Run time:", datetime.datetime.now())

    # --------------------
    # Settings
    # --------------------
    tickers = [
        "AMD","INTC","MU","QCOM","AMAT","LRCX","KLAC","ADI","NXPI","MCHP"
    ]

    BENCHMARK = "SPY"

    START_DATE = "2013-01-01"

    FEATURE_CFG = FeatureConfig()

    RECO_THRESH = RecommendationThresholds(buy=0.55, hold_low=0.40, hold_high=0.55)

    PORTFOLIO_CFG = PortfolioConfig(initial_capital=100_000.0, top_n=10, max_exposure=0.90)

    OUT_PLOTS = Path("outputs/plots")
    OUT_REPORT = Path("outputs/reports/performance_report.txt")

    # --------------------
    # Load data
    # --------------------
    cfg = DataConfig(
        tickers=tickers + [BENCHMARK],
        start=START_DATE,
        cache_dir=Path("data"),
        use_cache=False
    )

    raw = load_data(cfg)
    spy_df = raw.pop(BENCHMARK)

    raw = ensure_min_history(raw, min_days=800)

    # --------------------
    # Model
    # --------------------
    feat_cols = feature_columns(FEATURE_CFG)

    latest_preds = []
    per_ticker_probs = {}

    print("Training models...")

    for ticker, df in raw.items():

        feats = add_features(df, FEATURE_CFG)

        candidates = train_and_select_model(feats, feat_cols, test_size=0.2)
        best = select_best_model(candidates)

        fitted_full = clone(best.pipeline).fit(feats[feat_cols], feats["target"].values)

        reg_pipe = _make_random_forest_regressor()

        horizon = 5
        y_reg = (feats["Close"].shift(-horizon) - feats["Close"]) / feats["Close"]
        valid_idx = y_reg.notna()

        reg_pipe.fit(feats.loc[valid_idx, feat_cols], y_reg.loc[valid_idx])

        best_full = best.__class__(
            name=best.name,
            pipeline=fitted_full,
            feature_names=best.feature_names,
            metrics=best.metrics
        )

        pred = predict_latest(
            ticker,
            feats.iloc[[-1]],
            df,
            best_full,
            regressor=reg_pipe,
            compute_target=True,
        )

        latest_preds.append(pred)

        probs = walk_forward_predict_proba(
            feats,
            feat_cols,
            model_name=best.name,
            train_years=5,
            step_years=1,
        )

        # 🔥 기대수익 낮으면 매수 금지
        if pred.target_price is not None and pred.target_price < 0.02:
            probs = probs * 0

        per_ticker_probs[ticker] = probs

    # --------------------
    # Recommendations
    # --------------------
    print("\nLatest predictions:")

    latest_preds_sorted = sorted(latest_preds, key=lambda p: p.prob_up, reverse=True)

    for p in latest_preds_sorted:

        rec = recommendation_from_probability(p.prob_up, RECO_THRESH)

        expected_return = float(p.target_price) if p.target_price is not None else None

        # 🔥 BUY 기준 강화 (1% → 2%)
        if expected_return is not None and expected_return <= 0.02:
            rec = "HOLD"

        if rec == "BUY" and expected_return is not None:
            target_price = p.current_price * (1.0 + expected_return)

            print(
                f"- {p.ticker} | BUY | "
                f"{p.current_price:.2f} → {target_price:.2f} "
                f"({expected_return*100:+.2f}%) | prob={p.prob_up:.2f}"
            )
        else:
            print(f"- {p.ticker} | {rec} | prob={p.prob_up:.2f}")

    # --------------------
    # Backtest
    # --------------------
    price_by_ticker = {
        t: df["Close"] for t, df in raw.items()
    }

    bt = run_portfolio_backtest(
        price_by_ticker=price_by_ticker,
        prob_by_ticker=per_ticker_probs,
        cfg=PORTFOLIO_CFG
    )

    spy_px = spy_df["Close"].loc[bt.equity_curve.index]
    strategy_eq = bt.equity_curve.loc[spy_px.index]
    benchmark_eq = PORTFOLIO_CFG.initial_capital * (spy_px / spy_px.iloc[0])

    report = summarize_performance(strategy_eq, benchmark_eq, bt.trades)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with OUT_REPORT.open("w") as f:
        f.write(f"Total return: {report.cumulative_return:.2%}\n")
        f.write(f"Sharpe: {report.sharpe_ratio:.2f}\n")
        f.write(f"Max DD: {report.max_drawdown:.2%}\n")

    plot_equity_curve(strategy_eq, OUT_PLOTS / "equity_curve.png")

    # --------------------
    # JSON OUTPUT
    # --------------------
    import json

    data = []

    for p in latest_preds_sorted:

        rec = recommendation_from_probability(p.prob_up, RECO_THRESH)

        expected_return = float(p.target_price) if p.target_price is not None else 0.0

        # 🔥 단일 기준 유지 (frontend랑 동일)
        if expected_return < 0.02:
            rec = "HOLD"

        data.append({
            "ticker": p.ticker,
            "prob_up": round(float(p.prob_up), 3),
            "price": float(p.current_price),
            "target_return": float(expected_return),
            "signal": rec
        })

    Path("website/data").mkdir(parents=True, exist_ok=True)

    with open("website/data/predictions.json", "w") as f:
        json.dump(data, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
