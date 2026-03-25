"""
Quant Trading System — Main entry point
Fixes applied:
  1. Caching enabled by default (FORCE_REFRESH flag to override)
  2. Parallel ticker processing via ThreadPoolExecutor
  3. Look-ahead bias removed from backtest (was #1 cause of Sharpe instability)
  4. Walk-forward history extended to use maximum available data
  5. HTML dashboard generated instead of just static PNGs + txt
  6. Duplicate tickers deduplicated
  7. Mathematical notes inline
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from backtest import run_portfolio_backtest
from data_loader import DataConfig, ensure_min_history, load_data
from evaluation import PerformanceReport, summarize_performance
from feature_engineering import FeatureConfig, add_features, feature_columns
from model import (
    TrainedModel,
    _make_random_forest_regressor,
    select_best_model,
    train_and_select_model,
    walk_forward_predict_proba,
)
from portfolio import PortfolioConfig
from market_regime import compute_market_regime
from prediction import predict_latest
from strategy import RecommendationThresholds, recommendation_from_probability
from visualization import (
    generate_html_dashboard,
    plot_drawdown,
    plot_equity_curve,
    plot_feature_importance,
    plot_strategy_vs_benchmark,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level ticker worker (must be at module level for ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────
_print_lock = threading.Lock()


def _train_single_ticker(
    ticker: str,
    df: pd.DataFrame,
    spy_close: pd.Series,
    feat_cols: list[str],
    feat_cfg: FeatureConfig,
    wf_train_years: int,
    wf_step_years: int,
) -> tuple:
    """
    Feature engineering → model selection → walk-forward probs → latest prediction.
    Returns (ticker, best_model, probs_series, prediction, importances_series | None).
    Returns (ticker, None, None, None, None) on any failure.
    """
    try:
        feats = add_features(df, feat_cfg, benchmark_close=spy_close)
        if len(feats) < 400:
            return ticker, None, None, None, None

        # ── Model training (80/20 chronological split for selection)
        candidates = train_and_select_model(feats, feat_cols, test_size=0.2)
        best = select_best_model(candidates)

        # ── Walk-forward OOS probabilities
        # FIX: start test date from first viable date (after initial training window)
        # instead of defaulting to the last 20% of the data. This gives us a much
        # longer backtest window and more stable Sharpe estimates.
        feat_start = feats.index.min()
        wf_start = feat_start + pd.DateOffset(years=wf_train_years)

        probs = walk_forward_predict_proba(
            feats,
            feat_cols,
            model_name=best.name,
            train_years=wf_train_years,
            step_years=wf_step_years,
            start_test_date=wf_start,
        )

        # ── Refit on all data for latest-date prediction (no future leakage here
        #    because we're predicting the CURRENT last row, not historical rows)
        fitted_full = clone(best.pipeline).fit(feats[feat_cols], feats["target"].values)

        # ── Regression model for expected 5-day return (used for target price display)
        #    NOTE: target_price stored in StockPrediction is a RETURN (e.g. 0.05 = 5%),
        #    not an actual dollar price — naming quirk kept for backward compat.
        reg_pipe = _make_random_forest_regressor()
        horizon = 5
        y_reg = (feats["Close"].shift(-horizon) - feats["Close"]) / feats["Close"]
        valid_mask = y_reg.notna()
        if valid_mask.sum() >= 50:
            reg_pipe.fit(feats.loc[valid_mask, feat_cols], y_reg.loc[valid_mask])
        else:
            reg_pipe = None

        best_full = TrainedModel(
            name=best.name,
            pipeline=fitted_full,
            feature_names=best.feature_names,
            metrics=best.metrics,
        )

        feats_for_pred = feats.reindex(df.index).ffill().iloc[[-1]]
        pred = predict_latest(
            ticker,
            feats_for_pred,
            df,
            best_full,
            regressor=reg_pipe,
            compute_target=(reg_pipe is not None),
        )

        # ── Feature importances (only for RF, used for the first successful ticker)
        importances = None
        if best.name == "random_forest":
            clf = fitted_full.named_steps["clf"]
            rf_model = clf.estimator if hasattr(clf, "estimator") else clf
            if hasattr(rf_model, "feature_importances_"):
                importances = pd.Series(rf_model.feature_importances_, index=feat_cols)

        with _print_lock:
            roc = best.metrics.roc_auc
            acc = best.metrics.accuracy
            n_probs = len(probs)
            print(
                f"  ✓ {ticker:<6} model={best.name[:2].upper()} "
                f"ROC={roc:.3f} acc={acc:.3f} probs={n_probs}"
            )

        return ticker, best, probs, pred, importances

    except Exception as exc:  # noqa: BLE001
        with _print_lock:
            print(f"  ✗ {ticker}: {exc}")
        return ticker, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    run_start = datetime.datetime.now()
    print(f"Run started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # ──────────────────────────────────────────────────────────────────────────
    # ① USER SETTINGS
    # ──────────────────────────────────────────────────────────────────────────

    # Set FORCE_REFRESH=True only when you need fresh data.
    # Keeping cache saves ~10-20 min per run for large universes.
    FORCE_REFRESH: bool = False

    # Deduplicated ticker universe (keep unique only, preserving order)
    _raw_tickers = [
        # AI / Semiconductor
        "NVDA", "AMD", "SMCI", "AVGO", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "MU",
        "QCOM", "MRVL", "ON", "NXPI", "ADI", "MCHP", "TER", "ENTG", "SWKS", "QRVO",
        "INTC", "CDNS", "SNPS", "ANET", "ARM", "MPWR", "COHR", "LSCC",
        # High-growth software
        "MSFT", "GOOGL", "META", "AMZN", "CRM", "NOW", "SNOW", "DDOG", "NET", "CRWD",
        "ZS", "OKTA", "MDB", "PANW", "TEAM", "WDAY", "HUBS", "SHOP", "TTD",
        "PLTR", "PATH", "ESTC", "UPST", "AFRM", "COIN", "SQ",
        # EV / Clean Energy
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "QS", "PLUG", "RUN",
        "ENPH", "SEDG", "FSLR", "BE", "FCEL", "CHPT", "EVGO", "ARRY", "NEE",
        # Oil / Commodities
        "XOM", "CVX", "OXY", "DVN", "EOG", "APA", "MRO", "SLB", "HAL",
        "CHK", "AR", "BTU", "NUE", "STLD",
        # Biotech
        "MRNA", "BNTX", "VRTX", "REGN", "GILD", "AMGN", "BIIB", "ALNY", "EXAS",
        "CRSP", "NTLA", "ILMN", "RXRX", "IONS",
        # Fintech
        "PYPL", "SOFI", "HOOD", "ALLY", "LC", "MELI", "NU", "SE",
        # Consumer / Discretionary
        "NFLX", "DIS", "ROKU", "SPOT", "UBER", "LYFT", "DASH", "ABNB",
        "ETSY", "PINS", "SNAP", "CHWY", "CVNA", "DKNG", "PENN", "MGM", "WYNN", "RCL",
        # Industrial / Defense
        "CAT", "DE", "ETN", "PH", "ROK", "EMR", "DOV", "IR", "XYL", "HON",
        "LMT", "RTX", "NOC", "GD", "BA", "TDG", "HEI",
        # China / EM
        "BABA", "JD", "PDD", "BIDU", "NTES", "BEKE",
        # ETFs (high-beta)
        "ARKK", "SOXL", "TQQQ",
        # Large-cap quality
        "AAPL", "ADBE", "INTU", "ISRG", "ZTS", "DXCM", "IDXX", "TMO", "DHR",
        "HCA", "UNH", "LLY",
        "MA", "V", "AXP", "GS", "MS", "BLK", "SCHW", "CME", "ICE", "SPGI",
        "UPS", "FDX", "UNP", "CSX", "NSC", "ODFL", "DAL", "UAL",
        "HD", "LOW", "COST", "WMT", "TGT", "NKE", "SBUX", "MCD", "CMG",
        "PG", "KO", "PEP", "MDLZ", "CL", "EL", "GIS",
        "LIN", "APD", "ECL", "SHW", "PPG", "DD", "DOW", "LYB",
        "ITW", "SWK", "PNR",
    ]
    # Deduplicate while preserving order
    tickers: list[str] = list(dict.fromkeys(_raw_tickers))

    BENCHMARK = "SPY"
    VIX_TICKER = "^VIX"

    START_DATE = "2013-01-01"
    END_DATE: str | None = None

    FEATURE_CFG = FeatureConfig()
    USE_MARKET_REGIME: bool = True

    # Recommendation thresholds (baseline; adjusted by regime below)
    RECO_THRESH = RecommendationThresholds(buy=0.55, hold_low=0.40, hold_high=0.55)

    PORTFOLIO_CFG = PortfolioConfig(initial_capital=100_000.0, top_n=10, max_exposure=0.90)

    WALK_FORWARD_TRAIN_YEARS = 5
    WALK_FORWARD_STEP_YEARS = 1

    # Parallelism: ThreadPoolExecutor workers.
    # M4 MacBook Air (10-core): 4-6 is sweet spot.
    # Each RF uses n_jobs=2 inside (set in model.py), so total threads ≈ workers × 2.
    MAX_WORKERS = min(6, os.cpu_count() or 4)

    OUT_PLOTS = Path("outputs/plots")
    OUT_REPORT = Path("outputs/reports/performance_report.txt")
    WEBSITE_DIR = Path("website")

    # ──────────────────────────────────────────────────────────────────────────
    # ② DATA LOAD
    # ──────────────────────────────────────────────────────────────────────────
    if FORCE_REFRESH:
        shutil.rmtree("data", ignore_errors=True)
        print("Cache cleared (FORCE_REFRESH=True)")

    extra_tickers = [BENCHMARK, VIX_TICKER]
    cfg = DataConfig(
        tickers=tickers + extra_tickers,
        start=START_DATE,
        end=END_DATE,
        cache_dir=Path("data"),
        use_cache=not FORCE_REFRESH,
    )
    print(f"Loading data for {len(tickers)} tickers (cache={'ON' if not FORCE_REFRESH else 'OFF'}) ...")
    raw = load_data(cfg)

    spy_df = raw.pop(BENCHMARK, None)
    vix_df = raw.pop(VIX_TICKER, None)

    if spy_df is None:
        raise RuntimeError("Benchmark SPY failed to download.")

    if raw:
        latest_date = max(df.index[-1] for df in raw.values())
        print(f"Latest market data: {latest_date.date()}")

    # ──────────────────────────────────────────────────────────────────────────
    # ③ MARKET REGIME DETECTION
    # ──────────────────────────────────────────────────────────────────────────
    current_regime: str | None = None
    effective_reco_thresh = RECO_THRESH
    effective_portfolio_cfg = PORTFOLIO_CFG

    if USE_MARKET_REGIME and vix_df is not None:
        spy_px_regime = spy_df.get("Adj Close", spy_df["Close"])
        vix_series = vix_df["Close"]
        regime_series = compute_market_regime(spy_px_regime, vix_series)
        if not regime_series.empty:
            current_regime = str(regime_series.iloc[-1])
            print(f"Market regime: {current_regime.upper()}")

            if current_regime == "risk_off":
                effective_reco_thresh = RecommendationThresholds(
                    buy=max(RECO_THRESH.buy, 0.65),
                    hold_low=RECO_THRESH.hold_low,
                    hold_high=RECO_THRESH.hold_high,
                )
                effective_portfolio_cfg = PortfolioConfig(
                    initial_capital=PORTFOLIO_CFG.initial_capital,
                    top_n=max(1, PORTFOLIO_CFG.top_n // 2),
                    max_exposure=PORTFOLIO_CFG.max_exposure,
                )
            elif current_regime == "bull":
                effective_reco_thresh = RecommendationThresholds(
                    buy=min(RECO_THRESH.buy, 0.55),
                    hold_low=RECO_THRESH.hold_low,
                    hold_high=RECO_THRESH.hold_high,
                )

    # ──────────────────────────────────────────────────────────────────────────
    # ④ UNIVERSE FILTERING
    # ──────────────────────────────────────────────────────────────────────────
    raw = ensure_min_history(raw, min_days=1800)

    filtered: dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        if df.empty:
            continue
        price = float(df["Close"].iloc[-1])
        avg_vol = float(df["Volume"].rolling(30).mean().iloc[-1])
        if price > 3.0 and avg_vol > 500_000:
            filtered[ticker] = df

    raw = filtered
    print(f"Universe after liquidity filter: {len(raw)} tickers")

    if len(raw) < 2:
        raise RuntimeError("Not enough tickers after filtering.")

    # ──────────────────────────────────────────────────────────────────────────
    # ⑤ PARALLEL FEATURE ENGINEERING + MODEL TRAINING
    # ──────────────────────────────────────────────────────────────────────────
    feat_cols = feature_columns(FEATURE_CFG)
    spy_close = spy_df.get("Adj Close", spy_df["Close"])

    best_models: dict[str, TrainedModel] = {}
    latest_preds: list = []
    per_ticker_probs: dict[str, pd.Series] = {}
    rf_importances: pd.Series | None = None
    rf_importances_ticker: str | None = None

    print(f"\nTraining models in parallel (workers={MAX_WORKERS}) ...")

    futures_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for ticker, df in raw.items():
            fut = executor.submit(
                _train_single_ticker,
                ticker, df, spy_close, feat_cols, FEATURE_CFG,
                WALK_FORWARD_TRAIN_YEARS, WALK_FORWARD_STEP_YEARS,
            )
            futures_map[fut] = ticker

        for fut in as_completed(futures_map):
            ticker, best, probs, pred, importances = fut.result()
            if best is None:
                continue
            best_models[ticker] = best
            if probs is not None and not probs.empty:
                per_ticker_probs[ticker] = probs
            if pred is not None:
                latest_preds.append(pred)
            if rf_importances is None and importances is not None:
                rf_importances = importances
                rf_importances_ticker = ticker

    print(f"\nSuccessfully trained: {len(best_models)} tickers")

    # ──────────────────────────────────────────────────────────────────────────
    # ⑥ RECOMMENDATIONS (latest signals — display only, NOT fed into backtest)
    # ──────────────────────────────────────────────────────────────────────────
    # MATH FIX: recommendations are computed separately from the backtest.
    # Do NOT zero out historical probs based on today's signal — that is
    # look-ahead bias and was the primary cause of Sharpe instability.
    # The backtest uses only historical walk-forward OOS probabilities.

    print("\nLatest predictions + recommendations (sorted by P(Up)):")
    latest_preds_sorted = sorted(latest_preds, key=lambda p: p.prob_up, reverse=True)

    predictions_for_output: list[dict] = []

    for p in latest_preds_sorted:
        rec = recommendation_from_probability(p.prob_up, effective_reco_thresh)

        expected_return: float | None = None
        target_price_display: float | None = None

        # p.target_price stores the predicted 5-day RETURN (e.g. 0.05 = 5%), not a dollar price.
        # Correct naming would be `predicted_return`; kept as-is for backward compat.
        if p.target_price is not None and p.current_price is not None:
            expected_return = float(p.target_price)
            target_price_display = p.current_price * (1.0 + expected_return)

            # Override BUY → HOLD if expected return is negligible (< 2%)
            if rec == "BUY" and expected_return <= 0.02:
                rec = "HOLD"

        final_rec = rec

        tag = ""
        if final_rec == "BUY" and target_price_display is not None:
            tag = (
                f" | ${p.current_price:.2f} → ${target_price_display:.2f}"
                f" ({expected_return * 100:+.2f}%)"
            )
        print(f"  {p.ticker:<6} p={p.prob_up:.3f} => {final_rec}{tag}")

        predictions_for_output.append(
            {
                "ticker": p.ticker,
                "prob_up": round(p.prob_up, 4),
                "price": p.current_price,
                "target_return": round(expected_return, 4) if expected_return is not None else None,
                "signal": final_rec,
            }
        )

  
    print("\nRunning portfolio backtest ...")

    per_ticker_probs = {
        t: p for t, p in per_ticker_probs.items()
        if p is not None and len(p) > 100
    }

    price_by_ticker: dict[str, pd.Series] = {
        t: (df.get("Adj Close", df["Close"]))
        for t, df in raw.items()
        if t in per_ticker_probs
    }

    from data_loader import align_on_common_dates

    per_ticker_probs = align_on_common_dates(per_ticker_probs)
    price_by_ticker = align_on_common_dates(price_by_ticker)

    print(f"Tickers used: {len(per_ticker_probs)}")
    print(f"Backtest length: {len(next(iter(per_ticker_probs.values())))}")

    bt = run_portfolio_backtest(
        price_by_ticker=price_by_ticker,
        prob_by_ticker=per_ticker_probs,
        cfg=effective_portfolio_cfg,
    )

    # Benchmark (SPY buy & hold) aligned to the same dates as backtest
    spy_px = spy_df.get("Adj Close", spy_df["Close"])
    spy_px = spy_px.reindex(bt.equity_curve.index).ffill().dropna()
    strategy_eq = bt.equity_curve.reindex(spy_px.index).dropna()
    benchmark_eq = PORTFOLIO_CFG.initial_capital * (spy_px / spy_px.iloc[0])

    report = summarize_performance(strategy_eq, benchmark_eq, bt.trades)

    print(
        f"\nPerformance summary:"
        f"\n  Total return   : {report.cumulative_return:.2%}"
        f"\n  Ann. return    : {report.annualized_return:.2%}"
        f"\n  Sharpe ratio   : {report.sharpe_ratio:.3f}"
        f"\n  Max drawdown   : {report.max_drawdown:.2%}"
        f"\n  SPY B&H return : {report.buy_and_hold_return:.2%}"
        f"\n  Win rate       : {report.win_rate:.2%}"
        f"\n  # Trades       : {report.number_of_trades}"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # ⑧ SAVE OUTPUTS
    # ──────────────────────────────────────────────────────────────────────────
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("w", encoding="utf-8") as f:
        f.write("Quant Trading System — Performance Report\n")
        f.write("==========================================\n\n")
        f.write(f"Run time       : {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tickers        : {', '.join(sorted(raw.keys()))}\n")
        f.write(f"Benchmark      : {BENCHMARK}\n")
        f.write(f"Market regime  : {current_regime or 'N/A'}\n\n")
        f.write(f"Total return   : {report.cumulative_return:.2%}\n")
        f.write(f"Ann. return    : {report.annualized_return:.2%}\n")
        f.write(f"Sharpe ratio   : {report.sharpe_ratio:.3f}\n")
        f.write(f"Max drawdown   : {report.max_drawdown:.2%}\n")
        f.write(f"SPY B&H return : {report.buy_and_hold_return:.2%}\n")
        f.write(f"Win rate       : {report.win_rate:.2%}\n")
        f.write(f"# Trades       : {report.number_of_trades}\n")

    # Plots
    plot_equity_curve(strategy_eq, OUT_PLOTS / "equity_curve.png")
    plot_strategy_vs_benchmark(strategy_eq, benchmark_eq, OUT_PLOTS / "strategy_vs_spy.png")
    plot_drawdown(bt.drawdown.reindex(strategy_eq.index), OUT_PLOTS / "drawdown.png")
    if rf_importances is not None:
        plot_feature_importance(
            rf_importances,
            OUT_PLOTS / "feature_importance_random_forest.png",
            title=f"RF Feature Importance ({rf_importances_ticker})",
        )

    # Website output
    charts_dir = WEBSITE_DIR / "charts"
    data_dir = WEBSITE_DIR / "data"
    charts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["equity_curve.png", "strategy_vs_spy.png", "drawdown.png",
                  "feature_importance_random_forest.png"]:
        src = OUT_PLOTS / fname
        if src.exists():
            shutil.copy(src, charts_dir / fname)

    with open(data_dir / "predictions.json", "w") as f:
        json.dump(predictions_for_output, f, indent=2)

    # ── HTML dashboard (self-contained, images embedded as base64)
    generate_html_dashboard(
        predictions=predictions_for_output,
        report=report,
        plots_dir=OUT_PLOTS,
        out_path=WEBSITE_DIR / "index.html",
        regime=current_regime,
        run_timestamp=run_start.strftime("%Y-%m-%d %H:%M"),
        n_tickers_trained=len(best_models),
        backtest_start=str(strategy_eq.index[0].date()) if len(strategy_eq) else "N/A",
        backtest_end=str(strategy_eq.index[-1].date()) if len(strategy_eq) else "N/A",
    )

    elapsed = (datetime.datetime.now() - run_start).total_seconds() / 60
    print(f"\nDone in {elapsed:.1f} min")
    print(f"  Report  : {OUT_REPORT}")
    print(f"  Plots   : {OUT_PLOTS}/")
    print(f"  Website : {WEBSITE_DIR}/index.html")


if __name__ == "__main__":
    main()
