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
    shutil.rmtree("data", ignore_errors=True)  # 🔥 캐시 삭제

    print("Run time:", datetime.datetime.now())
    # --------------------
    # User-configurable settings
    # --------------------
    tickers = [

# 0–100 range (100)
"AMD","INTC","MU","QCOM","AMAT","LRCX","KLAC","ADI","NXPI","MCHP",
"SWKS","QRVO","TER","ENTG","ON","HPQ","DELL","HPE","WDC","STX",
"F","GM","T","VZ","CSCO","ORCL","IBM","EBAY","PYPL","SQ",
"PINS","SNAP","ETSY","LYFT","UBER","DASH","CHWY","CVNA","ROKU","SPOT",
"EA","TTWO","ATVI","PARA","WBD","NOK","ERIC","BIDU","JD","BABA",
"NIO","XPEV","LI","RIVN","LCID","QS","PLUG","RUN","ENPH","SEDG",
"FSLR","XOM","CVX","OXY","HAL","SLB","DVN","EOG","APA","FANG",
"MRO","NOV","CHK","AR","CNX","BTU","ARCH","CCL","RCL","NCLH",
"AAL","DAL","UAL","MGM","WYNN","LVS","MAR","HLT","YUM","CMG",
"KHC","MDLZ","GIS","K","HSY","CL","KMB","PG","MO","PM",

# 100–200 range (100)
"AAPL","MSFT","GOOGL","META","AMZN","NVDA","AVGO","ADBE","CRM","NOW",
"SNOW","DDOG","NET","CRWD","ZS","OKTA","MDB","PANW","TEAM","WDAY",
"HUBS","SHOP","DOCU","TTD","ANET","CDNS","SNPS","FTNT","WDAY","ZS",
"MA","V","AXP","DFS","COF","ALLY","USB","PNC","BK","TFC",
"GS","MS","BLK","SCHW","CME","ICE","SPGI","MCO","AON","MMC",
"UNH","LLY","TMO","ISRG","VRTX","ZTS","HCA","DHR","IDXX","DXCM",
"REGN","GILD","AMGN","BIIB","BMY","ABBV","PFE","MRK","ELV","CI",
"CAT","DE","ETN","PH","HON","LMT","RTX","NOC","GD","BA",
"UPS","FDX","UNP","CSX","NSC","CP","CNI","ODFL","JBHT","CHRW",

# 200–300 range (100)
"COST","HD","LOW","TGT","WMT","SBUX","MCD","NKE","ADP","PAYX",
"INTU","FIS","FISV","GPN","JKHY","CTSH","ACN","IBM","ORCL","SAP",
"LIN","APD","SHW","ECL","PPG","DD","DOW","LYB","IFF","EMN",
"CLX","CHD","EL","PG","KMB","CL","HSY","MKC","SJM","HRL",
"DEO","STZ","BF.B","TAP","SAM","YUM","CMG","DPZ","DRI","TXRH",
"ROST","TJX","BURL","ULTA","ORLY","AZO","AAP","BBY","DG","DLTR",
"KR","WBA","CVS","CI","ELV","ANTM","HUM","CNC","MOH","UHS",
"ICE","CBOE","NDAQ","CME","SPGI","MCO","MSCI","BLK","TROW","BEN",
"PNR","ITW","PH","ROK","EMR","ETN","DOV","SWK","IR","XYL"
]
  # add/remove tickers here
    BENCHMARK = "SPY"
    VIX_TICKER = "^VIX"

    START_DATE = "2013-01-01"  # 10+ years of daily data
    END_DATE = None  # or "YYYY-MM-DD"

    FEATURE_CFG = FeatureConfig()
    USE_FACTOR_MODEL = False  # Set True to use a single cross-sectional factor model.
    USE_MARKET_REGIME = False  # Set True to enable market regime detection.
    RECO_THRESH = RecommendationThresholds(buy=0.55, hold_low=0.40, hold_high=0.55)

    PORTFOLIO_CFG = PortfolioConfig(initial_capital=100_000.0, top_n=10, max_exposure=0.90)

    WALK_FORWARD_TRAIN_YEARS = 5
    WALK_FORWARD_STEP_YEARS = 1

    OUT_PLOTS = Path("outputs/plots")
    OUT_REPORT = Path("outputs/reports/performance_report.txt")

    # --------------------
    # Data load
    # --------------------
    extra_tickers = [BENCHMARK]
    if USE_MARKET_REGIME:
        extra_tickers.append(VIX_TICKER)
    cfg = DataConfig(
    tickers=tickers + extra_tickers,
    start=START_DATE,
    end=END_DATE,
    cache_dir=Path("data"),
    use_cache=False 
    )
    raw = load_data(cfg)
    spy_df = raw.pop(BENCHMARK, None)
    vix_df = raw.pop(VIX_TICKER, None) if USE_MARKET_REGIME else None

    if spy_df is None:
        raise RuntimeError("Benchmark SPY failed to download.")

    if raw:
        latest_date = max(df.index[-1] for df in raw.values())
        print("Latest market data:", latest_date)

    # --------------------
    # Optional market regime detection (SPY + VIX)
    # --------------------
    current_regime: str | None = None
    effective_reco_thresh = RECO_THRESH
    effective_portfolio_cfg = PORTFOLIO_CFG

    if USE_MARKET_REGIME and vix_df is not None:
        spy_px_for_regime = spy_df["Adj Close"] if "Adj Close" in spy_df.columns else spy_df["Close"]
        vix_series = vix_df["Close"]
        regime_series = compute_market_regime(spy_px_for_regime, vix_series)
        if not regime_series.empty:
            current_regime = str(regime_series.iloc[-1])
            print(f"Current market regime: {current_regime}")

            if current_regime == "risk_off":
                # More conservative: higher buy threshold, fewer positions.
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
                # Slightly more aggressive: allow lower buy threshold.
                effective_reco_thresh = RecommendationThresholds(
                    buy=min(RECO_THRESH.buy, 0.55),
                    hold_low=RECO_THRESH.hold_low,
                    hold_high=RECO_THRESH.hold_high,
                )

    raw = ensure_min_history(raw, min_days=800)  # ~3 years of trading days

    # --------------------
    # Universe filtering: remove illiquid / very low price names
    # --------------------
    filtered_raw: dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        if df.empty:
            continue
        price = float(df["Close"].iloc[-1])
        avg_volume = float(df["Volume"].rolling(30).mean().iloc[-1])
        if price > 3.0 and avg_volume > 500_000:
            filtered_raw[ticker] = df

    raw = filtered_raw

    if len(raw) < 2:
        raise RuntimeError("Not enough tickers with sufficient history after liquidity/price filtering.")

    # --------------------
    # Feature engineering + model training
    # --------------------
    feat_cols = feature_columns(FEATURE_CFG)
    feature_frames: dict[str, pd.DataFrame] = {}
    best_models = {}
    latest_preds = []
    per_ticker_probs: dict[str, pd.Series] = {}
    rf_importances = None
    rf_importances_ticker = None

    print("Training models (time-based split, auto-select best)...")
    # First, build feature frames for all tickers.
    for ticker, df in raw.items():
        benchmark_close = spy_df["Adj Close"] if "Adj Close" in spy_df.columns else spy_df["Close"]
        feats = add_features(df, FEATURE_CFG, benchmark_close=benchmark_close)
        feature_frames[ticker] = feats

    if USE_FACTOR_MODEL:
        # --------------------
        # Single cross-sectional factor model across all tickers
        # --------------------
        combined = []
        for ticker, feats in feature_frames.items():
            tmp = feats.copy()
            tmp["ticker_id"] = abs(hash(ticker)) % 1000
            combined.append(tmp)

        combined_df = pd.concat(combined).sort_index()

        feat_cols = feat_cols + ["ticker_id"]

        candidates = train_and_select_model(combined_df, feat_cols, test_size=0.2)
        best_global = select_best_model(candidates)

        # Refit best global model on all data for latest prediction output.
        fitted_full_global = clone(best_global.pipeline).fit(combined_df[feat_cols], combined_df["target"].values)
        best_full_global = best_global.__class__(
            name=best_global.name, pipeline=fitted_full_global, feature_names=best_global.feature_names, metrics=best_global.metrics
        )

        # Regression remains per-ticker; probabilities come from the global factor model.
        for ticker, feats in feature_frames.items():
            df = raw[ticker]

            reg_pipe = _make_random_forest_regressor()
            horizon = 5
            y_reg = (feats["Close"].shift(-horizon) - feats["Close"]) / feats["Close"]
            valid_idx = y_reg.notna()
            reg_pipe.fit(
                feats.loc[valid_idx, feat_cols],
                y_reg.loc[valid_idx]
            )

            # Use the latest available market date by reindexing features to the raw price index.
            feats_for_pred = feats.reindex(df.index).ffill().iloc[[-1]]
            pred = predict_latest(
                ticker,
                feats_for_pred,
                df,
                best_full_global,
                regressor=reg_pipe,
                compute_target=True,
            )
            latest_preds.append(pred)

            # Cross-sectional probabilities for backtest (no walk-forward in factor mode).
            per_ticker_probs[ticker] = predict_proba_series(feats, best_full_global)

        # Feature importance (only once, from global RF model if available).
        if rf_importances is None and best_global.name == "random_forest":
            clf = fitted_full_global.named_steps["clf"]
            rf_model = clf.estimator if hasattr(clf, "estimator") else clf
            if hasattr(rf_model, "feature_importances_"):
                rf_importances = pd.Series(rf_model.feature_importances_, index=feat_cols)
                rf_importances_ticker = "FACTOR_MODEL"
    else:
        # --------------------
        # Original per-ticker modelling and walk-forward probabilities
        # --------------------
        for ticker, feats in feature_frames.items():
            df = raw[ticker]

            candidates = train_and_select_model(feats, feat_cols, test_size=0.2)
            best = select_best_model(candidates)
            best_models[ticker] = best

            # Refit best model on all data for latest prediction output.
            fitted_full = clone(best.pipeline).fit(feats[feat_cols], feats["target"].values)
            # Train regression model for target price
            reg_pipe = _make_random_forest_regressor()

            # 5-day forward return regression target (percentage return).
            horizon = 5
            y_reg = (feats["Close"].shift(-horizon) - feats["Close"]) / feats["Close"]
            valid_idx = y_reg.notna()

            reg_pipe.fit(
                feats.loc[valid_idx, feat_cols],
                y_reg.loc[valid_idx]
            )
            best_full = best.__class__(
                name=best.name, pipeline=fitted_full, feature_names=best.feature_names, metrics=best.metrics
            )
            # Use the latest available market date by reindexing features to the raw price index.
            feats_for_pred = feats.reindex(df.index).ffill().iloc[[-1]]
            pred = predict_latest(
                ticker,
                feats_for_pred,
                df,
                best_full,
                regressor=reg_pipe,
                compute_target=True,
            )

            latest_preds.append(pred)

            # Walk-forward evaluation (summary) + walk-forward probabilities for backtest (test period only).
            folds = walk_forward_validate(
                feats,
                feat_cols,
                model_name=best.name,
                initial_train_years=WALK_FORWARD_TRAIN_YEARS,
                test_years=1,
                step_years=1,
            )
            if folds:
                roc_mean = pd.Series([f.metrics.roc_auc for f in folds]).dropna().mean()
                acc_mean = pd.Series([f.metrics.accuracy for f in folds]).mean()
                print(
                    f"- {ticker}: best={best.name} | holdout ROC-AUC={best.metrics.roc_auc:.3f} acc={best.metrics.accuracy:.3f} | "
                    f"walk-forward mean ROC-AUC={roc_mean:.3f} acc={acc_mean:.3f} ({len(folds)} folds)"
                )
            else:
                print(
                    f"- {ticker}: best={best.name} | holdout ROC-AUC={best.metrics.roc_auc:.3f} acc={best.metrics.accuracy:.3f} | "
                    f"walk-forward: insufficient data for folds"
                )

            probs = walk_forward_predict_proba(
                feats,
                feat_cols,
                model_name=best.name,
                train_years=WALK_FORWARD_TRAIN_YEARS,
                step_years=WALK_FORWARD_STEP_YEARS,
)

# 🔥 여기서 바로 필터 적용
            latest_pred = next((x for x in latest_preds if x.ticker == ticker), None)

            if latest_pred and latest_pred.target_price is not None:
                if latest_pred.target_price < 0.02:
                    probs = probs * 0  # 👉 이 종목 매수 금지

# ✅ 그 다음에 넣는다
            per_ticker_probs[ticker] = probs

            if rf_importances is None and best.name == "random_forest":
                clf = fitted_full.named_steps["clf"]
                # Unwrap calibrated classifiers (e.g., CalibratedClassifierCV) to access the underlying
                # RandomForestClassifier's feature_importances_ attribute.
                rf_model = clf.estimator if hasattr(clf, "estimator") else clf
                if hasattr(rf_model, "feature_importances_"):
                    rf_importances = pd.Series(rf_model.feature_importances_, index=feat_cols)
                    rf_importances_ticker = ticker

    # --------------------
    # Recommendations (latest)
    # --------------------
    print("\nLatest predictions + recommendations:")
    latest_preds_sorted = sorted(latest_preds, key=lambda p: p.prob_up, reverse=True)

    for p in latest_preds_sorted:

        rec = recommendation_from_probability(p.prob_up, effective_reco_thresh)

    # 🔥 핵심 필터 추가
        if p.target_price is not None:
            if p.target_price < 0.02:   # 기대수익 2% 미만 제거
                rec = "HOLD"

        if rec == "BUY" and p.target_price is not None and p.current_price is not None:

            # Regressor predicts forward return directly; derive target price.
            expected_return = float(p.target_price)
            target_price = p.current_price * (1.0 + expected_return)

            if expected_return <= 0.01:
                rec = "HOLD"

            if rec == "BUY":
                print(
                    f"- {p.ticker} asof={p.asof.date()} prob_up={p.prob_up:.3f} pred={p.pred_label} => BUY | "
                    f"${p.current_price:.2f} → ${target_price:.2f} ({expected_return*100:+.2f}%)"
                )
            else:
                print(
                    f"- {p.ticker} asof={p.asof.date()} prob_up={p.prob_up:.3f} pred={p.pred_label} => HOLD"
                )

        else:
            print(
                f"- {p.ticker} asof={p.asof.date()} prob_up={p.prob_up:.3f} pred={p.pred_label} => {rec}"
            )

    # --------------------
    # Portfolio backtest on common test dates
    # --------------------
    print("\nRunning portfolio backtest (top-3 by predicted probability, daily rebalance)...")
    price_by_ticker = {}
    for ticker, df in raw.items():
        # Use Adj Close where available for valuation
        px = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        price_by_ticker[ticker] = px

   # =========================
# 🔥 단순 전략 (adaptive 제거)
# =========================
    bt = run_portfolio_backtest(
        price_by_ticker=price_by_ticker,
        prob_by_ticker=per_ticker_probs,
        cfg=effective_portfolio_cfg
    )

# =========================
# 5️⃣ 이후 기존 코드 그대로
# =========================
    spy_px = spy_df["Adj Close"] if "Adj Close" in spy_df.columns else spy_df["Close"]
    spy_px = spy_px.loc[bt.equity_curve.index].dropna()
    strategy_eq = bt.equity_curve.loc[spy_px.index]
    benchmark_eq = PORTFOLIO_CFG.initial_capital * (spy_px / spy_px.iloc[0])
    report = summarize_performance(strategy_eq, benchmark_eq, bt.trades)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("w", encoding="utf-8") as f:
        f.write("Quant Trading System - Performance Report\n")
        f.write("========================================\n\n")
        f.write(f"Tickers: {', '.join(sorted(raw.keys()))}\n")
        f.write(f"Benchmark: {BENCHMARK}\n\n")
        f.write(f"Total return: {report.cumulative_return:.2%}\n")
        f.write(f"Annualized return: {report.annualized_return:.2%}\n")
        f.write(f"Sharpe ratio: {report.sharpe_ratio:.3f}\n")
        f.write(f"Max drawdown: {report.max_drawdown:.2%}\n")
        f.write(f"SPY buy-and-hold return: {report.buy_and_hold_return:.2%}\n")
        f.write(f"Number of trades: {report.number_of_trades}\n")
        f.write(f"Win rate: {report.win_rate:.2%}\n")

    # --------------------
    # Visualizations
    # --------------------
    plot_equity_curve(strategy_eq, OUT_PLOTS / "equity_curve.png")
    plot_strategy_vs_benchmark(strategy_eq, benchmark_eq, OUT_PLOTS / "strategy_vs_spy.png")
    plot_drawdown(bt.drawdown.loc[strategy_eq.index], OUT_PLOTS / "drawdown.png")
    if rf_importances is not None:
        plot_feature_importance(
            rf_importances,
            OUT_PLOTS / "feature_importance_random_forest.png",
            title=f"Random Forest Feature Importance ({rf_importances_ticker})",
        )

    print("\nDone.")
    print(f"- Report: {OUT_REPORT}")
    print(f"- Plots: {OUT_PLOTS}/")

    # --------------------
# Copy plots to website
# --------------------
    chart_dir = Path("website/charts")
    chart_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(OUT_PLOTS / "equity_curve.png", chart_dir / "equity_curve.png")
    shutil.copy(OUT_PLOTS / "strategy_vs_spy.png", chart_dir / "strategy_vs_spy.png")
    shutil.copy(OUT_PLOTS / "drawdown.png", chart_dir / "drawdown.png")

    import json

    data = []

    for p in latest_preds_sorted:
        data.append({
            "ticker": p.ticker,
            "prob_up": round(p.prob_up,3),
            "price": p.current_price,
            "target_return": p.target_price
        })

    Path("website/data").mkdir(parents=True, exist_ok=True)

    with open("website/data/predictions.json","w") as f:
        json.dump(data,f,indent=2)

    # --------------------
    # Copy report to website
    # --------------------
    report_dir = Path("website/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(OUT_REPORT, report_dir / "performance_report.txt")

if __name__ == "__main__":
    main()
