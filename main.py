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

    BENCHMARK = "SPY"
    VIX_TICKER = "^VIX"

    START_DATE = "2013-01-01"  # 10+ years of daily data
    END_DATE = None  # or "YYYY-MM-DD"

    FEATURE_CFG = FeatureConfig()
    USE_FACTOR_MODEL = False
    USE_MARKET_REGIME = False

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

    raw = ensure_min_history(raw, min_days=800)

    # --------------------
    # Universe filtering
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
        raise RuntimeError("Not enough tickers with sufficient history.")

    # --------------------
    # Model training
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
            train_years=WALK_FORWARD_TRAIN_YEARS,
            step_years=WALK_FORWARD_STEP_YEARS,
        )

        per_ticker_probs[ticker] = probs

    # --------------------
    # Output JSON
    # --------------------
    import json

    data = []

    for p in latest_preds:
        rec = recommendation_from_probability(p.prob_up, RECO_THRESH)

        data.append({
            "ticker": p.ticker,
            "prob_up": round(p.prob_up,3),
            "price": p.current_price,
            "target_return": p.target_price,
            "signal": rec
        })

    Path("website/data").mkdir(parents=True, exist_ok=True)

    with open("website/data/predictions.json","w") as f:
        json.dump(data,f,indent=2)


if __name__ == "__main__":
    main()
