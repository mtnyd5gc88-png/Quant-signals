## Quant Trading System (ML + Portfolio Backtest)

End-to-end Python project that:
- Downloads 10+ years of daily data from Yahoo Finance (`yfinance`)
- Builds technical features per stock
- Trains **Logistic Regression** and **Random Forest** classifiers to predict next-day direction
- Selects the best model per stock by ROC-AUC/accuracy (time-based split)
- Runs **walk-forward validation**
- Generates **Buy / Hold / Stay in Cash** recommendations
- Backtests a **top-3 equal-weight portfolio** with stop-loss / take-profit and max exposure
- Produces performance metrics + plots

### Project structure

`quant_trading_system/`
- `data_loader.py`
- `feature_engineering.py`
- `model.py`
- `prediction.py`
- `strategy.py`
- `portfolio.py`
- `risk_management.py`
- `backtest.py`
- `evaluation.py`
- `visualization.py`
- `main.py`

### Setup

Create a virtual environment and install dependencies:

```bash
cd quant_trading_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Outputs:
- Plots saved to `outputs/plots/`
- Report saved to `outputs/reports/performance_report.txt`

### Configuration

Edit values at the top of `main.py`:
- **Tickers** (easily add/remove)
- **Date range** (defaults to ~12 years to guarantee 10+ years usable after indicator warmup)
- **Capital**, thresholds, walk-forward window, etc.

### Notes / assumptions

- Uses adjusted close (`Adj Close`) for returns and indicators, and close as fallback.
- Signals are generated using features at day \(t\) and applied to the return from \(t\) to \(t+1\) (no lookahead).
- Stop-loss / take-profit checks use end-of-day close prices (daily bars).

