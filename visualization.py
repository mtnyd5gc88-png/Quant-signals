from __future__ import annotations

import os
from pathlib import Path

# In sandboxed environments, the default matplotlib config path may be unwritable.
_default_cache = Path(__file__).resolve().parent / "outputs" / "mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_default_cache))

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values, label="Strategy")
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_strategy_vs_benchmark(equity: pd.Series, benchmark: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values / equity.iloc[0], label="Strategy (normalized)")
    plt.plot(benchmark.index, benchmark.values / benchmark.iloc[0], label="SPY Buy&Hold (normalized)")
    plt.title("Strategy vs SPY Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_drawdown(drawdown: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3.5))
    plt.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    plt.plot(drawdown.index, drawdown.values, color="red", linewidth=1.0)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(
    importances: pd.Series,
    out_path: Path,
    title: str = "Random Forest Feature Importance",
    top_k: int = 15,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imp = importances.sort_values(ascending=False).head(top_k)[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(imp.index, imp.values)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

