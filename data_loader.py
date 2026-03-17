from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf
import requests

SESSION = requests.Session()


@dataclass(frozen=True)
class DataConfig:
    tickers: list[str]
    start: str  # "YYYY-MM-DD"
    end: Optional[str] = None  # "YYYY-MM-DD" or None for today
    cache_dir: Path = Path("data")
    use_cache: bool = True


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # flatten multi-index columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

   # yfinance sometimes returns no Adj Close
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Handle missing values: forward-fill prices, set volume missing to 0, then drop any residual NaNs.
    df[["Open", "High", "Low", "Close", "Adj Close"]] = df[
        ["Open", "High", "Low", "Close", "Adj Close"]
    ].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    df = df.dropna()

    return df


def download_ticker_history(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:

    import time
    import random

    df = None

    for attempt in range(5):

        try:

            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
                timeout=20,
            )

            if df is not None and not df.empty:
                return _standardize_ohlcv(df)

        except Exception as e:
            print(f"[WARN] download error {ticker}: {e}")

        sleep_time = 2 + random.random() * 2
        time.sleep(sleep_time)

    raise ValueError(f"No data returned for {ticker}")


def load_data(config: DataConfig) -> Dict[str, pd.DataFrame]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    data: Dict[str, pd.DataFrame] = {}
    import time
    import random

    for t in config.tickers:

        time.sleep(0.3 + random.random() * 0.5)

        try:

            cache_path = config.cache_dir / f"{t}_{config.start}_{config.end or 'today'}.csv"

            if config.use_cache and cache_path.exists():
                df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
                df = _standardize_ohlcv(df)
            else:
                df = download_ticker_history(t, config.start, config.end)
                df.to_csv(cache_path, index_label="Date")

            data[t] = df

        except Exception as e:

            print(f"[WARN] Failed to load {t}: {e}")
            continue

    return data


def align_on_common_dates(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align all tickers to the intersection of available dates."""
    if not data:
        return {}

    common = None
    for df in data.values():
        idx = df.index
        common = idx if common is None else common.intersection(idx)

    out: Dict[str, pd.DataFrame] = {}
    for t, df in data.items():
        out[t] = df.loc[common].copy()
    return out


def ensure_min_history(data: Dict[str, pd.DataFrame], min_days: int) -> Dict[str, pd.DataFrame]:
    """Drop tickers with insufficient rows after cleaning/alignment."""
    return {t: df for t, df in data.items() if len(df) >= min_days}

