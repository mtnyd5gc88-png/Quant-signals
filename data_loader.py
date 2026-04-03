from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import requests
import yfinance as yf

SESSION = requests.Session()


@dataclass(frozen=True)
class DataConfig:
    tickers:   list[str]
    start:     str                    # "YYYY-MM-DD"
    end:       Optional[str] = None   # "YYYY-MM-DD" or None → today
    cache_dir: Path = Path("data")
    use_cache: bool = True


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # flatten multi-index columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing  = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in ["Open", "High", "Low", "Close", "Adj Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df[["Open", "High", "Low", "Close", "Adj Close"]] = (
        df[["Open", "High", "Low", "Close", "Adj Close"]].ffill()
    )
    df["Volume"] = df["Volume"].fillna(0)
    df = df.dropna()
    return df


def download_ticker_history(
    ticker:      str,
    start:       str,
    end:         Optional[str] = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    import random
    import time

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
            print(f"[WARN] attempt {attempt+1} for {ticker}: {e}")
        time.sleep(2 + random.random() * 2)

    raise ValueError(f"No data returned for {ticker} after 5 attempts")


def load_data(config: DataConfig) -> Dict[str, pd.DataFrame]:
    import random
    import time

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, pd.DataFrame] = {}

    for t in config.tickers:
        time.sleep(0.3 + random.random() * 0.5)
        try:
            cache_path = (
                config.cache_dir
                / f"{t}_{config.start}_{config.end or 'today'}.csv"
            )
            if config.use_cache and cache_path.exists():
                df = pd.read_csv(
                    cache_path, parse_dates=["Date"], index_col="Date"
                )
                df = _standardize_ohlcv(df)
            else:
                df = download_ticker_history(t, config.start, config.end)
                df.to_csv(cache_path, index_label="Date")

            data[t] = df

        except Exception as e:
            print(f"[WARN] Failed to load {t}: {e}")

    return data


def align_on_common_dates(
    data: Dict[str, Union[pd.DataFrame, pd.Series]],
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    DataFrame과 Series 모두 지원.
    FIX:
      - 빈 dict 즉시 반환
      - common index가 비면 경고 + 원본 반환 (silent 빈 데이터 방지)
      - Series.loc[common] → Series 반환 보장
    """
    if not data:
        return {}

    common: Optional[pd.DatetimeIndex] = None
    for obj in data.values():
        idx    = obj.index
        common = idx if common is None else common.intersection(idx)

    # FIX: common이 비면 조용히 빈 데이터 반환하던 버그 제거
    if common is None or len(common) == 0:
        print("[WARN] align_on_common_dates: no common dates found — returning originals")
        return data

    out: Dict[str, Union[pd.DataFrame, pd.Series]] = {}
    for t, obj in data.items():
        aligned = obj.loc[common]
        # Series.copy() → Series, DataFrame.copy() → DataFrame 보장
        out[t] = aligned.copy()

    return out


def ensure_min_history(
    data: Dict[str, pd.DataFrame], min_days: int
) -> Dict[str, pd.DataFrame]:
    """min_days 미만 행 보유 티커 제거."""
    return {t: df for t, df in data.items() if len(df) >= min_days}
