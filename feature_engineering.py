from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    rsi_window: int = 14
    ma_windows: tuple[int, int, int] = (5, 20, 50)
    momentum_window: int = 10
    volatility_window: int = 20
    bollinger_window: int = 20
    bollinger_k: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


def _price_series(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        return df["Adj Close"]
    return df["Close"]


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_features(
    df: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
    *,
    benchmark_close: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Return a dataframe with engineered features and binary target:
      target = 1 if next-day return > 0 else 0
    """
    out = df.copy()
    close = _price_series(out)

    # Log return
    out["log_return"] = np.log(close / close.shift(1))

    # Moving averages
    ma5, ma20, ma50 = cfg.ma_windows
    out[f"ma_{ma5}"] = close.rolling(ma5).mean()
    out[f"ma_{ma20}"] = close.rolling(ma20).mean()
    out[f"ma_{ma50}"] = close.rolling(ma50).mean()
    out["ma20_ma50_ratio"] = out[f"ma_{ma20}"] / out[f"ma_{ma50}"]

    # Momentum (price difference)
    out[f"momentum_{cfg.momentum_window}"] = close - close.shift(cfg.momentum_window)

    # Volatility (rolling std of returns)
    out[f"vol_{cfg.volatility_window}"] = out["log_return"].rolling(cfg.volatility_window).std()

    # Volume change
    out["volume_pct_change"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    # RSI
    out[f"rsi_{cfg.rsi_window}"] = _rsi(close, cfg.rsi_window)

    # MACD
    macd_line = _ema(close, cfg.macd_fast) - _ema(close, cfg.macd_slow)
    macd_signal = _ema(macd_line, cfg.macd_signal)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    # Bollinger Bands
    mid = close.rolling(cfg.bollinger_window).mean()
    std = close.rolling(cfg.bollinger_window).std()
    out["bb_mid"] = mid
    out["bb_upper"] = mid + cfg.bollinger_k * std
    out["bb_lower"] = mid - cfg.bollinger_k * std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / mid

    # Additional alpha features (purely backward-looking).
    # Level-based momentum and volatility
    out["momentum_20"] = close.pct_change(20)
    out["momentum_5"] = close.pct_change(5)
    returns = close.pct_change()
    out["volatility_20"] = returns.rolling(20).std()
    out["volume_ratio"] = out["Volume"] / out["Volume"].rolling(20).mean()
    ma50 = close.rolling(50).mean()
    out["distance_from_ma50"] = (close - ma50) / ma50
    if benchmark_close is not None and not benchmark_close.empty:
        # Relative strength vs market (SPY) over 20 trading days.
        stock_return_20d = close.pct_change(20)
        spy_aligned = benchmark_close.reindex(close.index)
        spy_return_20d = spy_aligned.pct_change(20)
        out["relative_strength_20"] = stock_return_20d - spy_return_20d
    else:
        out["relative_strength_20"] = np.nan

    # Cross-sectional style ranking features (within each ticker's history).
    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)
    vol_20 = returns.rolling(20).std()
    out["ret_5"] = ret_5
    out["ret_20"] = ret_20
    out["volatility_20_cross"] = vol_20
    out["ret_rank_5"] = ret_5.rank(pct=True)
    out["ret_rank_20"] = ret_20.rank(pct=True)
    out["vol_rank_20"] = vol_20.rank(pct=True)

    # 5-day forward classification target (aligned horizon).
    horizon = 5
    future_close = close.shift(-horizon)
    out["target"] = np.where(future_close.notna(), (future_close > close).astype(int), np.nan)

    # Clean: drop rows with any NaNs in feature columns or target.
    feature_cols = feature_columns(cfg)
    out = out.dropna(subset=feature_cols + ["target"])
    out["target"] = out["target"].astype(int)

    return out


def feature_columns(cfg: FeatureConfig = FeatureConfig()) -> list[str]:
    ma5, ma20, ma50 = cfg.ma_windows
    return [
        "log_return",
        f"ma_{ma5}",
        f"ma_{ma20}",
        f"ma_{ma50}",
        "ma20_ma50_ratio",
        f"momentum_{cfg.momentum_window}",
        f"vol_{cfg.volatility_window}",
        "volume_pct_change",
        f"rsi_{cfg.rsi_window}",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
        "momentum_20",
        "momentum_5",
        "volatility_20",
        "volume_ratio",
        "distance_from_ma50",
        "relative_strength_20",
        "ret_5",
        "ret_20",
        "volatility_20_cross",
        "ret_rank_5",
        "ret_rank_20",
        "vol_rank_20",
    ]

