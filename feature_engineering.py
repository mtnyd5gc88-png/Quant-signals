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
    # When True, replaces raw-price features with stationary equivalents so
    # StandardScaler z-scores stay bounded across walk-forward windows.
    stationary_scale_features: bool = True


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
      target = 1 if close rises over next 5 days else 0

    When cfg.stationary_scale_features=True (default), raw-price features are
    replaced with stationary equivalents so StandardScaler z-scores remain
    bounded across all walk-forward windows regardless of price drift.
    """
    out = df.copy()
    close = _price_series(out)

    # Log return
    out["log_return"] = np.log(close / close.shift(1))

    # Moving average series (computed once; reused for ratios and distance_from_ma50)
    ma5_win, ma20_win, ma50_win = cfg.ma_windows
    ma5_ser  = close.rolling(ma5_win).mean()
    ma20_ser = close.rolling(ma20_win).mean()
    ma50_ser = close.rolling(ma50_win).mean()

    if cfg.stationary_scale_features:
        # Price-to-MA ratios: bounded near 1.0, stationary across years
        out["close_ma5_ratio"]  = close / ma5_ser
        out["close_ma20_ratio"] = close / ma20_ser
        out["close_ma50_ratio"] = close / ma50_ser
    else:
        # Raw MA levels: non-stationary, grow with price (legacy behavior)
        out[f"ma_{ma5_win}"]  = ma5_ser
        out[f"ma_{ma20_win}"] = ma20_ser
        out[f"ma_{ma50_win}"] = ma50_ser

    out["ma20_ma50_ratio"] = ma20_ser / ma50_ser   # already stationary in both modes

    # Momentum
    if cfg.stationary_scale_features:
        # Percentage change: stationary regardless of price level
        out["momentum_10_pct"] = close.pct_change(cfg.momentum_window)
    else:
        # Raw price difference: scales with price level (legacy behavior)
        out[f"momentum_{cfg.momentum_window}"] = close - close.shift(cfg.momentum_window)

    # Volatility (rolling std of log returns — stationary in both modes)
    out[f"vol_{cfg.volatility_window}"] = out["log_return"].rolling(cfg.volatility_window).std()

    # Volume change
    out["volume_pct_change"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    # RSI
    out[f"rsi_{cfg.rsi_window}"] = _rsi(close, cfg.rsi_window)

    # MACD
    macd_line = _ema(close, cfg.macd_fast) - _ema(close, cfg.macd_slow)
    macd_sig  = _ema(macd_line, cfg.macd_signal)
    if cfg.stationary_scale_features:
        # Divide by close to normalize EMA differences to a price-independent scale
        out["macd_pct"]        = macd_line / close
        out["macd_signal_pct"] = macd_sig  / close
        out["macd_hist_pct"]   = (macd_line - macd_sig) / close
    else:
        # Raw EMA differences: scale with price (legacy behavior)
        out["macd"]        = macd_line
        out["macd_signal"] = macd_sig
        out["macd_hist"]   = macd_line - macd_sig

    # Bollinger Bands (bb_width = 2*k*std/mid is already stationary)
    bb_mid = close.rolling(cfg.bollinger_window).mean()
    bb_std = close.rolling(cfg.bollinger_window).std()
    out["bb_upper"] = bb_mid + cfg.bollinger_k * bb_std
    out["bb_lower"] = bb_mid - cfg.bollinger_k * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_mid

    # Additional stationary features (unchanged in both modes)
    out["momentum_20"] = close.pct_change(20)
    out["momentum_5"]  = close.pct_change(5)
    returns = close.pct_change()
    out["volatility_20"] = returns.rolling(20).std()
    out["volume_ratio"]  = out["Volume"] / out["Volume"].rolling(20).mean()
    out["distance_from_ma50"] = (close - ma50_ser) / ma50_ser

    if benchmark_close is not None and not benchmark_close.empty:
        stock_return_20d = close.pct_change(20)
        spy_aligned      = benchmark_close.reindex(close.index)
        spy_return_20d   = spy_aligned.pct_change(20)
        out["relative_strength_20"] = stock_return_20d - spy_return_20d
    else:
        out["relative_strength_20"] = np.nan

    out["ret_5"]  = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)

    # 5-day forward classification target
    horizon      = 5
    future_close = close.shift(-horizon)
    out["target"] = np.where(future_close.notna(), (future_close > close).astype(int), np.nan)

    feature_cols = feature_columns(cfg)
    out = out.dropna(subset=feature_cols + ["target"])
    out["target"] = out["target"].astype(int)

    return out


def feature_columns(cfg: FeatureConfig = FeatureConfig()) -> list[str]:
    ma5_win, ma20_win, ma50_win = cfg.ma_windows
    if cfg.stationary_scale_features:
        ma_cols   = ["close_ma5_ratio", "close_ma20_ratio", "close_ma50_ratio"]
        mom_col   = "momentum_10_pct"
        macd_cols = ["macd_pct", "macd_signal_pct", "macd_hist_pct"]
    else:
        ma_cols   = [f"ma_{ma5_win}", f"ma_{ma20_win}", f"ma_{ma50_win}"]
        mom_col   = f"momentum_{cfg.momentum_window}"
        macd_cols = ["macd", "macd_signal", "macd_hist"]

    return [
        "log_return",
        *ma_cols,
        "ma20_ma50_ratio",
        mom_col,
        f"vol_{cfg.volatility_window}",
        "volume_pct_change",
        f"rsi_{cfg.rsi_window}",
        *macd_cols,
        "bb_width",
        "momentum_20",
        "momentum_5",
        "volatility_20",
        "volume_ratio",
        "distance_from_ma50",
        "relative_strength_20",
        "ret_5",
        "ret_20",
    ]

