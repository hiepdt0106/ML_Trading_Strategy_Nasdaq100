"""
src/features/price.py
─────────────────────
Nhóm 1 — Giá, Momentum, Technical Indicators, Volume (Chương 3.4)

Features (30 cột):
─────────────────────────────────────────────────────────────
  Returns:          ret_1d, ret_5d, ret_10d, ret_21d            (4)
  Momentum:         mom_63d                                      (1)
  Mean-reversion:   rsi_14d, position_in_range_20d                (2)
  Lagged returns:   ret_1d_lag1…lag5                              (5)
  MACD:             macd_signal, macd_hist                        (2)
  Bollinger:        bb_pctb_20d                                   (1)
  Stochastic:       stoch_k_14d                                   (1)
  ADX:              adx_14d                                       (1)
  SMA signals:      price_sma20, price_sma50, sma_cross_5_20     (3)
  Trend:            price_sma200, sma_cross_20_50,                (5)
                    mom_accel_5d, trend_strength_21d, mean_rev_5d
  Volume:           vol_change_1d, vol_change_5d,                 (4)
                    vol_ma_ratio_20d, dollar_volume_21d
  Abnormal vol:     abnormal_volume                               (1)
─────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import log_return as _log_return

log = logging.getLogger(__name__)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return signal_line, histogram


def _bollinger_pctb(close: pd.Series, window: int = 20, n_std: float = 2.0):
    sma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return pctb


def _stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    return k


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Raw directional moves
    up_move = high - prev_high
    down_move = prev_low - low

    # FIX L: Compare raw values BEFORE zeroing out
    # +DM: up_move > down_move AND up_move > 0
    plus_dm = pd.Series(0.0, index=high.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move

    # -DM: down_move > up_move AND down_move > 0
    minus_dm = pd.Series(0.0, index=high.index)
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    atr = tr.ewm(alpha=1 / window, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, min_periods=window).mean()
    return adx


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """30 features: giá, momentum, technicals, trend, volume."""
    log.info("Nhóm 1 — Giá, Momentum & Technical Indicators ...")
    df = df.copy()

    def _per_ticker(grp):
        c = grp["adj_close"]
        h = grp["adj_high"]
        l = grp["adj_low"]
        v = grp["adj_volume"]

        # ── Returns (4) ──
        for n in [1, 5, 10, 21]:
            grp[f"ret_{n}d"] = _log_return(c, n)

        # ── Momentum dài (1) ──
        grp["mom_63d"] = _log_return(c, 63)

        # ── Mean-reversion (2) ──
        grp["rsi_14d"] = _rsi(c, window=14)
        high_20 = h.rolling(20, min_periods=15).max()
        low_20 = l.rolling(20, min_periods=15).min()
        rng = high_20 - low_20
        # Vị trí giá trong range [0=low, 1=high]
        grp["position_in_range_20d"] = (c - low_20) / rng.replace(0, np.nan)

        # ── Lagged returns (5) ──
        ret_1d = grp["ret_1d"]
        for lag in range(1, 6):
            grp[f"ret_1d_lag{lag}"] = ret_1d.shift(lag)

        # ── MACD (2) ──
        macd_sig, macd_hist = _macd(c)
        grp["macd_signal"] = macd_sig / c.replace(0, np.nan)
        grp["macd_hist"] = macd_hist / c.replace(0, np.nan)

        # ── Bollinger %B (1) ──
        grp["bb_pctb_20d"] = _bollinger_pctb(c, window=20)

        # ── Stochastic %K (1) ──
        grp["stoch_k_14d"] = _stochastic_k(h, l, c, window=14)

        # ── ADX (1) ──
        grp["adx_14d"] = _adx(h, l, c, window=14)

        # ── SMA signals (3) ──
        sma20 = c.rolling(20, min_periods=15).mean()
        sma50 = c.rolling(50, min_periods=40).mean()
        sma5 = c.rolling(5, min_periods=3).mean()
        grp["price_sma20"] = c / sma20.replace(0, np.nan) - 1
        grp["price_sma50"] = c / sma50.replace(0, np.nan) - 1
        grp["sma_cross_5_20"] = sma5 / sma20.replace(0, np.nan) - 1

        # ── Trend features (5) — giúp XGB detect trend/reversal ──
        sma200 = c.rolling(200, min_periods=100).mean()
        # Vị trí giá so với SMA200 (trend filter dài hạn)
        grp["price_sma200"] = c / sma200.replace(0, np.nan) - 1
        # SMA20 vs SMA50 cross (trung hạn trend)
        grp["sma_cross_20_50"] = sma20 / sma50.replace(0, np.nan) - 1
        # Momentum acceleration: ret_5d hiện tại vs ret_5d 5 ngày trước
        ret_5d = grp.get("ret_5d", _log_return(c, 5))
        grp["mom_accel_5d"] = ret_5d - ret_5d.shift(5)
        # Trend strength: ret_21d / vol_21d (Sharpe-like per stock)
        ret_21d = grp.get("ret_21d", _log_return(c, 21))
        vol_21d_local = grp.get("ret_1d", _log_return(c, 1)).rolling(21, min_periods=10).std()
        grp["trend_strength_21d"] = ret_21d / vol_21d_local.replace(0, np.nan)
        # Mean reversion signal: distance from 5d mean (short-term overshoot)
        grp["mean_rev_5d"] = c / sma5.replace(0, np.nan) - 1

        # ── Volume (4) ──
        grp["vol_change_1d"] = np.log(v / v.shift(1))
        grp["vol_change_5d"] = np.log(v / v.shift(5))
        vol_ma_20 = v.rolling(20, min_periods=10).mean()
        grp["vol_ma_ratio_20d"] = v / vol_ma_20.replace(0, np.nan)
        dollar = np.log(c * v + 1)
        grp["dollar_volume_21d"] = dollar.rolling(21, min_periods=10).mean()

        # ── Abnormal volume (1) ──
        vol_mean = v.rolling(63, min_periods=20).mean()
        vol_std = v.rolling(63, min_periods=20).std()
        grp["abnormal_volume"] = (v - vol_mean) / vol_std.replace(0, np.nan)

        return grp

    df = df.groupby(level="ticker", group_keys=False).apply(_per_ticker)

    # NOTE: KHÔNG tạo _rank features ở đây!
    # cross_sectional_rank() trong train.py sẽ rank tất cả features.

    n_added = len([c for c in df.columns if c.startswith((
        "ret_", "mom_", "rsi_", "position_in",
        "macd_", "bb_", "stoch_", "adx_",
        "price_sma", "sma_cross",
        "trend_strength", "mean_rev", "mom_accel",
        "vol_change", "vol_ma_", "dollar_", "abnormal_",
    ))])
    log.info(f"  → {n_added} features")
    return df
