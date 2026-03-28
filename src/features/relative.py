"""
src/features/relative.py
────────────────────────
Nhóm 4 — Relative / Residual / Liquidity features

Features (tối đa 11 cột):
─────────────────────────────────────────────────────────
  Residual:       rolling_beta_63d, resid_ret_21d, idio_vol_21d    (3)  [cần bench]
  Relative:       rel_strength_21d, rel_strength_63d                (2)  [cần bench]
  Liquidity:      amihud_illiq_21d, turnover_21d                    (2)
  Downside:       downside_vol_21d, downside_beta_63d, max_dd_21d   (3)  [downside_beta cần bench]
  Dispersion:     market_dispersion_21d                             (1)
─────────────────────────────────────────────────────────

KHÔNG dùng groupby().apply() — chỉ dùng transform/vectorized
để tránh pandas MultiIndex broadcast bug.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import log_return as _log_return

log = logging.getLogger(__name__)


def _rolling_per_ticker(series: pd.Series, window: int, min_periods: int, func: str = "mean") -> pd.Series:
    """Rolling per ticker dùng transform — safe cho MultiIndex."""
    def _apply(x):
        r = x.rolling(window, min_periods=min_periods)
        return getattr(r, func)()
    return series.groupby(level="ticker").transform(_apply)


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tối đa 11 features. Tất cả tính bằng vectorized + transform.
    """
    log.info("Nhóm 4 — Relative / Residual / Liquidity ...")
    df = df.copy()

    if "ret_1d" not in df.columns:
        raise ValueError("Cần cột 'ret_1d'. Chạy add_price_features() trước.")

    has_bench = "bench_close" in df.columns
    ret = df["ret_1d"]
    c = df["adj_close"]
    v = df["adj_volume"].fillna(0)

    # ══════════════════════════════════════════════════════════════
    # 1. LIQUIDITY (không cần benchmark)
    # ══════════════════════════════════════════════════════════════

    # Amihud illiquidity
    dollar_vol = c * v
    daily_illiq = ret.abs() / dollar_vol.replace(0, np.nan)
    df["amihud_illiq_21d"] = _rolling_per_ticker(daily_illiq, 21, 10, "mean")

    # Turnover proxy
    vol_avg = _rolling_per_ticker(v, 63, 20, "mean")
    vol_21 = _rolling_per_ticker(v, 21, 10, "mean")
    df["turnover_21d"] = vol_21 / vol_avg.replace(0, np.nan)

    log.info(f"  amihud NaN: {df['amihud_illiq_21d'].isna().mean():.1%}")
    log.info(f"  turnover NaN: {df['turnover_21d'].isna().mean():.1%}")

    # ══════════════════════════════════════════════════════════════
    # 2. DOWNSIDE (không cần benchmark)
    # ══════════════════════════════════════════════════════════════

    # Downside volatility
    neg_only = ret.where(ret < 0)
    df["downside_vol_21d"] = _rolling_per_ticker(neg_only, 21, 5, "std")

    # Max drawdown 21d
    rolling_max = _rolling_per_ticker(c, 21, 10, "max")
    df["max_dd_21d"] = (c - rolling_max) / rolling_max.replace(0, np.nan)

    # ══════════════════════════════════════════════════════════════
    # 3. DISPERSION (cross-sectional, không cần benchmark)
    # ══════════════════════════════════════════════════════════════

    daily_disp = df.groupby(level="date")["ret_1d"].std()
    disp_21d = daily_disp.rolling(21, min_periods=10).mean()
    disp_map = disp_21d.to_dict()
    df["market_dispersion_21d"] = df.index.get_level_values("date").map(disp_map)

    # ══════════════════════════════════════════════════════════════
    # 4. BENCHMARK-DEPENDENT (chỉ khi có bench_close)
    # ══════════════════════════════════════════════════════════════

    if has_bench:
        first_ticker = df.index.get_level_values("ticker").unique()[0]
        bench_close_s = df.xs(first_ticker, level="ticker")["bench_close"]
        bench_ret_1d = _log_return(bench_close_s, 1)
        bench_ret_21d = _log_return(bench_close_s, 21)
        bench_ret_63d = _log_return(bench_close_s, 63)

        # Map benchmark returns vào panel
        dates = df.index.get_level_values("date")
        b1d = dates.map(bench_ret_1d.to_dict()).astype(float)
        b21d = dates.map(bench_ret_21d.to_dict()).astype(float)
        b63d = dates.map(bench_ret_63d.to_dict()).astype(float)

        # Rolling beta per ticker: cov(stock, bench) / var(bench)
        # Tính bằng expanding cov/var per ticker
        # Cần loop per ticker vì rolling.cov cần 2 series aligned
        betas = []
        resid_rets = []
        idio_vols = []
        rel_21s = []
        rel_63s = []
        down_betas = []

        for tkr in df.index.get_level_values("ticker").unique():
            mask = df.index.get_level_values("ticker") == tkr
            tkr_ret = ret[mask]
            tkr_close = c[mask]
            tkr_dates = tkr_ret.index.get_level_values("date")

            # Align benchmark returns to ticker dates
            br1d = pd.Series(tkr_dates.map(bench_ret_1d.to_dict()).astype(float).values,
                             index=tkr_ret.index)
            br21d = pd.Series(tkr_dates.map(bench_ret_21d.to_dict()).astype(float).values,
                              index=tkr_ret.index)
            br63d = pd.Series(tkr_dates.map(bench_ret_63d.to_dict()).astype(float).values,
                              index=tkr_ret.index)

            # Rolling beta
            cov = tkr_ret.rolling(63, min_periods=32).cov(br1d)
            var = br1d.rolling(63, min_periods=32).var()
            beta = cov / var.replace(0, np.nan)
            betas.append(beta)

            # Residual return 21d
            ret_21d = df.loc[mask, "ret_21d"] if "ret_21d" in df.columns else _log_return(tkr_close, 21)
            resid = ret_21d - beta * br21d
            resid_rets.append(resid)

            # Idiosyncratic vol
            resid_daily = tkr_ret - beta * br1d
            idio = resid_daily.rolling(21, min_periods=10).std()
            idio_vols.append(idio)

            # Relative strength
            rel_21s.append(ret_21d - br21d)
            ret_63d = df.loc[mask, "mom_63d"] if "mom_63d" in df.columns else _log_return(tkr_close, 63)
            rel_63s.append(ret_63d - br63d)

            # Downside beta
            neg_mask_b = br1d < 0
            neg_stock = tkr_ret.where(neg_mask_b)
            neg_bench = br1d.where(neg_mask_b)
            cov_down = neg_stock.rolling(63, min_periods=20).cov(neg_bench)
            var_down = neg_bench.rolling(63, min_periods=20).var()
            down_betas.append(cov_down / var_down.replace(0, np.nan))

        df["rolling_beta_63d"] = pd.concat(betas)
        df["resid_ret_21d"] = pd.concat(resid_rets)
        df["idio_vol_21d"] = pd.concat(idio_vols)
        df["rel_strength_21d"] = pd.concat(rel_21s)
        df["rel_strength_63d"] = pd.concat(rel_63s)
        df["downside_beta_63d"] = pd.concat(down_betas)

        log.info("  benchmark features: OK (per-ticker loop, no groupby.apply)")
    else:
        log.warning("  bench_close không có → skip 6 benchmark-dependent features")

    # ── Log ──
    added = [col for col in df.columns if col.startswith((
        "rolling_beta", "resid_ret", "idio_vol",
        "rel_strength", "amihud", "turnover",
        "downside_vol", "downside_beta", "max_dd_21d",
        "market_disp",
    ))]
    for feat in added:
        nan_pct = df[feat].isna().mean()
        if nan_pct > 0.5:
            log.warning(f"  ⚠ {feat}: {nan_pct:.0%} NaN")
        elif nan_pct > 0.03:
            log.info(f"  {feat}: {nan_pct:.1%} NaN (warmup)")
    log.info(f"  → {len(added)} features")
    return df