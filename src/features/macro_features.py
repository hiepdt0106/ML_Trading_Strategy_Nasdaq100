"""
src/features/macro_features.py
──────────────────────────────
Nhóm 3 — Biến động thị trường & zSpread (Chương 3.5.3)

Features (10 cột):
─────────────────────────────────────────────────────────
  VIX/VXN returns:  vix_ret_1d, vxn_ret_1d, vxn_ret_5d
  VXN regime:       vxn_zscore
  VXN dynamics:     vxn_ma5_ma21, vxn_accel
  Spread:           vix_vxn_spread
  zSpread:          zspread, zspread_ma5, zspread_change_5d
─────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import log_return as _log_return

log = logging.getLogger(__name__)


def add_macro_features(
    df: pd.DataFrame,
    vxn_window: int = 252,
) -> pd.DataFrame:
    """10 features thị trường + zSpread."""
    log.info("Nhóm 3 — Biến động thị trường & zSpread ...")
    df = df.copy()

    if "vix" not in df.columns or "vxn" not in df.columns:
        raise ValueError("Cần cột 'vix' và 'vxn'")

    # ── Macro features trên date level ──
    first_ticker = df.index.get_level_values("ticker").unique()[0]
    macro = df.xs(first_ticker, level="ticker")[["vix", "vxn"]].copy()

    # Returns
    macro["vix_ret_1d"] = _log_return(macro["vix"], 1)
    macro["vxn_ret_1d"] = _log_return(macro["vxn"], 1)
    macro["vxn_ret_5d"] = _log_return(macro["vxn"], 5)

    # VXN regime — zscore
    vxn_mean = macro["vxn"].rolling(vxn_window, min_periods=60).mean()
    vxn_std = macro["vxn"].rolling(vxn_window, min_periods=60).std()
    macro["vxn_zscore"] = (macro["vxn"] - vxn_mean) / vxn_std.replace(0, np.nan)

    # VXN dynamics
    vxn_ma5 = macro["vxn"].rolling(5, min_periods=3).mean()
    vxn_ma21 = macro["vxn"].rolling(21, min_periods=10).mean()
    macro["vxn_ma5_ma21"] = vxn_ma5 / vxn_ma21.replace(0, np.nan) - 1
    macro["vxn_accel"] = macro["vxn_ret_5d"] - macro["vxn_ret_5d"].shift(5)

    # Spread
    macro["vix_vxn_spread"] = macro["vxn"] - macro["vix"]

    # ── Merge macro → panel ──
    macro_features = [c for c in macro.columns if c not in ["vix", "vxn"]]
    panel_flat = df.reset_index()
    macro_reset = macro[macro_features].reset_index()
    panel_flat = panel_flat.merge(macro_reset, on="date", how="left")
    df = panel_flat.set_index(["date", "ticker"]).sort_index()

    # ── zSpread (per-ticker) ──
    if "vol_21d" in df.columns:
        vxn_daily = df["vxn"] / (100 * np.sqrt(252))
        df["zspread"] = df["vol_21d"] - vxn_daily
        df["zspread_ma5"] = df.groupby(level="ticker")["zspread"].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        df["zspread_change_5d"] = df.groupby(level="ticker")["zspread"].transform(
            lambda x: x - x.shift(5)
        )
    else:
        log.warning("Chưa có vol_21d → bỏ qua zSpread")

    added = [c for c in df.columns
             if c.startswith(("vix_ret", "vxn_ret", "vxn_z", "vxn_ma5",
                              "vxn_accel", "vix_vxn", "zspread"))]
    log.info(f"  → {len(added)} features")
    return df