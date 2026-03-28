"""
src/features/regime_features.py
───────────────────────────────
Nhóm 5 — Regime as Feature (thay vì regime as overlay)

Tạo P(high_vol) từ HMM rồi đưa vào feature set:
  - p_high_vol          : xác suất high-volatility regime [0, 1]
  - p_high_x_mom_63d    : interaction với momentum
  - p_high_x_vol_21d    : interaction với realized vol
  - p_high_x_resid_ret  : interaction với residual return (nếu có)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def add_regime_features(
    df: pd.DataFrame,
    lookback: int = 504,
    refit_freq: int = 63,
) -> pd.DataFrame:
    """Tạo regime features từ HMM, walk-forward style."""
    log.info("Nhóm 5 — Regime as Feature (HMM) ...")
    df = df.copy()

    if "vxn" not in df.columns or "vix" not in df.columns:
        raise ValueError("Cần cột 'vxn' và 'vix'")

    try:
        from src.regime.hmm import RegimeHMM
    except ImportError:
        log.warning("hmmlearn không available → fill 0.5")
        df["p_high_vol"] = 0.5
        return df

    # ── VXN/VIX series (date level) ──
    first_ticker = df.index.get_level_values("ticker").unique()[0]
    macro = df.xs(first_ticker, level="ticker")[["vxn", "vix"]].copy()
    all_dates = macro.index.sort_values()
    n_dates = len(all_dates)

    # ── Walk-forward HMM ──
    p_high_values = np.full(n_dates, 0.5)
    hmm = RegimeHMM(n_states=2, min_obs=lookback)

    refit_indices = list(range(lookback, n_dates, refit_freq))
    if not refit_indices or refit_indices[-1] != n_dates - 1:
        refit_indices.append(n_dates)

    for r, refit_idx in enumerate(refit_indices[:-1]):
        vxn_hist = macro["vxn"].iloc[:refit_idx]
        vix_hist = macro["vix"].iloc[:refit_idx]
        hmm.fit(vxn_hist, vix_hist)

        next_refit = refit_indices[r + 1]
        for i in range(refit_idx, min(next_refit, n_dates)):
            p_high_values[i] = hmm.predict_proba(
                macro["vxn"].iloc[:i], macro["vix"].iloc[:i]
            )

    # ── Map vào panel bằng merge (tránh .map() date mismatch) ──
    p_high_df = pd.DataFrame({
        "date": pd.to_datetime(all_dates).normalize(),
        "p_high_vol": p_high_values,
    }).set_index("date")

    flat = df.reset_index()
    flat["date"] = pd.to_datetime(flat["date"]).dt.normalize()

    if "p_high_vol" in flat.columns:
        flat = flat.drop(columns=["p_high_vol"])

    flat = flat.merge(p_high_df, on="date", how="left")
    flat["p_high_vol"] = flat["p_high_vol"].fillna(0.5)
    df = flat.set_index(["date", "ticker"]).sort_index()

    n_features = 1
    log.info(f"  p_high_vol: mean={df['p_high_vol'].mean():.3f}, "
             f"std={df['p_high_vol'].std():.3f}, "
             f"NaN={df['p_high_vol'].isna().sum()}")

    # ── Interaction terms (fillna 0 cho warmup) ──
    interactions = {
        "p_high_x_mom_63d":   "mom_63d",
        "p_high_x_vol_21d":   "vol_21d",
        "p_high_x_resid_ret": "resid_ret_21d",
    }
    for new_col, base_col in interactions.items():
        if base_col in df.columns:
            df[new_col] = df["p_high_vol"] * df[base_col].fillna(0)
            n_features += 1
            log.info(f"  {new_col} (base NaN → 0)")
        else:
            log.info(f"  {new_col} skipped ({base_col} not found)")

    log.info(f"  → {n_features} regime features")
    return df