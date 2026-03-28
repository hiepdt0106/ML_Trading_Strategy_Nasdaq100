"""
src/labeling/triple_barrier.py
──────────────────────────────
Triple Barrier Labeling — López de Prado (2018)

Triple Barrier:
  - Vertical barrier (H): giới hạn thời gian tối đa giữ vị thế
  - Upper barrier (PT)  : chốt lời khi return ≥  +pt_sl_mult × daily_vol
  - Lower barrier (SL)  : cắt lỗ khi return   ≤  −pt_sl_mult × daily_vol

Labels (binary):
  - 1 : chạm PT trước, hoặc hết H mà return > 0
  - 0 : chạm SL trước, hoặc hết H mà return ≤ 0

Anti-leakage:
  - daily_vol tính từ dữ liệu quá khứ (rolling backward)
  - Label scan forward nhưng KHÔNG dùng làm feature
  - t1 (thời điểm chạm barrier) được lưu để tính embargo
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def label(
    df: pd.DataFrame,
    horizon: int = 10,
    pt_sl_mult: float = 2.0,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Gán nhãn Triple Barrier.

    Parameters
    ----------
    df         : MultiIndex (date, ticker), cần cột 'adj_close'
    horizon    : H — số ngày vertical barrier
    pt_sl_mult : barrier = pt_sl_mult × daily_vol
    vol_window : rolling window tính daily_vol

    Returns
    -------
    DataFrame thêm cột: daily_vol, tb_label, tb_barrier, tb_return, t1, holding_td
    """
    log.info(
        f"Triple Barrier: H={horizon}, "
        f"pt_sl_mult={pt_sl_mult}, vol_window={vol_window}"
    )
    df = df.copy()

    def _label_ticker(grp: pd.DataFrame) -> pd.DataFrame:
        close = grp["adj_close"].to_numpy(dtype=float)
        open_ = grp["adj_open"].to_numpy(dtype=float) if "adj_open" in grp.columns else close.copy()
        high = grp["adj_high"].to_numpy(dtype=float) if "adj_high" in grp.columns else close.copy()
        low = grp["adj_low"].to_numpy(dtype=float) if "adj_low" in grp.columns else close.copy()
        dates = grp.index.get_level_values("date")
        n = len(close)

        log_ret = np.log(close[1:] / close[:-1])
        vol = pd.Series(log_ret).rolling(
            vol_window, min_periods=max(vol_window // 2, 5)
        ).std()
        vol = np.concatenate([[np.nan], vol.values])

        labels = np.full(n, np.nan)
        barriers = np.full(n, "", dtype=object)
        returns = np.full(n, np.nan)
        t1_dates = pd.array([pd.NaT] * n, dtype="datetime64[ns]")
        hold_td = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(vol[i]) or vol[i] <= 0 or close[i] <= 0:
                continue

            pt_thresh = float(pt_sl_mult * vol[i])
            sl_thresh = float(-pt_sl_mult * vol[i])
            pt_level = close[i] * np.exp(pt_thresh)
            sl_level = close[i] * np.exp(sl_thresh)
            end = min(i + horizon, n - 1)
            if end <= i:
                continue

            hit = "VERT"
            hit_idx = end
            final_ret = float(np.log(close[end] / close[i]))

            for j in range(i + 1, end + 1):
                hit_pt = bool(np.isfinite(high[j]) and high[j] >= pt_level)
                hit_sl = bool(np.isfinite(low[j]) and low[j] <= sl_level)

                if hit_pt and hit_sl:
                    # Cùng một bar chạm cả 2 barrier.
                    # Dùng open để xử lý gap rõ ràng; nếu open chưa đủ tách bạch,
                    # dùng close so với entry như một tie-break phù hợp với xu hướng bar.
                    open_ret = np.log(open_[j] / close[i]) if open_[j] > 0 else np.nan
                    close_ret = np.log(close[j] / close[i]) if close[j] > 0 else 0.0
                    if np.isfinite(open_ret) and open_ret >= pt_thresh:
                        hit = "PT"
                        final_ret = pt_thresh
                    elif np.isfinite(open_ret) and open_ret <= sl_thresh:
                        hit = "SL"
                        final_ret = sl_thresh
                    elif close_ret >= 0:
                        hit = "PT"
                        final_ret = pt_thresh
                    else:
                        hit = "SL"
                        final_ret = sl_thresh
                    hit_idx = j
                    break

                if hit_pt:
                    hit = "PT"
                    hit_idx = j
                    final_ret = pt_thresh
                    break
                if hit_sl:
                    hit = "SL"
                    hit_idx = j
                    final_ret = sl_thresh
                    break

            if hit == "PT":
                labels[i] = 1
            elif hit == "SL":
                labels[i] = 0
            else:
                labels[i] = 1 if final_ret > 0 else 0

            barriers[i] = hit
            returns[i] = final_ret
            t1_dates[i] = dates[hit_idx]
            hold_td[i] = hit_idx - i

        grp["daily_vol"] = vol
        grp["tb_label"] = labels
        grp["tb_barrier"] = barriers
        grp["tb_return"] = returns
        grp["t1"] = t1_dates
        grp["holding_td"] = hold_td
        return grp

    df = df.groupby(level="ticker", group_keys=False).apply(_label_ticker)

    valid = df["tb_label"].dropna()
    n_total = len(valid)
    if n_total > 0:
        n_pos = int((valid == 1).sum())
        n_neg = int((valid == 0).sum())
        barrier_counts = df.loc[valid.index, "tb_barrier"].value_counts()
        log.info(
            f"  Labels: {n_total:,} | "
            f"1={n_pos:,} ({n_pos/n_total:.1%}) | "
            f"0={n_neg:,} ({n_neg/n_total:.1%})"
        )
        log.info(f"  Barriers: {dict(barrier_counts)}")
    else:
        log.warning("  Không có label nào hợp lệ!")

    return df


def embargo_mask(
    df: pd.DataFrame,
    train_end: str | pd.Timestamp,
    horizon: int = 10,
) -> pd.Index:
    """Trả về index các mẫu trong vùng embargo (H ngày sau train_end)."""
    train_end = pd.Timestamp(train_end)
    dates_all = df.index.get_level_values("date").unique().sort_values()
    future_dates = dates_all[dates_all > train_end]

    embargo_dates = future_dates[:horizon] if len(future_dates) >= horizon else future_dates
    mask = df.index.get_level_values("date").isin(embargo_dates)
    embargo_idx = df.index[mask]

    log.info(
        f"Embargo: {len(embargo_dates)} ngày sau {train_end.date()} "
        f"→ {len(embargo_idx)} mẫu bị loại"
    )
    return embargo_idx
