"""
src/data/build_dataset.py
─────────────────────────
Merge stocks + macro + (optional) benchmark → dataset duy nhất.
"""
from __future__ import annotations

import logging
import pandas as pd

log = logging.getLogger(__name__)


def build_dataset(
    stocks: pd.DataFrame,
    macro:  pd.DataFrame,
    bench:  pd.Series | None = None,
) -> pd.DataFrame:
    """
    Merge stocks panel (MultiIndex date,ticker) với macro (index=date)
    và optional benchmark close (index=date).

    Backward compatible:
      dataset = build_dataset(stocks, macro)                # OK — no bench
      dataset = build_dataset(stocks, macro, bench=bench)   # OK — with bench
    """
    if stocks.index.names != ["date", "ticker"]:
        raise ValueError("stocks.index phải là MultiIndex ['date', 'ticker']")

    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index).normalize()
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)

    if not macro.index.is_unique:
        raise ValueError("macro.index có duplicate date")
    required_cols = {"vix", "vxn"}
    if not required_cols.issubset(macro.columns):
        raise ValueError(f"macro phải có cột {sorted(required_cols)}")

    # Merge stocks + macro
    stocks_flat = stocks.reset_index()
    stocks_flat["date"] = pd.to_datetime(stocks_flat["date"]).dt.normalize()

    macro_flat = macro.reset_index()
    if "index" in macro_flat.columns:
        macro_flat = macro_flat.rename(columns={"index": "date"})

    merged = stocks_flat.merge(macro_flat, on="date", how="left", validate="many_to_one")

    # Merge benchmark (optional)
    if bench is not None:
        bench = bench.copy()
        bench.index = pd.to_datetime(bench.index).normalize()
        if bench.index.tz is not None:
            bench.index = bench.index.tz_localize(None)
        bench_df = bench.to_frame("bench_close")
        bench_flat = bench_df.reset_index()
        if "index" in bench_flat.columns:
            bench_flat = bench_flat.rename(columns={"index": "date"})
        merged = merged.merge(bench_flat, on="date", how="left", validate="many_to_one")

    merged = merged.set_index(["date", "ticker"]).sort_index()

    # Check macro NaN
    macro_cols = [c for c in macro.columns if c in merged.columns]
    remaining_na = merged[macro_cols].isna().sum().sum()
    if remaining_na:
        raise ValueError(
            "Macro vẫn còn NaN sau merge. "
            "Hãy align/fill macro trong align_panel() trước khi build_dataset()."
        )

    # Check benchmark NaN (if present)
    if "bench_close" in merged.columns:
        bench_na = merged["bench_close"].isna().sum()
        if bench_na:
            log.warning(f"bench_close có {bench_na} NaN → ffill")
            merged["bench_close"] = merged.groupby(level="ticker")["bench_close"].ffill()

    log.info(
        "build_dataset: %s rows | columns=%s",
        f"{len(merged):,}", list(merged.columns)
    )
    return merged
