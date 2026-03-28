"""
src/data/fetch_macro.py
───────────────────────
Fetch VIX & VXN từ Yahoo Finance, cache vào disk.
"""
from __future__ import annotations

import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def _download(symbol: str, start: str, end: str, retries: int = 3) -> pd.Series:
    """Download 1 symbol từ Yahoo Finance, retry nếu lỗi."""
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                symbol, start=start, end=end,
                progress=False, auto_adjust=True, repair=False,
            )
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                s = raw["Close"].squeeze().copy()
                s.index = pd.to_datetime(s.index)
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                s.index = s.index.normalize()
                s.index.name = "date"
                s.name = symbol.replace("^", "").lower()
                return s.sort_index().dropna()
        except Exception as e:
            log.warning(f"{symbol} attempt {attempt} (download): {e}")

        time.sleep(2 * attempt)

        try:
            tkr = yf.Ticker(symbol)
            raw2 = tkr.history(start=start, end=end)
            if raw2 is not None and not raw2.empty:
                s = raw2["Close"].squeeze().copy()
                s.index = pd.to_datetime(s.index)
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                s.index = s.index.normalize()
                s.index.name = "date"
                s.name = symbol.replace("^", "").lower()
                return s.sort_index().dropna()
        except Exception as e:
            log.warning(f"{symbol} attempt {attempt} (Ticker): {e}")
            time.sleep(3 * attempt)

    raise RuntimeError(
        f"Không fetch được {symbol} sau {retries} lần.\n"
        "Thử: pip install --upgrade yfinance rồi restart kernel."
    )


def fetch_macro(
    start:     str,
    end:       str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch VIX & VXN. Cache vào cache_dir nếu được truyền vào."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = cache_dir / f"macro_vix_vxn_{start}_{end}.parquet"
        if fp.exists():
            log.info(f"fetch_macro: cache hit → {fp.name}")
            return pd.read_parquet(fp)

    log.info("Fetching ^VIX ...")
    vix = _download("^VIX", start, end)
    time.sleep(1)
    log.info("Fetching ^VXN ...")
    vxn = _download("^VXN", start, end)

    macro = pd.concat([vix, vxn], axis=1).sort_index().dropna()

    if cache_dir is not None:
        macro.to_parquet(fp)
        log.info(f"fetch_macro: đã cache → {fp.name}")

    log.info(
        f"fetch_macro OK: {len(macro)} ngày | "
        f"VIX [{macro['vix'].min():.1f}–{macro['vix'].max():.1f}] "
        f"VXN [{macro['vxn'].min():.1f}–{macro['vxn'].max():.1f}]"
    )
    return macro
