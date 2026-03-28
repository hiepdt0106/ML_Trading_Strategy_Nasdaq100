"""
src/data/fetch_benchmark.py
───────────────────────────
Fetch benchmark ETF (QQQ) từ Yahoo Finance cho residual features.

QQQ dùng để tính:
  - rolling beta per ticker
  - residual return = stock return - beta × QQQ return
  - idiosyncratic volatility
"""
from __future__ import annotations

import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def fetch_benchmark(
    ticker:    str = "QQQ",
    start:     str = "2014-01-01",
    end:       str = "2026-03-01",
    cache_dir: Path | None = None,
) -> pd.Series:
    """
    Fetch adjusted close của benchmark ETF.

    Returns
    -------
    pd.Series  index=date, name='bench_close'
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = cache_dir / f"bench_{ticker}_{start}_{end}.parquet"
        if fp.exists():
            log.info(f"fetch_benchmark: cache hit → {fp.name}")
            df = pd.read_parquet(fp)
            return df["bench_close"]

    log.info(f"Fetching benchmark {ticker} ...")
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                ticker, start=start, end=end,
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
                s.name = "bench_close"

                if cache_dir is not None:
                    s.to_frame().to_parquet(fp)
                    log.info(f"fetch_benchmark: cached → {fp.name}")

                log.info(
                    f"fetch_benchmark OK: {len(s)} ngày "
                    f"({s.index[0].date()} → {s.index[-1].date()})"
                )
                return s.sort_index().dropna()
        except Exception as e:
            log.warning(f"benchmark {ticker} attempt {attempt}: {e}")
            time.sleep(2 * attempt)

    raise RuntimeError(f"Không fetch được benchmark {ticker}")
