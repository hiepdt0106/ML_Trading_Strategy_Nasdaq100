"""
src/data/pipeline.py
────────────────────
Orchestrate toàn bộ data pipeline: fetch → align → QC → build dataset.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.build_dataset import build_dataset
from src.data.clean import align_panel, filter_valid, quality_check
from src.data.fetch_macro import fetch_macro
from src.data.fetch_benchmark import fetch_benchmark
from src.data.tiingo_client import fetch_ticker
from src.utils.io import save

log = logging.getLogger(__name__)


def run_data_pipeline(
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy toàn bộ pipeline: fetch → align → QC → merge."""
    cfg = load_config(config_path) if config_path else load_config()

    # ── Fetch stocks ──
    stocks = {}
    for ticker in cfg.data.tickers:
        stocks[ticker] = fetch_ticker(
            ticker=ticker,
            start=cfg.data.start_date,
            end=cfg.data.end_date,
            cache_dir=cfg.dir_cache / "tiingo",
        )

    stock_panel = pd.concat(stocks, names=["ticker"]).swaplevel(0, 1).sort_index()
    stock_panel.index = stock_panel.index.set_names(["date", "ticker"])

    # ── Fetch macro ──
    macro = fetch_macro(
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_dir=cfg.dir_cache / "macro",
    )

    # ── Fetch benchmark ──
    bench = fetch_benchmark(
        ticker=cfg.data.benchmark_ticker,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        cache_dir=cfg.dir_cache / "benchmark",
    )

    # ── Align (bench passed as kwarg) ──
    stocks_aligned, macro_aligned, bench_aligned = align_panel(
        stock_panel, macro,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        bench=bench,
    )

    # ── QC ──
    qc_report, valid_tickers = quality_check(
        stocks_aligned,
        min_days=cfg.data.min_trading_days,
        max_nan_ratio=cfg.data.max_nan_ratio,
        max_consec_nan=cfg.data.max_consec_nan,
    )
    stocks_filtered = filter_valid(stocks_aligned, valid_tickers)

    # ── Build dataset (bench passed as kwarg) ──
    dataset = build_dataset(stocks_filtered, macro_aligned, bench=bench_aligned)

    # ── Save ──
    save(stocks_filtered, cfg.dir_interim / "stock_panel_qc.parquet")
    save(dataset, cfg.dir_processed / "dataset.parquet")
    save(qc_report, cfg.dir_outputs / "metrics" / "qc_report.parquet")

    log.info("Pipeline complete!")
    return dataset, qc_report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_data_pipeline()
