"""Create tables and figures from saved backtest outputs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._common import add_common_args, setup_logging
from src.backtest.engine import (
    BacktestEngineConfig,
    compute_alpha_stats,
    compute_metrics,
    run_backtest,
)
from src.config import load_config
from src.utils.io import load

log = logging.getLogger(__name__)


def _save_equity_plot(eq_full: pd.DataFrame, eq_base: pd.DataFrame, eq_bench: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq_full.index, eq_full["equity"], label="ML Full")
    ax.plot(eq_base.index, eq_base["equity"], label="ML Base")
    ax.plot(eq_bench.index, eq_bench["equity"], label="Buy&Hold")
    ax.set_title("Equity Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_drawdown_plot(eq_full: pd.DataFrame, eq_base: pd.DataFrame, eq_bench: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for eq, label in [(eq_full, "ML Full"), (eq_base, "ML Base"), (eq_bench, "Buy&Hold")]:
        dd = eq["equity"] / eq["equity"].cummax() - 1.0
        ax.plot(dd.index, dd.values, label=label)
    ax.set_title("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_analysis(config_path: str | None = None) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    figures_dir = cfg.dir_figures
    metrics_dir = cfg.dir_outputs / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = load(cfg.dir_processed / "dataset_labeled.parquet")
    pred_ens_full = load(cfg.dir_processed / "predictions_ens_full.parquet")
    eq_full = load(cfg.dir_outputs / "equity_full.parquet")
    eq_base = load(cfg.dir_outputs / "equity_base.parquet")
    eq_bench = load(cfg.dir_outputs / "equity_benchmark.parquet")

    metrics_df = pd.DataFrame(
        {
            "ML_Full": compute_metrics(eq_full),
            "ML_Base": compute_metrics(eq_base),
            "Buy&Hold": compute_metrics(eq_bench),
        }
    ).T
    alpha_df = pd.DataFrame(
        {
            "ML_Full": compute_alpha_stats(eq_full, eq_bench),
            "ML_Base": compute_alpha_stats(eq_base, eq_bench),
        }
    ).T
    metrics_df.to_csv(metrics_dir / "performance_metrics_analysis.csv")
    alpha_df.to_csv(metrics_dir / "alpha_stats_analysis.csv")

    annual = pd.DataFrame(
        {
            "ML_Full": eq_full["daily_ret"].resample("YE").apply(lambda x: (1 + x).prod() - 1),
            "ML_Base": eq_base["daily_ret"].resample("YE").apply(lambda x: (1 + x).prod() - 1),
            "Buy&Hold": eq_bench["daily_ret"].resample("YE").apply(lambda x: (1 + x).prod() - 1),
        }
    )
    annual.to_csv(metrics_dir / "annual_returns.csv")

    topk_rows = []
    for k in [3, 5, 8, 10, 12, 15]:
        bt = BacktestEngineConfig(
            top_k=k,
            rebalance_days=cfg.strategy.rebalance_days,
            cost_bps=cfg.backtest.cost_bps,
            initial_capital=cfg.backtest.initial_capital,
        )
        eq, _ = run_backtest(df, pred_ens_full, bt)
        m = compute_metrics(eq)
        m["top_k"] = k
        topk_rows.append(m)
    pd.DataFrame(topk_rows).set_index("top_k").to_csv(metrics_dir / "sensitivity_topk.csv")

    cost_rows = []
    for cost in [0, 5, 10, 15, 20, 30]:
        bt = BacktestEngineConfig(
            top_k=cfg.strategy.top_k,
            rebalance_days=cfg.strategy.rebalance_days,
            cost_bps=cost,
            initial_capital=cfg.backtest.initial_capital,
        )
        eq, _ = run_backtest(df, pred_ens_full, bt)
        m = compute_metrics(eq)
        m["cost_bps"] = cost
        cost_rows.append(m)
    pd.DataFrame(cost_rows).set_index("cost_bps").to_csv(metrics_dir / "sensitivity_cost.csv")

    rebalance_rows = []
    for freq in [5, 10, 15, 21, 42]:
        bt = BacktestEngineConfig(
            top_k=cfg.strategy.top_k,
            rebalance_days=freq,
            cost_bps=cfg.backtest.cost_bps,
            initial_capital=cfg.backtest.initial_capital,
        )
        eq, _ = run_backtest(df, pred_ens_full, bt)
        m = compute_metrics(eq)
        m["rebalance_days"] = freq
        rebalance_rows.append(m)
    pd.DataFrame(rebalance_rows).set_index("rebalance_days").to_csv(metrics_dir / "sensitivity_rebalance.csv")

    _save_equity_plot(eq_full, eq_base, eq_bench, figures_dir / "equity_curves.png")
    _save_drawdown_plot(eq_full, eq_base, eq_bench, figures_dir / "drawdown.png")
    log.info("Saved analysis tables and figures to %s", cfg.dir_outputs)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_analysis(args.config)


if __name__ == "__main__":
    main()
