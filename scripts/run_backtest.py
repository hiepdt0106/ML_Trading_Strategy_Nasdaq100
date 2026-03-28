"""Run backtests for ML Full, ML Base and Buy&Hold benchmark."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Bootstrap project root for direct script execution on Windows / VSCode
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, setup_logging
from src.backtest.engine import (
    BacktestEngineConfig,
    compute_alpha_stats,
    compute_benchmark,
    compute_metrics,
    run_backtest,
    summarize_trade_log,
)
from src.config import load_config
from src.utils.io import load, save

log = logging.getLogger(__name__)


def _log_final_equity(name: str, eq_df: pd.DataFrame, initial_capital: float) -> None:
    if eq_df is None or len(eq_df) == 0:
        log.warning("%s: no equity records.", name)
        return

    final_equity = float(eq_df["equity"].iloc[-1])
    total_return = final_equity / float(initial_capital) - 1.0
    start_date = eq_df.index.min().date()
    end_date = eq_df.index.max().date()

    log.info(
        "%s | %s -> %s | Final: $%s (return: %.1f%%)",
        name,
        start_date,
        end_date,
        f"{final_equity:,.0f}",
        total_return * 100.0,
    )


def run_backtests(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path) if config_path else load_config()

    df = load(cfg.dir_processed / "dataset_labeled.parquet")
    pred_ens_full = load(cfg.dir_processed / "predictions_ens_full.parquet")
    pred_ens_base = load(cfg.dir_processed / "predictions_ens_base.parquet")

    bt_cfg = BacktestEngineConfig(
        top_k=cfg.strategy.top_k,
        rebalance_days=cfg.strategy.rebalance_days,
        cost_bps=cfg.backtest.cost_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        initial_capital=cfg.backtest.initial_capital,
    )

    eq_full, trades_full = run_backtest(df, pred_ens_full, bt_cfg)
    eq_base, trades_base = run_backtest(df, pred_ens_base, bt_cfg)
    eq_bench = compute_benchmark(df, eq_full.index, initial_capital=bt_cfg.initial_capital)

    # Log benchmark final explicitly (ML strategies are already logged inside run_backtest)
    _log_final_equity("Buy&Hold", eq_bench, bt_cfg.initial_capital)

    m_full = compute_metrics(eq_full)
    m_base = compute_metrics(eq_base)
    m_bench = compute_metrics(eq_bench)
    metrics_df = pd.DataFrame(
        {"ML_Full": m_full, "ML_Base": m_base, "Buy&Hold": m_bench}
    ).T

    alpha_df = pd.DataFrame(
        {
            "ML_Full": compute_alpha_stats(eq_full, eq_bench),
            "ML_Base": compute_alpha_stats(eq_base, eq_bench),
        }
    ).T

    trade_summary_df = pd.DataFrame(
        {
            "ML_Full": summarize_trade_log(trades_full),
            "ML_Base": summarize_trade_log(trades_base),
        }
    ).T

    metrics_dir = cfg.dir_outputs / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_dir / "backtest_metrics.csv")
    alpha_df.to_csv(metrics_dir / "alpha_stats.csv")
    trade_summary_df.to_csv(metrics_dir / "trade_summary.csv")
    trades_full.to_csv(metrics_dir / "trade_log_full.csv", index=False)
    trades_base.to_csv(metrics_dir / "trade_log_base.csv", index=False)

    save(eq_full, cfg.dir_outputs / "equity_full.parquet")
    save(eq_base, cfg.dir_outputs / "equity_base.parquet")
    save(eq_bench, cfg.dir_outputs / "equity_benchmark.parquet")

    log.info("Backtest metrics summary:\n%s", metrics_df.round(4).to_string())
    log.info("Saved backtest outputs to %s", cfg.dir_outputs)

    return metrics_df


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_backtests(args.config)


if __name__ == "__main__":
    main()