"""
src/utils/summary.py
────────────────────
In kết quả tổng hợp sau khi pipeline hoàn tất.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config

log = logging.getLogger(__name__)


def print_summary(config_path: str | Path | None = None) -> None:
    """In bảng kết quả + liệt kê output files."""
    cfg = load_config(config_path) if config_path else load_config()
    metrics_path = cfg.dir_outputs / "metrics" / "backtest_metrics.csv"
    alpha_path = cfg.dir_outputs / "metrics" / "alpha_stats.csv"

    if not metrics_path.exists():
        log.warning("Chưa có kết quả backtest. Chạy pipeline trước.")
        return

    df = pd.read_csv(metrics_path, index_col=0)
    initial = cfg.backtest.initial_capital

    print("\n" + "═" * 70)
    print("  RESULTS SUMMARY")
    print("═" * 70)
    print(f"  {'Strategy':<12} {'Equity':>12} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8}")
    print("  " + "─" * 58)

    for s in df.index:
        equity = (1 + df.loc[s, "Total_Return"]) * initial
        print(
            f"  {s:<12} ${equity:>10,.0f} {df.loc[s, 'CAGR']*100:>7.1f}% "
            f"{df.loc[s, 'Sharpe']:>7.2f} {df.loc[s, 'Max_Drawdown']*100:>7.1f}% "
            f"{df.loc[s, 'Calmar']:>7.2f}"
        )

    if alpha_path.exists():
        alpha = pd.read_csv(alpha_path, index_col=0)
        print("\n  " + "─" * 58)
        print(f"  {'Alpha vs B&H':<12} {'Annual α':>12} {'IR':>8} {'t-stat':>8} {'p-value':>8}")
        print("  " + "─" * 58)
        for s in alpha.index:
            print(
                f"  {s:<12} {alpha.loc[s, 'Avg_Annual_Alpha']*100:>+11.2f}% "
                f"{alpha.loc[s, 'Information_Ratio']:>7.3f} "
                f"{alpha.loc[s, 't_stat_alpha']:>7.3f} "
                f"{alpha.loc[s, 'p_value_alpha']:>7.3f}"
            )

    print("\n  Output files:")
    for name, desc in [
        ("equity_full.parquet", "ML Full equity curve"),
        ("equity_base.parquet", "ML Base equity curve"),
        ("equity_benchmark.parquet", "Buy & Hold equity curve"),
        ("metrics/backtest_metrics.csv", "KPI summary"),
        ("metrics/alpha_stats.csv", "Alpha statistics"),
        ("metrics/trade_log_full.csv", "Trade details"),
        ("reporting/", "Power BI tables"),
    ]:
        path = cfg.dir_outputs / name
        status = "✓" if path.exists() else "✗"
        print(f"    {status} {name:<35} {desc}")

    print("═" * 70 + "\n")