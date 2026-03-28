"""Export Power BI-ready reporting tables from thesis pipeline outputs.

Usage (from project root):
    python -m scripts.export_reporting --config configs/base.yaml --log-level INFO
or:
    python scripts/export_reporting.py --config configs/base.yaml --log-level INFO

Outputs under outputs/reporting/:
    - dim_date.csv
    - dim_strategy.csv
    - fact_strategy_daily.csv
    - fact_strategy_kpi.csv
    - fact_strategy_kpi_long.csv
    - fact_strategy_annual.csv
    - fact_alpha.csv
    - fact_walkforward.csv
    - fact_walkforward_summary.csv
    - fact_model_stability.csv
    - fact_sensitivity.csv
    - fact_trade_rebalance.csv
    - fact_trade_summary.csv
    - reporting_manifest.csv

Purpose:
    Create a clean reporting mart for Power BI instead of loading raw metrics files one by one.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, setup_logging
from src.backtest.engine import compute_alpha_stats, compute_metrics
from src.config import load_config
from src.utils.io import load

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %s (%s rows)", path.relative_to(ROOT), len(df))


def _standardize_equity(eq: pd.DataFrame, strategy: str, initial_capital: float) -> pd.DataFrame:
    df = eq.copy()
    if df.index.name == "date":
        df = df.reset_index()
    elif "date" not in df.columns:
        df = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})

    df["date"] = _ensure_datetime(df["date"])
    df["strategy"] = strategy
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df["daily_ret"] = pd.to_numeric(df.get("daily_ret", 0.0), errors="coerce").fillna(0.0)
    df["gross_ret"] = pd.to_numeric(df.get("gross_ret", df["daily_ret"]), errors="coerce").fillna(0.0)
    df["cost_ret"] = pd.to_numeric(df.get("cost_ret", 0.0), errors="coerce").fillna(0.0)
    df["is_entry_day"] = df.get("is_entry_day", False).fillna(False).astype(bool)

    if "rebalance_date" in df.columns:
        df["rebalance_date"] = _ensure_datetime(df["rebalance_date"])
    else:
        df["rebalance_date"] = pd.NaT

    if "signal_date" in df.columns:
        df["signal_date"] = _ensure_datetime(df["signal_date"])
    else:
        df["signal_date"] = pd.NaT

    df = df.sort_values("date").reset_index(drop=True)
    df["cum_return"] = df["equity"] / float(initial_capital) - 1.0
    df["running_peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["running_peak"] - 1.0
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["quarter"] = "Q" + df["date"].dt.quarter.astype(str)
    df["year_quarter"] = df["date"].dt.year.astype(str) + "-Q" + df["date"].dt.quarter.astype(str)
    df["day_of_week"] = df["date"].dt.day_name()
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    keep = [
        "date",
        "strategy",
        "equity",
        "cum_return",
        "drawdown",
        "daily_ret",
        "gross_ret",
        "cost_ret",
        "is_entry_day",
        "rebalance_date",
        "signal_date",
        "year",
        "month_num",
        "month",
        "quarter",
        "year_quarter",
        "day_of_week",
        "days_since_start",
    ]
    return df[keep]


def _build_dim_date(all_dates: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(all_dates).dropna().sort_values().unique(), name="date")
    dim = pd.DataFrame({"date": dates})
    dim["year"] = dim["date"].dt.year
    dim["quarter_num"] = dim["date"].dt.quarter
    dim["quarter"] = "Q" + dim["quarter_num"].astype(str)
    dim["month_num"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.month_name()
    dim["month_short"] = dim["date"].dt.strftime("%b")
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    dim["year_quarter"] = dim["date"].dt.year.astype(str) + "-Q" + dim["date"].dt.quarter.astype(str)
    dim["week_of_year"] = dim["date"].dt.isocalendar().week.astype(int)
    dim["day_of_month"] = dim["date"].dt.day
    dim["day_of_week_num"] = dim["date"].dt.weekday + 1
    dim["day_of_week"] = dim["date"].dt.day_name()
    dim["is_month_start"] = dim["date"].dt.is_month_start
    dim["is_month_end"] = dim["date"].dt.is_month_end
    dim["is_quarter_start"] = dim["date"].dt.is_quarter_start
    dim["is_quarter_end"] = dim["date"].dt.is_quarter_end
    dim["is_year_start"] = dim["date"].dt.is_year_start
    dim["is_year_end"] = dim["date"].dt.is_year_end
    return dim


def _build_dim_strategy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy": ["ML_Full", "ML_Base", "Buy&Hold"],
            "strategy_sort": [1, 2, 3],
            "feature_set": ["Full", "Base", "Benchmark"],
            "strategy_group": ["ML", "ML", "Benchmark"],
            "description": [
                "Ensemble of LR, RF, XGB using full feature set including macro and regime features.",
                "Ensemble of LR, RF, XGB using base feature set without macro/regime features.",
                "Equal-weight buy and hold benchmark aligned to ML holding dates.",
            ],
        }
    )


def _build_kpi_summary(
    eq_full: pd.DataFrame,
    eq_base: pd.DataFrame,
    eq_bench: pd.DataFrame,
    initial_capital: float,
    metrics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df = _load_csv_if_exists(metrics_path)
    if metrics_df is None:
        metrics_df = pd.DataFrame(
            {
                "ML_Full": compute_metrics(eq_full),
                "ML_Base": compute_metrics(eq_base),
                "Buy&Hold": compute_metrics(eq_bench),
            }
        ).T.reset_index().rename(columns={"index": "strategy"})
    else:
        first_col = metrics_df.columns[0]
        if first_col != "strategy":
            metrics_df = metrics_df.rename(columns={first_col: "strategy"})

    metrics_df["initial_capital"] = float(initial_capital)
    kpi_long = metrics_df.melt(id_vars=["strategy", "initial_capital"], var_name="metric_name", value_name="metric_value")
    return metrics_df, kpi_long


def _build_annual_returns(metrics_dir: Path) -> pd.DataFrame:
    annual = pd.read_csv(metrics_dir / "annual_returns.csv")
    first = annual.columns[0]
    annual = annual.rename(columns={first: "year_end"})
    annual["year_end"] = pd.to_datetime(annual["year_end"], errors="coerce")
    annual["year"] = annual["year_end"].dt.year
    annual_long = annual.melt(id_vars=["year_end", "year"], var_name="strategy", value_name="annual_return")
    annual_long = annual_long.sort_values(["year", "strategy"]).reset_index(drop=True)
    return annual_long


def _build_alpha(metrics_dir: Path, eq_full: pd.DataFrame, eq_base: pd.DataFrame, eq_bench: pd.DataFrame) -> pd.DataFrame:
    alpha = _load_csv_if_exists(metrics_dir / "alpha_stats.csv")
    if alpha is None:
        alpha = pd.DataFrame(
            {
                "ML_Full": compute_alpha_stats(eq_full, eq_bench),
                "ML_Base": compute_alpha_stats(eq_base, eq_bench),
            }
        ).T.reset_index().rename(columns={"index": "strategy"})
    else:
        first_col = alpha.columns[0]
        if first_col != "strategy":
            alpha = alpha.rename(columns={first_col: "strategy"})
    return alpha


def _build_walkforward(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(metrics_dir / "walkforward_full.csv")
    base = pd.read_csv(metrics_dir / "walkforward_base.csv")
    full["strategy"] = "ML_Full"
    full["feature_set"] = "Full"
    base["strategy"] = "ML_Base"
    base["feature_set"] = "Base"
    walk = pd.concat([full, base], ignore_index=True)
    walk_summary = (
        walk.groupby(["strategy", "feature_set", "model"], as_index=False)
        .agg(
            n_folds=("fold", "count"),
            daily_auc_mean=("daily_auc", "mean"),
            daily_auc_std=("daily_auc", "std"),
            global_auc_mean=("global_auc", "mean"),
            global_auc_std=("global_auc", "std"),
            top_k_ret_mean=("top_k_ret", "mean"),
            top_k_ret_std=("top_k_ret", "std"),
            train_size_mean=("train_size", "mean"),
            test_size_mean=("test_size", "mean"),
        )
    )

    stability = _load_csv_if_exists(metrics_dir / "model_stability_full.csv")
    if stability is None:
        stability = walk_summary[walk_summary["strategy"] == "ML_Full"].copy()
    else:
        first_col = stability.columns[0]
        if first_col != "model":
            stability = stability.rename(columns={first_col: "model"})
        stability["strategy"] = "ML_Full"
        stability["feature_set"] = "Full"
    return walk, walk_summary, stability


def _build_sensitivity(metrics_dir: Path) -> pd.DataFrame:
    def one_file(path: Path, scenario_name: str, parameter_col: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        first_col = df.columns[0]
        if first_col != parameter_col:
            df = df.rename(columns={first_col: parameter_col})
        df["scenario_type"] = scenario_name
        df["strategy"] = "ML_Full"
        return df

    topk = one_file(metrics_dir / "sensitivity_topk.csv", "top_k", "parameter_value")
    cost = one_file(metrics_dir / "sensitivity_cost.csv", "cost_bps", "parameter_value")
    reb = one_file(metrics_dir / "sensitivity_rebalance.csv", "rebalance_days", "parameter_value")
    wide = pd.concat([topk, cost, reb], ignore_index=True, sort=False)
    long = wide.melt(
        id_vars=["scenario_type", "strategy", "parameter_value"],
        var_name="metric_name",
        value_name="metric_value",
    )
    return long.sort_values(["scenario_type", "parameter_value", "metric_name"]).reset_index(drop=True)


def _build_trade_tables(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(metrics_dir / "trade_log_full.csv")
    base = pd.read_csv(metrics_dir / "trade_log_base.csv")

    full["strategy"] = "ML_Full"
    base["strategy"] = "ML_Base"
    trades = pd.concat([full, base], ignore_index=True, sort=False)

    for col in ["rebalance_date", "signal_date", "entry_date"]:
        if col in trades.columns:
            trades[col] = _ensure_datetime(trades[col])

    if "holdings" in trades.columns:
        trades["holdings_text"] = trades["holdings"].astype(str)
        trades["holding_count_from_text"] = trades["holdings_text"].str.count("'") // 2
    else:
        trades["holdings_text"] = ""
        trades["holding_count_from_text"] = np.nan

    trades["year"] = trades["rebalance_date"].dt.year
    trades["month"] = trades["rebalance_date"].dt.strftime("%Y-%m")

    summary = _load_csv_if_exists(metrics_dir / "trade_summary.csv")
    if summary is None:
        summary = (
            trades.groupby("strategy", as_index=False)
            .agg(
                N_Rebalances=("rebalance_date", "count"),
                Avg_Holdings=("n_holdings", "mean"),
                Avg_N_Sold=("n_sold", "mean"),
                Avg_N_Bought=("n_bought", "mean"),
                Avg_Turnover_Est=("turnover_est", "mean"),
                Avg_Cost_Per_Rebalance=("cost", "mean"),
            )
        )
    else:
        first_col = summary.columns[0]
        if first_col != "strategy":
            summary = summary.rename(columns={first_col: "strategy"})
    return trades, summary


def _build_manifest() -> pd.DataFrame:
    rows = [
        ["dim_date.csv", "Dimension", "Calendar table for slicers and time intelligence", "date"],
        ["dim_strategy.csv", "Dimension", "Strategy metadata and sort order", "strategy"],
        ["fact_strategy_daily.csv", "Fact", "Daily equity, returns, drawdown for ML_Full / ML_Base / Buy&Hold", "date, strategy"],
        ["fact_strategy_kpi.csv", "Fact", "Overall KPI table by strategy", "strategy"],
        ["fact_strategy_kpi_long.csv", "Fact", "Long-format KPI table for easier visual filtering", "strategy, metric_name"],
        ["fact_strategy_annual.csv", "Fact", "Annual returns by strategy", "year, strategy"],
        ["fact_alpha.csv", "Fact", "Alpha statistics vs benchmark", "strategy"],
        ["fact_walkforward.csv", "Fact", "Fold-level model diagnostics", "strategy, fold, model"],
        ["fact_walkforward_summary.csv", "Fact", "Aggregated walk-forward summary by strategy and model", "strategy, model"],
        ["fact_model_stability.csv", "Fact", "Model stability summary for ML_Full", "model"],
        ["fact_sensitivity.csv", "Fact", "Sensitivity analysis in long format", "scenario_type, parameter_value, metric_name"],
        ["fact_trade_rebalance.csv", "Fact", "Rebalance-level trade log for ML strategies", "strategy, rebalance_date"],
        ["fact_trade_summary.csv", "Fact", "Trade summary by strategy", "strategy"],
    ]
    return pd.DataFrame(rows, columns=["file_name", "table_type", "purpose", "grain"])


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_reporting(config_path: str | None = None) -> None:
    cfg = load_config(config_path) if config_path else load_config()

    out_dir = cfg.dir_outputs
    metrics_dir = out_dir / "metrics"
    reporting_dir = out_dir / "reporting"
    reporting_dir.mkdir(parents=True, exist_ok=True)

    required = [
        out_dir / "equity_full.parquet",
        out_dir / "equity_base.parquet",
        out_dir / "equity_benchmark.parquet",
        metrics_dir / "backtest_metrics.csv",
        metrics_dir / "annual_returns.csv",
        metrics_dir / "walkforward_full.csv",
        metrics_dir / "walkforward_base.csv",
        metrics_dir / "sensitivity_topk.csv",
        metrics_dir / "sensitivity_cost.csv",
        metrics_dir / "sensitivity_rebalance.csv",
        metrics_dir / "trade_log_full.csv",
        metrics_dir / "trade_log_base.csv",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu output đầu vào cho reporting. Hãy chạy run_models.py, run_backtest.py, run_analysis.py trước.\n"
            + "Missing:\n- "
            + "\n- ".join(missing)
        )

    initial_capital = float(cfg.backtest.initial_capital)
    eq_full = load(out_dir / "equity_full.parquet")
    eq_base = load(out_dir / "equity_base.parquet")
    eq_bench = load(out_dir / "equity_benchmark.parquet")

    fact_daily = pd.concat(
        [
            _standardize_equity(eq_full, "ML_Full", initial_capital),
            _standardize_equity(eq_base, "ML_Base", initial_capital),
            _standardize_equity(eq_bench, "Buy&Hold", initial_capital),
        ],
        ignore_index=True,
    ).sort_values(["date", "strategy"]).reset_index(drop=True)

    dim_date = _build_dim_date(fact_daily["date"])
    dim_strategy = _build_dim_strategy()
    kpi, kpi_long = _build_kpi_summary(
        eq_full=eq_full,
        eq_base=eq_base,
        eq_bench=eq_bench,
        initial_capital=initial_capital,
        metrics_path=metrics_dir / "backtest_metrics.csv",
    )
    annual = _build_annual_returns(metrics_dir)
    alpha = _build_alpha(metrics_dir, eq_full, eq_base, eq_bench)
    walk, walk_summary, stability = _build_walkforward(metrics_dir)
    sensitivity = _build_sensitivity(metrics_dir)
    trade_rebalance, trade_summary = _build_trade_tables(metrics_dir)
    manifest = _build_manifest()

    _save_csv(dim_date, reporting_dir / "dim_date.csv")
    _save_csv(dim_strategy, reporting_dir / "dim_strategy.csv")
    _save_csv(fact_daily, reporting_dir / "fact_strategy_daily.csv")
    _save_csv(kpi, reporting_dir / "fact_strategy_kpi.csv")
    _save_csv(kpi_long, reporting_dir / "fact_strategy_kpi_long.csv")
    _save_csv(annual, reporting_dir / "fact_strategy_annual.csv")
    _save_csv(alpha, reporting_dir / "fact_alpha.csv")
    _save_csv(walk, reporting_dir / "fact_walkforward.csv")
    _save_csv(walk_summary, reporting_dir / "fact_walkforward_summary.csv")
    _save_csv(stability, reporting_dir / "fact_model_stability.csv")
    _save_csv(sensitivity, reporting_dir / "fact_sensitivity.csv")
    _save_csv(trade_rebalance, reporting_dir / "fact_trade_rebalance.csv")
    _save_csv(trade_summary, reporting_dir / "fact_trade_summary.csv")
    _save_csv(manifest, reporting_dir / "reporting_manifest.csv")

    log.info("Done. Import only outputs/reporting/* into Power BI.")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    export_reporting(args.config)


if __name__ == "__main__":
    main()
