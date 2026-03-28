"""Run walk-forward model training and save ensemble predictions."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging

import pandas as pd

from scripts._common import add_common_args, setup_logging
from src.config import get_feature_cols, load_config, split_feature_cols
from src.models.train import build_ensemble, walk_forward_compare
from src.splits.walkforward import make_expanding_splits
from src.utils.io import load, save

log = logging.getLogger(__name__)


def run_models(config_path: str | None = None):
    cfg = load_config(config_path) if config_path else load_config()

    # ── Load labeled dataset ──
    df = load(cfg.dir_processed / "dataset_labeled.parquet")
    log.info("Loaded labeled dataset: %s", df.shape)

    # ── Derive feature columns ──
    feature_cols_full = get_feature_cols(df.columns.tolist())
    feature_cols_base, _ = split_feature_cols(feature_cols_full)

    splits = make_expanding_splits(
        df,
        first_test_year=cfg.walkforward.first_test_year,
        horizon=cfg.labeling.horizon,
        max_train_years=cfg.walkforward.max_train_years,
    )
    if not splits:
        raise RuntimeError(
            "Không tạo được fold nào. Kiểm tra first_test_year / dữ liệu / labeling."
        )

    results_full, pred_full, results_base, pred_base = walk_forward_compare(
        df,
        splits,
        feature_cols_full=feature_cols_full,
        feature_cols_base=feature_cols_base,
        target="tb_label",
        top_k=cfg.strategy.top_k,
    )

    pred_ens_full = build_ensemble(pred_full)
    pred_ens_base = build_ensemble(pred_base)

    metrics_dir = cfg.dir_outputs / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_full.to_csv(metrics_dir / "walkforward_full.csv", index=False)
    results_base.to_csv(metrics_dir / "walkforward_base.csv", index=False)
    save(pred_full, cfg.dir_processed / "predictions_full.parquet")
    save(pred_base, cfg.dir_processed / "predictions_base.parquet")
    save(pred_ens_full, cfg.dir_processed / "predictions_ens_full.parquet")
    save(pred_ens_base, cfg.dir_processed / "predictions_ens_base.parquet")

    stability = results_full.groupby("model").agg(
        daily_auc_mean=("daily_auc", "mean"),
        daily_auc_std=("daily_auc", "std"),
        top_k_mean=("top_k_ret", "mean"),
        top_k_std=("top_k_ret", "std"),
    )
    stability.to_csv(metrics_dir / "model_stability_full.csv")

    log.info("Saved walk-forward outputs to %s and %s", cfg.dir_processed, metrics_dir)
    return results_full, pred_ens_full, results_base, pred_ens_base


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_models(args.config)


if __name__ == "__main__":
    main()
