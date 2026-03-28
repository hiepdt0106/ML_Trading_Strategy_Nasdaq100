"""Generate engineered features from processed dataset."""
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
from src.config import get_feature_cols, load_config
from src.features import (
    add_macro_features,
    add_price_features,
    add_regime_features,
    add_relative_features,
    add_vol_features,
)
from src.utils.io import load, save

log = logging.getLogger(__name__)


def build_features(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path) if config_path else load_config()
    in_path = cfg.dir_processed / "dataset.parquet"
    out_path = cfg.dir_processed / "dataset_features.parquet"
    metrics_dir = cfg.dir_outputs / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = load(in_path)
    log.info("Loaded dataset: %s", df.shape)

    df = add_price_features(df)
    df = add_vol_features(df)
    df = add_macro_features(df, vxn_window=252)
    df = add_relative_features(df)
    df = add_regime_features(
        df,
        lookback=cfg.regime.lookback,
        refit_freq=cfg.regime.refit_frequency,
    )

    feature_cols = get_feature_cols(df.columns.tolist())
    nan_summary = df[feature_cols].isna().mean().sort_values(ascending=False)
    n_before = len(df)
    df = df.dropna(subset=feature_cols).sort_index()
    n_after = len(df)

    save(df, out_path)

    feature_summary = pd.DataFrame(
        {
            "feature": nan_summary.index,
            "nan_pct_before_drop": nan_summary.values,
        }
    )
    feature_summary.to_csv(metrics_dir / "feature_columns.csv", index=False)

    log.info(
        "Saved features: %s | rows %s -> %s (dropped %s)",
        out_path,
        f"{n_before:,}",
        f"{n_after:,}",
        f"{n_before - n_after:,}",
    )
    log.info("Total feature columns: %s", len(feature_cols))
    return df


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    build_features(args.config)


if __name__ == "__main__":
    main()
