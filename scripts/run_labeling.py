"""Apply triple-barrier labeling to feature dataset."""
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
from src.config import load_config
from src.labeling import label as triple_barrier_label
from src.utils.io import load, save

log = logging.getLogger(__name__)


def run_labeling(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path) if config_path else load_config()
    in_path = cfg.dir_processed / "dataset_features.parquet"
    out_path = cfg.dir_processed / "dataset_labeled.parquet"

    df = load(in_path)
    log.info("Loaded features: %s", df.shape)

    labeled = triple_barrier_label(
        df,
        horizon=cfg.labeling.horizon,
        pt_sl_mult=cfg.labeling.pt_sl_mult,
        vol_window=cfg.labeling.vol_window,
    )
    labeled = labeled.dropna(subset=["tb_label"]).copy()
    labeled["tb_label"] = labeled["tb_label"].astype(int)

    save(labeled, out_path)

    flat = labeled.reset_index()
    flat["year"] = flat["date"].dt.year
    label_summary = pd.DataFrame(
        {
            "metric": [
                "n_rows",
                "n_tickers",
                "label_pos",
                "label_neg",
                "pos_rate",
                "avg_holding_td",
            ],
            "value": [
                len(labeled),
                labeled.index.get_level_values("ticker").nunique(),
                int((labeled["tb_label"] == 1).sum()),
                int((labeled["tb_label"] == 0).sum()),
                float((labeled["tb_label"] == 1).mean()),
                float(labeled["holding_td"].mean()),
            ],
        }
    )
    barrier_by_year = flat.groupby(["year", "tb_barrier"]).size().unstack(fill_value=0)

    label_summary.to_csv(cfg.dir_outputs / "metrics" / "label_summary.csv", index=False)
    barrier_by_year.to_csv(cfg.dir_outputs / "metrics" / "barrier_by_year.csv")

    log.info("Saved labeled dataset: %s", out_path)
    return labeled


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_labeling(args.config)


if __name__ == "__main__":
    main()
