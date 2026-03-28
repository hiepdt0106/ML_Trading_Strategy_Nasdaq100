"""Run the full project pipeline from ETL to analysis and reporting."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, setup_logging
from scripts.export_reporting import export_reporting
from scripts.run_analysis import run_analysis
from scripts.run_backtest import run_backtests
from scripts.run_features import build_features
from scripts.run_labeling import run_labeling
from scripts.run_models import run_models
from src.data.pipeline import run_data_pipeline

log = logging.getLogger(__name__)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)

    log.info("[1/7] Running data pipeline...")
    run_data_pipeline(args.config)

    log.info("[2/7] Building features...")
    build_features(args.config)

    log.info("[3/7] Running labeling...")
    run_labeling(args.config)

    log.info("[4/7] Training models...")
    run_models(args.config)

    log.info("[5/7] Running backtests...")
    run_backtests(args.config)

    log.info("[6/7] Running analysis...")
    run_analysis(args.config)

    log.info("[7/7] Exporting reporting tables for Power BI...")
    export_reporting(args.config)

    log.info("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()