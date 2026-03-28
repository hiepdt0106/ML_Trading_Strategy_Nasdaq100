"""Run the data pipeline from the command line."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path before importing project modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, setup_logging
from src.data.pipeline import run_data_pipeline

log = logging.getLogger(__name__)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    run_data_pipeline(args.config)


if __name__ == "__main__":
    main()