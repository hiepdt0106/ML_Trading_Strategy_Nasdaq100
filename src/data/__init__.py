"""
src/data — clean/build import ngay, network modules lazy.
"""
from .clean import get_trading_days, align_panel, quality_check, filter_valid
from .build_dataset import build_dataset


def fetch_ticker(*args, **kwargs):
    from .tiingo_client import fetch_ticker as _fn
    return _fn(*args, **kwargs)


def fetch_macro(*args, **kwargs):
    from .fetch_macro import fetch_macro as _fn
    return _fn(*args, **kwargs)


def fetch_benchmark(*args, **kwargs):
    from .fetch_benchmark import fetch_benchmark as _fn
    return _fn(*args, **kwargs)


__all__ = [
    "fetch_ticker",
    "fetch_macro",
    "fetch_benchmark",
    "get_trading_days",
    "align_panel",
    "quality_check",
    "filter_valid",
    "build_dataset",
]
