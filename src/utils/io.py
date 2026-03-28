"""
src/utils/io.py
───────────────
Save / load parquet, đảm bảo MultiIndex giữ nguyên sau round-trip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to parquet, tự tạo parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load(path: str | Path) -> pd.DataFrame:
    """Load parquet, chuẩn hoá datetime index, assert no duplicates."""
    df = pd.read_parquet(Path(path))

    # Chuẩn hoá datetime trong index
    if isinstance(df.index, pd.MultiIndex) and "date" in df.index.names:
        idx = df.index.to_frame(index=False)
        idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
        df.index = pd.MultiIndex.from_frame(idx)
    elif df.index.name == "date":
        df.index = pd.to_datetime(df.index).normalize()

    assert not df.index.duplicated().any(), \
        f"File {path} có duplicate index sau khi load!"

    return df.sort_index()


def log_return(series: pd.Series, n: int = 1) -> pd.Series:
    """Log return n kỳ: ln(P_t / P_{t-n}). Dùng chung cho price, vol, macro."""
    return np.log(series / series.shift(n))
