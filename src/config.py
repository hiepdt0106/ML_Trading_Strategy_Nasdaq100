"""
src/config.py
─────────────
Đọc configs/base.yaml → dataclass config dùng chung.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str
    tickers: List[str]
    benchmark_ticker: str = "QQQ"
    min_trading_days: int = 200
    max_nan_ratio: float = 0.02
    max_consec_nan: int = 5


@dataclass(frozen=True)
class LabelingConfig:
    horizon: int = 10
    pt_sl_mult: float = 1.5
    vol_window: int = 20


@dataclass(frozen=True)
class StrategyConfig:
    top_k: int = 10
    rebalance_days: int = 10


@dataclass(frozen=True)
class RegimeConfig:
    lookback: int = 504
    refit_frequency: int = 63


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 10.0
    slippage_bps: float = 0.0
    initial_capital: float = 10_000

@dataclass
class WalkForwardConfig:
    first_test_year: int = 2020
    max_train_years: int | None = 8

@dataclass
class ProjectConfig:
    data: DataConfig
    labeling: LabelingConfig
    strategy: StrategyConfig
    walkforward: WalkForwardConfig
    regime: RegimeConfig
    backtest: BacktestConfig
    
    # Paths — tính từ ROOT
    dir_raw: Path = field(default_factory=lambda: ROOT / "data" / "raw")
    dir_interim: Path = field(default_factory=lambda: ROOT / "data" / "interim")
    dir_processed: Path = field(default_factory=lambda: ROOT / "data" / "processed")
    dir_cache: Path = field(default_factory=lambda: ROOT / "data" / "raw" / "cache")
    dir_outputs: Path = field(default_factory=lambda: ROOT / "outputs")
    dir_figures: Path = field(default_factory=lambda: ROOT / "outputs" / "figures")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE REGISTRY — nguồn duy nhất định nghĩa feature columns
# ═══════════════════════════════════════════════════════════════════════════════

# Columns KHÔNG phải features — phải loại khỏi feature set
NON_FEATURE_COLS = frozenset({
    # Raw OHLCV
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
    # Macro raw
    "vix", "vxn",
    # Label / target columns
    "tb_label", "tb_barrier", "tb_return", "daily_vol", "t1", "holding_td",
    # Benchmark
    "bench_close",
})

# Macro features (giống nhau cho mọi ticker cùng ngày)
# → KHÔNG rank cross-sectional (sẽ ra tie)
MACRO_FEATURE_NAMES = frozenset({
    "vix_ret_1d", "vxn_ret_1d", "vxn_ret_5d",
    "vxn_zscore", "vxn_ma5_ma21", "vxn_accel",
    "vix_vxn_spread",
})

# Features KHÔNG nên rank cross-sectional (dùng trong _SKIP_RANK):
# - Macro features: giống nhau cho mọi ticker → rank = tie
# - market_dispersion_21d: cross-sectional metric, giống cho mọi ticker
# - p_high_vol: regime probability, giống cho mọi ticker
# - Benchmark-relative features: giữ raw magnitude vì ý nghĩa kinh tế
#   (beta=1.5 khác beta=0.8 không phải chỉ ranking, mà là mức độ rủi ro hệ thống)
SKIP_RANK_FEATURES = MACRO_FEATURE_NAMES | frozenset({
    "market_dispersion_21d",
    "p_high_vol",
    "rolling_beta_63d",
    "resid_ret_21d",
    "idio_vol_21d",
    "rel_strength_21d",
    "rel_strength_63d",
    "downside_beta_63d",
})

# Prefix để nhận diện macro features khi tách ML Full vs ML Base
MACRO_FEATURE_PREFIXES = ("vix_", "vxn_", "zspread", "vix_vxn_", "p_high")


def get_feature_cols(df_columns: list[str]) -> list[str]:
    """Trả về danh sách feature columns từ DataFrame columns.

    Nguồn duy nhất cho logic chọn features — không hardcode _EXCLUDE
    trong notebook nữa.
    """
    return [c for c in df_columns if c not in NON_FEATURE_COLS]


def split_feature_cols(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Tách thành base_cols (giá + vol) và macro_cols (VIX/VXN/zSpread)."""
    macro_cols = [c for c in feature_cols if c.startswith(MACRO_FEATURE_PREFIXES)]
    base_cols = [c for c in feature_cols if c not in macro_cols]
    return base_cols, macro_cols


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Key bắt buộc phải có trong YAML
_REQUIRED = {
    "data":     ["start_date", "end_date", "tickers"],
    "labeling": ["horizon", "pt_sl_mult"],
    "strategy": ["top_k", "rebalance_days"],
    "backtest": ["cost_bps", "initial_capital"],
    "regime":   ["lookback"],
    "walkforward": ["first_test_year"],
}


def load_config(path: str | Path = ROOT / "configs" / "base.yaml") -> ProjectConfig:
    """Đọc YAML → ProjectConfig dataclass.

    Validate schema, type check, trả về config có type hints.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy config: {path}\n"
            f"ROOT hiện tại: {ROOT}"
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _validate(raw, path)

    cfg = ProjectConfig(
        data=DataConfig(**raw["data"]),
        labeling=LabelingConfig(**raw["labeling"]),
        strategy=StrategyConfig(**raw["strategy"]),
        walkforward=WalkForwardConfig(**raw["walkforward"]),
        regime=RegimeConfig(**raw["regime"]),
        backtest=BacktestConfig(**raw["backtest"]),
    )
    return cfg


def _validate(raw: dict, path: Path) -> None:
    """Kiểm tra schema, type, range — fail fast nếu sai."""
    from datetime import datetime

    for section, keys in _REQUIRED.items():
        if section not in raw:
            raise KeyError(f"[{path.name}] Thiếu section '{section}'")
        for k in keys:
            if k not in raw[section]:
                raise KeyError(f"[{path.name}] Thiếu key '{section}.{k}'")

    # ── data ──
    d = raw["data"]
    if not isinstance(d["tickers"], list) or len(d["tickers"]) == 0:
        raise ValueError("data.tickers phải là list không rỗng")
    try:
        start = datetime.strptime(d["start_date"], "%Y-%m-%d")
        end = datetime.strptime(d["end_date"], "%Y-%m-%d")
    except (ValueError, TypeError) as e:
        raise ValueError(f"start_date/end_date phải có format YYYY-MM-DD: {e}")
    if start >= end:
        raise ValueError("start_date phải nhỏ hơn end_date")
    if d.get("min_trading_days", 200) < 1:
        raise ValueError("min_trading_days phải >= 1")
    if not (0 < d.get("max_nan_ratio", 0.02) <= 1):
        raise ValueError("max_nan_ratio phải trong (0, 1]")

    # ── labeling ──
    lb = raw["labeling"]
    if lb["horizon"] < 1:
        raise ValueError("labeling.horizon phải >= 1")
    if lb["pt_sl_mult"] <= 0:
        raise ValueError("labeling.pt_sl_mult phải > 0")

    # ── strategy ──
    st = raw["strategy"]
    if st["top_k"] < 1:
        raise ValueError("strategy.top_k phải >= 1")
    if st["rebalance_days"] < 1:
        raise ValueError("strategy.rebalance_days phải >= 1")

    # ── regime ──
    rg = raw["regime"]
    if rg["lookback"] < 1:
        raise ValueError("regime.lookback phải >= 1")

    # ── walkforward ──
    wf = raw["walkforward"]
    if wf["first_test_year"] < 2000:
        raise ValueError("walkforward.first_test_year phải >= 2000")

    # ── backtest ──
    bt = raw["backtest"]
    if bt["cost_bps"] < 0:
        raise ValueError("backtest.cost_bps phải >= 0")
    if bt["initial_capital"] <= 0:
        raise ValueError("backtest.initial_capital phải > 0")
