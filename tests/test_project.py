"""
tests/test_project.py
─────────────────────
Unit tests cho logic thực sự — không chỉ string match.
Covers: config validation, feature registry, labeling, XGB refit,
        guard clauses, backtest safety, QC.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def read_utf8(path: Path) -> str:
    """Read text files explicitly as UTF-8 for Windows compatibility."""
    return path.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def test_config_loads():
    from src.config import load_config
    cfg = load_config()
    assert cfg.labeling.pt_sl_mult == 1.5
    assert cfg.strategy.rebalance_days == 10
    assert cfg.regime.lookback == 504
    assert cfg.backtest.cost_bps == 10


def test_config_validation_bad_dates():
    """Validation nên reject dates sai format hoặc ngược."""
    from src.config import _validate

    base = {
        "data": {"start_date": "2020-01-01", "end_date": "2014-01-01",
                 "tickers": ["AAPL"]},
        "labeling": {"horizon": 10, "pt_sl_mult": 1.5},
        "strategy": {"top_k": 5, "rebalance_days": 10},
        "walkforward": {"first_test_year": 2020},
        "regime": {"lookback": 504},
        "backtest": {"cost_bps": 10, "initial_capital": 10000},
    }
    with pytest.raises(ValueError, match="start_date"):
        _validate(base, Path("test.yaml"))


def test_config_validation_bad_ranges():
    """Validation nên reject giá trị ngoài range."""
    from src.config import _validate

    base = {
        "data": {"start_date": "2014-01-01", "end_date": "2020-01-01",
                 "tickers": ["AAPL"], "min_trading_days": 200,
                 "max_nan_ratio": 0.02},
        "labeling": {"horizon": 10, "pt_sl_mult": 1.5},
        "strategy": {"top_k": 0, "rebalance_days": 10},
        "walkforward": {"first_test_year": 2020},
        "regime": {"lookback": 504},
        "backtest": {"cost_bps": 10, "initial_capital": 10000},
    }
    with pytest.raises(ValueError, match="top_k"):
        _validate(base, Path("test.yaml"))


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_registry_excludes_non_features():
    from src.config import get_feature_cols

    cols = ["ret_1d", "vol_21d", "adj_close", "tb_label", "vix", "bench_close"]
    result = get_feature_cols(cols)
    assert "ret_1d" in result
    assert "vol_21d" in result
    assert "adj_close" not in result
    assert "tb_label" not in result
    assert "vix" not in result
    assert "bench_close" not in result


def test_split_feature_cols():
    from src.config import split_feature_cols

    feature_cols = ["ret_1d", "vol_21d", "vix_ret_1d", "vxn_zscore", "zspread"]
    base, macro = split_feature_cols(feature_cols)
    assert "ret_1d" in base
    assert "vix_ret_1d" in macro
    assert "zspread" in macro
    assert len(base) + len(macro) == len(feature_cols)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURES — no double-ranking, correct names
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_rank_features_in_price():
    """price.py KHÔNG tạo _rank features (tránh double-ranking)."""
    src = read_utf8(ROOT / "src" / "features" / "price.py")
    assert 'f"{col}_rank"' not in src


def test_no_rank_features_in_volatility():
    src = read_utf8(ROOT / "src" / "features" / "volatility.py")
    assert 'f"{col}_rank"' not in src


def test_position_in_range_not_close_to_high():
    """Feature phải tên position_in_range_20d, không phải close_to_high_20d."""
    src = read_utf8(ROOT / "src" / "features" / "price.py")
    assert "position_in_range_20d" in src
    assert "close_to_high_20d" not in src


def test_downside_vol_uses_where_not_clip():
    """downside_vol phải dùng .where(ret < 0), không .clip(upper=0)."""
    src = read_utf8(ROOT / "src" / "features" / "relative.py")
    assert "ret.where(ret < 0)" in src
    assert "ret.clip(upper=0)" not in src


def test_cross_sectional_rank_skips_macro():
    from src.models.train import _SKIP_RANK
    assert "vix_ret_1d" in _SKIP_RANK
    assert "vxn_zscore" in _SKIP_RANK
    assert "market_dispersion_21d" in _SKIP_RANK
    assert "p_high_vol" in _SKIP_RANK
    assert "rolling_beta_63d" in _SKIP_RANK


# ═══════════════════════════════════════════════════════════════════════════════
# LABELING
# ═══════════════════════════════════════════════════════════════════════════════

def test_triple_barrier_basic():
    """Giá tăng đều → nên có nhiều label=1."""
    from src.labeling.triple_barrier import label

    dates = pd.bdate_range("2020-01-01", periods=100)
    tickers = ["AAPL"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    prices = np.linspace(100, 150, len(dates))
    df = pd.DataFrame({
        "adj_close": prices,
        "adj_open": prices * 0.999,
        "adj_high": prices * 1.005,
        "adj_low": prices * 0.995,
        "adj_volume": np.full(len(dates), 1e6),
    }, index=idx)

    result = label(df, horizon=10, pt_sl_mult=1.5, vol_window=20)
    valid = result["tb_label"].dropna()
    assert len(valid) > 0
    assert valid.mean() > 0.5


def test_embargo_mask():
    from src.labeling.triple_barrier import embargo_mask

    dates = pd.bdate_range("2020-01-01", periods=50)
    tickers = ["AAPL", "MSFT"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame({"adj_close": 100.0}, index=idx)

    emb = embargo_mask(df, train_end="2020-01-15", horizon=5)
    emb_dates = emb.get_level_values("date").unique()
    assert len(emb_dates) == 5
    assert all(d > pd.Timestamp("2020-01-15") for d in emb_dates)


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS — XGB early stopping, global_auc guard
# ═══════════════════════════════════════════════════════════════════════════════

def test_xgb_uses_date_block_validation():
    """XGB validation phải chia theo date block, không theo row."""
    src = read_utf8(ROOT / "src" / "models" / "train.py")
    assert "train_dates_sorted" in src
    assert "val_start_date" in src
    # Không có refit logic (đã bỏ vì gây underfit)
    assert "model_refit" not in src


def test_xgb_val_split_by_date_block():
    """Val split dựa trên date, không cắt giữa ngày."""
    src = read_utf8(ROOT / "src" / "models" / "train.py")
    assert "val_start_date = train_dates_sorted" in src
    assert "int(n * 0.85)" not in src


def test_global_auc_guarded():
    """global_auc phải guard khi chỉ có 1 class."""
    src = read_utf8(ROOT / "src" / "models" / "train.py")
    assert "len(np.unique(y_test)) < 2" in src
    assert "g_auc = np.nan" in src


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST — guard, regime modes, trade timing
# ═══════════════════════════════════════════════════════════════════════════════

def test_backtest_guards_multi_model():
    """Backtest phải reject pred_df chứa nhiều models."""
    src = read_utf8(ROOT / "src" / "backtest" / "engine.py")
    assert 'pred_df["model"].nunique() != 1' in src


def test_backtest_signal_date_not_average():
    """Backtest dùng signal ngày rebalance, không average 5 ngày."""
    src = read_utf8(ROOT / "src" / "backtest" / "engine.py")
    assert "[-5:]" not in src
    assert "signal_date" in src


def test_backtest_trade_at_t_plus_1():
    src = read_utf8(ROOT / "src" / "backtest" / "engine.py")
    assert "all_dates > reb_date" in src


def test_compute_metrics():
    from src.backtest.engine import compute_metrics

    dates = pd.bdate_range("2020-01-01", periods=252)
    np.random.seed(42)
    equity = 10000 * (1 + np.random.normal(0.0003, 0.01, len(dates))).cumprod()
    daily_rets = np.diff(equity, prepend=equity[0]) / np.concatenate([[equity[0]], equity[:-1]])
    eq_df = pd.DataFrame({
        "equity": equity,
        "daily_ret": daily_rets,
    }, index=dates)
    eq_df.index.name = "date"

    metrics = compute_metrics(eq_df)
    assert "CAGR" in metrics
    assert "Sharpe" in metrics
    assert "Max_Drawdown" in metrics
    assert metrics["Max_Drawdown"] <= 0


# ═══════════════════════════════════════════════════════════════════════════════
# QC — checks all OHLCV columns
# ═══════════════════════════════════════════════════════════════════════════════

def test_qc_checks_ohlcv_columns():
    """QC nên loại ticker có volume NaN nhiều dù adj_close OK."""
    from src.data.clean import quality_check

    dates = pd.bdate_range("2020-01-01", periods=100)
    tickers = ["GOOD", "BAD_VOL"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    data = {
        "adj_close": 100.0,
        "adj_open": 99.0,
        "adj_high": 101.0,
        "adj_low": 98.0,
        "adj_volume": 1e6,
    }
    df = pd.DataFrame(data, index=idx)

    # BAD_VOL: adj_close OK nhưng volume 50% NaN
    bad_mask = (
        (df.index.get_level_values("ticker") == "BAD_VOL") &
        (df.index.get_level_values("date") < dates[50])
    )
    df.loc[bad_mask, "adj_volume"] = np.nan

    report, valid = quality_check(df, min_days=50, max_nan_ratio=0.10, max_consec_nan=60)
    assert "GOOD" in valid
    assert "BAD_VOL" not in valid


# ═══════════════════════════════════════════════════════════════════════════════
# HMM REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def test_hmm_predict_proba_default():
    from src.regime.hmm import RegimeHMM

    hmm = RegimeHMM()
    # Chưa fit → trả về 0.5
    assert hmm.predict_proba(pd.Series([20.0] * 10), pd.Series([15.0] * 10)) == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════

def test_readme_regime_description_accurate():
    """README mô tả regime as feature (đã implement qua regime_features.py)."""
    readme = read_utf8(ROOT / "README.md")
    # Phải nói regime as feature
    assert "regime as feature" in readme.lower() or "p_high_vol" in readme
    # Phải nói overlay đã loại bỏ
    assert "loại bỏ" in readme or "removed" in readme.lower()


def test_regime_features_module_exists():
    """regime_features.py phải tồn tại và có add_regime_features."""
    path = ROOT / "src" / "features" / "regime_features.py"
    assert path.exists()
    src = read_utf8(path)
    assert "def add_regime_features" in src
    assert "p_high_vol" in src


def test_project_structure():
    for p in ["data", "features", "labeling", "models", "splits", "backtest", "utils", "regime"]:
        assert (ROOT / "src" / p).exists(), f"Missing src/{p}"


def test_cost_model_documented_as_approximation():
    """Cost model phải được ghi rõ là approximation."""
    src = read_utf8(ROOT / "src" / "backtest" / "engine.py")
    assert "approximation" in src.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — mini end-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def test_mini_pipeline_end_to_end():
    """
    Synthetic end-to-end: features → label → split → train → backtest.
    Dùng 3 tickers, 500 ngày, đảm bảo pipeline không crash.
    """
    from src.features.price import add_price_features
    from src.features.volatility import add_vol_features
    from src.labeling.triple_barrier import label
    from src.splits.walkforward import make_expanding_splits
    from src.models.train import walk_forward_train
    from src.backtest.engine import BacktestEngineConfig, run_backtest, compute_metrics
    from src.config import get_feature_cols

    np.random.seed(42)
    dates = pd.bdate_range("2018-01-02", periods=600)
    tickers = ["AAA", "BBB", "CCC"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    n = len(dates)
    base_prices = {}
    for tkr in tickers:
        rets = np.random.normal(0.0005, 0.015, n)
        close = 100.0 * np.cumprod(1 + rets)
        base_prices[tkr] = close

    rows = []
    for d_i, d in enumerate(dates):
        for tkr in tickers:
            c = base_prices[tkr][d_i]
            rows.append({
                "adj_close": c,
                "adj_open": c * (1 + np.random.normal(0, 0.002)),
                "adj_high": c * (1 + abs(np.random.normal(0, 0.005))),
                "adj_low": c * (1 - abs(np.random.normal(0, 0.005))),
                "adj_volume": float(np.random.randint(1_000_000, 10_000_000)),
                "vix": 20.0 + np.random.normal(0, 2),
                "vxn": 22.0 + np.random.normal(0, 2),
            })

    df = pd.DataFrame(rows, index=idx)

    # Features
    df = add_price_features(df)
    df = add_vol_features(df)

    # Label
    df = label(df, horizon=10, pt_sl_mult=1.5, vol_window=20)

    # Get features + drop NaN
    feature_cols = get_feature_cols(df.columns.tolist())
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=feature_cols + ["tb_label"])
    assert len(df_clean) > 100, f"Too few rows after dropna: {len(df_clean)}"

    # Split
    splits = make_expanding_splits(df_clean, first_test_year=2020, horizon=10)
    assert len(splits) > 0, "No splits created"

    # Train (1 fold only for speed)
    results, preds = walk_forward_train(
        df_clean, splits[:1], feature_cols, target="tb_label", top_k=2
    )
    assert len(results) > 0
    assert "y_prob" in preds.columns

    # Backtest
    ens = preds.copy()
    ens["model"] = "ENS"
    cfg = BacktestEngineConfig(top_k=2, rebalance_days=10, cost_bps=10)
    eq, trades = run_backtest(df_clean, ens, cfg)
    assert len(eq) > 0
    assert "equity" in eq.columns

    metrics = compute_metrics(eq)
    assert "CAGR" in metrics
    assert "Sharpe" in metrics


def test_walk_forward_handles_nan_labels():
    """walk_forward_train should not crash when some labels are NaN."""
    from src.splits.walkforward import FoldSplit

    dates = pd.bdate_range("2019-01-02", periods=300)
    tickers = ["X", "Y"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    np.random.seed(123)
    df = pd.DataFrame({
        "feat1": np.random.randn(len(idx)),
        "feat2": np.random.randn(len(idx)),
        "adj_close": 100.0,
        "tb_label": np.random.choice([0, 1], size=len(idx)).astype(float),
        "tb_return": np.random.randn(len(idx)) * 0.01,
    }, index=idx)

    # Inject NaN labels
    df.loc[df.index[:20], "tb_label"] = np.nan

    train_mask = df.index.get_level_values("date").year < 2020
    test_mask = df.index.get_level_values("date").year == 2020

    fold = FoldSplit(
        fold=1, test_year=2020,
        train_idx=df.index[train_mask],
        test_idx=df.index[test_mask],
        train_end=pd.Timestamp("2019-12-31"),
        purge_n=0, embargo_n=0,
    )

    from src.models.train import walk_forward_train
    results, preds = walk_forward_train(
        df, [fold], ["feat1", "feat2"], target="tb_label", top_k=1
    )
    assert len(results) > 0, "Should produce results even with NaN labels"