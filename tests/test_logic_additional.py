from __future__ import annotations

import numpy as np
import pandas as pd


def test_walkforward_purges_t1_overlap_and_embargoes_first_h_days():
    from src.splits.walkforward import make_expanding_splits

    dates = pd.bdate_range("2019-12-20", periods=25)
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame({"feat": 1.0}, index=idx)

    # t1 = +3 business days, enough to create overlap across year boundary
    future_map = {d: dates[min(i + 3, len(dates) - 1)] for i, d in enumerate(dates)}
    df["t1"] = [future_map[d] for d, _ in idx]

    splits = make_expanding_splits(df, first_test_year=2020, horizon=2)
    assert len(splits) == 1

    fold = splits[0]
    train_df = df.loc[fold.train_idx]
    test_dates = fold.test_idx.get_level_values("date").unique().sort_values()

    # Purge: no training label is allowed to look into the test period
    assert not (train_df["t1"] > pd.Timestamp("2020-01-01")).any()

    # Embargo: first H trading days of test year are removed entirely
    original_test_dates = dates[dates.year == 2020]
    assert not any(d in test_dates for d in original_test_dates[:2])
    assert test_dates.min() == original_test_dates[2]


def test_cross_sectional_rank_ranks_stock_features_but_keeps_macro_raw():
    from src.models.train import cross_sectional_rank

    dates = pd.bdate_range("2020-01-01", periods=2)
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "ticker"])
    df = pd.DataFrame(
        {
            "ret_1d": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "vix_ret_1d": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        },
        index=idx,
    )

    ranked = cross_sectional_rank(df, ["ret_1d", "vix_ret_1d"])

    day1 = ranked.xs(dates[0], level="date")
    assert list(day1["ret_1d"]) == [1 / 3, 2 / 3, 1.0]
    assert list(day1["vix_ret_1d"]) == [10.0, 10.0, 10.0]

    day2 = ranked.xs(dates[1], level="date")
    assert list(day2["ret_1d"]) == [1.0, 2 / 3, 1 / 3]
    assert list(day2["vix_ret_1d"]) == [20.0, 20.0, 20.0]


def test_benchmark_first_day_is_open_to_close_then_close_to_close():
    from src.backtest.engine import compute_benchmark

    dates = pd.bdate_range("2020-01-01", periods=3)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    df = pd.DataFrame(
        {
            "adj_open": [10, 20, 11, 22, 12, 24],
            "adj_close": [11, 22, 12, 24, 13, 26],
        },
        index=idx,
    )

    bench = compute_benchmark(df, dates, initial_capital=100)

    # 50/50 split, buy once at first open => 5 shares A and 2.5 shares B
    expected_equity = [110.0, 120.0, 130.0]
    expected_daily_ret = [0.10, 120.0 / 110.0 - 1.0, 130.0 / 120.0 - 1.0]

    np.testing.assert_allclose(bench["equity"].to_numpy(), expected_equity, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bench["daily_ret"].to_numpy(), expected_daily_ret, rtol=0, atol=1e-12)


def test_backtest_turnover_cost_charged_only_on_entry_day():
    from src.backtest.engine import BacktestEngineConfig, run_backtest

    dates = pd.bdate_range("2020-01-01", periods=6)
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    rows = []
    for d in dates:
        for tkr, px in {"A": 10.0, "B": 20.0, "C": 30.0}.items():
            rows.append({"adj_open": px, "adj_close": px * 1.01})
    df = pd.DataFrame(rows, index=idx)

    pred_idx = pd.MultiIndex.from_product([dates[:4], tickers], names=["date", "ticker"])
    preds = pd.DataFrame(
        {
            # Rebalance 1 picks A,B ; rebalance 2 keeps same names => zero turnover
            "y_prob": [0.9, 0.8, 0.1, 0.95, 0.85, 0.05, 0.9, 0.8, 0.1, 0.92, 0.82, 0.02],
            "model": "ENS",
        },
        index=pred_idx,
    )

    eq, trades = run_backtest(
        df,
        preds,
        BacktestEngineConfig(top_k=2, rebalance_days=2, cost_bps=10, initial_capital=100),
    )

    assert len(trades) == 2
    assert trades["turnover_est"].iloc[0] == 1.0  # initial buy from cash
    assert trades["turnover_est"].iloc[1] == 0.0  # same holdings => no refresh cost
    assert trades["cost"].iloc[0] > 0
    assert trades["cost"].iloc[1] == 0.0

    entry_days = eq["is_entry_day"].to_numpy()
    assert eq.loc[entry_days, "cost_ret"].iloc[0] < 0
    assert (eq.loc[~entry_days, "cost_ret"] == 0.0).all()
