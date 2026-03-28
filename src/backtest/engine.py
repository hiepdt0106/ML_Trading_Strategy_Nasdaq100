"""
src/backtest/engine.py
──────────────────────
Thesis-ready backtest engine — Long-only, Top-K, Equal-weight, Rebalance

Main fixes vs previous version:
─────────────────────────────────────────────────────────────────
1. Transaction costs are now applied to BOTH equity and daily_ret.
   This avoids overstating Sharpe / Sortino / alpha statistics.

   Cost model note:
   - proportional turnover cost is still an approximation for equal-weight
     portfolios; it is suitable for thesis research, not production execution.

2. Costs are charged on the first holding day (trade at t+1 open),
   which keeps daily return accounting aligned with execution timing.

3. Equity output now stores gross_ret, cost_ret, daily_ret and flags
   for entry days, making the notebook analysis easier and cleaner.

Anti-leakage:
  - Signal at close(t) → trade at open(t+1)
  - No use of future information
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BacktestEngineConfig:
    """
    Cấu hình backtest engine.

    Notes
    -----
    cost_bps và slippage_bps được hiểu là chi phí MỘT CHIỀU
    (one-way) trên traded notional, tính theo basis points.

    Vì vậy:
    - mua mới hoàn toàn ~ chịu 1 leg cost
    - full refresh (bán hết + mua lại hết) ~ chịu cost ở cả 2 legs

    Ví dụ:
    - cost_bps = 10 nghĩa là 10 bps mỗi chiều
    - nếu turnover_est = 2.0 thì net cost xấp xỉ 20 bps
      (chưa tính slippage nếu có)
    """
    top_k: int = 10
    rebalance_days: int = 10
    cost_bps: float = 10.0      # one-way cost on traded notional
    slippage_bps: float = 0.0   # one-way slippage on traded notional
    initial_capital: float = 10_000


def _estimate_trade_cost(
    prev_holdings: set[str],
    new_holdings: set[str],
    total_cost_rate: float,
) -> tuple[set[str], set[str], float, float]:
    """
    Approximate proportional turnover cost for equal-weight portfolios.

    Definitions
    -----------
    total_cost_rate:
        one-way transaction cost applied to traded notional.
        It should already include both explicit fees and slippage
        if you want to model both.

    turnover_est:
        approximate traded notional as a fraction of portfolio NAV.

        Examples
        --------
        - Initial buy from cash into portfolio:
            turnover_est ≈ 1.0
        - Full refresh from one portfolio to a completely different one:
            turnover_est ≈ 2.0
          because we sell the old leg and buy the new leg.

    cost:
        estimated proportional NAV cost:
            cost = turnover_est * total_cost_rate
    """
    if not prev_holdings:
        bought = set(new_holdings)
        turnover_est = 1.0 if new_holdings else 0.0
        return set(), bought, turnover_est, turnover_est * total_cost_rate

    sold = prev_holdings - new_holdings
    bought = new_holdings - prev_holdings
    n_prev = max(len(prev_holdings), 1)
    n_new = max(len(new_holdings), 1)

    turnover_est = (len(sold) / n_prev) + (len(bought) / n_new)
    cost = turnover_est * total_cost_rate
    return sold, bought, turnover_est, cost


def _portfolio_return_for_day(
    df: pd.DataFrame,
    selected: list[str],
    hold_dates: pd.Index,
    j: int,
    d: pd.Timestamp,
) -> float:
    rets: list[float] = []
    for tkr in selected:
        try:
            if j == 0:
                p0 = (
                    df.loc[(d, tkr), "adj_open"]
                    if "adj_open" in df.columns
                    else df.loc[(d, tkr), "adj_close"]
                )
                p1 = df.loc[(d, tkr), "adj_close"]
            else:
                prev_d = hold_dates[j - 1]
                p0 = df.loc[(prev_d, tkr), "adj_close"]
                p1 = df.loc[(d, tkr), "adj_close"]

            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rets.append(float(p1 / p0 - 1.0))

        except KeyError:
            continue

    return float(np.mean(rets)) if rets else 0.0


def run_backtest(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: BacktestEngineConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest chiến lược ML — long-only, top-K, equal-weight.

    Parameters
    ----------
    df       : MultiIndex (date, ticker) — dataset với adj_close / adj_open
    pred_df  : MultiIndex (date, ticker) — predictions với cột y_prob
    cfg      : BacktestEngineConfig

    Returns
    -------
    equity_df  : index=date, columns=[equity, daily_ret, gross_ret, cost_ret, ...]
    trades_df  : log chi tiết mỗi lần rebalance
    """
    if cfg is None:
        cfg = BacktestEngineConfig()

    if "model" in pred_df.columns and pred_df["model"].nunique() != 1:
        raise ValueError(
            f"pred_df chứa {pred_df['model'].nunique()} models: "
            f"{pred_df['model'].unique().tolist()}. "
            f"Phải truyền predictions của 1 model hoặc ensemble. "
            f"Dùng build_ensemble() hoặc filter pred_df trước. "
        )

    total_cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 10_000

    all_dates = df.index.get_level_values("date").unique().sort_values()
    pred_dates = pred_df.index.get_level_values("date").unique().sort_values()
    pred_date_list = pred_dates.tolist()
    rebalance_dates = pred_date_list[::cfg.rebalance_days]

    log.info(
        "Backtest: %s rebalances, every %s days, %s→%s",
        len(rebalance_dates),
        cfg.rebalance_days,
        pred_dates[0].date() if len(pred_dates) else None,
        pred_dates[-1].date() if len(pred_dates) else None,
    )

    equity = float(cfg.initial_capital)
    prev_holdings: set[str] = set()
    eq_records: list[dict] = []
    trade_records: list[dict] = []

    for i, reb_date in enumerate(rebalance_dates):
        available = pred_dates[pred_dates <= reb_date]
        if len(available) == 0:
            continue
        signal_date = available[-1]

        mask = pred_df.index.get_level_values("date") == signal_date
        signal_slice = pred_df.loc[mask]
        signal_pool = signal_slice.set_index(
            signal_slice.index.get_level_values("ticker")
        )["y_prob"]

        if len(signal_pool) == 0:
            continue

        top_k_actual = min(cfg.top_k, len(signal_pool))
        selected_raw = signal_pool.nlargest(top_k_actual).index.tolist()
        # Optional stricter entry filter:
        # only keep names with a valid entry-day open if hold_dates exists
        selected = selected_raw
        new_holdings = set(selected)

        if i + 1 < len(rebalance_dates):
            next_reb = rebalance_dates[i + 1]
            hold_dates = all_dates[(all_dates > reb_date) & (all_dates <= next_reb)]
        else:
            next_reb = pd.NaT
            hold_dates = all_dates[all_dates > reb_date]

        entry_date = hold_dates[0] if len(hold_dates) > 0 else pd.NaT
        if len(hold_dates) > 0 and "adj_open" in df.columns:
            selected = [
                tkr for tkr in selected
                if ((entry_date, tkr) in df.index)
                and pd.notna(df.loc[(entry_date, tkr), "adj_open"])
                and float(df.loc[(entry_date, tkr), "adj_open"]) > 0
            ]
            new_holdings = set(selected)
        if len(hold_dates) == 0:
            trade_records.append({
                "rebalance_date": reb_date,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "holdings": selected,
                "n_holdings": len(selected),
                "n_sold": 0,
                "n_bought": 0,
                "turnover_est": 0.0,
                "cost": 0.0,
                "skipped_no_hold_dates": True,
            })
            continue

        sold, bought, turnover_est, cost = _estimate_trade_cost(
            prev_holdings,
            new_holdings,
            total_cost_rate,
        )

        for j, d in enumerate(hold_dates):
            gross_ret = _portfolio_return_for_day(df, selected, hold_dates, j, d)
            if j == 0:
                net_ret = (1.0 + gross_ret) * (1.0 - cost) - 1.0
                cost_ret = net_ret - gross_ret
            else:
                net_ret = gross_ret
                cost_ret = 0.0

            equity *= (1.0 + net_ret)
            eq_records.append({
                "date": d,
                "equity": equity,
                "daily_ret": net_ret,
                "gross_ret": gross_ret,
                "cost_ret": cost_ret,
                "is_entry_day": bool(j == 0),
                "rebalance_date": reb_date,
                "signal_date": signal_date,
            })

        trade_records.append({
            "rebalance_date": reb_date,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "holdings": selected,
            "n_holdings": len(selected),
            "n_sold": len(sold),
            "n_bought": len(bought),
            "turnover_est": turnover_est,
            "cost": cost,
            "skipped_no_hold_dates": False,
        })
        prev_holdings = new_holdings

    equity_df = pd.DataFrame(eq_records).set_index("date") if eq_records else pd.DataFrame(
        columns=["equity", "daily_ret", "gross_ret", "cost_ret", "is_entry_day", "rebalance_date", "signal_date"]
    )
    trades_df = pd.DataFrame(trade_records)

    equity_df.attrs["initial_capital"] = float(cfg.initial_capital)

    if len(equity_df) > 0:
        log.info(
            "  Final: $%s (return: %.1f%%)",
            f"{equity_df['equity'].iloc[-1]:,.0f}",
            (equity_df["equity"].iloc[-1] / cfg.initial_capital - 1) * 100,
        )

    return equity_df, trades_df


def compute_benchmark(
    df: pd.DataFrame,
    dates: pd.Index,
    initial_capital: float = 10_000,
) -> pd.DataFrame:
    """
    Buy & Hold equal-weight benchmark, aligned with ML timing:
    - Day 1 of holding window: open -> close
    - Subsequent days: close -> close

    Uses the same target dates as ML.
    Keeps the original forward-fill/dropna logic for stability.
    """
    all_dates = df.index.get_level_values("date").unique().sort_values()
    target_dates = all_dates[all_dates.isin(dates)]

    if len(target_dates) == 0:
        return pd.DataFrame(columns=["equity", "daily_ret"])

    close_px = (
        df["adj_close"]
        .unstack("ticker")
        .reindex(target_dates)
        .sort_index()
    )

    if "adj_open" in df.columns:
        open_px = (
            df["adj_open"]
            .unstack("ticker")
            .reindex(target_dates)
            .sort_index()
        )
    else:
        open_px = close_px.copy()

    start_open = open_px.iloc[0]
    valid_tickers = start_open[start_open.notna() & (start_open > 0)].index

    if len(valid_tickers) == 0:
        return pd.DataFrame(columns=["equity", "daily_ret"], index=target_dates)

    close_px = close_px[valid_tickers].copy()
    open_px = open_px[valid_tickers].copy()

    close_px = close_px.ffill()
    open_px = open_px.ffill()

    close_px = close_px.dropna(axis=1, how="any")
    open_px = open_px[close_px.columns]

    if close_px.shape[1] == 0:
        return pd.DataFrame(columns=["equity", "daily_ret"], index=target_dates)

    n_assets = close_px.shape[1]
    capital_per_asset = initial_capital / n_assets

    # Buy once at OPEN of first day
    shares = capital_per_asset / open_px.iloc[0]

    equity_series = close_px.mul(shares, axis=1).sum(axis=1)

    benchmark = pd.DataFrame(index=equity_series.index)
    benchmark["equity"] = equity_series
    benchmark["daily_ret"] = benchmark["equity"].pct_change()

    # First holding day should reflect open -> close return, not 0
    benchmark.loc[benchmark.index[0], "daily_ret"] = benchmark["equity"].iloc[0] / initial_capital - 1.0
    benchmark["gross_ret"] = benchmark["daily_ret"]
    benchmark["cost_ret"] = 0.0
    benchmark["is_entry_day"] = False
    benchmark.attrs["initial_capital"] = float(initial_capital)

    return benchmark


def compute_metrics(equity_df: pd.DataFrame, risk_free_rate: float = 0.04) -> dict:
    """CAGR, Sharpe, Sortino, Max Drawdown, Calmar, VaR, Win Rate."""
    if len(equity_df) < 2:
        return {}

    eq = equity_df["equity"]
    rets = equity_df["daily_ret"].fillna(0.0)

    initial_equity = equity_df.attrs.get("initial_capital")
    if initial_equity is None:
        first_ret = float(rets.iloc[0])
        denom = max(1.0 + first_ret, 1e-12)
        initial_equity = float(eq.iloc[0] / denom)

    n_days = (eq.index[-1] - eq.index[0]).days
    n_years = max(n_days / 365.25, 0.01)
    total_return = float(eq.iloc[-1] / initial_equity)
    cagr = total_return ** (1 / n_years) - 1

    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = rets - daily_rf
    sharpe = excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252)

    downside = rets[rets < 0]
    sortino = (rets.mean() - daily_rf) / max(downside.std(), 1e-10) * np.sqrt(252)

    running_max = eq.cummax()
    drawdown = eq / running_max - 1
    max_dd = float(drawdown.min())

    calmar = cagr / max(abs(max_dd), 1e-10)
    var_95 = float(np.percentile(rets, 5))
    win_rate = float((rets > 0).mean())

    return {
        "Total_Return": total_return - 1,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": max_dd,
        "Calmar": calmar,
        "VaR_95": var_95,
        "Win_Rate": win_rate,
        "Avg_Daily_Ret": float(rets.mean()),
        "Std_Daily_Ret": float(rets.std()),
        "N_Days": len(eq),
    }


def summarize_trade_log(trades_df: pd.DataFrame) -> dict:
    """Useful implementation stats for the research notebook."""
    if len(trades_df) == 0:
        return {}

    valid = trades_df.copy()
    if "skipped_no_hold_dates" in valid.columns:
        valid = valid[~valid["skipped_no_hold_dates"].fillna(False)]
    if len(valid) == 0:
        return {}

    summary = {
        "N_Rebalances": int(len(valid)),
        "Avg_Holdings": float(valid["n_holdings"].mean()),
        "Avg_N_Sold": float(valid["n_sold"].mean()),
        "Avg_N_Bought": float(valid["n_bought"].mean()),
        "Avg_Turnover_Est": float(valid["turnover_est"].mean()) if "turnover_est" in valid else np.nan,
        "Avg_Cost_Per_Rebalance": float(valid["cost"].mean()) if "cost" in valid else np.nan,
        "Pct_Full_Refresh": float((valid["n_sold"] == valid["n_holdings"]).mean()) if len(valid) else np.nan,
    }
    return summary


def compute_alpha_stats(
    ml_equity: pd.DataFrame,
    bh_equity: pd.DataFrame,
    hac_lags: int | None = 5,
) -> dict:
    """
    Compare ML strategy with benchmark using daily alpha:
        daily_alpha = r_ml - r_bh

    If hac_lags is not None, use a Newey-West / HAC robust test
    for the mean daily alpha. This is more appropriate than a naive
    iid t-test when daily returns may be autocorrelated or heteroskedastic.

    If hac_lags is None, fall back to scipy's one-sample t-test.
    """
    common = ml_equity.index.intersection(bh_equity.index)
    if len(common) == 0:
        return {}

    ml_ret = ml_equity.loc[common, "daily_ret"]
    bh_ret = bh_equity.loc[common, "daily_ret"]

    daily_alpha = (ml_ret - bh_ret).dropna()
    n = len(daily_alpha)
    if n == 0:
        return {}

    tracking_error = float(daily_alpha.std(ddof=1))
    ir = float(daily_alpha.mean() / max(tracking_error, 1e-10) * np.sqrt(252))
    avg_annual_alpha = float(daily_alpha.mean() * 252)

    if hac_lags is None:
        from scipy import stats

        t_stat, p_value = stats.ttest_1samp(daily_alpha, 0.0, nan_policy="omit")
        method = "naive_ttest_mean_alpha"
    else:
        import statsmodels.api as sm

        y = daily_alpha.to_numpy(dtype=float)
        X = np.ones((len(y), 1), dtype=float)  # mean(alpha)
        model = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(hac_lags)},
        )
        t_stat = float(model.tvalues[0])
        p_value = float(model.pvalues[0])
        method = f"HAC_mean_alpha_maxlags={int(hac_lags)}"

    return {
        "Information_Ratio": ir,
        "t_stat_alpha": float(t_stat),
        "p_value_alpha": float(p_value),
        "Avg_Annual_Alpha": avg_annual_alpha,
        "Tracking_Error_Annual": float(tracking_error * np.sqrt(252)),
        "N_Days": int(n),
        "Alpha_Test_Method": method,
    }


def compare_to_benchmark(
    ml_equity: pd.DataFrame,
    bh_equity: pd.DataFrame,
) -> dict:
    """Backward-compatible alias with clearer name."""
    return compute_alpha_stats(ml_equity, bh_equity)
