"""
src/data/clean.py
─────────────────
Align stock panel về lịch giao dịch NYSE + kiểm tra chất lượng.

Nguyên tắc:
- KHÔNG ffill OHLCV stocks — thiếu thì để NaN, để QC xử lý
- ffill chỉ dùng cho macro (VIX/VXN) vì lý do holiday mismatch
- QC phải chạy trên dữ liệu chưa bị fill
"""
from __future__ import annotations

import logging
import pandas as pd

log = logging.getLogger(__name__)


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Lấy danh sách ngày giao dịch NYSE."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=start, end_date=end)
        days = mcal.date_range(sched, frequency="1D")
        days = pd.DatetimeIndex(days).tz_localize(None).normalize()
        log.info(f"NYSE calendar: {len(days)} ngày ({days[0].date()} → {days[-1].date()})")
        return days
    except ImportError:
        raise ImportError(
            "pandas_market_calendars là bắt buộc cho pipeline này.\n"
            "Cài đặt: pip install pandas-market-calendars"
        )


def align_panel(
    stocks: pd.DataFrame,
    macro:  pd.DataFrame,
    start:  str,
    end:    str,
    bench:  pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Reindex stocks, macro, (optional) benchmark về đúng ngày giao dịch NYSE.

    Stocks: KHÔNG ffill — thiếu ngày nào để NaN ngày đó.
    Macro:  ffill(limit=1) chấp nhận được (holiday mismatch).
    Bench:  ffill(limit=1) như macro. Optional — nếu không truyền, return 2 giá trị.

    Backward compatible:
      stocks_al, macro_al = align_panel(stocks, macro, start, end)           # OK
      stocks_al, macro_al, bench_al = align_panel(stocks, macro, start, end, bench=bench)  # OK
    """
    trading_days = get_trading_days(start, end)

    # ── Align macro ──
    macro = macro.copy()
    macro.index = pd.to_datetime(macro.index).normalize()
    if macro.index.tz is not None:
        macro.index = macro.index.tz_localize(None)
    macro_al = macro.reindex(trading_days).ffill(limit=1)

    # ── Align benchmark (optional) ──
    bench_al = None
    if bench is not None:
        bench = bench.copy()
        bench.index = pd.to_datetime(bench.index).normalize()
        if bench.index.tz is not None:
            bench.index = bench.index.tz_localize(None)
        bench_al = bench.reindex(trading_days).ffill(limit=1)

    # ── Align stocks (KHÔNG ffill) ──
    tickers = stocks.index.get_level_values("ticker").unique()
    pieces = []
    for tkr in tickers:
        sub = stocks.xs(tkr, level="ticker").copy()
        sub.index = pd.to_datetime(sub.index).normalize()
        sub = sub.reindex(trading_days)
        sub.index.name = "date"
        sub["ticker"] = tkr
        pieces.append(sub.reset_index().set_index(["date", "ticker"]))

    stocks_al = pd.concat(pieces).sort_index()

    dates = stocks_al.index.get_level_values("date").unique()
    log.info(f"stocks_al : {stocks_al.shape} | {dates[0].date()} → {dates[-1].date()}")
    log.info(f"macro_al  : {macro_al.shape}")

    if bench_al is not None:
        log.info(f"bench_al  : {len(bench_al)} ngày")
        return stocks_al, macro_al, bench_al
    return stocks_al, macro_al


def quality_check(
    stocks:         pd.DataFrame,
    min_days:       int   = 200,
    max_nan_ratio:  float = 0.02,
    max_consec_nan: int   = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """
    QC từng ticker trên dữ liệu CHƯA fill.

    Kiểm tra NaN ratio cho tất cả OHLCV columns (không chỉ adj_close).
    Ticker bị loại nếu BẤT KỲ cột OHLCV nào vượt ngưỡng NaN.
    """
    _OHLCV = ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]

    rows = []
    for tkr, grp in stocks.groupby(level="ticker"):
        col = grp["adj_close"] if "adj_close" in grp.columns else grp.iloc[:, 0]
        n_total = len(grp)
        n_valid = col.notna().sum()
        nan_ratio_close = col.isna().mean()
        bad_price = (col.dropna() <= 0).any()
        has_dup = grp.index.get_level_values("date").duplicated().any()

        # Max consecutive NaN (adj_close)
        consec = col.isna().astype(int)
        max_consec = (consec * (consec.groupby(
            (consec != consec.shift()).cumsum()).cumcount() + 1
        )).max()

        # Check NaN ratio cho tất cả OHLCV columns
        ohlcv_cols = [c for c in _OHLCV if c in grp.columns]
        max_nan_any_col = max(
            grp[c].isna().mean() for c in ohlcv_cols
        ) if ohlcv_cols else nan_ratio_close

        valid = (
            n_valid >= min_days
            and max_nan_any_col <= max_nan_ratio
            and not bad_price
            and not has_dup
            and max_consec <= max_consec_nan
        )
        rows.append({
            "ticker":          tkr,
            "n_total":         n_total,
            "n_valid":         n_valid,
            "nan_pct_close":   round(nan_ratio_close * 100, 2),
            "nan_pct_max_col": round(max_nan_any_col * 100, 2),
            "max_consec":      int(max_consec),
            "bad_price":       bad_price,
            "has_dup":         has_dup,
            "valid":           valid,
        })

    report = pd.DataFrame(rows).sort_values("nan_pct_max_col", ascending=False)
    valid = report.loc[report["valid"], "ticker"].tolist()
    bad = report.loc[~report["valid"], "ticker"].tolist()

    if bad:
        log.warning(f"quality_check: loại {len(bad)} ticker: {bad}")
    log.info(f"quality_check: {len(valid)}/{len(rows)} ticker hợp lệ")
    return report, valid


def filter_valid(stocks: pd.DataFrame, valid_tickers: list[str]) -> pd.DataFrame:
    """Giữ lại chỉ các tickers hợp lệ."""
    mask = stocks.index.get_level_values("ticker").isin(valid_tickers)
    return stocks[mask]
