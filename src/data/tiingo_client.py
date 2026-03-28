"""
src/data/tiingo_client.py
─────────────────────────
HTTP client cho Tiingo EOD API.

Nguyên tắc:
- Dùng requests trực tiếp (không SDK) để kiểm soát hoàn toàn response
- Cache theo (ticker, start, end) → không tốn API call khi chạy lại
- Validate schema sau khi load cache — tránh dùng cache cũ bị lỗi
- Retry + exponential backoff khi gặp rate-limit (HTTP 429)
- KHÔNG ffill/fillna — trả về data gốc, để pipeline quyết định
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

if load_dotenv is not None and find_dotenv is not None:
    load_dotenv(find_dotenv(usecwd=True))

log = logging.getLogger(__name__)

_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
_RENAME = {
    "adjOpen":   "adj_open",
    "adjHigh":   "adj_high",
    "adjLow":    "adj_low",
    "adjClose":  "adj_close",
    "adjVolume": "adj_volume",
}
_KEEP = ["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]


def _api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Không tìm thấy TIINGO_API_KEY.\n"
            "Kiểm tra file .env có dòng: TIINGO_API_KEY=your_key_here\n"
            f"Thư mục hiện tại: {Path.cwd()}"
        )
    return key


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá index về naive UTC date + rename columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(s) for s in c).strip("_") for c in df.columns]

    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"

    df = df.rename(columns=_RENAME)
    keep = [c for c in _KEEP if c in df.columns]
    df = df[keep].sort_index()

    # Loại duplicate date (giữ bar cuối cùng)
    if df.index.duplicated().any():
        n_dup = df.index.duplicated().sum()
        log.warning(f"Có {n_dup} duplicate date → giữ bar cuối")
        df = df[~df.index.duplicated(keep="last")]

    return df


def _validate_cache(df: pd.DataFrame) -> bool:
    """Kiểm tra cache có đúng schema và dữ liệu cơ bản không."""
    if df is None or df.empty:
        return False
    if not all(c in df.columns for c in _KEEP):
        return False
    if df.index.name != "date":
        return False
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        return False
    if not df.index.is_unique:
        return False
    if not df.index.is_monotonic_increasing:
        return False
    if (df["adj_close"].dropna() <= 0).any():
        return False
    if (df["adj_volume"].dropna() < 0).any():
        return False
    return True


def fetch_ticker(
    ticker:    str,
    start:     str,
    end:       str,
    cache_dir: Path | None = None,
    retries:   int = 4,
) -> pd.DataFrame:
    """
    Fetch OHLCV adjusted của 1 ticker từ Tiingo.

    Returns
    -------
    DataFrame  index=date (naive UTC), columns=adj_open/high/low/close/volume
    Không fill NaN — pipeline quyết định xử lý missing data.
    """
    # ── Cache hit ──
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fp = cache_dir / f"{ticker}_{start}_{end}.parquet"
        if fp.exists():
            try:
                cached = pd.read_parquet(fp)
                if _validate_cache(cached):
                    log.debug(f"[{ticker}] cache hit ({len(cached)} ngày)")
                    return cached
                else:
                    log.warning(f"[{ticker}] cache invalid → fetch lại")
                    fp.unlink()
            except Exception as e:
                log.warning(f"[{ticker}] đọc cache lỗi: {e} → fetch lại")
                fp.unlink()

    # ── Fetch với retry ──
    params = {
        "startDate":    start,
        "endDate":      end,
        "resampleFreq": "daily",
        "token":        _api_key(),
    }
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                _URL.format(ticker=ticker),
                params=params,
                timeout=30,
            )

            if r.status_code == 429:
                wait = 2 ** attempt
                log.warning(f"[{ticker}] rate-limit → chờ {wait}s (attempt {attempt})")
                time.sleep(wait)
                continue

            r.raise_for_status()
            records = r.json()

            if not records:
                raise ValueError(f"Tiingo trả về list rỗng cho {ticker}")

            df = pd.DataFrame(records).set_index("date")
            df = _normalize(df)

            if not _validate_cache(df):
                raise ValueError(f"[{ticker}] data sau normalize không hợp lệ")

            log.info(
                f"[{ticker}] OK — {len(df)} ngày "
                f"({df.index[0].date()} → {df.index[-1].date()})"
            )

            if cache_dir is not None:
                df.to_parquet(fp)
                log.debug(f"[{ticker}] cached → {fp.name}")

            return df

        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403, 404}:
                raise RuntimeError(
                    f"[{ticker}] HTTP {status} - kiểm tra API key hoặc ticker"
                ) from exc
            wait = 2 ** attempt
            log.warning(f"[{ticker}] HTTP {status} | attempt {attempt}/{retries}: {exc}")
            time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning(f"[{ticker}] attempt {attempt}/{retries}: {exc}")
            time.sleep(wait)

    raise RuntimeError(
        f"[{ticker}] fetch thất bại sau {retries} lần.\n"
        f"Lỗi cuối: {last_exc}"
    )
