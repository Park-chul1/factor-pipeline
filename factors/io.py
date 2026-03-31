from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import yfinance as yf


PRICE_KEYS = ["open", "high", "low", "close", "adj_close", "volume"]
MASSIVE_BASE_URL = "https://api.massive.com"
MASSIVE_API_KEY_ENV_CANDIDATES = ["MASSIVE_API_KEY", "POLYGON_API_KEY"]


def normalize_ticker_for_yf(ticker: str) -> str:
    """
    Convert internal ticker format to Yahoo Finance ticker format.
    Examples:
      BRK.B -> BRK-B
      BF.B  -> BF-B
    """
    return ticker.replace(".", "-")


def denormalize_ticker_from_yf(ticker: str) -> str:
    """
    Optional reverse mapping if needed later.
    """
    return ticker.replace("-", ".")


def build_ticker_mapping(master_tickers: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      yf_tickers: tickers for yfinance query order-aligned with master_tickers
      yf_to_master: maps yf ticker -> original master ticker
    """
    yf_tickers = [normalize_ticker_for_yf(t) for t in master_tickers]
    yf_to_master = {yf_t: orig for yf_t, orig in zip(yf_tickers, master_tickers)}
    return yf_tickers, yf_to_master


def _empty_panel(T: int, N: int) -> Dict[str, np.ndarray]:
    return {
        "open": np.full((T, N), np.nan, dtype=np.float64),
        "high": np.full((T, N), np.nan, dtype=np.float64),
        "low": np.full((T, N), np.nan, dtype=np.float64),
        "close": np.full((T, N), np.nan, dtype=np.float64),
        "adj_close": np.full((T, N), np.nan, dtype=np.float64),
        "volume": np.full((T, N), np.nan, dtype=np.float64),
    }


# -----------------------------------------------------------------------------
# yfinance helpers (existing behavior preserved)
# -----------------------------------------------------------------------------


def _extract_field_from_download(
    df: pd.DataFrame,
    ticker: str,
    field: str,
) -> pd.Series | None:
    """
    Handle yfinance multi-index output robustly.
    Expected common structure:
      columns = MultiIndex[(ticker, field)]
    """
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)

        if ticker in lvl0:
            sub = df[ticker]
            if field in sub.columns:
                return sub[field]

        matches = [(a, b) for a, b in df.columns if a == ticker and b == field]
        if matches:
            return df[matches[0]]

    else:
        if field in df.columns:
            return df[field]

    return None


def download_yfinance_panel(
    tickers: List[str],
    start: str,
    end: str,
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Download OHLCV data from yfinance and return panel aligned to input ticker order.

    Returns:
      {
        "dates": DatetimeIndex,
        "tickers": original input tickers,
        "yf_tickers": yahoo-format tickers,
        "open": [T,N],
        "high": [T,N],
        "low": [T,N],
        "close": [T,N],
        "adj_close": [T,N],
        "volume": [T,N],
      }
    """
    if len(tickers) == 0:
        raise ValueError("tickers is empty")

    master_tickers = list(tickers)
    yf_tickers, _ = build_ticker_mapping(master_tickers)

    df = yf.download(
        tickers=yf_tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=progress,
        group_by="ticker",
        threads=True,
    )

    if len(yf_tickers) == 1 and not isinstance(df.columns, pd.MultiIndex):
        df = pd.concat({yf_tickers[0]: df}, axis=1)

    if df is None or len(df) == 0:
        raise ValueError("yfinance returned empty dataframe")

    dates = pd.DatetimeIndex(df.index).sort_values()
    T = len(dates)
    N = len(master_tickers)

    panel = _empty_panel(T, N)
    panel["dates"] = dates
    panel["tickers"] = master_tickers
    panel["yf_tickers"] = yf_tickers

    field_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }

    for j, yf_ticker in enumerate(yf_tickers):
        for yf_field, out_key in field_map.items():
            s = _extract_field_from_download(df, yf_ticker, yf_field)
            if s is None:
                continue
            s = s.reindex(dates)
            panel[out_key][:, j] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    return panel


# -----------------------------------------------------------------------------
# Massive helpers
# -----------------------------------------------------------------------------


def _resolve_massive_api_key(api_key: str | None = None) -> str:
    if api_key:
        return api_key

    for env_name in MASSIVE_API_KEY_ENV_CANDIDATES:
        value = os.getenv(env_name)
        if value:
            return value

    raise ValueError(
        "Massive API key not found. Set MASSIVE_API_KEY (or POLYGON_API_KEY) "
        "or pass api_key explicitly."
    )


def _normalize_date_like(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _normalize_massive_bar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Massive OHLCV rows to a common schema.

    Input may come from either:
      - custom bars endpoint (ticker omitted in each row)
      - grouped daily endpoint (ticker column typically named T)

    Output columns:
      ticker, date, open, high, low, close, adj_close, volume, vwap, n_trades, timestamp_ms
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "vwap",
                "n_trades",
                "timestamp_ms",
            ]
        )

    out = df.copy()

    rename_map = {
        "T": "ticker",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "n": "n_trades",
        "t": "timestamp_ms",
    }
    out = out.rename(columns=rename_map)

    for needed in ["open", "high", "low", "close", "volume", "vwap", "n_trades", "timestamp_ms"]:
        if needed not in out.columns:
            out[needed] = np.nan

    if "ticker" not in out.columns:
        out["ticker"] = None

    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
    ts = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    out["date"] = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

    numeric_cols = ["open", "high", "low", "close", "volume", "vwap", "n_trades"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["adj_close"] = out["close"]

    keep_cols = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "vwap",
        "n_trades",
        "timestamp_ms",
    ]
    out = out[keep_cols].sort_values(["date", "ticker"], na_position="last").reset_index(drop=True)
    return out


def _massive_get(
    path_or_url: str,
    params: dict | None = None,
    api_key: str | None = None,
    base_url: str = MASSIVE_BASE_URL,
    timeout: float = 30.0,
    max_retries: int = 3,
    sleep_s: float = 0.75,
    session: requests.Session | None = None,
) -> dict:
    """
    Perform one Massive GET request and return parsed JSON.

    `path_or_url` can be either a relative path like `/v3/reference/tickers`
    or a fully-qualified `next_url` returned by the API.
    """
    key = _resolve_massive_api_key(api_key)
    sess = session or requests.Session()

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        url = path_or_url
    else:
        url = f"{base_url.rstrip('/')}/{path_or_url.lstrip('/')}"

    final_params = dict(params or {})
    final_params.setdefault("apiKey", key)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = sess.get(url, params=final_params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()

            if isinstance(payload, dict) and payload.get("status") == "ERROR":
                raise RuntimeError(f"Massive API returned error payload: {payload}")

            return payload
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_retries:
                break
            time.sleep(sleep_s * attempt)

    raise RuntimeError(f"Massive request failed for {url}: {last_exc}")


def _massive_get_paginated(
    path_or_url: str,
    params: dict | None = None,
    api_key: str | None = None,
    base_url: str = MASSIVE_BASE_URL,
    results_key: str = "results",
    timeout: float = 30.0,
    max_retries: int = 3,
    sleep_s: float = 0.75,
    session: requests.Session | None = None,
) -> list[dict]:
    """
    Follow Massive pagination via `next_url` and concatenate `results`.
    """
    key = _resolve_massive_api_key(api_key)
    sess = session or requests.Session()

    items: list[dict] = []
    next_target: str | None = path_or_url
    next_params: dict | None = dict(params or {})

    while next_target:
        payload = _massive_get(
            path_or_url=next_target,
            params=next_params,
            api_key=key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            sleep_s=sleep_s,
            session=sess,
        )

        batch = payload.get(results_key, [])
        if isinstance(batch, list):
            items.extend(batch)
        elif isinstance(batch, dict):
            items.append(batch)

        next_url = payload.get("next_url")
        if not next_url:
            break

        if "apiKey=" not in next_url and "apikey=" not in next_url:
            sep = "&" if "?" in next_url else "?"
            next_url = f"{next_url}{sep}{urlencode({'apiKey': key})}"

        next_target = next_url
        next_params = None

    return items


def download_massive_tickers(
    market: str = "stocks",
    exchange: str | None = None,
    active: bool | None = None,
    date: str | None = None,
    ticker_type: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Download reference tickers from Massive /v3/reference/tickers.

    Returned columns are preserved from the API where available, with common
    convenience aliases:
      - ticker
      - primary_exchange
      - type
      - active
      - delisted_utc
      - market
      - locale
    """
    params: dict[str, Any] = {
        "market": market,
        "limit": int(limit),
    }
    if exchange is not None:
        params["exchange"] = exchange
    if active is not None:
        params["active"] = str(active).lower()
    if date is not None:
        params["date"] = str(date)
    if ticker_type is not None:
        params["type"] = ticker_type

    rows = _massive_get_paginated(
        path_or_url="/v3/reference/tickers",
        params=params,
        api_key=api_key,
        session=session,
    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "name",
                "market",
                "locale",
                "primary_exchange",
                "type",
                "active",
                "currency_name",
                "cik",
                "composite_figi",
                "share_class_figi",
                "last_updated_utc",
                "delisted_utc",
                "source_feed",
            ]
        )

    df = pd.DataFrame(rows)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip()

    for col in ["last_updated_utc", "delisted_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "active" in df.columns:
        df["active"] = df["active"].astype("boolean")

    df = df.sort_values([c for c in ["ticker", "active", "delisted_utc"] if c in df.columns]).reset_index(drop=True)
    return df


def download_massive_ticker_overview(
    ticker: str,
    date: str | None = None,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Download a single ticker overview as a one-row DataFrame.
    """
    params: dict[str, Any] = {}
    if date is not None:
        params["date"] = str(date)

    payload = _massive_get(
        path_or_url=f"/v3/reference/tickers/{ticker}",
        params=params,
        api_key=api_key,
        session=session,
    )
    result = payload.get("results", {})
    if not result:
        return pd.DataFrame()

    df = pd.DataFrame([result])
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.strip()
    for col in ["list_date", "delisted_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=False, errors="coerce")
    return df


def download_massive_daily_bars(
    ticker: str,
    start: str,
    end: str,
    adjusted: bool = True,
    limit: int = 50000,
    sort: str = "asc",
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Download one ticker's daily OHLCV bars from Massive custom bars endpoint.

    Returns normalized columns:
      ticker, date, open, high, low, close, adj_close, volume, vwap, n_trades, timestamp_ms
    """
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": sort,
        "limit": int(limit),
    }

    rows = _massive_get_paginated(
        path_or_url=f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        params=params,
        api_key=api_key,
        session=session,
    )

    if not rows:
        return _normalize_massive_bar_frame(pd.DataFrame()).assign(ticker=ticker)

    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    out = _normalize_massive_bar_frame(df)

    if out.empty:
        return out

    out = out.sort_values(["date", "timestamp_ms"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    out = out.reset_index(drop=True)
    return out


def download_massive_panel(
    tickers: List[str],
    start: str,
    end: str,
    adjusted: bool = True,
    api_key: str | None = None,
    sleep_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Download daily bars ticker-by-ticker from Massive and build a [T, N] panel.

    Returns:
      {
        "dates": DatetimeIndex,
        "tickers": input tickers,
        "open": [T,N],
        "high": [T,N],
        "low": [T,N],
        "close": [T,N],
        "adj_close": [T,N],
        "volume": [T,N],
        "vwap": [T,N],
        "n_trades": [T,N],
      }
    """
    if len(tickers) == 0:
        raise ValueError("tickers is empty")

    sess = requests.Session()
    bars_by_ticker: dict[str, pd.DataFrame] = {}
    all_dates: list[pd.DatetimeIndex] = []

    for idx, ticker in enumerate(tickers):
        bars = download_massive_daily_bars(
            ticker=ticker,
            start=start,
            end=end,
            adjusted=adjusted,
            api_key=api_key,
            session=sess,
        )
        bars_by_ticker[ticker] = bars
        if not bars.empty:
            all_dates.append(pd.DatetimeIndex(bars["date"]))

        if sleep_s > 0.0 and idx + 1 < len(tickers):
            time.sleep(sleep_s)

    if not all_dates:
        raise ValueError("Massive returned no daily bars for the requested tickers/date range")

    dates = pd.DatetimeIndex(sorted(pd.Index(np.concatenate([d.to_numpy() for d in all_dates])).unique()))
    T = len(dates)
    N = len(tickers)

    panel = _empty_panel(T, N)
    panel["vwap"] = np.full((T, N), np.nan, dtype=np.float64)
    panel["n_trades"] = np.full((T, N), np.nan, dtype=np.float64)
    panel["dates"] = dates
    panel["tickers"] = list(tickers)

    row_index = pd.Index(dates)
    for j, ticker in enumerate(tickers):
        bars = bars_by_ticker[ticker]
        if bars.empty:
            continue

        bars = bars.set_index("date").sort_index()
        idx = row_index.get_indexer(pd.DatetimeIndex(bars.index))
        valid = idx >= 0
        dst_rows = idx[valid]
        src = bars.iloc[np.where(valid)[0]]

        for key in PRICE_KEYS:
            if key in src.columns:
                panel[key][dst_rows, j] = pd.to_numeric(src[key], errors="coerce").to_numpy(dtype=np.float64)

        for key in ["vwap", "n_trades"]:
            if key in src.columns:
                panel[key][dst_rows, j] = pd.to_numeric(src[key], errors="coerce").to_numpy(dtype=np.float64)

    return panel


def download_massive_grouped_daily(
    date: str,
    adjusted: bool = True,
    include_otc: bool = False,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Download one day's grouped U.S. stock OHLCV from Massive.
    """
    params = {
        "adjusted": str(adjusted).lower(),
        "include_otc": str(include_otc).lower(),
    }
    payload = _massive_get(
        path_or_url=f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
        params=params,
        api_key=api_key,
        session=session,
    )

    rows = payload.get("results", [])
    df = pd.DataFrame(rows)
    return _normalize_massive_bar_frame(df)


def download_massive_grouped_daily_range(
    dates: pd.DatetimeIndex,
    adjusted: bool = True,
    include_otc: bool = False,
    api_key: str | None = None,
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """
    Download grouped daily bars for multiple dates and return a long table.
    """
    if len(dates) == 0:
        raise ValueError("dates is empty")

    sess = requests.Session()
    chunks: list[pd.DataFrame] = []

    for idx, dt in enumerate(pd.DatetimeIndex(dates)):
        date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        day_df = download_massive_grouped_daily(
            date=date_str,
            adjusted=adjusted,
            include_otc=include_otc,
            api_key=api_key,
            session=sess,
        )
        if not day_df.empty:
            chunks.append(day_df)

        if sleep_s > 0.0 and idx + 1 < len(dates):
            time.sleep(sleep_s)

    if not chunks:
        return _normalize_massive_bar_frame(pd.DataFrame())

    out = pd.concat(chunks, axis=0, ignore_index=True)
    out = out.sort_values(["date", "ticker", "timestamp_ms"]).drop_duplicates(
        subset=["date", "ticker"],
        keep="last",
    )
    return out.reset_index(drop=True)


def massive_long_to_panel(
    bars_long: pd.DataFrame,
    master_dates: pd.DatetimeIndex,
    master_tickers: List[str],
) -> Dict[str, Any]:
    """
    Convert long-form Massive OHLCV rows to the panel schema used elsewhere.
    """
    T = len(master_dates)
    N = len(master_tickers)

    panel = _empty_panel(T, N)
    panel["vwap"] = np.full((T, N), np.nan, dtype=np.float64)
    panel["n_trades"] = np.full((T, N), np.nan, dtype=np.float64)
    panel["dates"] = pd.DatetimeIndex(master_dates)
    panel["tickers"] = list(master_tickers)

    if bars_long is None or bars_long.empty:
        return panel

    bars = bars_long.copy()
    bars["ticker"] = bars["ticker"].astype(str).str.strip()
    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars = bars.dropna(subset=["ticker", "date"])
    bars = bars.sort_values(["date", "ticker", "timestamp_ms"], na_position="last")
    bars = bars.drop_duplicates(subset=["date", "ticker"], keep="last")

    date_pos = {d: i for i, d in enumerate(pd.DatetimeIndex(master_dates))}
    ticker_pos = {t: j for j, t in enumerate(master_tickers)}

    for row in bars.itertuples(index=False):
        i = date_pos.get(pd.Timestamp(row.date))
        j = ticker_pos.get(str(row.ticker))
        if i is None or j is None:
            continue

        for field in PRICE_KEYS + ["vwap", "n_trades"]:
            if hasattr(row, field):
                value = getattr(row, field)
                if pd.notna(value):
                    panel[field][i, j] = float(value)

    return panel


# -----------------------------------------------------------------------------
# Shared panel helpers
# -----------------------------------------------------------------------------


def align_panel_to_master(
    panel: Dict[str, Any],
    master_dates: pd.DatetimeIndex,
    master_tickers: List[str],
) -> Dict[str, Any]:
    """
    Reindex panel arrays to the requested master_dates and master_tickers.

    Assumes panel["tickers"] are original/internal ticker names.
    """
    src_dates = pd.DatetimeIndex(panel["dates"])
    src_tickers = list(panel["tickers"])

    T = len(master_dates)
    N = len(master_tickers)

    out = _empty_panel(T, N)
    out["dates"] = pd.DatetimeIndex(master_dates)
    out["tickers"] = list(master_tickers)
    out["yf_tickers"] = [normalize_ticker_for_yf(t) for t in master_tickers]

    extra_keys = [k for k, v in panel.items() if isinstance(v, np.ndarray) and v.ndim == 2 and k not in PRICE_KEYS]
    for key in extra_keys:
        out[key] = np.full((T, N), np.nan, dtype=np.float64)

    date_indexer = pd.Index(src_dates).get_indexer(master_dates)
    src_ticker_to_j = {t: j for j, t in enumerate(src_tickers)}

    array_keys = [k for k, v in panel.items() if isinstance(v, np.ndarray) and v.ndim == 2]
    for j_new, tkr in enumerate(master_tickers):
        j_old = src_ticker_to_j.get(tkr, -1)
        if j_old == -1:
            continue

        valid_dates = date_indexer >= 0
        src_rows = date_indexer[valid_dates]
        dst_rows = np.where(valid_dates)[0]

        for key in array_keys:
            out[key][dst_rows, j_new] = panel[key][src_rows, j_old]

    return out


def build_dollar_volume(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Dollar volume proxy.
    """
    out = np.full_like(close, np.nan, dtype=np.float64)
    mask = np.isfinite(close) & np.isfinite(volume)
    out[mask] = close[mask] * volume[mask]
    return out


def panel_diagnostics(panel: Dict[str, Any]) -> pd.DataFrame:
    """
    Quick sanity-check summary per field.
    """
    rows = []
    for key, value in panel.items():
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            continue
        rows.append(
            {
                "field": key,
                "shape": value.shape,
                "nan_frac": float(np.isnan(value).mean()),
                "finite_count": int(np.isfinite(value).sum()),
            }
        )
    return pd.DataFrame(rows)

