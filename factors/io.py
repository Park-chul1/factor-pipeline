from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


PRICE_KEYS = ["open", "high", "low", "close", "adj_close", "volume"]


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
        lvl1 = df.columns.get_level_values(1)

        if ticker in lvl0:
            sub = df[ticker]
            if field in sub.columns:
                return sub[field]

        # sometimes group_by or ordering can differ
        matches = [(a, b) for a, b in df.columns if a == ticker and b == field]
        if matches:
            return df[matches[0]]

    else:
        # single ticker case may return flat columns
        if field in df.columns:
            return df[field]

    return None


def download_yfinance_panel(
    tickers: List[str],
    start: str,
    end: str,
    progress: bool = True,
) -> Dict:
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

    # single ticker safeguard
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


def align_panel_to_master(
    panel: Dict,
    master_dates: pd.DatetimeIndex,
    master_tickers: List[str],
) -> Dict:
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

    date_indexer = pd.Index(src_dates).get_indexer(master_dates)
    src_ticker_to_j = {t: j for j, t in enumerate(src_tickers)}

    for j_new, tkr in enumerate(master_tickers):
        j_old = src_ticker_to_j.get(tkr, -1)
        if j_old == -1:
            continue

        valid_dates = date_indexer >= 0
        src_rows = date_indexer[valid_dates]
        dst_rows = np.where(valid_dates)[0]

        for key in PRICE_KEYS:
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


def panel_diagnostics(panel: Dict) -> pd.DataFrame:
    """
    Quick sanity-check summary per field.
    """
    rows = []
    for key in PRICE_KEYS:
        a = panel[key]
        rows.append(
            {
                "field": key,
                "shape": a.shape,
                "nan_frac": float(np.isnan(a).mean()),
                "finite_count": int(np.isfinite(a).sum()),
            }
        )
    return pd.DataFrame(rows)
