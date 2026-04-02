from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_business_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """
    Business-day date index used as the master time axis.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("end must be >= start")
    return pd.date_range(start_ts, end_ts, freq="B")


def filter_nasdaq_common_stocks(
    tickers_df: pd.DataFrame,
    exchange_col: str = "primary_exchange",
    ticker_col: str = "ticker",
    type_col: str = "type",
) -> pd.DataFrame:
    """
    Keep only NASDAQ-listed common stocks from Massive ticker metadata.

    Expected Massive-like columns:
      - ticker
      - primary_exchange
      - type
      - list_date (optional)
      - delisted_utc (optional)

    Common stock type is usually 'CS'.
    NASDAQ primary exchange is usually 'XNAS'.
    """
    required = [ticker_col, exchange_col, type_col]
    missing = [c for c in required if c not in tickers_df.columns]
    if missing:
        raise ValueError(f"tickers_df missing required columns: {missing}")

    df = tickers_df.copy()

    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
    df[exchange_col] = df[exchange_col].astype(str).str.strip().str.upper()
    df[type_col] = df[type_col].astype(str).str.strip().str.upper()

    df = df[(df[exchange_col] == "XNAS") & (df[type_col] == "CS")].copy()

    # Optional date normalization
    if "list_date" in df.columns:
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce").dt.normalize()

    if "delisted_utc" in df.columns:
        df["delisted_utc"] = pd.to_datetime(df["delisted_utc"], errors="coerce").dt.normalize()

    # Drop duplicate tickers deterministically:
    # prefer rows with earlier list_date and later/no delisted date.
    sort_cols = []
    ascending = []

    if "list_date" in df.columns:
        sort_cols.append("list_date")
        ascending.append(True)

    if "delisted_utc" in df.columns:
        # NaT last by default with na_position='last'
        sort_cols.append("delisted_utc")
        ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last")

    df = df.drop_duplicates(subset=[ticker_col], keep="first").reset_index(drop=True)
    return df


def build_nasdaq_universe_mask(
    tickers_df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    ticker_col: str = "ticker",
    list_date_col: str = "list_date",
    delisted_col: str = "delisted_utc",
) -> tuple[list[str], pd.DatetimeIndex, np.ndarray]:
    """
    Build a date-by-ticker boolean universe mask from listing / delisting info.

    Rules:
    - if list_date is missing, assume active from global start
    - if delisted_utc is missing, assume active until global end
    - active interval is inclusive on both ends
    """
    if ticker_col not in tickers_df.columns:
        raise ValueError(f"tickers_df missing required column: {ticker_col}")

    df = tickers_df.copy()
    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    dates = build_business_dates(start_ts, end_ts)

    tickers = sorted(df[ticker_col].dropna().unique().tolist())
    ticker_to_j = {t: j for j, t in enumerate(tickers)}

    T = len(dates)
    N = len(tickers)
    universe_mask = np.zeros((T, N), dtype=bool)

    if list_date_col in df.columns:
        df[list_date_col] = pd.to_datetime(df[list_date_col], errors="coerce").dt.normalize()
    else:
        df[list_date_col] = pd.NaT

    if delisted_col in df.columns:
        df[delisted_col] = pd.to_datetime(df[delisted_col], errors="coerce").dt.normalize()
    else:
        df[delisted_col] = pd.NaT

    for _, row in df.iterrows():
        ticker = row[ticker_col]
        j = ticker_to_j[ticker]

        list_dt = row[list_date_col]
        delist_dt = row[delisted_col]

        if pd.isna(list_dt):
            list_dt = start_ts
        if pd.isna(delist_dt):
            delist_dt = end_ts

        # clip to requested window
        active_start = max(list_dt, start_ts)
        active_end = min(delist_dt, end_ts)

        if active_end < active_start:
            continue

        active = (dates >= active_start) & (dates <= active_end)
        universe_mask[active, j] = True

    return tickers, dates, universe_mask


def refine_universe_mask_with_panel_data(
    universe_mask: np.ndarray,
    panel: dict,
    price_field: str = "adj_close",
) -> np.ndarray:
    """
    Refine the metadata-based universe mask using actual price availability.

    Keeps a name in the universe only where panel[price_field] is finite.
    This is important because listing metadata and price coverage often do not
    align perfectly.

    Expected:
      panel[price_field] shape == (T, N)
    """
    if price_field not in panel:
        raise KeyError(f"panel missing field: {price_field}")

    px = np.asarray(panel[price_field])
    if px.ndim != 2:
        raise ValueError(f"panel[{price_field!r}] must be 2D, got shape={px.shape}")

    mask = np.asarray(universe_mask, dtype=bool)
    if mask.shape != px.shape:
        raise ValueError(
            f"shape mismatch: universe_mask {mask.shape} vs panel[{price_field!r}] {px.shape}"
        )

    has_price = np.isfinite(px)
    refined = mask & has_price
    return refined


def summarize_universe_mask(
    universe_mask: np.ndarray,
    dates: Iterable[pd.Timestamp] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Simple diagnostics: number of active names per date.
    """
    mask = np.asarray(universe_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("universe_mask must be 2D")

    dates_idx = pd.DatetimeIndex(dates)
    if len(dates_idx) != mask.shape[0]:
        raise ValueError(
            f"dates length {len(dates_idx)} does not match mask rows {mask.shape[0]}"
        )

    active_count = mask.sum(axis=1).astype(int)

    out = pd.DataFrame({
        "date": dates_idx,
        "n_active": active_count,
    })
    return out

def build_nasdaq_universe_mask_from_dates(
    tickers_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    ticker_col: str = "ticker",
    list_date_col: str = "list_date",
    delisted_col: str = "delisted_utc",
) -> tuple[list[str], pd.DatetimeIndex, np.ndarray]:
    if ticker_col not in tickers_df.columns:
        raise ValueError(f"tickers_df missing required column: {ticker_col}")

    df = tickers_df.copy()
    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()

    dates = pd.DatetimeIndex(dates).normalize().sort_values().unique()
    if len(dates) == 0:
        raise ValueError("dates is empty")

    start_ts = dates[0]
    end_ts = dates[-1]

    tickers = sorted(df[ticker_col].dropna().unique().tolist())
    ticker_to_j = {t: j for j, t in enumerate(tickers)}

    T = len(dates)
    N = len(tickers)
    universe_mask = np.zeros((T, N), dtype=bool)

    if list_date_col in df.columns:
        df[list_date_col] = pd.to_datetime(df[list_date_col], errors="coerce").dt.normalize()
    else:
        df[list_date_col] = pd.NaT

    if delisted_col in df.columns:
        df[delisted_col] = pd.to_datetime(df[delisted_col], errors="coerce").dt.normalize()
    else:
        df[delisted_col] = pd.NaT

    for _, row in df.iterrows():
        ticker = row[ticker_col]
        j = ticker_to_j[ticker]

        list_dt = row[list_date_col]
        delist_dt = row[delisted_col]

        if pd.isna(list_dt):
            list_dt = start_ts
        if pd.isna(delist_dt):
            delist_dt = end_ts

        active_start = max(list_dt, start_ts)
        active_end = min(delist_dt, end_ts)

        if active_end < active_start:
            continue

        active = (dates >= active_start) & (dates <= active_end)
        universe_mask[active, j] = True

    return tickers, dates, universe_mask