from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def build_business_dates(start: str, end: str) -> pd.DatetimeIndex:
    """
    Build business-day dates for the analysis horizon.
    """
    return pd.bdate_range(start=start, end=end)


def normalize_ticker_for_yf(ticker: str) -> str:
    """
    Convert internal ticker format to Yahoo Finance ticker format.
    Example:
      BRK.B -> BRK-B
    """
    return str(ticker).strip().replace(".", "-")


def _read_annotated_csv(annotated_csv: str) -> pd.DataFrame:
    """
    Read annotated_sp500.csv robustly.

    Supports:
    1) Plain CSV with header:
         ticker,effective_dt,in_index
         ticker,effective_date,in_index
    2) Headerless 3-column CSV
    3) Influx annotated CSV export with leading # rows
    """
    # Influx annotated CSV: skip lines starting with '#'
    df = pd.read_csv(annotated_csv, comment="#")

    # Normalize column names
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()

        if cl in {"ticker", "symbol", "tic"}:
            rename_map[c] = "ticker"

        elif cl in {
            "effective_date",
            "effective_dt",
            "date",
            "time",
            "asof",
            "_time",
        }:
            rename_map[c] = "effective_date"

        elif cl in {
            "in_index",
            "active",
            "member",
            "is_member",
            "value",
            "state",
            "_value",
        }:
            rename_map[c] = "in_index"

    df = df.rename(columns=rename_map)

    required = {"ticker", "effective_date", "in_index"}
    if required.issubset(df.columns):
        return df[["ticker", "effective_date", "in_index"]].copy()

    # Fallback: headerless 3-column
    df2 = pd.read_csv(
        annotated_csv,
        comment="#",
        header=None,
        names=["ticker", "effective_date", "in_index"],
    )
    return df2


def load_annotated_events(annotated_csv: str) -> pd.DataFrame:
    """
    Load and clean event-style S&P500 membership annotations.

    Returns a DataFrame with columns:
      ticker: str
      effective_date: Timestamp (UTC-naive after normalization)
      in_index: int (0 or 1)
    """
    df = _read_annotated_csv(annotated_csv).copy()

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["effective_date"] = pd.to_datetime(df["effective_date"], format="%Y-%m-%dT%H:%M:%S%z", utc=True, errors="coerce")
    df["in_index"] = pd.to_numeric(df["in_index"], errors="coerce")

    df = df.dropna(subset=["ticker", "effective_date", "in_index"]).copy()
    df["in_index"] = df["in_index"].astype(int)

    # Normalize to naive timestamps after UTC conversion, so comparisons are consistent
    df["effective_date"] = df["effective_date"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Keep only 0/1 rows
    df = df[df["in_index"].isin([0, 1])].copy()

    # Sort for forward-fill style state propagation
    df = df.sort_values(["ticker", "effective_date"]).reset_index(drop=True)

    return df


def build_master_tickers(events: pd.DataFrame) -> List[str]:
    """
    Sorted unique ticker list appearing in the event file.
    """
    return sorted(events["ticker"].dropna().astype(str).unique().tolist())


def build_universe_mask_from_annotated(
    annotated_csv: str,
    start: str,
    end: str,
) -> Tuple[pd.DatetimeIndex, List[str], np.ndarray]:
    """
    Build universe mask [T, N] from event-style annotated CSV.

    Logic:
    - For each ticker, events define changes of membership state over time.
    - At each date t, ticker is in universe iff the latest event on or before t has in_index == 1.

    Returns:
      dates: DatetimeIndex [T]
      tickers: list[str] [N]
      universe_mask: bool ndarray [T, N]
    """
    events = load_annotated_events(annotated_csv)
    dates = build_business_dates(start, end)

    # Convert dates to naive timestamps for consistent comparison
    date_values = pd.DatetimeIndex(dates).tz_localize(None)
    tickers = build_master_tickers(events)

    T = len(date_values)
    N = len(tickers)
    mask = np.zeros((T, N), dtype=bool)

    for j, ticker in enumerate(tickers):
        sub = events.loc[events["ticker"] == ticker, ["effective_date", "in_index"]].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("effective_date")
        ev_dates = sub["effective_date"].to_numpy(dtype="datetime64[ns]")
        ev_vals = sub["in_index"].to_numpy(dtype=np.int8)

        # For each analysis date, find last event index <= date
        idx = np.searchsorted(ev_dates, date_values.to_numpy(dtype="datetime64[ns]"), side="right") - 1

        valid = idx >= 0
        state = np.zeros(T, dtype=bool)
        state[valid] = ev_vals[idx[valid]] == 1
        mask[:, j] = state

    return dates, tickers, mask


def summarize_universe_mask(
    dates: pd.DatetimeIndex,
    tickers: List[str],
    universe_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Quick diagnostics:
      date, number of names in universe
    """
    return pd.DataFrame(
        {
            "date": dates,
            "n_names": universe_mask.sum(axis=1),
        }
    )


def build_yf_tickers_from_master(master_tickers: List[str]) -> List[str]:
    """
    Convert internal tickers to yfinance-compatible tickers.
    """
    return [normalize_ticker_for_yf(t) for t in master_tickers]
