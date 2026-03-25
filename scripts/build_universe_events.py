from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def _to_utc_midnight(dt: pd.Timestamp) -> pd.Timestamp:
    # Normalize to UTC midnight (consistent with your test data)
    if pd.isna(dt):
        return dt
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return dt.normalize()


def build_events_from_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected (case-insensitive):
      - ticker
      - start (or start_date, in_date)
      - end   (or end_date, out_date) [optional]

    Output:
      - ticker, effective_dt (UTC), in_index (int 0/1)
    """
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        raise ValueError("Input must have a 'ticker' column.")

    # accept aliases
    start_col = cols.get("start") or cols.get("start_date") or cols.get("in_date") or cols.get("in")
    end_col = cols.get("end") or cols.get("end_date") or cols.get("out_date") or cols.get("out")

    if not start_col:
        raise ValueError("Input ranges must have a start column: start/start_date/in_date")

    out = []

    x = df.copy()
    x["ticker"] = x[cols["ticker"]].astype(str).str.strip()

    x["_start"] = pd.to_datetime(x[start_col], errors="coerce", utc=True)
    x["_start"] = x["_start"].map(_to_utc_midnight)

    if end_col:
        x["_end"] = pd.to_datetime(x[end_col], errors="coerce", utc=True)
        x["_end"] = x["_end"].map(_to_utc_midnight)
    else:
        x["_end"] = pd.NaT

    # start events
    s = x.loc[~x["_start"].isna(), ["ticker", "_start"]].rename(columns={"_start": "effective_dt"})
    s["in_index"] = 1
    out.append(s)

    # end events (if present)
    e = x.loc[~x["_end"].isna(), ["ticker", "_end"]].rename(columns={"_end": "effective_dt"})
    if not e.empty:
        e["in_index"] = 0
        out.append(e)

    ev = pd.concat(out, ignore_index=True).dropna(subset=["effective_dt"])
    ev = ev.sort_values(["ticker", "effective_dt", "in_index"]).drop_duplicates()
    return ev.reset_index(drop=True)


def build_events_from_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected (case-insensitive):
      - ticker
      - date (or effective_dt)
      - action (add/remove) OR in_index (0/1)

    Output: ticker, effective_dt (UTC), in_index (int 0/1)
    """
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        raise ValueError("Input must have a 'ticker' column.")
    date_col = cols.get("date") or cols.get("effective_dt") or cols.get("time")
    if not date_col:
        raise ValueError("Input actions must have a date/effective_dt column.")

    x = df.copy()
    x["ticker"] = x[cols["ticker"]].astype(str).str.strip()

    x["effective_dt"] = pd.to_datetime(x[date_col], errors="coerce", utc=True)
    x["effective_dt"] = x["effective_dt"].map(_to_utc_midnight)

    if "in_index" in cols:
        x["in_index"] = pd.to_numeric(x[cols["in_index"]], errors="coerce").astype("Int64")
    else:
        action_col = cols.get("action") or cols.get("event") or cols.get("type")
        if not action_col:
            raise ValueError("Need either in_index column or action(add/remove) column.")
        a = x[action_col].astype(str).str.lower().str.strip()
        x["in_index"] = a.map({"add": 1, "in": 1, "enter": 1, "include": 1, "remove": 0, "out": 0, "exit": 0, "exclude": 0})

    ev = x[["ticker", "effective_dt", "in_index"]].dropna(subset=["effective_dt", "in_index"])
    ev["in_index"] = ev["in_index"].astype(int)
    ev = ev.sort_values(["ticker", "effective_dt", "in_index"]).drop_duplicates()
    return ev.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV path from Norgate (or other source)")
    ap.add_argument("--mode", choices=["ranges", "actions"], required=True, help="Input schema type")
    ap.add_argument("--output", required=True, help="Output events CSV path")
    args = ap.parse_args()

    inp = Path(args.input)
    df = pd.read_csv(inp)

    if args.mode == "ranges":
        ev = build_events_from_ranges(df)
    else:
        ev = build_events_from_actions(df)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(outp, index=False)
    print(f"[OK] events rows={len(ev):,} -> {outp}")


if __name__ == "__main__":
    main()


    
