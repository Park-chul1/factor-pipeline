from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision


@dataclass(frozen=True)
class InfluxCfg:
    url: str
    token: str
    org: str
    bucket: str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="events CSV (ticker,effective_dt,in_index)")
    ap.add_argument("--url", default="http://localhost:8086")
    ap.add_argument("--org", default="quant_org")
    ap.add_argument("--bucket", default="quant_bucket")
    ap.add_argument("--token", required=True)
    ap.add_argument("--measurement", default="sp500_universe")
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.events)
    need = {"ticker", "effective_dt", "in_index"}
    if not need.issubset(df.columns):
        raise ValueError(f"events CSV must have columns {sorted(need)}")

    df["effective_dt"] = pd.to_datetime(df["effective_dt"], utc=True, errors="coerce")
    df = df.dropna(subset=["effective_dt", "ticker", "in_index"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["in_index"] = pd.to_numeric(df["in_index"], errors="coerce").astype(int)

    cfg = InfluxCfg(args.url, args.token, args.org, args.bucket)

    with InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org) as client:
        w = client.write_api()

        points = []
        for r in df.itertuples(index=False):
            p = (
                Point(args.measurement)
                .tag("ticker", r.ticker)
                .field("in_index", int(r.in_index))
                .time(r.effective_dt.to_pydatetime(), WritePrecision.S)
            )
            points.append(p)

            if len(points) >= args.batch:
                w.write(bucket=cfg.bucket, org=cfg.org, record=points)
                points.clear()

        if points:
            w.write(bucket=cfg.bucket, org=cfg.org, record=points)

    print(f"[OK] Ingested rows={len(df):,} into measurement={args.measurement}")


if __name__ == "__main__":
    main()


