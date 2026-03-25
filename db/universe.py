from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from influxdb_client import InfluxDBClient


@dataclass(frozen=True)
class InfluxCfg:
    url: str
    token: str
    org: str
    bucket: str


def _to_utc_dt(asof: datetime) -> datetime:
    """
    Ensure timezone-aware UTC datetime.
    - If naive datetime: assume UTC (best practice for DB timestamps).
    """
    if asof.tzinfo is None:
        return asof.replace(tzinfo=timezone.utc)
    return asof.astimezone(timezone.utc)


def get_universe_asof(
    cfg: InfluxCfg,
    asof: datetime,
    measurement: str = "sp500_universe",
    field: str = "in_index",
) -> List[str]:
    """
    Return tickers that are in the universe 'as of' asof datetime.

    Key trick:
    - Flux range stop is effectively exclusive in practice for boundary cases,
      so we query up to (asof + 1 day) and then take last() per ticker.
    """
    asof_utc = _to_utc_dt(asof)
    stop_utc = asof_utc + timedelta(days=1)

    # RFC3339 like: 2021-01-02T00:00:00Z
    stop_rfc3339 = stop_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    flux = f"""
from(bucket: "{cfg.bucket}")
  |> range(start: 0, stop: {stop_rfc3339})
  |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "{field}")
  |> group(columns: ["ticker"])
  |> last()
  |> filter(fn: (r) => r._value == 1)
  |> keep(columns: ["ticker"])
  |> distinct(column: "ticker")
"""

    with InfluxDBClient(url=cfg.url, token=cfg.token, org=cfg.org) as client:
        q = client.query_api()
        tables = q.query(flux)

    tickers: List[str] = []
    for table in tables:
        for record in table.records:
            t = record.values.get("ticker")
            if t:
                tickers.append(str(t))

    tickers.sort()
    return tickers

