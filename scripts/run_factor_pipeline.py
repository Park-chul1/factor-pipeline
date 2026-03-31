from __future__ import annotations

import os
import numpy as np
import pandas as pd

from factors.io import (
    download_massive_tickers,
    download_massive_grouped_daily_range,
    massive_long_to_panel,
)

from factors.universe_mask import (
    filter_nasdaq_common_stocks,
    build_nasdaq_universe_mask,
    refine_universe_mask_with_panel_data,
)
from factors.raw_factors import compute_raw_factors

from factors.build_x import build_X
from factors.estimate_f import estimate_factor_returns
from factors.io import align_panel_to_master


MIN_NAMES = 50
LOWER_Q = 0.01
UPPER_Q = 0.99
RIDGE = 1e-8


# ==============================
# CONFIG
# ==============================

START = "2019-01-01"
END = "2024-12-31"

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================
# 1. LOAD NASDAQ UNIVERSE
# ==============================

print("Downloading tickers...")

tickers_df = download_massive_tickers(
    market="stocks",
    exchange="XNAS",
    active=None,  # active + delisted 둘 다
)

print("Building NASDAQ universe mask...")

tickers_df = filter_nasdaq_common_stocks(tickers_df)

tickers, dates, universe_mask = build_nasdaq_universe_mask(
    tickers_df,
    start=START,
    end=END,
)

print(f"Universe size: {len(tickers)}")


# ==============================
# 2. DOWNLOAD PRICE DATA (GROUPED)
# ==============================

print("Downloading grouped daily bars...")

bars_long = download_massive_grouped_daily_range(dates)


# ==============================
# 3. LONG → PANEL
# ==============================

print("Building panel...")

panel = massive_long_to_panel(
    bars_long,
    master_dates=dates,
    master_tickers=tickers,
)

panel = align_panel_to_master(
    panel,
    master_dates=dates,
    master_tickers=tickers,
)

universe_mask = refine_universe_mask_with_panel_data(
    universe_mask,
    panel,
    price_field="adj_close",
)


# ==============================
# 4. FACTORS
# ==============================

print("Computing raw factors...")

raw = compute_raw_factors(panel)


# ==============================
# 5. BUILD X
# ==============================

print("Building X tensor...")
X, valid_mask, factor_names = build_X(
    raw_factors=raw,
    universe_mask=universe_mask,
    min_names=MIN_NAMES,
    lower_q=LOWER_Q,
    upper_q=UPPER_Q,
)

# ==============================
# 6. FORWARD RETURNS
# ==============================

print("Building forward returns...")

adj_close = panel["adj_close"]

r = np.full_like(adj_close, np.nan)

horizon = 1
r[:-horizon] = adj_close[horizon:] / adj_close[:-horizon] - 1


# ==============================
# 7. ESTIMATE FACTOR RETURNS
# ==============================

print("Estimating factor returns...")

f, day_valid = estimate_factor_returns(
    X=X,
    r=r,
    universe_mask=universe_mask,
    valid_mask=valid_mask,
    min_names=MIN_NAMES,
    ridge=RIDGE,
)

# ==============================
# 8. SAVE
# ==============================

print("Saving outputs...")

np.save(f"{OUT_DIR}/X.npy", X)
np.save(f"{OUT_DIR}/r.npy", r)
np.save(f"{OUT_DIR}/f.npy", f)

np.save(f"{OUT_DIR}/universe_mask.npy", universe_mask)

pd.Series(tickers).to_csv(f"{OUT_DIR}/tickers.csv", index=False)
pd.Series(dates).to_csv(f"{OUT_DIR}/dates.csv", index=False)

print("Done.")
