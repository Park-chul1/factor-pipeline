from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from factors.universe_mask import build_universe_mask_from_annotated, summarize_universe_mask
from factors.io import download_yfinance_panel, align_panel_to_master, panel_diagnostics
from factors.raw_factors import compute_raw_factors, summarize_raw_factors
from factors.build_x import build_X, summarize_X
from factors.estimate_f import (
    build_forward_returns,
    estimate_factor_returns,
    summarize_factor_returns,
)


# =========================
# Config
# =========================
ANALYSIS_START = "2024-02-01"
DATA_START = "2023-01-01"
END = "2024-12-31"

ANNOTATED_CSV = str(PROJECT_ROOT / "db" / "annotated_sp500.csv")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MIN_NAMES = 30
LOWER_Q = 0.01
UPPER_Q = 0.99
RETURN_HORIZON = 1
RIDGE = 1e-6


def save_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def save_list_csv(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(values).to_csv(path, index=False)


def main() -> None:
    print("=" * 80)
    print("1) Build universe mask on analysis horizon")
    print("=" * 80)

    analysis_dates, tickers, universe_mask = build_universe_mask_from_annotated(
        annotated_csv=ANNOTATED_CSV,
        start=ANALYSIS_START,
        end=END,
    )

    universe_summary = summarize_universe_mask(analysis_dates, tickers, universe_mask)
    print("analysis dates:", len(analysis_dates))
    print("tickers:", len(tickers))
    print("universe_mask shape:", universe_mask.shape)
    print(
        "universe names min/max:",
        int(universe_summary["n_names"].min()),
        int(universe_summary["n_names"].max()),
    )

    active_tickers = sorted(np.array(tickers)[universe_mask.any(axis=0)].tolist())
    print("active tickers in analysis period:", len(active_tickers))

    print("\n" + "=" * 80)
    print("2) Download yfinance panel on extended data horizon")
    print("=" * 80)

    panel = download_yfinance_panel(
        tickers=active_tickers,
        start=DATA_START,
        end=END,
        progress=True,
    )

    print(panel_diagnostics(panel))

    print("\n" + "=" * 80)
    print("3) Align panel to extended master dates")
    print("=" * 80)

    # 가격/팩터 계산용 extended dates
    data_dates = pd.bdate_range(DATA_START, END)

    panel = align_panel_to_master(
        panel=panel,
        master_dates=data_dates,
        master_tickers=active_tickers,
    )

    print("aligned panel close shape:", panel["close"].shape)

    print("\n" + "=" * 80)
    print("4) Compute raw factors on extended horizon")
    print("=" * 80)

    raw_factors = compute_raw_factors(panel)
    raw_summary = summarize_raw_factors(raw_factors)

    for name, stats in raw_summary.items():
        print(f"\nRAW FACTOR: {name}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("5) Expand analysis universe mask onto extended dates")
    print("=" * 80)

    # extended date index 위에 analysis universe를 얹는다
    T_ext = len(data_dates)
    N = len(active_tickers)
    universe_mask_ext = np.zeros((T_ext, N), dtype=bool)

    data_idx = pd.Index(data_dates)
    analysis_pos = data_idx.get_indexer(analysis_dates)

    if (analysis_pos < 0).any():
        raise ValueError("Some analysis_dates are not found in extended data_dates.")

    # tickers 순서는 active_tickers로 재정렬
    ticker_to_old_j = {t: j for j, t in enumerate(tickers)}
    keep_idx = [ticker_to_old_j[t] for t in active_tickers]
    universe_mask_small = universe_mask[:, keep_idx]

    universe_mask_ext[analysis_pos] = universe_mask_small

    print("extended universe_mask shape:", universe_mask_ext.shape)

    print("\n" + "=" * 80)
    print("6) Build X on extended horizon")
    print("=" * 80)

    X_ext, valid_mask_ext, factor_names = build_X(
        raw_factors=raw_factors,
        universe_mask=universe_mask_ext,
        min_names=MIN_NAMES,
        lower_q=LOWER_Q,
        upper_q=UPPER_Q,
    )

    print("X_ext shape:", X_ext.shape)
    print("valid_mask_ext shape:", valid_mask_ext.shape)
    print("factor_names:", factor_names)
    print("valid frac ext:", float(valid_mask_ext.mean()))

    x_summary = summarize_X(X_ext, factor_names)
    for row in x_summary:
        print(row)

    print("\n" + "=" * 80)
    print("7) Build forward returns on extended horizon")
    print("=" * 80)

    r_ext = build_forward_returns(panel["adj_close"], horizon=RETURN_HORIZON)
    print("r_ext shape:", r_ext.shape)
    print("r_ext finite frac:", float(np.isfinite(r_ext).mean()))

    print("\n" + "=" * 80)
    print("8) Slice back to analysis horizon")
    print("=" * 80)

    X = X_ext[analysis_pos]
    r = r_ext[analysis_pos]
    valid_mask = valid_mask_ext[analysis_pos]
    universe_mask_final = universe_mask_ext[analysis_pos]
    dates = analysis_dates
    tickers_final = active_tickers

    print("X shape:", X.shape)
    print("r shape:", r.shape)
    print("valid_mask shape:", valid_mask.shape)
    print("universe_mask_final shape:", universe_mask_final.shape)

    print("\n" + "=" * 80)
    print("9) Estimate factor returns")
    print("=" * 80)

    f, day_valid = estimate_factor_returns(
        X=X,
        r=r,
        universe_mask=universe_mask_final,
        valid_mask=valid_mask,
        min_names=MIN_NAMES,
        ridge=RIDGE,
    )

    print("f shape:", f.shape)
    print("day_valid frac:", float(day_valid.mean()))
    print("factor return summary:", summarize_factor_returns(f))

    valid_idx = np.where(day_valid)[0]
    print("num valid days:", len(valid_idx))
    if len(valid_idx) > 0:
        print("first valid date:", dates[valid_idx[0]])
        print("first valid f row:", f[valid_idx[0]])

    print("\n" + "=" * 80)
    print("10) Save outputs")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_array(OUTPUT_DIR / "X.npy", X)
    save_array(OUTPUT_DIR / "r.npy", r)
    save_array(OUTPUT_DIR / "f.npy", f)
    save_array(OUTPUT_DIR / "universe_mask.npy", universe_mask_final)
    save_array(OUTPUT_DIR / "valid_mask.npy", valid_mask)
    save_array(OUTPUT_DIR / "day_valid.npy", day_valid)

    save_list_csv(OUTPUT_DIR / "dates.csv", dates.astype(str))
    save_list_csv(OUTPUT_DIR / "tickers.csv", tickers_final)
    save_list_csv(OUTPUT_DIR / "factor_names.csv", factor_names)

    universe_summary.to_csv(OUTPUT_DIR / "universe_summary.csv", index=False)

    f_df = pd.DataFrame(f, columns=factor_names)
    f_df.insert(0, "date", dates.astype(str))
    f_df["day_valid"] = day_valid
    f_df.to_csv(OUTPUT_DIR / "factor_returns.csv", index=False)

    print("Saved to:", OUTPUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
