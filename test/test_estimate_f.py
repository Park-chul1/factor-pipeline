import numpy as np

from factors.io import download_yfinance_panel
from factors.raw_factors import compute_raw_factors
from factors.build_x import build_X
from factors.estimate_f import (
    build_forward_returns,
    estimate_factor_returns,
    summarize_factor_returns,
)


def main():
    tickers = ["AAPL", "MSFT", "BRK.B"]

    panel = download_yfinance_panel(
        tickers=tickers,
        start="2023-01-01",
        end="2024-12-31",
        progress=False,
    )

    raw = compute_raw_factors(panel)

    T, N = panel["adj_close"].shape
    universe_mask = np.ones((T, N), dtype=bool)

    X, valid_mask, factor_names = build_X(
        raw_factors=raw,
        universe_mask=universe_mask,
        min_names=2,   # 테스트용
        lower_q=0.01,
        upper_q=0.99,
    )

    r = build_forward_returns(panel["adj_close"], horizon=1)

    f, day_valid = estimate_factor_returns(
        X=X,
        r=r,
        universe_mask=universe_mask,
        valid_mask=valid_mask,
        min_names=2,   # 테스트용
        ridge=1e-6,
    )

    print("X shape:", X.shape)
    print("r shape:", r.shape)
    print("f shape:", f.shape)
    print("factor_names:", factor_names)
    print("valid day frac:", day_valid.mean())
    print("factor return summary:", summarize_factor_returns(f))

    print("\nfirst 5 valid day indices:", np.where(day_valid)[0][:5])
    print("first valid f row:")
    idx = np.where(day_valid)[0]
    if len(idx) > 0:
        print(f[idx[0]])
    else:
        print("No valid days")


if __name__ == "__main__":
    main()
