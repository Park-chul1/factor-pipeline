import numpy as np

from factors.io import download_yfinance_panel
from factors.raw_factors import compute_raw_factors
from factors.build_x import build_X, summarize_X


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

    print("X shape:", X.shape)
    print("valid_mask shape:", valid_mask.shape)
    print("factor_names:", factor_names)
    print("valid frac:", valid_mask.mean())

    rows = summarize_X(X, factor_names)
    for row in rows:
        print(row)

    # 한 시점 exposure matrix 예시
    print("\nOne day slice:")
    print("X[100].shape =", X[100].shape)
    print(X[100])


if __name__ == "__main__":
    main()
