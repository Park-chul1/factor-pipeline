from factors.io import download_yfinance_panel
from factors.raw_factors import compute_raw_factors
from factors.preprocess import preprocess_all_factors, summarize_processed_factor
import numpy as np


def main():
    tickers = ["AAPL", "MSFT", "BRK.B"]
    panel = download_yfinance_panel(
        tickers=tickers,
        start="2023-01-01",
        end="2024-12-31",
        progress=False,
    )

    raw = compute_raw_factors(panel)

    # 테스트용 universe: 전 날짜 전 종목 포함
    T, N = panel["adj_close"].shape
    universe_mask = np.ones((T, N), dtype=bool)

    processed, valid_mask = preprocess_all_factors(
        raw_factors=raw,
        universe_mask=universe_mask,
        min_names=2,
        lower_q=0.01,
        upper_q=0.99,
    )

    for name, a in processed.items():
        print("\nFACTOR:", name)
        print(summarize_processed_factor(a))

    print("\ncombined valid mask")
    print("shape:", valid_mask.shape)
    print("finite frac:", valid_mask.mean())


if __name__ == "__main__":
    main()
