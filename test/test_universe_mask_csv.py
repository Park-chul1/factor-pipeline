from factors.universe_mask import build_universe_mask_from_annotated, summarize_universe_mask


def main():
    dates, tickers, mask = build_universe_mask_from_annotated(
        annotated_csv="db/annotated_sp500.csv",
        start="2024-01-01",
        end="2024-03-31",
    )

    print("dates:", len(dates))
    print("tickers:", len(tickers))
    print("mask shape:", mask.shape)
    print("first 10 tickers:", tickers[:10])

    summary = summarize_universe_mask(dates, tickers, mask)
    print(summary.head())
    print(summary.tail())
    print("min names:", summary["n_names"].min())
    print("max names:", summary["n_names"].max())


if __name__ == "__main__":
    main()
