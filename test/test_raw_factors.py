from factors.io import download_yfinance_panel, panel_diagnostics
from factors.raw_factors import compute_raw_factors, summarize_raw_factors


def main():
    tickers = ["AAPL", "MSFT", "BRK.B"]

    panel = download_yfinance_panel(
        tickers=tickers,
        start="2023-01-01",
        end="2024-12-31",
        progress=False,
    )

    print(panel_diagnostics(panel))

    raw = compute_raw_factors(panel)
    summary = summarize_raw_factors(raw)

    for name, stats in summary.items():
        print("\nFACTOR:", name)
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
