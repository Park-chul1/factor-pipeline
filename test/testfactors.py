from factors.io import download_yfinance_panel, panel_diagnostics

tickers = ["AAPL", "MSFT", "BRK.B"]
panel = download_yfinance_panel(tickers, "2023-01-01", "2023-03-01", progress=False)

print(panel["dates"][:3])
print(panel["tickers"])
print(panel["yf_tickers"])
print(panel["close"].shape)

print(panel_diagnostics(panel))
