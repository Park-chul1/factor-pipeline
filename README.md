# Quant Factor Model

WSL 기반 Python 프로젝트.

## Current pipeline
- S&P 500 universe reconstruction from `db/annotated_sp500.csv`
- yfinance price download
- raw factor computation
- cross-sectional preprocessing
- 3D exposure tensor build: `X[T, N, K]`
- factor return estimation via cross-sectional regression
    - OLS 기반 팩터 수익률 추정: $f = (X^T X)^{-1} X^T r$
    - $X$: Factor Exposure Matrix, $r$: Asset Returns, $f$: Estimated Factor Returns

## Main entrypoint
```bash
python3 scripts/run_factor_pipeline.py
