from __future__ import annotations

from typing import Dict

import numpy as np


def _as_float64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def compute_returns(price: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Simple return over `lag` periods.

    price: [T, N]
    return:
      out[t, n] = price[t, n] / price[t-lag, n] - 1
      shape [T, N]
    """
    price = _as_float64(price)
    T, N = price.shape
    out = np.full((T, N), np.nan, dtype=np.float64)

    if lag <= 0:
        raise ValueError("lag must be >= 1")
    if T <= lag:
        return out

    prev = price[:-lag]
    curr = price[lag:]

    valid = np.isfinite(curr) & np.isfinite(prev) & (prev != 0.0)
    tmp = np.full_like(curr, np.nan, dtype=np.float64)
    tmp[valid] = curr[valid] / prev[valid] - 1.0

    out[lag:] = tmp
    return out


def rolling_mean_strict(a: np.ndarray, window: int) -> np.ndarray:
    """
    Strict rolling mean:
    compute only if all values in the window are finite.

    input: [T, N]
    output: [T, N]
    """
    a = _as_float64(a)
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float64)

    if window <= 0:
        raise ValueError("window must be >= 1")
    if T < window:
        return out

    for j in range(N):
        col = a[:, j]
        finite = np.isfinite(col)

        for t in range(window - 1, T):
            s = t - window + 1
            w = col[s:t + 1]
            f = finite[s:t + 1]
            if f.all():
                out[t, j] = w.mean()

    return out


def rolling_std_strict(a: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """
    Strict rolling std:
    compute only if all values in the window are finite.
    """
    a = _as_float64(a)
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float64)

    if window <= 1:
        raise ValueError("window must be >= 2 for std")
    if T < window:
        return out

    for j in range(N):
        col = a[:, j]
        finite = np.isfinite(col)

        for t in range(window - 1, T):
            s = t - window + 1
            w = col[s:t + 1]
            f = finite[s:t + 1]
            if f.all():
                out[t, j] = w.std(ddof=ddof)

    return out


def rolling_max_strict(a: np.ndarray, window: int) -> np.ndarray:
    """
    Strict rolling max:
    compute only if all values in the window are finite.
    """
    a = _as_float64(a)
    T, N = a.shape
    out = np.full((T, N), np.nan, dtype=np.float64)

    if window <= 0:
        raise ValueError("window must be >= 1")
    if T < window:
        return out

    for j in range(N):
        col = a[:, j]
        finite = np.isfinite(col)

        for t in range(window - 1, T):
            s = t - window + 1
            w = col[s:t + 1]
            f = finite[s:t + 1]
            if f.all():
                out[t, j] = w.max()

    return out


def factor_mom_21(adj_close: np.ndarray) -> np.ndarray:
    """
    1-month momentum proxy: 21 trading day return
    """
    return compute_returns(adj_close, lag=21)


def factor_mom_63(adj_close: np.ndarray) -> np.ndarray:
    """
    3-month momentum proxy: 63 trading day return
    """
    return compute_returns(adj_close, lag=63)


def factor_rev_5(adj_close: np.ndarray) -> np.ndarray:
    """
    5-day reversal:
      larger recent losers -> larger exposure
    so define as negative 5-day return
    """
    return -compute_returns(adj_close, lag=5)


def factor_vol_20(adj_close: np.ndarray) -> np.ndarray:
    """
    20-day realized volatility of daily returns
    """
    daily_ret = compute_returns(adj_close, lag=1)
    return rolling_std_strict(daily_ret, window=20, ddof=1)


def factor_log_dollar_vol_20(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    log of 20-day average dollar volume
    """
    close = _as_float64(close)
    volume = _as_float64(volume)

    dollar_vol = np.full_like(close, np.nan, dtype=np.float64)
    valid = np.isfinite(close) & np.isfinite(volume)
    dollar_vol[valid] = close[valid] * volume[valid]

    adv20 = rolling_mean_strict(dollar_vol, window=20)

    out = np.full_like(adv20, np.nan, dtype=np.float64)
    valid_adv = np.isfinite(adv20) & (adv20 > 0.0)
    out[valid_adv] = np.log(adv20[valid_adv])

    return out


def factor_dist_52w_high(adj_close: np.ndarray) -> np.ndarray:
    """
    Distance to rolling 52-week high:
      adj_close / rolling_252d_max - 1

    Closer to 0 means near high.
    More negative means farther below high.
    """
    adj_close = _as_float64(adj_close)
    high_252 = rolling_max_strict(adj_close, window=252)

    out = np.full_like(adj_close, np.nan, dtype=np.float64)
    valid = np.isfinite(adj_close) & np.isfinite(high_252) & (high_252 > 0.0)
    out[valid] = adj_close[valid] / high_252[valid] - 1.0

    return out


def compute_raw_factors(panel: Dict) -> Dict[str, np.ndarray]:
    """
    Main raw factor builder.

    panel expected keys:
      "close", "adj_close", "volume"

    returns:
      {
        "mom_21": [T,N],
        "mom_63": [T,N],
        "rev_5": [T,N],
        "vol_20": [T,N],
        "log_dollar_vol_20": [T,N],
        "dist_52w_high": [T,N],
      }
    """
    close = _as_float64(panel["close"])
    adj_close = _as_float64(panel["adj_close"])
    volume = _as_float64(panel["volume"])

    raw = {
        "mom_21": factor_mom_21(adj_close),
        "mom_63": factor_mom_63(adj_close),
        "rev_5": factor_rev_5(adj_close),
        "vol_20": factor_vol_20(adj_close),
        "log_dollar_vol_20": factor_log_dollar_vol_20(close, volume),
        #"dist_52w_high": factor_dist_52w_high(adj_close),
    }
    return raw


def summarize_raw_factors(raw_factors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Lightweight diagnostics for debugging.
    """
    out: Dict[str, Dict[str, float]] = {}

    for name, a in raw_factors.items():
        finite = np.isfinite(a)
        n_total = a.size
        n_finite = int(finite.sum())

        if n_finite > 0:
            vals = a[finite]
            out[name] = {
                "shape_T": float(a.shape[0]),
                "shape_N": float(a.shape[1]),
                "finite_frac": float(n_finite / n_total),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        else:
            out[name] = {
                "shape_T": float(a.shape[0]),
                "shape_N": float(a.shape[1]),
                "finite_frac": 0.0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

    return out
