from __future__ import annotations

from typing import Tuple

import numpy as np


def build_forward_returns(adj_close: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Forward return aligned to exposure date t.

    r[t, n] = adj_close[t + horizon, n] / adj_close[t, n] - 1

    output shape: [T, N]
    last `horizon` rows will be NaN
    """
    adj_close = np.asarray(adj_close, dtype=np.float64)
    T, N = adj_close.shape

    out = np.full((T, N), np.nan, dtype=np.float64)

    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if T <= horizon:
        return out

    p0 = adj_close[:-horizon]
    p1 = adj_close[horizon:]

    valid = np.isfinite(p0) & np.isfinite(p1) & (p0 != 0.0)
    tmp = np.full_like(p0, np.nan, dtype=np.float64)
    tmp[valid] = p1[valid] / p0[valid] - 1.0

    out[:-horizon] = tmp
    return out


def estimate_factor_returns_one_day(
    X_t: np.ndarray,
    r_t: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Cross-sectional factor return estimation for one day.

    X_t: [n_t, K]
    r_t: [n_t]

    returns:
      f_t: [K]

    Uses ridge-regularized normal equations:
      f_t = (X'X + ridge I)^(-1) X'r
    """
    X_t = np.asarray(X_t, dtype=np.float64)
    r_t = np.asarray(r_t, dtype=np.float64)

    if X_t.ndim != 2:
        raise ValueError("X_t must be 2D")
    if r_t.ndim != 1:
        raise ValueError("r_t must be 1D")
    if X_t.shape[0] != r_t.shape[0]:
        raise ValueError("X_t and r_t row counts must match")

    K = X_t.shape[1]
    XtX = X_t.T @ X_t
    Xtr = X_t.T @ r_t

    XtX = XtX + ridge * np.eye(K, dtype=np.float64)

    try:
        f_t = np.linalg.solve(XtX, Xtr)
    except np.linalg.LinAlgError:
        f_t = np.full(K, np.nan, dtype=np.float64)

    return f_t


def estimate_factor_returns(
    X: np.ndarray,
    r: np.ndarray,
    universe_mask: np.ndarray,
    valid_mask: np.ndarray,
    min_names: int = 30,
    ridge: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate factor returns day by day.

    Inputs:
      X: [T, N, K]
      r: [T, N]
      universe_mask: [T, N]
      valid_mask: [T, N]

    Returns:
      f: [T, K]
      day_valid: [T] bool
    """
    X = np.asarray(X, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    T, N, K = X.shape

    if r.shape != (T, N):
        raise ValueError(f"r shape {r.shape} does not match X first dims {(T, N)}")
    if universe_mask.shape != (T, N):
        raise ValueError("universe_mask shape mismatch")
    if valid_mask.shape != (T, N):
        raise ValueError("valid_mask shape mismatch")

    f = np.full((T, K), np.nan, dtype=np.float64)
    day_valid = np.zeros(T, dtype=bool)

    for t in range(T):
        mask_t = universe_mask[t] & valid_mask[t] & np.isfinite(r[t])

        # First pass filter
        if mask_t.sum() < max(min_names, K + 2):
            continue

        X_t = X[t, mask_t, :]
        r_t = r[t, mask_t]

        # Remove any row still containing NaNs
        row_ok = np.isfinite(X_t).all(axis=1) & np.isfinite(r_t)
        X_t = X_t[row_ok]
        r_t = r_t[row_ok]

        if X_t.shape[0] < max(min_names, K + 2):
            continue

        f[t] = estimate_factor_returns_one_day(X_t, r_t, ridge=ridge)
        day_valid[t] = np.isfinite(f[t]).all()

    return f, day_valid


def summarize_factor_returns(f: np.ndarray) -> dict:
    """
    Simple diagnostics for factor return matrix [T, K].
    """
    mask = np.isfinite(f)
    n_total = f.size
    n_finite = int(mask.sum())

    if n_finite == 0:
        return {
            "finite_frac": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    vals = f[mask]
    return {
        "finite_frac": float(n_finite / n_total),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }
