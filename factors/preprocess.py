from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def winsorize_cross_section(
    x: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> np.ndarray:
    """
    Winsorize one cross-sectional vector.
    NaNs are ignored.
    """
    out = x.copy()
    mask = np.isfinite(out)

    if mask.sum() == 0:
        return out

    vals = out[mask]
    lo = np.quantile(vals, lower_q)
    hi = np.quantile(vals, upper_q)

    out[mask] = np.clip(vals, lo, hi)
    return out


def zscore_cross_section(x: np.ndarray) -> np.ndarray:
    """
    Z-score one cross-sectional vector.
    NaNs are ignored.
    """
    out = x.copy()
    mask = np.isfinite(out)

    if mask.sum() == 0:
        return out

    vals = out[mask]
    mu = vals.mean()
    sd = vals.std(ddof=1)

    if not np.isfinite(sd) or sd < 1e-12:
        out[mask] = 0.0
        return out

    out[mask] = (vals - mu) / sd
    return out


def preprocess_factor_matrix(
    raw_factor: np.ndarray,
    universe_mask: np.ndarray,
    min_names: int = 30,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess one factor matrix [T, N].

    Steps per date t:
      1) keep only universe names
      2) ignore NaNs
      3) winsorize cross-section
      4) z-score cross-section

    Returns:
      processed_factor: [T, N]
      valid_mask: [T, N]
    """
    T, N = raw_factor.shape
    processed = np.full((T, N), np.nan, dtype=np.float64)
    valid_mask = np.zeros((T, N), dtype=bool)

    for t in range(T):
        mask_t = universe_mask[t] & np.isfinite(raw_factor[t])

        if mask_t.sum() < min_names:
            continue

        x = np.full(N, np.nan, dtype=np.float64)
        x[mask_t] = raw_factor[t, mask_t]

        x = winsorize_cross_section(x, lower_q=lower_q, upper_q=upper_q)
        x = zscore_cross_section(x)

        processed[t] = x
        valid_mask[t] = np.isfinite(x) & universe_mask[t]

    return processed, valid_mask


def preprocess_all_factors(
    raw_factors: Dict[str, np.ndarray],
    universe_mask: np.ndarray,
    min_names: int = 30,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Preprocess all factors independently.

    Returns:
      processed_factors: dict[name -> [T, N]]
      combined_valid_mask: [T, N]
        True only where all factors are valid
    """
    processed: Dict[str, np.ndarray] = {}
    valid_masks = []

    for name, raw in raw_factors.items():
        proc, valid = preprocess_factor_matrix(
            raw_factor=raw,
            universe_mask=universe_mask,
            min_names=min_names,
            lower_q=lower_q,
            upper_q=upper_q,
        )
        processed[name] = proc
        valid_masks.append(valid)

    combined_valid_mask = np.logical_and.reduce(valid_masks)
    return processed, combined_valid_mask


def summarize_processed_factor(a: np.ndarray) -> dict:
    mask = np.isfinite(a)
    n_total = a.size
    n_finite = int(mask.sum())

    if n_finite == 0:
        return {
            "finite_frac": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    vals = a[mask]
    return {
        "finite_frac": float(n_finite / n_total),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }
