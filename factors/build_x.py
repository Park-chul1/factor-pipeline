from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from factors.preprocess import preprocess_all_factors


def get_factor_names(processed_factors: Dict[str, np.ndarray]) -> List[str]:
    """
    Return a stable, deterministic factor order.
    """
    return sorted(processed_factors.keys())


def stack_factors(
    processed_factors: Dict[str, np.ndarray],
    factor_names: List[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Stack processed factor matrices into a 3D tensor.

    Input:
      processed_factors[name] -> [T, N]

    Output:
      X -> [T, N, K]
      factor_names -> ordered factor name list
    """
    if len(processed_factors) == 0:
        raise ValueError("processed_factors is empty")

    if factor_names is None:
        factor_names = get_factor_names(processed_factors)

    mats = [processed_factors[name] for name in factor_names]

    # Shape consistency check
    ref_shape = mats[0].shape
    for name, mat in zip(factor_names, mats):
        if mat.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch for factor '{name}': "
                f"expected {ref_shape}, got {mat.shape}"
            )

    X = np.stack(mats, axis=2).astype(np.float64, copy=False)
    return X, factor_names


def build_X(
    raw_factors: Dict[str, np.ndarray],
    universe_mask: np.ndarray,
    min_names: int = 30,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    factor_names: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Full builder:
      raw factors -> preprocess -> stack into X

    Returns:
      X: [T, N, K]
      valid_mask: [T, N]
      factor_names: list[str]
    """
    processed_factors, valid_mask = preprocess_all_factors(
        raw_factors=raw_factors,
        universe_mask=universe_mask,
        min_names=min_names,
        lower_q=lower_q,
        upper_q=upper_q,
    )

    X, factor_names = stack_factors(
        processed_factors=processed_factors,
        factor_names=factor_names,
    )

    return X, valid_mask, factor_names


def summarize_X(
    X: np.ndarray,
    factor_names: List[str],
) -> List[dict]:
    """
    Per-factor summary of the 3D exposure tensor.
    """
    T, N, K = X.shape

    if K != len(factor_names):
        raise ValueError(
            f"factor_names length {len(factor_names)} does not match X.shape[2] {K}"
        )

    rows = []
    for k, name in enumerate(factor_names):
        a = X[:, :, k]
        mask = np.isfinite(a)
        n_total = a.size
        n_finite = int(mask.sum())

        if n_finite == 0:
            rows.append(
                {
                    "factor": name,
                    "finite_frac": 0.0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )
            continue

        vals = a[mask]
        rows.append(
            {
                "factor": name,
                "finite_frac": float(n_finite / n_total),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )

    return rows
