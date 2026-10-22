from __future__ import annotations
import numpy as np
from itertools import product


def brute_force_qubo(Q: np.ndarray, offset: float) -> tuple[np.ndarray, float]:
    """Enumerate all 2^n binary vectors, return argmin and min energy.

    Works for any Q form (upper triangular, lower triangular, or symmetric)
    because x^T Q x sums all entries.
    """
    n = Q.shape[0]
    if n == 0:
        return np.array([], dtype=float), offset
    if n > 20:
        raise ValueError(f"brute_force_qubo: n={n} > 20, too large for exhaustive search")

    best_x = None
    best_energy = float("inf")

    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        energy = float(x @ Q @ x) + offset
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return best_x, best_energy


def energy(Q: np.ndarray, x: np.ndarray, offset: float) -> float:
    return float(x @ Q @ x) + offset


def is_upper_triangular(Q: np.ndarray, tol: float = 1e-8) -> bool:
    """True iff every entry strictly below the diagonal is ~0."""
    n = Q.shape[0]
    return bool(np.allclose(np.tril(Q, k=-1), 0.0, atol=tol))


def validate_Q(Q: np.ndarray, n_variables: int) -> list[str]:
    """Validate Q is square, upper triangular, and finite. Returns list of errors."""
    errors: list[str] = []
    if Q.shape != (n_variables, n_variables):
        errors.append(f"Q shape {Q.shape} does not match n_variables={n_variables}")
    if not is_upper_triangular(Q):
        errors.append(
            "Q is not upper triangular: write off-diagonal coefficients to Q[min(i,j), max(i,j)] only"
        )
    if not np.isfinite(Q).all():
        errors.append("Q contains non-finite values (NaN or Inf)")
    return errors
