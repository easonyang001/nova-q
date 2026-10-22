from __future__ import annotations
import traceback
from typing import Callable

import numpy as np

from ..schemas import QUBOFormulation, StructuredSpec, TestInstanceResult
from .brute_force import validate_Q
from .test_cases import TestCase

MAX_BRUTE_FORCE_VARS = 26     # 2^26 = 67M energies, vectorised: well under a minute
_CHUNK = 1 << 18              # bitstrings evaluated per numpy batch (keeps memory bounded)


def run_tests(
    build_qubo: Callable[[dict], tuple[np.ndarray, float]],
    test_cases: list[TestCase],
) -> tuple[list[TestInstanceResult], str]:
    """Run build_qubo on each pre-computed test case; check that the QUBO's
    argmin set overlaps with the case's optimal_bitstrings.

    Returns (results, failure_classification):
      - "success" if all cases pass
      - "coding_error" if Q construction fails or shape/triangularity wrong
      - "formulation_error" if Q is valid but its argmin disagrees with ground truth
      - "unverifiable" if the only problem is a case too large to enumerate: not the code's fault,
        so it must not be sent back to the Coder (it cannot change the test cases)
    """
    results: list[TestInstanceResult] = []
    for i, case in enumerate(test_cases):
        results.append(_run_single(build_qubo, case, i))

    n_pass = sum(1 for r in results if r.match)
    if n_pass == len(test_cases):
        return results, "success"

    coding_errors = sum(1 for r in results if r.status == "coding_error")
    formulation_errors = sum(1 for r in results if r.status == "formulation_error")
    if coding_errors == formulation_errors == 0:
        return results, "unverifiable"
    if coding_errors >= formulation_errors:
        return results, "coding_error"
    return results, "formulation_error"


def _run_single(
    build_qubo: Callable,
    case: TestCase,
    instance_id: int,
) -> TestInstanceResult:
    # Build Q
    try:
        Q, offset = build_qubo(case.instance)
        offset = float(offset) if isinstance(offset, (int, float)) else 0.0
    except Exception:
        return TestInstanceResult(
            instance_id=instance_id,
            status="coding_error",
            error_message=f"[{case.name}] build_qubo raised: {traceback.format_exc()[:600]}",
            match=False,
        )

    if Q.shape != (case.n_variables, case.n_variables):
        return TestInstanceResult(
            instance_id=instance_id,
            status="coding_error",
            error_message=(
                f"[{case.name}] Q shape {Q.shape} does not match expected "
                f"({case.n_variables},{case.n_variables})"
            ),
            match=False,
        )

    q_errors = validate_Q(Q, case.n_variables)
    if q_errors:
        return TestInstanceResult(
            instance_id=instance_id,
            status="coding_error",
            error_message=f"[{case.name}] " + "; ".join(q_errors),
            match=False,
        )

    n = case.n_variables
    if n > MAX_BRUTE_FORCE_VARS:        # 2^n enumeration would never finish (one benchmark hung for hours)
        return TestInstanceResult(
            instance_id=instance_id,
            status="unverifiable",
            error_message=(f"[{case.name}] {n} variables is too many for brute-force verification "
                           f"(limit {MAX_BRUTE_FORCE_VARS}); the test case is too large, not a code error"),
            match=False,
        )

    qubo_min, qubo_optimal = _enumerate_minima(np.asarray(Q, dtype=float), offset, n)
    problem_optimal = set(case.optimal_bitstrings)

    overlap = qubo_optimal & problem_optimal
    if overlap:
        return TestInstanceResult(
            instance_id=instance_id,
            status="success",
            qubo_optimum=qubo_min,
            problem_optimum=case.optimal_value,
            match=True,
        )

    # Diagnose: report the disagreement without exposing ground-truth bitstrings or values
    error_msg = (
        f"[{case.name}] Formulation error: the QUBO minimum does not coincide with "
        f"the correct optimal solution. "
        f"QUBO found {len(qubo_optimal)} optimal bitstring(s); "
        f"none overlap with the {len(problem_optimal)} correct optimal bitstring(s). "
        f"Check penalty weights, objective sign, and variable indexing."
    )
    return TestInstanceResult(
        instance_id=instance_id,
        status="formulation_error",
        error_message=error_msg,
        qubo_optimum=qubo_min,
        problem_optimum=case.optimal_value,
        match=False,
    )


def _enumerate_minima(Q: np.ndarray, offset: float, n: int, tol: float = 1e-6) -> tuple[float, set[tuple]]:
    """Exact minimum of x^T Q x + offset over all 2^n bitstrings, and every bitstring within tol of it.
    Bitstrings are tuples in itertools.product order (first variable = most significant bit)."""
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    best, optimal = np.inf, set()
    for start in range(0, 1 << n, _CHUNK):
        idx = np.arange(start, min(start + _CHUNK, 1 << n), dtype=np.int64)
        X = ((idx[:, None] >> shifts) & 1).astype(float)
        E = np.einsum("ij,ij->i", X @ Q, X) + offset
        m = float(E.min())
        if m < best - tol:                      # a strictly better level: earlier ties are not optimal
            best, optimal = m, set()
        if m <= best + tol:
            best = min(best, m)
            optimal |= {tuple(int(b) for b in X[k]) for k in np.flatnonzero(E <= best + tol)}
    optimal = {b for b in optimal if abs(float(np.asarray(b, float) @ Q @ np.asarray(b, float)) + offset - best) <= tol}
    return best, optimal
