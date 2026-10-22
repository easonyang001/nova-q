"""Deterministic tool layer (spec §6). Everything here is plain code; agents only call it."""
from __future__ import annotations
import itertools
import math

import numpy as np

from .problem import QUBO, ProblemInstance, qubo_energy


def verify_small(problem: ProblemInstance, penalty_weights: dict[str, float] | None = None,
                 max_n: int = 20) -> dict:
    """Brute-force check: the WHOLE argmin set is feasible and contains a true optimum.

    Stricter than the original Test Runner, which only required overlap with the optimum.
    """
    n = problem.n_variables
    if n > max_n:
        return {"checked": False, "reason": f"n={n} > {max_n}"}
    qubo, offset = problem.to_qubo(penalty_weights)
    xs = [list(x) for x in itertools.product((0, 1), repeat=n)]
    energies = np.array([qubo_energy(qubo, offset, x) for x in xs])
    argmin = [xs[i] for i in np.flatnonzero(np.isclose(energies, energies.min(), atol=1e-9))]
    infeasible = [x for x in argmin if not problem.is_feasible(x)]
    ref = problem.brute_force_reference(max_n) if problem.reference_objective is None else problem.reference_objective
    optimal = [x for x in argmin if problem.is_feasible(x) and math.isclose(problem.objective(x), ref)]
    return {"checked": True, "passed": not infeasible and bool(optimal),
            "argmin_size": len(argmin), "argmin_infeasible": len(infeasible), "argmin_optimal": len(optimal)}


def penalty_lower_bound(problem: ProblemInstance) -> dict[str, float]:
    """Heuristic floor per constraint: the largest objective gain from flipping one variable.

    Each penalty is integer-valued with minimum violation >= 1, so a weight above this value means
    no single flip that breaks a constraint can lower the energy. It is a heuristic (multi-variable
    moves are not covered); `verify_small` gives the exact answer on small instances.
    """
    gain = np.zeros(problem.n_variables)
    for (i, j), c in problem.objective_qubo.items():
        gain[i] += abs(c)
        if i != j:
            gain[j] += abs(c)
    bound = float(gain.max()) if gain.size else 0.0
    return {name: bound for name in problem.penalty_qubos}


def qubo_summary(problem: ProblemInstance, penalty_weights: dict[str, float] | None = None) -> dict:
    """Problem features: used as agent context and as the RAG retrieval key."""
    qubo, offset = problem.to_qubo(penalty_weights)
    n = problem.n_variables
    coefs = np.array([abs(c) for c in qubo.values() if c != 0.0]) if qubo else np.array([0.0])
    n_quad = sum(1 for (i, j) in qubo if i != j)
    obj_max = max((abs(c) for c in problem.objective_qubo.values()), default=0.0)
    return {
        "problem_type": problem.name,
        "sense": problem.sense,
        "n_variables": n,
        "n_quadratic_terms": n_quad,
        "density": round(2 * n_quad / (n * (n - 1)), 4) if n > 1 else 0.0,
        "coef_abs_max": float(coefs.max()),
        "coef_abs_min": float(coefs.min()),
        "coef_dynamic_range": float(coefs.max() / coefs.min()) if coefs.min() > 0 else None,
        "objective_coef_abs_max": float(obj_max),
        "constraints": {name: problem.constraint_types.get(name, "unknown") for name in problem.constraints},
        "has_reference": problem.reference_objective is not None,
    }


def scale_for_backend(qubo: QUBO, offset: float, factor: float, round_to_int: bool = False,
                      check_argmin_n: int = 16) -> tuple[QUBO, float, dict]:
    """Multiply the QUBO by `factor` (> 0), optionally rounding to integers (fixed-precision hardware).

    Positive scaling alone never changes the argmin; rounding can, so on small instances the argmin
    sets before and after are compared.
    """
    if factor <= 0:
        raise ValueError("scale factor must be > 0")
    scaled = {k: v * factor for k, v in qubo.items()}
    new_offset = offset * factor
    if round_to_int:
        scaled = {k: float(round(v)) for k, v in scaled.items() if round(v) != 0}
        new_offset = float(round(new_offset))
    report = {"factor": factor, "rounded": round_to_int, "argmin_preserved": None}
    n = 1 + max((max(k) for k in qubo), default=-1)
    if round_to_int and n <= check_argmin_n:
        xs = [list(x) for x in itertools.product((0, 1), repeat=n)]
        def argmin(q, o):
            e = np.array([qubo_energy(q, o, x) for x in xs])
            return {tuple(xs[i]) for i in np.flatnonzero(np.isclose(e, e.min()))}
        report["argmin_preserved"] = argmin(qubo, offset) == argmin(scaled, new_offset)
    return scaled, new_offset, report
