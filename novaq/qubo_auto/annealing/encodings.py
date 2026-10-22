"""Encoding library + QUBO compiler (spec v0.2, P2.5).

A problem is given as a typed intermediate representation (IR): binary decision variables, a quadratic
objective, and typed constraints. The Encoding Agent picks one encoding per constraint from ENCODINGS;
`compile_problem` builds the ProblemInstance deterministically. Feasibility and the objective are always
evaluated from the IR itself (never from penalties), so a bad encoding can only hurt solution quality,
never silently redefine the problem.

    exact encodings:        penalty >= 0, = 0 iff satisfied (for the best auxiliary values)
    approximate encodings:  no auxiliary variables, but the penalty can be non-zero on feasible points
                            (e.g. unbalanced penalization); verify_encoding reports them as approximate
"""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from .problem import QUBO, ProblemInstance, add_qubo, qubo_energy


# ---------------------------------------------------------------- IR
class LinearConstraint(BaseModel):
    name: str
    kind: Literal["linear"] = "linear"
    coeffs: dict[int, float]                     # variable index -> coefficient
    sense: Literal["eq", "le", "ge"]
    rhs: float

    def satisfied(self, x: list[int]) -> bool:
        lhs = sum(c * x[i] for i, c in self.coeffs.items())
        return (abs(lhs - self.rhs) < 1e-9 if self.sense == "eq"
                else lhs <= self.rhs + 1e-9 if self.sense == "le" else lhs >= self.rhs - 1e-9)


class ExclusionConstraint(BaseModel):
    """At most one of each listed pair."""
    name: str
    kind: Literal["exclusion"] = "exclusion"
    pairs: list[tuple[int, int]]

    def satisfied(self, x: list[int]) -> bool:
        return all(not (x[i] and x[j]) for i, j in self.pairs)


class ImplicationConstraint(BaseModel):
    """x_i = 1 implies x_j = 1, for each listed pair (i, j)."""
    name: str
    kind: Literal["implication"] = "implication"
    pairs: list[tuple[int, int]]

    def satisfied(self, x: list[int]) -> bool:
        return all(x[j] or not x[i] for i, j in self.pairs)


Constraint = LinearConstraint | ExclusionConstraint | ImplicationConstraint


class ProblemIR(BaseModel):
    name: str
    instance_id: str
    n_variables: int                                  # binary decision variables only
    sense: Literal["minimize", "maximize"]
    linear: dict[int, float] = Field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = Field(default_factory=dict)
    constant: float = 0.0
    constraints: list[Constraint] = Field(default_factory=list)
    reference_objective: float | None = None

    def objective(self, x: list[int]) -> float:
        return (self.constant + sum(c * x[i] for i, c in self.linear.items())
                + sum(c * x[i] * x[j] for (i, j), c in self.quadratic.items()))


# ---------------------------------------------------------------- encodings
@dataclass
class Encoded:
    penalty: QUBO                       # over decision + auxiliary variables (global indices)
    offset: float
    aux: list[int] = field(default_factory=list)
    exact: bool = True
    note: str = ""


def _square(coeffs: dict[int, float], const: float) -> tuple[QUBO, float]:
    """(sum_k c_k z_k + const)^2 for binary z."""
    q: QUBO = {}
    items = sorted(coeffs.items())
    for a, (i, ci) in enumerate(items):
        q[(i, i)] = q.get((i, i), 0.0) + ci * ci + 2 * const * ci
        for j, cj in items[a + 1:]:
            key = (min(i, j), max(i, j))
            q[key] = q.get(key, 0.0) + 2 * ci * cj
    return q, const * const


def _lhs_range(c: LinearConstraint) -> tuple[float, float]:
    lo = sum(v for v in c.coeffs.values() if v < 0)
    hi = sum(v for v in c.coeffs.values() if v > 0)
    return lo, hi


def _as_le(c: LinearConstraint) -> tuple[dict[int, float], float]:
    return (dict(c.coeffs), c.rhs) if c.sense == "le" else ({i: -v for i, v in c.coeffs.items()}, -c.rhs)


def enc_square(c: LinearConstraint, next_aux: int, **_) -> Encoded:
    if c.sense != "eq":
        raise ValueError("square encoding is for equalities")
    q, off = _square(c.coeffs, -c.rhs)
    return Encoded(q, off)


def _slack_le(c: LinearConstraint, next_aux: int, unary: bool) -> Encoded:
    coeffs, rhs = _as_le(c)
    lo = sum(v for v in coeffs.values() if v < 0)
    span = rhs - lo                                    # largest slack ever needed
    if span < 0:
        raise ValueError(f"constraint {c.name} is infeasible for every x")
    if any(abs(v - round(v)) > 1e-9 for v in list(coeffs.values()) + [rhs]):
        raise ValueError("slack encodings need integer coefficients and rhs")
    span = int(round(span))
    if unary:
        weights = [1] * span
    else:
        m = max(1, span.bit_length())
        weights = [2 ** k for k in range(m - 1)] + [span - (2 ** (m - 1) - 1)] if span > 0 else []
    aux = list(range(next_aux, next_aux + len(weights)))
    q, off = _square({**coeffs, **dict(zip(aux, map(float, weights)))}, -rhs)
    return Encoded(q, off, aux, note=f"{'unary' if unary else 'binary'} slack, {len(aux)} bits")


def enc_binary_slack(c, next_aux, **_):
    return _slack_le(c, next_aux, unary=False)


def enc_unary_slack(c, next_aux, **_):
    return _slack_le(c, next_aux, unary=True)


def enc_unbalanced(c: LinearConstraint, next_aux: int, l1: float = 1.0, l2: float = 1.0, **_) -> Encoded:
    """Unbalanced penalization (Montanez-Barrera et al., arXiv:2211.13914): with h = rhs - lhs >= 0 required,
    penalty = l2*h^2 - l1*h. No slack variables; small positive h is barely penalised, h < 0 strongly.
    Approximate: feasible points can carry a (small, possibly negative) penalty."""
    coeffs, rhs = _as_le(c)
    neg = {i: -v for i, v in coeffs.items()}            # h = rhs + sum(neg_i x_i)
    q, off = _square(neg, rhs)
    q = {k: l2 * v for k, v in q.items()}
    off = l2 * off - l1 * rhs
    for i, v in neg.items():
        q[(i, i)] = q.get((i, i), 0.0) - l1 * v
    return Encoded(q, off, exact=False, note=f"unbalanced l1={l1}, l2={l2}")


def unbalanced_max_reward(c: LinearConstraint, l1: float = 1.0, l2: float = 1.0) -> float:
    """Largest NEGATIVE penalty (a reward, per unit weight) unbalanced penalization gives a FEASIBLE point.
    With h = rhs - lhs in [h_lo, h_hi] on the feasible side, penalty = l2*h^2 - l1*h is lowest at h = l1/(2*l2).
    If that reward outweighs the objective, the solver is paid to stay far from the bound: e.g. a loose budget
    with l1=300, l2=1 makes 'select nothing' the QUBO optimum."""
    coeffs, rhs = _as_le(c)
    lo, hi = sum(v for v in coeffs.values() if v < 0), sum(v for v in coeffs.values() if v > 0)
    h_lo, h_hi = max(0.0, rhs - hi), rhs - lo
    if h_hi < h_lo:
        return 0.0
    h = min(max(l1 / (2 * l2), h_lo), h_hi)
    return max(0.0, l1 * h - l2 * h * h)


def enc_exclusion(c: ExclusionConstraint, next_aux: int, **_) -> Encoded:
    q: QUBO = {}
    for i, j in c.pairs:
        key = (min(i, j), max(i, j))
        q[key] = q.get(key, 0.0) + 1.0
    return Encoded(q, 0.0)


def enc_implication(c: ImplicationConstraint, next_aux: int, **_) -> Encoded:
    q: QUBO = {}
    for i, j in c.pairs:                                 # x_i (1 - x_j) = x_i - x_i x_j
        q[(i, i)] = q.get((i, i), 0.0) + 1.0
        key = (min(i, j), max(i, j))
        q[key] = q.get(key, 0.0) - 1.0
    return Encoded(q, 0.0)


# encoding -> {param: (min, max)}; encodings not listed take no parameters
ENCODING_PARAMS = {"unbalanced": {"l1": (1e-3, 1e4), "l2": (1e-4, 1e3)}}

# constraint kind / sense -> available encodings (first = default)
ENCODINGS = {
    ("linear", "eq"): {"square": enc_square},
    ("linear", "le"): {"binary_slack": enc_binary_slack, "unary_slack": enc_unary_slack, "unbalanced": enc_unbalanced},
    ("linear", "ge"): {"binary_slack": enc_binary_slack, "unary_slack": enc_unary_slack, "unbalanced": enc_unbalanced},
    ("exclusion", None): {"pairwise": enc_exclusion},
    ("implication", None): {"pairwise": enc_implication},
}


def options_for(c: Constraint) -> dict:
    return ENCODINGS[(c.kind, getattr(c, "sense", None))]


def encode(c: Constraint, choice: str | None, next_aux: int, **params) -> Encoded:
    opts = options_for(c)
    name = choice or next(iter(opts))
    if name not in opts:
        raise ValueError(f"encoding '{name}' not available for {c.name} (choose from {sorted(opts)})")
    return opts[name](c, next_aux, **params)


# ---------------------------------------------------------------- verification
def verify_encoding(c: Constraint, enc: Encoded, n_decision: int, max_enum: int = 16) -> dict:
    """Two-way check over all decision assignments (x) and auxiliary assignments (s):
    violated -> min_s penalty > 0;  satisfied -> min_s penalty == 0 (exact encodings).
    For approximate encodings, report how penalties on feasible and infeasible points compare."""
    used = sorted({i for k in enc.penalty for i in k if i < n_decision} |
                  set(getattr(c, "coeffs", {}) or {}) | {i for p in getattr(c, "pairs", []) for i in p})
    if len(used) + len(enc.aux) > max_enum:
        return {"checked": False, "reason": f"{len(used)} decision + {len(enc.aux)} aux variables > {max_enum}"}
    worst_feasible, best_infeasible, errors = -math.inf, math.inf, []
    size = (max(used + enc.aux) + 1) if (used or enc.aux) else 1
    for xb in itertools.product((0, 1), repeat=len(used)):
        z = [0] * size
        for i, b in zip(used, xb):
            z[i] = b
        best = math.inf
        for sb in itertools.product((0, 1), repeat=len(enc.aux)):
            for i, b in zip(enc.aux, sb):
                z[i] = b
            best = min(best, qubo_energy(enc.penalty, enc.offset, z))
        ok = c.satisfied(z + [0] * max(0, n_decision - len(z)))
        if ok:
            worst_feasible = max(worst_feasible, best)
            if enc.exact and abs(best) > 1e-9:
                errors.append(f"satisfied at {dict(zip(used, xb))} but min penalty {best:.4g} != 0")
        else:
            best_infeasible = min(best_infeasible, best)
            if best <= 1e-9:
                errors.append(f"violated at {dict(zip(used, xb))} but min penalty {best:.4g} <= 0")
        if len(errors) >= 3:
            break
    return {"checked": True, "exact": enc.exact, "passed": not errors if enc.exact else best_infeasible > worst_feasible,
            "errors": errors, "max_penalty_feasible": worst_feasible, "min_penalty_infeasible": best_infeasible}


# ---------------------------------------------------------------- compiler
def compile_problem(ir: ProblemIR, choices: dict[str, str] | None = None,
                    params: dict[str, dict] | None = None, weights: dict[str, float] | None = None) -> ProblemInstance:
    """IR + one encoding per constraint -> ProblemInstance (penalties separate, checks from the IR)."""
    choices, params, weights = choices or {}, params or {}, weights or {}
    sign = -1.0 if ir.sense == "maximize" else 1.0
    obj: QUBO = {(i, i): sign * c for i, c in ir.linear.items()}
    add_qubo(obj, {k: sign * v for k, v in ir.quadratic.items()})
    next_aux, penalties, notes = ir.n_variables, {}, {}
    for c in ir.constraints:
        enc = encode(c, choices.get(c.name), next_aux, **params.get(c.name, {}))
        next_aux += len(enc.aux)
        penalties[c.name] = (enc.penalty, enc.offset)
        notes[c.name] = f"{choices.get(c.name) or next(iter(options_for(c)))}{' (approximate)' if not enc.exact else ''}"
    n_dec = ir.n_variables
    floor = max((sum(abs(v) for k, v in obj.items() if i in k) for i in range(n_dec)), default=0.0)
    return ProblemInstance(
        name=ir.name, n_variables=next_aux, sense=ir.sense,
        objective=lambda x: ir.objective(list(x[:n_dec])),
        objective_qubo=obj, objective_offset=sign * ir.constant,
        constraints={c.name: (lambda x, c=c: c.satisfied(list(x[:n_dec]))) for c in ir.constraints},
        penalty_qubos=penalties,
        default_penalty_weights={c.name: weights.get(c.name, floor + 1.0) for c in ir.constraints},
        constraint_types={c.name: notes[c.name] for c in ir.constraints},
        reference_objective=ir.reference_objective, instance_id=ir.instance_id,
    )
