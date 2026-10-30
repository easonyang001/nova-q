"""What a planner asks after "here is the best plan": which limits are binding, and what would a bit more of one buy?

The delivered plan is the optimum of the stated objective under the stated limits. This module re-solves the same
verified model with one limit moved at a time and reports the effect, using the same compile-and-sample route as the
pipeline, so the numbers are as trustworthy as the delivery itself (a sampling solver, not a proof).

    report = sensitivity_report(ir, delivered_x)
    report["binding"]        constraints with no slack at the delivered plan
    report["sensitivity"]    per binding linear constraint: objective and plan change when its limit moves

Linear `le`/`ge` limits move by relative factors; `eq` limits with an integer right-hand side (exactly k) move by
one unit. Exclusion and implication constraints have no number to move and are only listed.
"""
from __future__ import annotations
import numpy as np

from .backends import SimulatedAnnealingBackend
from .encodings import LinearConstraint, ProblemIR, compile_problem

FACTORS = (0.9, 0.95, 1.05, 1.1)
TOL = 1e-9


def slack(c: LinearConstraint, x: list[int]) -> float:
    """Distance to the limit on the feasible side (0 = binding)."""
    lhs = sum(v * x[i] for i, v in c.coeffs.items())
    return {"le": c.rhs - lhs, "ge": lhs - c.rhs, "eq": -abs(lhs - c.rhs)}[c.sense]


def binding_constraints(ir: ProblemIR, x: list[int]) -> list[dict]:
    rows = []
    for c in ir.constraints:
        if isinstance(c, LinearConstraint):
            s = slack(c, x)
            rows.append({"name": c.name, "kind": "linear", "sense": c.sense, "rhs": c.rhs,
                         "slack": s, "binding": abs(s) <= TOL * max(1.0, abs(c.rhs)) or c.sense == "eq"})
        else:
            rows.append({"name": c.name, "kind": c.kind, "binding": None})
    return rows


def solve_best(ir: ProblemIR, reads: int = 500, seed: int = 0) -> tuple[list[int], float] | None:
    """Best feasible decoded sample of the compiled QUBO (decision variables only), or None."""
    prob = compile_problem(ir)
    qubo, off = prob.to_qubo()
    scale = max([abs(v) for v in qubo.values()] + [1e-12])
    q = {(int(i), int(j)): float(v) / scale for (i, j), v in qubo.items()}
    result = SimulatedAnnealingBackend().run(q, off / scale, prob.n_variables, {"num_reads": reads, "seed": seed})
    best = None
    for sample in result.samples:
        x = [int(v) for v in sample[:ir.n_variables]]
        if all(c.satisfied(x) for c in ir.constraints):
            value = ir.objective(x)
            if best is None or (value > best[1] if ir.sense == "maximize" else value < best[1]):
                best = (x, value)
    return best


def with_rhs(ir: ProblemIR, name: str, rhs: float) -> ProblemIR:
    out = ir.model_copy(deep=True)
    for c in out.constraints:
        if c.name == name:
            c.rhs = rhs
    return out


def moves(c: LinearConstraint) -> list[tuple[str, float]]:
    integral = float(c.rhs).is_integer() and all(float(v).is_integer() for v in c.coeffs.values())
    if c.sense == "eq":
        return [(f"{d:+d}", c.rhs + d) for d in (-1, 1)] if integral else []
    if not integral:
        return [(f"{f - 1:+.0%}", c.rhs * f) for f in FACTORS]
    # the slack encodings need an integer limit: round, and fall back to one unit when rounding moves nothing
    out = {}
    for f in FACTORS:
        rhs = float(round(c.rhs * f))
        if rhs != c.rhs:
            out.setdefault(rhs, f"{f - 1:+.0%}")
    if not out:
        out = {c.rhs - 1: "-1", c.rhs + 1: "+1"}
    return [(label, rhs) for rhs, label in sorted(out.items())]


def planning_limits(ir: ProblemIR, rows: list[dict], max_constraints: int) -> list[LinearConstraint]:
    """The limits a planner would move: binding budgets and capacities (le/ge) first, then totals over most of the
    variables ('exactly 3 chargers'). Local one-hot rows of a tour or an assignment ('each stop once') are skipped:
    moving one of them makes the plan invalid, not cheaper."""
    binding = {r["name"] for r in rows if r["binding"]}
    linear = [c for c in ir.constraints if isinstance(c, LinearConstraint) and c.name in binding]
    ranked = ([c for c in linear if c.sense != "eq"]
              + [c for c in linear if c.sense == "eq" and len(c.coeffs) >= 0.5 * ir.n_variables])
    return ranked[:max_constraints]


def sensitivity_report(ir: ProblemIR, x: list[int], reads: int = 500, seed: int = 0, max_constraints: int = 3) -> dict:
    base = solve_best(ir, reads, seed)
    rows = binding_constraints(ir, x)
    out = {"delivered_objective": ir.objective(x), "resolved_objective": None if base is None else base[1],
           "binding": [r["name"] for r in rows if r["binding"]], "constraints": rows, "sensitivity": [],
           "note": "re-solved with the pipeline's sampler; compare each row with 'resolved_objective', not with the "
                   "delivered plan, so solver noise cancels"}
    for c in planning_limits(ir, rows, max_constraints):
        for label, rhs in moves(c):
            found = solve_best(with_rhs(ir, c.name, rhs), reads, seed)
            entry = {"constraint": c.name, "move": label, "rhs": rhs, "objective": None, "plan_changes": None}
            if found is not None:
                entry["objective"] = found[1]
                if base is not None:
                    entry["delta_objective"] = found[1] - base[1]
                    entry["plan_changes"] = int(np.sum(np.array(found[0]) != np.array(base[0])))
            out["sensitivity"].append(entry)
    return out
