"""Sensitivity report on a small knapsack and a small cardinality model (qubo_auto/annealing/sensitivity.py)."""
import itertools

import numpy as np

from qubo_auto.annealing.modeler import to_ir
from qubo_auto.annealing.sensitivity import binding_constraints, sensitivity_report, solve_best

V = [60, 100, 120, 70, 30, 90, 50, 40]
W = [10, 20, 30, 15, 5, 25, 12, 8]


def knapsack(budget):
    n = len(V)
    return to_ir({"name": "knap", "variable_names": [f"x{i}" for i in range(n)], "sense": "maximize",
                  "variable_keys": [{"items": str(i)} for i in range(n)], "linear": {i: float(V[i]) for i in range(n)},
                  "quadratic": [], "constraints": [{"name": "budget", "kind": "linear", "sense": "le", "rhs": float(budget),
                                                    "coeffs": {i: float(W[i]) for i in range(n)}}]}, "knap")


def exact(ir):
    best = max((x for x in itertools.product((0, 1), repeat=ir.n_variables) if all(c.satisfied(list(x)) for c in ir.constraints)),
               key=lambda x: ir.objective(list(x)))
    return list(best), ir.objective(list(best))


def test_solver_finds_the_enumerated_optimum():
    ir = knapsack(60)
    assert abs(solve_best(ir)[1] - exact(ir)[1]) < 1e-9


def test_binding_constraint_is_reported_and_loose_one_is_not():
    ir = knapsack(60)
    x, _ = exact(ir)
    lhs = sum(W[i] for i in range(len(W)) if x[i])
    rows = binding_constraints(ir, x)
    assert rows[0]["binding"] == (lhs == 60)
    loose = knapsack(10_000)
    assert binding_constraints(loose, exact(loose)[0])[0]["binding"] is False


def test_more_budget_never_hurts_and_less_never_helps():
    ir = knapsack(60)
    report = sensitivity_report(ir, exact(ir)[0])
    by = {r["move"]: r for r in report["sensitivity"]}
    assert report["resolved_objective"] == exact(ir)[1]
    if "budget" in report["binding"]:
        assert by["+10%"]["delta_objective"] >= -1e-9 and by["-10%"]["delta_objective"] <= 1e-9
        for r in report["sensitivity"]:
            target = knapsack(r["rhs"])
            assert abs(r["objective"] - exact(target)[1]) < 1e-9


def test_cardinality_limit_moves_by_one():
    n = 6
    ir = to_ir({"name": "pick", "variable_names": [f"x{i}" for i in range(n)], "sense": "maximize",
                "variable_keys": [{"s": str(i)} for i in range(n)], "linear": {i: float(10 + i) for i in range(n)},
                "quadratic": [], "constraints": [{"name": "k", "kind": "linear", "sense": "eq", "rhs": 3.0,
                                                  "coeffs": {i: 1.0 for i in range(n)}}]}, "pick")
    x, _ = exact(ir)
    report = sensitivity_report(ir, x)
    moves = {r["move"]: r for r in report["sensitivity"]}
    assert set(moves) == {"-1", "+1"}
    assert moves["+1"]["delta_objective"] == 10 + 2 and moves["-1"]["delta_objective"] == -(10 + 3)


def test_one_hot_rows_of_an_assignment_are_not_moved():
    """3 items x 2 slots, each item exactly once: local rows; only the global capacity is a planning limit."""
    idx = {(i, k): 2 * i + k for i in range(3) for k in range(2)}
    cons = [{"name": f"once_{i}", "kind": "linear", "sense": "eq", "rhs": 1.0, "coeffs": {idx[i, 0]: 1.0, idx[i, 1]: 1.0}}
            for i in range(3)]
    cons.append({"name": "slot0_cap", "kind": "linear", "sense": "le", "rhs": 1.0, "coeffs": {idx[i, 0]: 1.0 for i in range(3)}})
    ir = to_ir({"name": "assign", "variable_names": [f"x{v}" for v in range(6)], "sense": "maximize",
                "variable_keys": [{"i": str(v)} for v in range(6)],
                "linear": {idx[i, 0]: 5.0 for i in range(3)} | {idx[i, 1]: 1.0 for i in range(3)},
                "quadratic": [], "constraints": cons}, "assign")
    x, _ = exact(ir)
    moved = {r["constraint"] for r in sensitivity_report(ir, x)["sensitivity"]}
    assert moved == {"slot0_cap"}
