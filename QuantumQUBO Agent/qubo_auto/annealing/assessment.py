"""Practical feasibility and quality checks; an optimum is not required.

An infeasibility finding is about the supplied model, not proof that the user's
real-world request is impossible. Unknown/time-limited results never imply infeasibility.
"""
from __future__ import annotations
import math
import random

from .encodings import LinearConstraint, ProblemIR
from .runtime import run_task


def _milp(ir: ProblemIR, time_limit: float):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint as ScipyConstraint, milp
    from scipy.sparse import coo_matrix
    rows, cols, vals, lo, hi = [], [], [], [], []
    def row(coeffs, lower, upper):
        j = len(lo)
        for i, v in coeffs.items():
            rows.append(j); cols.append(i); vals.append(v)
        lo.append(lower); hi.append(upper)
    for c in ir.constraints:
        if isinstance(c, LinearConstraint):
            row(c.coeffs, c.rhs if c.sense in ("eq", "ge") else -np.inf,
                c.rhs if c.sense in ("eq", "le") else np.inf)
        else:
            for i, j in c.pairs:
                if c.kind == "exclusion":
                    coeffs = {i: 1.0}
                    coeffs[j] = coeffs.get(j, 0.0) + 1.0
                    row(coeffs, -np.inf, 1.0)
                else:
                    coeffs = {i: 1.0}
                    coeffs[j] = coeffs.get(j, 0.0) - 1.0
                    row(coeffs, -np.inf, 0.0)
    objective = np.zeros(ir.n_variables)
    if not ir.quadratic:
        for i, v in ir.linear.items():
            objective[i] = v if ir.sense == "minimize" else -v
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lo), ir.n_variables)).tocsc()
    result = milp(objective, integrality=np.ones(ir.n_variables), bounds=Bounds(0, 1),
                  constraints=ScipyConstraint(matrix, lo, hi), options={"time_limit": time_limit})
    x = None if result.x is None else [int(v >= 0.5) for v in result.x]
    return {"status": int(result.status), "solution": x, "message": str(result.message)}


def _exchange_improve(ir: ProblemIR, x: list, feasible, sign: int, max_rounds: int = 50, max_ones: int = 400) -> list:
    """Constraint-preserving exchange moves for assignment-like models (tours, groups, p-median assignments).

    One- and two-bit flips cannot leave a feasible point when every row and column of an assignment is fixed (the
    TSP baseline stayed 28% above nearest-neighbour + 2-opt, Melbourne 2026-10-21). An exchange takes two chosen
    variables p, q and two unchosen r, s, where r shares a constraint with p and with q, and s does too, and flips
    all four (for a position tour: two stops swap positions). Only the constraint structure is used, so it applies to
    any model; every candidate is checked on the original constraints. Best improvement, bounded rounds."""
    groups = [set(c.coeffs) for c in ir.constraints if isinstance(c, LinearConstraint) and c.sense == "eq"]
    member: dict[int, list[int]] = {}
    for g, vars_ in enumerate(groups):
        for v in vars_:
            member.setdefault(v, []).append(g)
    if not groups:
        return x
    best, best_val = list(x), sign * ir.objective(x)
    for _ in range(max_rounds):
        ones = [i for i, v in enumerate(best) if v == 1 and i in member][:max_ones]
        move, move_val = None, best_val
        for a in range(len(ones)):
            p = ones[a]
            gp = set(member[p])
            for b in range(a + 1, len(ones)):
                q = ones[b]
                gq = set(member[q])
                if gp & gq:
                    continue                                      # same row/column: not an exchange pair
                # unchosen variables sharing a constraint with both p and q
                cand = {v for g in gp for v in groups[g] if best[v] == 0 and set(member.get(v, [])) & gq}
                for r in cand:
                    for s in cand:
                        if s <= r or set(member[r]) & set(member[s]):
                            continue
                        y = list(best)
                        y[p] = y[q] = 0
                        y[r] = y[s] = 1
                        if not feasible(y):
                            continue
                        val = sign * ir.objective(y)
                        if val < move_val - 1e-9:
                            move, move_val = y, val
        if move is None:
            break
        best, best_val = move, move_val
    return best


def inspect_problem(ir: ProblemIR, time_limit: float = 2, local_moves: int = 200) -> dict:
    issues = []
    coefficients = list(ir.linear.values()) + list(ir.quadratic.values()) + [ir.constant]
    for c in ir.constraints:
        if isinstance(c, LinearConstraint):
            coefficients += list(c.coeffs.values()) + [c.rhs]
    if ir.n_variables < 1 or any(not math.isfinite(v) for v in coefficients):
        return {"feasibility": "invalid", "issues": ["模型沒有決策變數或包含 NaN／無限大；先檢查資料與建模"],
                "baseline_solution": None, "baseline_objective": None, "proven_optimal": False}
    for c in ir.constraints:
        if not isinstance(c, LinearConstraint):
            continue
        lower = sum(min(0.0, v) for v in c.coeffs.values())
        upper = sum(max(0.0, v) for v in c.coeffs.values())
        if ((c.sense in ("eq", "ge") and c.rhs > upper + 1e-9)
                or (c.sense in ("eq", "le") and c.rhs < lower - 1e-9)):
            issues.append(f"限制 {c.name} 要求 {c.sense} {c.rhs:g}，但目前資料最多／最少只能達到 [{lower:g}, {upper:g}]；"
                          "請檢查容量、單位或需求，增加 penalty 無法解決")
    if issues:
        return {"feasibility": "infeasible", "issues": issues, "baseline_solution": None,
                "baseline_objective": None, "proven_optimal": False}
    try:
        result = run_task(__name__, "_milp", ir, time_limit, timeout=time_limit + 10)
    except Exception as e:
        result = {"status": 4, "solution": None, "message": str(e)}
    if result["status"] == 2:
        intervals = {}
        for constraint in ir.constraints:
            if not isinstance(constraint, LinearConstraint):
                continue
            key = tuple(sorted(constraint.coeffs.items()))
            bucket = intervals.setdefault(key, [])
            lower = constraint.rhs if constraint.sense in ("eq", "ge") else -math.inf
            upper = constraint.rhs if constraint.sense in ("eq", "le") else math.inf
            for other, other_lower, other_upper in bucket:
                if max(lower, other_lower) > min(upper, other_upper) + 1e-9:
                    issues.append(f"具體衝突：{other} 與 {constraint.name} 對同一組變數要求互斥範圍")
            bucket.append((constraint.name, lower, upper))
        issues.append("目前模型的限制互相衝突，獨立可行性檢查確認無解；先回查資料與模型，不再盲目調 penalty。"
                      "系統不會自行刪除已確認的需求")
    def feasible(x):
        return all(c.satisfied(x) for c in ir.constraints)
    candidates = [x for x in (result["solution"], [0] * ir.n_variables, [1] * ir.n_variables)
                  if x is not None and feasible(x)]
    rng = random.Random(0)
    sign = 1 if ir.sense == "minimize" else -1
    best = min(candidates, key=lambda x: sign * ir.objective(x)) if candidates else None
    # Cheap improvement for quadratic models, with a fixed work bound. Always check original constraints.
    for _ in range(local_moves if best is not None and ir.quadratic else 0):
        x = list(best)
        for i in rng.sample(range(ir.n_variables), min(ir.n_variables, rng.choice((1, 2)))):
            x[i] = 1 - x[i]
        if feasible(x) and sign * ir.objective(x) < sign * ir.objective(best):
            best = x
    if best is not None and ir.quadratic:
        best = _exchange_improve(ir, best, feasible, sign)
    status = "infeasible" if result["status"] == 2 else "feasible" if best is not None else "unknown"
    if status == "unknown":
        issues.append("限時檢查尚未找到可行方案；這不代表無解，可以繼續求解")
    return {"feasibility": status, "issues": issues, "baseline_solution": best,
            "baseline_objective": ir.objective(best) if best is not None else None,
            "proven_optimal": result["status"] == 0 and not ir.quadratic and best is not None,
            "method": "time-limited HiGHS + bounded local improvement", "solver_message": result["message"]}


def evaluate_delivery(ir: ProblemIR, result, inspection: dict, tolerance: float = 0.01) -> dict:
    """Validate the delivered solution against this exact IR and compare with a feasible baseline."""
    x = result.best_solution
    valid = (x is not None and len(x) >= ir.n_variables
             and all(b in (0, 1) for b in x[:ir.n_variables]))
    failed = [c.name for c in ir.constraints if valid and not c.satisfied(x)]
    if not valid or failed:
        return {"status": "no_solution", "message": "尚未取得可交付的可行方案",
                "next_stage": "solve" if inspection["feasibility"] == "feasible" else "diagnose",
                "evidence": failed or inspection["issues"], "baseline_objective": inspection.get("baseline_objective"),
                "quality_gap": None}
    objective = ir.objective(x)
    baseline = inspection.get("baseline_objective")
    gap = None if baseline is None else ((objective - baseline) if ir.sense == "minimize" else (baseline - objective)) / max(abs(baseline), 1.0)
    limited = gap is None or gap > tolerance
    return {"status": "limited" if limited else "usable", "objective": objective,
            "message": "已有可行方案，品質尚未確認或仍落後簡單基準" if limited else "已有可行方案，達到簡單基準的品質",
            "next_stage": "solve" if gap is not None and gap > tolerance else "deliver",
            "evidence": [], "baseline_objective": baseline, "quality_gap": gap}
