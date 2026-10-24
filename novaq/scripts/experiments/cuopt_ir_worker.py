"""Solve one agent-built model (its IR, before any QUBO penalty) with NVIDIA cuOpt MILP. Runs in the cuOpt env.

The IR is binary variables, a linear + quadratic objective and linear / exclusion / implication constraints. cuOpt
takes no QUBO, so each product x_i x_j becomes a continuous y_ij with the standard linearisation
(y <= x_i, y <= x_j, y >= x_i + x_j - 1): the MILP has exactly the feasible set and objective of the agent's model.

    /research/conda-envs/cuopt-env/bin/python cuopt_ir_worker.py in.json out.json [time_limit]
"""
import json
import sys
import time


def main():
    import rmm
    rmm.reinitialize(managed_memory=True)            # the LLM servers hold most of each GPU
    from cuopt.linear_programming.problem import CONTINUOUS, INTEGER, MAXIMIZE, MINIMIZE, LinearExpression, Problem
    from cuopt.linear_programming.solver_settings import SolverSettings
    ir = json.load(open(sys.argv[1]))
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    prob = Problem("nova-q-ir")
    x = [prob.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"x{i}") for i in range(ir["n_variables"])]
    ov, oc = [], []                                  # objective as one LinearExpression (thousands of terms)
    for i, c in ir["linear"]:
        ov.append(x[i]); oc.append(float(c))
    for i, j, c in ir["quadratic"]:
        if i == j:                                  # x^2 = x for a binary variable
            ov.append(x[i]); oc.append(float(c))
            continue
        y = prob.addVariable(lb=0, ub=1, vtype=CONTINUOUS)
        prob.addConstraint(LinearExpression([y, x[i]], [1.0, -1.0], 0.0) <= 0)
        prob.addConstraint(LinearExpression([y, x[j]], [1.0, -1.0], 0.0) <= 0)
        prob.addConstraint(LinearExpression([y, x[i], x[j]], [1.0, -1.0, -1.0], 0.0) >= -1)
        ov.append(y); oc.append(float(c))
    obj = LinearExpression(ov, oc, float(ir.get("constant", 0.0)))
    for con in ir["constraints"]:
        if con["kind"] == "linear":
            items = [(int(i), float(c)) for i, c in con["coeffs"].items()]
            lhs = LinearExpression([x[i] for i, _ in items], [c for _, c in items], 0.0)
            rhs = float(con["rhs"])
            prob.addConstraint(lhs == rhs if con["sense"] == "eq" else lhs <= rhs if con["sense"] == "le" else lhs >= rhs)
        elif con["kind"] == "exclusion":
            for i, j in con["pairs"]:
                prob.addConstraint(LinearExpression([x[i], x[j]], [1.0, 1.0], 0.0) <= 1)
        elif con["kind"] == "implication":
            for i, j in con["pairs"]:
                prob.addConstraint(LinearExpression([x[i], x[j]], [1.0, -1.0], 0.0) <= 0)
    prob.setObjective(obj, sense=MAXIMIZE if ir["sense"] == "maximize" else MINIMIZE)
    s = SolverSettings()
    s.set_parameter("time_limit", limit)
    t = time.time()
    prob.solve(s)
    sol = [int(round(v.getValue())) for v in x]
    json.dump({"objective": float(prob.ObjValue), "seconds": time.time() - t, "status": str(prob.Status),
               "solution": sol}, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
