"""Small hand-built instances for exercising backends (oracle-QUBO condition)."""
from __future__ import annotations
import itertools
import math

import numpy as np

from .problem import QUBO, ProblemInstance


def milp_reference(c, a_ub=None, b_ub=None, a_eq=None, b_eq=None, maximize=False,
                   time_limit: float = 120.0) -> float | None:
    """Exact 0/1 optimum with scipy's HiGHS MILP (spec §6 reference_solve). None if not proven optimal."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    c = np.asarray(c, dtype=float)
    cons = []
    if a_ub is not None and len(a_ub):
        cons.append(LinearConstraint(np.asarray(a_ub, dtype=float), -np.inf, np.asarray(b_ub, dtype=float)))
    if a_eq is not None and len(a_eq):
        cons.append(LinearConstraint(np.asarray(a_eq, dtype=float), np.asarray(b_eq, dtype=float), np.asarray(b_eq, dtype=float)))
    res = milp(-c if maximize else c, constraints=cons, integrality=np.ones_like(c), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.status != 0:
        return None
    return float(round(-res.fun if maximize else res.fun, 9))


def select_k_with_conflicts(
    values: list[float] | None = None,
    k: int = 3,
    conflicts: list[tuple[int, int]] | None = None,
    weight_select: float = 10.0,
    weight_conflict: float = 10.0,
    instance_id: str = "select_k_aed8",
) -> ProblemInstance:
    """Pick exactly k items maximising total value; conflicting pairs can't both be picked.

    Same structure as the AED site-selection example used with CGA / Fujitsu / D-Wave.
    """
    values = values or [5, 4, 7, 3, 6, 2, 8, 4]
    conflicts = conflicts or [(0, 2), (2, 6), (4, 6), (1, 3), (5, 7)]
    n = len(values)

    # (sum x - k)^2 = (1 - 2k) sum x_i + 2 sum_{i<j} x_i x_j + k^2
    select: QUBO = {(i, i): 1.0 - 2 * k for i in range(n)}
    select.update({(i, j): 2.0 for i in range(n) for j in range(i + 1, n)})
    conflict: QUBO = {(min(i, j), max(i, j)): 1.0 for i, j in conflicts}

    inst = ProblemInstance(
        name="select_k_with_conflicts",
        n_variables=n,
        sense="maximize",
        objective=lambda x: float(sum(v * xi for v, xi in zip(values, x))),
        objective_qubo={(i, i): -float(v) for i, v in enumerate(values)},
        constraints={
            "select_k": lambda x: sum(x) == k,
            "conflicts": lambda x: all(not (x[i] and x[j]) for i, j in conflicts),
        },
        penalty_qubos={"select_k": (select, float(k * k)), "conflicts": (conflict, 0.0)},
        default_penalty_weights={"select_k": weight_select, "conflicts": weight_conflict},
        constraint_types={"select_k": "cardinality_equality", "conflicts": "pairwise_exclusion"},
        instance_id=instance_id,
    )
    inst.reference_objective = inst.brute_force_reference()
    return inst


def max_cut(edges: list[tuple[int, int, float]] | None = None, instance_id: str = "max_cut_case8") -> ProblemInstance:
    """Weighted Max-Cut (no constraints). Default graph = benchmarks/max_cut test case 8 (optimum 49)."""
    edges = edges or [(0, 1, 3), (0, 2, 8), (0, 7, 2), (1, 2, 5), (1, 3, 4), (2, 3, 6), (2, 4, 1),
                      (3, 5, 7), (4, 5, 2), (4, 6, 9), (5, 6, 3), (5, 7, 4), (6, 7, 5)]
    n = 1 + max(max(i, j) for i, j, _ in edges)

    # maximise sum w (x_i + x_j - 2 x_i x_j)  ->  minimise its negation
    obj: QUBO = {}
    for i, j, w in edges:
        obj[(i, i)] = obj.get((i, i), 0.0) - w
        obj[(j, j)] = obj.get((j, j), 0.0) - w
        obj[(min(i, j), max(i, j))] = obj.get((min(i, j), max(i, j)), 0.0) + 2 * w

    inst = ProblemInstance(
        name="max_cut",
        n_variables=n,
        sense="maximize",
        objective=lambda x: float(sum(w for i, j, w in edges if x[i] != x[j])),
        objective_qubo=obj,
        instance_id=instance_id,
    )
    inst.reference_objective = inst.brute_force_reference()
    return inst


def select_k_random(n: int = 30, k: int = 6, conflict_density: float = 0.15, seed: int = 0,
                    weight: float | None = None) -> ProblemInstance:
    """Scalable select-k-with-conflicts instance. Reference by exact enumeration of k-subsets."""
    rng = np.random.default_rng(seed)
    values = [int(v) for v in rng.integers(1, 21, size=n)]
    conflicts = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < conflict_density]
    w = weight if weight is not None else float(max(values)) + 1.0     # just above the single-flip bound
    inst = select_k_with_conflicts(values, k, conflicts, w, w, instance_id=f"select_k_n{n}_k{k}_d{conflict_density}_s{seed}")
    if math.comb(n, k) <= 200_000:
        inst.reference_objective = _best_k_subset(values, k, conflicts)
    else:
        a_ub = np.zeros((len(conflicts), n))
        for r, (i, j) in enumerate(conflicts):
            a_ub[r, i] = a_ub[r, j] = 1
        inst.reference_objective = milp_reference(values, a_ub, np.ones(len(conflicts)), np.ones((1, n)), [k], maximize=True)
    return inst


def _best_k_subset(values: list[int], k: int, conflicts: list[tuple[int, int]]) -> float | None:
    bad = set(conflicts)
    best = None
    for combo in itertools.combinations(range(len(values)), k):
        if any((a, b) in bad for a, b in itertools.combinations(combo, 2)):
            continue
        total = sum(values[i] for i in combo)
        best = total if best is None or total > best else best
    return None if best is None else float(best)


def knapsack_random(n: int = 12, capacity_ratio: float = 0.4, seed: int = 0,
                    weight: float | None = None) -> ProblemInstance:
    """0/1 knapsack with binary-encoded slack: sum w x + sum c s = C. Reference by DP.

    Variables 0..n-1 are items, n..n+m-1 are slack bits. The penalty has squared weights,
    so its coefficients grow much faster than the objective's (a realistic precision stress).
    """
    rng = np.random.default_rng(seed)
    w_items = [int(v) for v in rng.integers(1, 21, size=n)]
    values = [int(v) for v in rng.integers(1, 31, size=n)]
    cap = max(1, int(capacity_ratio * sum(w_items)))
    m = cap.bit_length()
    slack = [2 ** b for b in range(m - 1)] + [cap - (2 ** (m - 1) - 1)]   # sums to exactly 0..cap
    a = w_items + slack
    total = n + m

    # (a.z - C)^2 = sum (a_i^2 - 2 C a_i) z_i + sum_{i<j} 2 a_i a_j z_i z_j + C^2
    pen: QUBO = {(i, i): float(a[i] ** 2 - 2 * cap * a[i]) for i in range(total)}
    pen.update({(i, j): float(2 * a[i] * a[j]) for i in range(total) for j in range(i + 1, total)})

    best = [0] * (cap + 1)                                   # DP over capacity
    for wi, vi in zip(w_items, values):
        for c in range(cap, wi - 1, -1):
            best[c] = max(best[c], best[c - wi] + vi)

    inst = ProblemInstance(
        name="knapsack",
        n_variables=total,
        sense="maximize",
        objective=lambda x: float(sum(v * xi for v, xi in zip(values, x[:n]))),
        objective_qubo={(i, i): -float(v) for i, v in enumerate(values)},
        constraints={"capacity": lambda x: sum(wi * xi for wi, xi in zip(w_items, x[:n])) <= cap},
        penalty_qubos={"capacity": (pen, float(cap * cap))},
        default_penalty_weights={"capacity": weight if weight is not None else float(max(values)) + 1.0},
        constraint_types={"capacity": "inequality_with_binary_slack"},
        reference_objective=float(best[cap]),
        instance_id=f"knapsack_n{n}_r{capacity_ratio}_s{seed}",
    )
    return inst


def graph_coloring_random(n: int = 10, k: int = 3, edge_prob: float = 0.4, seed: int = 0,
                          weight: float | None = None) -> ProblemInstance:
    """Min-cost graph colouring: one-hot x[v,c] (index v*k + c), random cost per (node, colour).

    Edges are only drawn between nodes of different colours in a planted colouring, so a feasible
    colouring always exists. Reference by exhaustive search over k^n assignments.
    """
    rng = np.random.default_rng(seed)
    planted = rng.integers(0, k, size=n)
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)
             if planted[u] != planted[v] and rng.random() < edge_prob]
    cost = rng.integers(1, 11, size=(n, k)).astype(float)
    idx = lambda v, c: v * k + c

    one: QUBO = {}                                # sum_v (sum_c x_vc - 1)^2  (offset n)
    for v in range(n):
        for c in range(k):
            one[(idx(v, c), idx(v, c))] = -1.0
            for c2 in range(c + 1, k):
                one[(idx(v, c), idx(v, c2))] = 2.0
    edge: QUBO = {(idx(u, c), idx(v, c)): 1.0 for u, v in edges for c in range(k)}

    def colours(x):
        return [[c for c in range(k) if x[idx(v, c)]] for v in range(n)]

    if k ** n <= 200_000:
        best = None
        for assign in itertools.product(range(k), repeat=n):
            if any(assign[u] == assign[v] for u, v in edges):
                continue
            total = sum(cost[v, assign[v]] for v in range(n))
            best = total if best is None or total < best else best
    else:
        a_eq = np.zeros((n, n * k))
        for v in range(n):
            a_eq[v, v * k:(v + 1) * k] = 1
        a_ub = np.zeros((len(edges) * k, n * k))
        for r, ((u, v), c) in enumerate(itertools.product(edges, range(k))):
            a_ub[r, idx(u, c)] = a_ub[r, idx(v, c)] = 1
        best = milp_reference(cost.ravel(), a_ub, np.ones(len(a_ub)), a_eq, np.ones(n))

    inst = ProblemInstance(
        name="graph_coloring",
        n_variables=n * k,
        sense="minimize",
        objective=lambda x: float(sum(cost[v, c] * x[idx(v, c)] for v in range(n) for c in range(k))),
        objective_qubo={(idx(v, c), idx(v, c)): float(cost[v, c]) for v in range(n) for c in range(k)},
        constraints={
            "one_color": lambda x: all(len(cs) == 1 for cs in colours(x)),
            "edge_conflict": lambda x: all(not (x[idx(u, c)] and x[idx(v, c)]) for u, v in edges for c in range(k)),
        },
        penalty_qubos={"one_color": (one, float(n)), "edge_conflict": (edge, 0.0)},
        default_penalty_weights={"one_color": weight or 11.0, "edge_conflict": weight or 11.0},
        constraint_types={"one_color": "one_hot", "edge_conflict": "pairwise_exclusion"},
        reference_objective=None if best is None else float(best),
        instance_id=f"graph_coloring_n{n}_k{k}_p{edge_prob}_s{seed}",
    )
    return inst


def _qoblib(fn_name: str):
    def load(name: str, **kw):
        from . import qoblib
        return getattr(qoblib, fn_name)(name, **kw)
    return load


EXAMPLES = {"select_k": select_k_with_conflicts, "max_cut": max_cut,
            "select_k_random": select_k_random, "knapsack": knapsack_random,
            "graph_coloring": graph_coloring_random,
            "qoblib_market_split": _qoblib("market_split"), "qoblib_mis": _qoblib("max_independent_set")}


def knapsack_ir(n: int = 12, capacity_ratio: float = 0.4, seed: int = 0):
    """The same instance as knapsack_random(n, capacity_ratio, seed), as a typed IR (for the encoding library)."""
    from .encodings import LinearConstraint, ProblemIR
    rng = np.random.default_rng(seed)
    w_items = [int(v) for v in rng.integers(1, 21, size=n)]
    values = [int(v) for v in rng.integers(1, 31, size=n)]
    cap = max(1, int(capacity_ratio * sum(w_items)))
    best = [0] * (cap + 1)
    for wi, vi in zip(w_items, values):
        for c in range(cap, wi - 1, -1):
            best[c] = max(best[c], best[c - wi] + vi)
    return ProblemIR(name="knapsack", instance_id=f"knapsack_n{n}_r{capacity_ratio}_s{seed}", n_variables=n,
                     sense="maximize", linear={i: float(v) for i, v in enumerate(values)},
                     constraints=[LinearConstraint(name="capacity", coeffs={i: float(w) for i, w in enumerate(w_items)},
                                                   sense="le", rhs=float(cap))],
                     reference_objective=float(best[cap]))


IR_EXAMPLES = {"knapsack_ir": knapsack_ir}     # problems given as a typed IR: encodings can change in the loop
