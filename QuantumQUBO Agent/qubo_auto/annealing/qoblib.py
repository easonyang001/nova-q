"""QOBLIB instances as ProblemInstances (benchmark source for the feedback loop).

QOBLIB — Quantum Optimization Benchmarking Library, Koch et al., Nature Computational Science
(arXiv:2504.03832), https://github.com/ZIB-AOPT/QOBLIB. Instance data CC BY 4.0, code Apache 2.0.
Changes made here: instances are re-encoded with one penalty term per constraint (so penalty
weights can be tuned separately); QOBLIB's own QUBO files are not used.

Expected location: $QOBLIB_DIR or /research/datasets/QOBLIB (a sparse clone is enough).
Reference values come from QOBLIB's .opt.sol (proven optimal) or .bst.sol (best known) files and
are checked against our formulation when the solution vector is available.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

from .problem import QUBO, ProblemInstance

ROOT = Path(os.environ.get("QOBLIB_DIR", "/research/datasets/QOBLIB"))


def _data_lines(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def _solution_file(cls_dir: Path, name: str) -> tuple[Path | None, str | None]:
    for kind in ("opt", "bst"):
        f = cls_dir / "solutions" / f"{name}.{kind}.sol"
        if f.exists():
            return f, kind
    return None, None


# ---------------------------------------------------------------- Market Split (01)
def market_split(name: str, root: Path = ROOT) -> ProblemInstance:
    """Find x in {0,1}^n with A x = b. One penalty (b_i - a_i.x)^2 per row; objective is 0 (feasibility)."""
    cls = root / "01-marketsplit"
    lines = _data_lines(cls / "instances" / f"{name}.dat")
    m, n = map(int, lines[0].split())
    rows = [list(map(int, l.split())) for l in lines[1:1 + m]]
    a, b = [r[:n] for r in rows], [r[n] for r in rows]

    penalties: dict[str, tuple[QUBO, float]] = {}
    for i in range(m):
        q: QUBO = {(j, j): float(a[i][j] ** 2 - 2 * b[i] * a[i][j]) for j in range(n) if a[i][j]}
        q.update({(j, k): float(2 * a[i][j] * a[i][k]) for j in range(n) for k in range(j + 1, n)
                  if a[i][j] and a[i][k]})
        penalties[f"row_{i}"] = (q, float(b[i] ** 2))

    def row_ok(i):
        return lambda x: sum(a[i][j] * x[j] for j in range(n)) == b[i]

    sol_file, kind = _solution_file(cls, name)
    reference = None
    if sol_file is not None:
        x = list(map(int, _data_lines(sol_file)[0].split()))
        assert len(x) == n and all(sum(a[i][j] * x[j] for j in range(n)) == b[i] for i in range(m)), \
            f"QOBLIB solution for {name} does not satisfy A x = b"
        reference = 0.0
    return ProblemInstance(
        name="market_split", n_variables=n, sense="minimize",
        objective=lambda x: 0.0, objective_qubo={},
        constraints={f"row_{i}": row_ok(i) for i in range(m)},
        penalty_qubos=penalties, default_penalty_weights={f"row_{i}": 1.0 for i in range(m)},
        constraint_types={f"row_{i}": "linear_equality" for i in range(m)},
        reference_objective=reference, instance_id=f"qoblib_ms_{name}",
    )


# ---------------------------------------------------------------- Maximum Independent Set (07)
def max_independent_set(name: str, root: Path = ROOT, weight: float = 2.0) -> ProblemInstance:
    """Max sum x_v s.t. no edge has both ends selected. Penalty x_u x_v per edge (QOBLIB uses weight 2)."""
    cls = root / "07-independentset"
    n, edges = 0, set()
    for line in _data_lines(cls / "instances" / f"{name}.gph"):
        parts = line.split()
        if parts[0] == "p":
            n = int(parts[2])
        elif parts[0] == "e":
            u, v = int(parts[1]) - 1, int(parts[2]) - 1          # DIMACS is 1-indexed
            if u != v:
                edges.add((min(u, v), max(u, v)))
    edges = sorted(edges)

    sol_file, kind = _solution_file(cls, name)
    reference = None
    if sol_file is not None:
        text = sol_file.read_text(encoding="utf-8")
        m = re.search(r"Objective value\s*=\s*([-\d.]+)", text)
        reference = float(m.group(1)) if m else None
        chosen = {int(l) - 1 for l in _data_lines(sol_file) if re.fullmatch(r"\d+", l)}
        if chosen and reference is not None:
            assert len(chosen) == reference and not any(u in chosen and v in chosen for u, v in edges), \
                f"QOBLIB solution for {name} is not an independent set of size {reference}"
    return ProblemInstance(
        name="max_independent_set", n_variables=n, sense="maximize",
        objective=lambda x: float(sum(x)), objective_qubo={(v, v): -1.0 for v in range(n)},
        constraints={"independence": lambda x: all(not (x[u] and x[v]) for u, v in edges)},
        penalty_qubos={"independence": ({e: 1.0 for e in edges}, 0.0)},
        default_penalty_weights={"independence": weight},
        constraint_types={"independence": "pairwise_exclusion"},
        reference_objective=reference, instance_id=f"qoblib_mis_{name}",
    )


def list_instances(cls: str, max_vars: int | None = None, root: Path = ROOT) -> list[tuple[str, int, str | None]]:
    """(name, n_variables, 'opt'|'bst'|None) for market_split / max_independent_set instances."""
    out = []
    if cls == "market_split":
        d = root / "01-marketsplit"
        for f in sorted((d / "instances").glob("*.dat")):
            n = int(_data_lines(f)[0].split()[1])
            out.append((f.stem, n, _solution_file(d, f.stem)[1]))
    elif cls == "max_independent_set":
        d = root / "07-independentset"
        for f in sorted((d / "instances").glob("*.gph")):
            header = next(l for l in _data_lines(f) if l.startswith("p"))
            out.append((f.stem, int(header.split()[2]), _solution_file(d, f.stem)[1]))
    else:
        raise ValueError(cls)
    return [t for t in out if max_vars is None or t[1] <= max_vars]
