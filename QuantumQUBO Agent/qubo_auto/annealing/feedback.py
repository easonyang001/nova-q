"""Feedback Extractor: turn any backend's samples into one backend-agnostic set of metrics.

Improvement between rounds is judged on the shared metrics only (original-problem objective,
feasibility). QUBO energy is reported for diagnosis but is not comparable across rounds,
because changing penalty weights changes the energy landscape.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .backends import SolverRunResult
from .problem import ProblemInstance


class Feedback(BaseModel):
    problem: str
    backend: str
    simulated: bool
    n_samples: int                                   # total reads (with occurrences)
    n_unique: int
    feasibility_rate: float
    constraint_violation_rates: dict[str, float] = Field(default_factory=dict)
    best_feasible_objective: float | None = None
    mean_feasible_objective: float | None = None
    best_feasible_solution: list[int] | None = None
    reference_objective: float | None = None
    gap: float | None = None                         # relative gap to reference; 0 = optimal
    diversity: float                                 # n_unique / n_samples
    best_energy: float                               # diagnostic only
    wall_time_s: float
    solver_time_s: float | None = None
    backend_specific: dict[str, Any] = Field(default_factory=dict)


def extract_feedback(problem: ProblemInstance, result: SolverRunResult) -> Feedback:
    total = sum(result.num_occurrences)
    violations = {name: 0 for name in problem.constraints}
    feasible: list[tuple[float, int, list[int]]] = []   # (objective, occurrences, x)

    for x, occ in zip(result.samples, result.num_occurrences):
        ok = True
        for name, check in problem.constraints.items():
            if not check(x):
                violations[name] += occ
                ok = False
        if ok:
            feasible.append((problem.objective(x), occ, x))

    n_feasible = sum(occ for _, occ, _ in feasible)
    best = mean = best_x = gap = None
    if feasible:
        pick = max if problem.sense == "maximize" else min
        best, _, best_x = pick(feasible, key=lambda t: t[0])
        mean = sum(obj * occ for obj, occ, _ in feasible) / n_feasible
        ref = problem.reference_objective
        if ref is not None:
            diff = (ref - best) if problem.sense == "maximize" else (best - ref)
            gap = diff / abs(ref) if ref != 0 else diff

    return Feedback(
        problem=problem.name,
        backend=result.backend,
        simulated=result.simulated,
        n_samples=total,
        n_unique=len(result.samples),
        feasibility_rate=n_feasible / total if total else 0.0,
        constraint_violation_rates={k: v / total for k, v in violations.items()} if total else {},
        best_feasible_objective=best,
        mean_feasible_objective=mean,
        best_feasible_solution=best_x,
        reference_objective=problem.reference_objective,
        gap=gap,
        diversity=len(result.samples) / total if total else 0.0,
        best_energy=min(result.energies),
        wall_time_s=result.wall_time_s,
        solver_time_s=result.solver_time_s,
        backend_specific=result.metadata,
    )


def log_run(path: Path, result: SolverRunResult, feedback: Feedback, extra: dict[str, Any] | None = None) -> None:
    """Append one execution (config, raw samples, metrics) to a JSONL log for replay/analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": time.time(), **(extra or {}),
              "result": result.model_dump(), "feedback": feedback.model_dump()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
