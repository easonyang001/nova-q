"""Problem-level verification: judge a QUBO by what its ground states MEAN, not by their bit layout.

The original test (test_runner.run_tests) compares the QUBO's argmin bitstrings with ground-truth bitstrings
that an LLM derived from the benchmark text. That breaks whenever the LLM miscounts variables, fills slack
bits wrongly, or copies an inconsistent reference solution, and it rejects any correct encoding that lays the
variables out differently. Here instead:

  * a test case holds the instance, the benchmark's stated optimal VALUE (human-written, in the problem's own
    sign convention) and, when the benchmark gives one, a reference solution in problem terms;
  * the Coder writes decode(instance, x) -> solution next to build_qubo;
  * a Checker, written from the problem statement alone (it never sees the formulation or the code), writes
    evaluate(instance, solution) -> (feasible, objective);
  * the Checker is trusted only after it reproduces the stated optimal value on the reference solution;
  * a case passes when EVERY ground state of the QUBO decodes to a feasible solution with the stated optimal
    value (a ground state the solver may return must never be a wrong answer).
"""
from __future__ import annotations
import json
import math
import random
import signal
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..schemas import TestInstanceResult
from .brute_force import validate_Q
from .test_runner import MAX_BRUTE_FORCE_VARS, _enumerate_minima

VALUE_TOL = 1e-6
MAX_GROUND_STATES = 4096          # decoded per case; more means a massively degenerate QUBO (reported, not decoded)
CALL_TIMEOUT_S = 30               # build_qubo / decode / evaluate are LLM code: never let one hang the run


@dataclass
class ReferenceCase:
    name: str
    description: str
    instance: dict
    optimal_value: float                      # problem convention (maximisation values stay positive)
    reference_solution: Any | None = None     # the benchmark's stated optimal solution, in problem terms


def save_reference_cases(cases: list[dict], solution_format: str, cases_dir: Path) -> list[Path]:
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "solution_format.txt").write_text(solution_format, encoding="utf-8")
    paths = []
    for i, c in enumerate(cases):
        name = c.get("name") or f"case_{i + 1}"
        p = cases_dir / f"{name}.json"
        p.write_text(json.dumps({"name": name, "description": c.get("description", ""), "instance": c["instance"],
                                 "optimal_value": float(c["optimal_value"]),
                                 "reference_solution": c.get("reference_solution")}, indent=2), encoding="utf-8")
        paths.append(p)
    return paths


def load_reference_cases(cases_dir: Path) -> tuple[list[ReferenceCase], str]:
    fmt_path = cases_dir / "solution_format.txt"
    fmt = fmt_path.read_text(encoding="utf-8") if fmt_path.exists() else ""
    cases = []
    for p in sorted(cases_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        cases.append(ReferenceCase(name=d.get("name", p.stem), description=d.get("description", ""),
                                   instance=d["instance"], optimal_value=float(d["optimal_value"]),
                                   reference_solution=d.get("reference_solution")))
    return cases, fmt


# ---------------------------------------------------------------- guarded calls into LLM-written code
class _Timeout(Exception):
    pass


def _call(fn: Callable, *args, timeout: int = CALL_TIMEOUT_S):
    def _raise(*_):
        raise _Timeout(f"took longer than {timeout}s")
    use_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _raise)
        signal.alarm(timeout)
    try:
        return fn(*args)
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def _evaluate(evaluate: Callable, instance: dict, solution) -> tuple[bool, float]:
    out = _call(evaluate, instance, solution)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        raise TypeError(f"evaluate must return (feasible, objective), got {out!r}"[:200])
    feasible, objective = bool(out[0]), float(out[1])
    return feasible, objective


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=VALUE_TOL)


# ---------------------------------------------------------------- checker validation
def validate_checker(evaluate: Callable, cases: list[ReferenceCase]) -> tuple[list[bool], list[str]]:
    """Which cases the Checker can be trusted on: its evaluate() must give the stated optimal value for the
    stated optimal solution. A case without a reference solution cannot anchor the Checker and is not trusted."""
    trusted, notes = [], []
    for c in cases:
        if c.reference_solution is None:
            trusted.append(False)
            notes.append(f"[{c.name}] no reference solution in the benchmark: the checker cannot be anchored")
            continue
        try:
            feasible, obj = _evaluate(evaluate, c.instance, c.reference_solution)
        except Exception as e:
            trusted.append(False)
            notes.append(f"[{c.name}] evaluate() raised on the reference solution: {type(e).__name__}: {str(e)[:200]}")
            continue
        ok = feasible and _same(obj, c.optimal_value)
        trusted.append(ok)
        if not ok:
            notes.append(f"[{c.name}] evaluate() says the benchmark's reference solution is "
                         f"{'feasible' if feasible else 'INFEASIBLE'} with objective {obj:.6g}, but the benchmark "
                         f"states it is optimal with objective {c.optimal_value:.6g}")
    return trusted, notes


# ---------------------------------------------------------------- the test itself
def run_reference_tests(
    build_qubo: Callable[[dict], tuple],
    decode: Callable[[dict, list], Any],
    evaluate: Callable[[dict, Any], tuple],
    cases: list[ReferenceCase],
) -> tuple[list[TestInstanceResult], str]:
    """Same return contract as test_runner.run_tests: (results, "success" | "coding_error" |
    "formulation_error" | "unverifiable"). Only cases the Checker was validated on should be passed in."""
    results = [_run_case(build_qubo, decode, evaluate, c, i) for i, c in enumerate(cases)]
    if all(r.match for r in results):
        return results, "success"
    coding = sum(r.status == "coding_error" for r in results)
    formulation = sum(r.status == "formulation_error" for r in results)
    if coding == formulation == 0:
        return results, "unverifiable"
    return results, "coding_error" if coding >= formulation else "formulation_error"


def _run_case(build_qubo, decode, evaluate, case: ReferenceCase, idx: int) -> TestInstanceResult:
    def fail(status: str, msg: str, **kw) -> TestInstanceResult:
        return TestInstanceResult(instance_id=idx, status=status, error_message=f"[{case.name}] {msg}",
                                  problem_optimum=case.optimal_value, match=False, **kw)
    try:
        Q, offset = _call(build_qubo, case.instance)
        Q = np.asarray(Q, dtype=float)
        offset = float(offset) if isinstance(offset, (int, float, np.floating, np.integer)) else 0.0
    except Exception:
        return fail("coding_error", f"build_qubo raised: {traceback.format_exc()[-600:]}")
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        return fail("coding_error", f"Q must be a square matrix, got shape {Q.shape}")
    n = Q.shape[0]
    q_errors = validate_Q(Q, n)
    if q_errors:
        return fail("coding_error", "; ".join(q_errors))
    if n > MAX_BRUTE_FORCE_VARS:
        return fail("unverifiable", f"{n} variables is too many for brute-force verification "
                                    f"(limit {MAX_BRUTE_FORCE_VARS}); the test case is too large, not a code error")

    qubo_min, ground = _enumerate_minima(Q, offset, n)
    if len(ground) > MAX_GROUND_STATES:
        ground = set(random.Random(0).sample(sorted(ground), MAX_GROUND_STATES))
    bad: list[str] = []
    for x in sorted(ground):
        try:
            sol = _call(decode, case.instance, list(x))
        except Exception as e:
            return fail("coding_error", f"decode raised on a ground state: {type(e).__name__}: {str(e)[:300]}")
        try:
            feasible, obj = _evaluate(evaluate, case.instance, sol)
        except Exception as e:
            return fail("coding_error", f"decode returned {str(sol)[:200]!s}, which is not a valid solution "
                                        f"in the required format ({type(e).__name__}: {str(e)[:200]})")
        if not feasible:
            bad.append(f"a ground state decodes to an INFEASIBLE solution: {json.dumps(sol, default=str)[:300]}")
        elif not _same(obj, case.optimal_value):
            bad.append(f"a ground state decodes to a feasible but NON-OPTIMAL solution (objective {obj:.6g}): "
                       f"{json.dumps(sol, default=str)[:300]}")
        if len(bad) >= 2:
            break
    if bad:
        # the expected optimum is withheld, as in the original runner: feedback must not leak the answer
        return fail("formulation_error",
                    f"the QUBO minimum (energy {qubo_min:.6g}, {len(ground)} ground state(s)) is not the correct "
                    f"optimal solution: " + " | ".join(bad) + ". Check penalty weights, objective sign, "
                    f"variable indexing and decode().", qubo_optimum=qubo_min)
    return TestInstanceResult(instance_id=idx, status="success", qubo_optimum=qubo_min,
                              problem_optimum=case.optimal_value, match=True)
