"""Front end -> feedback loop (spec P5): decompose a verified build_qubo into objective + per-constraint
penalties + original-problem checks, verify the decomposition by brute force, build a ProblemInstance.

The Decomposer is the only LLM step; everything it returns is checked here against the already-verified
build_qubo and the test cases' ground truth before the loop may use it.
"""
from __future__ import annotations
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .problem import QUBO, ProblemInstance

_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "decomposer.txt").read_text(encoding="utf-8")
REQUIRED = ("SENSE", "build_objective", "build_penalties", "decision_variables", "objective_value",
            "check_constraints")


# ---------------------------------------------------------------- code handling
def extract_code(raw: str) -> str:
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


def compile_module(code: str) -> dict[str, Any]:
    ns: dict[str, Any] = {"np": np}
    exec(compile(code, "<decomposition>", "exec"), ns)  # noqa: S102 - LLM code, verified below before use
    missing = [k for k in REQUIRED if k not in ns]
    if missing:
        raise ValueError(f"decomposition is missing: {missing}")
    if ns["SENSE"] not in ("minimize", "maximize"):
        raise ValueError(f"SENSE must be 'minimize' or 'maximize', got {ns['SENSE']!r}")
    return ns


def _energy(q: np.ndarray, off: float, x: np.ndarray) -> float:
    return float(x @ q @ x) + float(off)


def _bits(n: int, max_enum: int = 14, samples: int = 4000, seed: int = 0):
    """All bitstrings for small n, random ones otherwise."""
    if n <= max_enum:
        return [np.array(b, dtype=float) for b in itertools.product((0, 1), repeat=n)]
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2, size=n).astype(float) for _ in range(samples)]


# ---------------------------------------------------------------- verification
def verify_decomposition(ns: dict[str, Any], build_qubo: Callable, cases: list) -> list[str]:
    """Return a list of problems (empty = verified). `cases` are verification.test_cases.TestCase objects."""
    errors: list[str] = []
    for case in cases:
        tag, inst = f"[{case.name}]", case.instance
        try:
            q_ref, off_ref = build_qubo(inst)
            q_ref, off_ref = np.asarray(q_ref, float), float(off_ref)
            q_obj, off_obj = ns["build_objective"](inst)
            q_obj = np.asarray(q_obj, float)
            pens = {k: (np.asarray(q, float), float(o), float(w)) for k, (q, o, w) in ns["build_penalties"](inst).items()}
        except Exception as e:
            errors.append(f"{tag} raised {type(e).__name__}: {e}")
            continue
        n = q_ref.shape[0]
        shapes = [q_obj.shape] + [q.shape for q, _, _ in pens.values()]
        if any(s != (n, n) for s in shapes):
            errors.append(f"{tag} matrix shapes {shapes} differ from build_qubo's {(n, n)}")
            continue
        xs = _bits(n)
        sum_err = max(abs(_energy(q_ref, off_ref, x) - _energy(q_obj, off_obj, x)
                          - sum(w * _energy(q, o, x) for q, o, w in pens.values())) for x in xs)
        if sum_err > 1e-6 * max(1.0, float(np.abs(q_ref).max())):
            errors.append(f"{tag} objective + sum(default_weight * penalty) differs from build_qubo by up to {sum_err:.4g}")
        dec = [int(i) for i in ns["decision_variables"](inst)]
        if not dec or max(dec) >= n or len(set(dec)) != len(dec):
            errors.append(f"{tag} decision_variables must be distinct indices in 0..{n - 1}, got {dec}")
            continue
        exhaustive = n <= 14
        for name, (q, o, w) in pens.items():
            # min over auxiliary variables of the penalty, per assignment of the decision variables
            best: dict[tuple, float] = {}
            for x in xs:
                key = tuple(int(x[i]) for i in dec)
                v = _energy(q, o, x)
                best[key] = min(v, best.get(key, np.inf))
            if min(best.values()) < -1e-6:
                errors.append(f"{tag} penalty '{name}' is negative for some x (min {min(best.values()):.4g})")
            for key, v in best.items():
                x = [0] * n
                for i, b in zip(dec, key):
                    x[i] = b
                checks = ns["check_constraints"](inst, x)
                if name not in checks:
                    errors.append(f"{tag} check_constraints has no entry '{name}'")
                    break
                if not checks[name] and v <= 1e-9:
                    errors.append(f"{tag} penalty '{name}' is 0 for decision values {list(key)} that violate '{name}'")
                    break
                if checks[name] and v > 1e-9 and exhaustive:
                    errors.append(f"{tag} penalty '{name}' cannot reach 0 for decision values {list(key)} that "
                                  f"SATISFY '{name}' (min penalty {v:.4g}): the encoding is stricter than the "
                                  f"constraint, e.g. an inequality encoded as an equality without slack")
                    break
        target = -case.optimal_value if ns["SENSE"] == "maximize" else case.optimal_value
        for bs in case.optimal_bitstrings[:5]:
            x = list(bs)
            checks = ns["check_constraints"](inst, x)
            if not all(checks.values()):
                errors.append(f"{tag} known optimum {x} violates {[k for k, v in checks.items() if not v]}")
            val = ns["objective_value"](inst, x)
            if not math.isclose(val, target, rel_tol=1e-6, abs_tol=1e-6):
                errors.append(f"{tag} objective_value(known optimum) = {val}, expected {target} "
                              f"(ground truth {case.optimal_value} is in QUBO/minimisation convention)")
    return list(dict.fromkeys(errors))[:12]


# ---------------------------------------------------------------- ProblemInstance
def _to_qubo(q: np.ndarray) -> QUBO:
    n = q.shape[0]
    out: QUBO = {}
    for i in range(n):
        for j in range(i, n):
            c = q[i, j] + (q[j, i] if j != i else 0.0)
            if c != 0:
                out[(i, j)] = float(c)
    return out


def to_problem_instance(ns: dict[str, Any], instance: dict, name: str, instance_id: str,
                        reference: float | None = None) -> ProblemInstance:
    q_obj, off_obj = ns["build_objective"](instance)
    pens = ns["build_penalties"](instance)
    names = list(pens)
    return ProblemInstance(
        name=name, n_variables=int(np.asarray(q_obj).shape[0]), sense=ns["SENSE"],
        objective=lambda x: float(ns["objective_value"](instance, list(x))),
        objective_qubo=_to_qubo(np.asarray(q_obj, float)), objective_offset=float(off_obj),
        constraints={k: (lambda x, k=k: bool(ns["check_constraints"](instance, list(x))[k])) for k in names},
        penalty_qubos={k: (_to_qubo(np.asarray(pens[k][0], float)), float(pens[k][1])) for k in names},
        default_penalty_weights={k: float(pens[k][2]) for k in names},
        constraint_types={k: "llm_decomposed" for k in names},
        reference_objective=reference, instance_id=instance_id,
    )


# ---------------------------------------------------------------- agent loop
def decompose(spec, formulation, build_qubo_code: str, build_qubo: Callable, cases: list, client, model: str,
              temperature: float = 0.0, max_attempts: int = 3) -> tuple[str, dict[str, Any], list[str]]:
    """Ask the Decomposer, verify, retry with the errors. Returns (code, namespace, remaining_errors)."""
    feedback, code, ns, errors = "", "", {}, ["not attempted"]
    sample = json.dumps(cases[0].instance, indent=1) if cases else "{}"
    for attempt in range(max_attempts):
        prompt = (_PROMPT.replace("{spec_json}", spec.model_dump_json(indent=1))
                  .replace("{formulation_json}", formulation.model_dump_json(indent=1))
                  .replace("{build_qubo_code}", build_qubo_code).replace("{sample_instance}", sample)
                  .replace("{previous_feedback}", feedback))
        raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="decompose")
        code = extract_code(raw)
        try:
            ns = compile_module(code)
            errors = verify_decomposition(ns, build_qubo, cases)
        except Exception as e:
            ns, errors = {}, [f"code failed to load: {type(e).__name__}: {e}"]
        if not errors:
            return code, ns, []
        feedback = ("PREVIOUS ATTEMPT FAILED VERIFICATION. Fix every problem:\n- " + "\n- ".join(errors))
    return code, ns, errors
