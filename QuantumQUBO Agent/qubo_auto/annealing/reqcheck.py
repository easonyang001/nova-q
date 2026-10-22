"""Requirement-level validation for real-data models: an independent Checker agent against the Modeling Agent.

The Checker sees only the confirmed specification and the data declaration, never the model. It writes
evaluate(data, decisions) and sample_decisions(data, rng). Decisions are dicts of set ids, the same language as
the model's variable_keys, so both can be compared on the same instance:

- before solving: random valid decisions must be feasible in the model (helper variables optimised by MILP), the
  model must rank them like the Checker's objective, and decisions the Checker rejects (one removed, one added)
  must be infeasible in the model (catches a relaxed bound such as "at most 200" for "exactly 5");
- after solving: the delivered decisions must satisfy the Checker.

A disagreement is evidence for the Modeling Agent, not a verdict: the Checker can be wrong too (the benchmark
audit found wrong LLM-written checks), so the independent acceptance judge, not this module, decides success.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from . import schema
from .bridge import extract_code
from .runtime import run_task

_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "req_checker.txt").read_text(encoding="utf-8")
SAMPLES = 20
RANK_MIN = 0.8


# ---------------------------------------------------------------- the model with decisions fixed

def parse(raw: dict) -> dict:
    names = [str(v) for v in raw["variable_names"]]
    by_name = {v: i for i, v in enumerate(names)}

    def idx(k):
        if isinstance(k, int):
            return k
        k = str(k)
        return int(k) if k.lstrip("-").isdigit() else by_name[k]

    quad = {}
    for i, j, v in raw.get("quadratic", []):
        a, b = sorted((idx(i), idx(j)))
        quad[(a, b)] = quad.get((a, b), 0.0) + float(v)
    cons = []
    for c in raw.get("constraints", []):
        if c["kind"] == "linear":
            cons.append(("linear", {idx(k): float(v) for k, v in c["coeffs"].items()}, c["sense"], float(c["rhs"])))
        else:
            cons.append((c["kind"], [(idx(a), idx(b)) for a, b in c["pairs"]], None, None))
    keys = raw.get("variable_keys") or [{}] * len(names)
    return {"n": len(names), "sense": raw["sense"], "lin": {idx(k): float(v) for k, v in raw.get("linear", {}).items()},
            "quad": quad, "const": float(raw.get("constant", 0)), "cons": cons, "keys": keys,
            "index": {_kt(k): i for i, k in enumerate(keys) if k}}


def _kt(k: dict) -> tuple:
    return tuple(sorted((str(s), str(v)) for s, v in k.items()))


def fixed_value(m: dict, decisions: list[dict], time_limit: float = 20.0):
    """Model objective for exactly these decisions (other decision variables 0, helpers optimised).
    Returns (value | None, note); None = infeasible or not representable."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    chosen = set()
    for d in decisions:
        t = _kt(d)
        if t not in m["index"]:
            return None, f"no model variable for decision {d}"
        chosen.add(m["index"][t])
    fixed = {i: int(i in chosen) for i in m["index"].values()}
    free = [i for i in range(m["n"]) if i not in fixed]
    pos = {v: k for k, v in enumerate(free)}
    const = m["const"] + sum(m["lin"].get(i, 0) * b for i, b in fixed.items())
    c = np.zeros(len(free))
    for i, v in m["lin"].items():
        if i in pos:
            c[pos[i]] += v
    for (i, j), v in m["quad"].items():
        if i in fixed and j in fixed:
            const += v * fixed[i] * fixed[j]
        elif i in fixed:
            c[pos[j]] += v * fixed[i]
        elif j in fixed:
            c[pos[i]] += v * fixed[j]
        else:
            return None, "helper variables interact quadratically"
    rows, lo, hi = [], [], []
    for kind, a, sense, rhs in m["cons"]:
        entries = []
        if kind == "linear":
            row = np.zeros(len(free))
            for i, v in a.items():
                if i in pos:
                    row[pos[i]] += v
            entries.append((row, rhs - sum(v * fixed[i] for i, v in a.items() if i in fixed), sense))
        else:
            for i, j in a:
                row = np.zeros(len(free))
                sign_j = 1 if kind == "exclusion" else -1
                r = (1 - fixed.get(i, 0) - fixed.get(j, 0)) if kind == "exclusion" else (fixed.get(j, 0) - fixed.get(i, 0))
                if i in pos:
                    row[pos[i]] += 1
                if j in pos:
                    row[pos[j]] += sign_j
                entries.append((row, r, "le"))
        for row, r, s in entries:
            if not row.any():
                if (s == "le" and r < -1e-9) or (s == "ge" and r > 1e-9) or (s == "eq" and abs(r) > 1e-9):
                    return None, "infeasible"
                continue
            rows.append(row); lo.append(r if s in ("ge", "eq") else -np.inf); hi.append(r if s in ("le", "eq") else np.inf)
    if not free:
        return const, "ok"
    sign = 1 if m["sense"] == "minimize" else -1
    res = milp(sign * c, integrality=np.ones(len(free)), bounds=Bounds(0, 1),
               constraints=[LinearConstraint(np.array(rows), lo, hi)] if rows else None, options={"time_limit": time_limit})
    if res.x is None:
        return None, "infeasible" if res.status == 2 else f"milp status {res.status}"
    return const + float(c @ np.round(res.x)), "ok"


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- the Checker agent

def _exec(code: str) -> dict:
    import numpy
    from . import prep
    # the tested toolkit is available: a hand-written haversine without radians made a Checker wrong (2026-10-11)
    import sys
    ns = {"np": numpy, "random": random, "prep": prep}
    sys.modules.setdefault("prep", prep)        # generated code writes `import prep` (2026-10-12)
    exec(compile(code, "<checker>", "exec"), ns)
    return ns


def _samples(code: str, data: dict, n: int, seed: int, decision_sets: list[list[str]]):
    """Worker: n sampled solutions with their evaluation, plus two mutated (should-be-invalid) variants of each."""
    ns = _exec(code)
    rng = random.Random(seed)
    ids = {s: list(data[s]) for shape in decision_sets for s in shape if isinstance(data.get(s), list)}
    out = []
    for _ in range(n):
        sol = ns["sample_decisions"](data, rng)
        sol = [dict(d) for d in sol]
        ev = ns["evaluate"](data, sol)
        mutants = []
        if len(sol) > 1:
            drop = sol[:]
            drop.pop(rng.randrange(len(drop)))
            mutants.append(("one decision removed", drop))
        # add a decision of a kind the Checker itself produces: the model may label helper indicators too (covered
        # blocks), and a mutant of that kind crashed the Checker with KeyError (Berlin, 2026-10-12)
        own = [list(t) for t in dict.fromkeys(tuple(sorted(d)) for d in sol)]
        shape = rng.choice([sh for sh in decision_sets if sorted(sh) in own] or own or decision_sets)
        extra = {s: rng.choice(ids[s]) for s in shape if ids.get(s)}
        if len(extra) == len(shape) and _kt(extra) not in {_kt(d) for d in sol}:
            mutants.append(("one decision added", sol + [extra]))
        # one decision whose entity is replaced by another entity that is ALREADY in the solution (a stop visited
        # twice, a member in two groups): the remove / add mutants keep the counts per position and missed a model
        # that had lost its "each stop once" constraints (fault injection, 2026-10-12)
        if len(sol) > 1:
            j = rng.randrange(len(sol))
            for s_name in [k for k in sol[j] if isinstance(sol[j][k], str) or k in ids]:
                others = [d[s_name] for i, d in enumerate(sol) if i != j and d.get(s_name) != sol[j][s_name]]
                if others:
                    twin = dict(sol[j]); twin[s_name] = rng.choice(others)
                    if _kt(twin) not in {_kt(d) for d in sol}:
                        mutants.append(("one decision repeats another entity", sol[:j] + [twin] + sol[j + 1:]))
                    break
        mutants = [(why, m, ns["evaluate"](data, m)) for why, m in mutants]
        out.append((sol, ev, mutants))
    return json.loads(json.dumps({"sense": ns.get("OBJECTIVE_SENSE"), "items": out}, default=float))


def _evaluate(code: str, data: dict, decisions: list[dict]):
    return json.loads(json.dumps(_exec(code)["evaluate"](data, decisions), default=float))


def decision_sets(raw: dict) -> list[list[str]]:
    return sorted({tuple(sorted(k)) for k in raw.get("variable_keys", []) if k}, key=str)


def build_checker(spec_json: str, ds: dict, example: dict, raw_example: dict, client, model: str,
                  attempts: int = 3, temperature: float = 0.0) -> tuple[str, list[str]]:
    # LLM_MODEL_CHECKER: a different model for the independent checks, so they do not share the builder's
    # misreadings (both read a per-row update time as a period, YouBike 2026-10-12)
    model = __import__("os").environ.get("LLM_MODEL_CHECKER") or model
    shapes = decision_sets(raw_example)
    first = next((k for k in raw_example.get("variable_keys", []) if k), {})
    feedback, code, errors = "", "", ["not attempted"]
    for _ in range(attempts):
        prompt = (_PROMPT.replace("{spec}", spec_json[:6000]).replace("{schema}", schema.describe(ds))
                  .replace("{example}", json.dumps(example, ensure_ascii=False, default=str)[:3000])
                  .replace("{decision_example}", json.dumps(first, ensure_ascii=False))
                  .replace("{decision_sets}", json.dumps([list(s) for s in shapes]))
                  .replace("{previous_feedback}", feedback))
        try:
            raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="req_checker")
        except Exception as e:                                    # see datacheck.build: no check, not a failure
            return "", [f"requirement checker unavailable: {type(e).__name__}: {str(e)[:200]}"]
        code = extract_code(raw) if raw.strip() else ""
        errors = []
        try:
            got = run_task(__name__, "_samples", code, example, 3, 0, [list(s) for s in shapes], timeout=60)
            if got["sense"] not in ("maximize", "minimize"):
                errors.append('define OBJECTIVE_SENSE = "maximize" or "minimize" at module level')
            for sol, ev, _ in got["items"]:
                if not isinstance(ev, dict) or "feasible" not in ev or "objective" not in ev:
                    errors.append("evaluate must return {'feasible': bool, 'violations': [...], 'objective': float}")
                    break
                if not ev["feasible"]:
                    errors.append(f"evaluate rejects a solution from your own sample_decisions: {ev.get('violations')}")
                    break
                bad = [d for d in sol if set(d) not in [set(s) for s in shapes]
                       or any(str(v) not in {str(x) for x in example.get(s, [])} for s, v in d.items())]
                if bad:
                    errors.append(f"sample_decisions returned decisions not made of the sets {shapes} and their ids: {bad[:2]}")
                    break
        except Exception as e:
            errors.append(f"{type(e).__name__}: {str(e)[:1500]}")
        if not errors:
            return code, []
        feedback = ("PREVIOUS ATTEMPT FAILED:\n- " + "\n- ".join(errors) +
                    "\n\nYOUR PREVIOUS CODE (fix it, keep what works):\n```python\n" + code[:12000] + "\n```")
    return code, errors


class Validator:
    """Plugs the Checker into run_workflow: preflight(ns, data, raw) and solution(ns, data, raw, x) return issues."""

    def __init__(self, code: str, shapes: list, samples: int = SAMPLES, seed: int = 0):
        self.code, self.shapes, self.samples, self.seed = code, [tuple(s) for s in shapes], samples, seed
        self.last: dict[str, Any] = {}

    def preflight(self, data: dict, raw: dict) -> list[str]:
        m, shapes = parse(raw), decision_sets(raw)
        if [tuple(s) for s in shapes] != self.shapes:
            self.last = {"skipped": f"decision sets changed from {self.shapes} to {shapes}; Checker not comparable"}
            return []
        report = {"valid_feasible_in_model": 0, "valid": 0, "invalid_rejected": 0, "invalid": 0, "not_comparable": 0}
        issues = []
        try:
            got = run_task(__name__, "_samples", self.code, data, self.samples, self.seed,
                           [list(s) for s in shapes], timeout=300)
        except Exception as e:
            self.last = {"error": str(e)[:500]}
            return []                                     # a broken Checker is no evidence against the model
        mv, cv = [], []
        direction = 1 if got["sense"] == "maximize" else -1
        for sol, ev, mutants in got["items"]:
            if not ev.get("feasible"):
                continue                                  # the Checker contradicts itself: skip, not evidence
            v, note = fixed_value(m, sol)
            if note.startswith("no model variable"):
                # the model leaves some decisions out on purpose (e.g. a fixed start of a closed tour to break
                # symmetry): an equivalent solution exists but is written differently. Not evidence of an error
                # (a false alarm in a 2026-10-11 run); the independent acceptance judge still checks the result.
                report["not_comparable"] += 1
                continue
            report["valid"] += 1
            if v is None:
                if len(issues) < 3:
                    issues.append(f"the model rejects a solution that satisfies the confirmed requirements ({note}): "
                                  f"{sol[:6]}{' ...' if len(sol) > 6 else ''}")
                continue
            report["valid_feasible_in_model"] += 1
            mv.append(v if m["sense"] == "maximize" else -v)
            cv.append(float(ev["objective"]))
            for why, mut, mev in mutants:
                if mev.get("feasible"):
                    continue
                mval, mnote = fixed_value(m, mut)
                if mnote.startswith("no model variable"):
                    continue
                report["invalid"] += 1
                if mval is None:
                    report["invalid_rejected"] += 1
                elif sum(1 for i in issues if "accepts" in i) < 2:
                    issues.append(f"the model accepts a solution that breaks the requirements ({why}; checker: "
                                  f"{'; '.join(mev.get('violations', []))[:200]})")
        if len(mv) >= 5:
            rank = _spearman(np.array(mv), direction * np.array(cv))
            report["rank_agreement"] = round(rank, 3)
            if not np.isnan(rank) and rank < RANK_MIN:
                issues.append(f"the model's objective ranks valid solutions differently from the confirmed objective "
                              f"(rank correlation {rank:.2f} on {len(mv)} random valid solutions): check the objective "
                              f"terms, their signs and whether items are counted once")
        self.last = report
        return issues

    def solution(self, data: dict, raw: dict, x: list) -> list[str]:
        decisions = [k for k, b in zip(raw.get("variable_keys", []), x) if b and k]
        try:
            ev = run_task(__name__, "_evaluate", self.code, data, decisions, timeout=120)
        except Exception:
            return []
        return [] if ev.get("feasible") else [f"the delivered solution breaks the confirmed requirements: "
                                              f"{'; '.join(ev.get('violations', []))[:600]}"]

