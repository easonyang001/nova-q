"""Data-level requirement check by double computation.

An independent Data Checker agent sees the confirmed specification, the declaration of the public data and the
raw-table profiles, never the adapter's code. It writes expected(tables) -> {"counts": {set: n}, "totals": {param:
sum}}: how many entities each set must have after the specification's filters (area, category, period, level, one
entity per real place) and the sums of numeric parameters taken straight from the tables. The adapter's output is
compared with it. Ids are not compared: an adapter may number entities itself.

Reducible sets are not counted (reduce_points shrinks them on purpose), but their weight totals must still agree.
A disagreement is evidence for the adapter, not a verdict: the Data Checker can be wrong too.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema
from .bridge import extract_code
from .datakit import profiles
from .runtime import run_task

_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "data_checker.txt").read_text(encoding="utf-8")
TOTAL_TOLERANCE = 0.005


def _expected(code: str, tables: dict) -> dict:
    from . import prep
    import sys
    ns = {"pd": pd, "np": np, "prep": prep}
    sys.modules.setdefault("prep", prep)        # generated code writes `import prep`
    exec(compile(code, "<data_checker>", "exec"), ns)
    out = ns["expected"]({k: v.copy() for k, v in tables.items()})
    return json.loads(json.dumps(out, default=float))


def build(spec_text: str, ds: dict, tables: dict, client, model: str, attempts: int = 2,
          situations: str = "") -> tuple[str, dict, list[str]]:
    """Returns (code, expected, errors). `situations`: the data situations the system detected in the tables (the
    same text the Adapter gets); without it a checker summed rows split by sex together with their total rows
    three times (2026-10-11)."""
    model = __import__("os").environ.get("LLM_MODEL_CHECKER") or model          # see reqcheck.build_checker
    view = schema.public_view(ds)
    numeric = {p: d for p, d in view.get("params", {}).items() if d.get("type") in ("int", "number")}
    described = schema.describe({"sets": view.get("sets", {}), "params": numeric}) if schema.is_structured(ds) else \
        json.dumps(ds, ensure_ascii=False)
    feedback, code, errors = "", "", ["not attempted"]
    for _ in range(attempts):
        prompt = (_PROMPT.replace("{spec}", spec_text[:4000]).replace("{schema}", described)
                  .replace("{profiles}", profiles(tables)).replace("{situations}", situations or "(none detected)")
                  .replace("{previous_feedback}", feedback))
        try:
            raw = client.call(model=model, prompt=prompt, temperature=0.0, step_name="data_checker")
        except Exception as e:
            # the checker could not run (a 32k-context checker model refused a long prompt and failed the whole data
            # stage, 2026-10-13): no check is no evidence, never a failure of the data itself
            return "", {}, [f"data checker unavailable: {type(e).__name__}: {str(e)[:200]}"]
        code = extract_code(raw) if raw.strip() else ""
        try:
            got = run_task(__name__, "_expected", code, tables, timeout=120)
            if not isinstance(got, dict) or not isinstance(got.get("counts", {}), dict) \
                    or not isinstance(got.get("totals", {}), dict):
                raise ValueError("expected() must return {'counts': {...}, 'totals': {...}}")
            # names must be the declaration's own: a check under another name ('stations' for 'sites') was
            # silently skipped and a fabricated extra site went through (2026-10-11)
            bad = [k for k in got.get("counts", {}) if k not in view.get("sets", {})] + \
                  [k for k in got.get("totals", {}) if k not in numeric]
            if bad:
                raise ValueError(f"unknown names {bad}: use exactly the set names {list(view.get('sets', {}))} in "
                                 f"counts and the parameter names {list(numeric)} in totals")
            return code, got, []
        except Exception as e:
            errors = [f"{type(e).__name__}: {str(e)[:1200]}"]
            feedback = ("PREVIOUS ATTEMPT FAILED:\n- " + errors[0] + "\n\nYOUR PREVIOUS CODE:\n```python\n"
                        + code[:8000] + "\n```")
    return code, {}, errors


TABLE_SOURCES = ("public", "private")


def compare(expected: dict, data: dict, ds: dict) -> list[str]:
    """Disagreements between the independent expectation and the adapter's output."""
    issues = []
    sets = ds.get("sets", {}) if schema.is_structured(ds) else {}
    params = ds.get("params", {}) if schema.is_structured(ds) else {}
    for name, n in (expected.get("counts") or {}).items():
        # public and private sets both come from the tables the checker also read ("private" for downloaded open
        # data hid a 38-vs-35 mismatch the checker had found, Basel 2026-10-24); user / mock / model sets do not
        if name not in sets or sets[name].get("source") not in TABLE_SOURCES or sets[name].get("reducible") \
                or not isinstance(data.get(name), list):
            continue
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if n == 0:
            # the specification needs these entities: an expected count of 0 is the checker's own mistake (it kept
            # each row's update time as a period and found nothing, YouBike 2026-10-12), not evidence against the data
            continue
        if len(data[name]) != n:
            issues.append(f"set {name!r} has {len(data[name])} entities, but an independent computation from the raw "
                          f"tables and the specification's filters (area, category, period, level, one entity per "
                          f"real place) expects {n}. Check which rows you keep and whether duplicates of the same "
                          f"place are merged")
    for name, total in (expected.get("totals") or {}).items():
        if name not in params or params[name].get("source") not in TABLE_SOURCES or data.get(name) is None:
            continue
        try:
            total = float(total)
            ours = float(np.nansum(np.asarray(data[name], dtype=float)))
        except (TypeError, ValueError):
            continue
        if abs(ours - total) > TOTAL_TOLERANCE * max(abs(total), 1.0):
            issues.append(f"parameter {name!r} sums to {ours:,.6g}, but an independent computation from the raw "
                          f"tables expects {total:,.6g}. Rows may be missing, duplicated, from the wrong period or "
                          f"include totals of a higher level")
    return issues
