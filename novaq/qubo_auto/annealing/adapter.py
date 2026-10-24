"""Data Adapter agent: real tables -> the validated model's input, without the LLM reading the data.

The agent sees table profiles (datakit.profiles) and the model's DATA_SCHEMA plus the shape of its example input,
writes load_data(tables), and the code is checked on the FULL tables: every schema key present, the same JSON
type as the example, JSON round trip, and build_ir() accepting the result. Retries show the agent its previous
code (repair, don't rewrite). The user approves the result in the UI before it is used.
"""
from __future__ import annotations
import json
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bridge import extract_code
from . import prep
from .prep import situations as prep_situations
from .datakit import profiles
from .modeler import describe_error, to_ir, to_jsonable
from .runtime import run_task
from . import schema

_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "adapter.txt").read_text(encoding="utf-8")
_PLAN_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "adapter_plan.txt").read_text(encoding="utf-8")
# "plan": the agent writes a data plan executed by tested tools (dataplan.py) and writes code only after the plan
# attempts failed; "code": free-form adapter code only (the earlier design, kept for the ablation)
ADAPTER_MODE = __import__("os").environ.get("QUBO_ADAPTER_MODE", "plan")
PLAN_ATTEMPTS = 2


def _kind(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _shape(example: Any, depth: int = 0) -> Any:
    """A short picture of the example input: types, list lengths, first elements."""
    if isinstance(example, dict):
        return {k: _shape(v, depth + 1) for k, v in list(example.items())[:30]}
    if isinstance(example, list):
        return [f"list of {len(example)}", _shape(example[0], depth + 1)] if example else "empty list"
    return example if depth == 0 or isinstance(example, (int, float, str, bool)) else _kind(example)


_NUMBER_SUFFIX = re.compile(r"\s*(?:[_#-]\s*|\()\d+\)?$")


def repeated_names(ds: dict, data: dict, limit: int = 8) -> list[str]:
    """Names that occur more than once in one data-derived set: as ids, in a name-like parameter, or as an id plus the
    same id with an added number ("X" and "X_1"). [] when there are none."""
    if not schema.is_structured(ds):
        return []
    found: list[str] = []
    for name, d in ds["sets"].items():
        if d.get("source") not in ("public", "private") or not isinstance(data.get(name), list):
            continue
        columns = [[str(i) for i in data[name]]]
        columns += [[str(v) for v in data[k]] for k, p in ds.get("params", {}).items()
                    if p.get("index") == [name] and isinstance(data.get(k), list)
                    and re.search(r"name|title|名", k, re.I) and len(data[k]) == len(data[name])]
        for values in columns:
            seen = set(values)
            for v in values:
                base = _NUMBER_SUFFIX.sub("", v).strip()
                if (values.count(v) > 1 or (base != v and base in seen)) and base not in found:
                    found.append(base)
    return found[:limit]


PORTAL_ROW_ID = re.compile(r"^row-[0-9a-z]{4}([~._-][0-9a-z]{4}){2}$")      # Socrata ':id' values


def check_output(model_ns: dict[str, Any], data: Any, example: dict, metadata=None, sources=None) -> tuple[list[str], dict]:
    """Problems with load_data's output for this model ([] = usable), plus a size report."""
    errors, report = [], {}
    if not isinstance(data, dict):
        return [f"load_data must return a dict, got {_kind(data)}"], report
    try:
        data = json.loads(json.dumps(to_jsonable(data), allow_nan=False))
    except (TypeError, ValueError) as e:
        return [f"the output cannot be stored as JSON ({e}); use lists, numbers and strings only"], report
    ds = model_ns["DATA_SCHEMA"]
    missing = [k for k in schema.field_keys(ds) if k not in data]
    if missing:
        errors.append(f"missing DATA_SCHEMA keys {missing}")
    if schema.is_structured(ds):
        errors += [e for e in schema.data_errors(ds, data) if not e.endswith("is missing")]
    from .contracts import check_data
    contract_errors, review = check_data(data, model_ns.get("DATA_CONTRACT"), metadata, sources)
    errors.extend(contract_errors)
    report["review_required"] = review
    report["data_metadata"] = metadata or {}
    set_names = set(ds["sets"]) if schema.is_structured(ds) else set()
    for name in sorted(set_names & set(data)):
        ids = [str(i) for i in data[name]] if isinstance(data[name], list) else []
        if ids and sum(bool(PORTAL_ROW_ID.match(i)) for i in ids) > len(ids) / 2:
            # a portal's internal row id names nothing the user can recognise, and the answer cannot say which
            # proposal or place was chosen (Somerville, Socrata ':id' row-xxxx~xxxx.xxxx, 2026-10-24)
            errors.append(f"set {name!r} uses the portal's internal row ids ({ids[0]!r} ...) as identifiers; use a "
                          f"column that names or codes each member (title, name, official code)")
    for k, v in example.items():
        if k in set_names:
            continue                      # set ids may be strings or integers; data_errors checks them properly
        if k in data and _kind(data[k]) != _kind(v):
            errors.append(f"key {k!r} is a {_kind(data[k])} but the model expects a {_kind(v)} (like its example)")
        elif isinstance(v, list) and v and k in data and data[k] and _kind(data[k][0]) != _kind(v[0]):
            errors.append(f"items of {k!r} are {_kind(data[k][0])} but the model expects {_kind(v[0])}")
    if errors:
        return errors, report
    try:
        ir = to_ir(model_ns["build_ir"](data), "adapted")
    except Exception as e:
        return [describe_error(e, model_ns.get("__source__", "")) +
                "\n    (raised by the MODEL's build_ir on your output: make the output match the example's format)"], report
    report.update(n_variables=ir.n_variables, n_constraints=len(ir.constraints), data=data)
    return [], report


class _Recorder:
    """The `prep` module as the agent sees it: every tool call is recorded, so required tools are checked at run
    time (writing `prep.x` in a comment does not count)."""
    def __init__(self):
        self.called: set[str] = set()

    def __getattr__(self, name):
        obj = getattr(prep, name)
        if callable(obj) and not isinstance(obj, type):
            def wrapped(*a, **kw):
                self.called.add(name)
                return obj(*a, **kw)
            return wrapped
        return obj


class _TrackedTables(dict):
    """Records which tables the code reads; tables it ignores impose no requirements."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.used: set[str] = set()

    def __getitem__(self, k):
        self.used.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k in self:
            self.used.add(k)
        return super().get(k, default)

    def items(self):
        self.used.update(self.keys())
        return super().items()

    def values(self):
        self.used.update(self.keys())
        return super().values()


class _SystemLog(prep.PrepLog):
    """Sources are written by the system from download metadata; the agent cannot add or change them."""
    def source(self, *a, **kw):
        self.agent_tried_source = True


def enforcement_errors(situations, recorder: _Recorder, used: set[str], code: str) -> list[str]:
    errors = []
    for sit in situations:
        relevant = (len(used) >= 2) if sit.table == "*" else (sit.table in used)
        if sit.kind == "areas identified only by codes" and not any(q + sit.column + q in code for q in "\"'"):
            # the code column is never touched: nothing is selected by codes (NYC budgeting filtered by the council
            # district the user named and was still forced into in_area on 'Borough Code', 2026-10-22)
            relevant = False
        if relevant and not (set(sit.tools) & recorder.called):
            errors.append("REQUIRED TOOL NOT USED: " + sit.text())
    errors += [f"FORBIDDEN: {m}" for m in prep_situations.forbidden_patterns(code)]
    return errors


# Real tables (polygon files, national lists with 100k rows) need more than the 30 s default; same order as the data check.
ADAPTER_TIMEOUT_S = 180


def _execute_adapter(code, tables, sources, situations):
    recorder, log = _Recorder(), _SystemLog()
    for src in sources or []:
        prep.PrepLog.source(log, **{k: src.get(k, "") for k in ("title", "url", "publisher", "license", "retrieved")})
    ns = {"pd": pd, "np": np, "prep": recorder, "__source__": code}
    import inspect, sys, types
    exposed = types.ModuleType("prep")
    exposed.__getattr__ = recorder.__getattr__
    previous = sys.modules.get("prep")
    sys.modules["prep"] = exposed
    try:
        exec(compile(code, "<model>", "exec"), ns)
        if "load_data" not in ns:
            raise ValueError("define load_data(tables, log)")
        copies = _TrackedTables({k: v.copy() for k, v in tables.items()})
        fn = ns["load_data"]
        data = fn(copies, log) if len(inspect.signature(fn).parameters) >= 2 else fn(copies)
    finally:
        if previous is None:
            sys.modules.pop("prep", None)
        else:
            sys.modules["prep"] = previous
    errors = enforcement_errors(situations, recorder, copies.used, code)
    errors += [f"DATA PROBLEM FOUND BY A TOOL: {f}" for f in log.flags]
    return to_jsonable(data), errors, {"tools_used": sorted(recorder.called), "notes": str(ns.get("ADAPTER_NOTES", ""))[:1500],
                                    "prep_log": log.markdown(), "flags": log.flags,
                                    "data_metadata": ns.get("ADAPTER_METADATA", {})}


def _metadata_block(model_ns: dict, sources, plan_mode: bool) -> str:
    """Contract, sources and the metadata request, placed BEFORE the prompt's closing output instruction (appended
    after it, a plan-mode prompt ended by asking for a Python variable while the answer had to be JSON)."""
    where = 'the "metadata" object of your answer' if plan_mode else "a module-level dict ADAPTER_METADATA"
    return ("DATA CONTRACT (preserve exact units/period/geography/granularity):\n"
            + json.dumps(model_ns.get("DATA_CONTRACT", {}), ensure_ascii=False)
            + "\nAVAILABLE SOURCE METADATA:\n" + json.dumps(sources or [], ensure_ascii=False)
            + f"\nFill {where} per output field: unit, period, geography, granularity, source_urls, assumptions. "
            "Use only actual source URLs. assumptions contains ONLY unconfirmed choices/substitutions needing review; "
            "use [] when simply applying the confirmed spec. If unknown, leave unknown; never claim verification "
            "without evidence.\n")


def build_adapter(model_ns: dict[str, Any], tables: dict[str, pd.DataFrame], spec_text: str, client, model: str,
                  notes: str = "", attempts: int = 3, temperature: float = 0.0,
                  sources: list[dict] | None = None, data_check: bool = False) -> tuple[str, dict, list[str], dict]:
    """Returns (code, data, errors, report). report: n_variables, n_constraints, notes, prep_log (markdown).
    `sources`: provenance of the files (title, url, publisher, license), copied into the record."""
    example = to_jsonable(model_ns["make_example_data"](6, 0))
    situations = prep_situations.detect(tables)
    required = "\n".join(f"- {s.text()}" for s in situations) or "(none detected)"
    expected, check_errors, check_code = {}, [], ""
    if data_check and schema.is_structured(model_ns["DATA_SCHEMA"]):
        # independent double computation of the data-level requirements (filters, dedup, level, period)
        from . import datacheck
        check_code, expected, check_errors = datacheck.build(spec_text, model_ns["DATA_SCHEMA"], tables, client, model,
                                                    situations=required)
    feedback, code, errors, report = "", "", ["not attempted"], {}
    # "agent": the same plan attempts, but each one first LOOKS at the data with read-only tools (dataagent.py)
    plan_attempts = PLAN_ATTEMPTS if ADAPTER_MODE in ("plan", "agent") else 0
    agent_history: list[str] = []
    names_questioned = False
    attempts += plan_attempts          # the code attempts keep their budget; plan attempts come first
    for attempt in range(attempts):
        if attempt == plan_attempts and plan_attempts:
            feedback = ""                                     # switching to code: the plan feedback does not apply
        template = _PLAN_PROMPT if attempt < plan_attempts else _PROMPT
        prompt = (template.replace("{spec}", spec_text[:3000])
                  .replace("{schema}", schema.describe(model_ns["DATA_SCHEMA"]))
                  .replace("{example}", json.dumps(_shape(example), ensure_ascii=False, default=str)[:3000])
                  .replace("{profiles}", profiles(tables))
                  .replace("{notes}", notes or "(none)")
                  .replace("{required}", required)
                  .replace("{previous_feedback}", _metadata_block(model_ns, sources, attempt < plan_attempts) + feedback))
        if attempt < plan_attempts and ADAPTER_MODE == "agent":
            from . import dataagent
            raw = dataagent.plan(prompt, tables, client, model, agent_history, temperature=temperature,
                                 ds=model_ns["DATA_SCHEMA"] if schema.is_structured(model_ns["DATA_SCHEMA"]) else None)
            try:
                from . import dataplan
                code = dataplan.plan_to_code(json.loads(raw))
            except Exception as e:
                code = f"def load_data(tables, log):\n    raise ValueError({('the plan was not valid JSON: ' + str(e))[:300]!r})\n"
        elif attempt < plan_attempts:
            raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="adapter_plan",
                              json_mode=True)
            try:
                from . import dataplan
                code = dataplan.plan_to_code(json.loads(raw))
            except Exception as e:
                code = f"def load_data(tables, log):\n    raise ValueError({('the plan was not valid JSON: ' + str(e))[:300]!r})\n"
        else:
            raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="adapter")
            code = extract_code(raw) if raw.strip() else ""
        try:
            out, tool_errors, execution = run_task(__name__, "_execute_adapter", code, tables, sources, situations,
                                                        timeout=ADAPTER_TIMEOUT_S)
            if isinstance(out, dict) and schema.is_structured(model_ns["DATA_SCHEMA"]):
                out = schema.fill_model_sets(model_ns["DATA_SCHEMA"], out)     # positions, group labels: system ids
            errors, report = check_output(model_ns, out, example, execution.get("data_metadata"), sources)
            errors += tool_errors
            if schema.is_structured(model_ns["DATA_SCHEMA"]):
                for tool, param in schema.required_tools(model_ns["DATA_SCHEMA"]).items():
                    if tool not in execution.get("tools_used", []):
                        errors.append(f"REQUIRED TOOL NOT USED: DATA_SCHEMA declares {param!r} as derived by "
                                      f"prep.{tool}; compute it with that tool")
            missing_metadata = set(model_ns.get("DATA_CONTRACT", {})) - set(execution.get("data_metadata", {}))
            if missing_metadata:
                # paperwork, not a data error: it cost whole attempts (2026-10-11). Declare unknown, flag for review;
                # source URLs come from the download record anyway.
                meta = dict(execution.get("data_metadata") or {})
                meta.update({k: {"assumptions": ["metadata not declared by the adapter"]} for k in missing_metadata})
                execution["data_metadata"] = meta
                report["review_required"] = list(report.get("review_required", [])) + \
                    [f"adapter 沒有聲明這些欄位的來源與單位：{', '.join(sorted(missing_metadata))}"]
            report.update(execution)
        except Exception as e:
            errors = [f"{type(e).__name__}: {e}"]
        if not errors and expected:
            from . import datacheck
            mismatch = datacheck.compare(expected, report["data"], model_ns["DATA_SCHEMA"])
            report["data_check"] = {"expected": expected, "issues": mismatch}
            if mismatch and attempt < attempts - 1:
                # the checker can be wrong as well (it once kept the whole city while the adapter correctly kept
                # one district, and the adapter then chose a wrong district code to reach that number, 2026-10-12):
                # show how it computed its number, so the adapter verifies the differing step instead of chasing it
                errors = [f"DATA CHECK: {m}" for m in mismatch] + [
                    "The independent computation can be wrong too. It was:\n```python\n" + (check_code or "")[:3000]
                    + "\n```\nCompare it with your steps one by one against the DATA SCOPE and the actual table values "
                    "(e.g. which area name an area code stands for). Fix YOUR code only where it is wrong; if your "
                    "filters are right, keep them and say why in ADAPTER_NOTES. Never pick a code, filter or row set "
                    "just to reach the expected number"]
            elif mismatch:            # the checker can be wrong too: keep the data, flag it for review
                report["review_required"] = list(report.get("review_required", [])) + \
                    [f"獨立資料核對不一致：{m}" for m in mismatch]
        elif data_check:
            report["data_check"] = {"expected": {}, "issues": [], "checker_errors": check_errors}
        if not errors and attempt < attempts - 1:
            clash = repeated_names(model_ns["DATA_SCHEMA"], report.get("data") or {})
            if clash and not names_questioned:
                # one question, not a loop: the agent decides from the DATA SCOPE (Basel AED: one faculty's two
                # records 200 m apart became "Theologische Fakultät" and "Theologische Fakultät_1", 2026-10-24)
                names_questioned = True
                errors = [f"these members of the same set share a name (or differ only by an added number): {clash}. "
                          f"Check the DATA SCOPE: if rows with the same name are one place, merge them by name "
                          f"(merge_same_place with max_m null); if they are different places, name them with a "
                          f"column of the data (address, district). Never number them to make ids unique"]
            elif clash:
                report["review_required"] = list(report.get("review_required", [])) + \
                    [f"同一集合中有同名成員（資料代理判斷為不同地點）：{clash}"]
        if not errors:
            report["adapter_mode"] = (ADAPTER_MODE if attempt < plan_attempts else "code")
            if agent_history:
                report["agent_steps"] = agent_history
            return code, report.pop("data"), [], report
        if any("not found" in e or "KeyError" in e for e in errors):
            # the adapter looked a table up by its dataset title instead of its key (London, 2026-10-12)
            errors.append(f"The tables dict has exactly these keys: {sorted(tables)}")
        if attempt < plan_attempts and ADAPTER_MODE == "agent":
            agent_history.append("- the system's checks REJECTED your plan: " + " | ".join(e[:800] for e in errors))
        if attempt < plan_attempts:
            feedback = ("\nPREVIOUS PLAN FAILED. Fix every problem (change only the steps involved):\n- "
                        + "\n- ".join(e[:2500] for e in errors) + "\n\nYOUR PREVIOUS PLAN:\n" + raw[:6000])
        else:
            feedback = ("PREVIOUS ATTEMPT FAILED. Fix every problem:\n- " + "\n- ".join(errors)
                        + "\n\nYOUR PREVIOUS CODE (fix the problems, keep what works, return the complete code):\n```python\n"
                        + code[:12000] + "\n```")
    return code, {}, errors, report
