"""Data agent: LOOK at the data with read-only tools, then write a data plan (dataplan.py) that tested tools execute.

The agent decides the next step; the data plan is how it acts. Most 'looks right but is wrong' data errors of the
2026-10-12 runs needed one look at the data, not more rules:
  a column of per-row update times taken as periods (one look at its distinct values),
  area codes guessed from memory (one look at which code lies in which named area),
  green spaces taken as playgrounds (one look at a few rows), two key systems joined (one look at both keys).
As in OGD4All (arXiv 2602.00012) the agent observes before it commits; unlike a free notebook it may only use the
read-only tools below and a bounded number of steps, and every step is recorded.

Each turn the model returns ONE JSON object:
    {"explore": {"tool": "<name>", ...arguments}}       look at something
    {"plan": {...a data plan...}, "notes": ..., "metadata": ...}   try a plan (dry run on the full tables)
    {"gap": "<what is missing>"}                         the tables cannot give what the model needs
A dry run that fails comes back as an observation; a plan that runs is returned to the adapter, whose checks
(contract, required tools, data double-check) decide.
"""
from __future__ import annotations
import json
from typing import Any

import numpy as np
import pandas as pd

from . import dataplan, prep

MAX_STEPS = 10
STEP_TOKENS = 8192       # one turn decides one small step: a 32k thinking budget ran out 12 times in one AED replay
MIN_EXPLORE = 2          # look before you plan: a plan before two looks is sent back
TOOLS_DOC = """EXPLORATION TOOLS (read-only; one per turn):
- {"explore": {"tool": "columns", "table": "<key>"}}                       columns, types, missing share, examples
- {"explore": {"tool": "values", "table": "<key>", "column": "<col>"}}     distinct values with counts (top 25) and
                                                                            how many distinct values per row
- {"explore": {"tool": "rows", "table": "<key>", "column": "<col>", "contains": "<text>"}}   up to 5 matching rows
  (omit column/contains for the first rows)
- {"explore": {"tool": "keys", "left": ["<table>", "<col>"], "right": ["<table>", "<col>"]}}   key examples on both
  sides and how many match after normalising text: check BEFORE a join
- {"explore": {"tool": "code_areas", "table": "<key>", "code_column": "<col>", "lat": "<col>", "lon": "<col>",
   "areas": ["大安區, 臺北市", ...], "country": "tw"}}                        which code lies in which named area
When you have seen enough, answer {"plan": {...}, "notes": "...", "metadata": {...}} (the DATA PLAN format above);
it is tried at once on the full tables and you see the result. Or {"gap": "..."}.
Look before you decide: check what a column's values really are before filtering by it, which code is which area
before using codes, what the rows of a table really are before using them as a set's entities, and both keys
before a join."""


def _table(tables, name):
    if name not in tables:
        raise KeyError(f"no table {name!r}; tables: {sorted(tables)}")
    return tables[name]


def _short(obj, n=1500) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)[:n]


def explore(tables: dict[str, pd.DataFrame], call: dict) -> str:
    """One read-only tool call; the result is table text, so it is neutralised before it reaches a prompt."""
    return prep.safety.observation(_explore(tables, call))


def _explore(tables: dict[str, pd.DataFrame], call: dict) -> str:
    """Run one read-only tool; any error is returned as text (it is an observation, not a failure)."""
    try:
        tool = call.get("tool")
        if tool == "columns":
            df = _table(tables, call["table"])
            return _short({c: {"type": str(df[c].dtype), "missing": round(float(df[c].isna().mean()), 3),
                               "examples": df[c].dropna().astype(str).head(3).tolist()} for c in df.columns[:60]}, 3000)
        if tool == "values":
            df = _table(tables, call["table"])
            s = df[call["column"]]
            vc = s.astype(str).value_counts().head(25)
            return _short({"rows": len(df), "distinct": int(s.nunique()), "distinct_per_row": round(s.nunique() / max(1, len(df)), 3),
                           "top": vc.to_dict()}, 2000)
        if tool == "rows":
            df = _table(tables, call["table"])
            if call.get("column") and call.get("contains") is not None:
                hit = df[df[call["column"]].astype(str).str.contains(str(call["contains"]), case=False, na=False, regex=False)]
            else:
                hit = df
            return _short({"matching": len(hit), "rows": hit.head(5).to_dict(orient="records")}, 2500)
        if tool == "keys":
            (lt, lc), (rt, rc) = call["left"], call["right"]
            a = _table(tables, lt)[lc].dropna().astype(str).map(prep.normalize_text)
            b = _table(tables, rt)[rc].dropna().astype(str).map(prep.normalize_text)
            common = set(a) & set(b)
            return _short({"left_examples": a.head(5).tolist(), "right_examples": b.head(5).tolist(),
                           "left_distinct": int(a.nunique()), "right_distinct": int(b.nunique()),
                           "matching_distinct": len(common), "examples_matching": sorted(common)[:5]})
        if tool == "code_areas":
            df = _table(tables, call["table"])
            la, lo = pd.to_numeric(df[call["lat"]], errors="coerce").to_numpy(), pd.to_numeric(df[call["lon"]], errors="coerce").to_numpy()
            out = {}
            for area in call["areas"]:
                hit = prep.points_in_polygon(la, lo, prep.area_polygon(area, call.get("country")))
                codes = df.loc[hit, call["code_column"]].astype(str).value_counts().head(3)
                out[area] = {"points_inside": int(hit.sum()), "codes_inside": codes.to_dict()}
            return _short(out)
        return f"unknown tool {tool!r}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:600]


def dry_run(answer: dict, tables: dict[str, pd.DataFrame], ds: dict | None = None) -> str | None:
    """Try the plan on the full tables (outside the sandbox) and compare its output with the model's input format;
    None when it runs and fits. A plan that ran but missed a set ('groups') or added an undeclared key was only
    rejected later by the system's checks, which cost a whole attempt (AED replay, 2026-10-13)."""
    plan = {k: answer.get("plan", {}).get(k) for k in ("prepare", "table", "steps", "output")}
    if ds:
        # a set the model defines itself (source user or mock: tour positions, groups) is not a column of the data
        own = [n for n, d in (ds.get("sets") or {}).items() if (d or {}).get("source") in ("user", "mock", "model")]
        from_table = [n for n in own if isinstance(((plan.get("output") or {}).get("sets") or {}).get(n), str)]
        if from_table:
            return (f"sets {from_table} are defined by the model (source user/mock), not by the data: do not take "
                    f"them from a table column; write them under \"settings\": a fixed list of ids, or "
                    f"{{\"index_of\": \"<set>\"}} for one id per member of that set (tour positions)")
    steps = list(plan.get("steps") or []) + [x for v in (plan.get("prepare") or {}).values() for x in (v or [])]
    if any((st or {}).get("op") == "custom" for st in steps):
        return "custom steps are not accepted in this mode: express the step with the listed tools"
    try:
        data, seen, log = dataplan.execute(plan, {k: v.copy() for k, v in tables.items()})
    except Exception as e:
        return f"{e}"[:2500]
    sizes = {k: len(v) for k, v in data.items() if isinstance(v, list)}
    if not all(sizes.values()):
        return f"the plan ran but produced empty lists: {sizes}"
    if ds:
        from . import schema
        data = schema.fill_model_sets(ds, data)                      # the system generates 'model' sets
        need = set(schema.field_keys(ds))
        missing, extra = sorted(need - set(data)), sorted(set(data) - need)
        if missing or extra:
            return (f"the plan ran, but its output does not match the model's input format: missing {missing}, "
                    f"not declared {extra}. Output exactly these keys: {sorted(need)}")
    return None


def looked_so_far(history: list[str]) -> int:
    return sum(h.startswith("- explore") for h in history)


def _as_plan_format(base_prompt: str) -> str:
    """The base prompt ends with 'Return JSON only: <plan>': for one-shot planning. Here it would contradict 'start
    by exploring' (Gemma answered with a plan every turn, Melbourne 2026-10-21), so it is relabelled as the format."""
    return base_prompt.replace("Return JSON only:\n", 'DATA PLAN FORMAT (the content of {"plan": ...}; not this '
                               "turn's answer by itself):\n", 1)


def plan(base_prompt: str, tables: dict[str, pd.DataFrame], client, model: str, history: list[str],
         temperature: float = 0.0, max_steps: int = MAX_STEPS, ds: dict | None = None) -> str:
    """Explore, then return the raw JSON of a data plan (dataplan.plan_to_code format) or a gap. `history` is kept
    by the caller across adapter attempts, so a later attempt sees what was already looked at and tried."""
    last_plan = ""
    for _ in range(max_steps):
        transcript = "\n".join(history[-14:])[-12000:]
        prompt = (_as_plan_format(base_prompt) + "\n\nTHIS TURN: you work step by step. The DATA PLAN FORMAT above is "
                  'what goes INSIDE {"plan": ...}. '
                  + ("Start by exploring.\n" if looked_so_far(history) < MIN_EXPLORE else "\n") + TOOLS_DOC + "\n\nWHAT YOU HAVE DONE SO FAR:\n"
                  + (transcript or "(nothing yet)") + '\n\nReturn ONE JSON object now: {"explore": ...}, {"plan": ...} or {"gap": ...}.')
        try:
            raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="data_agent",
                              json_mode=True, max_tokens=STEP_TOKENS)
        except TypeError:                                    # a client without the max_tokens argument (tests)
            raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="data_agent", json_mode=True)
        try:
            answer = json.loads(raw)
        except json.JSONDecodeError:
            history.append(f"- your answer was not valid JSON: {raw[:200]!r}")
            continue
        if "plan" not in answer and ("steps" in answer or "table" in answer or "prepare" in answer):
            # a bare plan, as the base prompt's own answer format shows: accept it as a plan
            answer = {"plan": {k: answer.get(k) for k in ("prepare", "table", "steps", "output")},
                      "notes": answer.get("notes", ""), "metadata": answer.get("metadata") or {}}
        looked = looked_so_far(history)
        if "plan" in answer and looked < MIN_EXPLORE:
            if sum(h.startswith("- you proposed a plan after") for h in history) >= 1:
                # a model that keeps planning without looking (Gemma, Melbourne 2026-10-21: the same rejection
                # 6+ times, every step spent): do the looking for it, so the next plan is judged on real data
                for name in list(tables)[:3]:
                    call = {"tool": "rows", "table": name}
                    history.append(f"- explore (automatic) {_short(call, 200)} -> {explore(tables, call)}")
                    call = {"tool": "columns", "table": name}
                    history.append(f"- explore (automatic) {_short(call, 200)} -> {explore(tables, call)}")
                looked = sum(h.startswith("- explore") for h in history)
            else:
                history.append(f"- you proposed a plan after {looked} look(s); first LOOK at the data with the "
                               f"exploration tools (at least {MIN_EXPLORE}): check the columns you filter or join on")
                continue
        if "plan" in answer and looked < MIN_EXPLORE:
            continue
        if "explore" in answer:
            result = explore(tables, answer["explore"] or {})
            history.append(f"- explore {_short(answer['explore'], 300)} -> {result}")
            continue
        if answer.get("gap"):
            return json.dumps({"gap": answer["gap"]}, ensure_ascii=False)
        if "plan" in answer:
            last_plan = json.dumps({**(answer["plan"] or {}), "notes": answer.get("notes", ""),
                                    "metadata": answer.get("metadata") or {}}, ensure_ascii=False)
            problem = dry_run(answer, tables, ds)
            if problem is None:
                history.append(f"- plan {_short(answer['plan'], 600)} -> ran on the full tables")
                return last_plan
            history.append(f"- plan {_short(answer['plan'], 600)} -> FAILED: {problem}")
            continue
        history.append(f"- answer had neither explore, plan nor gap: {raw[:200]!r}")
    return last_plan or json.dumps({"gap": "the data agent used all its steps without a plan that runs"})
