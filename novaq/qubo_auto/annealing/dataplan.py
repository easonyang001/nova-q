"""Data plan: the Data Adapter DECIDES the preparation steps, tested tools EXECUTE them, one step at a time.

Free-form pandas from the local model failed in ways no rule could list in advance (steps in the wrong order, area
codes recalled from memory, a reference point added as a stop, a table looked up by its title; 2026-10-12 runs). As in
OGD4All (arXiv 2602.00012) and PEARL (arXiv 2607.18256), each step is executed at once and its result (rows before and
after, columns, three example rows) is shown before the next step is chosen; a failing step is redone alone.

A plan is JSON:
    {"table": "<table key>",
     "steps": [{"op": "in_area", "lat": "緯度", "lon": "經度", "areas": ["大安區, 臺北市"], "code_column": "區域代碼"},
               {"op": "exclude", "column": "場所分類", "values": ["其他"]},
               {"op": "merge_same_place", "name": "場所名稱", "lat": "緯度", "lon": "經度"}],
     "output": {"sets": {"sites": "場所名稱"}, "params": {"lat": "緯度", "lon": "經度"},
                "settings": {"k": 3}, "derived": {"dist": {"tool": "haversine_matrix", "lat": "lat", "lon": "lon"}}}}
Every op is one prep tool (or a plain, checked table operation); "custom" is the only escape hatch and is flagged for
the user's review.
"""
from __future__ import annotations
import json
from typing import Any

import numpy as np
import pandas as pd

from . import prep


def _cols(df: pd.DataFrame, *names) -> None:
    missing = [n for n in names if n is not None and n not in df.columns]
    if missing:
        raise KeyError(f"no column {missing}; the table has {list(df.columns)[:40]}")


def _join(df, step, tables, log):
    right = tables[step["table"]]
    return prep.match(df, right, left_on=step["left_on"], right_on=step["right_on"], how=step.get("how", "left"), log=log)


OPS: dict[str, Any] = {
    "latest_period": lambda df, s, t, log: prep.latest_period(df, s["columns"], log=log),
    "keep_category": lambda df, s, t, log: prep.keep_category(df, s["column"], s["value"], log=log),
    "attach_parent": lambda df, s, t, log: prep.attach_parent(df, s["column"], new=s.get("new", "parent"), log=log),
    "keep_level": lambda df, s, t, log: prep.keep_level(df, s["column"], s.get("levels", "里村"), log=log),
    "drop_total_rows": lambda df, s, t, log: prep.drop_total_rows(df, s.get("columns"), log=log),
    "to_number": lambda df, s, t, log: df.assign(**{c: prep.to_number(df[c], log=log) for c in s["columns"]}),
    "to_wgs84": lambda df, s, t, log: prep.to_wgs84(df, s["x"], s["y"], crs=s.get("crs"), log=log),
    "in_area": lambda df, s, t, log: prep.in_area(df, s["lat"], s["lon"], s["areas"], country=s.get("country"),
                                                   code_column=s.get("code_column"), log=log),
    "within_radius": lambda df, s, t, log: prep.within_radius(df, s["lat"], s["lon"], tuple(s["center"]), s["radius_m"], log=log),
    "dedupe": lambda df, s, t, log: prep.dedupe(df, s["keys"], log=log),
    "merge_same_place": lambda df, s, t, log: prep.merge_same_place(df, s["name"], s["lat"], s["lon"],
                                                                     s.get("max_m", 150), log=log),
    "drop_stray_points": lambda df, s, t, log: prep.drop_stray_points(df, s["lat"], s["lon"], log=log),
    # a reducible set larger than the model allows: k representatives (weighted k-means), never the first k rows.
    # Without this op an agent could not obey "reduce to 150 with prep.reduce_points" (format sweep, 2026-10-14)
    "reduce_points": lambda df, s, t, log: prep.reduce_points(df, int(s["k"]), lat=s.get("lat", "lat"),
                                                              lon=s.get("lon", "lon"), weight=s.get("weight"), log=log),
    "join": _join,
    # distance of every row to ONE point (a station, a city centre): the model may declare it as a parameter.
    # Without this op a plan could not produce it and the agent reported a data gap (Melbourne v7, 2026-10-21)
    "distance_to": lambda df, s, t, log: _distance_to(df, s, log),
}


def _distance_to(df, step, log):
    _cols(df, step["lat"], step["lon"])
    lat0, lon0 = step["center"]
    new = step.get("new", "distance_m")
    out = df.assign(**{new: np.round(prep.distance_to(df[step["lat"]].astype(float), df[step["lon"]].astype(float),
                                                      lat0, lon0), 1)})
    log.step(f"distance_to {new} from ({lat0}, {lon0})", len(df), len(out))
    return out


def _plain(df, step, log):
    """Plain table operations, checked: keep / exclude values, drop missing, rename, aggregate."""
    op = step["op"]
    before = len(df)
    if op in ("keep", "exclude"):
        _cols(df, step["column"])
        vals = {prep.normalize_text(v) for v in step["values"]}
        hit = df[step["column"]].map(prep.normalize_text).isin(vals)
        if op == "keep" and not hit.any():
            raise ValueError(f"keep: none of {step['values']} occurs in {step['column']!r}; values there: "
                             f"{df[step['column']].dropna().astype(str).unique()[:15].tolist()}")
        out = df[hit if op == "keep" else ~hit]
    elif op == "drop_missing":
        _cols(df, *step["columns"])
        out = df.dropna(subset=step["columns"])
    elif op == "rename":
        _cols(df, *step["columns"].keys())
        out = df.rename(columns=step["columns"])
    elif op == "aggregate":
        _cols(df, *step["by"], *step["sum"])
        out = df.groupby(step["by"], as_index=False)[step["sum"]].sum()
    else:
        raise KeyError(f"unknown op {op!r}; available: {sorted(OPS) + ['keep', 'exclude', 'drop_missing', 'rename', 'aggregate', 'custom']}")
    out = out.reset_index(drop=True)
    log.step(f"{op} {json.dumps({k: v for k, v in step.items() if k != 'op'}, ensure_ascii=False)[:120]}", before, len(out))
    return out


def observe(df: pd.DataFrame, before: int | None = None) -> str:
    """What the agent sees after a step."""
    head = df.head(3).to_dict(orient="records")
    return (f"rows {before} -> {len(df)}" if before is not None else f"rows {len(df)}") + \
        f"; columns {list(df.columns)[:30]}; first rows {json.dumps(head, ensure_ascii=False, default=str)[:900]}"


def run_step(df: pd.DataFrame, step: dict, tables: dict, log) -> pd.DataFrame:
    op = step.get("op")
    if op == "custom":
        # escape hatch: a pandas expression on `df`, flagged for review in the data report
        out = eval(step["expr"], {"pd": pd, "np": np, "prep": prep}, {"df": df})   # noqa: S307 (local, logged)
        log.flags.append(f"REVIEW: custom step {step.get('why', '')!r}: {step['expr'][:200]}")
        log.step(f"custom: {step.get('why', '')}", len(df), len(out))
        return out.reset_index(drop=True)
    if op in OPS:
        return OPS[op](df, step, tables, log).reset_index(drop=True)
    return _plain(df, step, log)


def assemble(df: pd.DataFrame, output: dict) -> dict:
    """The model's data from the final table: sets are id columns, params columns aligned with them, settings
    scalars, derived params computed by the named toolkit function."""
    data: dict[str, Any] = {}
    cols = [c for c in (output.get("sets") or {}).values() if isinstance(c, str)]
    shared = sorted({c for c in cols if cols.count(c) > 1})
    if shared:
        # positions filled with the station ids (Melbourne v7, 2026-10-21): model and checker then read two
        # different tours from the same answer
        raise ValueError(f"sets {[n for n, c in output['sets'].items() if c in shared]} all take column {shared}: each "
                         f"set needs its own ids. A set the model defines (positions of a tour, groups, slots) goes "
                         f"under settings, e.g. \"settings\": {{\"positions\": {{\"index_of\": \"sites\"}}}} (one id per site)")
    sized: dict[str, str] = {}
    for name, col in (output.get("sets") or {}).items():
        if isinstance(col, dict) and "index_of" in col:  # one id per member of another set: tour positions
            sized[name] = col["index_of"]
            continue
        if isinstance(col, list):                       # a set defined by the user, written under sets
            data[name] = [str(x) for x in col]
            continue
        if col not in df.columns:
            raise KeyError(f"set {name!r}: no column {col!r}; the table has {list(df.columns)[:30]}. A set the user "
                           f"defines (groups, teams, positions) goes under settings as a list of ids")
        _cols(df, col)
        ids = df[col].astype(str).map(str.strip).tolist()
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        if dup:
            raise ValueError(f"set {name!r}: ids repeat {dup}; merge the same place (merge_same_place / dedupe) or "
                             f"make ids unique with a parent column")
        data[name] = ids
    for name, col in (output.get("params") or {}).items():
        if not isinstance(col, str):          # a value, not a column: a setting written under params ("k": 3)
            data[name] = col
            continue
        _cols(df, col)
        vals = df[col].tolist()
        data[name] = [None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
                      for v in vals]
    for name, value in (output.get("settings") or {}).items():
        if isinstance(value, dict) and "index_of" in value:
            sized[name] = value["index_of"]
        else:
            data[name] = value
    for name, of in sized.items():
        if of not in data or not isinstance(data[of], list):
            raise KeyError(f"{name!r}: index_of names {of!r}, which is not a set of this output")
        data[name] = [str(i) for i in range(len(data[of]))]
    for name, d in (output.get("derived") or {}).items():
        if d.get("tool") == "haversine_matrix":
            m = prep.haversine_matrix(data[d["lat"]], data[d["lon"]])
            data[name] = np.round(m, 1).tolist()
        elif d.get("tool") == "distance_to":
            lat0, lon0 = d["center"]
            data[name] = np.round(prep.distance_to(data[d["lat"]], data[d["lon"]], lat0, lon0), 1).tolist()
        elif d.get("tool") == "road_matrix":
            m = prep.road_matrix(data[d["lat"]], data[d["lon"]])[d.get("value", "distance_m")]
            data[name] = np.round(np.asarray(m, float), 1).tolist()
        else:
            raise KeyError(f"derived {name!r}: unsupported tool {d.get('tool')!r}")
    return data


def execute(plan: dict, tables: dict[str, pd.DataFrame], log=None) -> tuple[dict, list[str], Any]:
    """Run a whole plan; returns (data, observations, log). Raises with the failing step's index and op.
    Inside the adapter sandbox the toolkit is exposed as module `prep` with call recording: use that one, so the
    required-tool checks see the plan's tool calls."""
    import sys
    global prep
    real, exposed = prep, sys.modules.get("prep")
    if exposed is not None and exposed is not real:
        prep = exposed
    try:
        return _execute(plan, tables, log)
    finally:
        prep = real


def _run_steps(df, steps, tables, log, seen, label):
    for i, step in enumerate(steps or []):
        before, flags = len(df), len(log.flags)
        try:
            df = run_step(df, step, tables, log)
        except Exception as e:
            raise ValueError(f"{label} step {i} ({step.get('op')}): {type(e).__name__}: {e}" + _so_far(seen)) from e
        problems = [f for f in log.flags[flags:] if not f.startswith("REVIEW")]
        if problems:      # a tool found a data problem (e.g. parents already filtered away): stop at this step
            raise ValueError(f"{label} step {i} ({step.get('op')}): DATA PROBLEM FOUND BY A TOOL: "
                             f"{'; '.join(problems)}" + _so_far(seen))
        seen.append(f"{label} step {i} {step.get('op')}: " + observe(df, before))
    return df


def _execute(plan: dict, tables: dict[str, pd.DataFrame], log=None) -> tuple[dict, list[str], Any]:
    log = log or prep.PrepLog()
    original = tables

    class _Local(dict):            # prepared tables; others are read (and so marked as used) only when a step needs them
        def __missing__(self, key):
            self[key] = original[key].copy()
            return self[key]
    tables = _Local()
    seen: list[str] = []
    # each table's own cleaning BEFORE it is joined: a single pipeline put the population table's steps (period,
    # category, parent) on the boundary table (smoke19, 2026-10-12)
    for name, steps in (plan.get("prepare") or {}).items():
        if name not in original:
            raise KeyError(f"prepare: table {name!r} is not one of {sorted(original)}")
        seen.append(f"prepare {name}: " + observe(tables[name]))
        tables[name] = _run_steps(tables[name], steps, tables, log, seen, f"prepare[{name}]")
    if plan.get("table") not in original:
        raise KeyError(f"plan table {plan.get('table')!r} is not one of {sorted(original)}")
    df = tables[plan["table"]].copy()
    seen.append(f"start {plan['table']}: " + observe(df))
    df = _run_steps(df, plan.get("steps"), tables, log, seen, "main")
    try:
        return assemble(df, plan.get("output") or {}), seen, log
    except Exception as e:
        raise ValueError(f"output: {type(e).__name__}: {e}" + _so_far(seen)) from e


def _so_far(seen: list[str]) -> str:
    """The results of the steps that ran, so the next plan is chosen from what the data really looked like."""
    return "\nRESULTS OF THE STEPS THAT RAN:\n" + "\n".join(x[:600] for x in seen)


def plan_to_code(answer: dict) -> str:
    """An adapter module that runs the plan; the rest of the pipeline (isolated run, checks, data double-check,
    required-tool tracking) treats it like any adapter."""
    if answer.get("gap"):
        return f"def load_data(tables, log):\n    raise ValueError({('DATA GAP: ' + str(answer['gap']))!r})\n"
    plan = {k: answer.get(k) for k in ("prepare", "table", "steps", "output")}
    return (
        "import json\nfrom qubo_auto.annealing import dataplan\n\n"
        f"PLAN = json.loads({json.dumps(plan, ensure_ascii=False)!r})\n"
        f"ADAPTER_NOTES = {str(answer.get('notes', ''))!r}\n"
        f"ADAPTER_METADATA = json.loads({json.dumps(answer.get('metadata') or {}, ensure_ascii=False)!r})\n\n\n"
        "def load_data(tables, log):\n"
        "    data, _seen, _ = dataplan.execute(PLAN, tables, log)\n"
        "    return data\n")
