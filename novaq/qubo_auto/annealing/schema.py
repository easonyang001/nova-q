"""DATA_SCHEMA as index sets + parameters (the AMPL/Pyomo idea): one problem-independent way to declare any data.

The Modeler decides WHICH sets and parameters a problem needs; this module fixes HOW they are declared and checks
that a data instance matches the declaration, so the Data Adapter fills a known shape instead of guessing.

    DATA_SCHEMA = {
      "sets": {"<set>": {"meaning": str, "source": "public|user|private|mock",
                         "reducible": {"method": "reduce_points", "weight": "<param>"}}},      # reducible optional
      "params": {"<param>": {"meaning": str, "index": ["<set>", ...], "type": "number|int|binary|text|datetime",
                             "source": ..., "unit": str, "sparse": bool,
                             "derived": {"tool": "<prep function or 'code'>", "from": [...], "how": str},
                             "value": ..., "min": ..., "max": ..., "max_missing_fraction": 0.0,
                             "period": str, "geography": str, "granularity": str}}}

Data instance (flat JSON dict): a set is a list of unique ids; a dense parameter is a nested list in set order (a
scalar when index is []); a sparse parameter is a list of [id, ..., value] rows. Canonical units: lengths in m,
durations in s, coordinates WGS84 as deg_lat / deg_lon, timestamps as ISO text.
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any

SOURCES = ("public", "user", "private", "mock", "model")
TYPES = ("number", "int", "binary", "text", "datetime")
SET_KEYS = {"meaning", "source", "reducible", "period", "geography", "granularity", "size_of"}
PARAM_KEYS = {"meaning", "index", "type", "source", "unit", "sparse", "derived", "value", "min", "max",
              "max_missing_fraction", "period", "geography", "granularity"}
CONTRACT_KEYS = ("unit", "period", "geography", "granularity", "max_missing_fraction", "value", "min", "max")
# non-canonical unit -> what to use instead (convert in the adapter, rescale inside build_ir if needed)
UNIT_FIX = {"km": "m", "kilometer": "m", "kilometre": "m", "mile": "m", "miles": "m", "cm": "m",
            "min": "s", "mins": "s", "minute": "s", "minutes": "s", "h": "s", "hr": "s", "hour": "s", "hours": "s",
            "ms": "s", "day": "s", "days": "s", "deg": "deg_lat or deg_lon", "degree": "deg_lat or deg_lon",
            "degrees": "deg_lat or deg_lon", "latitude": "deg_lat", "longitude": "deg_lon",
            "twd97": "deg_lat / deg_lon (convert with prep.to_wgs84)", "epsg:3826": "deg_lat / deg_lon"}


def prep_tools() -> set[str]:
    from . import prep
    return {n for n in dir(prep) if not n.startswith("_") and callable(getattr(prep, n))
            and not isinstance(getattr(prep, n), type)}


def is_structured(ds: Any) -> bool:
    return isinstance(ds, dict) and set(ds) == {"sets", "params"} and all(isinstance(ds[k], dict) for k in ds)


def field_keys(ds: Any) -> list[str]:
    """Every key a data instance must contain (sets and params; legacy flat schemas: their keys)."""
    if is_structured(ds):
        return list(ds["sets"]) + list(ds["params"])
    return list(ds or {})


def _clean(d: Any) -> Any:
    """Optional fields set to None mean 'not given' (models often write "min": None)."""
    return {k: v for k, v in d.items() if v is not None} if isinstance(d, dict) else d


def normalize(ds: Any) -> Any:
    if not is_structured(ds):
        return ds
    return {"sets": {k: _clean(v) for k, v in ds["sets"].items()},
            "params": {k: _clean(v) for k, v in ds["params"].items()}}


def declaration_errors(ds: Any) -> list[str]:
    ds = normalize(ds)
    if not is_structured(ds):
        return ['DATA_SCHEMA must be {"sets": {...}, "params": {...}} (index sets + parameters); see the format rules']
    errors, sets, params = [], ds["sets"], ds["params"]
    for name in set(sets) & set(params):
        errors.append(f"{name!r} is declared both as a set and as a parameter")
    for name, d in sets.items():
        if not isinstance(d, dict):
            errors.append(f"set {name!r}: declaration must be a dict")
            continue
        errors += [f"set {name!r}: unknown field {k!r} (allowed: {', '.join(sorted(SET_KEYS))})" for k in set(d) - SET_KEYS]
        if not str(d.get("meaning", "")).strip():
            errors.append(f"set {name!r}: give its meaning")
        if d.get("source") not in SOURCES:
            errors.append(f"set {name!r}: source must be one of {SOURCES}")
        if d.get("source") == "model" and d.get("size_of") not in set(sets) | set(params):
            errors.append(f"set {name!r}: a 'model' set (an index the formulation defines: tour positions, group labels, "
                          f"time slots) needs \"size_of\": the set with one id per member, or a user parameter giving the count")
        red = d.get("reducible")
        if red is not None:
            if not isinstance(red, dict) or red.get("method") != "reduce_points":
                errors.append(f"set {name!r}: reducible must be {{'method': 'reduce_points', 'weight': <param or null>}}")
            elif red.get("weight") is not None and red["weight"] not in params:
                errors.append(f"set {name!r}: reducible weight {red['weight']!r} is not a declared parameter")
    tools = prep_tools()
    for name, d in params.items():
        if not isinstance(d, dict):
            errors.append(f"param {name!r}: declaration must be a dict")
            continue
        errors += [f"param {name!r}: unknown field {k!r} (allowed: {', '.join(sorted(PARAM_KEYS))})"
                   for k in set(d) - PARAM_KEYS]
        if not str(d.get("meaning", "")).strip():
            errors.append(f"param {name!r}: give its meaning")
        index = d.get("index")
        if not isinstance(index, list):
            errors.append(f"param {name!r}: index must be a list of set names ([] for a single value)")
            index = []
        errors += [f"param {name!r}: index {s!r} is not a declared set" for s in index if s not in sets]
        if d.get("type") not in TYPES:
            errors.append(f"param {name!r}: type must be one of {TYPES}")
        if d.get("source") not in SOURCES:
            errors.append(f"param {name!r}: source must be one of {SOURCES}")
        if d.get("source") == "model":
            errors.append(f"param {name!r}: only sets can have source 'model'")
        if d.get("sparse") and len(index) < 2:
            errors.append(f"param {name!r}: sparse needs an index of at least two sets")
        if "value" in d and d.get("source") != "user":
            errors.append(f"param {name!r}: 'value' is only for user settings fixed by the confirmed specification")
        unit = str(d.get("unit", "")).strip()
        if "deg_lat" in unit and "deg_lon" in unit or "," in unit:
            errors.append(f"param {name!r}: one unit per parameter; declare latitude and longitude as two params "
                          f"(unit deg_lat and unit deg_lon), each indexed by the set")
        elif unit.lower() in UNIT_FIX:
            errors.append(f"param {name!r}: unit {unit!r} is not canonical; use {UNIT_FIX[unit.lower()]}")
        derived = d.get("derived")
        if derived is not None:
            if not isinstance(derived, dict) or not derived.get("tool"):
                errors.append(f"param {name!r}: derived must be {{'tool': <prep function or 'code'>, 'from': [...], 'how': str}}")
            else:
                if derived["tool"] != "code" and derived["tool"] not in tools:
                    errors.append(f"param {name!r}: derived tool {derived['tool']!r} is not a prep function "
                                  f"(use 'code' for a computation without a toolkit function)")
                for src in derived.get("from", []) or []:
                    if src not in params and src not in sets and not str(src).startswith("raw:"):
                        errors.append(f"param {name!r}: derived from {src!r}, which is neither declared nor 'raw:<column>'")
    return errors


def _key(v) -> str:
    return str(v)


def _leaves(value, depth):
    if depth == 0:
        yield value
    else:
        for item in value:
            yield from _leaves(item, depth - 1)


def _shape_error(value, sizes, path="") -> str | None:
    if not sizes:
        return None
    if not isinstance(value, list):
        return f"expected a list at {path or 'top level'}"
    if len(value) != sizes[0][1]:
        return f"length {len(value)} at {path or 'top level'} but set {sizes[0][0]!r} has {sizes[0][1]} ids"
    for i, item in enumerate(value):
        err = _shape_error(item, sizes[1:], f"{path}[{i}]")
        if err:
            return err
    return None


def _value_error(v, d) -> str | None:
    kind = d.get("type")
    if kind in ("number", "int", "binary"):
        if isinstance(v, bool) and kind != "binary":
            return f"boolean {v!r} where a {kind} is expected"
        if not isinstance(v, (int, float)):
            return f"{v!r} is not a number"
        if not math.isfinite(v):
            return f"{v!r} is not finite"
        if kind == "int" and float(v) != int(v):
            return f"{v!r} is not an integer"
        if kind == "binary" and v not in (0, 1):
            return f"{v!r} is not 0/1"
        unit = d.get("unit")
        if unit == "deg_lat" and not -90 <= v <= 90:
            return f"latitude {v} outside [-90, 90] (projected coordinates? use prep.to_wgs84)"
        if unit == "deg_lon" and not -180 <= v <= 180:
            return f"longitude {v} outside [-180, 180] (projected coordinates? use prep.to_wgs84)"
        if "min" in d and v < d["min"]:
            return f"{v} below the declared minimum {d['min']}"
        if "max" in d and v > d["max"]:
            return f"{v} above the declared maximum {d['max']}"
    elif kind == "text":
        if not isinstance(v, str):
            return f"{v!r} is not text"
    elif kind == "datetime":
        try:
            datetime.fromisoformat(str(v))
        except ValueError:
            return f"{v!r} is not an ISO date/time"
    return None


def data_errors(ds: dict, data: dict, check_values: bool = True) -> list[str]:
    """Does a data instance match a structured DATA_SCHEMA? (call only when is_structured(ds))
    check_values=False for small synthetic examples, which may scale confirmed settings (k=20 cannot hold for 6 ids)."""
    ds = normalize(ds)
    errors, ids = [], {}
    if isinstance(data, dict) and {"sets", "params"} <= set(data) and not ({"sets", "params"} & set(field_keys(ds))):
        return ["the data is nested under 'sets'/'params' like the schema; data must be FLAT: one top-level key "
                "per set (its id list) and one per parameter (its values), e.g. {'sites': [...], 'cost': [...], 'k': 5}"]
    declared = set(field_keys(ds))
    extra = [k for k in data if k not in declared]
    if extra:
        errors.append(f"keys {extra} are not declared in DATA_SCHEMA (declare them or drop them)")
    for name in ds["sets"]:
        if name not in data:
            errors.append(f"set {name!r} is missing")
            continue
        v = data[name]
        if not isinstance(v, list) or not v:
            errors.append(f"set {name!r} must be a non-empty list of ids")
            continue
        bad = [x for x in v if isinstance(x, bool) or not isinstance(x, (str, int))]
        if bad:
            errors.append(f"set {name!r}: ids must be strings or integers, got {bad[:3]}")
            continue
        keys = [_key(x) for x in v]
        if len(set(keys)) != len(keys):
            dup = sorted({k for k in keys if keys.count(k) > 1})[:5]
            errors.append(f"set {name!r}: ids repeat ({dup}); ids must be unique")
        ids[name] = set(keys)
    # (0, 0) is not a place: missing coordinates written as 0 ("Null Island", 125 parking lots, 2026-10-11)
    lat_p = [n for n, d in ds["params"].items() if d.get("unit") == "deg_lat" and not d.get("sparse")]
    for la in lat_p:
        lo = next((n for n, d in ds["params"].items() if d.get("unit") == "deg_lon"
                   and d.get("index") == ds["params"][la].get("index") and not d.get("sparse")), None)
        if lo and isinstance(data.get(la), list) and isinstance(data.get(lo), list) and len(ds["params"][la].get("index", [])) == 1:
            zeros = sum(1 for a, b in zip(data[la], data[lo]) if a == 0 and b == 0)
            if zeros:
                errors.append(f"params {la!r}/{lo!r}: {zeros} entities at (0, 0), which is not a real location; the "
                              f"source wrote missing coordinates as 0. Use another coordinate field of the same rows "
                              f"(e.g. projected x/y converted with prep.to_wgs84) or drop them and say so")
    for name, d in ds["params"].items():
        if name not in data:
            errors.append(f"param {name!r} is missing")
            continue
        index, v = d.get("index", []), data[name]
        if any(s not in ids for s in index):
            continue                                    # the set itself is already reported
        if d.get("sparse"):
            if not isinstance(v, list):
                errors.append(f"param {name!r} (sparse) must be a list of [{', '.join(index)}, value] rows")
                continue
            values = []
            for row in v:
                if not isinstance(row, list) or len(row) != len(index) + 1:
                    errors.append(f"param {name!r}: sparse row {row!r} must be [{', '.join(index)}, value]")
                    break
                unknown = [(s, r) for s, r in zip(index, row) if _key(r) not in ids[s]]
                if unknown:
                    errors.append(f"param {name!r}: sparse row {row!r} uses id {unknown[0][1]!r} not in set {unknown[0][0]!r}")
                    break
                values.append(row[-1])
        else:
            err = _shape_error(v, [(s, len(data[s])) for s in index])
            if err:
                errors.append(f"param {name!r}[{', '.join(index)}]: {err}")
                continue
            values = list(_leaves(v, len(index)))
        missing = [x for x in values if x is None]
        limit = float(d.get("max_missing_fraction", 0) or 0)
        if values and len(missing) / len(values) > limit:
            errors.append(f"param {name!r}: {len(missing)} of {len(values)} values missing (allowed {limit:.0%})")
        for x in values:
            if x is None:
                continue
            err = _value_error(x, d)
            if err:
                errors.append(f"param {name!r}: {err}")
                break
        if (check_values and d.get("source") == "public" and d.get("type") in ("number", "int") and len(index) == 1
                and len(values) >= 5 and len({x for x in values if x is not None}) == 1):
            # measured data are never all equal: a default filled in for missing data (population 1 everywhere)
            errors.append(f"param {name!r}: all {len(values)} values are {values[0]!r}; a measured public value is "
                          f"never constant. If the tables do not contain it, report the data gap instead of a default")
        if check_values and "value" in d and v != d["value"]:
            errors.append(f"param {name!r}: {v!r} differs from the confirmed setting {d['value']!r}")
    return errors


def contract(ds: dict) -> dict:
    """The same declaration as a DATA_CONTRACT (contracts.check_data) for every set and parameter."""
    ds = normalize(ds)
    out = {}
    for group in ("sets", "params"):
        for name, d in ds[group].items():
            rules = {"source_kind": d.get("source")}
            rules.update({k: d[k] for k in CONTRACT_KEYS if k in d})
            out[name] = rules
    return out


def required_tools(ds: dict) -> dict[str, str]:
    """prep functions the adapter must call: {tool: parameter that declares it}."""
    ds = normalize(ds)
    tools = prep_tools()
    return {d["derived"]["tool"]: name for name, d in ds["params"].items()
            if isinstance(d.get("derived"), dict) and d["derived"].get("tool") in tools}


def describe(ds: Any) -> str:
    """Compact text for the adapter prompt and the UI."""
    ds = normalize(ds)
    if not is_structured(ds):
        import json
        return json.dumps(ds, ensure_ascii=False, indent=1)
    lines = ["SETS (each one is a list of unique ids in the data):"]
    for name, d in ds["sets"].items():
        red = d.get("reducible")
        extra = (f"; GENERATED BY THE SYSTEM (one id per member of {d.get('size_of')}): do not provide it" if
                 d.get("source") == "model" else
                 f"; may be reduced only with prep.reduce_points (weight: {red.get('weight')})" if red else
                 "; keep ALL ids (not reducible)")
        lines.append(f"- {name} [{d.get('source')}]: {d.get('meaning')}{extra}")
    lines.append("PARAMETERS:")
    for name, d in ds["params"].items():
        index = d.get("index", [])
        shape = (f"[{', '.join(index)}]" if index else " (single value)")
        form = (f"SPARSE list of [{', '.join(index)}, value] rows" if d.get("sparse") else
                "nested list in set order" if index else "scalar")
        bits = [d.get("type", "?"), form, f"source {d.get('source')}"]
        if d.get("unit"):
            bits.append(f"unit {d['unit']}")
        if "value" in d:
            bits.append(f"= {d['value']!r} (confirmed setting)")
        for k in ("min", "max", "period", "geography", "granularity"):
            if k in d:
                bits.append(f"{k} {d[k]}")
        der = d.get("derived")
        if isinstance(der, dict):
            bits.append(f"derived by {'prep.' + der['tool'] if der.get('tool') != 'code' else 'code'}"
                        f" from {', '.join(map(str, der.get('from', []) or []))}" + (f" ({der['how']})" if der.get("how") else ""))
        lines.append(f"- {name}{shape}: {d.get('meaning')} — " + "; ".join(bits))
    return "\n".join(lines)


def public_view(ds: Any) -> Any:
    """What a data search should look for: only the public sets/params (user settings and private data are not
    searched online). Legacy schemas are returned unchanged."""
    ds = normalize(ds)
    if not is_structured(ds):
        return ds
    params = {k: v for k, v in ds["params"].items() if v.get("source") == "public"}
    used = {s for v in params.values() for s in v.get("index", [])}        # sets a public table is indexed by
    sets = {k: v for k, v in ds["sets"].items() if v.get("source") == "public" or k in used}
    return {"sets": sets, "params": params}


class _Tracking(dict):
    """A data dict that records which keys the model reads (iteration counts as reading everything)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: set[str] = set()

    def __getitem__(self, k):
        self.read.add(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.read.add(k)
        return super().get(k, default)

    def __contains__(self, k):
        return super().__contains__(k)

    def _all(self):
        self.read.update(super().keys())

    def keys(self):
        self._all()
        return super().keys()

    def items(self):
        self._all()
        return super().items()

    def values(self):
        self._all()
        return super().values()

    def __iter__(self):
        self._all()
        return super().__iter__()

    def copy(self):
        self._all()
        return dict(super().items())


def _used_keys(code: str, data: dict) -> list[str]:
    """Runs in a worker process: keys read by build_ir, plus describe_solution on all-one and all-zero solutions
    (an empty solution reads no names, a full one reads every per-item field)."""
    import numpy as np
    ns = {"np": np, "__source__": code}
    exec(compile(code, "<model>", "exec"), ns)
    tracked = _Tracking(data)
    raw = ns["build_ir"](tracked)
    n = len(raw.get("variable_names", []))
    for x in ([1] * n, [0] * n):
        try:
            ns["describe_solution"](tracked, x)
        except Exception:
            pass
    return sorted(tracked.read)


def unused_params(ds: dict, code: str, data: dict) -> list[str]:
    """Declared parameters the model never reads (sets are index anchors and may be read only via lengths)."""
    from .runtime import run_task
    read = set(run_task(__name__, "_used_keys", code, data))
    return [p for p in ds["params"] if p not in read]


def variable_key_errors(ds: dict, data: dict, raw: dict) -> list[str]:
    """build_ir's variable_keys: for every binary variable, the set ids it decides about ({} for helper/slack
    variables). This lets anyone read a solution as ids of the data, independently of the model's own names."""
    ds = normalize(ds)
    keys, n = raw.get("variable_keys"), len(raw.get("variable_names", []))
    if not isinstance(keys, list) or len(keys) != n:
        return [f"build_ir must return variable_keys: a list with one dict per variable ({n}), mapping set names to "
                f"the ids the variable decides about, e.g. {{'sites': 'S3'}} or {{'nurses': 'N1', 'days': 2}}; "
                f"{{}} for helper variables"]
    ids = {s: {_key(v) for v in data.get(s, [])} for s in ds["sets"]}
    for i, k in enumerate(keys):
        if not isinstance(k, dict):
            return [f"variable_keys[{i}] must be a dict"]
        for s, v in k.items():
            if s not in ids:
                return [f"variable_keys[{i}] uses {s!r}, which is not a declared set (declared sets: {sorted(ids)}; "
                        f"keys must be these exact set names, not index or role names). If variable {i} is a HELPER "
                        f"(slack, bits of an auxiliary number, an indicator that follows from the decisions), give it "
                        f"{{}}. Only if it is part of the decision the user asked for, declare {s!r} under DATA_SCHEMA "
                        f"'sets' and return its ids from make_example_data. A variable over two members of the SAME set (an "
                        f"assignment of each point to a site, a pair) that follows from the chosen decisions is a helper: give it {{}}"]
            if _key(v) not in ids[s]:
                sample = list(data.get(s, []))[:3]
                return [f"variable_keys[{i}] uses id {v!r}, which is not in set {s!r} (its ids look like {sample}): "
                        f"use the set's ids, not positions (data['{s}'][j], not j). If variable {i} is a helper that "
                        f"follows from the decisions (an assignment of a point to its nearest open site, a covered "
                        f"indicator), give it {{}}"]
    if not any(keys):
        return ["variable_keys are all empty: say which set ids each decision variable refers to"]
    seen_keys: dict[tuple, int] = {}
    for i, k in enumerate(keys):
        if not k:
            continue
        t = tuple(sorted((s, _key(v)) for s, v in k.items()))
        if t in seen_keys:
            return [f"variables {seen_keys[t]} and {i} have the same variable_keys {k}. A variable that decides about "
                    f"two things needs both ids (a stop AND its position: {{'stations': 'S1', 'positions': 3}}); "
                    f"declare the second set (e.g. positions) under DATA_SCHEMA 'sets'. Helper variables (covered / "
                    f"served indicators, slack) get {{}}"]
        seen_keys[t] = i
    # every member of a decision set must be decidable: a model that fixes a decision by leaving its variables out
    # (a fixed tour start for symmetry breaking) cannot be compared with independent checks, and its decoded solution
    # silently lacks that member (YouBike tour, 2026-10-12)
    used = {s: {_key(v) for k in keys if k for t, v in k.items() if t == s} for s in {t for k in keys if k for t in k}}
    for s, got in used.items():
        absent = [v for v in data.get(s, []) if _key(v) not in got]
        if absent:
            return [f"set {s!r} has members with no decision variable: {absent[:5]}. Create variables for EVERY member "
                    f"(do not fix a decision such as the start of a tour by leaving its variables out; symmetric "
                    f"solutions are fine). A member that can never be chosen must be removed in the data stage"]
    return []


def decisions(raw: dict, x: list) -> list[dict]:
    """The chosen decisions of a 0/1 solution, as set ids (helper variables omitted)."""
    return [k for k, b in zip(raw.get("variable_keys", []), x) if b and k]


def model_sets(ds: Any) -> dict[str, str]:
    """{set: size_of} for the sets the formulation itself defines (tour positions, group labels, time slots)."""
    ds = normalize(ds)
    if not is_structured(ds):
        return {}
    return {n: d.get("size_of") for n, d in ds["sets"].items() if isinstance(d, dict) and d.get("source") == "model"}


def fill_model_sets(ds: Any, data: dict) -> dict:
    """Generate the ids of 'model' sets from the prepared data: "0".."n-1", n = members of size_of (a set) or its
    value (a user parameter). The data never supplies them (station ids taken as tour positions, 2026-10-21)."""
    out = dict(data)
    for name, of in model_sets(ds).items():
        src = out.get(of)
        n = len(src) if isinstance(src, list) else int(src) if isinstance(src, (int, float)) and not isinstance(src, bool) else None
        if n is not None:
            out[name] = [str(i) for i in range(n)]
    return out
