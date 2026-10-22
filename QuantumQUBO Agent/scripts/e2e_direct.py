"""Baseline arm for the end-to-end experiment: direct LLM modelling and solving.

The same local model gets the request (sentence + the user's fixed answers), searches the web itself like a plain
browsing agent (search_files: raw web-search results, pick links, download, one level of links on a page) and writes
one module that loads the data, builds the binary model and describes a solution. Its only feedback is the Python error of the previous
attempt, with the same attempt count as the Modeling Agent. No Clarifier, toolkit, schema/contract/requirement
checks, baseline fallback or repair. The output format (sets/params data, variable_keys) is the one the
independent judge reads; it is a reporting format, not a check.
"""
from __future__ import annotations
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from qubo_auto.annealing.bridge import extract_code
from qubo_auto.annealing.datakit import load_tables, profiles
from qubo_auto.annealing.modeler import to_ir, to_jsonable
from qubo_auto.annealing.runtime import run_task

PROMPT = """You are an optimisation engineer. Solve the user's request as a QUBO (binary model) that an annealer will solve.

REQUEST:
{request}

DATA FILES: already loaded as pandas DataFrames in the dict `tables` (name -> DataFrame). Profiles:
{profiles}

Write ONE Python module (pandas as pd and numpy as np are available) that defines:

DATA_SCHEMA = {{"sets": {{"<set>": {{"meaning": str, "source": "public|user|private|mock"}}}},
               "params": {{"<param>": {{"meaning": str, "index": ["<set>", ...], "type": "number|int|binary|text",
                                       "source": "public|user|private|mock", "unit": str}}}}}}
    # coordinates as two params with unit "deg_lat" and "deg_lon"; lengths in m

def load_data(tables) -> dict:
    # FLAT JSON dict: each set -> list of unique ids; each param -> nested lists in set order (scalar if index []).

def build_ir(data) -> dict:
    # {{"name": str, "sense": "minimize" | "maximize", "variable_names": [str, ...],
    #  "variable_keys": [{{"<set>": <id>, ...}}, ...],   # per variable: ALL the set ids it decides about (a distinct
    #                                                  # combination per decision variable); {{}} for helpers
    #  "linear": {{"<i>": coef}}, "quadratic": [[i, j, coef], ...], "constant": float,
    #  "constraints": [{{"name": str, "kind": "linear", "coeffs": {{"<i>": coef}}, "sense": "le|ge|eq", "rhs": number}},
    #                  {{"name": str, "kind": "exclusion", "pairs": [[i, j], ...]}},
    #                  {{"name": str, "kind": "implication", "pairs": [[i, j], ...]}}]}}
    # Hard constraints go in "constraints" (the system adds penalties); integer coefficients.

def describe_solution(data, x) -> dict:
    # what is chosen, from the data

{feedback}
Output: Python code only, in a single ```python ... ``` block.
"""


_QUERIES = """You need public data files to solve this request:
{request}

Write up to 3 web search queries that would find downloadable data files (CSV, JSON, Excel, shapefile ZIP) for it.
Return JSON only: {{"queries": ["...", ...]}}"""

_PICK = """Request:
{request}

Candidate links (from a web search or from a page you opened):
{links}

Pick up to {k} links that most likely ARE downloadable data files (or pages that link to them) for this request.
Return JSON only: {{"urls": ["<one of the urls above>", ...]}}"""

DATA_EXT = (".csv", ".json", ".geojson", ".xlsx", ".xls", ".zip")


def _ask_json(client, model, prompt, step):
    raw = client.call(model=model, prompt=prompt, temperature=0.0, step_name=step, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _get(url: str):
    """Same TLS/IPv4 handling as the system (some government certificates fail Python's strict X.509 checks);
    this is network setup, not search or data skill."""
    from qubo_auto.annealing.websearch import _session
    return _session().get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (research data download)"})


def search_files(request: str, client, model: str, dest: Path, max_files: int = 3, log=print):
    """A plain browsing agent: web search (raw results, no source ranking), pick links, download; if a page is HTML,
    list the data-file links on it and pick again. No open-data platform APIs, no relevance screening."""
    import re
    from urllib.parse import urljoin
    from ddgs import DDGS
    dest.mkdir(parents=True, exist_ok=True)
    record = {"queries": [], "results": [], "downloaded": []}
    queries = _ask_json(client, model, _QUERIES.format(request=request), "direct_queries").get("queries", [])[:3]
    record["queries"] = queries
    hits = []
    for q in queries:
        try:
            hits += [{"title": h.get("title", ""), "url": h.get("href", "")} for h in DDGS().text(q, max_results=8)]
        except Exception as e:
            log(f"[direct] search failed: {e}")
    record["results"] = hits
    links = "\n".join(f"- {h['url']} | {h['title']}" for h in hits)
    urls = [u for u in _ask_json(client, model, _PICK.format(request=request, links=links, k=max_files),
                                 "direct_pick").get("urls", []) if u in {h["url"] for h in hits}]
    files = []
    for url in urls:
        if len(files) >= max_files:
            break
        try:
            r = _get(url)
            r.raise_for_status()
        except Exception as e:
            log(f"[direct] download failed {url}: {e}")
            continue
        ctype = r.headers.get("content-type", "").lower()
        if "html" in ctype and not url.lower().endswith(DATA_EXT):
            found = sorted({urljoin(url, h) for h in re.findall(r'href="([^"]+)"', r.text)
                            if h.lower().split("?")[0].endswith(DATA_EXT) or "download" in h.lower()})[:40]
            if not found:
                continue
            sub = _ask_json(client, model, _PICK.format(request=request, links="\n".join(f"- {u}" for u in found),
                                                        k=max_files - len(files)), "direct_pick_page").get("urls", [])
            for u in [u for u in sub if u in found][: max_files - len(files)]:
                try:
                    rr = _get(u)
                    rr.raise_for_status()
                    files.append(_save(rr, u, dest)); record["downloaded"].append(u)
                except Exception as e:
                    log(f"[direct] download failed {u}: {e}")
            continue
        files.append(_save(r, url, dest)); record["downloaded"].append(url)
    return files, record


def _save(r, url: str, dest: Path) -> Path:
    import hashlib
    raw = r.content
    ctype = r.headers.get("content-type", "").lower()
    ext = next((e for e in DATA_EXT if url.lower().split("?")[0].endswith(e)), None) or (
        ".zip" if raw[:2] == b"PK" else ".json" if "json" in ctype else ".csv")
    path = dest / f"file_{hashlib.sha256(url.encode()).hexdigest()[:8]}{ext}"
    path.write_bytes(raw)
    return path


def _run(code: str, tables: dict):
    ns = {"pd": pd, "np": np}
    exec(compile(code, "<direct>", "exec"), ns)
    data = json.loads(json.dumps(to_jsonable(ns["load_data"]({k: v.copy() for k, v in tables.items()}))))
    raw = ns["build_ir"](data)
    to_ir(raw, "direct")
    # the reporting format the judge reads (not a modelling check): one distinct key per decision variable
    keys = [tuple(sorted((str(a), str(b)) for a, b in k.items())) for k in raw.get("variable_keys", []) if k]
    if len(raw.get("variable_keys", [])) != len(raw["variable_names"]) or not keys:
        raise ValueError("variable_keys must have one dict per variable, naming the set ids it decides about")
    if len(set(keys)) != len(keys):
        raise ValueError("variable_keys repeat: each decision variable needs its own id combination, e.g. "
                         "{'stations': 'S1', 'positions': 3}; use {} for helper variables")
    return {"schema": ns["DATA_SCHEMA"], "data": data, "raw": json.loads(json.dumps(to_jsonable(raw)))}


def build(request: str, files: list[Path], client, model: str, attempts: int = 4, log=print):
    tables = load_tables(files)
    feedback, code, last = "", "", "not attempted"
    for attempt in range(attempts):
        prompt = PROMPT.format(request=request, profiles=profiles(tables), feedback=feedback)
        raw = client.call(model=model, prompt=prompt, temperature=0.0, step_name="direct")
        code = extract_code(raw) if raw.strip() else ""
        try:
            out = run_task("scripts.e2e_direct", "_run", code, tables, timeout=120)
            return code, out, []
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:3000]
            log(f"[direct] attempt {attempt + 1} failed: {last[:200]}")
            feedback = ("YOUR PREVIOUS MODULE FAILED WITH THIS ERROR:\n" + last +
                        "\n\nPREVIOUS MODULE:\n```python\n" + code[:20000] + "\n```")
    return code, None, [last]
