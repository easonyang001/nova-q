"""Modeling Agent: confirmed specification -> typed model (ProblemIR) + example data + solution reader.

The LLM writes a small module (DATA_SCHEMA, make_example_data, build_ir, describe_solution). Everything it
produces is checked here before use: the IR must validate, indices must be in range, a small instance must
have a feasible solution, and describe_solution must run.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bridge import extract_code
from .encodings import ExclusionConstraint, ImplicationConstraint, LinearConstraint, ProblemIR
from .runtime import generated_module
from . import schema

_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "modeler.txt").read_text(encoding="utf-8")
REQUIRED = ("DATA_SCHEMA", "make_example_data", "build_ir", "describe_solution")


def load_module(code: str) -> dict[str, Any]:
    ns = generated_module(code)
    missing = [k for k in REQUIRED if k not in ns]
    if missing:
        raise ValueError(f"model module is missing: {missing}")
    if "DATA_CONTRACT" not in ns and schema.is_structured(ns["DATA_SCHEMA"]) \
            and not schema.declaration_errors(ns["DATA_SCHEMA"]):
        # one declaration: the sets/params schema carries the contract fields (source, unit, period, value ...)
        ns["DATA_CONTRACT"] = schema.contract(ns["DATA_SCHEMA"])
    return ns


def to_ir(raw: dict, instance_id: str) -> ProblemIR:
    """Plain dict from build_ir -> validated ProblemIR (+ variable names in `name_map`)."""
    names = [str(v) for v in raw["variable_names"]]
    n = len(names)
    by_name = {v: i for i, v in enumerate(names)}

    def idx(k) -> int:
        """Variable reference -> index: accepts an index (int or numeric string) or a variable name."""
        if isinstance(k, int):
            return k
        k = str(k)
        if k.lstrip("-").isdigit():
            return int(k)
        if k in by_name:
            return by_name[k]
        raise ValueError(f"unknown variable reference {k!r}: use an index 0..{n - 1} or a name from variable_names "
                         f"(e.g. {names[:3]})")

    cons = []
    for c in raw.get("constraints", []):
        if c["kind"] == "linear":
            cons.append(LinearConstraint(name=c["name"], coeffs={idx(k): float(v) for k, v in c["coeffs"].items()},
                                         sense=c["sense"], rhs=float(c["rhs"])))
        elif c["kind"] == "exclusion":
            cons.append(ExclusionConstraint(name=c["name"], pairs=[(idx(a), idx(b)) for a, b in c["pairs"]]))
        elif c["kind"] == "implication":
            cons.append(ImplicationConstraint(name=c["name"], pairs=[(idx(a), idx(b)) for a, b in c["pairs"]]))
        else:
            raise ValueError(f"constraint '{c.get('name')}' has unsupported kind {c.get('kind')!r}")
    quad: dict[tuple[int, int], float] = {}
    for i, j, v in raw.get("quadratic", []):
        i, j = idx(i), idx(j)
        if i == j:
            raise ValueError(f"quadratic term ({i},{i}) on the diagonal: put it in 'linear' (x*x = x for binary x)")
        key = (min(i, j), max(i, j))
        quad[key] = quad.get(key, 0.0) + float(v)
    ir = ProblemIR(name=raw.get("name", "model"), instance_id=instance_id, n_variables=n, sense=raw["sense"],
                   linear={idx(k): float(v) for k, v in raw.get("linear", {}).items()}, quadratic=quad,
                   constant=float(raw.get("constant", 0.0)), constraints=cons)
    bad = [i for i in list(ir.linear) + [k for p in quad for k in p] if not 0 <= i < n]
    for c in cons:
        bad += [i for i in (list(getattr(c, "coeffs", {})) + [k for p in getattr(c, "pairs", []) for k in p]) if not 0 <= i < n]
    if bad:
        raise ValueError(f"variable indices out of range 0..{n - 1}: {sorted(set(bad))[:10]}")
    cnames = [c.name for c in cons]
    if len(cnames) != len(set(cnames)):
        dup = sorted({n for n in cnames if cnames.count(n) > 1})
        raise ValueError(f"constraint names must be unique; repeated: {dup[:5]}. Give each constraint created in a loop "
                         f"its own name with the loop index, e.g. f\"{dup[0]}_{{i}}\" (REQUIREMENT_MAP can then use "
                         f"the prefix pattern \"{dup[0]}_*\")")
    return ir


def brute_force(ir: ProblemIR, max_n: int = 18) -> tuple[float | None, list[int] | None, int]:
    """(best objective, argmax/argmin, number of feasible points) by enumeration, or (None, None, -1) if too big."""
    if ir.n_variables > max_n:
        return None, None, -1
    best, best_x, n_feas = None, None, 0
    for bits in itertools.product((0, 1), repeat=ir.n_variables):
        x = list(bits)
        if all(c.satisfied(x) for c in ir.constraints):
            n_feas += 1
            v = ir.objective(x)
            if best is None or (v > best if ir.sense == "maximize" else v < best):
                best, best_x = v, x
    return best, best_x, n_feas


def reference(ir: ProblemIR) -> float | None:
    """Exact optimum: enumeration for small models, MILP when the objective is linear."""
    if ir.n_variables <= 18:
        return brute_force(ir)[0]
    if ir.quadratic:
        return None
    from .examples import milp_reference
    n = ir.n_variables
    c = np.zeros(n)
    for i, v in ir.linear.items():
        c[i] = v
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for con in ir.constraints:
        if isinstance(con, LinearConstraint):
            row = np.zeros(n)
            for i, v in con.coeffs.items():
                row[i] = v
            if con.sense == "le":
                a_ub.append(row), b_ub.append(con.rhs)
            elif con.sense == "ge":
                a_ub.append(-row), b_ub.append(-con.rhs)
            else:
                a_eq.append(row), b_eq.append(con.rhs)
        else:
            for i, j in con.pairs:
                row = np.zeros(n)
                if isinstance(con, ExclusionConstraint):
                    row[i] = row[j] = 1
                    a_ub.append(row), b_ub.append(1)
                else:                                   # x_i <= x_j
                    row[i], row[j] = 1, -1
                    a_ub.append(row), b_ub.append(0)
    val = milp_reference(c, a_ub or None, b_ub or None, a_eq or None, b_eq or None, maximize=ir.sense == "maximize")
    return None if val is None else val + ir.constant


def missing_hard_constraints(spec_json: str, report: dict) -> list[str]:
    """The confirmed spec lists hard constraints but the model lists none: they were folded into the objective as
    fixed penalties (seen 2026-10-10: a CVRP model with 0 constraints), so nothing can be tuned or checked."""
    try:
        hard = [c for c in json.loads(spec_json).get("constraints", []) if c.get("strength") == "hard"]
    except (json.JSONDecodeError, AttributeError):
        return []
    if hard and report.get("constraints") == []:
        return [f"the specification has {len(hard)} hard constraint(s) ({', '.join(c.get('name', '?') for c in hard[:6])}) "
                f"but the model lists none in 'constraints'. List every hard constraint there (linear, exclusion or "
                f"implication); do not fold them into the objective as penalties"]
    return []


def trivially_separable(ir) -> bool:
    """Linear objective, and every variable sits in at most ONE "choose K" constraint (coefficients 1): sorting by
    objective coefficient inside each group is exactly optimal, so no annealer is needed (seen 2026-10-07: a
    p-median spec turned into "pick the 20 best individual scores"). Overlapping constraints (assignment: one per
    row AND one per column; set cover; connectivity) are NOT sortable; a first version of this check missed that
    and wrongly rejected every assignment, set-cover and network-design model (2026-10-11)."""
    if any(ir.quadratic.values()) or not ir.constraints:
        return False
    seen: set[int] = set()
    for c in ir.constraints:
        if getattr(c, "kind", "") != "linear" or any(v != 1 for v in c.coeffs.values()):
            return False
        if seen & set(c.coeffs):
            return False                              # a variable in two constraints: the choices interact
        seen |= set(c.coeffs)
    return True


def looped(text: str, repeats: int = 40) -> bool:
    """A degenerate answer: one non-trivial line repeated many times."""
    from collections import Counter
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
    return bool(lines) and Counter(lines).most_common(1)[0][1] >= repeats


def extract_module(raw: str, required: tuple[str, ...] = REQUIRED) -> str:
    """The usable module in an LLM answer. Answers sometimes hold several code blocks, prose inside a block
    ("I will provide the model with x_i ...") or an echoed instruction followed by a second copy; taking the first
    block then fails with "unterminated string" (2026-10-10, 1 of 3 runs). Try every block, trimming trailing
    non-code lines, and keep the version that compiles and defines the most required names."""
    import re
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)(?:```|\Z)", raw, re.DOTALL) or [raw]
    best, best_score = extract_code(raw), (-1, 0)
    for b in blocks:
        lines = b.rstrip().splitlines()
        for cut in range(len(lines), max(0, len(lines) - 40), -1):      # drop up to 40 trailing lines
            cand = "\n".join(lines[:cut]).strip()
            try:
                compile(cand, "<model>", "exec")
            except SyntaxError:
                continue
            score = (sum(f"def {r}" in cand or f"{r} =" in cand for r in required), len(cand))
            if score > best_score:
                best, best_score = cand, score
            break
    return best


def penalty_in_objective(ir) -> list[str]:
    """Equality constraints whose squared penalty lambda*(sum c_i x_i - b)^2 was also written into the objective:
    every pair of their variables is coupled with 2*lambda*c_i*c_j for one lambda > 0 (seen 2026-10-07: a
    p-median model double-encoding 'choose exactly p sites')."""
    found = []
    for c in ir.constraints:
        if getattr(c, "kind", "") != "linear" or c.sense != "eq" or len(c.coeffs) < 3:
            continue
        idx = sorted(c.coeffs)
        lams = []
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1:]:
                q = ir.quadratic.get((i, j), ir.quadratic.get((j, i), 0.0))
                lams.append(q / (2 * c.coeffs[i] * c.coeffs[j]))
        if lams and min(lams) > 1e-9 and max(lams) - min(lams) <= 1e-6 * max(lams):
            found.append(c.name)
    return found


def to_jsonable(obj: Any) -> Any:
    """numpy arrays / scalars -> plain lists and numbers, recursively (data is stored as data.json)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    import datetime as _dt
    if isinstance(obj, (_dt.datetime, _dt.date)) or type(obj).__name__ == "Timestamp":
        return obj.isoformat()                      # dates in data or report keys broke JSON storage (2026-10-12)
    return obj


def validate(ns: dict[str, Any], small_size: int = 6, require_visualize: bool = False,
             strict_schema: bool = False) -> tuple[list[str], dict]:
    """Run the module on a small synthetic instance and check it. Returns (errors, report).
    strict_schema: DATA_SCHEMA must use the sets/params format (new models); legacy flat schemas still load."""
    errors, report = [], {}
    ds = ns["DATA_SCHEMA"]
    structured = schema.is_structured(ds)
    if structured or strict_schema:
        # report as many independent problems as possible in one round: returning at the first one used up all four
        # attempts on four different problems, one per attempt (London, 2026-10-12)
        errors += schema.declaration_errors(ds)
        if "DATA_CONTRACT" not in ns and not errors:
            ns["DATA_CONTRACT"] = schema.contract(ds)
    contract = ns.get("DATA_CONTRACT")
    if contract is not None:
        if not isinstance(contract, dict):
            return ["DATA_CONTRACT must be a dict"], report
        missing_contract = set(schema.field_keys(ds)) - set(contract)
        if missing_contract:
            errors.append("DATA_CONTRACT is missing DATA_SCHEMA fields: " + ", ".join(sorted(missing_contract)))
    try:
        raw = ns["make_example_data"](small_size, 0)
        try:      # the UI stores data as JSON and reloads it, so build_ir must work on the JSON form
            data = json.loads(json.dumps(to_jsonable(raw)))
        except (TypeError, ValueError) as e:
            errors.append(f"make_example_data returned data that cannot be stored as JSON ({e}); "
                          f"use dicts, lists, numbers and strings only")
            return errors, report
        if structured:
            bad = schema.data_errors(ds, data, check_values=False)
            if bad:
                errors += [f"make_example_data does not follow DATA_SCHEMA: {b}" for b in bad]
        try:
            built = ns["build_ir"](data)
        except Exception as e:
            errors.append(describe_error(e, ns.get("__source__", "")) + "\n    (build_ir receives the data after a "
                          "JSON round trip: arrays arrive as nested lists; convert with np.asarray where needed)")
            return errors, report
        if structured and not schema.declaration_errors(ds):
            try:
                errors += schema.variable_key_errors(ds, data, built)
            except Exception:
                pass
        try:
            ir = to_ir(built, "validation")
        except Exception as e:
            errors.append(describe_error(e, ns.get("__source__", "")))
            return errors, report
        if errors:
            return errors, report
        if structured and ns.get("__source__"):
            unused = schema.unused_params(ds, ns["__source__"], data)
            if unused:
                # a warning, not an error: as an error it set off a rewrite cascade (2026-10-11) for fields that are
                # legitimately useful elsewhere (coordinates for figures and for independent checks)
                report.setdefault("warnings", []).append(
                    f"DATA_SCHEMA 宣告了模型沒有讀取的參數：{unused}")
        if not any(ir.linear.values()) and not any(ir.quadratic.values()):
            # passes every other check, yet every feasible choice scores the same (seen: a p-median model whose
            # objective was dropped, leaving only "choose exactly K sites")
            errors.append("the objective is empty (no non-zero linear or quadratic term): every feasible solution "
                          "scores the same. Encode the specification's objective. If it needs a min/max over choices "
                          "(e.g. distance to the NEAREST open site, as in p-median), add assignment variables "
                          "y[i,j] = 'demand j served by site i' with constraints sum_i y[i,j] = 1 and y[i,j] <= x[i]; "
                          "y follows from x, so its variable_keys are {}")
        if trivially_separable(ir):
            if str(ns.get("MODEL_CHOICE", "")).lower().startswith("separable"):
                report.setdefault("warnings", []).append(
                    "這個模型可用排序直接求解（目標全線性、限制只有選 K 個），Modeling Agent 聲明這是刻意的："
                    + str(ns["MODEL_CHOICE"])[:300])
            else:
                errors.append("the model is solvable by sorting: the objective is purely linear and the only "
                              "constraints are 'choose K'. Choices do not interact, so no annealer is needed and "
                              "interactions in the specification (overlap, distance, nearest facility ...) are lost. "
                              "Re-model so the objective keeps those interactions; if the specification truly has none, "
                              "start MODEL_CHOICE with 'separable:' and give the reason")
        dup = penalty_in_objective(ir)
        if dup:
            errors.append(f"the objective contains a squared penalty for the hard constraint(s) {dup} (uniform positive "
                          f"couplings between all their variables). Remove it from the objective: list hard constraints "
                          f"in 'constraints' only; the system builds and tunes the penalties")
        report["n_variables"] = ir.n_variables
        report["constraints"] = [f"{c.name} ({c.kind}{'/' + c.sense if hasattr(c, 'sense') else ''})" for c in ir.constraints]
        best, x, n_feas = brute_force(ir)
        size = small_size
        while n_feas == -1 and size > 2:
            # too many variables to enumerate (assignment models grow as N*M): prove correctness on a smaller instance
            size -= 1
            data = json.loads(json.dumps(to_jsonable(ns["make_example_data"](size, 0))))
            ir = to_ir(ns["build_ir"](data), "validation")
            best, x, n_feas = brute_force(ir)
        report["checked_size"] = size if n_feas != -1 else None
        report["_data"] = data
        try:
            report["_var_names"] = list(ns["build_ir"](data).get("variable_names", []))
        except Exception:
            report["_var_names"] = []
        report["model_choice"] = str(ns.get("MODEL_CHOICE", ""))[:600]
        if n_feas == 0:
            errors.append(f"no feasible solution on the small example (size={small_size}): hard constraints conflict. "
                          f"A common cause is a parameter not scaled to `size`, e.g. selecting a fixed 20 sites when "
                          f"make_example_data({small_size}, seed) only creates {small_size} candidates; derive such "
                          f"numbers from the data")
        elif x is not None:
            report["small_example_optimum"] = best
            report["small_example_solution"] = ns["describe_solution"](data, x)
            report["small_example_x"] = list(x)
            if require_visualize:
                errors += check_visualize(ns, data, x, report)
        missing = [k for k in schema.field_keys(ds) if k not in data]
        if missing:
            errors.append(f"make_example_data does not provide DATA_SCHEMA keys {missing}")
    except Exception as e:
        errors.append(describe_error(e, ns.get("__source__", "")))
    return errors, report


def check_visualize(ns: dict[str, Any], data: dict, x: list, report: dict) -> list[str]:
    """visualize() must return a drawable spec for a real solution (the figure on the results page)."""
    from .render import check_spec
    if "visualize" not in ns:
        return ["define visualize(data, x) returning a 'map', 'schedule' or 'routes' spec (see the module contract)"]
    try:
        spec = ns["visualize"](data, list(x))
    except Exception as e:
        return [describe_error(e, ns.get("__source__", ""))]
    problems = check_spec(to_jsonable(spec))
    if not problems:
        report["figure_type"] = spec["type"]
    return [f"visualize(): {p}" for p in problems]


def figure(ns: dict[str, Any], data: dict, x: list, path: Path) -> Path | None:
    """Render the model's figure for solution x; None when the model has no (valid) visualize()."""
    from .render import check_spec, render
    if "visualize" not in ns:
        return None
    try:
        spec = to_jsonable(ns["visualize"](data, list(x)))
    except Exception:
        return None
    return None if check_spec(spec) else render(spec, path)


def describe_error(e: Exception, code: str) -> str:
    """The exception plus WHERE in the generated model it happened: without a line the Modeling Agent retried
    blindly (seen: "TypeError: only 0-dimensional arrays can be converted to Python scalars", 3 attempts)."""
    import re
    import traceback
    lines = code.splitlines()
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<model>"]
    msg = f"{type(e).__name__}: {e}"
    for f in frames[-3:]:
        src = lines[f.lineno - 1].strip() if f.lineno and 0 < f.lineno <= len(lines) else ""
        msg += f"\n    in {f.name}(), line {f.lineno}: {src}"
    if not frames:
        # the model ran in a worker process: its frames are only in the traceback text, which names the line but
        # not its code, and the agent retried the same bug four times (Taipei chargers, 2026-10-12)
        for no, name in re.findall(r'File "<model>", line (\d+), in (\w+)', str(e))[-3:]:
            no = int(no)
            if 0 < no <= len(lines):
                msg += f"\n    in {name}(), line {no}: {lines[no - 1].strip()}"
    return msg


PATTERNS_PATH = Path(__file__).parent.parent.parent / "knowledge" / "modeling_patterns.jsonl"
_PICK = """You are preparing to write a binary optimisation model for this confirmed specification:
{spec}

A library of GENERAL modelling patterns is available (titles only):
{catalogue}

Pick at most {k} patterns that are building blocks of THIS problem (none if none fits). Return JSON only:
{{"ids": ["<pattern id>", ...], "why": "<one sentence>"}}"""


def load_patterns(path: Path = PATTERNS_PATH) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def select_patterns(spec_json: str, client, model: str, k: int = 3) -> tuple[list[str], str]:
    """Retrieval step: the Modeling Agent picks up to k general patterns by title; their text is returned for the
    modelling prompt. The patterns are building blocks, not the formulation of this problem."""
    lib = {p["id"]: p for p in load_patterns()}
    if not lib:
        return [], ""
    catalogue = "\n".join(f"- {p['id']}: {p['title']} [{', '.join(p.get('tags', [])[:5])}]" for p in lib.values())
    raw = client.call(model=model, prompt=_PICK.replace("{spec}", spec_json[:6000]).replace("{catalogue}", catalogue)
                      .replace("{k}", str(k)), temperature=0.0, step_name="pattern_pick", json_mode=True)
    try:
        ids = [i for i in json.loads(raw).get("ids", []) if i in lib][:k]
    except (json.JSONDecodeError, AttributeError):
        ids = []
    text = "\n".join(f"[{i}] {lib[i]['title']}: {lib[i]['text']} (source: {lib[i]['source']})" for i in ids)
    return ids, text


def build_model(spec_json: str, qa_json: str, client, model: str, max_attempts: int = 4,
                temperature: float = 0.0, backend: str = "", previous_code: str = "",
                repair_feedback: str = "", patterns: bool | list = False) -> tuple[str, dict[str, Any], list[str], dict]:
    """Ask the Modeling Agent, validate, retry with the errors. Returns (code, namespace, errors, report).
    patterns=True first retrieves up to three general modelling patterns (knowledge/modeling_patterns.jsonl);
    a list of pattern ids reuses an earlier choice (repairs keep the same patterns)."""
    if isinstance(patterns, list):
        lib = {p["id"]: p for p in load_patterns()}
        pattern_ids = [i for i in patterns if i in lib]
        pattern_text = "\n".join(f"[{i}] {lib[i]['title']}: {lib[i]['text']} (source: {lib[i]['source']})"
                                 for i in pattern_ids)
    else:
        pattern_ids, pattern_text = select_patterns(spec_json, client, model) if patterns else ([], "")
    pattern_block = ("GENERAL MODELLING PATTERNS (optional building blocks; use one only where it fits the binding requirements, "
                     "the formulation is yours):\n" + pattern_text + "\n") if pattern_text else ""
    feedback = ("EXECUTION FEEDBACK: " + repair_feedback + "\nPreserve every confirmed requirement. Do not weaken hard "
                "constraints, fabricate data or change the objective to obtain feasibility. You may change the variables or the "
                "formulation; the requirements stay as confirmed.\n"
                + "PREVIOUS MODULE:\n```python\n" + previous_code[:24000] + "\n```\n") if repair_feedback else ""
    code, ns, errors, report = "", {}, ["not attempted"], {}
    for _ in range(max_attempts):
        prompt = (_PROMPT.replace("{spec}", spec_json).replace("{qa}", qa_json)
                  .replace("{backend}", backend or "(not specified)")
                  .replace("{patterns}", pattern_block)
                  .replace("{previous_feedback}", feedback))
        try:
            raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="model")
        except Exception as e:
            if "Timeout" not in type(e).__name__:
                raise
            raw = ""                    # a timed-out answer is one failed attempt, like an exhausted reasoning budget
        if not raw.strip():
            # the reasoning model spent its whole budget thinking: keep the real feedback, don't replace it
            # with "code failed to load" (that lost the actual problem in a 2026-10-07 run)
            errors = errors if errors != ["not attempted"] else ["empty answer"]
            feedback = ("YOUR PREVIOUS ANSWER WAS EMPTY (you ran out of thinking time or budget before writing code). Think "
                        "briefly and write the module directly; keep it short.\n" + feedback)
            continue
        if looped(raw):
            # seen 2026-10-11: a no-reasoning retry argued with itself in code comments until the token limit
            errors = errors if errors != ["not attempted"] else ["looping answer"]
            feedback = ("YOUR PREVIOUS ANSWER REPEATED THE SAME LINES UNTIL IT WAS CUT OFF. Write the module directly; "
                        "do not deliberate in comments.\n" + feedback)
            continue
        code = extract_module(raw)
        try:
            ns = load_module(code)
            errors, report = validate(ns, strict_schema=True)
        except Exception as e:
            ns, errors = {}, [f"code failed to load: {type(e).__name__}: {e}"]
        errors += missing_hard_constraints(spec_json, report)
        if ns and not errors:
            from .contracts import check_requirements
            errors += check_requirements(json.loads(spec_json), ns, ns["build_ir"](report["_data"]))
        report["patterns"] = pattern_ids
        if not errors:
            add_visualization(code, ns, report, client, model, temperature)
            return ns["__source__"], ns, [], report
        # show the agent its own previous module: retries used to rewrite from scratch and threw away a nearly
        # correct model (2026-10-10: attempt 3 had one index bug, attempt 4 regressed to an empty objective)
        feedback = ("PREVIOUS ATTEMPT FAILED VALIDATION. Fix every problem:\n- " + "\n- ".join(errors)
                    + "\n\nChange only the lines these problems involve; keep the formulation and everything that "
                      "already works.\nYOUR PREVIOUS MODULE (return the complete corrected module):\n```python\n"
                    + code[:24000] + "\n```")
    return code, ns, errors, report


_VIS_PROMPT = (Path(__file__).parent.parent / "prompts" / "annealing" / "visualizer.txt").read_text(encoding="utf-8")


def _shorten(obj: Any, n: int = 8) -> Any:
    if isinstance(obj, dict):
        return {k: _shorten(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shorten(v, n) for v in obj[:n]] + (["..."] if len(obj) > n else [])
    return obj


def sample_solution(ns: dict[str, Any], data: dict, reads: int = 50) -> list[int] | None:
    """A good (preferably feasible) solution of the model by quick simulated annealing, decision variables only."""
    try:
        from dwave.samplers import SimulatedAnnealingSampler
        import dimod
        from .encodings import compile_problem
        ir = to_ir(ns["build_ir"](data), "sample")
        prob = compile_problem(ir)
        qubo, offset = prob.to_qubo(prob.default_penalty_weights)
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
        for i in range(prob.n_variables):
            bqm.add_variable(i)
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=reads, num_sweeps=2000, seed=0)
        best = None
        for sample, energy in sorted(((s, e) for s, e in ss.data(["sample", "energy"])), key=lambda t: t[1]):
            x = [int(sample.get(i, 0)) for i in range(ir.n_variables)]
            if all(c.satisfied(x) for c in ir.constraints):
                return x
            best = best or x
        return best
    except Exception:
        return None


def add_visualization(code: str, ns: dict[str, Any], report: dict, client, model: str, temperature: float = 0.0,
                      attempts: int = 2) -> bool:
    """Second, small task after the model validated: write visualize(). The model works without it (no figure),
    so a failure here never fails the model. Splitting it off keeps the Modeling Agent's task small enough for a
    local model (one combined task kept running out of thinking budget)."""
    data, x = report.get("_data"), report.get("small_example_x")
    if data is not None and x is None:
        # too many variables for brute force (e.g. 8 staff x 7 days x 2 shifts at any size): the figure step was
        # silently skipped for every scheduling model (2026-10-11). Any good solution will do for drawing.
        x = sample_solution(ns, data)
        report["small_example_x"] = x
    if "visualize" in ns:
        problems = check_visualize(ns, data, x, report) if data is not None and x is not None else []
        if not problems:
            return True
        ns.pop("visualize", None)
    if data is None or x is None:
        report.setdefault("warnings", []).append("沒有成果圖：小實例沒有可行解可以用來測試繪圖")
        return False
    feedback = ""
    for _ in range(attempts):
        prompt = (_VIS_PROMPT.replace("{schema}", json.dumps(ns.get("DATA_SCHEMA", {}), ensure_ascii=False))
                  .replace("{sample}", json.dumps(_shorten(data), ensure_ascii=False)[:4000])
                  .replace("{described}", json.dumps(to_jsonable(report.get("small_example_solution")), ensure_ascii=False, default=str)[:2000])
                  .replace("{n_vars}", str(len(x)))
                  .replace("{var_names}", ", ".join(map(str, report.get("_var_names", [])[:30])))
                  .replace("{previous_feedback}", feedback))
        raw = client.call(model=model, prompt=prompt, temperature=temperature, step_name="visualizer")
        fn_code = extract_code(raw) if raw.strip() else ""
        try:
            trial = load_module(code.rstrip() + "\n\n" + fn_code)
            problems = check_visualize(trial, data, x, report)
        except Exception as e:
            problems = [f"{type(e).__name__}: {e}"]
        if not problems:
            ns["visualize"] = trial["visualize"]
            ns["__source__"] = code.rstrip() + "\n\n\n" + fn_code.strip() + "\n"
            return True
        feedback = "PREVIOUS ATTEMPT FAILED:\n- " + "\n- ".join(problems) + "\nPrevious function:\n```python\n" + fn_code[:6000] + "\n```"
    report.setdefault("warnings", []).append("沒有成果圖：visualize() 兩次都沒通過檢查（模型本身可以正常使用）")
    return False
