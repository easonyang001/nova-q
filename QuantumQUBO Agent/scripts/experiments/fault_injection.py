"""E2: fault injection. Which check layer catches which error, and how often a layer raises a false alarm.

Seeds are runs that passed the independent acceptance (data, model and delivery). Each fault is injected into the
seed's final data or into its model's IR; then every layer looks at the faulty version:
  L1  data contract / schema checks           (schema.data_errors, contracts.check_data)
  L2  requirement Checker vs model             (the run's own checker.py, reqcheck.Validator.preflight)
  J   independent acceptance judge             (ground truth: is the fault actually harmful?)
The unmodified seed measures false alarms. No LLM is called (the data double-check, which needs one, is reported
separately). A development tool until the seeds and the fault list are frozen.

    python scripts/experiments/fault_injection.py e2e_smoke16_20261012 e2e_smoke17_20261012
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "acceptance", ROOT / "scripts"):
    sys.path.insert(0, str(p))
from judge import judge                                                    # noqa: E402
from qubo_auto.annealing import contracts, modeler as M, reqcheck, schema  # noqa: E402
import run_e2e_batch as B                                                  # noqa: E402

TRUTHS = [ROOT / "results/acceptance_truth/truth_20261011.json", ROOT / "results/acceptance_truth/truth_intl_20261011.json"]


# ---------------------------------------------------------------- faults
def _main_set(ds, data):
    sets = [s for s, d in schema.normalize(ds)["sets"].items() if d.get("source") == "public" and isinstance(data.get(s), list)]
    return max(sets, key=lambda s: len(data[s])) if sets else None


def _aligned(ds, s):
    return [n for n, d in schema.normalize(ds)["params"].items() if d.get("index") == [s]]


def data_faults(ds, data):
    s = _main_set(ds, data)
    if not s:
        return {}
    n, out = len(data[s]), {}
    keep = [i for i in range(n) if i % 10 != 3]                         # D1: about 10 % of the entities lost
    d = copy.deepcopy(data)
    for name in [s] + _aligned(ds, s):
        d[name] = [d[name][i] for i in keep]
    out["D1 entities missing"] = d
    d = copy.deepcopy(data)                                             # D2: one place listed twice under two ids
    d[s] = d[s] + [str(d[s][0]) + "_2"]
    for name in _aligned(ds, s):
        d[name] = d[name] + [d[name][0]]
    out["D2 duplicate place"] = d
    lat = next((p for p in _aligned(ds, s) if "lat" in p.lower()), None)
    lon = next((p for p in _aligned(ds, s) if "lon" in p.lower()), None)
    if lat and lon:                                                     # D3: lat / lon swapped for three entities
        d = copy.deepcopy(data)
        for i in range(min(3, n)):
            d[lat][i], d[lon][i] = d[lon][i], d[lat][i]
        out["D3 swapped coordinates"] = d
    pop = next((p for p in _aligned(ds, s) if p.lower() in ("pop", "population", "weight", "w")), None)
    if pop:                                                             # D4: a measured value off by x10
        d = copy.deepcopy(data)
        for i in range(min(3, n)):
            d[pop][i] = d[pop][i] * 10
        out["D4 values x10"] = d
    if lat and lon:
        d = copy.deepcopy(data)                                         # D5: a higher-level total row mixed in
        d[s] = d[s] + ["總計"]
        for name in _aligned(ds, s):
            col = d[name]
            nums = [v for v in col if isinstance(v, (int, float))]
            d[name] = col + ([sum(nums)] if name == pop else [sum(nums) / len(nums)] if nums and len(nums) == len(col) else [col[0]])
        out["D5 total row added"] = d
        d = copy.deepcopy(data)                                         # D6: three entities from outside the area
        extra = list(range(min(3, n)))
        d[s] = d[s] + [f"{d[s][i]}_outside" for i in extra]
        for name in _aligned(ds, s):
            d[name] = d[name] + [d[name][i] + (0.05 if name == lat else 0) if isinstance(d[name][i], (int, float))
                                 else d[name][i] for i in extra]
        out["D6 entities outside the area"] = d
    return out


def model_faults(raw):
    out = {}
    cons = raw.get("constraints", [])
    if cons:
        family = cons[0]["name"].rsplit("_", 1)[0]                      # M1: one constraint family dropped
        r = copy.deepcopy(raw)
        r["constraints"] = [c for c in cons if not c["name"].startswith(family)]
        out[f"M1 constraints '{family}*' dropped"] = r
    r = copy.deepcopy(raw)                                              # M2: objective sense flipped
    r["linear"] = {k: -v for k, v in (r.get("linear") or {}).items()}
    r["quadratic"] = [[a, b, -v] for a, b, v in (r.get("quadratic") or [])]
    out["M2 objective flipped"] = r
    if raw.get("quadratic"):                                            # M3: interactions lost (e.g. count once)
        r = copy.deepcopy(raw)
        r["quadratic"] = []
        out["M3 quadratic terms dropped"] = r
    eq = [c for c in cons if c.get("kind", "linear") == "linear" and c.get("sense") == "eq"]
    if eq:                                                              # M4: a count off by one
        big = max(eq, key=lambda c: len(c.get("coeffs", {})))
        r = copy.deepcopy(raw)
        for c in r["constraints"]:
            if c["name"] == big["name"]:
                c["rhs"] = c["rhs"] + 1
        out[f"M4 '{big['name']}' rhs + 1"] = r
    keys = raw.get("variable_keys") or []
    first = next((k for k in keys if k), None)
    if first:                                                          # M5: one entity can never be chosen
        s0, v0 = next(iter(first.items()))
        idx = [i for i, k in enumerate(keys) if k and k.get(s0) == v0]
        r = copy.deepcopy(raw)
        r["constraints"] = list(r.get("constraints", [])) + [{"name": "fault_fixed_out", "kind": "linear", "sense": "eq",
                                                             "rhs": 0, "coeffs": {str(i): 1 for i in idx}}]
        out[f"M5 entity {v0} fixed out"] = r
    lin = raw.get("linear") or {}
    if len(set(lin.values())) > 1:                                     # M6: objective weights from the wrong column
        import random
        r = copy.deepcopy(raw)
        vals = list(lin.values())
        random.Random(0).shuffle(vals)
        r["linear"] = dict(zip(lin.keys(), vals))
        out["M6 objective weights permuted"] = r
    return out


# ---------------------------------------------------------------- layers
def layer_data(ns, data, report):
    errs = schema.data_errors(ns["DATA_SCHEMA"], data)
    try:
        e2, _ = contracts.check_data(data, ns.get("DATA_CONTRACT"), report.get("data_metadata"), report.get("sources"))
        errs += e2
    except Exception as e:
        errs.append(f"{type(e).__name__}: {e}")
    return errs


def layer_checker(checker_code, data, raw):
    try:
        v = reqcheck.Validator(checker_code, reqcheck.decision_sets(raw))
        return v.preflight(data, raw)
    except Exception as e:
        return [f"checker error {type(e).__name__}"]


def build_layers(d, wf, ns, data, raw, client, model, builds):
    """LLM-built checkers for one seed, `builds` times: Checker code (L2) and the data cross-check's expected counts (L3)."""
    from qubo_auto.annealing import clarify as C, datacheck
    from qubo_auto.annealing.datakit import load_tables
    from qubo_auto.annealing.prep import situations as sit
    sess = json.loads((d / "session.json").read_text())
    draft = C.DraftSpec.model_validate(sess["draft"])
    spec_json = draft.model_dump_json(indent=1)
    files = [f for f in (d / "data_files").iterdir() if not f.name.startswith("_")]
    tables = load_tables(files)
    situations = "\n".join(f"- {x.text()}" for x in sit.detect(tables)) or "(none detected)"
    example = M.to_jsonable(ns["make_example_data"](20, 0))
    example = json.loads(json.dumps(example))
    out = []
    for b in range(builds):
        code, errs = reqcheck.build_checker(spec_json, ns["DATA_SCHEMA"], example, ns["build_ir"](example), client, model)
        _, expected, cerrs = datacheck.build(C.spec_for_data(draft), ns["DATA_SCHEMA"], tables, client, model,
                                            situations=situations)
        out.append({"checker": "" if errs else code, "expected": expected if not cerrs else {}})
    return out


def layer_crosscheck(expected, ns, data):
    from qubo_auto.annealing import datacheck
    if not expected:
        return ["no cross-check"]
    return datacheck.compare(expected, data, ns["DATA_SCHEMA"])


def harmful(case, truth, ds, data, raw):
    try:
        v = judge(case, truth, ds, data, raw, None, {})
    except Exception:
        return None
    return not (v.get("data", {}).get("ok") and v.get("model", {}).get("ok"))


def main(tags, checker_model=None, builds=0):
    """builds = 0: the run's own saved checker only (no LLM). builds > 0: build the checkers anew with checker_model
    (e.g. local2:gemma-4-31b) that many times per seed and add the data cross-check layer."""
    client = None
    if builds:
        from qubo_auto.llm_client import LLMClient
        client = LLMClient(ROOT / "results/experiments/fault_injection_transcripts", "e2", max_tokens=5_000_000,
                           use_cache=False)
        checker_model = checker_model or B.MODEL
    truth = {}
    for t in TRUTHS:
        truth.update(json.loads(t.read_text()))
    cases = {c["id"]: c for c in json.loads((ROOT / "acceptance/cases.json").read_text())["cases"]}
    rows = []
    for tag in tags:
        for rj in json.loads((ROOT / "results" / tag / "rejudged.json").read_text()):
            if not rj["joint"]:
                continue
            d = ROOT / "results" / tag / rj["arm"] / rj["run"]
            wf = json.loads((d / "workflow.json").read_text())
            ns, data = M.load_module(wf["_final_code"]), wf["_final_data"]
            raw = ns["build_ir"](data)
            checker = (d / "checker.py").read_text() if (d / "checker.py").exists() else ""
            report = (wf.get("report") or {})
            tr, _ = B.live_truth(cases[rj["case"]], truth, d, B.system_downloads(d / "data_files"))
            built = build_layers(d, wf, ns, data, raw, client, checker_model, builds) if builds else \
                [{"checker": checker, "expected": {}}]
            variants = [("none (seed)", data, raw)]
            variants += [(k, v, ns["build_ir"](v) if not schema.data_errors(ns["DATA_SCHEMA"], v) else raw)
                         for k, v in data_faults(ns["DATA_SCHEMA"], data).items()]
            variants += [(k, data, r) for k, r in model_faults(raw).items()]
            for fault, dv, rv in variants:
                l1 = layer_data(ns, dv, report)
                hurt = harmful(rj["case"], tr, ns["DATA_SCHEMA"], dv, rv)
                for b, layers in enumerate(built):
                    l2 = layer_checker(layers["checker"], dv, rv) if layers["checker"] else ["no checker"]
                    l3 = layer_crosscheck(layers["expected"], ns, dv) if builds else ["not run"]
                    usable2, usable3 = bool(layers["checker"]), bool(layers["expected"])
                    row = {"seed": f"{tag}/{rj['case']}", "fault": fault, "build": b, "checker_model": checker_model or "saved",
                           "L1_data": bool(l1), "L2_checker": usable2 and bool(l2) and not l2[0].startswith("checker error"),
                           "L3_crosscheck": usable3 and bool(l3), "L2_available": usable2, "L3_available": usable3,
                           "judge_harmful": hurt, "L1_why": "; ".join(l1)[:160], "L2_why": "; ".join(l2)[:160],
                           "L3_why": "; ".join(l3)[:160]}
                    rows.append(row)
                    print(f"{row['seed']:<38} {fault:<34} b{b} L1={row['L1_data']!s:<5} L2={row['L2_checker']!s:<5} "
                          f"L3={row['L3_crosscheck']!s:<5} harmful={hurt}", flush=True)
    name = (checker_model or "saved").replace(":", "_").replace("/", "_")
    out = ROOT / f"results/experiments/fault_injection_dev_{name}_b{builds}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    caught = [r for r in rows if r["fault"] != "none (seed)" and r["judge_harmful"]]
    any_layer = lambda r: r["L1_data"] or r["L2_checker"] or r["L3_crosscheck"]
    print(f"harmful (fault, build) pairs: {len(caught)}; caught by L1 {sum(r['L1_data'] for r in caught)}, L2 "
          f"{sum(r['L2_checker'] for r in caught)}, L3 {sum(r['L3_crosscheck'] for r in caught)}, any {sum(map(any_layer, caught))}")
    seeds = [r for r in rows if r["fault"] == "none (seed)"]
    print(f"false alarms on {len(seeds)} seed builds: L1 {sum(r['L1_data'] for r in seeds)}, L2 "
          f"{sum(r['L2_checker'] for r in seeds)}, L3 {sum(r['L3_crosscheck'] for r in seeds)}; checkers usable: "
          f"L2 {sum(r['L2_available'] for r in seeds)}, L3 {sum(r['L3_available'] for r in seeds)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--checker-model", default=None, help="e.g. local2:gemma-4-31b; default: the main model")
    ap.add_argument("--builds", type=int, default=0, help="rebuild the LLM checkers this many times per seed")
    a = ap.parse_args()
    main(a.tags, a.checker_model, a.builds)
