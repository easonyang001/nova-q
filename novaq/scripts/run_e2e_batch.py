"""End-to-end batch: three arms on the same cases, budget and independent judge (acceptance/).

  full       sentence -> Clarifier (fixed user answers) -> Modeler -> data search, download, toolkit-enforced
             adapter -> checks -> solver, with cross-stage repair (data->model before solving; workflow repairs)
  no_repair  the same with cross-stage repair switched off (each agent still retries within its own stage)
  direct     direct LLM modelling: sentence + answers + the official raw files, Python-error feedback only

Every run starts empty (no model, data or LLM cache). Code, prompts, budget and judge are frozen in manifest.json
before the first run; failures are kept, never retried by hand.

    python scripts/run_e2e_batch.py --tag e2e_20261011 --repeats 3
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acceptance"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
# defaults for every new experiment (user decision, 2026-10-13): look-then-plan data agent, and the independent
# checkers on a different model (Gemma on GPU 1). Set these variables explicitly to override, e.g. for an ablation.
os.environ.setdefault("QUBO_ADAPTER_MODE", "agent")
os.environ.setdefault("LLM_MODEL_CHECKER", "local2:gemma-4-31b")
os.environ.setdefault("QUBO_BACKEND", "sa")          # local SA until the data stage is settled (2026-10-13)
# local reasoning answers can take minutes; a 180 s client timeout turned slow answers into stage failures
os.environ.setdefault("LLM_REQUEST_TIMEOUT_S", "900")
import yaml

from qubo_auto.annealing import clarify as C
from qubo_auto.annealing import modeler as M
from qubo_auto.annealing.acquisition import prepare_real_data
from qubo_auto.annealing.artifacts import ArtifactStore, atomic_write
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.workflow import run_workflow
from qubo_auto.annealing.loop import run_loop
from qubo_auto.annealing.repair import make_repair, prepare_with_repair
from qubo_auto.annealing import reqcheck
from qubo_auto.annealing.guardrail import HardwareBudget
from scripts import e2e_direct
from qubo_auto.annealing.datakit import load_tables
from qubo_auto.llm_client import LLMClient
from judge import judge

MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
# the mechanism ladder: each arm adds one mechanism to the one before it (R0 is the direct-LLM baseline)
ARMS = {
    "pipeline":   {"checker": False, "repair": False, "patterns": False, "web": True},    # R1
    "no_repair":  {"checker": True,  "repair": False, "patterns": False, "web": True},    # R2
    "full":       {"checker": True,  "repair": True,  "patterns": False, "web": True},    # R3
    "full_rag":   {"checker": True,  "repair": True,  "patterns": True,  "web": True},    # R3 + modelling patterns
    "full_noweb": {"checker": True,  "repair": True,  "patterns": False, "web": False},   # R3 without web lookup
}


def digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest(cases, truth_paths):
    # everything that can change a result: code, prompts, retrieved knowledge and the solver configuration
    code = (list((ROOT / "qubo_auto").rglob("*.py")) + list((ROOT / "qubo_auto" / "prompts").rglob("*.txt"))
            + list((ROOT / "knowledge").glob("*.jsonl")) + [ROOT / "config_loop.yaml"]
            + [ROOT / "scripts/run_e2e_batch.py", ROOT / "scripts/e2e_direct.py"])
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_head": git, "working_tree_note": "uncommitted changes are captured by the hashes below",
            "code_prompts_knowledge_config_sha256": digest(code), "acceptance_sha256": digest(list((ROOT / "acceptance").glob("*.py"))
                                                                               + [ROOT / "acceptance/cases.json"]),
            "truth": [str(t) for t in truth_paths],
            "truth_sha256": hashlib.sha256(b"".join(t.read_bytes() for t in truth_paths)).hexdigest(),
            "llm_model": MODEL, "thinking": os.environ.get("LLM_ENABLE_THINKING", "(default)"),
            "llm_request_timeout_s": os.environ.get("LLM_REQUEST_TIMEOUT_S"),
            "adapter_mode": os.environ.get("QUBO_ADAPTER_MODE"), "backend": os.environ.get("QUBO_BACKEND"), "checker_model": os.environ.get("LLM_MODEL_CHECKER"),
            "checker_thinking": os.environ.get("LLM_ENABLE_THINKING_LOCAL2", "(default off)"),
            "max_completion_tokens": os.environ.get("LLM_MAX_COMPLETION_TOKENS", "(default)"),
            "budget": cases["budget"], "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def hardware_budget(cfg):
    hb = cfg["hardware_budget"]
    return HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})


def run_one(case, arm, budget, truth, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rec = {"case": case["id"], "arm": arm, "stages": {}, "failed_stage": None, "started": time.time(), "repairs": []}
    client = LLMClient(out / "transcripts", "e2e", max_tokens=500_000, use_cache=False)
    say = lambda m: (print(f"  [{case['id']}] {m}", flush=True), (out / "log.txt").open("a").write(m + "\n"))

    def stage(name, fn):
        t = time.time()
        try:
            value = fn()
            rec["stages"][name] = {"ok": True, "seconds": round(time.time() - t, 1)}
            return value
        except Exception as e:
            rec["stages"][name] = {"ok": False, "seconds": round(time.time() - t, 1),
                                   "error": f"{type(e).__name__}: {e}"[:3000]}
            rec["failed_stage"] = rec["failed_stage"] or name
            say(f"{name} failed: {type(e).__name__}: {str(e)[:300]}")
            (out / f"{name}_traceback.txt").write_text(traceback.format_exc())
            return None

    # solver override (user decision 2026-10-13): while the data stage is the question, solve locally with SA so the
    # one-job-at-a-time CGA service is not the bottleneck; the main experiment's backend is decided at freezing
    budget = dict(budget, backend=os.environ.get("QUBO_BACKEND", budget["backend"]))
    rec["backend"] = budget["backend"]
    if arm == "direct":
        return run_direct(case, budget, truth, out, rec, client, stage, say)
    mech = ARMS[arm]
    rec["mechanisms"] = mech
    budget = dict(budget, data_check=mech["checker"])     # requirement checks cover the data as well as the model

    # 1. Clarifier with the fixed user answers; confirmation = checkpoint 1
    def clarify():
        s = C.new_session(case["sentence"], budget["backend"])
        s = C.step(s, client, MODEL, web=mech["web"])
        for _ in range(budget["clarify_rounds"] - 1):
            if s.ready_to_confirm and not s.pending_questions:
                break
            s = C.step(s, client, MODEL, answers=case["answers"], web=mech["web"])
        try:
            s = C.confirm(s)
        except ValueError:
            s = C.step(s, client, MODEL, answers=case["answers"], web=mech["web"])
            s = C.confirm(s)
        s.save(out / "session.json")
        return s
    s = stage("clarify", clarify)
    if s is None:
        return finish(rec, out, client)
    rec["human"] = {"answers_given": len(s.rounds), "spec_confirmed": 1, "data_approved": 1}
    spec_json, qa = s.draft.model_dump_json(indent=1), json.dumps(s.rounds, ensure_ascii=False, indent=1)
    spec_text = C.spec_for_data(s.draft)
    say("spec confirmed: " + s.draft.problem_summary[:200])

    # 2. Modeler
    def model():
        code, ns, errors, report = M.build_model(spec_json, qa, client, MODEL, max_attempts=budget["model_attempts"],
                                                 backend=f"{s.backend}: {C.BACKEND_LIMITS.get(s.backend, '')}",
                                                 patterns=mech["patterns"])
        rec["patterns"] = report.get("patterns", [])
        if mech["patterns"]:
            budget["pattern_ids"] = report.get("patterns", [])          # repairs keep the same patterns
        atomic_write(out / "model.py", code)
        if errors:
            raise ValueError("model validation failed: " + " | ".join(errors))
        return code, ns, report
    got = stage("model", model)
    if got is None:
        return finish(rec, out, client)
    code, ns, model_report = got
    rec["model_choice"] = model_report.get("model_choice", "")
    rec["schema"] = ns["DATA_SCHEMA"]

    # 2b. independent requirement Checker (spec + data declaration only; never the model code)
    def checker():
        example = model_report["_data"]
        try:
            # the 6-item validation example can be smaller than a confirmed count (choose 8 sites): the Checker's
            # sampler then failed on it (Berlin, 2026-10-12). 20 items, as the UI's default instance
            bigger = json.loads(json.dumps(M.to_jsonable(ns["make_example_data"](20, 0))))
            raw_example = ns["build_ir"](bigger)
            example = bigger
        except Exception:
            raw_example = ns["build_ir"](example)
        ccode, errors = reqcheck.build_checker(spec_json, ns["DATA_SCHEMA"], example, raw_example, client, MODEL)
        atomic_write(out / "checker.py", ccode)
        if errors:
            raise ValueError("checker failed: " + " | ".join(errors))
        return reqcheck.Validator(ccode, reqcheck.decision_sets(raw_example))
    validator = stage("checker", checker) if mech["checker"] else None
    if rec["failed_stage"] == "checker":
        rec["failed_stage"] = None                     # without a Checker the run continues, unvalidated

    # 3. data search, download and preprocessing; approval = checkpoint 2 (auto-approved, judged independently)
    backend_text = f"{s.backend}: {C.BACKEND_LIMITS.get(s.backend, '')}"

    def data():
        nonlocal code, ns
        if mech["repair"]:
            new_code, data, errors, report, repairs = prepare_with_repair(
                code, spec_json, qa, spec_text, client, MODEL, backend_text, out / "data_files", budget,
                max_model_repairs=budget["max_model_repairs_before_solve"], log=say)
            rec["repairs"] += repairs
            if new_code != code:
                code, ns = new_code, M.load_module(new_code)
                atomic_write(out / "model_repaired.py", code)
        else:
            data, errors, report = prepare_real_data(code, spec_text, client, MODEL, out / "data_files",
                                                     search_rounds=budget["data_search_rounds"],
                                                     max_downloads=budget["max_downloads"], log=say,
                                                     data_check=budget["data_check"])
        if errors:
            raise ValueError("data preparation failed: " + " | ".join(map(str, errors))[:2500])
        atomic_write(out / "instance.json", json.dumps(data, ensure_ascii=False))
        return data, report
    got = stage("data", data)
    rec["data_found"] = found_files(out / "data_files")
    if got is None:
        return finish(rec, out, client)
    data, prep = got
    rec["prep"] = {k: prep.get(k) for k in ("tools_used", "notes", "prep_log", "flags", "review_required", "sources",
                                             "substitution",
                                             "n_variables", "n_constraints", "data_check")}
    raw_ir = stage("build_ir", lambda: ns["build_ir"](data))

    # 4. solve on the fixed backend and budget
    def solve():
        cfg = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8"))
        cfg["loop"]["max_solver_calls"] = budget["max_solver_calls"]
        repairs = budget["max_workflow_repairs"] if mech["repair"] else 0
        cfg.setdefault("workflow", {})["max_repairs"] = repairs
        store = ArtifactStore(out / "artifacts")
        store.commit(code, data, "e2e real data", {"source": "real", "report": prep, "confirmed_spec": s.draft.model_dump(),
                                                   "data_review_approved": True})
        repair = make_repair(spec_json, qa, spec_text, client, MODEL, backend_text, out / "data_files", budget,
                             log=say) if repairs else None
        result = run_workflow(store, get_backend(budget["backend"]), lambda: make_policy(budget["policy"]), cfg,
                              budget=hardware_budget(cfg), repair=repair, validator=validator, log=say)
        events = [json.loads(l) for l in (store.root / "workflow.jsonl").read_text(encoding="utf-8").splitlines()]
        rec["requirement_checks"] = [{k: e.get(k) for k in ("status", "version", "issues", "report")}
                                     for e in events if e.get("stage") == "requirement_check"]
        rec["workflow_events"] = [f"{e['stage']}:{e['status']}" for e in events]
        rec["repairs"] += [{"workflow": m} for m in result.get("repairs", [])] if isinstance(result, dict) else []
        final = store.read(result["version"]) if result.get("version") else store.current()  # the delivered version
        result["_final_code"], result["_final_data"] = final["code"], final["data"]
        atomic_write(out / "workflow.json", json.dumps(result, ensure_ascii=False, default=str))
        return result
    result = stage("solve", solve)          # also when build_ir failed: the preflight sees it and may route a repair
    if result is not None:
        rec["delivery"] = {k: result["delivery"].get(k) for k in ("status", "message", "solution_source", "quality_gap",
                                                                  "objective", "baseline_objective", "review_required",
                                                                  "formula_changes", "solver_raw")}

    # 5. independent judge, on the model and data the solution belongs to
    if result is not None and result.get("_final_code") != code:
        code, data = result["_final_code"], result["_final_data"]
        ns = M.load_module(code)
        raw_ir = ns["build_ir"](data)
        atomic_write(out / "model_final.py", code)
        atomic_write(out / "instance_final.json", json.dumps(data, ensure_ascii=False))
    elif result is not None and result.get("_final_data") != data:
        data = result["_final_data"]
        raw_ir = ns["build_ir"](data)
        atomic_write(out / "instance_final.json", json.dumps(data, ensure_ascii=False))
    judge_truth, rec["truth_used"] = live_truth(case, truth, out, system_downloads(out / "data_files"))
    verdict = stage("judge", lambda: judge(case["id"], judge_truth, ns["DATA_SCHEMA"], data, raw_ir,
                                           (result or {}).get("solution"), {}))
    rec["verdict"] = verdict
    return finish(rec, out, client, result)


def run_direct(case, budget, truth, out, rec, client, stage, say):
    request = case["sentence"] + "\n使用者補充：" + case["answers"]
    rec["human"] = {"answers_given": 1, "spec_confirmed": 0, "data_approved": 0}

    def find():
        files, record = e2e_direct.search_files(request, client, MODEL, out / "data_files", log=say)
        rec["search"] = record
        readable = []
        for f in files:
            try:
                load_tables([f]); readable.append(f)
            except Exception as e:
                say(f"[direct] unreadable {f.name}: {e}")
        if not readable:
            raise ValueError("no readable data file found")
        return readable
    files = stage("data", find)
    rec["data_found"] = [f.name for f in files or []]
    if files is None:
        return finish(rec, out, client)

    def build():
        code, got, errors = e2e_direct.build(request, files, client, MODEL, attempts=budget["model_attempts"], log=say)
        atomic_write(out / "direct.py", code)
        if errors:
            raise ValueError("direct module failed: " + errors[0])
        atomic_write(out / "instance.json", json.dumps(got["data"], ensure_ascii=False))
        return got
    got = stage("model", build)
    if got is None:
        return finish(rec, out, client)
    rec["schema"] = got["schema"]

    def solve():
        cfg = yaml.safe_load((ROOT / "config_loop.yaml").read_text(encoding="utf-8"))
        cfg["loop"]["max_solver_calls"] = budget["max_solver_calls"]
        ir = M.to_ir(got["raw"], f"direct_{case['id']}")
        res = run_loop(None, get_backend(budget["backend"]), make_policy(budget["policy"]), cfg, ir=ir,
                       root=out, log=say, seed=0, budget=hardware_budget(cfg))
        x = res.best_solution[:ir.n_variables] if res.best_solution else None
        feasible = x is not None and all(c.satisfied(x) for c in ir.constraints)
        result = {"solution": x if feasible else None, "solver_calls": res.solver_calls,
                  "delivery": {"status": "usable" if feasible else "no_solution", "solution_source": "annealer",
                               "objective": res.best_objective}}
        atomic_write(out / "workflow.json", json.dumps(result, ensure_ascii=False, default=str))
        return result
    result = stage("solve", solve)
    if result is not None:
        rec["delivery"] = result["delivery"]
    downloaded = dict(zip(rec.get("search", {}).get("downloaded", []), files or []))
    judge_truth, rec["truth_used"] = live_truth(case, truth, out, downloaded)
    rec["verdict"] = stage("judge", lambda: judge(case["id"], judge_truth, got["schema"], got["data"], got["raw"],
                                                  (result or {}).get("solution"), {}))
    return finish(rec, out, client, result)


def live_truth(case, truth, out: Path, downloaded: dict[str, Path]):
    """For live feeds: if the run downloaded the official file, the judge uses a truth recomputed from that file.
    Returns (truth to judge with, note)."""
    live = case.get("live")
    if not live:
        return truth, "static truth"
    import truth as T
    for url, path in downloaded.items():
        if live["official_url_contains"] in str(url) and Path(path).exists():
            snap = dict(truth)
            snap[case["id"]] = getattr(T, live["truth_from_file"])(Path(path), **live.get("kwargs", {}))
            atomic_write(out / "truth_snapshot.json", json.dumps(snap[case["id"]], ensure_ascii=False))
            return snap, f"truth recomputed from the run's own download of the official feed ({path.name})"
    return truth, "run did not use the official feed: judged against the pre-downloaded official snapshot"


def system_downloads(directory: Path) -> dict[str, Path]:
    src = directory / "_sources.json"
    if not src.exists():
        return {}
    out = {}
    for name, rec in json.loads(src.read_text(encoding="utf-8")).items():
        for key in ("resource_url", "url"):
            if rec.get(key):
                out[rec[key]] = directory / name
    return out


def found_files(directory: Path) -> list[str]:
    """Readable data files the run obtained itself (the 'data found' gate, separate from processing them right)."""
    names = []
    for f in sorted(directory.glob("*")) if directory.exists() else []:
        if f.is_file() and not f.name.startswith("_"):
            try:
                load_tables([f]); names.append(f.name)
            except Exception:
                pass
    return names


def finish(rec, out, client=None, result=None):
    rec["cost"] = {"llm_tokens": getattr(client, "tokens_used", None),
                   "solver_calls": (result or {}).get("total_solver_calls", (result or {}).get("solver_calls"))}
    rec["seconds"] = round(time.time() - rec.pop("started"), 1)
    atomic_write(out / "record.json", json.dumps(rec, ensure_ascii=False, indent=1, default=str))
    v = rec.get("verdict") or {}
    print(f"== {rec['arm']}/{rec['case']}: failed_stage={rec['failed_stage']} found={bool(rec.get('data_found'))} "
          f"data={v.get('data', {}).get('ok')} "
          f"model={v.get('model', {}).get('ok')} delivery={v.get('delivery', {}).get('ok')} ({rec['seconds']} s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--arms", nargs="*", default=["direct", "pipeline", "no_repair", "full", "full_rag"],
                    choices=["direct", *ARMS])
    ap.add_argument("--truth", type=Path, nargs="*", default=[ROOT / "results/acceptance_truth/truth_20261011.json",
                                                              ROOT / "results/acceptance_truth/truth_intl_20261011.json"])
    args = ap.parse_args()
    cases = json.loads((ROOT / "acceptance/cases.json").read_text(encoding="utf-8"))
    truth = {}
    for path in args.truth:                               # Taiwanese and international truths, one dict
        truth.update(json.loads(path.read_text(encoding="utf-8")))
    base = ROOT / "results" / args.tag
    base.mkdir(parents=True, exist_ok=True)
    man = base / "manifest.json"
    if man.exists():
        frozen = json.loads(man.read_text())
        now = manifest(cases, args.truth)
        changed = [k for k in ("code_prompts_knowledge_config_sha256", "acceptance_sha256", "truth_sha256")
                   if frozen.get(k) != now[k]]
        if changed:
            raise SystemExit(f"code/prompts/judge changed since this batch started ({changed}); use a new --tag")
    else:
        atomic_write(man, json.dumps(manifest(cases, args.truth), indent=1))
    for rep in range(args.repeats):                      # interleaved: every arm and case once, then repeat
        for case in cases["cases"]:
            if args.cases and case["id"] not in args.cases:
                continue
            for arm in args.arms:
                out = base / arm / f"{case['id']}_r{rep}"
                if (out / "record.json").exists():
                    continue                             # finished runs are never redone
                print(f"--- {arm} {case['id']} repeat {rep}", flush=True)
                run_one(case, arm, cases["budget"], truth, out)


if __name__ == "__main__":
    main()
