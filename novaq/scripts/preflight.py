"""Go / no-go checks before a main batch. Every check must pass; nothing is submitted to any solver.

    python scripts/preflight.py --tag e2e_main_YYYYMMDD
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acceptance"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
os.environ.setdefault("LLM_MODEL_CHECKER", "local2:gemma-4-31b")
PY = sys.executable
TRUTHS = [ROOT / "results/acceptance_truth/truth_20261011.json", ROOT / "results/acceptance_truth/truth_intl_20261011.json"]


def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
    return ok


def unit_tests():
    r = subprocess.run([PY, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=1200)
    return r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]


def judge_selftests():
    r = subprocess.run([PY, "acceptance/selftest.py"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    lines = [l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL"))]
    return r.returncode == 0, "; ".join(lines)


def cases_have_truth_and_judge():
    from judge import CASES
    cases = json.loads((ROOT / "acceptance/cases.json").read_text())["cases"]
    truth = {}
    for t in TRUTHS:
        truth.update(json.loads(t.read_text()))
    missing = [c["id"] for c in cases if c["id"] not in truth or c["id"] not in CASES]
    return not missing, f"{len(cases)} cases" + (f"; missing truth or judge: {missing}" if missing else "")


def llm_server():
    import urllib.request
    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    with urllib.request.urlopen(base + "/models", timeout=10) as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    want = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
    return want in ids, f"serving {ids}"


def checker_server():
    """The independent checkers run on a second model (LLM_MODEL_CHECKER, default local2:gemma-4-31b)."""
    import urllib.request
    want = os.environ.get("LLM_MODEL_CHECKER", "local2:gemma-4-31b")
    if not want.startswith("local2:"):
        return True, f"checker model {want} (main server)"
    with urllib.request.urlopen("http://localhost:8001/v1/models", timeout=10) as r:
        ids = [m["id"] for m in json.loads(r.read())["data"]]
    return want.split(":", 1)[1] in ids, f"port 8001 serves {ids}"


def cga_reachable():
    u = urlparse(os.environ["CGA_URI"])
    with socket.create_connection((u.hostname, u.port or 80), timeout=10):
        pass
    return True, f"TCP connect to {u.hostname}:{u.port} ok (no job submitted)"


def hardware_budget():
    import yaml
    from qubo_auto.annealing.guardrail import HardwareBudget
    hb = yaml.safe_load((ROOT / "config_loop.yaml").read_text())["hardware_budget"]
    b = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})
    used, lim = b.usage("cga"), b.limits.get("cga", {})
    left = lim.get("max_calls", 0) - used.get("calls", 0)
    return left > 2000, f"CGA calls left {left}"


def disk():
    free = shutil.disk_usage(ROOT).free / 2**30
    return free > 20, f"{free:.0f} GB free"


def nothing_else_running():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    busy = [l[:90] for l in out.splitlines() if any(k in l for k in ("run_e2e_batch.py", "data_sweep_100.py",
            "data_robustness.py", "exp_size_tuning.py", "exp_paper_backends.py")) and "preflight" not in l]
    return not busy, "idle" if not busy else f"still running: {busy}"


def manifest_consistent(tag):
    def fn():
        man = ROOT / "results" / tag / "manifest.json"
        if not man.exists():
            return True, "new batch: manifest will be written at start"
        import run_e2e_batch as B                      # noqa: imported for its manifest()
        frozen = json.loads(man.read_text())
        now = B.manifest(json.loads((ROOT / "acceptance/cases.json").read_text()), TRUTHS)
        changed = [k for k in ("code_prompts_knowledge_config_sha256", "acceptance_sha256", "truth_sha256")
                   if frozen.get(k) != now[k]]
        return not changed, "unchanged" if not changed else f"changed since the batch started: {changed}"
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    results = [check("unit tests", unit_tests), check("judge self-tests", judge_selftests),
               check("every case has a truth and a judge", cases_have_truth_and_judge),
               check("local LLM server", llm_server), check("checker model server", checker_server),
               check("CGA reachable", cga_reachable),
               check("hardware budget", hardware_budget), check("disk space", disk),
               check("no other experiment running", nothing_else_running),
               check("manifest", manifest_consistent(args.tag))]
    print("GO" if all(results) else "NO-GO", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
