"""Resume a saved penalty search without repeating completed solver rounds.

Example: python scripts/resume_penalty.py results/ui/SESSION/checkpoints/FILE.json --max-calls 20
The budget is the TOTAL number of calls, including those already saved.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from qubo_auto.annealing.backends import get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.encodings import ProblemIR
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import run_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-calls", type=int, help="new total solver-call budget")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    saved = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    context = saved["context"]
    cfg = context["config"]
    if args.max_calls is not None:
        cfg["loop"]["max_solver_calls"] = args.max_calls
    backend = get_backend(context["backend"])
    policy = make_policy(context["policy"], seed=context["policy_seed"])
    if not getattr(policy, "resumable", False) or context["ir"] is None:
        parser.error("this checkpoint needs an IR and a penalty-search policy")
    budget = None
    if context["real"]:
        hb = cfg["hardware_budget"]
        budget = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})
    result = run_loop(None, backend, policy, cfg, ir=ProblemIR.model_validate(context["ir"]),
        seed=context["seed"], encodings=context["encodings"], simulate=not context["real"],
        checkpoint=args.checkpoint, resume=True, root=ROOT, budget=budget)
    print(f"{result.stop_reason}; total calls={result.solver_calls}; best objective={result.best_objective}")


if __name__ == "__main__":
    main()
