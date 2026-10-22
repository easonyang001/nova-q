"""Run the closed-loop workflow (spec §8) on one problem instance.

Usage:
    python scripts/run_loop.py --problem select_k_random --n 30 --k 6 --backend sa --policy rule
    python scripts/run_loop.py --problem knapsack --n 14 --backend sa --policy agent            # multi-agent + RAG
    python scripts/run_loop.py --problem knapsack --n 14 --backend sa --policy agent --mode single --no-rag
    python scripts/run_loop.py --problem select_k --backend fujitsu_da --policy agent --simulate  # stand-in
    python scripts/run_loop.py --problem select_k --backend fujitsu_da --policy agent             # real hardware
                                                                     (needs hardware_budget in config_loop.yaml)
Policies: default | rule | random | optuna | agent
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.annealing.backends import BACKENDS, get_backend
from qubo_auto.annealing.baselines import make_policy
from qubo_auto.annealing.examples import EXAMPLES, IR_EXAMPLES
from qubo_auto.annealing.guardrail import HardwareBudget
from qubo_auto.annealing.loop import run_loop


def make_problem(args):
    if args.problem in IR_EXAMPLES:
        return None, IR_EXAMPLES[args.problem](n=args.n, seed=args.instance_seed)
    kwargs = {}
    if args.problem.startswith("qoblib_"):
        if not args.name:
            raise SystemExit("--name is required for QOBLIB problems (e.g. karate, ms_03_050_002)")
        return EXAMPLES[args.problem](args.name), None
    if args.problem in ("select_k_random", "knapsack"):
        kwargs = {"n": args.n, "seed": args.instance_seed}
        if args.problem == "select_k_random":
            kwargs.update(k=args.k, conflict_density=args.density)
        if args.weight is not None:
            kwargs["weight"] = args.weight
    return EXAMPLES[args.problem](**kwargs), None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", choices=sorted(EXAMPLES) + sorted(IR_EXAMPLES), default="select_k_random")
    ap.add_argument("--name", default=None, help="QOBLIB instance name for qoblib_* problems")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--density", type=float, default=0.15)
    ap.add_argument("--instance-seed", type=int, default=0)
    ap.add_argument("--weight", type=float, default=None, help="initial penalty weight (formula default if unset)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="sa")
    ap.add_argument("--policy", default="penalty_tpe", choices=["default", "rule", "random", "optuna", "agent",
                    "penalty_tpe", "penalty_grid", "penalty_cem", "penalty_sequential"])
    ap.add_argument("--compiler", choices=["native", "pyqubo"], default=None)
    ap.add_argument("--checkpoint", type=Path, help="save after every round")
    ap.add_argument("--resume", action="store_true", help="resume an IR penalty-search checkpoint")
    ap.add_argument("--mode", choices=["multi", "single"], default=None, help="agent mode (overrides config)")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--fixed-encoding", action="store_true", help="IR problems: keep default encodings (no Encoding Agent)")
    ap.add_argument("--simulate", action="store_true", help="run real backends on their local stand-in")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config_loop.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.compiler:
        cfg.setdefault("penalty_search", {})["compiler"] = args.compiler
    if args.mode:
        cfg["agents"]["mode"] = args.mode
    if args.no_rag:
        cfg["rag"]["enabled"] = False
    if args.fixed_encoding:
        cfg["agents"]["encoding_agent"] = False

    problem, ir = make_problem(args)
    if ir is not None:
        from qubo_auto.annealing.encodings import compile_problem
        problem = compile_problem(ir)
    backend = get_backend(args.backend)
    rag = None
    if cfg["rag"]["enabled"]:
        from qubo_auto.annealing.rag import KnowledgeBase
        rag = KnowledgeBase.from_config(cfg["rag"], ROOT)
    policy = make_policy(args.policy, seed=args.seed, cfg=cfg, root=ROOT, rag=rag)
    hb = cfg["hardware_budget"]
    budget = HardwareBudget(ROOT / hb["ledger"], {k: v for k, v in hb.items() if k != "ledger"})

    print(f"Problem : {problem.instance_id}  n={problem.n_variables}  reference={problem.reference_objective}")
    print(f"Backend : {backend.name}{' (simulated)' if args.simulate else ''}   Policy: {policy.name}"
          f"{' / ' + cfg['agents']['mode'] if args.policy == 'agent' else ''}   RAG: {'on' if rag else 'off'}\n")
    res = run_loop(None if ir else problem, backend, policy, cfg, simulate=args.simulate, budget=budget, rag=rag,
                   seed=args.seed, root=ROOT, ir=ir, checkpoint=args.checkpoint, resume=args.resume)
    print(f"\nStop reason : {res.stop_reason}   solver calls: {res.solver_calls}   LLM tokens: {res.llm_tokens}"
          f"   wall: {res.wall_time_s:.1f}s")
    print(f"Best        : objective={res.best_objective} (reference {res.reference_objective}, gap {res.best_gap}) "
          f"feasibility={res.best_feasibility_rate} at round {res.best_round}")


if __name__ == "__main__":
    main()
