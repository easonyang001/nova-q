"""Clarify a real-world problem with the user before any QUBO is built (Clarification Agent).

    python scripts/clarify.py new --backend fujitsu_da --text "我想做一個都市計畫的問題..."
    python scripts/clarify.py new --backend fujitsu_da --file problem.txt
    python scripts/clarify.py answer results/clarify/<id>.json --text "Q1: ...  Q2: ..."
    python scripts/clarify.py show results/clarify/<id>.json
    python scripts/clarify.py confirm results/clarify/<id>.json      # only you should run this

Each step prints the agent's current understanding and its open questions. The session file is what later
steps read; they refuse sessions that are not confirmed.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from qubo_auto.annealing.clarify import BACKEND_LIMITS, Session, confirm, new_session, render, step
from qubo_auto.llm_client import LLMClient


def _text(args) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    if getattr(args, "text", None):
        return args.text
    raise SystemExit("give --text or --file")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_new = sub.add_parser("new")
    p_new.add_argument("--backend", choices=sorted(BACKEND_LIMITS), required=True)
    p_new.add_argument("--text"), p_new.add_argument("--file")
    p_ans = sub.add_parser("answer")
    p_ans.add_argument("session"), p_ans.add_argument("--text"), p_ans.add_argument("--file")
    sub.add_parser("show").add_argument("session")
    sub.add_parser("confirm").add_argument("session")
    args = ap.parse_args()

    model = os.environ.get("LLM_MODEL") or "qwen3.6-35b-a3b"
    if args.cmd == "new":
        s = new_session(_text(args), args.backend)
        path = ROOT / "results" / "clarify" / f"{s.id}.json"
    else:
        path = Path(args.session) if Path(args.session).is_absolute() else Path.cwd() / args.session
        s = Session.load(path)

    if args.cmd in ("new", "answer"):
        client = LLMClient(transcript_dir=ROOT / "transcripts", run_id=f"clarify_{s.id}", max_tokens=400_000, use_cache=False)
        s = step(s, client, model, answers=_text(args) if args.cmd == "answer" else None)
    elif args.cmd == "confirm":
        s = confirm(s)
    s.save(path)
    print(render(s))
    print(f"\nSession: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
