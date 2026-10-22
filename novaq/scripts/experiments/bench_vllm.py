"""Throughput benchmark for the local vLLM server using real agent prompts from transcripts.

    python scripts/experiments/bench_vllm.py --requests 64 --concurrency 32
"""
from __future__ import annotations
import argparse
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="comma-separated for several replicas")
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / "transcripts" / "loop_*" / "diagnoser_*.json")))
    files += sorted(glob.glob(str(ROOT / "transcripts" / "loop_*" / "refiner_*.json")))
    prompts = [json.loads(Path(f).read_text())["prompt"] for f in files[: args.requests]]
    if len(prompts) < args.requests:
        prompts = (prompts * (args.requests // max(len(prompts), 1) + 1))[: args.requests]
    clients = [OpenAI(base_url=u.strip(), api_key="EMPTY") for u in args.base_url.split(",")]

    def one(item: tuple[int, str]) -> tuple[int, int, float]:
        i, prompt = item
        client = clients[i % len(clients)]                  # round-robin across replicas
        t = time.time()
        r = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": prompt}],
                                           temperature=0, max_tokens=args.max_tokens,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        return r.usage.prompt_tokens, r.usage.completion_tokens, time.time() - t

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        out = list(ex.map(one, enumerate(prompts)))
    wall = time.time() - t0
    gen = sum(o[1] for o in out)
    print(json.dumps({"requests": len(out), "concurrency": args.concurrency, "wall_s": round(wall, 1),
                      "prompt_tokens": sum(o[0] for o in out), "completion_tokens": gen,
                      "gen_tok_per_s": round(gen / wall, 1),
                      "mean_latency_s": round(sum(o[2] for o in out) / len(out), 1)}))


if __name__ == "__main__":
    main()
