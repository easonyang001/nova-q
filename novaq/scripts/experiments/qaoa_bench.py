"""CUDA-Q QAOA simulation cost vs qubit count: GPU state vector ('nvidia') vs CPU ('qpp-cpu').

Random QUBOs with the density of the real instances (~each variable coupled to ~half of the others, as in the
one-hot / tour models), p = 2, a fixed number of cost evaluations. Run in the CUDA-Q environment:

    CUDA_VISIBLE_DEVICES=0 /home/var/miniconda3/envs/cudaq-env/bin/python -I scripts/experiments/qaoa_bench.py \
        --out results/experiments/qaoa_cudaq_20261020/bench.jsonl --gpu-max 31 --cpu-max 24
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qubo_auto" / "annealing"))
import qaoa_cudaq  # noqa: E402


def random_qubo(n, density=0.5, seed=0):
    rng = np.random.default_rng(seed)
    q = [[i, i, float(rng.normal())] for i in range(n)]
    q += [[i, j, float(rng.normal())] for i in range(n) for j in range(i + 1, n) if rng.random() < density]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-max", type=int, default=31)
    ap.add_argument("--cpu-max", type=int, default=24)
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--start", type=int, default=10)
    a = ap.parse_args()
    for n in range(a.start, a.gpu_max + 1):
        for target, limit in (("nvidia", a.gpu_max), ("qpp-cpu", a.cpu_max)):
            if n > limit:
                continue
            spec = {"n": n, "qubo": random_qubo(n), "offset": 0.0,
                    "params": {"layers": a.layers, "shots": 1000, "maxiter": a.evals, "restarts": 1, "target": target}}
            t0 = time.time()
            r = qaoa_cudaq.run(spec)
            rec = {"n_qubits": n, "target": target, "layers": a.layers, "n_terms": r["n_terms"],
                   "s_per_evaluation": r["timing"]["s_per_evaluation"], "sample_s": r["timing"]["sample_s"],
                   "evaluations": r["timing"]["evaluations"], "wall_s": time.time() - t0}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


if __name__ == "__main__":
    main()
