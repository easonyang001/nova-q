"""Multi-GPU QAOA on CUDA-Q.

mqpu: the nvidia-mqpu target treats each GPU as one virtual QPU. The same batch of QAOA cost evaluations (one
      agent-built instance, many parameter sets, as in restarts / parameter sweeps) runs on 1 GPU in sequence and
      on all GPUs with observe_async; the result is throughput (evaluations per second).
mgpu: the nvidia-mgpu target splits ONE state vector across the GPUs of an MPI job, so the qubit limit grows by one
      per doubling of GPUs. Launch with mpirun; rank 0 writes the timing.

    CUDA_VISIBLE_DEVICES=0,1 python -I qaoa_multigpu.py mqpu --instance .../ev_chargers_s_n7.json --out mqpu.jsonl
    CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 python -I qaoa_multigpu.py mgpu --qubits 30 31 32 33 --out mgpu.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qubo_auto" / "annealing"))
from qaoa_cudaq import qubo_to_ising  # noqa: E402


def build(cudaq, n, qubo, offset):
    h, J, _ = qubo_to_ising(n, qubo, offset)
    scale = max([abs(v) for v in h] + [abs(v) for v in J.values()] + [1e-12])
    hs = [float(v / scale) for v in h]
    ei, ej = [int(a) for a, _ in J], [int(b) for _, b in J]
    jv = [float(v / scale) for v in J.values()]

    @cudaq.kernel
    def kernel(nq: int, nl: int, hz: list[float], ia: list[int], ib: list[int], jz: list[float], theta: list[float]):
        q = cudaq.qvector(nq)
        h(q)
        for l in range(nl):
            g = theta[l]
            b = theta[nl + l]
            for k in range(len(ia)):
                x.ctrl(q[ia[k]], q[ib[k]])
                rz(2.0 * g * jz[k], q[ib[k]])
                x.ctrl(q[ia[k]], q[ib[k]])
            for i in range(nq):
                rz(2.0 * g * hz[i], q[i])
            for i in range(nq):
                rx(2.0 * b, q[i])

    ham = 0.0 * cudaq.spin.i(0)
    for i, v in enumerate(hs):
        if v != 0.0:
            ham += v * cudaq.spin.z(i)
    for a, b, v in zip(ei, ej, jv):
        ham += v * cudaq.spin.z(a) * cudaq.spin.z(b)
    return kernel, ham, (hs, ei, ej, jv)


def random_qubo(n, density=0.5, seed=0):
    rng = np.random.default_rng(seed)
    q = [[i, i, float(rng.normal())] for i in range(n)]
    q += [[i, j, float(rng.normal())] for i in range(n) for j in range(i + 1, n) if rng.random() < density]
    return q


def mqpu(a):
    import cudaq
    inst = json.loads(Path(a.instance).read_text())
    n, qubo, off = inst["n_qubits"], inst["qubo"], inst["offset"]
    cudaq.set_target("nvidia", option="mqpu")
    gpus = cudaq.get_target().num_qpus()
    kernel, ham, (hs, ei, ej, jv) = build(cudaq, n, qubo, off)
    rng = np.random.default_rng(0)
    thetas = [list(map(float, rng.uniform(0, 1, 2 * a.layers))) for _ in range(a.evals)]
    cudaq.observe(kernel, ham, n, a.layers, hs, ei, ej, jv, thetas[0]).expectation()        # warm-up / JIT
    recs = []
    for use in sorted({1, gpus}):
        t0 = time.time()
        futs = [cudaq.observe_async(kernel, ham, n, a.layers, hs, ei, ej, jv, th, qpu_id=k % use)
                for k, th in enumerate(thetas)]
        vals = [f.get().expectation() for f in futs]
        dt = time.time() - t0
        recs.append({"mode": "mqpu", "instance": Path(a.instance).stem, "n_qubits": n, "layers": a.layers,
                     "gpus": use, "evaluations": a.evals, "wall_s": dt, "evals_per_s": a.evals / dt,
                     "mean_value": float(np.mean(vals))})
        print(recs[-1], flush=True)
    with open(a.out, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def mgpu(a):
    import cudaq
    cudaq.mpi.initialize()
    rank, size = cudaq.mpi.rank(), cudaq.mpi.num_ranks()
    cudaq.set_target("nvidia", option="mgpu")
    for n in a.qubits:
        kernel, ham, (hs, ei, ej, jv) = build(cudaq, n, random_qubo(n), 0.0)
        th = [0.3] * a.layers + [0.6] * a.layers
        try:
            cudaq.observe(kernel, ham, n, a.layers, hs, ei, ej, jv, th).expectation()      # warm-up
            t0 = time.time()
            for _ in range(a.evals):
                cudaq.observe(kernel, ham, n, a.layers, hs, ei, ej, jv, th).expectation()
            dt = (time.time() - t0) / a.evals
            rec = {"mode": "mgpu", "n_qubits": n, "gpus": size, "layers": a.layers, "s_per_evaluation": dt, "ok": True}
        except Exception as e:                       # out of memory marks the limit for this GPU count
            rec = {"mode": "mgpu", "n_qubits": n, "gpus": size, "layers": a.layers, "ok": False,
                   "error": f"{type(e).__name__}: {str(e)[:200]}"}
        if rank == 0:
            print(rec, flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if not rec["ok"]:
            break
    cudaq.mpi.finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mqpu", "mgpu"])
    ap.add_argument("--instance")
    ap.add_argument("--qubits", type=int, nargs="*", default=[29, 30, 31, 32, 33])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--evals", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mqpu(a) if a.mode == "mqpu" else mgpu(a)


if __name__ == "__main__":
    main()
