"""QAOA for a QUBO on CUDA-Q (GPU state-vector simulation). Standalone: runs in the CUDA-Q environment
(numpy, scipy, cudaq only), so the main environment never imports cudaq. The QAOABackend in backends.py calls it
as a subprocess:

    $CUDAQ_PYTHON qubo_auto/annealing/qaoa_cudaq.py in.json out.json

in.json  {"n": int, "qubo": [[i, j, c], ...], "offset": float, "params": {...}}
out.json {"samples": [[bits], ...], "counts": [...], "theta": [...], "history": [...], "timing": {...}, ...}

QUBO x^T Q x + offset with x = (1 - z) / 2 becomes the Ising H = sum h_i Z_i + sum J_ij Z_i Z_j + const.
H is rescaled to max |coefficient| = 1 before optimisation, so the angle ranges are comparable across instances.
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np


def qubo_to_ising(n: int, qubo: list, offset: float):
    h, J, const = np.zeros(n), {}, float(offset)
    for i, j, c in qubo:
        i, j, c = int(i), int(j), float(c)
        if i == j:                                   # c x_i = c (1 - z_i) / 2
            const += c / 2
            h[i] -= c / 2
        else:                                        # c x_i x_j = c (1 - z_i - z_j + z_i z_j) / 4
            a, b = min(i, j), max(i, j)
            const += c / 4
            h[a] -= c / 4
            h[b] -= c / 4
            J[(a, b)] = J.get((a, b), 0.0) + c / 4
    J = {k: v for k, v in J.items() if v != 0.0}
    return h, J, const


def energies_of(bits: np.ndarray, n: int, qubo: list, offset: float) -> np.ndarray:
    """QUBO energy of each row of a 0/1 matrix."""
    e = np.full(len(bits), float(offset))
    for i, j, c in qubo:
        e += float(c) * bits[:, int(i)] * bits[:, int(j)]
    return e


def run(spec: dict) -> dict:
    import cudaq
    from scipy.optimize import minimize

    n, qubo, offset = int(spec["n"]), spec["qubo"], float(spec.get("offset", 0.0))
    p = spec.get("params", {})
    layers = int(p.get("layers", 2))
    shots = int(p.get("shots", 10000))
    maxiter = int(p.get("maxiter", 200))
    restarts = int(p.get("restarts", 3))
    cvar = float(p.get("cvar_alpha", 1.0))          # < 1: CVaR objective (mean of the best alpha fraction of shots)
    cvar_shots = int(p.get("cvar_shots", 2000))
    seed = int(p.get("seed", 0))
    target = p.get("target", "nvidia")
    cudaq.set_target(target)
    cudaq.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    h, J, const = qubo_to_ising(n, qubo, offset)
    scale = max([abs(v) for v in h] + [abs(v) for v in J.values()] + [1e-12])
    hs = [float(v / scale) for v in h]
    ei = [int(a) for a, _ in J]
    ej = [int(b) for _, b in J]
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

    history, n_eval, t_eval = [], 0, 0.0

    def cost(theta):
        nonlocal n_eval, t_eval
        t0 = time.time()
        if cvar < 1.0:
            res = cudaq.sample(kernel, n, layers, hs, ei, ej, jv, [float(t) for t in theta], shots_count=cvar_shots)
            bits, cnt = [], []
            for bs, c in res.items():
                bits.append([int(ch) for ch in bs])
                cnt.append(int(c))
            e = energies_of(np.array(bits, float), n, qubo, offset) / scale
            order = np.argsort(e)
            e, w = e[order], np.array(cnt, float)[order]
            k = cvar * w.sum()
            take = np.minimum(w, np.maximum(k - (np.cumsum(w) - w), 0))
            val = float((e * take).sum() / take.sum())
        else:
            val = cudaq.observe(kernel, ham, n, layers, hs, ei, ej, jv, [float(t) for t in theta]).expectation()
        t_eval += time.time() - t0
        n_eval += 1
        history.append(float(val))
        return val

    t_start = time.time()
    best = None
    for r in range(restarts):
        # linear-ramp start (annealing-like schedule), jittered on later restarts
        g0 = np.linspace(0.1, 0.8, layers) * (1 + 0.3 * rng.standard_normal(layers) * (r > 0))
        b0 = np.linspace(0.8, 0.1, layers) * (1 + 0.3 * rng.standard_normal(layers) * (r > 0))
        res = minimize(cost, np.concatenate([g0, b0]), method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
        if best is None or res.fun < best.fun:
            best = res
    t_opt = time.time() - t_start

    t0 = time.time()
    counts = cudaq.sample(kernel, n, layers, hs, ei, ej, jv, [float(t) for t in best.x], shots_count=shots)
    t_sample = time.time() - t0
    samples, occ = [], []
    for bitstr, c in counts.items():                 # character k of the bit string is qubit k
        samples.append([int(ch) for ch in bitstr])   # |1> on qubit k  <=>  z_k = -1  <=>  x_k = 1
        occ.append(int(c))
    return {"samples": samples, "counts": occ, "theta": [float(t) for t in best.x],
            "expectation_scaled": float(best.fun), "ising_scale": float(scale), "ising_const": float(const),
            "history": history, "layers": layers, "shots": shots, "target": target, "n_qubits": n,
            "n_terms": len(hs) + len(jv),
            "timing": {"optimise_s": t_opt, "sample_s": t_sample, "evaluations": n_eval,
                       "s_per_evaluation": t_eval / max(n_eval, 1)}}


if __name__ == "__main__":
    spec = json.loads(open(sys.argv[1]).read())
    out = run(spec)
    with open(sys.argv[2], "w") as f:
        json.dump(out, f)
