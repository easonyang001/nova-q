"""QAOA p=1 energy landscape <H>(gamma, beta) of an agent-built real-data instance, computed on all GPUs with the
nvidia-mqpu target (one virtual QPU per GPU), plus the COBYLA path from the linear-ramp start.

    compute (CUDA-Q env):  CUDA_VISIBLE_DEVICES=0,1 python -I qaoa_landscape.py compute --instance X.json --out L.json
    plot    (main env):    python qaoa_landscape.py plot --data L.json --out landscape.pdf
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def compute(a):
    import cudaq
    from scipy.optimize import minimize
    sys.path.insert(0, str(HERE))
    from qaoa_multigpu import build
    inst = json.loads(Path(a.instance).read_text())
    n, qubo, off = inst["n_qubits"], inst["qubo"], inst["offset"]
    cudaq.set_target("nvidia", option="mqpu")
    gpus = cudaq.get_target().num_qpus()
    kernel, ham, (hs, ei, ej, jv) = build(cudaq, n, qubo, off)
    gam = np.linspace(a.gmin, a.gmax, a.grid)
    bet = np.linspace(0.0, np.pi / 2, a.grid)
    t0 = time.time()
    futs = [cudaq.observe_async(kernel, ham, n, 1, hs, ei, ej, jv, [float(g), float(b)], qpu_id=k % gpus)
            for k, (g, b) in enumerate((g, b) for b in bet for g in gam)]
    vals = np.array([f.get().expectation() for f in futs]).reshape(len(bet), len(gam))
    grid_s = time.time() - t0
    path = []

    def cost(th):
        v = cudaq.observe(kernel, ham, n, 1, hs, ei, ej, jv, [float(t) for t in th]).expectation()
        path.append([float(th[0]), float(th[1]), float(v)])
        return v

    minimize(cost, np.array([0.1, 0.8]), method="COBYLA", options={"maxiter": 120, "rhobeg": 0.2})
    out = {"instance": Path(a.instance).stem, "n_qubits": n, "gpus": gpus, "gamma": gam.tolist(), "beta": bet.tolist(),
           "energy": vals.tolist(), "grid_seconds": grid_s, "evaluations": int(vals.size), "path": path}
    Path(a.out).write_text(json.dumps(out))
    print(f"{n} qubits, {vals.size} evaluations on {gpus} GPUs in {grid_s:.1f}s; path {len(path)} steps")


def plot(a):
    sys.path.insert(0, str(HERE))
    import qaoa_figures  # noqa: F401  (fonts and rcParams)
    import matplotlib.pyplot as plt
    d = json.loads(Path(a.data).read_text())
    E = np.array(d["energy"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    import matplotlib.patheffects as pe
    im = ax.imshow(E, origin="lower", aspect="auto", cmap="viridis", interpolation="bicubic",
                   extent=[d["gamma"][0], d["gamma"][-1], d["beta"][0], d["beta"][-1]])
    ax.contour(d["gamma"], d["beta"], E, levels=12, colors="white", linewidths=0.4, alpha=0.6)
    p = np.array(d["path"])
    halo = [pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()]
    ax.plot(p[:, 0], p[:, 1], color="#ff5722", lw=2.2, marker=".", ms=4, label="COBYLA path", path_effects=halo)
    ax.plot(*p[0, :2], "o", color="white", mec="#222222", mew=1.5, ms=10, label="start", zorder=5)
    ax.plot(*p[-1, :2], "*", color="#ff5722", mec="white", mew=1.2, ms=18, label="minimum found", zorder=5)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_xlim(d["gamma"][0], d["gamma"][-1])
    ax.set_ylim(d["beta"][0], d["beta"][-1])
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\langle H\rangle$ (rescaled)")
    ax.legend(loc="lower right", fontsize=12, frameon=True, facecolor="white", edgecolor="#888888", framealpha=0.95,
              borderpad=0.6, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=400)                   # the energy map is a raster part: print quality
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "plot"])
    ap.add_argument("--instance")
    ap.add_argument("--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--gmax", type=float, default=1.0)
    ap.add_argument("--gmin", type=float, default=0.0)
    a = ap.parse_args()
    compute(a) if a.mode == "compute" else plot(a)


if __name__ == "__main__":
    main()
