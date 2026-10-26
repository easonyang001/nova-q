"""Poster heat map: which verification gate catches which injected fault (E6 fault injection, Gemma checker).

Rows = fault types (D = data, M = model), only harmful injections (judge says the fault changes the answer);
columns = gate 1 (schema checks, L1), gate 2 (requirement checker, L2), gate 3 (data cross-check, L3), any gate.

    python scripts/experiments/fault_gate_figure.py --out poster/gtc2027/figures/fault_gates.pdf
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import qaoa_figures  # noqa: E402,F401  (fonts and rcParams)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SRC = ROOT / "results/experiments/fault_injection_dev_local2_gemma-4-31b_b3.json"
LABEL = {"D1": "D1 entities missing", "D2": "D2 duplicate place", "D3": "D3 swapped lat/lon",
         "D4": "D4 value off by 10x", "D5": "D5 total row mixed in", "D6": "D6 outside the area",
         "M1": "M1 constraint dropped", "M2": "M2 objective flipped", "M3": "M3 pair terms dropped",
         "M4": "M4 count off by one", "M5": "M5 entity blocked"}
GATES = [("L1_data", "G1\nschema"), ("L2_checker", "G2\nchecker"), ("L3_crosscheck", "G3\ncross-\ncheck")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    agg = collections.defaultdict(collections.Counter)
    for r in json.loads(SRC.read_text()):
        if r["fault"].startswith("none") or not r["judge_harmful"]:
            continue
        k = r["fault"][:2]
        agg[k]["n"] += 1
        for key, _ in GATES:
            agg[k][key] += bool(r[key])
        agg[k]["any"] += any(bool(r[key]) for key, _ in GATES)
    rows = sorted(agg)
    M = np.array([[agg[k][key] / agg[k]["n"] for key, _ in GATES] + [agg[k]["any"] / agg[k]["n"]] for k in rows])
    cmap = LinearSegmentedColormap.from_list("gate", ["#ffffff", "#f6c9ad", "#C2410C"])
    scope = {"L1_data": "D", "L2_checker": "M", "L3_crosscheck": "D"}     # which fault family each gate is built for
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    shown = M.copy()
    for i, k in enumerate(rows):
        for j, (key, _) in enumerate(GATES):
            if scope[key] != k[0]:
                shown[i, j] = np.nan                                    # out of scope: drawn separately
    ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    from matplotlib.patches import Rectangle
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(shown[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2f2f2", edgecolor="#d9d9d9",
                                       hatch="///", lw=0))
                ax.text(j, i, f"{v:.0%}" if v else "\u2014", ha="center", va="center", fontsize=12, color="#8a8a8a")
                continue
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=14,
                    color="white" if v > 0.6 else "#222222", fontweight="bold" if j == 3 else "normal")
    ax.set_xticks(range(4), [g for _, g in GATES] + ["any\ngate"], fontsize=13.5, linespacing=1.0)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [LABEL[k] for k in rows], fontsize=14)
    ax.axvline(2.5, color="#222222", lw=1.2)
    ax.axhline(5.5, color="#222222", lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    out = ROOT / a.out
    fig.savefig(out, dpi=400)                     # hatching and the heat map are raster parts: print quality
    fig.savefig(out.with_suffix(".png"), dpi=300)
    tot = {g: (sum(agg[k]["any"] for k in rows if k[0] == g), sum(agg[k]["n"] for k in rows if k[0] == g)) for g in "DM"}
    print(out, tot)


if __name__ == "__main__":
    main()
