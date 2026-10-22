"""Poster map of one real end-to-end delivery: the agent's EV-charger answer for all 53 villages of Da'an District.

Villages, centroids and population come from the acceptance truth (official census + boundary data); the polygons are
matched by village code (truth 'cptvid' = boundary 'NEW'), so every village is drawn. The chosen sites come from the
run's delivered solution (workflow_state.json) and the agent model's variable names.

    python scripts/experiments/delivery_map.py --run results/e2e_smoke16_20261012/full_rag/ev_chargers_s_r0 \
        --boundaries results/e2e_cfgBt_r1_20261013c/full_rag/ev_chargers_s_r0/data_files/台北市村里界.geojson \
        --truth results/acceptance_truth/truth_20261011.json:ev_chargers_s --out poster/gtc2027/figures/delivery_map.pdf
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import qaoa_figures  # noqa: E402,F401  (fonts and rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from qubo_auto.annealing.modeler import load_module  # noqa: E402

BLUE, COVER, OUT, EDGE = "#0072B2", "#c6dcef", "#ececec", "#4d4d4d"


def hav(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [p[0] for p in polys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--boundaries", required=True)
    ap.add_argument("--truth", required=True, help="truth JSON and case id, 'path:case'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=1000)
    a = ap.parse_args()

    tp, case = a.truth.rsplit(":", 1)
    vill = json.loads((ROOT / tp).read_text())[case]["villages"]
    run = ROOT / a.run
    ns = load_module((run / "artifacts/model.py").read_text())
    data = json.loads((run / "artifacts/data.json").read_text())
    data = data.get("data", data)
    names = ns["build_ir"](data)["variable_names"]
    x = json.loads((run / "artifacts/workflow_state.json").read_text())["solution"]
    chosen = {n[2:] for n, v in zip(names, x) if v and n.startswith("x_")}
    sel = [i for i, v in enumerate(vill) if v["village"] in chosen]
    cov = {i for i, v in enumerate(vill)
           if any(hav(v["lat"], v["lon"], vill[s]["lat"], vill[s]["lon"]) <= a.radius for s in sel)}
    pop_all = sum(v["pop"] for v in vill)
    pop_cov = sum(vill[i]["pop"] for i in cov)

    by_code = {str(v["cptvid"]): i for i, v in enumerate(vill)}
    geo = json.loads(Path(ROOT / a.boundaries).read_text())
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    drawn, edges = set(), Counter()
    key = lambda p: (round(p[0], 6), round(p[1], 6))
    for f in geo["features"]:
        i = by_code.get(str(f["properties"].get("NEW")))
        if i is None:
            continue
        drawn.add(i)
        for ring in rings(f["geometry"]):
            xs, ys = zip(*ring)
            ax.fill(xs, ys, facecolor=COVER if i in cov else OUT, edgecolor="white", lw=0.8, zorder=1)
            for u, v in zip(ring, ring[1:]):
                edges[tuple(sorted((key(u), key(v))))] += 1
    for (u, v), n in edges.items():                     # district outline: edges used by one village only
        if n == 1:
            ax.plot([u[0], v[0]], [u[1], v[1]], color=EDGE, lw=1.4, zorder=2, solid_capstyle="round")

    big = max(v["pop"] for v in vill)
    rest = [i for i in range(len(vill)) if i not in sel]
    ax.scatter([v["lon"] for v in vill], [v["lat"] for v in vill], s=[8 + 40 * v["pop"] / big for v in vill],
               color="#333333", edgecolor="white", lw=0.4, zorder=3)          # demand, size = residents
    lat0 = sum(v["lat"] for v in vill) / len(vill)
    dla, dlo = a.radius / 111320, a.radius / (111320 * math.cos(math.radians(lat0)))
    t = [2 * math.pi * k / 180 for k in range(181)]
    for s in sel:
        lo, la = vill[s]["lon"], vill[s]["lat"]
        ax.plot([lo + dlo * math.cos(u) for u in t], [la + dla * math.sin(u) for u in t], color=BLUE, lw=1.6,
                ls=(0, (5, 2)), zorder=4)
        ax.scatter([lo], [la], s=190, marker="s", color=BLUE, edgecolor="white", lw=1.8, zorder=5)

    lons = [v["lon"] for v in vill]; lats = [v["lat"] for v in vill]
    x0, y0 = min(lons) - 0.001, min(lats) - 0.004               # 1 km scale bar
    ax.plot([x0, x0 + dlo], [y0, y0], color="#222222", lw=2.5, solid_capstyle="butt", zorder=6)
    ax.text(x0 + dlo / 2, y0 + 0.0008, "1 km", ha="center", va="bottom", fontsize=15, zorder=6)

    handles = [
        Line2D([], [], ls="", marker="o", color="#333333", markeredgecolor="white", ms=7,
               label="demand (size = residents)"),
        Line2D([], [], ls="", marker="s", color=BLUE, markeredgecolor="white", ms=11, label=f"chosen site ({len(sel)})"),
        Line2D([], [], color=BLUE, lw=1.8, ls=(0, (5, 2)), label="1 km service radius"),
        Patch(facecolor=COVER, edgecolor="#9a9a9a", lw=0.5, label=f"demand covered\n({pop_cov / pop_all:.0%} of residents)"),
        Patch(facecolor=OUT, edgecolor="#9a9a9a", lw=0.5, label="demand not covered"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=16, frameon=False,
              handletextpad=0.6, labelspacing=1.0, borderaxespad=0.0)
    ax.set_aspect(1 / math.cos(math.radians(lat0)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    out = ROOT / a.out
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"{len(vill)} villages ({len(drawn)} drawn), chosen {sorted(chosen)}, covered {len(cov)} villages, "
          f"{pop_cov} of {pop_all} residents ({pop_cov / pop_all:.1%})")


if __name__ == "__main__":
    main()
