"""Poster map: the agent's real Da'an District data, its reduced 7-point instance and the optimal 3 charger sites.

Li boundaries come from the GeoJSON the pipeline downloaded in that run (polygons that contain one of the 53 li
centroids); the optimum comes from full enumeration of the reduced QUBO.

    python scripts/experiments/qaoa_map.py --dir results/experiments/qaoa_cudaq_20261020
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import qaoa_figures  # noqa: E402,F401  (fonts and rcParams)
from qaoa_compare import enumerate_min  # noqa: E402


def rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return [p[0] for p in polys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--instance", default="ev_chargers_s_n7")
    ap.add_argument("--no-legend", action="store_true", help="legend goes into the poster caption instead")
    a = ap.parse_args()
    base = ROOT / a.dir
    inst = json.loads((base / "instances" / f"{a.instance}.json").read_text())
    run = ROOT / inst["source_run"]
    full = json.loads((run / "artifacts/data.json").read_text())
    full = full.get("data", full)
    small = inst["data"]
    _, states = enumerate_min(inst["n_qubits"], inst["qubo"], inst["offset"])
    names = inst["variable_names"]
    chosen = {names[k][2:] for k in range(inst["n_decision"]) if states[0][k] and names[k].startswith("x_")}

    geo = json.loads(next((run / "data_files").glob("*.geojson")).read_text())
    pts = list(zip(full["lon"], full["lat"]))
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    for f in geo["features"]:
        for ring in rings(f["geometry"]):
            path = MPath(ring)
            mx, my = sum(x for x, _ in ring) / len(ring), sum(y for _, y in ring) / len(ring)
            near = min((mx - x) ** 2 + (my - y) ** 2 for x, y in pts) ** 0.5 < 0.003   # ~300 m
            if near or any(path.contains_point(p) for p in pts):
                xs, ys = zip(*ring)
                ax.fill(xs, ys, facecolor="#f2f2f2", edgecolor="#b5b5b5", lw=0.5, zorder=1)
    pmax = max(full["population"])
    ax.scatter(full["lon"], full["lat"], s=[8 + 30 * p / pmax for p in full["population"]], color="#9a9a9a",
               lw=0, zorder=2, label="53 li (size = residents)")
    r = small["coverage_radius"]
    lat0 = sum(small["lat"]) / len(small["lat"])
    for name, lon, lat in zip(small["li"], small["lon"], small["lat"]):
        if name in chosen:
            dlat, dlon = r / 111_320, r / (111_320 * math.cos(math.radians(lat0)))
            t = [2 * math.pi * k / 120 for k in range(121)]
            ax.plot([lon + dlon * math.cos(u) for u in t], [lat + dlat * math.sin(u) for u in t],
                    color="#0072B2", lw=1.0, zorder=3)
    rep = [(lo, la) for n, lo, la in zip(small["li"], small["lon"], small["lat"]) if n not in chosen]
    ax.scatter(*zip(*rep), s=55, facecolor="white", edgecolor="#222222", lw=1.2, zorder=4,
               label="7 representatives (k-means)")
    sel = [(lo, la) for n, lo, la in zip(small["li"], small["lon"], small["lat"]) if n in chosen]
    ax.scatter(*zip(*sel), s=90, marker="s", color="#0072B2", edgecolor="#222222", lw=1.0, zorder=5,
               label="optimal 3 sites, 1 km radius")
    ax.set_aspect(1 / math.cos(math.radians(lat0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if not a.no_legend:
        ax.legend(loc="lower left", ncol=1, handletextpad=0.3, fontsize=13, borderaxespad=0)
    fig.tight_layout()
    out = base / "figures" / ("map_ev_chargers_nolegend.pdf" if a.no_legend else "map_ev_chargers.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(out, sorted(chosen))


if __name__ == "__main__":
    main()
