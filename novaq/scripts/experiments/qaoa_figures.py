"""Poster figures + summary table for the solver comparison (reads compare.jsonl and bench.jsonl).

Plain journal style: one panel per problem, colour-blind-safe palette (Okabe-Ito), the poster's typeface
(Nimbus Sans = Helvetica metrics), light horizontal grid only, no shadows or gradients.

    python scripts/experiments/qaoa_figures.py --dir results/experiments/qaoa_cudaq_20261020
"""
import argparse
import collections
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _f in ("NimbusSans-Regular.otf", "NimbusSans-Bold.otf", "NimbusSans-Italic.otf"):
    _p = Path("/usr/share/fonts/opentype/urw-base35") / _f
    if _p.exists():
        font_manager.fontManager.addfont(str(_p))

ROOT = Path(__file__).resolve().parents[2]
CASES = [("ev_chargers_s", "EV chargers, Taipei (MCLP)"), ("youbike_tour", "Bike tour, Taipei (TSP)"),
         ("london_ambulance_s", "Ambulances, London (p-median)"), ("aed_groups", "AED zones, Taipei (groups)"),
         ("nyc_pb_budget", "Budget, New York (knapsack)"),
         ("sg_hawker_groups", "Hawker zones, Singapore (groups)")]
# Okabe-Ito palette
SOLVERS = [("sa", "SA", "#E69F00", "o"), ("sqa", "SQA", "#D55E00", "s"), ("dwave_qpu", "D-Wave QPU", "#009E73", "D"),
           ("cga", "CGA", "#CC79A7", "v"), ("qaoa_p3", "QAOA $p$=3 (CUDA-Q)", "#0072B2", "^")]
plt.rcParams.update({
    "font.family": "Nimbus Sans", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def by_instance(recs):
    by = collections.defaultdict(dict)
    for r in recs:
        by[r["instance"]][r["solver"]] = r
    return {k: v for k, v in by.items() if "exact" in v}


def fig_popt(by, out):
    present = {s for rs in by.values() for s in rs}
    solvers = [s for s in SOLVERS if s[0] in present]
    fig, axes = plt.subplots(1, len(CASES), figsize=(3.25 * len(CASES), 3.3), sharey=True)
    for ax, (case, title) in zip(axes, CASES):
        rows = sorted((rs for rs in by.values() if rs["exact"]["case"] == case), key=lambda rs: rs["exact"]["n_qubits"])
        q = [rs["exact"]["n_qubits"] for rs in rows]
        ax.plot(q, [rs["exact"]["p_random"] for rs in rows], color="#999999", ls="--", lw=1.2, label="random guess")
        for key, label, color, marker in solvers:
            pts = [(rs["exact"]["n_qubits"], max(rs[key]["p_optimal"], 1e-6)) for rs in rows if key in rs]
            if pts:
                ax.plot(*zip(*pts), color=color, marker=marker, ms=5, lw=1.5, label=label)
        ax.set_title(title, loc="left")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 1.6)
        ax.set_xticks(q)
        ax.set_xlabel("qubits")
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("P(optimal) per sample")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(l), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_bench(bench, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for target, label, color in (("nvidia", "GPU, 1$\\times$ L40S", "#0072B2"), ("qpp-cpu", "CPU", "#555555")):
        pts = sorted((b["n_qubits"], b["s_per_evaluation"]) for b in bench if b["target"] == target)
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, marker="o", ms=3.5, lw=1.5)
        ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=color)
    g = {b["n_qubits"]: b["s_per_evaluation"] for b in bench if b["target"] == "nvidia"}
    c = {b["n_qubits"]: b["s_per_evaluation"] for b in bench if b["target"] == "qpp-cpu"}
    both = sorted(set(g) & set(c))
    if both:
        n = both[-1]
        ax.annotate(f"{c[n] / g[n]:.0f}$\\times$", (n, (c[n] * g[n]) ** 0.5), xytext=(8, 0), textcoords="offset points",
                    va="center")
        ax.plot([n, n], [g[n], c[n]], color="#999999", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel("seconds per cost evaluation")
    ax.set_xlim(9, 34.5)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_energy(by, out):
    """Normalised energy gap (lower is better): mean line with a +-1 std ribbon per solver, one panel per problem."""
    present = {s for rs in by.values() for s in rs}
    solvers = [s for s in SOLVERS if s[0] in present]
    fig, axes = plt.subplots(1, len(CASES), figsize=(3.25 * len(CASES), 3.5), sharey=True)
    for ax, (case, title) in zip(axes, CASES):
        rows = sorted((rs for rs in by.values() if rs["exact"]["case"] == case), key=lambda rs: rs["exact"]["n_qubits"])
        q = [rs["exact"]["n_qubits"] for rs in rows]
        ax.axhline(1.0, color="#999999", ls="--", lw=1.1, label="random guess")
        ax.axhline(0.0, color="#222222", lw=0.8)
        for key, label, color, marker in solvers:
            pts = [(rs["exact"]["n_qubits"], rs[key]["gap_mean"], rs[key]["gap_std"]) for rs in rows if key in rs]
            if not pts:
                continue
            x, m, sd = map(list, zip(*pts))
            lo = [max(a - b, 0.0) for a, b in zip(m, sd)]
            hi = [a + b for a, b in zip(m, sd)]
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(x, m, color=color, marker=marker, ms=4.5, lw=1.6, label=label)
        ax.set_title(title, loc="left")
        ax.set_xticks(q)
        ax.set_xlabel("qubits")
        ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0, 0.01, 0.1, 1])
        ax.set_yticklabels(["0", "0.01", "0.1", "1"])
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("energy gap to optimum\n(0 = optimum, 1 = random)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(l), handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig_energy_rep(recs, out, floor=5e-4):
    """Repeat instances (3 per size). Same statistic everywhere: line = mean over instances, band = mean +- 1 s.d.
    (clipped at `floor` on the log axis). Left: all problems pooled into qubit bins; right: one panel per problem.
    Every solver gets a band, drawn with the same style."""
    import re
    import numpy as np
    from matplotlib.gridspec import GridSpec
    by = collections.defaultdict(dict)
    for r in recs:
        by[r["instance"]][r["solver"]] = r
    rows = []
    for inst, rs in by.items():
        m = re.match(r"(.+)_n(\d+)_s(\d+)$", inst)
        if not m or "exact" not in rs:
            continue
        for key, *_ in SOLVERS:
            if key in rs and "gap_mean" in rs[key]:
                rows.append({"case": m.group(1), "points": int(m.group(2)), "seed": int(m.group(3)),
                             "qubits": rs["exact"]["n_qubits"], "solver": key, "gap": rs[key]["gap_mean"]})
    lw = {"qaoa_p3": 2.6}
    ls = {"cga": "--"}

    def draw(ax, xs, groups, key, color, marker, scale=1.0):
        m = np.array([np.mean(g) for g in groups]); sd = np.array([np.std(g) for g in groups])
        lo, hi = np.maximum(m - sd, floor), np.maximum(m + sd, floor)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(xs, np.maximum(m, floor), color=color, marker=marker, ms=5 * scale, lw=lw.get(key, 1.5) * scale,
                ls=ls.get(key, "-"))

    ncol = math.ceil(len(CASES) / 2)                   # small panels: two rows
    fig = plt.figure(figsize=(9.4 + 2.3 * ncol, 4.6))
    gs = GridSpec(2, 2 + ncol, figure=fig, width_ratios=[2.3, 0.08] + [1] * ncol, wspace=0.28, hspace=0.55, top=0.86)
    ax = fig.add_subplot(gs[:, 0])
    bins = [(9, 12), (13, 17), (18, 22), (23, 27)]
    for key, label, color, marker in SOLVERS:
        xs, groups = [], []
        for a, b in bins:
            sel = [r for r in rows if r["solver"] == key and a <= r["qubits"] <= b]
            if len(sel) >= 2:
                xs.append(np.mean([r["qubits"] for r in sel])); groups.append([r["gap"] for r in sel])
        if xs:
            draw(ax, xs, groups, key, color, marker)
            ax.plot([], [], color=color, marker=marker, lw=lw.get(key, 1.5), ls=ls.get(key, "-"), label=label)
    ax.axhline(1.0, color="#999999", ls=":", lw=1.2, label="random guess")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 2.0)
    # readable values instead of powers of ten; the bottom edge holds gaps below the floor (the optimum itself)
    ax.set_yticks([floor, 1e-2, 1e-1, 1.0], ["≈0 (optimum)", "0.01", "0.1", "1 (random)"])
    ax.minorticks_off()
    ax.set_xlabel("qubits")
    ax.set_ylabel("energy gap to optimum (log scale, lower is better)")
    ax.set_title(f"{len(CASES)} agent-built models on real city data, pooled (3 instances per size)", loc="left")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6, which="major")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=6, fontsize=10.5, handlelength=2.0,
               columnspacing=1.4)
    ax.text(0.99, 0.02, "line = mean, band = $\pm$1 s.d. over instances", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.5, color="#555555")
    for k, (case, title) in enumerate(CASES):
        sub = fig.add_subplot(gs[k // ncol, 2 + k % ncol])
        for key, label, color, marker in SOLVERS:
            pts = collections.defaultdict(list)
            for r in rows:
                if r["case"] == case and r["solver"] == key:
                    pts[r["points"]].append(r)
            if not pts:
                continue
            ks = sorted(pts)
            draw(sub, [np.mean([r["qubits"] for r in pts[q]]) for q in ks], [[r["gap"] for r in pts[q]] for q in ks],
                 key, color, marker, scale=0.8)
        sub.axhline(1.0, color="#999999", ls=":", lw=1)
        sub.set_yscale("log")
        sub.set_ylim(floor * 0.8, 2.0)
        sub.set_yticks([floor, 1e-2, 1.0], ["≈0", "0.01", "random"])
        sub.minorticks_off()
        sub.set_title(title, loc="left", fontsize=10.5)
        sub.tick_params(labelsize=9.5)
        if k // ncol or k + ncol >= len(CASES):          # bottom of its column
            sub.set_xlabel("qubits", fontsize=10.5)
    if len(CASES) < 2 * ncol:
        # the free cell: median gap per problem and solver (the pooled plot on the left already shows all instances)
        from matplotlib.colors import LogNorm
        sub = fig.add_subplot(gs[1, 2 + ncol - 1])
        short = {"sa": "SA", "sqa": "SQA", "dwave_qpu": "QPU", "cga": "CGA", "qaoa_p3": "QAOA"}
        rows_lab = {"ev_chargers_s": "EV", "youbike_tour": "Tour", "london_ambulance_s": "Amb.", "aed_groups": "AED",
                    "nyc_pb_budget": "Budget"}
        keys = [k for k, *_ in SOLVERS]
        M = np.array([[np.median([r["gap"] for r in rows if r["case"] == c and r["solver"] == k] or [np.nan])
                       for k in keys] for c, _ in CASES])
        sub.imshow(np.maximum(M, floor), cmap="Blues", norm=LogNorm(vmin=floor, vmax=1.0), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt = "≈0" if v < floor else (f"{v:.3f}" if v < 0.1 else f"{v:.2f}")
                sub.text(j, i, txt, ha="center", va="center", fontsize=8,
                         color="white" if v >= 0.05 else "#222222")
        sub.set_xticks(range(len(keys)), [short[k] for k in keys], fontsize=8.5)
        sub.set_yticks(range(len(CASES)), [rows_lab.get(c, c) for c, _ in CASES], fontsize=9)
        sub.tick_params(length=0)
        for side in ("left", "bottom"):
            sub.spines[side].set_visible(False)
        sub.set_title("Median gap per problem", loc="left", fontsize=10.5)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

def table(by, out):
    rows = []
    for inst, rs in sorted(by.items(), key=lambda kv: (kv[1]["exact"]["case"], kv[1]["exact"]["n_qubits"])):
        e = rs["exact"]
        row = {"instance": inst, "case": e["case"], "points": e["n_points"], "qubits": e["n_qubits"],
               "p_random": e["p_random"]}
        for s in ("sa", "sqa", "dwave_qpu", "cga", "qaoa_p1", "qaoa_p2", "qaoa_p3"):
            if s in rs:
                row[s] = {k: rs[s].get(k) for k in ("found_optimum", "p_optimal", "p_feasible", "wall_s", "solver_s")}
        rows.append(row)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return rows


def latex_table(by, out):
    """P(optimal) per sample for every instance and solver, as LaTeX rows for the poster (\\input)."""
    lab = {"ev_chargers_s": "EV chargers", "youbike_tour": "Bike tour", "nyc_pb_budget": "NY budget",
           "london_ambulance_s": "Ambulance",
           "aed_groups": "AED zoning", "sg_hawker_groups": "SG hawker zones"}

    def num(v):
        return f"{v:.2f}" if v >= 0.1 else f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"

    def sci(v):
        e = math.floor(math.log10(v))
        m = round(v / 10 ** e)
        if m == 10:
            m, e = 1, e + 1
        return f"${m}{{\\times}}10^{{{e}}}$"

    import re
    groups = collections.defaultdict(list)                      # (case, points) -> instances (one per seed)
    for inst, rs in by.items():
        m = re.match(r"(.+)_n(\d+)(?:_s\d+)?$", inst)
        groups[(rs["exact"]["case"], int(m.group(2)))].append(rs)
    lines = []
    for case, _ in CASES:
        for pts in sorted(p for c, p in groups if c == case):
            g = groups[(case, pts)]
            qs = sorted({rs["exact"]["n_qubits"] for rs in g})
            q = str(qs[0]) if len(qs) == 1 else f"{qs[0]}--{qs[-1]}"
            cells = [f"{lab[case]}, {pts} pts", q, sci(sum(rs["exact"]["p_random"] for rs in g) / len(g))]
            for key in ("sa", "sqa", "dwave_qpu", "cga", "qaoa_p3"):
                v = [rs[key]["p_optimal"] for rs in g if key in rs]
                cells.append(num(sum(v) / len(v)) if v else "--")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[1.5mm]")
    out.write_text("\n".join(lines[:-1]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--results", default="compare.jsonl", help="results file inside --dir")
    a = ap.parse_args()
    base = ROOT / a.dir
    (base / "figures").mkdir(exist_ok=True)
    by = by_instance(load(base / a.results))
    if any(k.endswith("_s1") for k in by):
        fig_energy_rep(load(base / a.results), base / "figures/energy_rep.png")
    if any("gap_mean" in r for rs in by.values() for r in rs.values()):
        fig_energy(by, base / "figures/energy_by_problem.png")
    fig_popt(by, base / "figures/popt_by_problem.png")
    bench = load(base / "bench.jsonl")
    if bench:
        fig_bench(bench, base / "figures/gpu_vs_cpu.png")
    latex_table(by, base / "figures" / "table_popt.tex")
    for r in table(by, base / "summary.json"):
        print(f"{r['instance']:<22} q={r['qubits']:<3} rand={r['p_random']:.1e} " + " ".join(
            f"{s}={r[s]['p_optimal']:.4f}{'' if r[s]['found_optimum'] else '(miss)'}"
            for s in ("sa", "sqa", "dwave_qpu", "cga", "qaoa_p3") if s in r))


if __name__ == "__main__":
    main()
