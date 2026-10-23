"""Poster timeline of one real end-to-end run: stage durations from the run's record.json, the data stage split at
the end of its last search call (transcript timestamps), the gates drawn where they act and each step's output.

    python scripts/experiments/run_timeline.py --run results/e2e_smoke16_20261012/full_rag/ev_chargers_s_r0 \
        --out poster/gtc2027/figures/run_timeline.pdf
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import qaoa_figures  # noqa: E402,F401  (fonts and rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

AGENT, GATE, SOLVE, MUTED = "#1F4E79", "#C2410C", "#2e7d32", "#555555"


def call_ends(run: Path) -> dict[str, float]:
    """Last end time (s, unix) of each LLM step in the run's transcripts."""
    out = {}
    for p in (run / "transcripts" / "e2e").glob("*.json"):
        d = json.loads(p.read_text())
        step = p.name.rsplit("_", 2)[0]
        out[step] = max(out.get(step, 0), d["timestamp"])
    return out


def segments(run: Path):
    rec = json.loads((run / "record.json").read_text())
    st = {k: v["seconds"] / 60 for k, v in rec["stages"].items()}
    t_clar = st["clarify"]
    t_model = t_clar + st["model"]
    t_check = t_model + st["checker"]
    t_data = t_check + st["data"]
    t_end = t_data + st["solve"]
    ends = call_ends(run)
    # align transcript clock to the record: the requirement checker's last call ends its stage
    off = t_check - ends["req_checker"] / 60
    t_search = min(max(ends["scout_rank"] / 60 + off, t_check), t_data)
    tries = sum(1 for p in (run / "transcripts" / "e2e").glob("adapter*.json")) or 1
    rc = next((json.loads(l) for l in (run / "artifacts" / "workflow.jsonl").read_text().splitlines()
               if json.loads(l).get("stage") == "requirement_check"), {}).get("report", {})
    v = rec.get("verdict", {})
    data_ck = v.get("data", {}).get("checks", {})
    deliv = v.get("delivery", {}).get("checks", {})
    g2 = (f"{rc.get('valid_feasible_in_model', '?')}/{rc.get('valid', '?')} kept\n"
          f"{rc.get('invalid_rejected', '?')}/{rc.get('invalid', '?')} rejected") if rc else "checked"
    # notes from the run's own record, whatever the task
    srcs = (rec.get("prep") or {}).get("sources") or []
    from urllib.parse import urlparse
    pubs = sorted({(urlparse(str(s_.get("url") or s_.get("page") or "")).hostname or "").replace("www.", "")
                   for s_ in srcs} - {""})
    found = f"{len(srcs)} official file" + ("s" if len(srcs) != 1 else "") if srcs else "files found"
    n_ent, ent = data_ck.get("instance_entities", "?"), str(data_ck.get("entity_set", "items")).replace("_", " ")
    solver = {"sa": "SA", "cga": "CGA", "qaoa_cudaq": "QAOA, CUDA-Q", "sqa": "SQA"}.get(rec.get("backend", ""), rec.get("backend", ""))
    chosen = deliv.get("decisions", "?")
    return t_end, [
        # (start, end, title, color, output below the bar, gate badges)
        (0, t_clar, "Clarify", AGENT, "spec confirmed", []),
        (t_clar, t_model, "Model", AGENT, "", [1]),
        (t_model, t_check, "Check", GATE, g2, [2]),
        (t_check, t_search, "Search data", AGENT, pubs[0] if pubs else found, []),
        (t_search, t_data, "Prepare data", AGENT, f"{n_ent} {ent}", [1, 3]),
        (t_data, t_end, f"Solve ({solver})", SOLVE, f"{chosen} chosen,\nrules hold", [4]),
    ]


def badge(ax, x, y, n):
    ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.22", fc=GATE, ec="none"), zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    total, segs = segments(ROOT / a.run)
    fig, ax = plt.subplots(figsize=(8.2, 1.95))
    h = 1.3
    out_rows = [-1.0, -1.0]            # alternate rows so neighbouring notes never overlap
    wide = total * 0.085                                    # a title fits inside a bar this long (in minutes)
    step = total * 0.022                                    # gap between gate badges
    for i, (x0, x1, title, color, note, gates) in enumerate(segs):
        d = x1 - x0
        ax.barh(0, max(d, total * 0.006), left=x0, height=h, color=color, edgecolor="white", lw=2)
        mid = x0 + d / 2
        dur = f"{d:.1f} min" if d >= 0.1 else "<0.1 min"
        if d >= wide:
            ax.text(mid, 0.0, f"{title}\n{dur}", ha="center", va="center", color="white", fontsize=10,
                    fontweight="bold", linespacing=1.05)
        else:                                               # narrow stage: title above the bar
            ax.text(mid, h / 2 + 0.08, f"{title}\n{dur}", ha="center", va="bottom", color=color, fontsize=10,
                    fontweight="bold", linespacing=1.05)
        for k, g in enumerate(gates):                       # gate badges on the lower edge of the stage
            bx = x1 - step * 0.6 - step * (len(gates) - 1 - k) if d >= step * len(gates) else x1 + step * (0.6 + k)
            badge(ax, bx, -h / 2, g)
        y = out_rows[i % 2]
        ax.text(mid, y, note, ha="center", va="top", color=MUTED, fontsize=9, linespacing=1.05)
    ax.set_xlim(-0.05, total * 1.035)
    ax.set_ylim(-1.85, 1.6)
    ax.set_yticks([])
    ticks = list(range(0, int(total) + 1, 2))
    ax.set_xticks(ticks + [round(total, 1)])
    ax.set_xticklabels([str(t) for t in ticks] + [f"{total:.1f} min"])
    ax.tick_params(labelsize=10)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", -1.85))
    ax.legend(handles=[Patch(color=AGENT, label="LLM agent"),
                       Patch(color=GATE, label="check"),
                       Patch(color=SOLVE, label="solver")],
              loc="upper left", bbox_to_anchor=(0.0, 1.1), ncol=3, frameon=False, fontsize=9,
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    out = ROOT / a.out
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"total {total:.2f} min", [(s[2].replace(chr(10), ' '), round(s[1] - s[0], 2)) for s in segs])


if __name__ == "__main__":
    main()
