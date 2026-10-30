"""How much do the judge's two thresholds decide? Re-score saved runs under other values, without running anything.

The judge (acceptance/judge.py) passes a model when every random valid decision is feasible, every invalid one is
rejected and the Spearman rank agreement is >= RANK_THRESHOLD (0.9); it passes a delivery when the plan meets the
original request and trails the raw-data reference by at most QUALITY_TOLERANCE (10%). Both measured values are in
each record.json, so the pass count can be recomputed for any pair of thresholds. Data checks are taken as recorded.

    python scripts/rescore_thresholds.py results/e2e_test_g0_20261022 results/e2e_test_g1_20261022 ...
    python scripts/rescore_thresholds.py                      # every saved run
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANKS = [0.8, 0.9, 0.95, 0.99]
TOLERANCES = [0.0, 0.01, 0.05, 0.10, 0.20]


def fraction(text) -> bool | None:
    try:
        a, b = str(text).split("/")
        return int(a) == int(b)
    except (ValueError, AttributeError):
        return None


def model_ok(model: dict, rank_threshold: float) -> bool:
    c = model.get("checks", {})
    valid, invalid, rank = fraction(c.get("valid_decisions_feasible")), fraction(c.get("invalid_decisions_rejected")), c.get("rank_agreement")
    if valid is None or invalid is None or rank is None:
        return bool(model.get("ok"))                  # stage did not get that far: keep the recorded verdict
    return valid and invalid and rank >= rank_threshold


def delivery_ok(delivery: dict, tolerance: float) -> bool:
    c = delivery.get("checks", {})
    gap = c.get("gap_vs_reference")
    if gap is None or "meets_original_requirements" not in c:
        return bool(delivery.get("ok"))
    return bool(c["meets_original_requirements"]) and gap <= tolerance + 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=[str(ROOT / "results")])
    ap.add_argument("--out")
    a = ap.parse_args()
    recs = []
    for d in a.dirs:
        for f in sorted(Path(d).rglob("record.json")):
            r = json.loads(f.read_text())
            v = r.get("verdict")
            if isinstance(v, dict) and all(isinstance(v.get(k), dict) for k in ("data", "model", "delivery")):
                recs.append(v)
    n = len(recs)
    recorded = sum(all(v[k].get("ok") for k in ("data", "model", "delivery")) for v in recs)
    lines = ["# Judge thresholds: re-scored pass counts", "",
             f"{n} judged runs from {', '.join(Path(d).name for d in a.dirs)}; recorded passes {recorded} "
             f"(rank >= 0.9, tolerance 10%). Rows: delivery tolerance; columns: rank-agreement threshold.", "",
             "| Tolerance | " + " | ".join(f"rank >= {r}" for r in RANKS) + " |",
             "|---|" + "--:|" * len(RANKS)]
    for tol in TOLERANCES:
        cells = []
        for rank in RANKS:
            p = sum(bool(v["data"].get("ok")) and model_ok(v["model"], rank) and delivery_ok(v["delivery"], tol) for v in recs)
            cells.append(f"{p}/{n}")
        label = "exact optimum" if tol == 0 else f"{tol:.0%}"
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    gaps = sorted(v["delivery"]["checks"]["gap_vs_reference"] for v in recs
                  if isinstance(v["delivery"].get("checks", {}).get("gap_vs_reference"), (int, float)))
    ranks = sorted(v["model"]["checks"]["rank_agreement"] for v in recs
                   if isinstance(v["model"].get("checks", {}).get("rank_agreement"), (int, float)))
    lines += ["", f"Delivery gaps recorded: {len(gaps)}; above 0: {sum(g > 1e-12 for g in gaps)}, above 1%: "
              f"{sum(g > 0.01 for g in gaps)}, above 5%: {sum(g > 0.05 for g in gaps)}, largest {max(gaps) if gaps else '-'}.",
              f"Rank agreements recorded: {len(ranks)}; below 0.95: {sum(r < 0.95 for r in ranks)}, below 0.9: "
              f"{sum(r < 0.9 for r in ranks)}, smallest {min(ranks) if ranks else '-'}.", ""]
    text = "\n".join(lines)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
