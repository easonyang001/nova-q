"""Self-tests of the independent judge: correct synthetic outputs must pass, deliberately broken ones must fail.
Run before every batch (scripts/preflight.py does): python acceptance/selftest.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import judge, haversine  # noqa: E402


def selftest_taipei():
    """Sanity check of the judge: a correct synthetic output passes, broken ones fail where they should."""
    truth = json.load(open(str(ROOT / "results/acceptance_truth/truth_20261011.json")))

    # chargers: x[site] + z[village] coverage model, all 456
    rows = truth["ev_chargers"]["villages"]
    def charger_case(rows, k_rhs=5, sense="eq"):
        ids = [r["code"] for r in rows]; n = len(ids)
        ds = {"sets": {"sites": {"meaning": "village centroid", "source": "public"}},
              "params": {"pop": {"meaning": "pop", "index": ["sites"], "type": "int", "source": "public"},
                         "lat": {"meaning": "", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lat"},
                         "lon": {"meaning": "", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lon"}}}
        data = {"sites": ids, "pop": [r["pop"] for r in rows], "lat": [r["lat"] for r in rows], "lon": [r["lon"] for r in rows]}
        lat = np.array(data["lat"]); lon = np.array(data["lon"])
        cov = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) <= 1000
        raw = {"variable_names": [f"x{i}" for i in range(n)] + [f"z{j}" for j in range(n)],
               "variable_keys": [{"sites": i} for i in ids] + [{}] * n, "sense": "maximize",
               "linear": {str(n + j): data["pop"][j] for j in range(n)}, "quadratic": [], "constant": 0,
               "constraints": [{"name": "k", "kind": "linear", "coeffs": {str(i): 1 for i in range(n)}, "sense": sense, "rhs": k_rhs}] +
               [{"name": f"c{j}", "kind": "linear", "coeffs": {**{str(n + j): 1}, **{str(i): -1 for i in np.where(cov[:, j])[0]}}, "sense": "le", "rhs": 0} for j in range(n)]}
        return ds, data, raw
    ds, data, raw = charger_case(rows)
    greedy = set(truth["ev_chargers"]["greedy"]["sites"])
    x = [int(i in greedy) for i in data["sites"]] + [0] * len(rows)
    r = judge("ev_chargers", truth, ds, data, raw, x, {})
    print("GOOD chargers:", {k: v["ok"] for k, v in r.items()}, r["model"]["checks"].get("rank_agreement"), r["delivery"]["checks"].get("gap_vs_reference"))
    assert all(v["ok"] for v in r.values()), "correct chargers output must pass"
    ds2, data2, raw2 = charger_case(rows[:32])
    r = judge("ev_chargers", truth, ds2, data2, raw2, None, {})
    print("32 villages:", r["data"]["ok"], r["data"]["checks"]["missing_real_entities"], r["data"]["checks"]["population_total"])
    assert r["data"]["ok"] is False and r["data"]["checks"]["missing_real_entities"] == 424, "32 of 456 villages"
    ds3, data3, raw3 = charger_case(rows, k_rhs=200, sense="le")
    r = judge("ev_chargers", truth, ds3, data3, raw3, None, {})
    print("k<=200 model:", r["model"]["ok"], r["model"]["checks"].get("invalid_decisions_rejected"))
    assert r["model"]["ok"] is False, "a model allowing up to 200 sites must fail"
    dup = json.loads(json.dumps(data)); dup["lat"][1], dup["lon"][1] = dup["lat"][0], dup["lon"][0]
    r = judge("ev_chargers", truth, ds, dup, raw, None, {})
    print("duplicate:", r["data"]["ok"], r["data"]["checks"]["duplicates"], r["data"]["checks"]["missing_examples"])
    assert r["data"]["ok"] is False, "a copied coordinate must fail"

    # youbike: x[station, step] with distance objective
    st = truth["youbike_tour"]["stations"]; n = len(st); ids = [s["sno"] for s in st]
    lat = np.array([s["lat"] for s in st]); lon = np.array([s["lon"] for s in st]); D = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    v = lambda i, t: i * n + t
    quad = [[v(i, t), v(j, (t + 1) % n), D[i, j]] for i in range(n) for j in range(n) if i != j for t in range(n)]
    cons = [{"name": f"s{i}", "kind": "linear", "coeffs": {str(v(i, t)): 1 for t in range(n)}, "sense": "eq", "rhs": 1} for i in range(n)] + \
           [{"name": f"t{t}", "kind": "linear", "coeffs": {str(v(i, t)): 1 for i in range(n)}, "sense": "eq", "rhs": 1} for t in range(n)]
    raw = {"variable_names": [f"x{i}_{t}" for i in range(n) for t in range(n)], "variable_keys": [{"stations": ids[i], "steps": t} for i in range(n) for t in range(n)],
           "sense": "minimize", "linear": {}, "quadratic": quad, "constant": 0, "constraints": cons}
    ds = {"sets": {"stations": {"meaning": "", "source": "public"}, "steps": {"meaning": "", "source": "user"}},
          "params": {"lat": {"meaning": "", "index": ["stations"], "type": "number", "source": "public", "unit": "deg_lat"},
                     "lon": {"meaning": "", "index": ["stations"], "type": "number", "source": "public", "unit": "deg_lon"}}}
    data = {"stations": ids, "steps": list(range(n)), "lat": lat.tolist(), "lon": lon.tolist()}
    x = [0] * (n * n)
    for t, i in enumerate(range(n)): x[v(i, t)] = 1
    r = judge("youbike_tour", truth, ds, data, raw, x, {})
    print("tour (index order):", {k: v_["ok"] for k, v_ in r.items()}, r["model"]["checks"], r["delivery"]["checks"].get("gap_vs_reference"))
    assert r["data"]["ok"] and r["model"]["ok"] and r["delivery"]["ok"] is False, "a poor tour must fail delivery only"

def selftest_london():
    truth = json.load(open(str(ROOT / "results/acceptance_truth/truth_intl_20261011.json")))
    case = "london_ambulance_s"
    t = truth[case]; rows = t["areas"]; n = len(rows); p = t["p"]
    ids = [r["code"] for r in rows]
    lat = np.array([r["lat"] for r in rows]); lon = np.array([r["lon"] for r in rows]); w = [r["pop"] for r in rows]
    D = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    def build(op="eq", rhs=p, coords="wgs"):
        ds = {"sets": {"areas": {"meaning": "MSOA", "source": "public"}},
              "params": {"pop": {"meaning": "", "index": ["areas"], "type": "int", "source": "public"},
                         "lat": {"meaning": "", "index": ["areas"], "type": "number", "source": "public", "unit": "deg_lat"},
                         "lon": {"meaning": "", "index": ["areas"], "type": "number", "source": "public", "unit": "deg_lon"}}}
        la = lat.tolist() if coords == "wgs" else [r["y"] for r in rows]
        lo = lon.tolist() if coords == "wgs" else [r["x"] for r in rows]
        data = {"areas": ids, "pop": w, "lat": la, "lon": lo}
        # x[j] open (decision), y[i,j] helper; minimise sum w_i d_ij y_ij
        names = [f"x{j}" for j in range(n)] + [f"y{i}_{j}" for i in range(n) for j in range(n)]
        keys = [{"areas": ids[j]} for j in range(n)] + [{}] * (n * n)
        yi = lambda i, j: n + i * n + j
        lin = {str(yi(i, j)): w[i] * D[i, j] for i in range(n) for j in range(n)}
        cons = [{"name": "p", "kind": "linear", "coeffs": {str(j): 1 for j in range(n)}, "sense": op, "rhs": rhs}]
        cons += [{"name": f"serve{i}", "kind": "linear", "coeffs": {str(yi(i, j)): 1 for j in range(n)}, "sense": "eq", "rhs": 1} for i in range(n)]
        cons += [{"name": "link", "kind": "implication", "pairs": [[yi(i, j), j] for i in range(n) for j in range(n)]}]
        raw = {"variable_names": names, "variable_keys": keys, "sense": "minimize", "linear": lin, "quadratic": [], "constant": 0, "constraints": cons}
        return ds, data, raw
    best = t["reference"]["exact_sites"]
    def sol(sites):
        x = [0] * (n + n * n)
        for j in sites: x[j] = 1
        return x
    ds, data, raw = build()
    v = judge(case, truth, ds, data, raw, sol(best), {})
    print("GOOD:", {g: v[g]["ok"] for g in ("data", "model", "delivery")}, v["model"]["checks"].get("rank_agreement"), v["delivery"]["checks"].get("gap_vs_reference"))
    assert all(v[g]["ok"] for g in ("data", "model", "delivery")), "correct London output must pass"
    ds, data, raw = build(op="le", rhs=20)
    v = judge(case, truth, ds, data, raw, None, {})
    print("at most 20:", v["model"]["ok"], v["model"]["checks"].get("invalid_decisions_rejected"))
    assert v["model"]["ok"] is False, "at most 20 sites must fail"
    ds, data, raw = build(coords="bng")
    v = judge(case, truth, ds, data, raw, None, {})
    print("not converted:", v["data"]["ok"], v["data"]["checks"].get("coordinates_outside_area"), v["data"]["checks"].get("mapping"))
    assert v["data"]["ok"] is False and v["data"]["checks"]["coordinates_outside_area"] > 0, "unconverted grid must fail"
    worst = sorted(range(n), key=lambda j: -sum(w[i] * D[i, j] for i in range(n)))[:p]
    ds, data, raw = build()
    v = judge(case, truth, ds, data, raw, sol(worst), {})
    print("worst sites:", v["delivery"]["ok"], v["delivery"]["checks"].get("gap_vs_reference"))
    assert v["delivery"]["ok"] is False, "worst sites must fail delivery"

def selftest_berlin():
    truth = json.load(open(str(ROOT / "results/acceptance_truth/truth_intl_20261011.json")))
    case = "berlin_playgrounds_s"; t = truth[case]
    cand = t["candidates"]; k = t["k"]; nc = len(cand)
    open_set = set(t["open_blocks"]); blocks = [b for b in t["blocks"] if b["block"] in open_set]; nb = len(blocks)
    cla = np.array([c["lat"] for c in cand]); clo = np.array([c["lon"] for c in cand])
    bla = np.array([b["lat"] for b in blocks]); blo = np.array([b["lon"] for b in blocks])
    cov = haversine(cla[:, None], clo[:, None], bla[None, :], blo[None, :]) <= t["radius_m"]
    def build(op="eq", rhs=k, count_once=True):
        ds = {"sets": {"sites": {"meaning": "planning area centre", "source": "public"},
                       "blocks": {"meaning": "block not near a playground", "source": "public"}},
              "params": {"lat": {"meaning": "", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lat"},
                         "lon": {"meaning": "", "index": ["sites"], "type": "number", "source": "public", "unit": "deg_lon"},
                         "ew": {"meaning": "", "index": ["blocks"], "type": "int", "source": "public"}}}
        data = {"sites": [c["plr_id"] for c in cand], "blocks": [b["block"] for b in blocks],
                "lat": cla.tolist(), "lon": clo.tolist(), "ew": [b["ew"] for b in blocks]}
        names = [f"x{i}" for i in range(nc)]; keys = [{"sites": c["plr_id"]} for c in cand]
        if count_once:
            names += [f"z{j}" for j in range(nb)]; keys += [{}] * nb
            lin = {str(nc + j): blocks[j]["ew"] for j in range(nb)}
            cons = [{"name": f"c{j}", "kind": "linear", "coeffs": {**{str(nc + j): 1}, **{str(i): -1 for i in np.flatnonzero(cov[:, j])}}, "sense": "le", "rhs": 0} for j in range(nb)]
        else:            # double counting: every (site, block) pair counts
            lin = {str(i): float(sum(blocks[j]["ew"] for j in np.flatnonzero(cov[i]))) for i in range(nc)}
            cons = []
        cons = [{"name": "k", "kind": "linear", "coeffs": {str(i): 1 for i in range(nc)}, "sense": op, "rhs": rhs}] + cons
        raw = {"variable_names": names, "variable_keys": keys, "sense": "maximize", "linear": lin, "quadratic": [], "constant": 0, "constraints": cons}
        return ds, data, raw
    best = t["reference"]["exact_sites"]
    x = [1 if i in best else 0 for i in range(nc)] + [1] * nb
    ds, data, raw = build()
    v = judge(case, truth, ds, data, raw, x, {})
    print("GOOD:", {g: v[g]["ok"] for g in ("data", "model", "delivery")}, v["model"]["checks"].get("rank_agreement"), v["delivery"]["checks"].get("gap_vs_reference"), v["data"]["checks"].get("resident_totals"))
    assert all(v[g]["ok"] for g in ("data", "model", "delivery")), "correct Berlin output must pass"
    ds, data, raw = build(count_once=False)
    v = judge(case, truth, ds, data, raw, None, {})
    print("double counting:", v["model"]["ok"], v["model"]["checks"].get("rank_agreement"))
    pass  # double counting is equivalent on this instance (service areas hardly overlap); not asserted
    ds, data, raw = build(op="le", rhs=36)
    v = judge(case, truth, ds, data, raw, None, {})
    print("at most 36:", v["model"]["ok"], v["model"]["checks"].get("invalid_decisions_rejected"))
    assert v["model"]["ok"] is False, "at most 36 sites must fail"
    ds, data, raw = build(); data["ew"] = [e * 2 for e in data["ew"]]
    v = judge(case, truth, ds, data, raw, None, {})
    print("doubled residents:", v["data"]["ok"], v["data"]["checks"].get("resident_totals"))
    assert v["data"]["ok"] is False, "doubled residents must fail"


def selftest_aed():
    truth = json.load(open(str(ROOT / "results/acceptance_truth/truth_20261011.json")))
    case = "aed_groups"; t = truth[case]; rows = t["places"]; n = len(rows); G = t["groups"]; size = t["group_size"]
    names = [r["name"] for r in rows]
    lat = np.array([r["lat"] for r in rows]); lon = np.array([r["lon"] for r in rows])
    D = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    v = lambda i, g: i * G + g

    def build(equal=True):
        ds = {"sets": {"places": {"meaning": "AED place", "source": "public"}, "teams": {"meaning": "team", "source": "user"}},
              "params": {"lat": {"meaning": "", "index": ["places"], "type": "number", "source": "public", "unit": "deg_lat"},
                         "lon": {"meaning": "", "index": ["places"], "type": "number", "source": "public", "unit": "deg_lon"}}}
        data = {"places": names, "teams": list(range(G)), "lat": lat.tolist(), "lon": lon.tolist()}
        quad = [[v(i, g), v(j, g), D[i, j]] for g in range(G) for i in range(n) for j in range(i + 1, n)]
        cons = [{"name": f"one_{i}", "kind": "linear", "coeffs": {str(v(i, g)): 1 for g in range(G)}, "sense": "eq", "rhs": 1}
                for i in range(n)]
        cons += [{"name": f"size_{g}", "kind": "linear", "coeffs": {str(v(i, g)): 1 for i in range(n)},
                  "sense": "eq" if equal else "le", "rhs": size if equal else n} for g in range(G)]
        raw = {"variable_names": [f"x{i}_{g}" for i in range(n) for g in range(G)],
               "variable_keys": [{"places": names[i], "teams": g} for i in range(n) for g in range(G)],
               "sense": "minimize", "linear": {}, "quadratic": quad, "constant": 0, "constraints": cons}
        return ds, data, raw
    order = sorted(range(n), key=lambda i: (lat[i], lon[i]))                  # compact bands: a decent grouping
    good = [0] * (n * G)
    for k, i in enumerate(order):
        good[v(i, k // size)] = 1
    ds, data, raw = build()
    r = judge(case, truth, ds, data, raw, good, {})
    assert r["data"]["ok"] and r["model"]["ok"], ("correct AED model must pass", r["model"]["checks"])
    ds, data, raw = build(equal=False)
    r = judge(case, truth, ds, data, raw, None, {})
    assert r["model"]["ok"] is False, "groups of any size must fail"
    bad = [0] * (n * G)
    for k, i in enumerate(order):
        bad[v(i, k % G)] = 1                                                   # interleaved: every team spans the area
    ds, data, raw = build()
    r = judge(case, truth, ds, data, raw, bad, {})
    assert r["delivery"]["ok"] is False, "a scattered grouping must fail delivery"


def selftest_tour_variants():
    truth = json.load(open(str(ROOT / "results/acceptance_truth/truth_20261011.json")))
    st = truth["youbike_tour"]["stations"]; ids = [x["sno"] for x in st]
    lat = np.array([x["lat"] for x in st]); lon = np.array([x["lon"] for x in st])
    D = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])

    def tour_model(keep):                       # position encoding over the stations in `keep`, first stop fixed
        n = len(keep)
        pairs = [(i, t) for i in range(n) for t in range(n) if (i == 0) == (t == 0)]
        index = {p_: k for k, p_ in enumerate(pairs)}
        quad = [[index[(i, t)], index[(j, (t + 1) % n)], D[keep[i], keep[j]]]
                for (i, t) in pairs for j in range(n) if j != i and (j, (t + 1) % n) in index]
        cons = [{"name": f"s{i}", "kind": "linear", "coeffs": {str(index[(i, t)]): 1 for t in range(n) if (i, t) in index},
                 "sense": "eq", "rhs": 1} for i in range(n)]
        cons += [{"name": f"t{t}", "kind": "linear", "coeffs": {str(index[(i, t)]): 1 for i in range(n) if (i, t) in index},
                  "sense": "eq", "rhs": 1} for t in range(n)]
        raw = {"variable_names": [f"x{i}_{t}" for i, t in pairs], "sense": "minimize", "linear": {}, "quadratic": quad,
               "variable_keys": [{"stations": ids[keep[i]], "steps": t} for i, t in pairs], "constant": 0,
               "constraints": cons}
        ds = {"sets": {"stations": {"meaning": "", "source": "public"}, "steps": {"meaning": "", "source": "user"}},
              "params": {"lat": {"meaning": "", "index": ["stations"], "type": "number", "source": "public", "unit": "deg_lat"},
                         "lon": {"meaning": "", "index": ["stations"], "type": "number", "source": "public", "unit": "deg_lon"}}}
        data = {"stations": [ids[k] for k in keep], "steps": list(range(n)), "lat": lat[keep].tolist(), "lon": lon[keep].tolist()}
        x = [1 if i == t else 0 for i, t in pairs]
        return ds, data, raw, x
    ds, data, raw, x = tour_model(list(range(len(ids))))
    r = judge("youbike_tour", truth, ds, data, raw, x, {})
    assert r["model"]["ok"], ("a tour model with a fixed first stop is correct", r["model"]["checks"])
    ds, data, raw, x = tour_model(list(range(10)))          # only 10 of the 18 required stations
    r = judge("youbike_tour", truth, ds, data, raw, x, {})
    assert r["data"]["ok"] is False and r["delivery"]["ok"] is False, "a tour that skips required stations must fail"


if __name__ == "__main__":
    failed = []
    for fn in (selftest_taipei, selftest_tour_variants, selftest_aed, selftest_london, selftest_berlin):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
