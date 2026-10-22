"""Joining tables from different sources (docs/data_landscape.md §3.6) and shrinking data to a solvable size (§3.7).

Joins always report what did not match, so a join that silently loses rows (e.g. '台北' vs '臺北', '大安里' vs
'大安區大安里') is visible to the user before they approve the data.
"""
from __future__ import annotations
import difflib
import re

import numpy as np
import pandas as pd

from .text import normalize_text

_ADMIN_SUFFIX = re.compile(r"(市|縣|區|鄉|鎮|村|里|鄰)$")


def admin_key(name, strip_parents: bool = False) -> str:
    """Comparable key for a Taiwanese place name: normalised text; optionally keeps only the last level
    ('臺北市大安區龍坡里' -> '龍坡里')."""
    t = normalize_text(name).replace(" ", "")
    if strip_parents:
        parts = re.findall(r".+?(?:市|縣|區|鄉|鎮|村|里)", t)
        t = parts[-1] if parts else t
    return t


def match(left: pd.DataFrame, right: pd.DataFrame, left_on: str | list[str] | None = None,
          right_on: str | list[str] | None = None, fuzzy: float | None = 0.88, how: str = "left", log=None,
          max_unmatched: float = 0.10, **aliases) -> pd.DataFrame:
    """Join on normalised keys (several columns allowed, e.g. [district, village]); unmatched single-column keys
    can be matched fuzzily (similarity >= fuzzy). The log lists how many rows matched exactly, fuzzily, or not."""
    # accept the names a model tends to guess (left_keys / right_keys / on); a wrong keyword cost a whole retry
    left_on = left_on or aliases.pop("left_keys", None) or aliases.get("on")
    right_on = right_on or aliases.pop("right_keys", None) or aliases.pop("on", None)
    if aliases:
        raise TypeError(f"match(): unknown arguments {sorted(aliases)}; use left_on=, right_on=")
    lo = [left_on] if isinstance(left_on, str) else list(left_on)
    ro = [right_on] if isinstance(right_on, str) else list(right_on)
    lk = left[lo].apply(lambda r: "|".join(admin_key(v) for v in r), axis=1)
    rk = right[ro].apply(lambda r: "|".join(admin_key(v) for v in r), axis=1)
    # keys that are not unique merge different places into one (two villages named 中央里 in different districts
    # were joined by village name alone, 2026-10-11): the log flags it and the adapter rejects the step
    dup_right = int(rk.duplicated(keep=False).sum())
    dup_left = int(lk.duplicated(keep=False).sum())
    auto_parent = None
    if (dup_left or dup_right) and len(lo) == 1:
        # repeated names but BOTH tables carry a parent level (district, county, Land ...): join on [parent, name].
        # A model added the parent column and still joined on the name alone (2026-10-11); the tool does it.
        lp, rp = parent_columns(left, lo), parent_columns(right, ro)
        if len(lp) == 1 and len(rp) == 1:
            lk2 = left[[lp[0], lo[0]]].apply(lambda r: "|".join(admin_key(v) for v in r), axis=1)
            rk2 = right[[rp[0], ro[0]]].apply(lambda r: "|".join(admin_key(v) for v in r), axis=1)
            if lk2.duplicated().sum() < max(1, dup_left) and rk2.duplicated().sum() < max(1, dup_right) \
                    and len(set(lk2) & set(rk2)) >= 0.5 * min(len(set(lk2)), len(set(rk2))):
                auto_parent = (lp[0], rp[0])
                lo, ro, lk, rk = [lp[0]] + lo, [rp[0]] + ro, lk2, rk2
                dup_right = int(rk.duplicated(keep=False).sum())
                dup_left = int(lk.duplicated(keep=False).sum())
    right_keys = dict(zip(rk, right.index))
    mapped, n_exact, n_fuzzy, unmatched = [], 0, 0, []
    # codes and ids must match exactly: 63000010002 and 63000010003 are different villages, yet string similarity
    # "matched" 32 of them in a real run (2026-10-11). Fuzzy matching is for names only.
    codes = lk.str.replace("|", "", regex=False).str.fullmatch(r"[\d.\-]+").mean() > 0.5
    if codes:
        fuzzy = None
    for k in lk:
        if k in right_keys:
            mapped.append(right_keys[k]); n_exact += 1
            continue
        if fuzzy and len(lo) == 1:
            hit = difflib.get_close_matches(k, list(right_keys), n=1, cutoff=fuzzy)
        elif fuzzy:
            # several key columns: parents must agree exactly, only the last part may differ (variant characters
            # such as 糖蔀里 / 糖廍里 in two official tables, 2026-10-11)
            head, last = k.rsplit("|", 1)
            same_parent = {rk_.rsplit("|", 1)[1]: rk_ for rk_ in right_keys if rk_.rsplit("|", 1)[0] == head}
            near = difflib.get_close_matches(last, list(same_parent), n=1, cutoff=min(fuzzy, 0.6))
            hit = [same_parent[near[0]]] if near else []
        else:
            hit = []
        if hit:
            mapped.append(right_keys[hit[0]]); n_fuzzy += 1
        else:
            mapped.append(None); unmatched.append(k)
    joined = left.reset_index(drop=True).copy()
    extra = right.loc[[m for m in mapped if m is not None]].reset_index(drop=True)
    extra.columns = [c if c not in joined.columns else f"{c}_right" for c in extra.columns]
    ok = [m is not None for m in mapped]
    mask = np.array(ok, dtype=bool)
    cols = {}
    for c in extra.columns:                       # object column first, so text and numbers both fit
        col = pd.Series([np.nan] * len(joined), dtype="object")
        col[mask] = extra[c].to_numpy()
        cols[c] = col.infer_objects()
    joined = pd.concat([joined, pd.DataFrame(cols)], axis=1)
    if how == "inner":
        joined = joined[ok].reset_index(drop=True)
    if log is not None:
        note = (f"完全對上 {n_exact}、模糊對上 {n_fuzzy}、對不上 {len(unmatched)}"
                + (f"（例：{'、'.join(unmatched[:5])}）" if unmatched else ""))
        if dup_right or dup_left:
            example = rk[rk.duplicated(keep=False)].head(2).tolist() or lk[lk.duplicated(keep=False)].head(2).tolist()
            note += (f"；⚠️ 比對鍵不唯一（左 {dup_left}、右 {dup_right} 列重複，例：{'、'.join(example)}），"
                     f"不同地方可能被當成同一個")
            lp, rp = parent_columns(left, lo), parent_columns(right, ro)
            missing = [side for side, cols in (("left", lp), ("right", rp)) if not cols]
            how = ""
            if missing:
                how = (f" The {' and '.join(missing)} table has no parent-level column. If its ORIGINAL table lists the "
                       f"parent (e.g. district) as rows above its children, call prep.attach_parent(original_table, "
                       f"name_column, new='parent') BEFORE removing those rows with keep_level, then join on "
                       f"[parent, name]; if it has an administrative code, its leading digits may identify the parent.")
            log.flags.append(f"ambiguous join on {lo} / {ro}: keys repeat (e.g. {', '.join(example)}); add the parent "
                             f"level to the keys. Columns that look like a parent level: left {lp}, right {rp}.{how}")
        if len(left) and len(unmatched) > max_unmatched * len(left) and n_exact + n_fuzzy > 0:
            log.flags.append(f"join on {lo} / {ro} left {len(unmatched)} of {len(left)} rows unmatched "
                             f"(e.g. {unmatched[:3]}): the keys probably differ in format or level"
                             + ("; codes are matched exactly only, try names" if codes else ""))
        if n_exact + n_fuzzy == 0 and len(left) and len(right):
            log.flags.append(f"join on {lo} / {ro} matched NOTHING. Example keys: left {lk.head(3).tolist()}, right "
                             f"{rk.head(3).tolist()}. Check that the columns hold the same thing at the same level")
        if auto_parent:
            note = f"名稱重複，自動加上上層鍵 {auto_parent[0]} ↔ {auto_parent[1]}；" + note
        log.step(f"合併（{'+'.join(lo)} ↔ {'+'.join(ro)}）", len(left), len(joined), note)
    return joined


def parent_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Columns whose values are mostly upper-level place names (區/鄉/鎮/市/縣), for join hints."""
    from .table import admin_level
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        vals = df[c].dropna().astype(str).head(200)
        if len(vals) and (vals.map(lambda v: admin_level(v) in ("區", "鄉", "鎮", "市", "縣") and bool(admin_level(v))).mean() > 0.8):
            out.append(c)
    return out


def nearest(lat_a, lon_a, lat_b, lon_b) -> tuple[np.ndarray, np.ndarray]:
    """For each point in A: index of the nearest point in B and the distance (m)."""
    from .geo import haversine_matrix
    la = np.concatenate([np.asarray(lat_a, float), np.asarray(lat_b, float)])
    lo = np.concatenate([np.asarray(lon_a, float), np.asarray(lon_b, float)])
    d = haversine_matrix(la, lo)[: len(lat_a), len(lat_a):]
    return d.argmin(axis=1), d.min(axis=1)


def reduce_points(df: pd.DataFrame, k: int, lat: str = "lat", lon: str = "lon", weight: str | None = None,
                  log=None, seed: int = 0) -> pd.DataFrame:
    """Shrink many points to k representatives by (weighted) k-means: each cluster is represented by its member
    closest to the centroid, carrying the summed weight. Never 'take the first k rows'."""
    from sklearn.cluster import KMeans
    if len(df) <= k:
        return df.copy()
    xy = df[[lat, lon]].to_numpy(float)
    w = df[weight].to_numpy(float) if weight else None
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(xy, sample_weight=w)
    reps = []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        best = members[np.argmin(((xy[members] - km.cluster_centers_[c]) ** 2).sum(axis=1))]
        row = df.iloc[best].copy()
        row["cluster_size"] = len(members)
        if weight:
            row[weight] = float(w[members].sum())
        reps.append(row)
    out = pd.DataFrame(reps).reset_index(drop=True)
    if log is not None:
        log.step(f"規模縮減：k-means 分 {k} 群，每群取最接近中心的代表點", len(df), len(out),
                 f"{'權重（' + weight + '）加總到代表點' if weight else '未加權'}")
    return out
