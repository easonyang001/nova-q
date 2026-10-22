"""Retry real-data preparation with alternate sources and explicit failure feedback."""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import modeler, opendata
from .adapter import build_adapter
from .artifacts import atomic_write
from .datakit import load_tables
from .provenance import file_digest, source_record
from .progress import record_progress
import hashlib


def prepare_real_data(code: str, spec: str, client, model: str, directory: Path, *,
                      notes: str = "", search_rounds: int = 2, max_downloads: int = 4, log=print,
                      data_check: bool = False):
    directory.mkdir(parents=True, exist_ok=True)
    notes = notes or ""
    ns = modeler.load_module(code)
    record_progress(directory.parent, "data", "running")
    src_path = directory / "_sources.json"
    sources = json.loads(src_path.read_text(encoding="utf-8")) if src_path.exists() else {}
    identity = hashlib.sha256((code + "\0" + spec + "\0" + notes).encode()).hexdigest()
    state_path = directory / "_acquisition_state.json"
    saved = json.loads(state_path.read_text()) if state_path.exists() else {}
    tried = set(saved.get("tried", [])) if saved.get("identity") == identity else set()
    # files already downloaded for this run are not fetched again after a model repair changes the identity
    tried |= {u for s in sources.values() for u in (s.get("resource_url"), s.get("url")) if u}
    errors, report = [], {}
    downloads = 0
    searched: list[str] = []
    osm_deferred = False
    gaps: list[str] = []          # kept across rounds (errors are rebuilt every round)
    taken: set[tuple] = set()          # (dataset, purpose) already downloaded, across rounds
    for round_index in range(search_rounds + 1):
        files = [p for p in sorted(directory.glob("*")) if p.is_file() and not p.name.startswith("_")]
        file_versions = {p.name: file_digest(p) for p in files}
        if src_path.exists():
            file_versions["_sources.json"] = file_digest(src_path)
        cache_path = directory / "_prepared.json"
        cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        if cached.get("identity") == identity and cached.get("files") == file_versions:
            record_progress(directory.parent, "data", "ready", reused=True)
            return cached["data"], [], cached["report"]
        tables, readable = {}, []
        for path in files:
            try:
                tables.update(load_tables([path]))
                readable.append(path)
            except Exception as e:
                log(f"[資料] {path.name} 讀取失敗：{e}；嘗試其他來源")
        if tables:
            log(f"[預處理] 第 {round_index + 1} 次，使用 {len(readable)} 份可讀資料")
            provenance = [sources.get(p.name, {"title": p.name, "publisher": "使用者上傳"}) for p in readable]
            try:
                adapter_code, data, errors, report = build_adapter(ns, tables, spec, client, model,
                    notes=notes + "\n" + "\n".join(errors), sources=provenance, data_check=data_check)
                atomic_write(directory / "_adapter.py", adapter_code)
                if not errors:
                    report["sources"] = provenance
                    atomic_write(cache_path, json.dumps(modeler.to_jsonable({"identity": identity, "files": file_versions,
                                 "data": data, "report": report}), ensure_ascii=False, allow_nan=False, default=str))
                    record_progress(directory.parent, "data", "ready")
                    return data, [], report
            except Exception as e:
                errors = [f"預處理失敗：{type(e).__name__}: {e}"]
        else:
            errors = ["沒有可讀的真實資料；需要搜尋其他來源或上傳檔案"]
        if round_index == search_rounds or downloads >= max_downloads:
            break
        log("[找資料] 依目前缺口重新搜尋，下載失敗時換候選來源")
        try:
            previous = "\n".join(searched[-12:])
            found = opendata.scout(spec + "\nDATA GAPS TO RESOLVE:\n" + "\n".join(list(errors) + gaps)
                                   + ("\nPREVIOUS SEARCHES (no usable data; use different wording):\n" + previous
                                      if previous else ""), ns["DATA_SCHEMA"], client, model)
            ranked = opendata.rank_candidates(spec, ns["DATA_SCHEMA"], found["candidates"], client, model)
            plan = found.get("plan", {})
            searched += [f"- query {q!r}" for q in plan.get("govtw_queries", [])]
            searched += [f"  rejected {c['title']}: {c.get('reason', '')}" for c in ranked
                         if not c.get("relevant")][:6]
        except Exception as e:
            errors.append(f"搜尋失敗：{type(e).__name__}: {e}")
            continue
        crowd_only = [c for c in ranked if c.get("relevant") and ("osm" in c or c.get("source") == "OpenStreetMap")]
        official_found = any(c.get("relevant") and "osm" not in c and c.get("source") != "OpenStreetMap" for c in ranked)
        if crowd_only and not official_found and round_index < search_rounds and not osm_deferred:
            # only crowd-sourced points so far: search once more for the official source before accepting OSM
            # (an unlucky query missed Taipei's official YouBike feed and OSM lost 2 of 38 stations, 2026-10-12)
            osm_deferred = True
            gaps.append("OFFICIAL SOURCE NOT FOUND YET: only OpenStreetMap matched. Search again for the official "
                        "publisher's dataset: city name + the service's exact name, and '即時' / 'real-time' for live data")
            log("[找資料] 只找到群眾編輯資料，先再搜尋一輪官方來源")
            continue
        for candidate in ranked:
            key = candidate.get("url") or candidate["id"]
            # one file per dataset and purpose per round: four copies of one report used every download slot while
            # the data that was still missing was never fetched (Berlin, 2026-10-12)
            # the dataset: its title without the file part and without year/version noise, so the CSV, ZIP and
            # GeoJSON exports of one centroid table count once (London, four copies over two rounds, 2026-10-12)
            base = re.sub(r"\s*\((v\d+|[^)]*export[^)]*)\)\s*$", "", candidate.get("title", "").split(" — ")[0].strip(), flags=re.I)
            same = (base.lower() or candidate.get("page", ""), candidate.get("feeds", ""))
            if same in taken and same[0]:
                continue
            # The user's explicitly requested publisher takes precedence over a
            # title-only LLM rejection. Contents still undergo full validation.
            if key in tried or not (candidate.get("relevant") or candidate.get("requested_source")) or not candidate.get("readable"):
                continue
            tried.add(key)
            atomic_write(state_path, json.dumps({"identity": identity, "tried": sorted(tried), "status": "downloading"}))
            downloads += 1
            try:
                path = opendata.fetch_candidate(candidate, directory)
                taken.add(same)
                sources[path.name] = source_record(candidate, path)
                atomic_write(src_path, json.dumps(sources, ensure_ascii=False))
                log(f"[下載] {path.name}")
            except Exception as e:
                log(f"[下載失敗] {candidate['title']}：{e}；改試下一個來源")
            if downloads >= max_downloads:
                break
    record_progress(directory.parent, "data", "needs_input", errors=errors)
    return {}, errors, report
