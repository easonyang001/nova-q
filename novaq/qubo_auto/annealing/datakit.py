"""Load user / open-data files and describe them for the LLM WITHOUT showing it the data.

The LLM never reads raw data (a real dataset easily exceeds the context window). It sees a bounded profile of
every table: columns, types, ranges, missing values and a few sample rows, then writes code that runs on the
full tables (see adapter.py).
"""
from __future__ import annotations
import io
import csv
import zipfile
import json
from pathlib import Path
from typing import Any

import pandas as pd

MAX_TABLE_CHARS = 3000        # profile size per table
MAX_TOTAL_CHARS = 12000       # all profiles together


def _delimited(raw, suffix=".csv"):
    text = _decode(raw)
    head = text.lstrip()[:200].lower()
    if head.startswith(("<!doctype html", "<html")) or "<head>" in head:
        # a landing page saved under a data name (seen: a Spanish ".csv" that was an HTML page)
        raise ValueError("this file is a web page, not a data table; look for the download link on the page")
    try:
        separator = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        separator = "\t" if suffix == ".tsv" else ","
    skip = _preamble_lines(text, separator)
    try:
        return pd.read_csv(io.StringIO(text), sep=separator, skiprows=skip)
    except pd.errors.ParserError:
        # a few broken rows (a truncated last line, an unclosed quote) must not void the whole file
        rows = list(csv.reader(io.StringIO(text), delimiter=separator))[skip:]
        width = max(set(map(len, rows)), key=[len(r) for r in rows].count) if rows else 0
        good = [r for r in rows if len(r) == width]
        df = pd.DataFrame(good[1:], columns=good[0]) if good else pd.DataFrame()
        for c in df.columns:
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() == df[c].replace("", None).notna().sum():   # every value was a number
                df[c] = converted
        skipped = len(rows) - len(good)
        df.attrs["load_warnings"] = [f"{skipped} malformed row(s) skipped (expected {width} fields)" if skipped else
                                     "read with a tolerant parser (an unclosed quote at the end of the file)"]
        return df


def _preamble_lines(text: str, separator: str, probe: int = 60) -> int:
    """Title / notes lines above the real header: the header is the first line with the usual number of fields."""
    rows = list(csv.reader(io.StringIO("\n".join(text.splitlines()[:probe])), delimiter=separator))
    if len(rows) < 3:
        return 0
    counts = [len(r) for r in rows]
    usual = max(set(counts[len(counts) // 3:]), key=counts[len(counts) // 3:].count)
    if usual <= 1:
        return 0
    return next((i for i, c in enumerate(counts) if c == usual), 0)


def _gml(raw):
    """Adjacency table preserving all nodes, including isolated vertices."""
    import networkx as nx
    graph = nx.parse_gml(_decode(raw).splitlines(), label=None)
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("This loader requires a simple undirected GML graph")
    nodes = sorted(graph.nodes())
    values = nx.to_numpy_array(graph, nodelist=nodes, weight=None, dtype=int)
    table = pd.DataFrame(values, columns=[f"adj_{i}" for i in range(len(nodes))])
    table.insert(0, "node_id", nodes)
    return table


def _decode(raw: bytes) -> str:
    """UTF-8 first. Anything else goes to charset_normalizer, which told Big5, GBK, Shift-JIS and EUC-JP/KR apart in the
    robustness tests; a fixed cp950 -> big5 order read EUC-JP files as Chinese-looking nonsense without an error
    (cp950 happily decodes most two-byte encodings). cp950/big5 remain the fallback for Taiwanese files."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    from charset_normalizer import from_bytes
    best = from_bytes(raw).best()
    if best:
        encoding = "cp950" if best.encoding == "big5" else best.encoding     # cp950 is Big5 plus the ETEN extensions
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return str(best)
    for encoding in ("cp950", "big5"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def _geojson_table(obj: dict) -> pd.DataFrame:
    rows = []
    for f in obj.get("features", []):
        props = dict(f.get("properties") or {})
        geom = f.get("geometry") or {}
        coords = geom.get("coordinates")
        pts: list = []

        def walk(c):
            if isinstance(c, (list, tuple)) and len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2]):
                pts.append(c[:2])
            elif isinstance(c, (list, tuple)):
                for x in c:
                    walk(x)
        walk(coords)
        if pts:                                   # point, or centroid of a line / polygon's vertices
            props["lon"] = sum(p[0] for p in pts) / len(pts)
            props["lat"] = sum(p[1] for p in pts) / len(pts)
        props["geometry_type"] = geom.get("type")
        rows.append(props)
    return pd.DataFrame(rows)


def _shapefile_tables(raw: bytes, stem: str) -> dict[str, pd.DataFrame]:
    """A zipped Shapefile (or several): attributes + centroid of each shape. Coordinates stay in the file's own
    system; Taiwanese government layers are usually TWD97 / TM2 metres (x_ east, y_ north), which is fine for
    distances. The .prj text is kept in a column so the adapter can see the coordinate system."""
    import zipfile
    import shapefile                                # pyshp
    out = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        for shp in [n for n in names if n.lower().endswith(".shp")]:
            base = shp[:-4]
            def part(ext):
                hit = next((n for n in names if n.lower() == (base + ext).lower()), None)
                return io.BytesIO(z.read(hit)) if hit else None
            prj = part(".prj")
            for enc in ("utf-8", "cp950"):
                try:
                    r = shapefile.Reader(shp=part(".shp"), dbf=part(".dbf"), shx=part(".shx"), encoding=enc)
                    fields = [f[0] for f in r.fields[1:]]
                    rows = []
                    for sr in r.iterShapeRecords():
                        pts = sr.shape.points
                        row = dict(zip(fields, sr.record))
                        if pts:
                            row["x"] = sum(p[0] for p in pts) / len(pts)
                            row["y"] = sum(p[1] for p in pts) / len(pts)
                        rows.append(row)
                    df = pd.DataFrame(rows)
                    df["crs_prj"] = prj.read().decode("utf-8", "replace")[:200] if prj else ""
                    out[f"{stem}.{Path(base).name}" if len(out) or shp.count("/") else stem] = df
                    break
                except UnicodeDecodeError:
                    continue
    return out


def load_tables(paths: list[str | Path]) -> dict[str, pd.DataFrame]:
    """Every readable table in the files: CSV/TSV/TXT, XLSX (one table per sheet), JSON, GeoJSON."""
    tables: dict[str, pd.DataFrame] = {}
    for p in map(Path, paths):
        stem, suf = p.stem, p.suffix.lower()
        raw = p.read_bytes()
        if suf in (".zip", ".shp") or raw[:2] == b"PK" and suf not in (".xlsx", ".xlsm"):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if any(n.lower().endswith(".shp") for n in archive.namelist()):
                    tables.update(_shapefile_tables(raw, stem))
                else:
                    for info in archive.infolist():
                        extension = Path(info.filename).suffix.lower()
                        if extension == ".gml" and info.file_size <= 64 * 1024 * 1024:
                            tables[f"{stem}#{Path(info.filename).stem}"] = _gml(archive.read(info))
                        if extension in (".json", ".geojson") and info.file_size <= 64 * 1024 * 1024:
                            # a portal's "geojson" download can be a zip holding the GeoJSON (Basel AED, 2026-10-24)
                            tables.update(_json_tables(archive.read(info), f"{stem}#{Path(info.filename).stem}"))
                        if extension in (".csv", ".tsv", ".txt") and info.file_size <= 64 * 1024 * 1024:
                            # A graph README is not a data table.
                            if not (extension == ".txt" and any(n.lower().endswith(".gml") for n in archive.namelist())):
                                tables[f"{stem}#{Path(info.filename).stem}"] = _delimited(archive.read(info), extension)
                    if not any(k.startswith(stem + "#") for k in tables):
                        raise ValueError(f"{p.name}: archive contains no readable CSV/TSV/text, JSON or Shapefile")
        elif suf in (".xlsx", ".xlsm", ".xls"):
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
            for sheet, df in sheets.items():
                tables[stem if len(sheets) == 1 else f"{stem}#{sheet}"] = df
        elif suf == ".gml":
            tables[stem] = _gml(raw)
        elif suf in (".json", ".geojson"):
            tables.update(_json_tables(raw, stem))
        else:                                      # delimited text
            tables[stem] = _delimited(raw, suf)
    return {k: _clean_cells(_flat_cells(v)) for k, v in tables.items()}


def _json_tables(raw: bytes, stem: str) -> dict[str, pd.DataFrame]:
    obj = json.loads(_decode(raw))
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        return {stem: _geojson_table(obj)}
    if isinstance(obj, list):
        return {stem: pd.json_normalize(obj)}
    out: dict[str, pd.DataFrame] = {}
    if isinstance(obj, dict):
        lists = _record_lists(obj)
        for k, v in lists.items() or {"data": [obj]}.items():
            out[f"{stem}.{k}" if lists else stem] = pd.json_normalize(v)
    return out


def _record_lists(obj, path: str = "", depth: int = 0) -> dict[str, list]:
    """Lists of records anywhere in a JSON object (APIs wrap them: {"data": {"park": [...]}}), keyed by their path.
    Only lists of at least 2 records nested up to 4 levels; the top level was searched alone before (2026-10-11:
    1,758 parking lots read as one row)."""
    found: dict[str, list] = {}
    if depth > 4 or not isinstance(obj, dict):
        return found
    for k, v in obj.items():
        key = f"{path}.{k}" if path else str(k)
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, dict) for x in v[:20]):
            found[key] = v
        elif isinstance(v, dict):
            found.update(_record_lists(v, key, depth + 1))
    return found


def _clean_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Text cells without edge spaces (a district name with a trailing space missed an exact filter, 2026-10-25), and a column of numbers written
    with decimal commas, alone or mixed with points ('52,5368' and '52.4451'), as numbers."""
    import re as _re
    from .prep.text import decimal_comma
    df = df.copy()
    num = _re.compile(r"^[-+]?\d+([.,]\d+)?$")
    for c in df.columns:
        if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue                                    # pandas 3 reads text as the "str" dtype, not object
        col = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        df[c] = col
        texts = [v for v in col.dropna().tolist() if isinstance(v, str) and v != ""]
        if texts and all(num.match(v) for v in texts) and decimal_comma(texts):
            df[c] = pd.to_numeric(col.map(lambda v: v.replace(",", ".") if isinstance(v, str) else v), errors="coerce")
    return df


def _flat_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One value per cell: JSON lists / nested objects left in cells (e.g. a parking lot with a list of entrances)
    become JSON text. Every tool after this sees an ordinary table (unhashable cells crashed one, 2026-10-11)."""
    nested = [c for c in df.columns if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (list, dict))).any()]
    if not nested:
        return df
    df = df.copy()
    for c in nested:
        df[c] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (list, dict)) else v)
    return df


def profile(name: str, df: pd.DataFrame, max_chars: int = MAX_TABLE_CHARS) -> str:
    lines = [f"TABLE {name!r}: {len(df)} rows x {df.shape[1]} columns"]
    lines += [f"LOAD WARNING: {w}" for w in df.attrs.get("load_warnings", [])]
    for col in df.columns:
        s = df[col]
        miss = f", {s.isna().mean():.0%} missing" if s.isna().any() else ""
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            # a True/False column describes as count/unique/top, without min/max: it crashed the profile (2026-10-13)
            d = s.describe()
            lines.append(f"- {col!r} ({s.dtype}{miss}): min {d['min']:.6g}, max {d['max']:.6g}, mean {d['mean']:.6g}")
        else:
            vals = s.dropna().astype(str)
            top = ", ".join(f"{v!r}" for v in vals.value_counts().head(4).index)
            lines.append(f"- {col!r} (text, {vals.nunique()} distinct{miss}): e.g. {top}")
    sample = df.head(3).to_dict(orient="records")
    lines.append("first rows: " + json.dumps(sample, ensure_ascii=False, default=str))
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars] + " …(profile truncated)"


def profiles(tables: dict[str, pd.DataFrame]) -> str:
    out, used = [], 0
    for name, df in tables.items():
        p = profile(name, df, max_chars=min(MAX_TABLE_CHARS, max(800, (MAX_TOTAL_CHARS - used))))
        out.append(p)
        used += len(p)
        if used >= MAX_TOTAL_CHARS:
            out.append(f"… {len(tables) - len(out)} more table(s) not shown")
            break
    return "\n\n".join(out)
