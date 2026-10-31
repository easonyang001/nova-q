"""Data-layer robustness: take a clean table, make it dirty in the ways real portals do, and require that the
loader / prep tools give the clean table back.

Every case here is a failure type seen in the evaluation runs (docs/data_landscape.md, README test sets) or a close
relative of one. When a new portal breaks the data layer, add a mutation here first, then fix the loader.
"""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from qubo_auto.annealing import prep
from qubo_auto.annealing.datakit import load_tables

NAMES_ASCII = [f"Station {i:02d}" for i in range(24)]
NAMES_CJK = [f"{c}里" for c in "大安仁愛信義光復敦化忠孝復興樂業建國通化臥龍古亭龍泉福德錦安新生"[:24]]
NAMES_JA = [f"{c}公園" for c in "中央北南東西桜梅松竹緑青赤白黒金銀花鳥風月雪山川海"[:24]]
NAMES_KO = [f"{c}공원" for c in "중앙북남동서벚매화소나무대나무푸른붉은흰검은금은꽃새바람달눈산강바다"[:24]]
NAMES_ACCENT = [f"Café Zürich {i:02d} - Straße" for i in range(24)]


def clean(names, seed=0):
    rng = np.random.default_rng(seed)
    n = len(names)
    return pd.DataFrame({
        "name": names,
        "lat": np.round(25.0 + rng.random(n) * 0.05, 6),
        "lon": np.round(121.5 + rng.random(n) * 0.05, 6),
        "pop": rng.integers(100, 9000, n),
        "area": np.round(rng.random(n) * 10 + 0.5, 3),
    })


def read(tmp_path, filename, raw):
    p = tmp_path / filename
    p.write_bytes(raw)
    tables = load_tables([p])
    assert len(tables) == 1, f"expected one table, got {list(tables)}"
    return next(iter(tables.values()))


def assert_same(got, want):
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    for c in want.columns:
        if pd.api.types.is_numeric_dtype(want[c]):
            assert np.allclose(pd.to_numeric(got[c]), want[c]), c
        else:
            assert got[c].astype(str).str.strip().tolist() == want[c].tolist(), c


def csv_text(df, sep=",", decimal="."):
    return df.to_csv(index=False, sep=sep, decimal=decimal)


# ---------------------------------------------------------------- encodings
@pytest.mark.parametrize("names,encoding", [
    (NAMES_ASCII, "utf-8"), (NAMES_ASCII, "utf-8-sig"), (NAMES_ASCII, "utf-16"),
    (NAMES_CJK, "utf-8"), (NAMES_CJK, "utf-8-sig"), (NAMES_CJK, "cp950"), (NAMES_CJK, "big5"),
    (NAMES_ACCENT, "utf-8"), (NAMES_ACCENT, "utf-8-sig"), (NAMES_ACCENT, "cp1252"), (NAMES_ACCENT, "latin-1"),
])
def test_text_encodings(tmp_path, names, encoding):
    df = clean(names)
    assert_same(read(tmp_path, "t.csv", csv_text(df).encode(encoding)), df)


# ---------------------------------------------------------------- delimiters, preambles, decimal commas
@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_delimiters(tmp_path, sep):
    df = clean(NAMES_ASCII)
    assert_same(read(tmp_path, "t.csv", csv_text(df, sep).encode()), df)


def test_semicolon_with_decimal_commas(tmp_path):
    df = clean(NAMES_ASCII)
    assert_same(read(tmp_path, "t.csv", csv_text(df, ";", ",").encode()), df)


@pytest.mark.parametrize("lines", [
    ["Open data portal export", "Generated 2026-10-01"],
    ["Title", "", "Source: city office", "Licence: CC-BY", "Updated monthly"],
])
def test_preamble_lines_above_header(tmp_path, lines):
    df = clean(NAMES_ASCII)
    assert_same(read(tmp_path, "t.csv", ("\n".join(lines) + "\n" + csv_text(df)).encode()), df)


def test_trailing_spaces_in_cells(tmp_path):
    df = clean(NAMES_ASCII)
    padded = df.copy()
    padded["name"] = padded["name"] + "  "
    assert_same(read(tmp_path, "t.csv", csv_text(padded).encode()), df)


def test_truncated_last_line_does_not_void_the_file(tmp_path):
    df = clean(NAMES_ASCII)
    got = read(tmp_path, "t.csv", csv_text(df).encode().rstrip(b"\n")[:-12])
    assert len(got) >= len(df) - 1


def test_html_page_saved_as_csv_is_refused(tmp_path):
    p = tmp_path / "t.csv"
    p.write_bytes(b"<!DOCTYPE html><html><head><title>Download</title></head><body>Click</body></html>")
    with pytest.raises(ValueError, match="web page"):
        load_tables([p])


# ---------------------------------------------------------------- containers
def zipped(**members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_csv_inside_zip(tmp_path):
    df = clean(NAMES_ASCII)
    assert_same(read(tmp_path, "t.zip", zipped(**{"data.csv": csv_text(df)})), df)


def test_geojson_inside_zip(tmp_path):
    """Basel AED, 2026-10-24: a portal's 'geojson' download was a zip holding the GeoJSON."""
    df = clean(NAMES_ASCII)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": r["name"], "pop": int(r["pop"]), "area": r["area"]},
         "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]}} for _, r in df.iterrows()]}
    got = read(tmp_path, "t.zip", zipped(**{"aed.geojson": json.dumps(fc)}))
    assert len(got) == len(df)
    assert np.allclose(got["lat"], df["lat"]) and np.allclose(got["lon"], df["lon"])


def test_json_records_wrapped_in_api_envelopes(tmp_path):
    df = clean(NAMES_ASCII)
    body = {"status": "ok", "data": {"stations": df.to_dict("records"), "meta": {"n": len(df)}}}
    got = read(tmp_path, "t.json", json.dumps(body).encode())
    assert len(got) == len(df)
    assert_same(got[list(df.columns)], df)


def test_json_nested_list_cells_become_text(tmp_path):
    rows = [{"id": i, "entrances": [{"x": 1}, {"x": 2}]} for i in range(5)]
    got = read(tmp_path, "t.json", json.dumps(rows).encode())
    assert len(got) == 5 and got["entrances"].map(lambda v: isinstance(v, str)).all()


def test_xlsx_single_sheet(tmp_path):
    df = clean(NAMES_CJK)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    assert_same(read(tmp_path, "t.xlsx", buf.getvalue()), df)


# ---------------------------------------------------------------- prep tools
def test_total_rows_are_dropped():
    df = clean(NAMES_CJK)
    dirty = pd.concat([df, pd.DataFrame([{"name": "總計", "lat": 0, "lon": 0, "pop": df["pop"].sum(), "area": df["area"].sum()}]),
                       pd.DataFrame([{"name": "Total", "lat": 0, "lon": 0, "pop": 1, "area": 1}])], ignore_index=True)
    assert_same(prep.drop_total_rows(dirty, ["name"]), df)


def test_swapped_lat_lon_are_fixed():
    df = clean(NAMES_ASCII)
    dirty = df.copy()
    dirty.loc[[2, 9, 17], ["lat", "lon"]] = df.loc[[2, 9, 17], ["lon", "lat"]].to_numpy()
    assert_same(prep.fix_swapped(dirty, bbox=(24.9, 121.4, 25.2, 121.7)), df)


def test_twd97_coordinates_convert_back():
    from pyproj import Transformer
    df = clean(NAMES_ASCII)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True).transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    got = prep.to_wgs84(pd.DataFrame({"x": x, "y": y}), "x", "y")
    assert np.allclose(got["lon"], df["lon"], atol=1e-5) and np.allclose(got["lat"], df["lat"], atol=1e-5)


def test_british_grid_needs_a_declared_system_not_a_guess():
    """UK metres were once taken for web mercator (2026-10-12): the loader must refuse to guess."""
    df = pd.DataFrame({"x": [530000.0, 531000.0, 532500.0], "y": [180000.0, 181000.0, 182500.0]})
    with pytest.raises(ValueError, match="coordinate system"):
        prep.to_wgs84(df, "x", "y")
    got = prep.to_wgs84(df, "x", "y", crs="EPSG:27700")
    assert got["lat"].between(51.3, 51.7).all() and got["lon"].between(-0.5, 0.3).all()


@pytest.mark.parametrize("raw,want", [
    (["1,234", "5,678"], [1234.0, 5678.0]),
    (["NT$300", "NT$1,250"], [300.0, 1250.0]),
    (["（123）", "(45)"], [-123.0, -45.0]),
    (["52,5368", "52.4451"], [52.5368, 52.4451]),
    (["-", "n/a", "7"], [np.nan, np.nan, 7.0]),
])
def test_number_formats(raw, want):
    got = prep.to_number(pd.Series(raw)).to_numpy()
    assert np.allclose(got, want, equal_nan=True)


def test_duplicate_rows_are_removed():
    df = clean(NAMES_ASCII)
    dirty = pd.concat([df, df.iloc[[3, 7]]], ignore_index=True)
    assert_same(prep.dedupe(dirty, ["name"]).sort_values("name").reset_index(drop=True), df.sort_values("name").reset_index(drop=True))


def test_same_name_far_apart_is_kept_by_default_and_merged_by_name_only():
    """Basel AED, 2026-10-24: one name is one place, even more than 150 m apart."""
    df = pd.DataFrame({"name": ["Faculty", "Faculty", "Museum"], "lat": [47.550, 47.552, 47.56], "lon": [7.58, 7.58, 7.59]})
    assert len(prep.merge_same_place(df, "name", "lat", "lon")) == 3
    assert len(prep.merge_same_place(df, "name", "lat", "lon", max_m=None)) == 2


# ---------------------------------------------------------------- harder cases
@pytest.mark.parametrize("names,encoding", [
    (NAMES_JA, "shift_jis"), (NAMES_JA, "cp932"), (NAMES_JA, "euc_jp"), (NAMES_JA, "utf-16"),
    (NAMES_KO, "cp949"), (NAMES_KO, "euc_kr"),
    (NAMES_CJK, "gb18030"), (NAMES_CJK, "gbk"),
])
def test_east_asian_encodings(tmp_path, names, encoding):
    df = clean(names)
    assert_same(read(tmp_path, "t.csv", csv_text(df).encode(encoding)), df)


def test_quoted_fields_with_commas_and_newlines(tmp_path):
    df = clean(NAMES_ASCII)
    df["name"] = [f'"Main", Hall\nbuilding {i}' for i in range(len(df))]
    assert_same(read(tmp_path, "t.csv", csv_text(df).encode()), df)


@pytest.mark.parametrize("fmt", ["{:,.0f}", "{:,.0f}".replace(",", " "), "{:,.0f}"])
def test_thousands_separators_in_quoted_numbers(tmp_path, fmt):
    df = clean(NAMES_ASCII)
    df["pop"] = df["pop"] * 1000
    shown = df.copy()
    shown["pop"] = [fmt.format(v) for v in df["pop"]]
    got = read(tmp_path, "t.csv", csv_text(shown).encode())
    assert prep.to_number(got["pop"]).tolist() == df["pop"].astype(float).tolist()


def test_swiss_apostrophe_thousands():
    got = prep.to_number(pd.Series(["1'234", "56'789", "12"]))
    assert got.tolist() == [1234.0, 56789.0, 12.0]


def test_duplicate_column_names_do_not_lose_columns(tmp_path):
    raw = "name,value,value\nA,1,2\nB,3,4\n".encode()
    got = read(tmp_path, "t.csv", raw)
    assert got.shape == (2, 3)


def test_empty_trailing_columns_and_blank_rows(tmp_path):
    df = clean(NAMES_ASCII)
    text = csv_text(df).replace("\n", ",,\n").rstrip(",\n") + "\n\n\n"
    got = read(tmp_path, "t.csv", text.encode())
    assert len(got.dropna(how="all")) == len(df)


def test_header_after_blank_line(tmp_path):
    df = clean(NAMES_ASCII)
    assert_same(read(tmp_path, "t.csv", ("\n\n" + csv_text(df)).encode()), df)


def test_crlf_and_old_mac_line_endings(tmp_path):
    df = clean(NAMES_ASCII)
    assert_same(read(tmp_path, "t.csv", csv_text(df).replace("\n", "\r\n").encode()), df)
    assert_same(read(tmp_path, "t2.csv", csv_text(df).replace("\n", "\r").encode()), df)


def test_geojson_polygons_get_centroids(tmp_path):
    ring = [[7.0, 47.0], [7.0, 47.1], [7.1, 47.1], [7.1, 47.0], [7.0, 47.0]]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": i}, "geometry": {"type": "Polygon", "coordinates": [ring]}} for i in range(3)]}
    got = read(tmp_path, "t.geojson", json.dumps(fc).encode())
    assert got["lat"].between(47.0, 47.1).all() and got["lon"].between(7.0, 7.1).all()


def test_geojson_with_null_geometry_keeps_the_row(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": 1}, "geometry": None},
        {"type": "Feature", "properties": {"id": 2}, "geometry": {"type": "Point", "coordinates": [7.0, 47.0]}}]}
    assert len(read(tmp_path, "t.geojson", json.dumps(fc).encode())) == 2


def test_utf8_bom_with_semicolons_and_decimal_commas(tmp_path):
    df = clean(NAMES_ASCII)
    assert_same(read(tmp_path, "t.csv", csv_text(df, ";", ",").encode("utf-8-sig")), df)


def test_percent_and_currency_symbols():
    got = prep.to_number(pd.Series(["12.5%", "7%"]))
    assert got.tolist() == [12.5, 7.0]
    assert prep.to_number(pd.Series(["$1,250.50", "€ 300"])).tolist() == [1250.5, 300.0]


def test_currency_magnitude_suffixes_and_trailing_question_marks():
    got = prep.to_number(pd.Series(["$2.846M", "$10,000?", "€1.5 million", "£3K", "$2bn"])).tolist()
    assert np.allclose(got, [2_846_000, 10_000, 1_500_000, 3_000, 2e9])


def test_metres_are_not_millions():
    assert np.isnan(prep.to_number(pd.Series(["5m", "12 m", "3k"]))).all()


def test_xlsx_title_and_note_rows_above_the_header(tmp_path):
    df = clean(NAMES_ASCII)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame([["City statistics 2025"], ["Source: statistics office"], [None]]).to_excel(w, index=False, header=False)
        df.to_excel(w, index=False, startrow=3)
    assert_same(read(tmp_path, "t.xlsx", buf.getvalue()), df)


def test_xlsx_every_sheet_is_a_table(tmp_path):
    df = clean(NAMES_CJK)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame({"note": ["read me"]}).to_excel(w, sheet_name="README", index=False)
        df.to_excel(w, sheet_name="data", index=False)
    p = tmp_path / "m.xlsx"
    p.write_bytes(buf.getvalue())
    tables = load_tables([p])
    assert set(tables) == {"m#README", "m#data"}
    assert_same(tables["m#data"], df)


def test_shapefile_points_in_a_zip_keep_coordinates_and_projection(tmp_path):
    import shapefile
    df = clean(NAMES_ASCII)
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POINT)
    w.field("name", "C")
    w.field("pop", "N")
    for _, r in df.iterrows():
        w.point(r["lon"], r["lat"])
        w.record(r["name"], int(r["pop"]))
    w.close()
    prj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    got = read(tmp_path, "s.zip", zipped(**{"p.shp": shp.getvalue(), "p.shx": shx.getvalue(), "p.dbf": dbf.getvalue(), "p.prj": prj}))
    got = got.sort_values("name").reset_index(drop=True)
    want = df.sort_values("name").reset_index(drop=True)
    assert got["name"].tolist() == want["name"].tolist()
    lonlat = prep.to_wgs84(got, "x", "y", prj=str(got["crs_prj"].iloc[0]))
    assert np.allclose(lonlat["lat"], want["lat"]) and np.allclose(lonlat["lon"], want["lon"])


def test_large_file_column_gets_one_type_not_mixed_chunks(tmp_path):
    """pandas guesses types per chunk by default: numbers in early rows and text later gave a mixed column."""
    rows = ["code,value"] + [f"{i},{i}" for i in range(300_000)] + ["X12,1"]
    got = read(tmp_path, "big.csv", ("\n".join(rows) + "\n").encode())
    assert len({type(v) for v in got["code"]}) == 1


def test_very_long_cells_such_as_wkt_geometry(tmp_path):
    df = clean(NAMES_ASCII)
    df["geom"] = ["MULTIPOLYGON (((" + ", ".join(f"{-73.9 + k * 1e-6} {40.7 + k * 1e-6}" for k in range(9000)) + ")))"] * len(df)
    got = read(tmp_path, "t.csv", csv_text(df).encode())
    assert len(got) == len(df) and got["geom"].str.len().min() > 131072


def test_column_names_lose_edge_spaces_including_no_break_spaces(tmp_path):
    df = clean(NAMES_ASCII)
    shown = df.rename(columns={"name": "\xa0 name", "pop": "pop  "})
    assert_same(read(tmp_path, "t.csv", csv_text(shown).encode()), df)


@pytest.mark.parametrize("filename", ["t.geojson", "t.json", "t.zip", "t.xlsx", "t.xls"])
def test_csv_under_a_wrong_name_is_read_by_content(tmp_path, filename):
    """Toronto labelled 32 CSV downloads GeoJSON (corpus, 2026-10-30)."""
    df = clean(NAMES_ASCII)
    got = read(tmp_path, filename, csv_text(df).encode())
    assert_same(got, df)
    assert any("delimited text" in w for w in got.attrs.get("load_warnings", []))


def test_json_under_a_csv_name_is_read_as_json(tmp_path):
    df = clean(NAMES_ASCII)
    got = read(tmp_path, "t.csv", json.dumps(df.to_dict("records")).encode())
    assert_same(got[list(df.columns)], df)


@pytest.mark.parametrize("filename", ["t.json", "t.geojson", "t.zip"])
def test_web_page_under_any_data_name_is_refused(tmp_path, filename):
    p = tmp_path / filename
    p.write_bytes(b"<!DOCTYPE html><html><head><title>Not found</title></head></html>")
    with pytest.raises(ValueError, match="web page"):
        load_tables([p])


def test_excel_inside_zip(tmp_path):
    df = clean(NAMES_ASCII)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    assert_same(read(tmp_path, "t.zip", zipped(**{"stats.xlsx": buf.getvalue()})), df)


def test_archive_without_tables_says_what_it_holds(tmp_path):
    p = tmp_path / "docs.zip"
    p.write_bytes(zipped(**{"report.pdf": b"%PDF-1.4", "map.jpg": b"\xff\xd8"}))
    with pytest.raises(ValueError, match=r"holds \.jpg, \.pdf"):
        load_tables([p])


@pytest.mark.parametrize("with_cpg", [True, False])
def test_shapefile_attributes_in_cp1252(tmp_path, with_cpg):
    """A UK layer's dbf held cp1252 text ('Road\xa0\xa0', 'Café'); utf-8 then cp950 failed or garbled it."""
    import shapefile
    names = [f"Café {i} \u2013 Springdale Road\u00a0" for i in range(12)]
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POINT, encoding="cp1252")
    w.field("name", "C", size=60)
    for i, n in enumerate(names):
        w.point(-0.1 + i * 1e-3, 51.5)
        w.record(n)
    w.close()
    members = {"p.shp": shp.getvalue(), "p.shx": shx.getvalue(), "p.dbf": dbf.getvalue()}
    if with_cpg:
        members["p.cpg"] = b"1252"
    got = read(tmp_path, "s.zip", zipped(**members))
    assert sorted(got["name"]) == sorted(n.strip() for n in names)


def test_cp1252_dashes_and_curly_quotes_are_not_control_characters(tmp_path):
    names = [f"Café ‘{i}’ – King’s Road" for i in range(24)]
    df = clean(names)
    assert_same(read(tmp_path, "t.csv", csv_text(df).encode("cp1252")), df)


def test_taiwanese_shapefile_without_cpg_stays_chinese(tmp_path):
    import shapefile
    names = [f"臺北市大安區{c}里活動中心" for c in "仁愛信義光復敦化忠孝復興樂業建國"]
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POINT, encoding="cp950")
    w.field("name", "C", size=60)
    for n in names:
        w.point(121.5, 25.0)
        w.record(n)
    w.close()
    got = read(tmp_path, "s.zip", zipped(**{"p.shp": shp.getvalue(), "p.shx": shx.getvalue(), "p.dbf": dbf.getvalue()}))
    assert sorted(got["name"]) == sorted(names)


def test_double_encoded_utf8_in_the_source_is_repaired(tmp_path):
    """Seattle and Canadian files hold UTF-8 that was once read as Windows-1252 and saved again."""
    names = [f"Café {i} – King’s Road, unités" for i in range(24)]
    df = clean(names)
    mangled = df.copy()
    mangled["name"] = [n.encode("utf-8").decode("cp1252") for n in names]
    assert_same(read(tmp_path, "t.csv", csv_text(mangled).encode("utf-8")), df)


@pytest.mark.parametrize("text", ["SÃO PAULO", "Ãrvore", "Â", "Âge du bronze"])
def test_ordinary_text_with_a_tilde_a_is_left_alone(tmp_path, text):
    from qubo_auto.annealing.datakit import _unmangle
    assert _unmangle(text) == text


def test_degree_named_columns_with_projected_values_are_flagged():
    from qubo_auto.annealing.datakit import coordinate_name_warnings, profile
    df = pd.DataFrame({"name": list("abcdefgh"), "latitude": [1.84e6, 1.85e6, 99999, 99999, 99999, 1.86e6, 1.87e6, 1.88e6],
                       "longitude": [6.44e6, 6.45e6, 99999, 99999, 99999, 6.46e6, 6.47e6, 6.48e6]})
    warnings_ = coordinate_name_warnings(df)
    assert len(warnings_) == 2 and "99999" in warnings_[0]
    assert "WARNING:" in profile("t", df)
    assert coordinate_name_warnings(clean(NAMES_ASCII)) == []


def test_legacy_xls_with_a_title_row(tmp_path):
    xlwt = pytest.importorskip("xlwt")                # test-only: writes the legacy format
    df = clean(NAMES_CJK)
    book = xlwt.Workbook(encoding="utf-8")
    sheet = book.add_sheet("data")
    sheet.write(0, 0, "臺北市各里人口統計")
    for j, c in enumerate(df.columns):
        sheet.write(2, j, c)
    for i, row in enumerate(df.itertuples(index=False), start=3):
        for j, v in enumerate(row):
            sheet.write(i, j, v.item() if hasattr(v, "item") else v)
    buf = io.BytesIO()
    book.save(buf)
    assert_same(read(tmp_path, "t.xls", buf.getvalue()), df)


def test_dates_as_excel_header_cells_do_not_break_the_profile(tmp_path):
    """Canadian workbooks had dates as column headers: the model-facing profile crashed on them."""
    import datetime
    from qubo_auto.annealing.datakit import profile
    df = pd.DataFrame({"region": ["A", "B", "C"], datetime.datetime(2024, 1, 1): [1, 2, 3], 2025.0: [4, 5, 6]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    got = read(tmp_path, "t.xlsx", buf.getvalue())
    assert list(got.columns) == ["region", "2024-01-01", "2025"]
    assert "2024-01-01" in profile("t", got)


def test_repeated_column_names_in_a_file_with_broken_rows(tmp_path):
    """The tolerant parser (used when a row is broken) took a frame for a column when a name repeated."""
    text = 'name,value,value\nA,1,2\nB,3,4\nC,5,6\nD,"7,8\n'
    got = read(tmp_path, "t.csv", text.encode())
    assert list(got.columns)[:3] == ["name", "value", "value.1"] and len(got) >= 3


def test_geojson_declaring_web_mercator_is_converted(tmp_path):
    from pyproj import Transformer
    df = clean(NAMES_ASCII)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    fc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
          "features": [{"type": "Feature", "properties": {"name": n}, "geometry": {"type": "Point", "coordinates": [a, b]}}
                       for n, a, b in zip(df["name"], x, y)]}
    got = read(tmp_path, "t.geojson", json.dumps(fc).encode())
    assert np.allclose(got["lat"], df["lat"]) and np.allclose(got["lon"], df["lon"])
    assert any("EPSG:3857" in w for w in got.attrs["load_warnings"])


def test_two_level_excel_header_names_repeated_columns_by_their_group(tmp_path):
    """'Primary' under two budget blocks became 'Primary' and 'Primary.1' (UK workbooks)."""
    rows = [["Council budget statement 2025/26", None, None, None, None],
            [None, "Schools block", None, "High needs block", None],
            ["Area", "Primary", "Secondary", "Primary", "Secondary"],
            ["North", 1, 2, 3, 4], ["South", 5, 6, 7, 8], ["East", 9, 10, 11, 12]]
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    got = read(tmp_path, "t.xlsx", buf.getvalue())
    assert list(got.columns) == ["Area", "Primary (Schools block)", "Secondary (Schools block)",
                                 "Primary (High needs block)", "Secondary (High needs block)"]
    assert got["Primary (High needs block)"].tolist() == [3, 7, 11]


def test_empty_extra_sheets_are_dropped_but_an_all_empty_workbook_is_kept(tmp_path):
    df = clean(NAMES_ASCII)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        df.to_excel(w, sheet_name="data", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Sheet2", index=False)
    p = tmp_path / "m.xlsx"
    p.write_bytes(buf.getvalue())
    assert list(load_tables([p])) == ["m#data"]


@pytest.mark.parametrize("flat", [True, False])
def test_esri_file_geodatabase_in_a_zip(tmp_path, flat):
    """Seven UK downloads were zipped file geodatabases, some without their .gdb folder (flat)."""
    pytest.importorskip("pyogrio")
    shapely = pytest.importorskip("shapely")
    from pyogrio.raw import write
    df = clean(NAMES_ASCII)
    gdb = tmp_path / "src" / "places.gdb"
    gdb.parent.mkdir()
    points = shapely.to_wkb(shapely.points(df["lon"].to_numpy(), df["lat"].to_numpy()))
    write(str(gdb), points, [df["name"].to_numpy(dtype=object), df["pop"].to_numpy()], ["name", "pop"],
          layer="places", driver="OpenFileGDB", geometry_type="Point", crs="EPSG:4326")
    members = {(f.name if flat else f"places.gdb/{f.name}"): f.read_bytes() for f in gdb.iterdir()}
    got = read(tmp_path, "g.zip", zipped(**members)).sort_values("name").reset_index(drop=True)
    want = df.sort_values("name").reset_index(drop=True)
    assert got["name"].tolist() == want["name"].tolist()
    assert np.allclose(got["y"], want["lat"]) and np.allclose(got["x"], want["lon"])


def test_replacement_characters_in_the_source_are_reported(tmp_path):
    """Los Angeles and San Francisco files carried U+FFFD in the published bytes: the letters are gone upstream."""
    df = clean([f"Library�s branch {i}" for i in range(24)])
    got = read(tmp_path, "t.csv", csv_text(df).encode("utf-8"))
    assert any("replacement characters" in w for w in got.attrs.get("load_warnings", []))


def test_profile_says_when_cells_are_lists_of_numbers():
    from qubo_auto.annealing.datakit import profile
    df = pd.DataFrame({"stop": list("abcdef"), "routes": ["1,3,4", "12", "2, 6, 28", "1,3", "8,9", "36,62"]})
    assert "LISTS of numbers" in profile("t", df)
    assert "LISTS" not in profile("t", clean(NAMES_ASCII))
