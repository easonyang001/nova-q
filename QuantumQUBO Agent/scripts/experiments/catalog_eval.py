"""Recall of the catalogue index (semantic) vs the portals' keyword search for the data the dev cases need.

Each need is written as the scout would describe it (what, which measure, where), NOT as the publisher's title;
`gold` lists words that the correct dataset's title must contain. A development diagnostic only.

    python scripts/experiments/catalog_eval.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qubo_auto.annealing import catalog, opendata, platforms          # noqa: E402

NEEDS = [  # (portal, need as a description, keyword query as the scout wrote it in failed runs, gold title words)
    ("data.gov.tw", "population of every village (里) in Taipei City, latest month / 臺北市 各里 人口數",
     "臺北市 里 人口", ["各里", "人口"]),
    ("data.gov.tw", "boundary polygons or centre coordinates of Taipei villages / 臺北市 里 邊界 座標",
     "臺北市 里 座標", ["里界圖"]),
    ("data.gov.tw", "real-time list of all YouBike 2.0 stations in Taipei with coordinates / 臺北市 YouBike 站點 經緯度",
     "臺北市 YouBike2.0 站點 經緯度", ["YouBike", "即時"]),
    ("data.gov.tw", "locations of AED defibrillators in Taipei City with venue name and category / 臺北市 AED 設置地點",
     "臺北市 AED 位置", ["AED"]),
    ("data.gov.tw", "national list of AED installation sites in Taiwan / 全國 AED 設置場所",
     "AED 設置場所", ["AED"]),
    ("data.gov.tw", "locations of public elementary schools in Taipei / 臺北市 國民小學 位置",
     "臺北市 國民小學", ["國民小學"]),
    ("data.gov.uk", "population-weighted centre coordinates of every MSOA (middle super output area) in England, 2021",
     "Camden MSOA boundaries centroids coordinates", ["Middle layer Super Output Areas", "Population Weighted Centroids"]),
    ("data.gov.uk", "population-weighted centre coordinates of every LSOA in England and Wales, 2021 census geography",
     "Camden LSOA centroids", ["Lower layer Super Output Areas", "Population Weighted Centroids"]),
    ("data.gov.uk", "locations of London cycle hire docking stations (Santander Cycles)",
     "Santander cycles docking stations London", ["cycle"]),
    ("data.gov.uk", "locations of fire stations in England", "fire stations England", ["fire station"]),
    ("datenregister.berlin.de", "planning areas (LOR Planungsräume) of Berlin with geometry",
     "Berlin LOR Planungsräume Geometrie", ["Planungsr"]),
    ("datenregister.berlin.de", "residents per statistical block in Berlin / Einwohner je Block",
     "Einwohnerdichte Block Berlin", ["Einwohner"]),
    ("datenregister.berlin.de", "locations of public playgrounds in Berlin / Spielplätze Berlin",
     "Spielplätze Berlin", ["Spielpl"]),
]


def hit(titles, gold):
    t = [x.lower() for x in titles]
    return next((i + 1 for i, x in enumerate(t) if all(g.lower() in x for g in gold)), None)


def main():
    print(f"{'need':<62} {'semantic':>9} {'keyword':>8}")
    sem_hits = kw_hits = n = 0
    for portal, need, kw, gold in NEEDS:
        if portal not in catalog.available():
            continue
        n += 1
        sem = [c["title"] for c in catalog.search([need], [portal], k=10)]
        sem_rank = hit(list(dict.fromkeys(t.split(" — ")[0] for t in sem)), gold)
        if portal == "data.gov.tw":
            kwt = [h["title"] for h in opendata.search_govtw(kw, limit=10)]
        else:
            kwt = [c["title"] for c in platforms.search("ckan", portal, kw, limit=10)]
        kw_rank = hit(list(dict.fromkeys(t.split(" — ")[0] for t in kwt)), gold)
        sem_hits += sem_rank is not None and sem_rank <= 5
        kw_hits += kw_rank is not None and kw_rank <= 5
        print(f"{need[:60]:<62} {str(sem_rank):>9} {str(kw_rank):>8}")
    print(f"found in top 5: semantic {sem_hits}/{n}, keyword {kw_hits}/{n}")


if __name__ == "__main__":
    main()
