# Data portal and format coverage

What the data layer has been exercised on, and what it has not. Sources: `qubo_auto/annealing/catalog.py` and
`platforms.py` (portals and APIs), README result tables (end-to-end tasks), `tests/test_data_robustness.py` (file
formats). "Passed" counts are README's: first three finished runs per task, simulated annealing, Gemma-4-31B-it.

## Portals and APIs

| Portal / API | Kind | End-to-end task (passed) | Notes |
|---|---|---|---|
| data.gov.tw | CKAN-like national | Taipei EV chargers (3/3), YouBike tour (2/3, live API), AED zones (3/3) | development cities; formats include Big5 and TWD97/67 coordinates |
| ONS Census 2021 (data.gov.uk CKAN or Nomis API) | CKAN, Nomis | London ambulance stations (1/3) | weakest dev task; census area codes; the README does not say which route each run used |
| datenregister.berlin.de | CKAN | Berlin playgrounds (development only, not in README tables) | a Berlin toilet file mixed decimal commas and points (development run) |
| data.gov.sg | national | Singapore hawker-centre zones (2/3) | one failure fell back to OpenStreetMap |
| data.cityofnewyork.us | Socrata | New York participatory budget (3/3; QAOA 3/3) | |
| data.cambridgema.gov | Socrata | Cambridge participatory budget (3/3; QAOA 3/3) | |
| data.somervillema.gov | Socrata | Somerville participatory budget (2/3 on frozen code) | portal row id taken as a name (fixed) |
| opendata.paris.fr | Opendatasoft | Paris Vélib' tour (3/3) | |
| City of Melbourne registry | city portal | Melbourne bike-dock tour (3/3) | one pass followed an unreviewed repair |
| opendata.swiss | CKAN | Basel AED groups (0/3 on frozen code, 3/3 after fixes) | GeoJSON inside a zip; same-name merging |
| data.gov.ie | CKAN | Dublin bike-station tour (2/3 on frozen code) | Clarifier misread a landmark |
| govdata.de | CKAN | none | in the catalogue only |
| open.canada.ca | CKAN | none | in the catalogue only |
| data.gov.au | CKAN | none | in the catalogue only |
| data.go.jp | CKAN | none | Japanese encodings are tested at file level only |
| data.overheid.nl | CKAN | none | in the catalogue only |
| ArcGIS Hub | ArcGIS | none | |
| data.europa.eu | EU API | none | |
| OpenStreetMap | Overpass-style | used as a fallback (Singapore failure) | crowd-sourced, ranked below official sources |

Roughly half of the catalogue's portals have no end-to-end task. A pass rate says nothing about them.

## File formats

| Format / situation | Tested in `test_data_robustness.py` |
|---|---|
| CSV: UTF-8, UTF-8 BOM, UTF-16, CP950, Big5, GBK, GB18030, Shift-JIS, CP932, EUC-JP, CP949, EUC-KR, CP1252, Latin-1 | yes |
| Delimiters `,` `;` tab `\|`; CRLF and CR line endings; quoted fields with commas and newlines | yes |
| Title and note lines above the header; blank lines before it; empty trailing columns; truncated last line | yes |
| Decimal commas, thousands groups, Swiss `1'234`, currency, percent, accounting negatives | yes |
| HTML page saved as `.csv` | yes (refused) |
| Zip containing CSV or GeoJSON | yes |
| GML graph in a zip (README file ignored) | yes, in `test_contracts_recovery.py` |
| GeoJSON points, polygons (centroid), null geometry | yes |
| JSON records wrapped in API envelopes; nested list cells | yes |
| XLSX: one sheet, several sheets, title and note rows above the header | yes |
| Shapefile points in a zip (x/y plus the .prj, converted by `to_wgs84`) | yes |
| Total rows, swapped latitude/longitude, TWD97 and British National Grid coordinates, duplicates, same name far apart | yes |
| Legacy `.xls`, WFS, XML | **no** |
| Paginated or rate-limited APIs, authentication, redirects, partial downloads | **no** |
| Very large files (the 64 MB member limit) | **no** |
| Table text that addresses a model | yes, `tests/test_data_safety.py` |

The robustness tests use synthetic tables made dirty in these ways; the original failing files of the evaluation runs
are not versioned (raw downloads are excluded), so each real failure is covered only through a synthetic stand-in.

## Adding a case

1. Reproduce the failure type as a small mutation in `test_data_robustness.py` (clean table in, dirty bytes out).
2. Watch it fail, fix the loader or the prep tool, keep the test.
3. Add the portal or format to the tables above.
