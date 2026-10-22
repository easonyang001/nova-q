# Test set (frozen code bd9e90bef707697d; CGA-address change only, see PREREGISTRATION), 2026-10-22
Scoring: first 3 finished runs per task. No pipeline code changes while running; failures are recorded, not fixed.

| # | run | task | verdict | notes |
|---|---|---|---|---|
| 1 | g1 r0 | sg_hawker_groups | PASS | 12 centres; ArcGIS Hub copy of the Hawker Centres layer; gap +6.2 % vs reference (within 10 %) ; 44 min |
| 2 | g0 r0 | sg_hawker_groups | FAIL (data) | search found no official source in two rounds and fell back to OpenStreetMap amenity=food_court: 49 food courts instead of 12 hawker centres. The data checker saw the same table, so gates ① ③ could not tell; the judge did. 67 min |
| 3 | g1 r0 | paris_velib_tour | PASS | 10 stations from opendata.paris.fr; route 3,103 m = reference; 54.5 min |
| 4 | g0 r0 | paris_velib_tour | PASS | 10 stations; route = reference; 50.5 min |
| 5 | g1 r0 | cambridge_pb_budget | PASS | 16 projects from data.cambridgema.gov; 10 funded, $518,000, 24,769 votes = exact optimum (vote-order rule 22,004); 18.7 min |
| 6 | g0 r0 | cambridge_pb_budget | PASS | exact optimum again (24,769 votes); 29.5 min |
| 7 | g1 r1 | sg_hawker_groups | PASS | ArcGIS Hub copy again; same grouping as run 1 (+6.2 %); 46.9 min. Singapore final: 2/3 |
| (8) | g0 r1 | sg_hawker_groups | FAIL (data), not counted (4th run) | OpenStreetMap food courts again |
| 9 | g1 r1 | paris_velib_tour | PASS | route = reference; 49.6 min. Paris final: 3/3 |
| 10 | g1 r1 | cambridge_pb_budget | PASS | exact optimum; 29.9 min. Cambridge final: 3/3 |

## Final (first 3 finished runs per task)
| Task | Passed |
|---|:---:|
| Singapore hawker-centre zones | 2/3 |
| Paris Vélib' tour | 3/3 |
| Cambridge participatory budget | 3/3 |
| **Total** | **8/9** |

The single counted failure (and the uncounted 4th Singapore run) is the same data-search miss: no official source
found, fallback to OpenStreetMap food courts. No pipeline code was changed during the test set (apart from the
CGA-address change recorded in PREREGISTRATION.md, which does not affect SA runs).
