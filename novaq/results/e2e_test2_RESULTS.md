# Test set 2 (frozen code e9f33f6315f8db90, LLM_TEMPERATURE=0.7), 2026-10-24
Scoring: first 3 finished runs per task. Runs are independent samples (temperature 0.7), unlike test set 1
(temperature 0). No pipeline code changes while running; failures are recorded, not fixed.

| # | run | task | verdict | notes |
|---|---|---|---|---|
| 1 | g0 r0 | basel_aed_groups | FAIL (data) | 38 places instead of 35: `merge_same_place` merged same-name rows only within 150 m; three places (Departement Geschichte, MAB Musikakademie, Theologische Fakultät) have two records farther apart, although the user said same-name places are one place; 35.9 min |
| 2 | g1 r0 | basel_aed_groups | FAIL (data) | same 38 places; 47.9 min |
| 3 | g0 r0 | dublin_bike_tour | PASS | 9 stations from data.gov.ie; tour 2,713 m = reference; 30.9 min |
| 4 | g0 r0 | somerville_pb_budget | FAIL (data) | right 20 proposals, totals and budget, but the Socrata row id (`:id`, e.g. `row-3j4m~khmq.utc9`) used as the proposal identifier, so no delivered proposal can be named; also a false "substitute source" flag (the file was on data.somervillema.gov, found through catalog.data.gov); 19.1 min |
| 5 | g0 r1 | basel_aed_groups | FAIL (model) | the model was right (one-team-per-site constraints present) but validation shrank the example to 4 sites, which the model rounds to 0 (multiple of 5 teams); the requirement check on the empty instance reported the constraints missing, and three repairs could not fix a correct model; 17.3 min. Basel final: 0/3 |
| 6 | g1 r0 | dublin_bike_tour | FAIL (model) | the clarifier recorded "start from Connolly station and return" although the user answered "any station can be the start"; the model added the station as a depot, so model and true metric rank random tours differently (Spearman 0.892 < 0.9). The delivered tour happens to be optimal (2,713 m); 85.3 min |
| 7 | g1 r0 | somerville_pb_budget | PASS | 20 proposals; score 103,058 = exact optimum; 19.9 min |
| 8 | g0 r1 | dublin_bike_tour | PASS | tour = reference after a requirement-check repair of the model; 56.7 min. Dublin final: 2/3 |
| 9 | g0 r1 | somerville_pb_budget | PASS | exact optimum; 19.1 min. Somerville final: 2/3 |
| (10) | g1 r1 | basel_aed_groups | stopped, not counted (4th run) | |

## Final (first 3 finished runs per task)
| Task | Passed |
|---|:---:|
| Basel AED groups | 0/3 |
| Dublin bike tour | 2/3 |
| Somerville participatory budget | 2/3 |
| **Total** | **4/9** |

## What failed, and the generic fix (applied after the test set, outside the frozen code)
1. Same-name places farther apart than 150 m stayed separate although the request says one name = one place
   (runs 1, 2). Fix: `merge_same_place` can merge by name only (`max_m=None`, mean coordinates); the data agent uses it
   when the confirmed spec says so.
2. Model validation shrank the example to an empty instance and reported correct constraints as missing (run 5).
   Fix: requirement mapping is checked on the full-size example; a shrunk instance with no variables is not accepted
   as verified.
3. A system row id (`:id`) became the entity identifier (run 4). Fix: identifiers must be a meaningful name or code
   column, never a portal's internal row id.
4. "Substitute source" flagged a file that came from the requested portal through a catalogue (run 4). Fix: the check
   also looks at the download URL.
5. The clarifier turned an area-defining landmark into a start point against the user's answer (run 6).
   Fix: a deterministic check between the user's answers and the confirmed spec for start/end points.
