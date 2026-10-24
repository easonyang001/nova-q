# Post-fix validation 2 (2026-10-24), reported apart from the frozen test set 2
- Code: 990851b (fixes for the test set 2 failures). Same config as test set 2 (Gemma doer+checker, thinking on,
  temperature 0.7, 1200 s, 64000 tokens, agent data prep, SA).
- Purpose: check in full runs that the five fixes hold: Basel (name-only merge, validator on the full example),
  Dublin (clarifier checked against the user's answers), Somerville (no portal row ids, substitute flag).
- Cases basel_aed_groups, dublin_bike_tour, somerville_pb_budget; 1 repeat per GPU (2 runs per task).
  These runs are not test-set results: the fixes were made after seeing the test set 2 failures.
- RESULT g0 Basel: PASS (name-only merge, 35 places, gap 0). RESULT g1 Basel: FAIL (data): the data agent merged
  within 100 m only (38 places); the independent check found 35 but compare() skipped the set because the model
  labelled it "private". Fixed in 6f62b94 (after these batches started: they keep the code they loaded, 990851b).
- RESULT Dublin g0, g1: PASS, PASS (rank 1.0, tour = reference). RESULT Somerville g0: PASS (no substitute flag).
- Follow-up e2e_postfix3_g0_20261024 (code 6f62b94, data cross-check covers "private" sets): basel_aed_groups,
  2 runs on GPU 0, same config.
- RESULT Somerville g1: PASS. Post-fix 2 overall: Basel 1/2 (the failure's cause fixed in 6f62b94), Dublin 2/2, Somerville 2/2.
- RESULT postfix3 g0 r0 Basel (6f62b94): FAIL (data): the cross-check ran but itself counted 40, and the data agent
  numbered same-name places ("…_1") instead of merging them (38). Fix d4ee5d3: repeated names in one set are questioned
  once against the data scope. postfix3 r1 stopped (it ran the old code); e2e_postfix4_g0_20261024 runs Basel 2x on d4ee5d3.
- RESULT postfix4 g0 r0 Basel (d4ee5d3): FAIL (data): the official GeoJSON download was a zip holding the .geojson,
  which the reader did not open; no other readable AED table. Fix 4e: JSON/GeoJSON inside archives. postfix4 r1
  stopped (old code); e2e_postfix5_g0_20261024 runs Basel 2x on the new code.
- RESULT postfix5 g0 r0 Basel (5bb038a): FAIL (data): the draft turned the user's "same name = one place" into "same
  location = one place"; the checker counted by coordinates (40) and the data agent chased it with a forbidden
  coordinate dedupe. Fix: data agents and checker also get the user's own words, which win. postfix5 r1 stopped;
  e2e_postfix6_g0_20261024 runs Basel 3x on the new code.
- RESULT postfix6 g0 r0 Basel (a67d9a1): data PASS (35 places, the user's words reached the agents), model PASS,
  delivery FAIL: SA's grouping 98% above the reference and the exchange polish found no move (group sizes were <=/>=
  rows, the polish paired moves through equalities only). Fix de85c44; the polish now reaches the reference on that
  very delivery. g0 r1 stopped (old code); e2e_postfix7_g0_20261024 runs Basel 2x on de85c44. g1 r0 finishes on a67d9a1.
- RESULT postfix6 g1 r0 Basel (a67d9a1): PASS. g1 r1 stopped; e2e_postfix7_g1_20261024 runs Basel 1x on de85c44.
- RESULT postfix7 (de85c44): g0 r0 PASS (polish 52370.5 -> 26503.5 = reference), g1 r0 PASS.
