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
