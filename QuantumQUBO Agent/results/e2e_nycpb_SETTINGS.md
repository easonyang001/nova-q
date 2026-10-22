# NYC participatory budgeting (new problem type: knapsack / budget selection), 2026-10-22
- Case nyc_pb_budget: NYC Open Data wwhr-5ven, Vote Year 2017, Council District 3, 15 ballot projects, budget $1,000,000.
  Truth (full enumeration): optimum 7,103 votes (7 projects, $980,000); city rule (fund by votes) = actual winners 5,381.
- Judge class Budget (acceptance/judge.py): data = votes exact + cost proportional + budget value present; model =
  30 random fundable sets rank like votes, over-budget sets infeasible; delivery within budget, gap <= 10 % of optimum.
  Self-test: 7/7 expected verdicts (correct, city-rule delivery, relaxed budget, wrong sign, altered votes, cost in
  thousands, over-budget delivery).
- Config = Melbourne v7 (Gemma doer+checker, thinking on, 1200 s, 64000 tokens, agent data prep, SA) with all fixes
  up to 2026-10-21 night. Tags e2e_nycpb_g0/g1, 2 repeats each.
- Stop rule (fixed before start): first 3 finished runs = x/3; then stop.

## v2 (restart, 2026-10-22): e2e_nycpb_g0/g1 killed before any run finished (not counted)
- Cause: system rule "areas identified only by codes" forced prep.in_area on 'Borough Code' although the plan never
  used that column (it kept Council District = 3, as the user said) -> the correct plan was rejected.
- Fix (generic): the rule applies only if the adapter code / plan references the code column; situation text and
  adapter prompts say "ONLY IF". Test added; 181 tests pass. Same config otherwise. Tags e2e_nycpb2_g0/g1.
- RESULT 1: e2e_nycpb2_g1 r0 PASS (15 projects all mapped; 7 projects, $980,000, 7,103 votes = exact optimum;
  26.0 min; model stage 3.3 min, no thinking timeout).
- RESULT 2: e2e_nycpb2_g0 r0 PASS (37.3 min). Score so far 2/2.
- RESULT 3: e2e_nycpb2_g1 r1 PASS (19.7 min). FINAL (stop rule): NYC budgeting 3/3; remaining g0 r1 stopped.
