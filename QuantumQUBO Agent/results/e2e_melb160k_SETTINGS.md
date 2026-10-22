# Melbourne bike tour, 160k context (2026-10-21)
- Gemma-4-31B-it FP8 on each L40S, vLLM max_model_len 163840, FP8 KV cache, max 8 seqs
- doer + checker Gemma, thinking on, LLM_MAX_COMPLETION_TOKENS=64000, LLM_REQUEST_TIMEOUT_S=3600
- agent data prep, SA backend (+ exchange polish), 4 repeats per GPU
- Stop rule (fixed before start): the first 3 runs to finish (both GPUs, by finish time) are reported as x/3; then stop.
  >=1 pass -> Melbourne replaces Berlin in the poster table; 0 -> keep Berlin 0/3.

## v6 (2026-10-21 15:5x) — supersedes the 160k batch above
- same servers (160k context); LLM_REQUEST_TIMEOUT_S=1200 (timeout -> one no-thinking retry), LLM_MAX_COMPLETION_TOKENS=64000
- fix: data agent no longer loops on "plan before two looks" (second time it explores the tables itself)
- tags e2e_melb6_g0/g1. Stop rule unchanged: first 3 finished runs = x/3. If no Melbourne pass by 03:00, switch to basel_aed_groups.

## 2026-10-21 16:55 UTC — prompt audit (applies to batches started AFTER this time; v6 runs use the old prompts)
- clarifier: decisions in user terms only (no x_ij / encodings / aux-variable counts); method choices never recorded
  as assumptions; a landmark that only defines the area is never added as a stop/start unless the user says so.
- modeler: spec's variable notation / formula_sketch / size_estimate declared NOT binding; binding = decisions,
  objectives, HARD constraint names, numbers, data scope, answers. Repair may change the formulation.
- req_checker: checks requirements, not the spec's formula sketches.
- adapter: no "aggregate to districts" (contradicted "keep every id"); metadata request moved before the output
  instruction and phrased per mode (JSON "metadata" vs ADAPTER_METADATA).
- dataagent: base prompt's "Return JSON only: <plan>" relabelled as DATA PLAN FORMAT (it contradicted "explore first").
- tests: 178 passed.

## v7 (2026-10-21 ~17:00 UTC) — user: "直接重跑"; supersedes v6 (v6 killed mid-run, not counted)
- identical env to v6 (timeout 1200 s, 64000 tokens, Gemma doer+checker thinking on, agent data prep, SA), NEW prompts
- tags e2e_melb7_g0/g1. Stop rule unchanged: first 3 finished runs = x/3; no Melbourne pass by 03:00 -> basel_aed_groups.
- 18:10 UTC (code only, running v7 processes unaffected): data plan gained op "distance_to" and derived tools
  distance_to / road_matrix. v7 g1's model declared a distance-to-station param; the plan agent had no tool for it
  and reported a gap (prep.distance_to existed only in code mode). Test added; 179 tests pass.
- RESULT 1 (18:3x UTC): e2e_melb7_g1 r0 PASS (data/model/delivery all ok; 11 docks = truth; route 5311 m = reference,
  baseline 6555 m; 71.4 min, model stage 27 min incl. one 20-min thinking timeout). Melbourne replaces Berlin per stop rule.
- 18:5x UTC: g0 r0 requirement check disagreed (rank 0.10): its data plan mapped set "positions" to the station-id
  column; model reads positions in list order, checker sorted ids numerically -> different tours. Generic fixes
  (next batch): plan may not map two sets to one column; sets with source user/mock may not come from a column;
  new {"index_of": "<set>"} sizes such a set from the data; checker prompt: set order = list order. 180 tests pass.
  The g1 pass is genuine (positions P0..P10).
- RESULT 2 (19:3x UTC): e2e_melb7_g0 r0 PASS after one cross-stage repair (requirement-check disagreement ->
  model repair -> data re-prepared with positions 0..10; repair flagged "formula changed, needs human review",
  accepted in batch mode). Route 5311 m = reference. 113.9 min. Score so far 2/2.
- RESULT 3 (20:0x UTC): e2e_melb7_g1 r1 PASS (route 5311 m = reference; no repair). Same spec and model code as r0
  (temperature 0); its data plan also used station ids as position ids, harmless here because this run's checker and
  the judge read positions in list order. FINAL (stop rule, first 3 finished): Melbourne 3/3. Batches stopped.
