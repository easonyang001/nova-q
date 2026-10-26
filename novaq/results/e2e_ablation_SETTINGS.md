# Gate ablation (2026-10-24), frozen code (hash in each run's manifest.json)
- Question: what do the checker agents and repair loop change at delivery time? Same tasks, same code, same LLM
  (Gemma-4-31B-it doer, temperature 0.7, thinking on, 1200 s, 64000 tokens, agent data prep, SA), two arms:
  - full_rag (GPU 0): every gate, repair on;
  - pipeline (GPU 1): no checker agent (gates 2 and 4 off), no repair, no modelling patterns. Gate 1 (schema and
    brute-force model check inside the Modeler) and the data cross-check stay, because they are part of building.
- Tasks: the six held-out tasks of test sets 1 and 2 (sg_hawker_groups, paris_velib_tour, cambridge_pb_budget,
  basel_aed_groups, dublin_bike_tour, somerville_pb_budget); 1 run per task per arm.
- Measures (fixed before start): judged passes; and "wrong deliveries" = runs that delivered a plan the judge rejects.
  A gate that stops a run is counted as no delivery, not as wrong.

# Cross-family end to end (2026-10-25), same frozen code
- Question: does a checker from a second LLM family help in full runs, not only on injected faults?
- Doer Gemma-4-31B-it (GPU 0, port 8000); checker Qwen3.6-27B (dense, FP8) served on GPU 1 (port 8001) in place of the
  second Gemma server; otherwise the same settings and the same six held-out tasks, 1 run each.
- Compared with the full_rag arm of the gate ablation above (Gemma doer + Gemma checker, same tasks and code).

## RESULT (gate ablation, 2026-10-24/25)
| Arm | Passed | Wrong deliveries | Median minutes |
|---|:---:|:---:|:---:|
| full_rag (all gates, repair) | 6/6 | 0 | 33 |
| pipeline (no checker, no repair) | 6/6 | 0 | 35 |
On the fixed code these six tasks no longer produce a wrong model or data, so the checker had nothing to stop (one
requirement-check issue raised on Dublin, no repair needed). The ablation shows no cost of the gates in pass rate or
time, but it cannot show their benefit; that evidence remains the injected-fault study (97/99 data, 66/78 model).

## RESULT (cross-family end to end, 2026-10-25)
| Doer + checker | Passed | Failures |
|---|:---:|---|
| Gemma + Gemma (ablation full_rag arm) | 6/6 | |
| Gemma + Qwen3.6-27B | 4/6 | Basel: data agent deduplicated by coordinates (40 places); the Qwen checker counted 40 as well, so the cross-check agreed with the wrong data. Singapore: a tool-argument mismatch (merge_same_place(name_col=...)) unrelated to the checker |
In full runs a second-family checker did not raise the pass rate on these six tasks; the cross-family benefit stands
only for injected faults (98% vs 73%). The tool-argument mismatch is fixed after this run.
