# Evaluation plan: what is still missing and how to run it

Experiments that need the GPUs, the local LLM servers, the annealer services or another person. Everything the code
side of them needs already exists unless stated. Order matters: A1 can change what the paper claims.

Common rules (from `acceptance/PREREGISTRATION.md`): freeze the code and record its hash before building reference
answers; every agent samples at `LLM_TEMPERATURE=0.7` so repeats are independent samples; three runs per task, the
first three finished runs count; report every run, failures included; run `scripts/preflight.py --tag <tag>` first and
start only when all checks pass; summarise with `scripts/report_runs.py`.

## A1. Direct-LLM baseline (R0)

The `direct` arm exists (`scripts/e2e_direct.py`, `--arms direct` in `run_e2e_batch.py`). It gets the request and the
user's fixed answers, searches the web itself, writes one module and has the Modeling Agent's attempt count, with the
same backend and the same judge. It has no Clarifier, toolkit, checks or repair. The saved results contain only the
R1–R3 batches.

```bash
export LLM_TEMPERATURE=0.7
python scripts/preflight.py --tag e2e_r0_YYYYMMDD
python scripts/run_e2e_batch.py --tag e2e_r0_YYYYMMDD --repeats 3 --arms direct full \
    --truth results/acceptance_truth/truth_20261011.json results/acceptance_truth/truth_intl_20261011.json \
            results/acceptance_truth/truth_melbourne_20261021.json results/acceptance_truth/truth_nyc_pb_20261022.json \
            results/acceptance_truth/truth_test_20261022.json results/acceptance_truth/truth_test2_20261024.json \
            results/acceptance_truth/truth_basel_20261021.json
```

Run `full` in the same batch, same temperature and same code, so the comparison is paired. Report R0 against R3 per
task and overall with Wilson intervals; report the data, model and delivery criteria separately (H2: does R1 beat R0 on
data?). Nothing here is a result until it has run.

## A2. Ambiguous real requests (Clarifier)

The evaluation answers the Clarifier from fixed scripts. Collect 20–30 requests from people who did not build the
system (one sentence each, no answers prepared), let the Clarifier ask, have a person answer in their own words, and
record: questions asked, assumptions the system made, and whether the confirmed specification matches what the person
meant (the person rates each assumption: right, wrong, harmless). The number to report is the share of confirmed
specifications with at least one wrong assumption, because no later gate can catch it.

## A3. Independent repeats and the headline number

Dev tasks and test set 1 ran at temperature 0; their repeats are close to copies. Re-run them at 0.7 on frozen code
(R3 only is enough) and use only frozen-code, temperature-0.7 results in the abstract and highlights. Post-fix runs stay
in their own table.

## A4. A test set with unseen problem structures

All tasks so far are coverage, tour, balanced partition, p-median and knapsack, and the modelling-pattern library was
written around them. Add at least two structures that are not in `knowledge/*.jsonl` (the pre-registration planned set
cover; assignment with capacities or scheduling are candidates). Build the reference answers after freezing the code
(record the hash), run once, report every run.

## A5. Judge independence and tolerance

1. A second person builds the reference answer for at least one task from the raw files, without seeing the existing
   `truth.py`. Compare entity sets, totals and the optimum; list the differences.
2. Report delivery quality at three tolerances (exact optimum, within 1%, within 10%); a 10% criterion can hide real gaps (the
   README reports a delivery on Singapore's hawker zones 5.8% worse than the cuOpt optimum).
3. Rank agreement (Spearman) is measured on random valid plans; add plans near the optimum (one or two swaps from it).

## A6. Value of the gates, measured on faults nobody tuned for

The fault-injection study used faults chosen by the authors. Have another person write 15–20 realistic faults (a data
error and a modelling error each) into otherwise correct runs, apply them, and report catch rate and false rejections.
Also report how often the doer and the checker make the same wrong count (the Basel/Qwen case), since correlated errors
are what a same-family checker cannot see.

## A7. Fair solver comparison

`scripts/experiments/solver_tuning.py` (equal trials, solver parameters and penalty multiplier tuned on separate
instances, scored on the original objective). Build the instances with `qaoa_instances.py` first (they are not
versioned). Hardware budgets: Fujitsu DA 300 minutes per month, D-Wave QPU access time; `--allow-real` prints the
planned call counts before sending anything. CGA exposes only its time limit, so its search is the penalty multiplier.
Then regenerate the comparison table from `summary.md` and replace the energy-gap table, which is not comparable across
penalty weights.

## A8. Sensitivity of thresholds

`docs/constants_inventory.md` lists 199 hard-coded values. Pick those that decide a gate or the data kept (Spearman
cut-off, quality tolerance, 150 m merge radius, total tolerance, number of random valid plans, `MAX_STEPS`,
`MIN_EXPLORE`), vary each over a sensible range on the frozen tasks, and report how the pass count moves.

## Recording what ran

Every batch should carry its `manifest.json` (code hash, model, temperature, backend) in the repository, and every run
its `backend`. For live APIs (bike stations) store the downloaded snapshot and its SHA-256 next to the run; without it
the run cannot be reproduced.
