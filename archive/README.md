# Archive

Files that the current system does not use, kept for reference. Nothing here is imported or run by
the code in `QuantumQUBO Agent/`. They were moved here on 2026-10-04 to keep the working tree clean.

## `original_project/`: from the original QuantumQUBO Agent (Mondal & Parvez, MIT License)

| Item | What it is | Why archived |
|---|---|---|
| `index.html` | The original project website | Not part of the code |
| `ICML-Page-5-1.png`, `ICML-Page-5.drawio.pdf` | Architecture figure from the original paper | Referenced only by the original README section |
| `statistics.ipynb` | Original notebook for the paper's statistics | Analysis of the original benchmark CSVs |
| `deliverables/` | LaTeX/HTML/PDF formulations produced by the original authors' runs | Generated output |
| `results/` | Original benchmark-suite CSVs and logs (`benchmark_suite*.csv`, `benchmark_suite_logs/`, `no_retries_full_pipeline_42.csv`) | The original paper's results |
| `tests/test_pipeline.py`, `qubo_auto/benchmarks_oracle.py` | Original test file and the oracle only it used | The test never ran: it imports `is_symmetric`, which the original `brute_force.py` does not define |

To restore an item, move it back to the same relative place under `QuantumQUBO Agent/` (or the repo root for `index.html`).

## `our_runs/`: our own superseded outputs

| Item | What it is |
|---|---|
| `p2_smoke/` | Smoke run of the P2 experiment pipeline (4 instances, 1 seed). Superseded by `results/experiments/p2_sa_v1/` |

`scripts/ablation_scripts/` and `config_no_retries.yaml` were archived at first but moved back, because the README's documented commands use them.
