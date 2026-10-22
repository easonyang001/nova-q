# NOVA-Q: Verified LLM Agents From Natural Language and Open Data to QAOA on CUDA-Q

**N**atural-language → **O**pen-data → **V**erified **A**gents for **Q**uantum.
A user describes a real planning problem in one sentence ("place 3 EV chargers in Da'an District, Taipei, to cover the
most residents"). Local LLM agents clarify it, find and prepare official open data, write a QUBO model, and solve it on
annealers and on QAOA with NVIDIA CUDA-Q. Independent checks run around every solve, and a frozen judge scores the
delivered answer against ground truth built from the raw files.

Chia-Chen Yang¹, Chia-Ho Ou¹,² · ¹National Pingtung University, Taiwan · ²Tohoku University, Japan ·
cho@mail.nptu.edu.tw

> **Built on [QuantumQUBO Agent](https://github.com/QuitttCat/QuantumQUBOAgent)** by Niloy Kumar Mondal and Md Rizwan
> Parvez (ICML 2026 Workshop, arXiv:2609.10629), MIT License. Its formulation agents are the starting point of our
> Modeler; everything else listed below is new. The earlier README of this repository, including the original
> project's README unchanged, is in [docs/README_development_log.md](docs/README_development_log.md).

Poster (NVIDIA GTC 2027): [`QuantumQUBO Agent/poster/gtc2027/poster.pdf`](QuantumQUBO%20Agent/poster/gtc2027/poster.pdf)

---

## How it works

```
one sentence ─► Clarifier ─► confirmed spec ─► Modeler ─► QUBO model ──────────────┐
                                     │                    │                         │
                                     │          ② blind requirement checker         │
                                     ▼                    ▼                         ▼
                        open-data search (10 portals) ─► data agent ─► data ─► ④ solvers ─► delivery
                                                  ① schema checks  ③ data cross-check        ┊
                         failed check → evidence goes back to the Modeler or the data agent  ⑤ judge
                                                                                    (evaluation only)
```

| Gate | What it checks | Who checks |
|---|---|---|
| ① data schema & model | units, period, area, keys of the prepared data; the model on a small instance by brute force | code |
| ② requirements | random valid plans must be feasible and ranked like the user's objective; mutated plans must be rejected | an LLM checker that never sees the model |
| ③ data cross-check | entity counts and totals recomputed from the raw tables | an LLM checker that never sees the preparation code |
| ④ solution | the delivered plan against the original constraints and a simple baseline | code |
| ⑤ acceptance judge | data, model and delivery against hand-built ground truth | frozen code, never seen by the system |

A plain-language description of every agent (Traditional Chinese): [docs/agents_overview_zh.md](QuantumQUBO%20Agent/docs/agents_overview_zh.md).

## Results

**Real cities, official open data** (3 runs per task; Gemma-4-31B-it as doer and checker, local SA solver):

| City & task | Problem type | Official source | Instance | Passed |
|---|---|---|---|---|
| Taipei: EV chargers | max coverage | village census | 53 villages, 0.28 M people | 3/3 |
| Taipei: bike tour | tour (TSP) | YouBike live API | 18 stations | 2/3 |
| Taipei: AED zones | balanced groups | AED registry | 48 sites | 3/3 |
| London: ambulances | p-median | ONS Census 2021 | 27 areas, 0.21 M people | 1/3 |
| Melbourne: bike tour | tour (TSP) | dock registry (data.gov.au) | 11 docks | 3/3 |
| New York: budget | knapsack | NYC Open Data ballot results | 15 projects, $1 M | 3/3 |
| **Passed** | | | | **15/18** |

New York, Council District 3, 2017: the delivered plan funds 7 projects with 7,103 votes for $980,000, the exact
optimum; the city's vote-order rule funded 4 projects with 5,381 votes for the same $1 M budget (+32%).

**Test set (code frozen before the reference answers were built):** Singapore hawker centres (balanced groups),
Paris Vélib' tour, Cambridge (MA) participatory budget. Running; results will be added here.

**Verification gates, injected harmful faults:** caught 97/99 data faults and 66/78 model faults; a checker from a
second LLM family catches 98% of the data faults (same family: 73%); no false rejections on 300 valid decisions.

**Solvers on the same verified QUBOs** (39 instances, 9–27 qubits, optimum known by enumeration; energy gap
0 = optimum, 1 = random guess, median):

| SQA | SA | D-Wave QPU | QAOA p=3 (CUDA-Q) | CGA |
|---|---|---|---|---|
| 0.001 | 0.009 | 0.028 | 0.16 | 0.66 |

QAOA samples the optimum on every instance but trails the annealers, because penalties cut its feasible fraction.

![Energy gap to the optimum](QuantumQUBO%20Agent/results/experiments/qaoa_cudaq_20261020/figures/energy_rep.png)

## NVIDIA stack

- LLM agents: vLLM on 2 × NVIDIA L40S (48 GB), FP8 weights: Gemma-4-31B-it and Qwen3.6-35B-A3B (Apache-2.0). No cloud API.
- QAOA: CUDA-Q 0.14, `nvidia` state-vector target. One L40S holds up to 32 qubits; QAOA cost evaluation at 21 qubits
  takes 0.079 s on the GPU vs 31.5 s on the CPU (400×). `nvidia-mqpu` on 2 GPUs gives 1.65–1.94× throughput.
- Next: H200 / B200 (33–34 qubits per GPU, `nvidia-mgpu` over NVLink) and larger agents.

## Run it

```bash
cd "QuantumQUBO Agent"
cp .env.example .env            # LLM endpoint; optional D-Wave token and CGA server address
pip install -r requirements.txt

# local LLM (separate vLLM environment), one server per GPU
bash scripts/serve_vllm.sh

# guided web UI: describe -> answer questions -> model & data -> solve -> results  (http://localhost:7860)
python scripts/ui.py

# end-to-end batch with the independent judge
python scripts/run_e2e_batch.py --tag my_batch --repeats 3 --cases nyc_pb_budget --arms full_rag \
    --truth results/acceptance_truth/truth_nyc_pb_20261022.json
python scripts/rejudge.py my_batch

# tests
python -m pytest -q tests
```

QAOA runs in its own CUDA-Q environment (`CUDAQ_PYTHON`, see `qubo_auto/annealing/backends.py`). The solver comparison
is `scripts/experiments/qaoa_instances.py` → `qaoa_compare.py` → `qaoa_figures.py`.

## Repository layout

| Path (inside `QuantumQUBO Agent/`) | Contents |
|---|---|
| `qubo_auto/annealing/` | agents, data tools (`prep/`, `dataplan.py`, `dataagent.py`), checkers, encodings, solver back-ends, workflow and repair |
| `qubo_auto/prompts/annealing/` | prompts of every agent |
| `acceptance/` | judge, ground-truth builders, task list, pre-registration |
| `scripts/` | UI, batch runner, re-judging; `experiments/` for the solver comparison and poster figures |
| `knowledge/` | general modelling patterns retrieved by the Modeler |
| `poster/gtc2027/` | poster source (LaTeX) and figures |
| `results/` | settings notes, reference answers, per-run judge records, solver-comparison data (raw downloads, LLM transcripts and caches stay local) |
| `tests/` | unit tests |

The original QuantumQUBO Agent front end (`qubo_auto/agents`, `benchmarks/`, `deliverables/`) is kept as released.

## License and credits

MIT License (see [LICENSE](LICENSE)); the original QuantumQUBO Agent copyright notice is kept. Please cite the original
work ([CITATION.cff](CITATION.cff)) when using this code. Benchmark instances from
[QOBLIB](https://github.com/ZIB-AOPT/QOBLIB) (CC BY 4.0) are used unmodified as data. Open data belongs to its
publishers (Taipei City Government, UK ONS, City of Melbourne, NYC Open Data, data.gov.sg, Ville de Paris, City of
Cambridge) under their licences.
