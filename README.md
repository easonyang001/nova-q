<div align="center">

<img src="assets/nova-q-banner.jpg" alt="NOVA-Q: Verified LLM Agents From Natural Language and Open Data to QAOA on CUDA-Q" width="80%">

**N**atural-language → **O**pen-data → **V**erified **A**gents for **Q**uantum

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg)
![CUDA-Q 0.14](https://img.shields.io/badge/CUDA--Q-0.14-76B900.svg)
![NVIDIA L40S](https://img.shields.io/badge/GPU-2%C3%97%20NVIDIA%20L40S-76B900.svg)

Chia-Chen Yang · National Pingtung University, Taiwan

[Agent guide (繁體中文)](novaq/docs/agents_overview_zh.md)

</div>

---

## Contents

1. [Abstract](#abstract)
2. [Highlights](#highlights)
3. [Motivation](#motivation)
4. [A complete run, step by step](#a-complete-run-step-by-step)
5. [System design](#system-design)
6. [Verification design](#verification-design)
7. [Evaluation protocol](#evaluation-protocol)
8. [Results](#results)
9. [Hardware and software](#hardware-and-software)
10. [Limitations](#limitations)
11. [Installation and usage](#installation-and-usage)
12. [Repository structure](#repository-structure)
13. [Citation, license and acknowledgements](#citation)

---

## Abstract

Turning a real planning request into a QUBO requires finding and cleaning data, choosing a formulation and tuning
penalties: expertise that most users do not have. NOVA-Q takes a single sentence, such as *"place 3 EV chargers in
Da'an District, Taipei, to cover the most residents"*, and lets local LLM agents clarify the request, retrieve and
prepare official open data, formulate a QUBO, and solve it on simulated, quantum and GPU annealers and on QAOA with
NVIDIA CUDA-Q. Because LLM output can look right and still be wrong, every solve is surrounded by independent checks:
three gates before the solver and one after, run by checker agents that never see the work they check. A frozen
acceptance judge scores each delivery against ground truth built directly from the raw public files.

## Highlights

- **End to end on real cities.** 15 of 18 runs across Taipei, London, Melbourne and New York pass the independent
  judge, using official open data and local models only (no cloud LLM). On a held-out test set built after freezing
  the code (Singapore, Paris, Cambridge MA), 8 of 9 pass.
- **Verification that catches real faults.** Injected harmful faults are caught 97/99 (data) and 66/78 (model); a
  checker from a second LLM family catches 98% of data faults versus 73% for the doer's own family. No valid plan was
  wrongly rejected (300/300).
- **Measurable impact.** On New York's 2017 participatory budget (Council District 3), the delivered plan funds
  7,103 votes for the same $1 M budget, versus 5,381 under the city's vote-order rule (+32%). It equals the exact
  optimum.
- **One verified QUBO, five solvers.** Simulated annealing, simulated quantum annealing, the D-Wave Advantage QPU, a
  GPU annealer and QAOA on CUDA-Q, compared on 39 instances with known optima.

## Motivation

Large language models can write a QUBO from a problem statement, but on real requests three things go wrong:

1. **The request is underspecified.** "Cover the most residents" leaves open the radius, whether a resident covered
   twice counts twice, and which administrative level the data use.
2. **The data are not ready.** Official tables mix periods, include subtotal rows, use area codes instead of names,
   store coordinates in local projections, or list one place several times.
3. **The model can be silently wrong.** A relaxed bound ("at most 200" instead of "exactly 5"), a flipped objective or
   a double-counted term still produces a QUBO that solves and returns a plausible answer.

Existing systems avoid these problems rather than detect them: they assume a correct intermediate model, learn from
test cases the user writes, check the translation only on standard textbook problems, or start from a hand-built model.
NOVA-Q targets the realistic setting (a vague sentence, raw public data, no user-written tests) and makes verification,
not generation, the central design element.

## A complete run, step by step

The New York participatory-budget task, as it ran in the evaluation (about 20–37 minutes per run on one L40S):

| Step | What happens | Output in this run |
|---|---|---|
| 1. Request | *"District 3's 2017 participatory budget is $1 M; choose the projects to fund, total cost at most $1 M, as many votes as possible. Use NYC Open Data."* | |
| 2. Clarifier | asks only questions that change the model; decides the rest as explicit assumptions the user confirms | asked about extra hard limits and dependencies between projects; assumed no category quotas and votes as the only value |
| 3. Modeler | writes a typed binary model: variables, objective, hard constraints, the data it needs, and a map from each user requirement to its constraints | one binary variable per project, one budget inequality (0-1 knapsack) |
| 4. Gate ② | a checker agent, given only the specification, writes its own evaluator; random valid plans must be feasible in the model and ranked alike, mutated plans must be infeasible | 20/20 valid plans kept, 13/13 over-budget plans rejected, rank agreement 1.0 |
| 5. Data search | searches a catalogue of 10 national portals and the open-data platform APIs; ranks candidates by their actual columns | `Participatory Budgeting Projects`, data.cityofnewyork.us |
| 6. Data agent | looks at the table with read-only tools, then writes a preparation plan made of tested steps | kept Vote Year 2017 and Council District 3: 15 projects |
| 7. Gates ① and ③ | the prepared data must match the declared units, period and area; a second checker recomputes counts and totals from the raw table | consistent |
| 8. Solve | the model is compiled to a QUBO (penalties and slack variables chosen and tuned by code) and solved | 7 projects, $980,000, 7,103 votes |
| 9. Gate ④ | the delivered plan is checked against the original constraints and compared with a simple baseline | feasible, equal to the baseline's best |
| 10. Judge ⑤ | the frozen judge compares data, model and delivery with ground truth enumerated from the raw file | pass (the exact optimum) |

If a gate fails, its evidence (which requirement, which plan, which numbers) goes back to the stage that can fix it:
the data agent for data problems and the Modeler for model problems. A repair may change the formulation but never the
confirmed requirements, the objective direction or a user setting.

## System design

### Agents

| Agent | Role | Sees | Never sees |
|---|---|---|---|
| Clarifier | turns the request into a confirmed specification: decisions, objectives, hard and soft constraints, data scope, assumptions; may look up laws and standards on the web | the request, the user's answers | any formulation |
| Pattern retrieval | picks up to three general modelling patterns (e.g. cardinality, one-hot assignment, position-based tour, coverage counted once) | the specification | |
| Modeler | writes the binary model, the data schema and the requirement map | the specification, the patterns | the data checker |
| Requirement checker (gate ②) | writes an independent evaluator and a sampler of valid plans | the specification, the data schema | the model |
| Data scout | proposes searches; the system runs them and downloads candidates | the specification, the data schema | |
| Data agent | explores the tables, then writes a preparation plan from tested tools | table profiles, tool results | |
| Data checker (gate ③) | recomputes expected counts and totals from the raw tables | the specification, the raw tables | the preparation plan |
| Acceptance judge (gate ⑤) | scores data, model and delivery against ground truth | the system's outputs | is never seen by the system |

All agents run on local models served by vLLM. Doer and checker can be assigned to different model families.

### Open data

The data layer searches a pre-built catalogue of 10 national portals (Taiwan, Japan, Australia, Ireland, UK, the
Netherlands, Berlin, Germany, Canada, Switzerland) plus CKAN, Socrata, Opendatasoft, ArcGIS, EU and Nomis APIs and
OpenStreetMap. Candidates are ranked by a preview of their real columns, official sources are preferred over
crowd-sourced ones, and the latest period of a dataset is kept.

Preparation is done by **tested tools, not free-form code**. The data agent first inspects columns, distinct values,
sample rows, key overlaps and which area code lies in which named area; then it writes a plan of steps such as
`latest_period`, `keep_category`, `keep_level`, `to_wgs84`, `in_area` (real boundaries), `within_radius`,
`merge_same_place`, `reduce_points` and `join`. Each step runs on the full table and its result is shown before the
next decision. Public values that are missing are reported as a data gap; they are never filled with defaults.

### From model to QUBO

The Modeler describes only variables, objective and hard constraints. Code compiles them into a QUBO: equality and
inequality constraints become penalty terms with binary or unary slack encodings, implications and exclusions become
pairwise terms, and penalty weights are chosen and tuned by the solver loop rather than by the LLM. The same compiled
QUBO is passed unchanged to every solver.

### Solvers

| Solver | Implementation |
|---|---|
| SA | dwave-samplers simulated annealing, followed by constraint-preserving exchange moves |
| SQA | OpenJij simulated quantum annealing |
| D-Wave QPU | D-Wave Advantage, 1,000 reads |
| GPU annealer | a commercial GPU annealing service |
| QAOA | CUDA-Q 0.14 on the `nvidia` state-vector target, p = 3, COBYLA, 3 restarts, 10,000 shots; `nvidia-mqpu` for parallel evaluations |

## Verification design

Three principles guide the gates:

- **Independence.** A checker never sees the artefact it checks. The requirement checker writes its own evaluator
  from the specification alone; the data checker recomputes the expected data from the raw tables alone.
- **Agreement, not authority.** A disagreement is evidence, not a verdict: the checker can be wrong too. The evidence
  is returned to the stage that can fix it, and the repaired version must pass all gates again.
- **Diversity.** A checker from a different model family catches more faults than one from the same family, because
  the two make different mistakes (98% versus 73% on injected data faults).

The fault-injection study inserts harmful faults one at a time into otherwise correct runs: missing or duplicated
entities, swapped coordinates, values off by a factor of 10, total rows mixed in, points outside the area, dropped
constraints, flipped objectives, dropped pairwise terms, counts off by one and blocked entities.

## Evaluation protocol

- **Tasks.** Each task is a one-sentence request with fixed answers to the Clarifier's questions, so runs are
  comparable. Every run starts empty: no cached model, data or LLM answers.
- **Ground truth.** Built by hand-written code directly from the official raw files, before the runs, and never visible
  to the system.
- **Pass criteria.** A run passes only if all three hold:
  1. *data*: every entity maps one-to-one to a real entity, nothing is missing or duplicated, coordinates and measured
     values match;
  2. *model*: 30 random valid plans are feasible and rank like the true metric (Spearman ≥ 0.9), and invalid plans are
     infeasible;
  3. *delivery*: the plan satisfies the original request and is within 10% of a reference computed from the raw data.
- **Repeats.** Three runs per task; the first three finished runs count. Stop rules are written down before a batch
  starts ([acceptance/PREREGISTRATION.md](novaq/acceptance/PREREGISTRATION.md)).
- **Held-out test set.** The pipeline code is frozen (hash recorded) before the reference answers of new tasks are
  built, and is not changed while they run.

## Results

### Real cities, official open data

Three runs per task; Gemma-4-31B-it as doer and checker; simulated annealing as the solver.

| City and task | Problem type | Official source | Instance | Passed |
|---|---|---|---|:---:|
| Taipei: EV chargers | maximum coverage | village census | 53 villages, 0.28 M residents | 3/3 |
| Taipei: bike-station tour | travelling salesman | YouBike live API | 18 stations | 2/3 |
| Taipei: AED inspection zones | balanced partition | AED registry | 48 sites | 3/3 |
| London: ambulance stations | p-median | ONS Census 2021 | 27 areas, 0.21 M residents | 1/3 |
| Melbourne: bike-station tour | travelling salesman | City of Melbourne dock registry | 11 docks | 3/3 |
| New York: participatory budget | knapsack | NYC Open Data ballot results | 15 projects, $1 M | 3/3 |
| **Total** | | | | **15/18** |

With a Qwen3.6-35B-A3B doer, the first four tasks pass 5/12.

### Held-out test set

The pipeline code was frozen (hash recorded) before the reference answers of three new tasks were built, and was not
changed while they ran. Same configuration as above; the first three finished runs per task count.

| City and task | Problem type | Official source | Instance | Passed |
|---|---|---|---|:---:|
| Singapore: hawker-centre zones | balanced partition | data.gov.sg Hawker Centres | 12 centres, 3 zones | 2/3 |
| Paris: Vélib' station tour | travelling salesman | opendata.paris.fr | 10 stations | 3/3 |
| Cambridge, MA: participatory budget | knapsack | data.cambridgema.gov ballot | 16 projects, $525k | 3/3 |
| **Total** | | | | **8/9** |

The failed run found no official source and fell back to OpenStreetMap food courts (49 places instead of 12 hawker
centres); the judge caught it. All three Cambridge runs return the exact optimum (24,769 votes, versus 22,004 under the
vote-order rule). Run-by-run notes: [results/e2e_test_RESULTS.md](novaq/results/e2e_test_RESULTS.md).

### Verification gates

Injected harmful faults, 9–18 per type; doer Qwen3.6-35B-A3B, checker Gemma-4-31B-it.

| Gate | Data faults caught | Model faults caught |
|---|:---:|:---:|
| any gate | 97/99 | 66/78 |

No false rejections in 15 verified runs (300/300 valid decisions accepted).

### Solvers on the same verified QUBOs

39 instances from four agent-built models (3 instances per size, k-means on random 80% subsets of the real points),
9–27 qubits, optimum known by full enumeration. The energy gap is
(E − E<sub>opt</sub>) / (Ē<sub>random</sub> − E<sub>opt</sub>): 0 is the optimum and 1 a random guess.

| Solver | SQA | SA | D-Wave QPU | QAOA p=3 (CUDA-Q) | GPU annealer |
|---|:---:|:---:|:---:|:---:|:---:|
| Median energy gap | 0.001 | 0.009 | 0.028 | 0.16 | 0.66 |
| Instances where the optimum was sampled | 39/39 | 39/39 | 37/39 | 39/39 | 39/39 |

QAOA samples the optimum on every instance but trails the annealers, because penalty terms reduce its feasible
fraction.

<p align="center">
  <img src="novaq/results/experiments/qaoa_cudaq_20261020/figures/energy_rep.png" width="900"
       alt="Energy gap to the optimum per solver and problem">
</p>

## Hardware and software

| Component | Setup |
|---|---|
| LLM agents | vLLM on 2 × NVIDIA L40S (48 GB), FP8 weights: Gemma-4-31B-it and Qwen3.6-35B-A3B (Apache-2.0); one server per GPU, 160k-token context |
| QAOA | CUDA-Q 0.14, `nvidia` state-vector target (cuStateVec); up to 32 qubits on one L40S |
| GPU vs CPU | QAOA cost evaluation at 21 qubits: 0.079 s on one L40S vs 31.5 s on the CPU (400×) |
| Multi-GPU | `nvidia-mqpu` on 2 GPUs: 1.65–1.94× QAOA throughput; `nvidia-mgpu` over PCIe was slower than one GPU |

Planned: H200 and B200 GPUs (33–34-qubit state vectors per GPU, `nvidia-mgpu` over NVLink), larger agent models and
constraint-preserving QAOA mixers.

## Limitations

- **Scale.** The solver comparison stops at 27 qubits so that optima are known; one L40S simulates at most 32 qubits.
- **Larger instances.** City-scale versions of the tasks pass far less often; the main failures are in data
  preparation.
- **Repeats.** Runs use temperature 0, so the three repeats of a task often share the same specification and model;
  they are not fully independent samples.
- **Judge.** Ground truth and judge are written by the authors; they are frozen before the runs and published here for
  inspection.

## Installation and usage

```bash
git clone https://github.com/easonyang001/nova-q.git
cd nova-q/novaq
pip install -r requirements.txt
cp .env.example .env          # LLM endpoint; optional D-Wave token and GPU-annealer address
```

The local LLM runs in a separate vLLM environment (`scripts/serve_vllm.sh`); QAOA runs in a CUDA-Q environment whose
interpreter is set with `CUDAQ_PYTHON`.

```bash
# guided web interface: describe -> answer questions -> model and data -> solve -> results
python scripts/ui.py                                   # http://localhost:7860

# end-to-end batch with the independent judge
python scripts/run_e2e_batch.py --tag my_batch --repeats 3 --cases nyc_pb_budget --arms full_rag \
    --truth results/acceptance_truth/truth_nyc_pb_20261022.json
python scripts/rejudge.py my_batch

# solver comparison and figures
python scripts/experiments/qaoa_instances.py --out results/experiments/my_cmp/instances_rep --seeds 0 1 2
python scripts/experiments/qaoa_compare.py --dir results/experiments/my_cmp --instances instances_rep --layers 3
python scripts/experiments/qaoa_figures.py --dir results/experiments/my_cmp --results compare.jsonl

# unit tests
python -m pytest -q tests
```

## Repository structure

| Path (under `novaq/`) | Contents |
|---|---|
| `qubo_auto/annealing/` | agents, data tools (`prep/`, `dataplan.py`, `dataagent.py`), checkers (`reqcheck.py`, `datacheck.py`), QUBO encodings, solver back-ends, workflow and repair |
| `qubo_auto/prompts/annealing/` | agent prompts |
| `acceptance/` | acceptance judge, ground-truth builders, task definitions, pre-registration |
| `scripts/` | web interface, batch runner, re-judging; `experiments/` for the solver study and figures |
| `knowledge/` | general modelling patterns available to the Modeler |
| `poster/gtc2027/figures/` | figures and the scripts' outputs used on the poster |
| `docs/` | agent guide and experiment notes |
| `results/` | run settings, reference answers, per-run judge records, solver-comparison data |
| `tests/` | unit tests |
| `legacy/` | benchmark suite and reports of the earlier text-to-QUBO front end (not used by NOVA-Q) |

Raw downloads, LLM transcripts and caches are not versioned.

## Citation

```bibtex
@misc{yang2027novaq,
  title        = {{NOVA-Q}: Verified {LLM} Agents From Natural Language and Open Data to {QAOA} on {CUDA-Q}},
  author       = {Yang, Chia-Chen},
  year         = {2027},
  howpublished = {NVIDIA GTC 2027 poster},
  url          = {https://github.com/easonyang001/nova-q}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

NOVA-Q extends the formulation agents of QuantumQUBO Agent (N. K. Mondal and M. R. Parvez, ICML 2026 Workshop,
arXiv:2609.10629; MIT License). Benchmark instances from [QOBLIB](https://github.com/ZIB-AOPT/QOBLIB) (CC BY 4.0) are
used unmodified. Open data are provided by Taipei City Government, the UK Office for National Statistics, the City of
Melbourne, NYC Open Data, data.gov.sg, Ville de Paris and the City of Cambridge under their respective licences. The
project history is kept in [docs/README_development_log.md](docs/README_development_log.md).
