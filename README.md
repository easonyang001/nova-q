<div align="center">

# NOVA-Q

**Verified LLM Agents From Natural Language and Open Data to QAOA on CUDA-Q**

**N**atural-language → **O**pen-data → **V**erified **A**gents for **Q**uantum

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg)
![CUDA-Q 0.14](https://img.shields.io/badge/CUDA--Q-0.14-76B900.svg)
![NVIDIA L40S](https://img.shields.io/badge/GPU-2%C3%97%20NVIDIA%20L40S-76B900.svg)

Chia-Chen Yang<sup>1</sup>, Chia-Ho Ou<sup>1,2</sup><br>
<sup>1</sup>National Pingtung University, Taiwan  ·  <sup>2</sup>Tohoku University, Japan<br>
Contact: cho@mail.nptu.edu.tw

[Poster (NVIDIA GTC 2027)](QuantumQUBO%20Agent/poster/gtc2027/poster.pdf)

</div>

---

## Abstract

Turning a real planning request into a QUBO requires finding and cleaning data, choosing a formulation and tuning
penalties: expertise that most users do not have. NOVA-Q takes a single sentence, such as *"place 3 EV chargers in
Da'an District, Taipei, to cover the most residents"*, and lets local LLM agents clarify the request, retrieve and
prepare official open data, formulate a QUBO, and solve it on simulated, quantum and GPU annealers and on QAOA with
NVIDIA CUDA-Q. Because LLM output can look right and still be wrong, every solve is surrounded by independent checks:
three gates before the solver and one after, with checker agents that never see the work they check. A frozen
acceptance judge scores each delivery against ground truth built directly from the raw public files.

## Highlights

- **End to end on real cities.** 15 of 18 runs across Taipei, London, Melbourne and New York pass the independent judge,
  using official open data and local models only.
- **Verification that catches real faults.** Injected harmful faults are caught 97/99 (data) and 66/78 (model); a
  checker from a second LLM family catches 98% of data faults versus 73% for the same family.
- **Measurable impact.** On New York's 2017 participatory budget (Council District 3), the delivered plan funds
  7,103 votes for the same $1 M budget, versus 5,381 under the city's vote-order rule (+32%), and equals the exact optimum.
- **One verified QUBO, five solvers.** SA, SQA, D-Wave Advantage QPU, a GPU annealer and QAOA on CUDA-Q, compared on
  39 instances with known optima.

## Method

```
request ─► Clarifier ─► confirmed spec ─► Modeler ─► QUBO model ──────────────────────┐
                              │                      │                                  │
                              │            ② blind requirement checker                  │
                              ▼                      ▼                                  ▼
            open-data search (10 national portals) ─► data agent ─► instance ─► ④ solvers ─► delivery
                                          ① schema & model checks  ③ data cross-check        ┊
            a failed check returns its evidence to the Modeler or the data agent        ⑤ acceptance judge
                                                                                         (evaluation only)
```

| Gate | Checks | Performed by |
|:---:|---|---|
| ① | units, period, area and keys of the prepared data; the model by brute force on a small instance | code |
| ② | random valid plans are feasible and ranked like the user's objective; mutated plans are rejected | LLM checker without access to the model |
| ③ | entity counts and totals recomputed from the raw tables | LLM checker without access to the preparation code |
| ④ | the delivered plan against the original constraints and a simple baseline | code |
| ⑤ | data, model and delivery against hand-built ground truth | frozen judge, not visible to the system |

A plain-language guide to every agent (Traditional Chinese) is in
[docs/agents_overview_zh.md](QuantumQUBO%20Agent/docs/agents_overview_zh.md).

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

**Held-out test set.** The pipeline code was frozen before the reference answers of three new tasks were built:
Singapore hawker-centre zones (balanced partition), a Paris Vélib' station tour (travelling salesman) and the
Cambridge, MA participatory budget (knapsack). Results will be reported here when the runs finish.

### Solvers on the same verified QUBOs

39 instances from four agent-built models, 9–27 qubits, optimum known by full enumeration. The energy gap is
(E − E<sub>opt</sub>) / (Ē<sub>random</sub> − E<sub>opt</sub>): 0 is the optimum and 1 a random guess.

| Solver | SQA | SA | D-Wave QPU | QAOA p=3 (CUDA-Q) | GPU annealer |
|---|:---:|:---:|:---:|:---:|:---:|
| Median energy gap | 0.001 | 0.009 | 0.028 | 0.16 | 0.66 |

QAOA samples the optimum on every instance but trails the annealers, because penalty terms reduce its feasible fraction.

<p align="center">
  <img src="QuantumQUBO%20Agent/results/experiments/qaoa_cudaq_20261020/figures/energy_rep.png" width="900"
       alt="Energy gap to the optimum per solver and problem">
</p>

## Hardware and software

| Component | Setup |
|---|---|
| LLM agents | vLLM on 2 × NVIDIA L40S (48 GB), FP8 weights: Gemma-4-31B-it and Qwen3.6-35B-A3B (Apache-2.0); no cloud API |
| QAOA | CUDA-Q 0.14, `nvidia` state-vector target; up to 32 qubits on one L40S; 400× faster than CPU at 21 qubits |
| Multi-GPU | `nvidia-mqpu` on 2 GPUs: 1.65–1.94× QAOA throughput |
| Annealers | dwave-samplers SA, OpenJij SQA, D-Wave Advantage QPU, GPU annealer |

Planned: H200 and B200 GPUs (33–34-qubit state vectors per GPU, `nvidia-mgpu` over NVLink) and larger agent models.

## Installation

```bash
git clone https://github.com/easonyang001/nova-q.git
cd "nova-q/QuantumQUBO Agent"
pip install -r requirements.txt
cp .env.example .env          # LLM endpoint; optional D-Wave token and GPU-annealer address
```

The local LLM runs in a separate vLLM environment (`scripts/serve_vllm.sh`), and QAOA in a CUDA-Q environment whose
interpreter is set with `CUDAQ_PYTHON`.

## Usage

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

| Path (under `QuantumQUBO Agent/`) | Contents |
|---|---|
| `qubo_auto/annealing/` | agents, data tools, checkers, encodings, solver back-ends, workflow and repair |
| `qubo_auto/prompts/annealing/` | agent prompts |
| `acceptance/` | acceptance judge, ground-truth builders, task definitions, pre-registration |
| `scripts/` | web interface, batch runner, re-judging; `experiments/` for the solver study and figures |
| `knowledge/` | general modelling patterns available to the Modeler |
| `poster/gtc2027/` | poster source and figures |
| `results/` | run settings, reference answers, per-run judge records, solver-comparison data |
| `tests/` | unit tests |

Raw downloads, LLM transcripts and caches are not versioned.

## Citation

```bibtex
@misc{yang2027novaq,
  title        = {{NOVA-Q}: Verified {LLM} Agents From Natural Language and Open Data to {QAOA} on {CUDA-Q}},
  author       = {Yang, Chia-Chen and Ou, Chia-Ho},
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
