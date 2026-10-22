# Development log (former README of this repository)

> Kept for the record: the research plan, progress notes and the original QuantumQUBO Agent README (unchanged, at the end).
> The current overview is the [top-level README](../README.md).

# Feedback-Driven Annealing Agent

> **Built on the original project: [QuantumQUBO Agent](https://github.com/QuitttCat/QuantumQUBOAgent)**
> by Niloy Kumar Mondal and Md Rizwan Parvez (ICML 2026 Workshop, [OpenReview](https://openreview.net/forum?id=9YTedapat4)), released under the [MIT License](LICENSE).
> All code under `QuantumQUBO Agent/` was written by the original authors, except the files listed in
> [Our Changes So Far](#our-changes-so-far). We do **not** claim authorship of the original pipeline.
> The original README is kept unchanged in [Original Project: QuantumQUBO Agent](#original-project-quantumqubo-agent).
>
> **Benchmark data:** instances from [QOBLIB](https://github.com/ZIB-AOPT/QOBLIB), the Quantum Optimization Benchmarking Library
> (Koch et al., *Nature Computational Science*, [arXiv:2504.03832](https://arxiv.org/abs/2504.03832)), CC BY 4.0. They are used unmodified as
> data. Our loader (`qubo_auto/annealing/qoblib.py`) re-encodes them with one penalty term per constraint and does not use QOBLIB's QUBO files.

**The full research specification is [`docs/research_spec.md`](docs/research_spec.md)** (v0.2, Traditional Chinese; v0.1 in `docs/history/`).
It is the authoritative document. This README summarises it and records progress.

---

## Repository Layout

```
QuantumQUBOAgent/
├── README.md                    this overview (+ original README at the bottom)
├── LICENSE, CITATION.cff        original MIT license and citation (kept)
├── docs/
│   ├── research_spec.md         research specification (authoritative)
│   └── related_work.md          related work, external resources, novelty check
├── archive/                     files the current system does not use (see archive/README.md)
└── QuantumQUBO Agent/
    ├── qubo_auto/               Python package
    │   ├── agents/ prompts/ verification/   front end (original): Planner, Judge, Formulizer, Debugger, Coder, Writer
    │   ├── pipeline.py, schemas.py, llm_client.py
    │   ├── annealing/           our feedback loop
    │   │   ├── problem.py       ProblemInstance (objective + one penalty per constraint)
    │   │   ├── backends.py      adapters: SA, D-Wave QPU, Fujitsu DA
    │   │   ├── guardrail.py     parameter checks, config caps, hardware budget
    │   │   ├── feedback.py      Feedback Extractor + run log
    │   │   ├── tools.py         verify_small, penalty_lower_bound, qubo_summary, scale_for_backend
    │   │   ├── loop.py          closed-loop workflow engine
    │   │   ├── agents.py        Configurator / Diagnoser / Refiner (+ single-agent mode)
    │   │   ├── baselines.py     default, rule, random, optuna
    │   │   ├── rag.py           knowledge base (4 stores)
    │   │   ├── examples.py      problem generators (+ MILP references)
    │   │   └── qoblib.py        QOBLIB loader (Market Split, MIS)
    │   └── prompts/annealing/   agent prompts
    ├── knowledge/               curated RAG stores (backend docs, formulation patterns, tuning)
    ├── benchmarks/              100 natural-language benchmark problems (front end)
    ├── scripts/
    │   ├── run_single.py, run_benchmark_suite.py   front end (NL -> QUBO)
    │   ├── run_backend.py       one QUBO on one or more backends
    │   ├── run_loop.py          closed loop on one problem
    │   ├── serve_vllm.sh        local LLM server
    │   ├── experiments/         exp_p2_compare.py, exp_dwave_chain_strength.py, bench_vllm.py
    │   └── ablation_scripts/    original paper's ablations (No Judge / No Planner / Direct Method)
    ├── tests/test_annealing.py  offline tests (`pytest`; pytest.ini sets the import path)
    ├── config.yaml              front-end config  (config_no_retries.yaml: original ablation)
    ├── config_loop.yaml         feedback-loop config
    ├── results/                 run logs (gitignored) + experiments/p2_sa_v1 (kept)
    └── transcripts/             LLM call logs (gitignored)
```

## In One Sentence

Given an optimization problem (in natural language, or as a ready-made QUBO), the system builds the
QUBO, configures an annealing backend, runs it, analyses the result, and uses **real execution
feedback** to adjust penalty weights, scaling and solver parameters before solving again. It forms a
closed loop across **D-Wave, Fujitsu Digital Annealer and CGA**.

---

## Research Questions

| ID | Question | Role |
|---|---|---|
| **RQ1** | What fraction of mathematically verified QUBOs perform poorly on real annealers, and why? | Motivation / characterisation |
| **RQ2** | Under a fixed solver-call budget, does closed-loop agent refinement beat rule-based heuristics and Bayesian optimisation (Optuna)? | **Core** |
| **RQ3** | How different are the best settings for the same problem across backends, and can the agent transfer experience from one backend to another? | Highlight: answers "why an LLM?" |
| **RQ4** *(spec v0.2)* | Does choosing each constraint's QUBO encoding adaptively, based on constraint type and backend (e.g. slack vs. unbalanced, one-hot vs. domain-wall), improve hardware results? How many encoding errors does verification catch? | Extension: closed loop at the formulation level |

**Expected contributions**
1. **Hardware-in-the-loop QUBO workflow.** Formulations run on real annealers, and execution results feed back into corrections. QuantumQUBO Agent does not do this and lists it as future work.
2. **Semantic-preserving refinement.** The agent may only act within a defined action space, and deterministic code guarantees that every adjusted QUBO still encodes the original problem.
3. **Backend-agnostic feedback extraction.** Results from all three backends are reduced to one shared set of metrics.
4. **Budget-matched empirical comparison** against heuristics and Bayesian optimisation at the same solver-call budget, including an analysis of which failure types an LLM handles better.
5. **Constraint-wise encoding selection** *(spec v0.2)*. An Encoding Agent chooses an encoding for each constraint, and code assembles the QUBO from a verified encoding library. Two-way verification catches QUBOs that pass the test cases but are encoded wrongly. Here, *two-way* means: a violated constraint must give a penalty > 0, and a satisfied constraint must allow some choice of auxiliary variables that gives a penalty of 0.

**Relationship to QuantumQUBO Agent.** We reference and re-implement its NL → QUBO front end, with
attribution under the MIT License. NL → QUBO is **not** claimed as a main contribution; the main
contribution is the execution-feedback loop. The original pipeline also serves as a baseline and as
a source of problems.

---

## Design Principles

1. **Fixed interfaces, swappable implementations.** Changing the model, backend or prompt must not affect other components.
2. **The LLM decides; it never computes.** Conversion, execution, verification and metrics are deterministic code, and every LLM output is validated before it runs.
3. **Everything is configuration.** Models, temperature, retries, budgets, stop conditions and the backend list live in config files.
4. **The number of agents can scale.** Multi-agent and single-agent modes are switched by configuration; the switch itself is an ablation.
5. **Simulate first, then use hardware.** Every backend has a local stand-in. Development and bulk statistics run on the stand-in; hardware is used for validation.
6. **Log everything.** Every LLM call, solver run and decision is logged so it can be replayed, analysed, and reused as RAG experience.
7. **Degrade gracefully.** When any layer fails, the system falls back to the last working configuration instead of crashing.

---

## Architecture

```
  FRONT END (optional) — Formulation, referencing QuantumQUBO Agent
  ┌──────────────────────────────────────────────────────────────────────┐
  │ NL problem → Planner ─→ Formulizer ─→ Encoding Agent ─→ QUBO         │
  │              (LLM)       (LLM: typed     (LLM: one        Compiler*  │
  │                          constraints)    encoding per        ↓       │
  │                                          constraint)     Verifier*   │
  └───────────────────────────────────────────────────────┬──────────────┘
                                                          │ verified QUBO
       Oracle condition: a known-correct QUBO enters here ──┤
                                                          ▼
  CORE — Execution Feedback Loop (the main subject of this research)
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Configurator Agent ──► Guardrail* ──► Backend Adapter* ──► Backend  │
  │   (cold start)            ▲                               D-Wave /   │
  │                           │                               Fujitsu /  │
  │                           │                               CGA / sim  │
  │                           │                                  │       │
  │                     Refiner Agent ◄── Diagnoser Agent ◄── Feedback   │
  │                     (legal actions     (diagnosis label   Extractor* │
  │                      only)              + evidence)          │       │
  │                                              │               ▼       │
  │                                    stop? ─► best solution + report   │
  │      formulation_suspect / precision_loss ─► back to Encoding Agent  │
  └──────────────────────────────────────────────────────────────────────┘
        RAG knowledge base ⇢ Configurator / Diagnoser / Refiner
        Execution log (from Feedback Extractor) ⇢ RAG experience store

  * = deterministic code (no LLM)
```

The mermaid version is in [spec §3](docs/research_spec.md#3-系統架構).

---

## Components: Reused / Replaced / New

| Component | Source | Treatment | Status |
|---|---|---|---|
| Planner Agent | QuantumQUBO Agent | Concept reused, prompt rewritten | Original version runs locally (Phase 1) |
| Judge Agent | QuantumQUBO Agent | Reused, merged into the Verifier flow | Original version runs locally |
| Formulizer Agent | QuantumQUBO Agent | Reused, **plus** each penalty labelled with its constraint | Original version runs locally |
| Coder Agent | QuantumQUBO Agent | **Replaced** by a deterministic QUBO Compiler (PyQUBO or symbolic expansion) | Planned (P5) |
| Debugger Agent | QuantumQUBO Agent | Only for natural-language test data; deterministic generators otherwise | Unchanged |
| Test Runner | QuantumQUBO Agent | **Stricter**: the *entire* argmin set must be feasible, not just overlap the optimum | Planned. Confirmed weakness: [`test_runner.py:90`](QuantumQUBO%20Agent/qubo_auto/verification/test_runner.py#L90) only checks overlap |
| Writer Agent | QuantumQUBO Agent | Not needed | — |
| Instance generators | QuantumQUBO Agent repo | Reused, extended to scalable sizes | Planned |
| **Encoding Agent** *(spec v0.2)* | new | Chooses an encoding per constraint; the QUBO is compiled from a verified encoding library | **Next to build** (P2.5) |
| **Configurator / Diagnoser / Refiner** | new | Core of this research | **Built** (multi- and single-agent modes; experiments pending) |
| **Backend Adapter, Feedback Extractor, Guardrail** | new | Core of this research | **P0 done** (see below) |
| **RAG knowledge base** | new | Cold start + cross-backend experience | **Built v0** (3 curated stores + auto-written experience store; ablations pending) |

---

## Agents

All agents share **one local LLM** (see [Local LLM](#local-llm-2--l40s)). Agents differ only in role,
context, prompt and output schema, which is defined as JSON schema in configuration.

**Front end (optional):** **Clarification Agent** *(spec v0.2)*. Before anything is formulated, it drafts its understanding (variables, objectives, hard/soft constraints, required data, size vs. backend limits, assumptions) and asks the user ranked questions, marking which must be answered. Downstream steps refuse a session the user hasn't confirmed (`scripts/clarify.py`). · Planner (NL → structured spec, with hard/soft constraints) · **Encoding Agent** *(spec v0.2)* (one encoding choice per constraint, with reasons; input: typed constraints, backend traits, RAG formulation patterns, diagnoses from the loop) · Formulizer
(spec → symbolic QUBO + constraint label per penalty + reasoning for the initial penalty weights;
writes maths only, no code) · Judge (pass/fail + issues).

**Core:**

| Agent | Job | Input | Output |
|---|---|---|---|
| **Configurator** | First-run settings (cold start) | QUBO summary (size, density, coefficient range, constraint structure), target backend, RAG results | Penalty weights, scaling strategy, solver parameters |
| **Diagnoser** | What went wrong this round | Feedback summary, history, RAG | Diagnosis label + evidence + suggested direction |
| **Refiner** | Turns the diagnosis into a concrete change | Diagnosis, current config, legal actions | Next config (legal actions only) |

**Diagnosis labels (v0, extensible):** `penalty_too_weak` · `penalty_too_strong` · `precision_loss` ·
`chain_break` (D-Wave) · `insufficient_sampling` · `schedule_mismatch` · `formulation_suspect`
(returns to the front end) · `converged`.

**Scalable configurations (for ablation):** multi-agent · single-agent (one agent plays all roles,
same tools and information) · backend-specialist (optional: one agent per backend plus a coordinator).

---

## Deterministic Tool Layer

| Tool | Purpose | Status |
|---|---|---|
| `check_params` | Name exists, type correct, within legal range (checked against `describe()`) | **Done** |
| `run` | Execute through a Backend Adapter | **Done** |
| `extract_feedback` | Unified metrics from a sample set | **Done** |
| `verify_small` | Brute-force check that the whole argmin set is feasible and contains the true optimum | **Done** |
| `compile_qubo` | Symbolic formulation → matrix | Planned |
| `qubo_to_ising` / `ising_to_qubo` | Format conversion | Planned |
| `penalty_lower_bound` | Penalty floor that keeps the feasible optimum | **Done** (heuristic: largest single-flip objective gain; exact check via `verify_small`) |
| `scale_for_backend` | Rescale for backend precision while preserving the argmin | **Done** (factor + optional integer rounding, argmin check on small instances) |
| `reference_solve` | Reference solution (Gurobi / CP-SAT / long SA) | **Done**: exact per generator (enumeration, knapsack DP) and scipy **HiGHS MILP** for larger instances; MILP agrees with enumeration on all 30 cross-check instances |
| `qubo_summary` | Problem features for agents and the RAG key | **Done** |

### Backend Adapter interface (implemented)

```
describe()             tunable params + legal ranges, precision limits, billing unit
submit(qubo, params)   run on the backend → raw result
normalize(raw)         → unique samples, occurrences, solver time (s), backend-specific metadata
simulate(qubo, params) local stand-in (simulated annealing)
run(...)               check_params → submit → normalize → SolverRunResult
```

| Backend | Adapter | Status | Backend-specific feedback kept |
|---|---|---|---|
| Simulated annealing (`sa`) | `dwave-samplers` | **Working**; stand-in for every backend | beta range |
| **Fujitsu DA v3c** (`fujitsu_da`) | REST async job | **Working on hardware** (first real backend); parameter limits **probed** | job id, raw timing (ms), queue wait, DA energies, result status |
| **D-Wave QPU** (`dwave_qpu`) | Ocean `EmbeddingComposite` | **Working on hardware** (Advantage_system4) | chain-break fraction, chain strength, qubits used, max chain length, timing (µs) |
| CGA | — | Waiting for API confirmation (spec §6) | — |

Implementation choices that affect experiments:
- **Energies are always recomputed locally** from `(QUBO, offset)`, so they are comparable across backends. Fujitsu does not receive the offset; its own energies are kept in metadata and were checked to match after adding the offset back.
- **The full sample set is kept** (unique samples + occurrence counts), not just the best sample. Feedback needs it for feasibility rates, per-constraint violation rates and diversity.
- **All times are converted to seconds.** Each backend's raw timing fields are stored unchanged in metadata.
- **Fujitsu jobs are always deleted** after the result is read, including on failure, because the account has a job-count limit.
- **Fujitsu behaviour we measured, which the adapter now handles:**
  - Submission always returns HTTP 200, and bad parameters only surface later as job status `Error` with a message. So the guardrail must catch them before submission.
  - A `Waiting` or `Running` job can be neither cancelled nor deleted.
  - Jobs can sit in the queue for minutes. Queue wait (default limit 30 min) is therefore separate from run time, and any job that can't be deleted is recorded in `results/fujitsu_orphan_jobs.jsonl` for `cleanup_orphans()`, which only touches jobs we recorded.
- **Penalties are stored per constraint, each with unit weight.** `ProblemInstance.to_qubo(weights)` can only rescale named penalties; the objective coefficients cannot be changed. This is how the code enforces the spec's "objective coefficients must not be modified" guardrail.

---

## RAG Knowledge Base (P3)

Four stores, each switchable for ablation:

| Store | Content | Used by | Updated |
|---|---|---|---|
| Backend docs | D-Wave / Fujitsu / CGA parameters, limits, vendor guidance | Configurator, Refiner | Manual, versioned |
| Formulation patterns | One-hot, domain-wall, slack encodings and their hardware behaviour | Formulizer, Diagnoser | Manual + literature |
| Tuning knowledge | Penalty-setting methods, chain-strength heuristics, etc. | Diagnoser, Refiner | Manual |
| Execution experience | Problem features → config → result → diagnosis → was the change effective | All | **Automatic** (from the run log) |

Retrieval rules:
- The retrieval key is **problem features**, not the full natural-language text.
- At most top-k = 3–5 results are returned.
- Every retrieval is logged.

The experience store is **split by train/test** to prevent contamination. For RQ3, which backends'
records it contains is controlled explicitly.

---

## Closed-Loop Workflow

1. **Prepare**: get a QUBO (from the front end or an oracle). Continue only if `verify_small` passes.
2. **Configure**: the Configurator proposes settings from problem features and RAG.
3. **Check**: the Guardrail validates the settings. Invalid settings are returned with reasons, up to N = 2 retries.
4. **Execute**: the Backend Adapter runs the job.
5. **Extract**: the Feedback Extractor computes metrics and writes them to the log.
6. **Diagnose**: the Diagnoser labels the problem. `formulation_suspect` goes back to the front end (bounded).
7. **Refine**: the Refiner picks the next step from the legal actions.
8. **Stop?** Otherwise go back to step 3.
9. **Output**: the best solution of the whole run **by the original-problem metric**, not the last round's solution.

**Stop conditions** (defaults, configurable):
- solver-call budget reached (10)
- no improvement for K = 3 rounds
- gap threshold to the reference reached
- the Diagnoser reports `converged`

**Guardrails (all code):**
- `check_params` against `describe()`
- objective coefficients immutable; penalties only above `penalty_lower_bound`; scaling must pass an argmin-consistency check
- duplicate-config detection
- the historical best is always kept, and "revert to best" is a legal action
- LLM output is schema-validated, then retried, then replaced by a fallback rule
- hardware quota is checked before every real call

**Feedback has two levels:**
- **Shared** (identical across backends): feasibility rate, per-constraint violation rates, best and mean objective, gap to the reference, diversity, time. **Only this level decides whether a round improved.**
- **Backend-specific**: used only to diagnose and tune that backend's own parameters.

QUBO energy is **not comparable across rounds**, because changing penalties reshapes the landscape.

**Action space (stage 1 opens the first five):**
1. penalty weights
2. coefficient scaling
3. number of samples/reads
4. solver parameters
5. backend-specific parameters
6. formulation rewrite (later; high risk, must be re-verified by the front end)

---

## Experiment Design (summary; details in [spec §10](docs/research_spec.md#10-實驗設計))

- **Benchmarks:**
  - Main (constrained, penalty-sensitive): Knapsack, Graph Coloring, Set Cover, Facility Location or Scheduling.
  - Control: Max-Cut.
  - Stress test: TSP.
  - Instances are generated programmatically at several sizes.
- **Conditions:** *Oracle QUBO* (loop only → RQ2) vs. *end-to-end* from natural language. The difference between them is the loss caused by formulation errors.
- **Baselines** (same solver-call budget):
  1. Default parameters + formula-based penalty
  2. Rule-based heuristic
  3. Random search
  4. Bayesian optimisation (Optuna)
  5. Raw QuantumQUBO Agent output on hardware (RQ1)
- **Ablations:**
  - multi-agent vs. single-agent
  - with/without RAG, and with each store removed
  - with/without the Diagnoser
  - local model vs. a frontier API model
- **Metrics:**
  - quality: feasibility, best feasible objective, gap
  - efficiency: solver calls to reach the target, wall time
  - cost: tokens, hardware usage
  - stability: across seeds and instances
  - behaviour: diagnosis accuracy, invalid-action rate
- **Statistics:**
  - simulation: ≥ 10 seeds per condition × instance; hardware: 3–5 runs
  - LLM temperature 0, with a non-zero robustness check
  - Wilcoxon signed-rank tests
  - configs, prompts, logs and seeds are published

---

## Roadmap

| Phase | Content | Done when | Status |
|---|---|---|---|
| *Prerequisite* | Original pipeline on a **local LLM** (vLLM, 2 × L40S) | Max-Cut NL → QUBO → brute force succeeds without paid APIs | ✅ **Done** 2026-10-04 |
| **P0 Infrastructure** | Backend Adapter (1 real + stand-in), Feedback Extractor, Guardrail, log | Same QUBO runs on the stand-in and on one real machine with unified metrics | ✅ **Done** 2026-10-04 (SA + Fujitsu DA + D-Wave QPU) |
| **P1 Motivation (RQ1)** | Scale up QUBOs that QuantumQUBO Agent got right and run them on hardware | Failure rate + failure-cause taxonomy | ⬜ Needs a hardware budget |
| **P2 Core loop** | Configurator / Diagnoser / Refiner + baselines | Budget-matched agent vs. baselines comparison | 🟨 Framework built; **SA comparison done** 2026-10-04 (agents beat all 4 baselines overall, p < 0.01; Knapsack ceiling = formulation); hardware loops pending |
| **P2.5 Encoding selection** *(spec v0.2)* | Encoding library, `verify_encoding`, Encoding Agent, re-encoding inside the loop | Re-encoding beats a fixed encoding on at least one class (Knapsack: slack vs. unbalanced) | 🟨 **Encoding library + compiler built**; first evidence: unbalanced encoding cuts the Knapsack gap from 17–23% to 3–5% on SA defaults. Encoding Agent next |
| **P3 RAG & ablation** | Four knowledge stores, ablations | Contributions of multi-agent and RAG separated | 🟨 RAG v0 built; **first ablation done**: no significant effect of RAG or multi-agent yet (p = 0.17 / 0.10); RAG v1 planned |
| **P4 Cross-backend (RQ3)** | 2nd and 3rd backend, transfer experiments | ≥ 2 backends compared | ⬜ (Fujitsu + D-Wave adapters already working; CGA pending API) |
| **P5 Front-end integration** | NL → QUBO front end feeding the loop | Oracle vs. end-to-end gap quantified | ⬜ |

Each phase should end with a result that can go in the paper. If time runs short, P4 and P5 can shrink
without losing the core contribution.

### Open questions (to confirm with the advisor, spec §12)

- [ ] Is NL → QUBO a main contribution, or only the system's front end?
- [ ] Main backend: **Fujitsu was chosen to go first** for P0. Should it stay the main backend?
- [ ] Quota and billing for each of the three backends
- [ ] Which system is CGA, and what can its API set and return?
- [ ] Target venue

---

## Current Progress

### P0: Infrastructure ✅ (2026-10-04)

The same QUBOs ran on the SA stand-in and on **both real backends**, and all three produced the same set of
metrics. Two problems were used:
- `select_k_with_conflicts`: choose K = 3 of 8 sites maximising value, with conflict pairs. This is the AED site-selection structure (8 variables, optimum 17).
- `max_cut`: the control problem with no constraints (8 nodes, optimum 49).

| Problem | Metric | SA stand-in (100 reads) | **Fujitsu DA v3c** (`time_limit_sec=3`) | **D-Wave Advantage_system4** (100 reads, 20 µs) |
|---|---|---|---|---|
| select_k | Feasibility rate | 98% | 100% | 96% |
| | Violations (`select_k` / `conflicts`) | 2% / 0% | 0% / 0% | 2% / 2% |
| | Best feasible objective / gap | 17 / 0 | 17 / 0 | 17 / 0 |
| | Mean feasible objective | 16.01 | 16.80 | 15.24 |
| | Solver time / wall time | 0.014 s / 0.02 s | 3.02 s / 9.1 s | 0.028 s QPU / 2.9 s |
| | Backend-specific | — | — | chain strength 20.7, chain breaks 0%, 12 qubits, max chain 2 |
| max_cut | Best objective / gap | 49 / 0 | 49 / 0 | 49 / 0 |
| | Mean objective | 48.31 | 47.60 | 48.09 |
| | Solver time / wall time | 0.012 s / 0.02 s | 3.02 s / 9.0 s | 0.025 s QPU / 2.3 s |
| | Backend-specific | — | — | no chains (8 qubits, chain length 1) |

All backends find the optimum of these tiny instances, so they do not yet discriminate between
settings. Even here, though, mean solution quality and the constraint-violation pattern already
differ by backend, and that is the kind of signal the Diagnoser will work from. Total hardware use for
P0 validation: 3 Fujitsu jobs (≈ 3 s each) and 53.6 ms of D-Wave QPU access time.

Also verified:
- Fujitsu energy + offset = local energy for every sample.
- DA timing is in milliseconds.
- Every Fujitsu job is deleted after the run; the 4 older jobs on the account are not ours and were left alone.
- The Guardrail rejects unknown parameters (`chainstrength`), wrong types (`num_reads=1.5`) and out-of-range values (`annealing_time=5000`) in one pass and lists every reason, before anything is sent to the QPU.
- 7 offline unit tests pass (`pytest tests/test_annealing.py`).

**Not yet done in P0 scope:** no hardware-quota guardrail yet. It is needed before running loops on real
hardware. D-Wave does not report remaining QPU time through its API, so the quota check will use a
local budget plus the per-call `qpu_access_time`.

### Full 100-benchmark front-end run on the local LLM (2026-10-04)

`scripts/experiments/run_suite_parallel.py --name full_pipeline_local --workers 24` ran the original pipeline unchanged on all 100 benchmarks: local Qwen3.6-35B-A3B-FP8, 64k context, seed 42, about 5.5 h, $0. Results are in `results/full_pipeline_local_42.csv`.

| | Original paper (OpenRouter) | Ours (local vLLM) |
|---|---|---|
| Success | **68 / 100** | **67 / 100** |
| Both succeed / both fail | 55 / 20 | |
| Only one succeeds | 13 (paper only) | 12 (ours only) |

- The overall rate matches, so the local model is a drop-in replacement for the paid API.
- **Per-benchmark outcomes are unstable between single runs:** 25 benchmarks flipped. Claims about *which* problems the front end solves need several seeds per benchmark; this matters for P1 and for the paper.
- One benchmark (`graph_bandwidth`) hung for about 2 h in brute-force verification, because the original Test Runner never enforces `max_n_vars`. The Test Runner now refuses to enumerate more than 24 variables (`MAX_BRUTE_FORCE_VARS`).
- An earlier run with a 32k context truncated long reasoning, and some of its failures came from that cut-off. The context is now 64k.

### Prerequisite: Local LLM ✅ (2026-10-04)

The original Max-Cut example ran end to end on the local model:

```
Planner ✓ → Judge(spec) ✓ → Formulizer ✓ → Judge(QUBO) ✓ → Debugger (8 cases) ✓
→ Coder ✓ → brute-force verification 8/8 → SUCCESS
```

| | Original paper run (OpenRouter) | Our run (local vLLM, 2 × L40S) |
|---|---|---|
| Model(s) | qwen3.6-35b-a3b + qwen3-coder-next | qwen3.6-35b-a3b (FP8) for all agents |
| Result | success, 8/8 (after 2 coder retries) | success, 8/8 (first attempt at every stage) |
| Wall time / tokens | 363.6 s / 49,194 | 107.1 s / 28,892 |
| API cost | paid credits | **$0** |

One run with one seed shows the setup works. It is not a benchmark result.

With thinking **off**, the same model failed Max-Cut: the Judge correctly rejected three formulations.
Reasoning matters for formulation quality.

### P2 + P3: Agents, workflow and RAG — framework built (2026-10-04)

**How the spec maps to the code** (`QuantumQUBO Agent/qubo_auto/annealing/`):

| Spec | Code | Notes |
|---|---|---|
| §8 closed-loop workflow | `loop.py` → `run_loop` | Deterministic engine; policies only *propose* configs |
| §8.2 stop conditions | `run_loop` | solver-call budget, `patience` rounds without improvement, gap target, agent `converged` |
| §8.3 guardrails | `loop.validate_config`, `guardrail.py` | unknown names, types and ranges (+ config caps); penalty ≥ `penalty_lower_bound`; scale range; duplicate configs; hardware budget (ledger); bounded retries → rule fallback; best-so-far kept |
| §5.2 Configurator / Diagnoser / Refiner | `agents.py` + `prompts/annealing/*.txt` | JSON-schema outputs (Pydantic), one re-ask on invalid output, then rule fallback |
| §5.3 multi- vs single-agent | `agents.mode: multi \| single` | Same tools, information and budget |
| §5.2 diagnosis labels | `agents.DiagnosisLabel` | the 8 labels from the spec, enforced by the schema |
| §9.2 action space (stage 1) | `set_penalty`, `set_param`, `set_scale`, `revert_to_best` | Objective coefficients are not reachable from any action |
| §7 RAG (4 stores) | `rag.py`, `knowledge/*.jsonl`, `results/experience.jsonl` | Feature- and tag-keyed retrieval; backend-scoped docs; experience never includes the current or test-split instances; shown and cited ids logged per round |
| §10.3 baselines | `baselines.py` | `default`, `rule`, `random`, `optuna` (Optuna is not installed yet) |
| §2.3 configuration | `config_loop.yaml` | Budgets, caps, agent mode, RAG stores, test split, hardware budget |
| §2.6 logging | `results/loop_runs.jsonl`, `annealing_runs.jsonl`, `transcripts/` | Every round, every backend run, every LLM call |

**Smoke test (SA stand-in, one instance and one seed each; this shows the system works, it is not a result):**

| Instance | Policy | Solver calls | Best / reference | What happened |
|---|---|---|---|---|
| knapsack n=14 (20 vars incl. slack) | default | 4 | 167 / 170 | Reruns of the same config; stopped after 3 rounds without improvement |
| | rule | 5 | 167 / 170 | Kept doubling the penalty because of a 1% violation rate, and the objective got worse (*penalty_too_strong*) |
| | random | 5 | 167 / 170 | — |
| | **agent (multi + RAG)** | **2** | **170 / 170** | Configurator kept the penalty at its floor (cited `fp-slack`, `tk-penalty-strong`); Diagnoser: `insufficient_sampling`; Refiner: num_reads 100 → 200 |
| select_k n=30, k=6 | **agent (single + RAG)** | **2** | **110 / 110** | Same pattern; retrieved the knapsack experience record from the run above |

Fixed during the smoke tests:
- The first agent version stopped after one round at a 1.8% gap, because the agents were not told the gap target or the remaining budget. Every prompt now includes a loop-status block.
- RAG returned D-Wave documents for SA runs; docs are now scoped to their backend.

Guardrail, fallback and isolation paths are covered by offline tests (16 pass): penalty floor, unknown names, config caps, policy crash → rule fallback, no repeated configs, hardware budget refusal before any call, experience isolation, backend-scoped docs, graph-colouring QUBO correctness, MILP reference vs. enumeration.

### Knowledge base filled by our own experiments (2026-10-04)

Where documentation was missing, we measured it ourselves. Every entry records its source and whether it was a single run.

| Knowledge entry | How it was obtained | Main finding |
|---|---|---|
| `bd-fujitsu-params` | Out-of-range probes against DA v3c (rejected jobs cost no solve time) | `num_output_solution` 1–1024 (default 5), `num_group` 1–16, `gs_level` 0–100, `gs_cutoff` ≥ 0, `time_limit_sec` integer ≥ 1, unknown names rejected |
| `bd-fujitsu-ops` | Same probes | Validation is asynchronous; running jobs can't be cancelled or deleted |
| `tk-chain-measured` | `scripts/experiments/exp_dwave_chain_strength.py` (5 QPU calls) | On a dense 30-variable QUBO, Ocean's default chain strength left breaks in **99% of samples** (1% feasible, gap 26%). 2–4× removed the breaks, but feasibility stayed at 4%, so chain strength and penalties have to be tuned together (`precision_loss`) |
| `bd-cross-backend-measured` | Same QUBOs on SA, Fujitsu and D-Wave | Dense select_k n=30: Fujitsu 100% feasible and optimal, SA 87% and optimal, D-Wave 1% at gap 26%. Knapsack (dynamic range 320): Fujitsu optimal in 1 s; SA gap 6.5% |

These single runs are early **RQ1/RQ3 evidence**: the same verified QUBO behaves very differently across backends, and on D-Wave it is far from usable without tuning.

**One incident to report:** a probe with `time_limit_sec=1801`, sent to find the upper limit, was accepted and could not be cancelled. It stayed `Running` for about 30 minutes and delayed other jobs, although the account dashboard did not show that time as charged. The probing procedure now never pushes time-type parameters upward.

**Hardware budget** (`config_loop.yaml`, cumulative via `results/hardware_ledger.jsonl`). `run_backend.py` and the experiment scripts also check and record every real call.
- **Fujitsu: enabled with a 60-minute cap** (20% of the account's **300 DA-minutes per month**), re-enabled for the urban-planning project; raise it in `config_loop.yaml`. We estimated up to ~30 minutes for the 2026-10-04 runs (mostly one mistaken 1801 s probe), but the account dashboard still showed the quota intact afterwards, so the probe was apparently not charged its full time limit. Probing time-type parameters upward is never done.
- **D-Wave: 500 calls / 30 s of QPU time.**

### P2 formal experiment: `scripts/experiments/exp_p2_compare.py --tag p2_sa_v1` (completed 2026-10-04)

- **Condition:** oracle QUBO on the SA stand-in.
- **Instances:**
  - Knapsack n = 40 and 60
  - select_k n = 60 (k = 10) and n = 80 (k = 12)
  - 2 test instances each, with exact references from DP or MILP
  - Sizes were calibrated so SA defaults do *not* already reach the optimum (knapsack gap ≈ 15–21%, select_k ≈ 0–2%).
- **Runs:** 10 seeds per instance; every policy gets the same solver-call budget (10), patience (3) and caps.
- **RAG train/test split (§7.3):** the multi-agent+RAG policy first runs on 8 separate *training* instances to build the experience store. During testing the store is frozen, and test instances are excluded from retrieval.
- **Policies:**
  - baselines: default, rule, random, optuna
  - agents: agent-multi-rag, agent-single-rag, agent-multi-norag (which gives the multi/single and RAG ablations)
- **Outputs** in `results/experiments/p2_sa_v1/`: `runs.csv`, `summary.md` (mean gap, optimal count, calls, tokens, paired Wilcoxon tests), plus per-run loop and backend logs.

**Results (560 test runs, 0 crashes).** Gap = relative distance to the exact optimum (0 = optimal); no feasible solution counts as gap 1.

| Policy | Mean gap (all) | Knapsack gap | select_k gap | Optimal runs (select_k) | Mean solver calls | Mean LLM tokens / run |
|---|---|---|---|---|---|---|
| default | 0.0841 | 0.1680 | **0.0003** | **38/40** | 4.2 | 0 |
| rule | 0.0875 | 0.1712 | 0.0038 | 19/40 | 4.4 | 0 |
| random | 0.0896 | 0.1643 | 0.0149 | 23/40 | 5.9 | 0 |
| optuna (BO) | 0.0919 | 0.1749 | 0.0090 | 19/40 | 5.3 | 0 |
| agent-multi-rag | 0.0776 | 0.1509 | 0.0042 | 35/40 | 4.6 | 67k |
| **agent-single-rag** | 0.0740 | **0.1332** | 0.0149 | 26/40 | 5.9 | 50k |
| **agent-multi-norag** | **0.0729** | 0.1397 | 0.0062 | 25/40 | 6.1 | 55k |

Paired Wilcoxon signed-rank tests (same instance and seed, n = 80 pairs). Full table: `results/experiments/p2_sa_v1/summary.md`.
- **Every agent variant has a lower overall gap than every baseline**, p between 2.6e-7 and 6.4e-3. With a Bonferroni correction for the 15 comparisons (threshold 0.0033), all of these stay significant except *agent-multi-rag vs. default* (p = 0.0064).
- **Knapsack:** agents are clearly better. Agent-multi-norag 0.140 vs. default 0.168, p = 1.2e-5. But **no policy reached the optimum in any of the 240 Knapsack runs.**
- **select_k:** the default baseline is best. Rerunning the default config with a new seed hits the optimum in 38/40 runs; agent-multi-rag got 35/40 (p = 0.046) and agent-multi-norag 25/40 (p = 0.001).
- **Ablations:** neither multi- vs. single-agent (p = 0.10) nor RAG on vs. off (p = 0.17) made a significant difference. RAG v0 was numerically *worse* on Knapsack.

**What the per-round logs show:**
1. **The Knapsack ceiling is in the formulation, not the parameters.** Agents correctly diagnosed `penalty_too_strong` hundreds of times. Their attempts to lower the capacity penalty were rejected by the guardrail 51 times, correctly: below the floor (the largest item value), an over-capacity state can become optimal. The binary-slack encoding makes the penalty dominate the landscape (dynamic range ≈ 300+). Within the stage-1 action space no method fixes this. The likely fix is a formulation rewrite (spec §9.2, e.g. unbalanced penalization), which is RQ1-type evidence in its own right.
2. **RAG v0 steers agents toward sampling changes.** With RAG, agents changed penalties 33 times and parameters 285 times; without RAG, 176 penalty changes. The most cited entries were `tk-sampling` (≈ 1,000 times) and `bd-sa-params`. The curated stores are unbalanced, and experience retrieval does not yet prefer records where the change *worked*. Next: re-weight experience retrieval by `effective`, and balance the tuning store.
3. **Agents over-tune easy problems.** On select_k, changing configurations was worse than simply resampling. The agents need a "resample the best config" option and a rule to recognise when that is enough.
4. **Cost:** 50–67k LLM tokens per agent run on the local model ($0 API cost; about 20 s of 2 × L40S time per run at 32 concurrent runs).

**Limits:** this is the SA stand-in only, with 2 test instances per family and size, two problem families and one LLM. The result says that the loop and agents work and beat the baselines on penalty-sensitive Knapsack. It says nothing yet about hardware backends.

### QOBLIB as benchmark and RAG source (2026-10-04)

[QOBLIB](https://github.com/ZIB-AOPT/QOBLIB) has 10 problem classes and 1,264 instances, with optimal or best-known values and both QUBO and MIP models. We sparse-cloned 4 small classes to `/research/datasets/QOBLIB` (or `$QOBLIB_DIR`).

| Use | Decision | Why |
|---|---|---|
| **Benchmark** | ✅ Market Split (01) and Max Independent Set (07) loaded via `qoblib.py` | A peer-reviewed standard set with references. **All 115 Market Split solutions and 31 MIS optima (≤ 200 nodes) verify** against our encoding. Our Market Split coefficients match QOBLIB's own metrics (max \|q\| = 56,477 for `ms_03_050_002`) |
| **RAG: formulation patterns** | ✅ `fp-qoblib-marketsplit`, `fp-qoblib-mis` | QOBLIB model notes plus our own measurements |
| **RAG: experience from submissions** | ❌ Not used | **None of the 2,377 submissions for classes 01/02/03/07 used an annealer**; the 223 quantum-hardware entries are gate-based (mostly QAOA). Submissions have no solver-parameter field, and adding best solutions to RAG would leak answers |

**Difficulty on the SA stand-in:**
- **MIS:** SA defaults find the proven optimum on every tested graph up to 200 nodes, so MIS will serve as the **D-Wave** benchmark (sparse graphs, short chains).
- **Market Split:** SA defaults find **no** feasible sample at n = 20; even 1000 reads × 20000 sweeps finds about 1 in 1000. It will go into the next SA comparison, scored by **success rate**.

**Research positioning:** there are no annealer results on these QOBLIB classes yet, so D-Wave and closed-loop results on them would be new, and could be contributed back to QOBLIB.

**Fixes made while integrating:**
- Penalty weights must now be **strictly above** the floor. A weight equal to the floor can tie feasible and infeasible states; for MIS the floor is 1.
- Formulation-pattern entries were wrongly hidden when tagged with another backend; `fp-cardinality` was invisible to the SA agents in P2 v1. Only backend docs and tuning entries are backend-scoped now.

### External resources and closest related work (2026-10-04)

Collected in [`docs/related_work.md`](docs/related_work.md). Eight new knowledge entries were added, summarised in our own words with sources: D-Wave best practices, a Fujitsu DA guide, chain-strength and annealing-time studies, and penalty-weight methods.

**Novelty risk.** *Practical Quantum CIM Empowerment via All-Domestic-Core Agentic Large Model* ([arXiv:2605.23934](https://arxiv.org/abs/2605.23934), Apr 2026) already has LLM agents iterating constraint weights with Coherent Ising Machine hardware in the loop. It must be cited. Our differentiators:
- several annealing backends, with cross-backend transfer (RQ3)
- semantic-preserving guardrails
- budget-matched baselines (rule, random, BO)
- multi/single and RAG ablations

Their abstract reports none of these.

### P2.5: Encoding library and first RQ4 evidence (2026-10-04)

**Why (found by the first end-to-end run).** The front end encoded Knapsack's capacity *inequality* as an *equality*, (Σwx − W)².
It passed the original verification only because every test case's optimum fills the knapsack exactly. Two fixes:
- The P5 bridge now also checks the reverse direction: if a constraint holds, some choice of auxiliary variables must make its penalty 0.
- A deterministic encoding library (`qubo_auto/annealing/encodings.py`), so the QUBO is no longer written by an LLM.

**Library:**
- Typed IR: binary variables, quadratic objective, and linear (=, ≤, ≥) / exclusion / implication constraints.
- Encodings:
  - linear equality: `square`
  - linear inequality: `binary_slack`, `unary_slack`, `unbalanced` (approximate)
  - exclusion and implication: `pairwise`
- `verify_encoding`: the two-way check for exact encodings; for approximate encodings it reports how feasible and infeasible points compare.
- `compile_problem`: IR + one choice per constraint → ProblemInstance. Feasibility and the objective are always computed from the IR, never from the penalties.
- 33 offline tests pass, including the inequality-as-equality bug.

**First evidence (SA defaults, the P2 Knapsack test instances, 5 seeds each):**

| Encoding | Variables (n = 40 / 60) | Mean gap | Feasibility |
|---|---|---|---|
| binary slack (the encoding used in P2) | 48 / 68 | 17–23% | 100% |
| unary slack | ≈ 205 / 310 | 33–43% | 100% |
| **unbalanced** (l1 = 10, l2 = 0.25) | **40 / 60** | **3.1–4.8%** | 100% |

The best P2 agents, tuning parameters on binary slack, reached 13–15%. **Changing the encoding beats parameter tuning by a wide margin.** The unbalanced parameters matter (l1 = 40, l2 = 0.25 gives 16–26%), so the Encoding Agent must choose encoding *parameters* too, and the loop must be able to tune them. Caveat: SA only, and the parameters were picked from a small grid. A budget-matched comparison with the agent choosing is next.

**Encoding Agent in the loop: first run** (Knapsack n = 40, test instance s0, SA, 10 solver calls; the knowledge base did *not* yet contain our encoding measurements):

| Rounds | What the agents did | Gap |
|---|---|---|
| 0 | Encoding Agent picked exact binary slack with weight 1000 ("preserve optimality") | 17.1% |
| 1–2 | Diagnoser: insufficient sampling; more sweeps and reads | 19–21% |
| 3–7 | Diagnoser: penalty too strong; weight lowered step by step to 30.1, just above the floor | 13.0% |
| **8** | Weight exhausted → **Refiner requested re-encoding → Encoding Agent switched to unbalanced (l1 = 30, l2 = 1)** | **7.1%** |
| 9 | More sweeps under the new encoding | **5.8%** |

For comparison, the rule baseline with the fixed binary-slack encoding and the same 10 calls ended at 10.7%, with SA effort at its cap. Cost: 137k LLM tokens and 21 min. This is the first end-to-end demonstration of the formulation-level loop (Diagnoser → Refiner → Encoding Agent).

We then measured slack vs. unbalanced on *training* instances only and added the result as `tk-encoding-knapsack-measured`: unbalanced with l1 ≈ ⅓ of the largest value scored 2–7%; binary slack scored 20–27%, and 26–31% with weight 1000. Re-running with this knowledge is the next RAG check.

### Next milestone

1. **Act on the P2 findings:** effectiveness-weighted experience retrieval, a balanced tuning store, a "resample best config" action, then re-run the ablations (`--tag p2_sa_v2`, 64 workers).
2. **P1 (RQ1)**: run the QuantumQUBO Agent's verified QUBOs scaled up on D-Wave (and the SA stand-in) with default settings, and classify the failures. The single runs above already show the effect.
3. **Hardware loops on D-Wave** (Fujitsu disabled because of its 300 min/month quota). D-Wave dense problems are the obvious first target.
4. **More generators** from spec §10.1: Set Cover, Facility Location / Scheduling, TSP stress test. Graph Coloring is done (`graph_coloring_random`).

---

## Local LLM (2 × L40S)

```
  All agents ──► qubo_auto/llm_client.py ──► OpenAI-compatible API (localhost:8000/v1)
                  (provider set by env vars)        │
                                                    ▼
                                    vLLM 0.30, tensor parallel = 2
                                    Qwen3.6-35B-A3B-FP8 (MoE, ~3B active)
                                       ┌────────────┴────────────┐
                                    L40S #0 (46 GB)           L40S #1 (46 GB)
```

One model is served once across both GPUs (16.8 GiB weights + 18.9 GiB KV cache per GPU, 32k context).

**Throughput, measured 2026-10-04** (`scripts/experiments/bench_vllm.py`, 64 real agent prompts, thinking on, ≤ 2048 output tokens):

| Serving layout | 32 concurrent | 64 concurrent |
|---|---|---|
| **Tensor parallel = 2** (one model across both GPUs) — *used* | **1,407 tok/s** | **2,083 tok/s** |
| Two independent single-GPU replicas | 1,223 tok/s | 1,643 tok/s |

- Tensor parallelism wins because both GPUs' memory bandwidth serves every decode step, and it leaves more room for KV cache.
- vLLM's `--data-parallel-size` doesn't create independent replicas for this MoE model: it couples the ranks via expert parallelism, and it failed to start here.
- Run agent experiments with about 64 concurrent workers. Both GPUs sit at 100% utilisation during experiments.
Every agent sends requests to this same server. We use the same model family as the original
paper so results stay comparable; it is also the model the spec names (§12).

| Variable | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `local` · `openrouter` · `openai` | `openrouter` (original behaviour) |
| `LLM_BASE_URL` | Any OpenAI-compatible endpoint | `local` → `http://localhost:8000/v1` |
| `LLM_MODEL` | Route **every** agent to this model | unset → per-agent models from `config.yaml` |
| `LLM_API_KEY` | Override the key | `local` → dummy `EMPTY` |
| `LLM_ENABLE_THINKING` | `local` only: Qwen3.x reasoning | `true`; Qwen3.6 ignores the original `/no_think` and reasoned on OpenRouter too |

Environment as deployed:

| Item | Value |
|---|---|
| GPUs / driver | 2 × NVIDIA L40S 46 GB · driver 565.57 (CUDA ≤ 12.7) |
| vLLM env | `/research/conda-envs/quantum-agent-local` (Python 3.11): an offline clone of `catalyst-qml`, which already had `torch 2.13.0+cu126` (vLLM 0.30.0's exact pin). vLLM `+cu129` wheel installed with the cu126 torch/CUDA libraries locked; `torchcodec` swapped to `0.17.0+cu126` |
| Model cache | `/research/cache/huggingface` (`/` has little free space) |
| Pipeline / backends | base Python (no GPU needed); D-Wave Ocean, `dwave-samplers` and `requests` are already installed |

Known issues:
- Base miniconda's torch is a CUDA 13 build and crashes on this driver. Don't use base for vLLM.
- FlashInfer's sampler JIT-compiles with the system `nvcc` 12.0, which is too old, so `serve_vllm.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0`.
- The original `tests/test_pipeline.py` never ran (it imports an `is_symmetric` that doesn't exist) and is archived; `pytest tests` now passes (18 tests).
- `run_single.py` writes LaTeX output to `QuantumQUBO Agent/deliverables/` (gitignored). The original authors' deliverables were moved to `archive/original_project/deliverables/`.

---

## How to Run

Every command below was executed end to end on 2026-10-04 (see *README verification* further down).

**Test UI** (Gradio, local only). It walks through clarify → model → solve:
```bash
/research/conda-envs/qubo-ui/bin/python scripts/ui.py      # open http://localhost:7860 (VS Code forwards the port)
```
- `qubo-ui` is a small venv on top of base Python, created so that installing Gradio would not change base packages (Gradio would otherwise upgrade `click` and downgrade `tomlkit`).
- The four tabs:
  1. **Clarification**: draft, 14-item checklist, 3–5 questions; the user confirms the spec.
  2. **Modelling**: Modeling Agent → validated model, synthetic or uploaded data, encoding check.
  3. **Solve**: SA / D-Wave / Fujitsu (or a local stand-in), live loop log, hardware budget.
  4. **Records**.
- Artifacts are saved under `results/clarify/` and `results/ui/<session>/`. The vLLM server must be running.

**One-time setup** (run from `QuantumQUBO Agent/`):
```bash
cd "QuantumQUBO Agent"
pip install -r requirements.txt          # base Python: front end + annealing loop
cp -n .env.example .env                  # -n: never overwrite an existing .env (it holds your keys)
# edit .env:  LLM_PROVIDER=local, LLM_MODEL=qwen3.6-35b-a3b, DWAVE_API_TOKEN=..., (FUJITSU_API_KEY=...)
```

**Start the local LLM** (needed by every agent and by the front end; keep it running in its own terminal):
```bash
conda activate /research/conda-envs/quantum-agent-local
bash scripts/serve_vllm.sh               # ready when the log shows "Application startup complete"
```

Credentials live only in `QuantumQUBO Agent/.env` (gitignored); never put them in code. In local mode, any model name
passed to the client (for example the original configs' `qwen/qwen3.6-35b-a3b`) is mapped to `LLM_MODEL`.

```bash
# Backends (P0): same QUBO, unified metrics; every run is appended to results/annealing_runs.jsonl
python scripts/run_backend.py --problem select_k --backend sa                       # local
python scripts/run_backend.py --problem select_k --backend sa --backend fujitsu_da  # Fujitsu (60-min cap; checked before sending)
python scripts/run_backend.py --problem select_k --backend fujitsu_da --simulate    # stand-in, no hardware
python scripts/run_backend.py --backend fujitsu_da --describe                       # tunable params
python scripts/run_backend.py --problem select_k --backend sa --penalty select_k=2 --param sa.num_sweeps=50   # deliberately weak penalty
python scripts/run_backend.py --problem qoblib_mis --name karate --backend sa       # QOBLIB instance
pytest                                                                              # offline tests (pytest.ini sets the path)

# Closed loop (P2/P3): --policy agent needs the vLLM server; SA stand-in by default
python scripts/run_loop.py --problem knapsack --n 14 --backend sa --policy agent               # multi-agent + RAG
python scripts/run_loop.py --problem knapsack --n 14 --backend sa --policy agent --mode single --no-rag
python scripts/run_loop.py --problem select_k_random --n 30 --k 6 --backend sa --policy rule   # baselines: default|rule|random|optuna
python scripts/run_loop.py --problem select_k --backend fujitsu_da --policy rule --simulate     # a backend's local stand-in

# Front end (NL -> QUBO) on the local LLM
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt
```

## Our Changes So Far

| File | Change |
|---|---|
| `docs/research_spec.md` | **New**: research specification v0.1 |
| `QuantumQUBO Agent/qubo_auto/annealing/` | **New**. P0: `problem.py` (per-constraint penalties, objective immutable), `backends.py` (adapter interface; SA, Fujitsu DA, D-Wave QPU), `guardrail.py` (`check_params`, config caps, hardware budget ledger), `feedback.py` (Feedback Extractor + JSONL log). Tools: `tools.py` (`verify_small`, `penalty_lower_bound`, `qubo_summary`, `scale_for_backend`). Problems: `examples.py` (`select_k`, `max_cut`, `select_k_random`, `knapsack`). P2: `loop.py` (workflow engine), `baselines.py`, `agents.py`. P3: `rag.py` |
| `QuantumQUBO Agent/qubo_auto/annealing/qoblib.py` | **New**: QOBLIB loader (Market Split, Max Independent Set) with solution checks |
| `QuantumQUBO Agent/qubo_auto/prompts/annealing/` | **New**: Configurator, Diagnoser, Refiner and single-agent prompts |
| `QuantumQUBO Agent/knowledge/` | **New**: curated RAG stores (backend docs, formulation patterns, tuning knowledge) with sources; entries measured by our own experiments are marked *verified* |
| `QuantumQUBO Agent/scripts/experiments/` | **New**: `exp_p2_compare.py` (P2 comparison), `exp_dwave_chain_strength.py` (knowledge-base experiment), `bench_vllm.py` (LLM throughput) |
| `archive/` | **New** (2026-10-04 cleanup): unused original-project files and superseded runs, listed in `archive/README.md` |
| `QuantumQUBO Agent/requirements.txt` | Added the annealing-loop dependencies |
| `QuantumQUBO Agent/config_loop.yaml`, `scripts/run_loop.py` | **New**: loop configuration and CLI |
| `QuantumQUBO Agent/scripts/run_backend.py` | **New**: run one QUBO on several backends |
| `QuantumQUBO Agent/tests/test_annealing.py` | **New**: 18 offline tests (P0, engine, guardrails, RAG, generators, MILP reference, QOBLIB) |
| `QuantumQUBO Agent/qubo_auto/llm_client.py` | Endpoint set by `LLM_PROVIDER` / `LLM_BASE_URL` / `LLM_API_KEY` instead of hard-coded OpenRouter; `LLM_ENABLE_THINKING` for `local` |
| `QuantumQUBO Agent/qubo_auto/pipeline.py` | `load_config` honours `LLM_MODEL` |
| `QuantumQUBO Agent/scripts/serve_vllm.sh` | **New**: vLLM on 2 × L40S |
| `QuantumQUBO Agent/.env.example`, `.gitignore` | Provider options documented; run log and `secrets.json` ignored |
| `README.md` | This overview; original README preserved below |

With no new environment variables set, the original pipeline behaves exactly as before.

---
---

# Original Project: QuantumQUBO Agent

> Everything below is the **original README of QuantumQUBO Agent**, kept as written by its authors
> (Niloy Kumar Mondal, Md Rizwan Parvez), MIT License. It documents the original pipeline, benchmarks,
> ablations and citation.
>
> **Moved in our 2026-10-04 cleanup:**
> `deliverables/`, `statistics.ipynb`, the original benchmark-suite results and the architecture figure are now under `archive/original_project/`
> (`scripts/ablation_scripts/` and `config_no_retries.yaml` were kept in place because the commands below use them).
>
> ⚠️ **If your `.env` is already configured, do not run `cp .env.example .env` below.** It would overwrite your keys; use `cp -n` instead.
> With `LLM_PROVIDER=local`, the OpenRouter model names below are served by the local model automatically.


### Automating Quadratic Unconstrained Binary Optimization (QUBO) Formulation Generation from Natural Language

[![ICML 2026 Workshop](https://img.shields.io/badge/ICML%202026-AI%20for%20Math%2C%20CS%20%26%20ML-blue?style=flat-square&logo=academia)](https://openreview.net/forum?id=9YTedapat4)
[![Paper](https://img.shields.io/badge/OpenReview-9YTedapat4-red?style=flat-square&logo=openreview)](https://openreview.net/forum?id=9YTedapat4)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Project Website](https://img.shields.io/badge/Website-QuantumQUBO%20Agent-purple?style=flat-square&logo=github)](https://quitttcat.github.io/QuantumQUBOAgent/)

> **Accepted** at the ICML 2026 Workshop: *AI as a Tool for Mathematics, Computer Science, and Machine Learning*
> [📄 Paper](https://openreview.net/forum?id=9YTedapat4) · [🌐 Project Website](https://quitttcat.github.io/QuantumQUBOAgent/)

---

![Architecture Diagram](archive/original_project/ICML-Page-5-1.png)


> **Note:** All commands below must be run from inside the `QuantumQUBO Agent/` directory.
> ```bash
> cd "QuantumQUBO Agent"
> ```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenRouter API key
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY=<your_key>

# 3. Run a single benchmark
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt

# 4. Run all benchmarks and save results to a named CSV
python scripts/run_benchmark_suite.py --seed 42 --name full_pipeline

# 5. Output: results/full_pipeline_42.csv  |  results/runs.jsonl  |  deliverables/  |  transcripts/
```

---




---

## Agents

| Agent | File | Stage | Role |
|---|---|---|---|
| Planner Agent | `agents/planner_agent.py` | 1 | NL → StructuredSpec |
| Judge Agent | `agents/judge_agent.py` | 1b, 2b | Correctness verification |
| Formulizer Agent | `agents/formulizer_agent.py` | 2 | StructuredSpec → QUBOFormulation |
| Debugger Agent | `agents/debugger_agent.py` | 2.5 | Plain-text → JSON test cases |
| Coder Agent | `agents/coder_agent.py` | 3 | QUBOFormulation → Python code |
| Writer Agent | `agents/writer_agent.py` | post | QUBOFormulation → LaTeX deliverable |

Each agent has a corresponding prompt in `qubo_auto/prompts/<agent_name>.txt`.

---

## Run Commands

### Single benchmark

```bash
# Run with plain-text test cases (Debugger builds JSON cases automatically)
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt

# Custom seed
python scripts/run_single.py benchmarks/knapsack --input benchmarks/knapsack/sample_cases.txt --seed 123

# Custom config file
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt --config config_no_retries.yaml
```

### Full benchmark suite

```bash
# Run all benchmarks — CSV saved to results/<name>_<seed>.csv
python scripts/run_benchmark_suite.py --seed 42 --name full_pipeline

# Run a subset of benchmarks
python scripts/run_benchmark_suite.py --seed 42 --name subset_run --benchmarks max_cut,knapsack,graph_coloring

# Skip specific benchmarks
python scripts/run_benchmark_suite.py --seed 42 --name full_pipeline --skip custom,travelling_salesman

# Preview list of benchmarks without running
python scripts/run_benchmark_suite.py --dry-run

# Use a different config (e.g. no-retries ablation)
python scripts/run_benchmark_suite.py --seed 42 --name no_retries_full_pipeline --config config_no_retries.yaml
```

---

## Ablation Studies

Four ablation configurations are provided under `scripts/ablation_scripts/`.

### A1 — No Judge (retries kept)

Removes both Judge verification gates (spec check + QUBO check). Planner, Formulizer, Debugger, Coder, and Test Runner are intact.

```bash
# All benchmarks
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42

# Single benchmark
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --benchmarks max_cut

# Multiple benchmarks
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --benchmarks max_cut,knapsack

# Custom CSV name
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --name ablation_no_judge

# No retries + No Judge combined
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --seed 42 --config config_no_retries.yaml --name no_judge_no_retries

# Preview
python "scripts/ablation_scripts/No Judge ablation study/run_no_judge.py" --dry-run
```

Results: `scripts/ablation_scripts/No Judge ablation study/results/<name>_<seed>.csv`

---

### A2 — No Retries (Judge kept)

Full pipeline with all retry budgets set to zero. One shot per stage — Judge gates are active but no repair budget.

```bash
# All benchmarks
python scripts/run_benchmark_suite.py --seed 42 --name no_retries_full_pipeline --config config_no_retries.yaml

# Single benchmark
python scripts/run_single.py benchmarks/max_cut --input benchmarks/max_cut/sample_cases.txt --config config_no_retries.yaml
```

Results: `results/no_retries_full_pipeline_42.csv`

---

### A3 — No Planner (NL direct to Formulizer)

Removes the Planner agent and both Judge gates. The raw NL problem goes directly to the Formulizer. Debugger receives NL + formulation output. Coder and Test Runner are intact.

```bash
# All benchmarks
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42

# Single benchmark
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --benchmarks max_cut

# Multiple benchmarks
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --benchmarks max_cut,knapsack

# Custom CSV name
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --seed 42 --name ablation_no_planner

# Preview
python "scripts/ablation_scripts/No Planner ablation study/run_no_planner.py" --dry-run
```

Results: `scripts/ablation_scripts/No Planner ablation study/results/<name>_<seed>.csv`

---

### A4 - Direct Method

## Caution : It reuses the json parsed test case generated by full architecture run. So that should be run first.

Single LLM call directly from the benchmark problem statement and example instance structures to `build_qubo(instance)`. No Planner, Formulizer, Judge, Debugger, or Coder pipeline stages are used.

```bash
# All benchmarks with the default configured model
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42

# All benchmarks with qwen/qwen3.6-35b-a3b
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --model qwen/qwen3.6-35b-a3b

# Single benchmark
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --benchmarks max_cut --model qwen/qwen3.6-35b-a3b

# Multiple benchmarks
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --benchmarks max_cut,knapsack --model qwen/qwen3.6-35b-a3b

# Custom CSV name
python "scripts/ablation_scripts/Direct Method/direct_method.py" --seed 42 --name ablation_direct_method --model qwen/qwen3.6-35b-a3b

# Preview
python "scripts/ablation_scripts/Direct Method/direct_method.py" --dry-run --model qwen/qwen3.6-35b-a3b
```

Results: `scripts/ablation_scripts/Direct Method/results/<name>_<model>_<seed>.csv`

---

### Ablation Summary

| Configuration | Planner | Judge | Retries | Success Rate |
|---|---|---|---|---|
| Full pipeline | ✓ | ✓ | ✓ | 68% |
| A1 — No Judge | ✓ | ✗ | ✓ | — |
| A2 — No Retries | ✓ | ✓ | ✗ | 35% |
| A1+A2 — No Judge, No Retries | ✓ | ✗ | ✗ | 36% |
| A3 — No Planner | ✗ | ✗ | ✓ | — |
| A4 - Direct Method | no | no | no | - |

---

## Configuration

### `config.yaml` (full pipeline)

```yaml
models:
  planner_agent:    qwen/qwen3.6-35b-a3b
  formulizer_agent: qwen/qwen3.6-35b-a3b
  judge_agent:      qwen/qwen3.6-35b-a3b
  coder_agent:      qwen/qwen3-coder-next
  debugger_agent:   qwen/qwen3.6-35b-a3b

temperatures:
  planner_agent:    0.0
  formulizer_agent: 0.0
  judge_agent:      0.0
  coder_agent:      0.0
  debugger_agent:   0.0

retries:
  restructure: 2    # Planner Agent max retries
  formulate:   2    # Formulizer Agent max retries
  code:        2    # Coder Agent max retries

verification:
  n_test_instances: 6   # test cases generated by Debugger
  max_n_vars:       12  # brute-force cap (cases above this are skipped)
  pass_threshold:   1.0 # fraction of test cases that must pass

limits:
  max_tokens_per_run: 200000
  use_cache: false
```

### `config_no_retries.yaml` (A2 ablation)

Identical to `config.yaml` but all retry budgets set to zero:

```yaml
retries:
  restructure: 0
  formulate:   0
  code:        0
```

Any OpenRouter model ID can be substituted. The pipeline is model-agnostic.

---

## Benchmark Folder Format

```
benchmarks/<name>/
    prompt.txt          ← natural-language problem statement
    sample_cases.txt    ← human-readable test cases (any format)
```

Test case JSON files are generated at runtime and written to `transcripts/<run_id>/cases/` — benchmark folders are never modified.

### Test case JSON schema

```json
{
  "name": "n4_example",
  "description": "...",
  "n_variables": 16,
  "instance": { "n_nodes": 4, "edges": [[0,1,1], [0,2,1]] },
  "ground_truth": {
    "optimal_value": -4.0,
    "optimal_bitstrings": [[0,0,1,1], [0,1,0,1]]
  }
}
```

`optimal_value` follows QUBO convention: negated for maximisation problems, raw for minimisation.

### Adding your own benchmark

1. Create `benchmarks/<name>/prompt.txt` with the problem statement in plain English.
2. Write `benchmarks/<name>/sample_cases.txt` with one or more test instances in any human-readable format.
3. Run:
   ```bash
   python scripts/run_single.py benchmarks/<name> --input benchmarks/<name>/sample_cases.txt
   ```

No code changes needed.

---

## Project Structure

```
qubo_auto/
├── qubo_auto/
│   ├── pipeline.py                   # Orchestrator — all stages + retry logic
│   ├── schemas.py                    # Pydantic models: StructuredSpec, QUBOFormulation, RunResult
│   ├── llm_client.py                 # OpenRouter wrapper, token + cost tracking, transcript logging
│   ├── agents/
│   │   ├── planner_agent.py          # Stage 1    : NL → StructuredSpec
│   │   ├── judge_agent.py            # Stage 1b, 2b : spec + QUBO correctness checks
│   │   ├── formulizer_agent.py       # Stage 2    : StructuredSpec → QUBOFormulation
│   │   ├── debugger_agent.py         # Stage 2.5  : plain-text → JSON test cases
│   │   ├── coder_agent.py            # Stage 3    : QUBOFormulation → Python build_qubo
│   │   └── writer_agent.py           # Post-success: LaTeX/HTML/PDF deliverable
│   ├── verification/
│   │   ├── brute_force.py            # Exhaustive 2^n search + Q matrix validation
│   │   ├── test_runner.py            # Run build_qubo vs ground-truth test cases
│   │   └── test_cases.py             # Load cases JSON into TestCase objects
│   └── prompts/
│       ├── planner_agent.txt
│       ├── judge_agent_spec.txt       # Judge prompt for spec verification
│       ├── judge_agent.txt            # Judge prompt for QUBO verification
│       ├── formulizer_agent.txt
│       ├── debugger_agent.txt
│       ├── coder_agent.txt
│       └── writer_agent.txt
├── benchmarks/
│   └── <name>/
│       ├── prompt.txt
│       └── sample_cases.txt
├── scripts/
│   ├── run_single.py                 # Run one benchmark
│   ├── run_benchmark_suite.py        # Run all benchmarks, write named CSV
│   └── ablation_scripts/
│       ├── No Judge ablation study/
│       │   ├── run_no_judge.py       # A1: pipeline without Judge gates
│       │   ├── results/              # CSV + logs
│       │   └── transcripts/
│       └── No Planner ablation study/
│           ├── run_no_planner.py     # A3: NL direct to Formulizer, no Planner/Judge
│           ├── results/
│           └── transcripts/
├── zero shot benchmark/
│   ├── QUBOBenchModified/            # Problem folders: prompt.txt + tests/case_N/
│   └── scripts/
│       └── direct_method_single.py       # Direct LLM → QUBO solve, no pipeline
├── transcripts/                      # Per-run LLM call logs (JSON, one file per step)
│   └── <run_id>/
│       ├── restructure_<ts>.json
│       ├── verify_spec_<ts>.json
│       ├── formulate_<ts>.json
│       ├── verify_formulation_<ts>.json
│       ├── parse_test_cases_<ts>.json
│       ├── code_<ts>.json
│       └── cases/                    # Generated test case JSON files
├── results/
│   ├── runs.jsonl                    # One JSON line per pipeline run
│   └── <name>_<seed>.csv            # Aggregated CSV from run_benchmark_suite.py
├── deliverables/
│   └── <benchmark_name>/
│       ├── <name>_<seed>.tex
│       ├── <name>_<seed>.html        # MathJax viewer (always generated)
│       └── <name>_<seed>.pdf         # Compiled PDF (requires pdflatex)
├── config.yaml                       # Full pipeline config
├── config_no_retries.yaml            # A2 ablation — zero retry budgets
└── .env                              # OPENROUTER_API_KEY
```

---

## Results Format

Each run appends one JSON line to `results/runs.jsonl`:

```json
{
  "run_id": "max_cut_42_a1b2c3d4",
  "benchmark": "max_cut",
  "seed": 42,
  "status": "success",
  "n_iterations": 2,
  "wall_time_s": 37.4,
  "models_used": {
    "planner_agent": "qwen/qwen3.6-35b-a3b",
    "formulizer_agent": "qwen/qwen3.6-35b-a3b",
    "coder_agent": "qwen/qwen3-coder-next"
  },
  "tokens_used": 9861,
  "cost_used": 0.0182,
  "verification": { "passed": 3, "total": 3 },
  "failure_modes": []
}
```

CSV columns written by `run_benchmark_suite.py`:

| Column | Description |
|---|---|
| `benchmark` | Benchmark folder name |
| `seed` | Run seed |
| `status` | `success` / `failed` / `script_failed` |
| `tests_passed` | Test cases passed |
| `total_tests` | Total test cases |
| `wall_time_s` | Wall clock time (seconds) |
| `tokens_used` | Total tokens consumed |
| `total_llm_cost` | Total LLM cost in USD |
| `planner_retries` | Stage 1 retry count |
| `formulizer_retries` | Stage 2 retry count |
| `coder_retries` | Stage 3 retry count |
| `debugger_retries` | Stage 2.5 retry count |

---

## Failure Modes

| Mode | Stage | Description |
|---|---|---|
| `spec_verification_failed` | 1b | Planner output missing variables or wrong objective |
| `formulation_verification_failed` | 2b | QUBO missing penalty terms or wrong objective direction |
| `coding_error` | 3 | Generated `build_qubo` raises exception or wrong Q shape |
| `formulation_error` | 3b | Q valid but argmin disagrees with ground truth |
| `token_budget_exceeded` | any | Run exceeded `max_tokens_per_run` |

---

## Running Tests (no API key needed)

```bash
pytest tests/
```

Covers: brute-force solver, code extraction, schema validation. No LLM calls made.

---

## Citation

If you use this code or the benchmarks in your research, please cite:

```bibtex
@inproceedings{mondal2026quantumqubo,
  title     = {Quantum{QUBO} Agent: Automating Quadratic Unconstrained Binary Optimization ({QUBO}) Formulation Generation from Natural Language},
  author    = {Niloy Kumar Mondal and Md Rizwan Parvez},
  booktitle = {ICML 2026 Workshop: AI as a Tool for Mathematics, Computer Science, and Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=9YTedapat4},
}
```

If you use the benchmark problems, also cite [QUBOBench](https://quitttcat.github.io/QUBOBench/).

---

## License

This project is released under the [MIT License](LICENSE).

© 2026 Niloy Kumar Mondal, Md Rizwan Parvez
