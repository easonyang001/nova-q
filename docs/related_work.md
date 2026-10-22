# Related work and external resources (collected 2026-10-04)

Working notes for the paper's related-work section and for the RAG knowledge base. Each item says what
it gives us and how close it is to our contribution (execution-feedback refinement of annealing
workflows across heterogeneous backends, budget-matched against baselines).

## Closest prior work: novelty check

| Work | What it does | Overlap with us | Our difference |
|---|---|---|---|
| **Practical Quantum CIM Empowerment via All-Domestic-Core Agentic Large Model** (Wang & Lu, [arXiv:2605.23934](https://arxiv.org/abs/2605.23934), Apr 2026) | LangGraph/LangChain LLM agents on a femtosecond-laser Coherent Ising Machine: QUBO/Ising calibration, **constraint-weight decision iteration**, validating published schemes | **High.** LLM iterates constraint weights with hardware in the loop | Single backend (CIM); the abstract reports no baselines, budget-matched comparison, guardrails or RAG. We cover several annealers (D-Wave, Fujitsu, CGA + SA stand-in), semantic-preserving guardrails, budget-matched baselines (rule, random, BO), multi/single and RAG ablations, and cross-backend transfer (RQ3). **Must be cited and positioned explicitly** |
| **QuantumQUBO Agent** (Mondal & Parvez, [arXiv:2609.10629](https://arxiv.org/abs/2609.10629)) | NL → QUBO multi-agent; lists hardware feedback as future work | Front end only | We build the hardware-feedback loop they list as future work |
| **LLM-QUBO** ([arXiv:2509.00099](https://arxiv.org/abs/2509.00099)) | End-to-end NL → QUBO → solve framework | Front end + solving | No execution-feedback refinement of solver/penalty configuration |
| **Automatic penalty weight tuning for PDPTW** ([Appl. Sci. 16(4):1690](https://doi.org/10.3390/app16041690)) | Feedback loop that adjusts constraint weights from intermediate solutions | Feedback on penalties | Hand-designed rule, single problem; corresponds to our rule-based baseline |
| **Formulation-Level Auto-Tuning for QUBO-based ML** ([arXiv:2607.18774](https://arxiv.org/pdf/2607.18774)) | Black-box optimisation of representation and penalty parameters | Penalty tuning as black-box optimisation | Corresponds to our BO (Optuna) baseline; no LLM, no backend-specific diagnosis |
| **ML-based penalty tuning, Minimum Bisection** ([arXiv:2509.19005](https://arxiv.org/abs/2509.19005)) | Gradient boosting predicts good penalty multipliers from graph features | Cold-start penalty choice | Relevant to our Configurator; needs training data per problem class |

## Penalty-weight methods (tuning knowledge, baselines)

- Verma & Lewis, *Penalty and partitioning techniques to improve performance of QUBO solvers*, Discrete Optimization 2020. Single-flip bound; our `penalty_lower_bound` is this quantity.
- Ayodele et al., *Exact and sequential penalty weights in QUBO with a digital annealer*, GECCO 2022 ([doi](https://dl.acm.org/doi/10.1145/3520304.3528925)). Sequential/adaptive weights on Fujitsu DA, a natural extra baseline.
- *Penalty Weights in QUBO Formulations: Permutation Problems* ([arXiv:2206.11040](https://arxiv.org/abs/2206.11040)).
- *Structure-Aware Penalty Weights Scaling for QUBO* ([Springer](https://link.springer.com/chapter/10.1007/978-3-032-23607-4_28)).
- Montañez-Barrera et al., *Unbalanced penalization* ([arXiv:2211.13914](https://arxiv.org/abs/2211.13914)), the formulation rewrite our Knapsack ceiling points to.

## D-Wave parameter studies (tuning knowledge)

- Pelofske, *Comparing three generations of D-Wave quantum annealers for minor embedded combinatorial optimization problems*, Quantum Sci. Technol. 2025 ([arXiv:2301.03009](https://arxiv.org/abs/2301.03009)): chain strength and annealing-time scans, Advantage2 at long anneals.
- *Benchmarking Chain Strength: An Optimal Approach for Quantum Annealing* ([NSF PAR](https://par.nsf.gov/biblio/10477832)).
- *Benchmarking Quantum Annealers with Near-Optimal Minor-Embedded Instances* ([arXiv:2405.01378](https://arxiv.org/abs/2405.01378)).
- Code: [HunpyoLee/OptimizeChainStrength](https://github.com/HunpyoLee/OptimizeChainStrength).

## Official documentation (backend knowledge)

- D-Wave: [QPU Solver Parameters](https://docs.dwavequantum.com/en/latest/quantum_research/solver_parameters.html), [QPU Configuration: Best Practices](https://docs.dwavequantum.com/en/latest/quantum_research/solver_configuration.html), [Basic QPU Configuration](https://docs.dwavequantum.com/en/latest/quantum_research/basic_config.html), white paper *Setting the Chain Strength* (14-1041A).
- Fujitsu: [Digital Annealer User's Guide](https://portal.aispf.global.fujitsu.com/apidoc/da/jp/da-guide-en.html) (8th ed., public); full parameter schemas are in the separate QUBO API V3 reference. [Fixstars Amplify Fujitsu client docs](https://amplify.fixstars.com/en/docs/amplify/v1/clients/fujitsu.html) list DA3/DA4 parameters.
- OpenJij / JijModeling ([tutorial](https://tutorial.openjij.org/en/intro.html)): an open-source SA/SQA stand-in with custom schedules, a possible second simulator.

## Benchmarks

- **QOBLIB** (Koch et al., Nature Comput. Sci., [arXiv:2504.03832](https://arxiv.org/abs/2504.03832)). Integrated (`qoblib.py`). No annealer submissions in classes 01/02/03/07.
- Quantum-inspired solver benchmark: D-Wave Hybrid vs. Toshiba SBM vs. Fujitsu DA ([arXiv:2104.14096](https://arxiv.org/abs/2104.14096)).

## How these are used

- `knowledge/backend_docs.jsonl`: `bd-dwave-reads-vs-time`, `bd-dwave-autoscale`, `bd-dwave-srt-postprocess`, `bd-dwave-schedule`, `bd-fujitsu-guide`.
- `knowledge/tuning.jsonl`: `tk-chain-literature`, `tk-annealing-time-tts`, `tk-penalty-methods`.
- All entries are short summaries in our own words, with the source cited. No documentation text is copied wholesale.
