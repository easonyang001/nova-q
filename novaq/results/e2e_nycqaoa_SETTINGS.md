# NYC participatory budget solved end to end with QAOA on CUDA-Q (2026-10-23)
- Same as e2e_nycpb2 (Gemma doer+checker, thinking on, 1200 s, 64000 tokens, agent data prep) with the solver
  QUBO_BACKEND=qaoa_cudaq (CUDA-Q 0.14, nvidia target; default policy p=2, 10^4 shots, 3 restarts, 200 COBYLA iters).
- Code af418a9 (audit fixes incl. gcd slack reduction: the NYC QUBO is 25 qubits).
- Each batch pins its QAOA to its own GPU (CUDA_VISIBLE_DEVICES=0 / 1); the LLM servers stay on 8000 / 8001.
- Stop rule (fixed before start): first 3 finished runs = x/3; delivery.solver_raw tells whether QAOA's own best sample
  (before the classical exchange polish) was the delivered plan.
