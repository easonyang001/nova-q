# NYC participatory budget solved end to end with QAOA on CUDA-Q (2026-10-23)
- Same as e2e_nycpb2 (Gemma doer+checker, thinking on, 1200 s, 64000 tokens, agent data prep) with the solver
  QUBO_BACKEND=qaoa_cudaq (CUDA-Q 0.14, nvidia target; default policy p=2, 10^4 shots, 3 restarts, 200 COBYLA iters).
- Code af418a9 (audit fixes incl. gcd slack reduction: the NYC QUBO is 25 qubits).
- Each batch pins its QAOA to its own GPU (CUDA_VISIBLE_DEVICES=0 / 1); the LLM servers stay on 8000 / 8001.
- Stop rule (fixed before start): first 3 finished runs = x/3; delivery.solver_raw tells whether QAOA's own best sample
  (before the classical exchange polish) was the delivered plan.
- RESULT 1: g1 r0 PASS (36.7 min): QAOA's own sample (no polish) = exact optimum (7 projects, $980,000, 7,103 votes);
  feasible fraction 0.78 of 10^4 shots; solve stage 140 s.
- RESULT 2: g0 r0 PASS (39.6 min): same; QAOA raw = optimum, feasible fraction 0.77. Score so far 2/2.
- RESULT 3: g0 r1 PASS (30.8 min): QAOA raw = optimum, no polish; solve 132 s. FINAL (stop rule): 3/3, every delivery
  was QAOA's own best sample (polished = false). Remaining run stopped.
