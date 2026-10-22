| Policy | Runs | Mean gap | Optimal (gap=0) | Mean solver calls | Mean LLM tokens | Fallback rounds | Crashes |
|---|---|---|---|---|---|---|---|
| default | 4 | 0.0477 | 2/4 | 2.50 | 0 | 0 | 0 |
| rule | 4 | 0.0477 | 2/4 | 3.00 | 0 | 0 | 0 |
| random | 4 | 0.0570 | 1/4 | 5.25 | 0 | 0 | 0 |
| optuna | 4 | 0.0507 | 1/4 | 5.25 | 0 | 0 | 0 |
| agent-multi-rag | 4 | 0.0520 | 2/4 | 3.75 | 42928 | 0 | 0 |
| agent-single-rag | 4 | 0.0406 | 2/4 | 4.25 | 32214 | 0 | 0 |
| agent-multi-norag | 4 | 0.0384 | 2/4 | 5.00 | 39098 | 0 | 0 |

Paired Wilcoxon signed-rank on gap (same instance and seed), two-sided:

| Comparison | n pairs | Mean gap A | Mean gap B | p-value |
|---|---|---|---|---|
| agent-multi-rag vs default | 4 | 0.0520 | 0.0477 | 0.5 |
| agent-multi-rag vs rule | 4 | 0.0520 | 0.0477 | 0.5 |
| agent-multi-rag vs random | 4 | 0.0520 | 0.0570 | 0.75 |
| agent-multi-rag vs optuna | 4 | 0.0520 | 0.0507 | 0.75 |
| agent-multi-rag vs agent-single-rag | 4 | 0.0520 | 0.0406 | 1 |
| agent-multi-rag vs agent-multi-norag | 4 | 0.0520 | 0.0384 | 0.5 |
| agent-single-rag vs default | 4 | 0.0406 | 0.0477 | 1 |
| agent-single-rag vs rule | 4 | 0.0406 | 0.0477 | 1 |
| agent-single-rag vs random | 4 | 0.0406 | 0.0570 | 0.25 |
| agent-single-rag vs optuna | 4 | 0.0406 | 0.0507 | 0.5 |
| agent-single-rag vs agent-multi-rag | 4 | 0.0406 | 0.0520 | 1 |
| agent-single-rag vs agent-multi-norag | 4 | 0.0406 | 0.0384 | 1 |
| agent-multi-norag vs default | 4 | 0.0384 | 0.0477 | 0.5 |
| agent-multi-norag vs rule | 4 | 0.0384 | 0.0477 | 0.5 |
| agent-multi-norag vs random | 4 | 0.0384 | 0.0570 | 0.5 |
| agent-multi-norag vs optuna | 4 | 0.0384 | 0.0507 | 0.25 |
| agent-multi-norag vs agent-multi-rag | 4 | 0.0384 | 0.0520 | 0.5 |
| agent-multi-norag vs agent-single-rag | 4 | 0.0384 | 0.0406 | 1 |
