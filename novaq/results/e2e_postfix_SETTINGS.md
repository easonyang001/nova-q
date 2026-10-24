# Post-fix validation (2026-10-24), reported apart from the frozen test set
- Code: main after af418a9 + metadata-download fix. Same config as the test set (Gemma doer+checker, thinking on,
  1200 s, 64000 tokens, agent data prep, SA).
- Purpose: check in full runs that (A) Singapore now finds the official data.gov.sg Hawker Centres layer and (F) the
  reasoning-loop detection keeps route modelling short. Cases sg_hawker_groups, paris_velib_tour; 1 repeat per GPU.
