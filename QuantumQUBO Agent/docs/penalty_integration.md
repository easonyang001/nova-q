# Penalty 搜尋整合版

本版把三個專案值得借鑑的部分接到同一個 `run_loop`；資料取得、預處理、模型修復與既有 backend 仍走原本 workflow。

| 來源 | 實際整合內容 | 使用方式 |
|---|---|---|
| [PyQUBO](https://pyqubo.readthedocs.io/en/latest/reference/model.html) | `Placeholder` 替換權重、`Constraint` 標籤、`decode_sample` 診斷 | **直接使用已安裝的 PyQUBO 套件** |
| [QHyper](https://qhyper.readthedocs.io/en/main/user_guide/solver_configuration.html) | penalty 視為搜尋參數、Grid / CEM 搜尋 | 借鑑搜尋設計，適配本專案逐輪 proposal API；**沒有直接呼叫 QHyper SDK** |
| [Kaiwu](https://kaiwu-sdk-docs.qboson.com/en/source/kaiwu.solver.html) | 逐步加重違反限制、累計可行結果停止、保存可行權重和 checkpoint | 借鑑 penalty-method 控制層；**沒有直接呼叫 Kaiwu SDK** |
| [Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) | Bayesian penalty 搜尋 | **直接使用 Optuna TPE**，兩次 startup 後開始適應回饋 |

Grid / CEM 是本地適配實作，並非把 QHyper 原始碼貼進來；因此沒有把整套 QHyper / Kaiwu 加成必要依賴。現成搜尋器使用 Optuna。後續若需要直接接這兩個 SDK，應另加其版本相依的 adapter。

## 如何使用

UI 的「誰來調參數」新增四個選項。預設是 `default`：先用預設 penalty 求解；有違反限制或品質不足的證據時再調參。若原因是錯資料、漏限制或需求衝突，應回上游處理，而不是調 penalty：

- `penalty_tpe`：只搜尋 penalty，solver 使用既有受限制的預設參數；與原本同時調 solver 的 `optuna` 分開。
- `penalty_grid`：依 logarithmic grid 逐組搜尋；lazy product 和候選上限避免一次生成巨大網格。
- `penalty_cem`：以原問題可行性及原目標排序，分批挑 elite 更新 log-weight 分布；有標準差下限及有限次 rejection。
- `penalty_sequential`：每輪將仍有違反的 penalty 加倍，上限由 `max_multiplier` 控制。

所有策略共用既有 SA / D-Wave / Fujitsu / CGA backend、guardrails、原問題評估與 solver-call budget；此版直接驗證的是本機 SA，沒有新增真實硬體測試。

`config_loop.yaml` 的 `penalty_search.compiler: pyqubo` 啟用編譯器。每種 encoding 組合只編譯一次，其後由 `feed_dict` 更新係數。`native` 可切回原編譯器。變數使用明確的 `x_0` → index 0 映射，保留 decision / slack 位置及完全沒出現在 Hamiltonian 中的變數。

```bash
python scripts/run_loop.py --problem knapsack_ir --n 12 --backend sa \
  --policy penalty_cem --no-rag --checkpoint /tmp/penalty.json
python scripts/resume_penalty.py /tmp/penalty.json --max-calls 20
```

Workflow / UI 會自動存到 session 的 `checkpoints/`，執行紀錄中附檔案路徑。UI 可在「工作進度與接續」重新整理、選擇同版本的紀錄，再按接續；使用保存的策略及設定，只調高累計求解次數上限。沒有選擇紀錄時會指出缺口，不會另啟工作。命令列也可使用上述 resume 指令。

## 診斷、停止和恢復

`encoded_penalty_diagnostics` 記錄每輪最佳可行 sample 的 PyQUBO 診斷；沒有可行 sample 時診斷最低能量 sample。原問題 `constraint_violation_rates` 仍是判斷可行性的依據：slack 不正確時 penalty 可非零，但 decision 仍可行。`unbalanced` 是近似 encoding，不使用「penalty 等於零」當限制檢查。

`loop.stop_after_feasible_count` 預設 `null`；例如設成 `2`，累計兩輪含有可行解即停止。其他停止條件仍可提早結束。它不是品質保證；需要改善品質時保留預設，以 baseline / 原目標評估。penalty floor 仍是原本 heuristic，沒有新增全域最佳解保證。

Checkpoint 保存完整 IR、設定、每輪回饋、可行 penalty 組合、求解次數與停止原因。四個新 penalty 策略由歷史重建搜尋狀態，接續不重跑已完成的求解。模型、backend、策略、seed 或其他設定不同會拒絕接續；允許提高**總** call budget。既有 Agent / 舊 Optuna 策略及任意 callable ProblemInstance 不支援接續，仍可保存紀錄。

在提交 solver 前先標記 pending，回傳並保存回饋後解除。如果程序在 solver 呼叫期間中斷，下次恢復會指出未完成的呼叫，要求先檢查 backend 紀錄，以免直接重複送件。這不是外部 backend 的交易機制，無法保證硬體端 exactly-once。

## 和參考專案的區別

參考的 penalty 搜尋在本專案中只負責「下一組權重」。外層還會檢查資料 / 模型、辨識已證實的不可能需求、必要時回到上游修復、保存模型資料版本，並用便宜可行 baseline 評估及交付成果。這是本專案完整閉環的差異；不把調 penalty 當成所有失敗的修復方式。

## 本次驗證

本機完整測試、UI 可行／不可行交付與 1→3 次 checkpoint 接續已驗證。144 次新 penalty 比較沒有崩潰，但平均 gap 未勝過預設策略；詳見 [結果解讀](../results/experiments/penalty_integration_20261011/interpretation.md)。資料／模型修復的受控比較另存於 [workflow recovery](../results/experiments/workflow_recovery_20261011/summary.md)，修復 callback 使用已知答案，不能代表 LLM 的自動診斷成功率。
