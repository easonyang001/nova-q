# D-Wave Hybrid：10 秒受控比較

使用實際 Leap Hybrid BQM 服務 `hybrid_binary_quadratic_model_version2p`，不是直接 QPU，也不是 SA 模擬。與原本 SA／CGA 比較沿用相同凍結模型：紅酒、60-item knapsack、80-item conflict selection。

## 預算與時間

- 使用者指定 `time_limit=10` 秒；每次提交前檢查服務對該模型的最低時間，不足就拒絕，不自動加時。
- default、penalty_tpe、agent，各 3 次外部重複；27 個條件，每條件最多 3 calls。
- 整批最多 81 calls、810 秒，使用累計硬體 ledger，順序提交避免預算競爭。帳戶已有的使用量不會被重設。
- 每次連線 timeout 30 秒、polling timeout 120 秒；與服務 time_limit 分開。提交後的外部錯誤停止整批，不盲目重送。
- 第一個執行程序的 ledger 將每次保守保留 10 秒。Hybrid 的 timing 位於 SampleSet.info 頂層；唯讀答案核對另存實際 charge_time、run_time、qpu_access_time，保留原始紀錄。
- Timing 按微秒換算；charge_time 是服務計費時間，qpu_access_time 是 Hybrid 內部的 QPU 使用量，兩者分開報告。[官方計時定義](https://docs.dwavequantum.com/en/latest/industrial_optimization/index_hybrid.html)。
- CGA 使用 3 秒，SA 使用 reads/sweeps；跨後端時間不代表相同算力，不能以此宣稱硬體優勢。

時間參數依據 [D-Wave 官方 sample 文件](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/dwave.system.samplers.LeapHybridSampler.sample.html)；[min_time_limit 文件](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/dwave.system.samplers.LeapHybridSampler.min_time_limit.html) 說明依問題規模計算最低時間。

## 執行及報告

從專案子目錄 `QuantumQUBO Agent` 執行。已完成條件會沿用保存結果，不再次提交；需 `.env` 的 D-Wave 與 LLM 設定。

```bash
python scripts/experiments/exp_dwave_hybrid_paper.py
python scripts/experiments/audit_dwave_hybrid_timing.py
python scripts/experiments/summarize_dwave_hybrid.py
python scripts/experiments/summarize_paper_results.py
python scripts/experiments/augment_paper_report.py
python scripts/experiments/update_paper_evidence.py
```

前一支腳本會提交付費／耗額度工作；timing audit 僅查詢已完成 problem IDs，沒有新增求解提交。結果、設定、每輪 feedback、服務 problem ID、checkpoint 與原始 timing 保存於 `results/experiments/dwave_hybrid_10s_20261011/`。完整圖表見 `results/paper_evidence_20261011/report.html`。
