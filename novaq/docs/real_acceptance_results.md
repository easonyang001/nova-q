# 真實資料驗收結果（2026-10-11）

10 題規格見 [acceptance_cases.md](acceptance_cases.md)。第 7 題有一次全新流程成功；第 8、10 題經開發修正、模型沿用及獨立檢查後完成交付。尚未完成全部 10 題的固定條件重複驗收。[完整論文實驗與成果報告](../results/paper_evidence_20261011/report.md) 包含求解器、編碼與較大路線實驗。

## 全新紅酒流程

[成果](../results/acceptance/wine_live_20261011_fresh2/delivery.md) · [驗收 JSON](../results/acceptance/wine_live_20261011_fresh2/acceptance.json)

指定 [UCI 官方資料頁](https://archive.ics.uci.edu/dataset/186/wine+quality)，由 scout 發現實際 ZIP 連結並下載。Qwen3.6-35B-A3B 寫模型及 adapter；PyQUBO 編譯、Optuna TPE 調 penalty、既有 local SA 求解。沒有接真實量子／退火硬體。約 64 秒、20,596 LLM tokens，4 次 solver calls。紀錄中的費用 0 代表本地服務未回報計價，不表示運算沒有成本。

- 原始紅酒 CSV：1,599 筆、11 個理化特徵；quality 不得作為候選特徵。
- 固定設定：恰選 5 個；目標為相關總和減去 0.5 倍兩兩相關總和。
- 對所有特徵重算原始資料相關矩陣；對全部 2,048 組二元決策獨立核對目標和可行性。
- 最終選擇：volatile acidity、residual sugar、total sulfur dioxide、sulphates、alcohol。
- 原目標 0.7497088246710516，等於本題獨立枚舉最佳值。交付來源為 annealer。
- 保存原檔、SHA256、來源頁／下載 URL、時間、資料 metadata、確認規格、版本、checkpoint、分數 CSV 和成果圖。

這個小問題可枚舉，是驗收檢查工具；其他題目仍以原限制與便宜可行基準驗收，不要求都有最佳解。

## 失敗與接續紀錄

`results/acceptance/` 保留每個嘗試：

| 目錄 | 結果 |
|---|---|
| wine_live_20261011 | 生成模型的 REQUIREMENT_MAP 指向不存在的限制，未求解 |
| wine_live_20261011_v2 | 官方 ZIP 被 title-only ranking 判為不相關，轉試失敗的 GitHub，未求解 |
| wine_live_20261011_v3 | ZIP 下載與預處理成功，但 adapter 沒有 metadata，未求解 |
| wine_live_20261011_v4 | 初次因描述被寫成 assumptions 停下；獨立驗證這些固定描述後，接續快取完成 |
| wine_live_20261011_fresh | 再次出現錯誤限制名稱對應，未求解 |
| wine_live_20261011_fresh2 | 回饋列出實際限制名稱後，全新流程完成；沒有快取／模型沿用或待確認項目 |

修正措施：指定來源優先、頁面描述和來源 URL 納入 ranking、CSV ZIP／分號格式讀取、缺少 ADAPTER_METADATA 必須重試、assumptions 僅記未確認的選擇、限制對應錯誤直接回饋實際名稱。

不能把這些修正後的開發嘗試當作預先固定條件的成功率評估。此驗收還包含人工寫好的確認規格與題目專用獨立檢查器；沒有證明任意自然語言問題、任意網站搜尋或模型語意都能自動通過。

## 其他驗證

- 自動測試：120 項通過；涵蓋契約、來源對應、需求漏列、同名界限被改、錯誤修復、快取失效和接續。
- UI：可行交付、不可能需求 0 次求解、刷新與 1→3 次累計接續、未選 checkpoint 的提示。
- [新 penalty 比較](../results/experiments/penalty_integration_20261011/interpretation.md)：144 次執行，沒有崩潰；新策略未全面優於 default。
- [受控修復比較](../results/experiments/workflow_recovery_20261011/summary.md)：有修復 callback 時 12/12 可修復案例交付，3/3 擅改設定被拒；callback 還原已知正確版本，非 LLM 自動修復能力。

## 新增真實成果

- [批發客戶分群](../results/acceptance/wholesale_live_20261011_v3/delivery.md)：官方 440 筆資料固定抽樣 24 人，3 組各 8 人；分群成本由可行基準 1021.7818 降至 856.8319，改善約 16.1%。沿用生成模型、重新取得資料與生成 adapter，沒有證明最佳解。
- [社團平衡分割](../results/acceptance/karate_live_20261011_v4/delivery.md)：官方 34 節點、78 邊，兩組各 17 人。退火最好切斷 22 條邊，交付較好的基準 17 條邊。保存下載碰撞、欄位排序錯誤與修復紀錄；沿用模型並修正 adapter，沒有證明最佳解。
- 另三次紅酒生成結果在驗收器修正後使用保存的模型／資料重新核對並交付；不能將事後重查稱為固定驗收條件的 100% 成功率。

## 重跑

從 `QuantumQUBO Agent` 目錄執行：

```bash
python scripts/run_real_acceptance.py --out results/acceptance/my_new_run
```

需要本機 LLM 設定和官方資料連線。`--model-file` 可沿用先前模型但仍重新驗證；同目錄的資料快取僅在需求、程式、備註和檔案雜湊一致時使用。資料語意未知仍需確認，不會為了可行解自行補造。
