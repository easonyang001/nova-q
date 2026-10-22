# 完整流程回復與 penalty 調參參考

本次目標是讓流程可完成、可修正、可交付，不要求所有應用題都有已證明的最佳解。

## 現在的操作

1. 確認需求、產生模型。
2. 在「建模與資料」選「自動備妥真實資料」。系統先使用已下載／上傳的可讀檔案，預處理失敗時把錯誤帶回 adapter；仍失敗則搜尋其他相關來源，下載失敗換候選。搜尋次數和下載數有上限。
3. 轉換沒有已知問題就保存為可求解版本；涉及缺值假設、模擬資料或矛盾時顯示問題，保留先前版本。既有手動選資料與核准方式也可使用。
4. 「開始求解」先做必要條件檢查及限時可行性檢查。已確認模型無解時不提交退火機；檢查逾時是 unknown，仍可求解。
5. 求解結果有具體 formulation 錯誤證據時，可回到 modeler。資料／模型矛盾則交給 router 判斷是否有可修正錯誤；若只能靠改需求解決，交付缺口與下一步，不自行放寬需求。
6. 最後區分可用、品質有限與沒有可行方案。退火結果落後已驗證可行的簡單基準時，保留並交付基準，標明來源。

## 實作位置與限制

- `assessment.py`：HiGHS 限時基準與可行性檢查；二次目標先求可行方案，再做固定次數的一／二變數局部改善。這是一般基準，不保證比每個問題專用 heuristic 好。
- `workflow.py`：外層回復；總 solver-call 預算跨模型／資料修正共用。修正後重新驗證，不跨版本直接比較 objective。會逐條檢查 REQUIREMENT_MAP；修復同名限制時，公式、係數、界限或目標變更也會被攔下，要求重新確認。使用者固定輸入不可藉改資料偷偷放寬。這仍不是自然語言公式的完整證明。
- `acquisition.py`：利用既有開放資料搜尋與 adapter，提供限次替代來源路徑。不能保證任意網站或任意應用題都找得到資料。
- `artifacts.py`：不可變的模型＋資料＋來源資訊版本、原子 active pointer、事件紀錄。可在 UI 還原版本，或接續四種 history-based penalty 搜尋 checkpoint。資料快取按模型、需求、備註和檔案雜湊核對；task_state.json 保存目前階段與下一步。既有 Agent／joint Optuna 內部狀態仍不支援接續。
- `runtime.py`：生成的模型函式與 adapter 在獨立程序執行，預設 30 秒逾時終止，UI 執行緒也適用。這是防卡住／崩潰的程序隔離，不是檔案或網路的安全沙箱。
- 成果綁定精確版本，圖表／說明生成失敗仍保存原始可行方案。

形式檢查與 LLM router 無法證明自然語言需求完全被正確建模；保留使用者確認規格、可檢視模型／處理紀錄與回退功能。自動修正採有證據且不更改需求的路徑。

## 可參考的現成 penalty 方法（查閱日期：2026-10-11）

| 工具／方法 | 可沿用的部分 | 與本專案的關係 |
|---|---|---|
| [QHyper HyperOptimizer](https://qhyper.readthedocs.io/en/main/user_guide/solver_configuration.html)／[程式庫](https://github.com/qc-lab/QHyper) | 將 penalty weights 當超參數，用 GridSearch 或 CEM 搜尋 | 最直接的現成調參參考；可包成 policy，用相同 solver 預算比較。不是資料／建模回復控制器 |
| [Kaiwu 自動 penalty 估計](https://kaiwu-sdk-docs.qboson.com/en/source/auto_penalty.html) | `get_min_penalty(obj, cons)` 估計初始懲罰係數 | 可參考 cold start 的權重估計；不應把單一翻轉的估計直接宣稱為全域最優保證 |
| [Kaiwu PenaltyMethodSolver，v1.1.2](https://kaiwu-sdk-docs.qboson.com/en/source/kaiwu.solver.html)／[新版 FAQ](https://kaiwu.qboson.com/docs/FAQ.html) | penalty method、可行解停止條件、checkpoint；新版介面稱 `hybrid.PenaltyMethodOptimizer` | 可參考自動懲罰回圈與接續執行。不同版本 API 不同，接入前需在隔離環境核對版本與授權 |
| [PyQUBO Placeholder／Constraint](https://pyqubo.readthedocs.io/en/latest/reference/express.html) | 編譯一次，把 penalty 當 placeholder 替換，回報 violated constraints | 建模與診斷工具；本身不等於自動搜尋最佳 penalty |
| [Exact and Sequential Penalty Weights … with a Digital Annealer](https://ore.exeter.ac.uk/rest/bitstreams/185242/retrieve)，Diez García、Ayodele、Moraglio，GECCO 2022 | exact／sequential penalty 方法，與 Fujitsu DA 直接相關 | 文獻與額外 baseline 候選；此次找到作者機構提供的論文，未確認有可直接使用的公開程式庫 |

PyQUBO 和 Optuna 已實際安裝並接入。Grid／CEM 與 sequential 控制借鑑 QHyper／Kaiwu，屬本地適配，沒有直接呼叫其 SDK。詳見 [整合說明](../QuantumQUBO%20Agent/docs/penalty_integration.md)。

建議把差異定位為：**使用既有 penalty 調參方法，但根據執行證據決定繼續調參、修資料、修模型，或回報需求缺口，並保存可交付成果與完整版本紀錄。** 是否構成研究新意還需要更完整的相關研究比較；不能僅因增加 LLM／agent 就宣稱新方法。

## 驗證

離線測試涵蓋：正常 SA 求解、資料來源下載失敗後替換、建模錯誤回修、容量不足與限制矛盾不提交退火機、檢查逾時保持 unknown、落後基準時保留可行基準、版本還原、生成程式在 UI 執行緒卡住後終止。

外部資料搜尋、實際 LLM 路由及真實退火硬體需要另外做連線驗收；離線測試不代表這些外部服務已完成整條實測。

## 資料契約與真實案例

DATA_CONTRACT 核對欄位、單位、期間、地區、粒度、缺值、範圍和實際下載來源對應。來源紀錄由程式記錄 SHA256、取得時間與發布頁面；未觀察到授權時標待確認，不自行宣稱政府資料授權。缺少 metadata 先讓 adapter 修正；真正未知的語意留待確認。LLM 網路請求預設 180 秒逾時。指定發布者的實際下載連結優先，仍須通過內容驗證。

[10 題驗收規格](../QuantumQUBO%20Agent/docs/acceptance_cases.md)包含原始需求檢查、便宜基準與故障注入。紅酒 pilot 已取得 1,599 筆真實資料、獨立核對相關矩陣與 2,048 組決策，再交付 5 個特徵；保存失敗與接續紀錄。這是指定官方來源的一題實測，不是十題驗收通過率，也不是不限來源的自主搜尋成功率。

後續全新紅酒流程也成功，沒有沿用模型或資料快取，約 64 秒、20,596 LLM tokens、4 次 SA 求解。成果包含名單、分數 CSV、圖表；詳細紀錄見 [真實驗收說明](../QuantumQUBO%20Agent/docs/real_acceptance_results.md)。
