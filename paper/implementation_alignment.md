# 論文與目前程式對照（2026-10-11）

依目前工作目錄實作更新，不依 Git 狀態推定檔案作者。本次沒有修改程式或提交硬體工作。

| 程式（相對 QuantumQUBO Agent/） | 論文更新與證據界線 |
| --- | --- |
| qubo_auto/annealing/schema.py | System：sets／params、索引、來源、單位、衍生工具。形狀正確不代表語意正確。 |
| qubo_auto/annealing/datacheck.py、adapter.py | System：選配的獨立數量／總量核對；未解決不一致交付審閱。 |
| qubo_auto/annealing/reqcheck.py | System：決策身分、可行性、排序與交付檢查。抽樣不能證明全模型正確。 |
| qubo_auto/annealing/platforms.py、opendata.py | System：CKAN、Socrata、Opendatasoft、ArcGIS、Europa、WFS；未宣稱全面實測。 |
| qubo_auto/annealing/prep/join.py、table.py、situations.py | System：行政父層補值、複合鍵、重複與未配對回報。 |
| qubo_auto/annealing/modeler.py、knowledge/modeling_patterns.jsonl | System：至多三個通用建模模式檢索；與舊調參 RAG 結果分開。 |
| qubo_auto/annealing/repair.py | System：契約失配返回模型、跨階段修復與次數限制；未宣稱修復成功率。 |
| scripts/ui.py | System／Experiments：預設 default；沒有據此宣稱自動切換 policy。 |
| scripts/run_e2e_batch.py、e2e_direct.py、acceptance/cases.json | Experiments：三題、五個預設比較組，三次重複，共 45 次；加入釐清階段不查網路的比較組後 54 次。 |
| acceptance/judge.py、truth.py | Experiments：外部資料／模型／交付驗收；真值不提供給 runtime agents。 |

新驗收題為台北充電站、YouBike 路線、AED 分組；充電站保留全部里，禁止抽樣。results/e2e_smoke* 有成功與失敗紀錄，但跨版本，不能合併成成功率。direct 與 pipeline 的工具與下載上限不同，不能把比較差異單獨歸因給 agent 數量。

保留既有實驗數字與調參結果置於附錄的結構。先前 120 項測試保留為歷史快照；本次 python -m pytest -q 為 146 passed。測試通過不等於真實資料端到端驗收通過。

下一個主要實驗：凍結程式、提示與真值後完成同一批次，分別報告資料正確率、模型正確率、可用交付率、聯合成功率，以及 checker／repair／建模 RAG 的增量效果。

附錄精簡：僅保留重現設定、統計方法與簡短功能驗證；關鍵後端比較及資料案例移至正文。先前完整附錄另存 supplementary/solver_details.pdf，原始詳細實驗檔案保留但不再納入論文。
