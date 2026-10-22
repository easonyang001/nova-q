# QCNC 2027 conference draft

此版本獨立於原本的長版論文，按 2026-10-11 查到的 QCNC 2027 投稿規定製作。稿件為英文，IEEE Computer Society conference 格式、雙欄、10pt、US Letter，作者欄與 PDF author metadata 為 Chia Chen Yang。包含圖與參考文獻共 8 頁，沒有附錄。

官方規定：
- https://www.qcnc.org/2027/submission.php — 審查稿最多 8 頁、雙盲、IEEE Computer Society Proceedings 格式。
- https://www.qcnc.org/2027/track2.php — Track 2 包含 AI-assisted quantum computing 與 hybrid quantum-classical systems；此為建議投稿方向，並非聲稱委員會已確認適合。

## 稿件定位

聚焦本地 LLM 如何協調資料取得、預處理、建模、驗證、求解與交付。三項貢獻是可檢查的端到端流程、分離資料／模型／解的驗證方式，以及分開呈現不同階段的實驗證據。Penalty 搜尋是配套，不主張新搜尋演算法。

保留既有 benchmark、合成端到端 suite、CGA／D-Wave Hybrid 與指定資料來源的案例數字。模型、資料、預算不同的實驗不合併為同一成功率。新資料／需求 Checker 與建模 RAG 有程式實作，但凍結版本的自主驗收比較尚未完成，稿中列為未完成的評估，沒有寫成已驗證的提升。Direct LLM 與結構化流程的比較也不被解讀為純 agent 數量消融。

圖表使用描述性名稱，沒有程式內部實驗組代號。新結果圖的後端數值讀取既有統計 JSON；benchmark 計數保留原長版的 69／72／70 與 4／6／6。

## 編譯

```bash
./build.sh
```

編譯需 IEEEtran、TikZ、pdfLaTeX 與 BibTeX；圖檔已包含，不需另外畫圖。`figures/plot_evidence.py` 在完整專案中可重新讀取 `results/paper_evidence_20261011` 重畫數值圖；使用單獨的原始碼包時可直接使用提供的 PDF 圖檔。

檔案：main.tex、sections/、figures/、references.bib。`validation.json` 記錄版面與作者 metadata 檢查。這是依現有證據完成的會議稿，尚未提交；若之後取得新自主驗收結果，應更新摘要、實驗與結論後再投稿。

2026-10-12 圖像修訂：流程與驗證圖改為自製向量插圖，使用角色分支、獨立計算的匯合、資料語意與修復回路註解。PDF 保留可選取文字，SVG 可再編輯；既有實驗數字未改動。

2026-10-12 skill 圖像修訂：架構圖先以影像生成探索構圖，再重建原生 Draw.io 與 SVG。圖中使用示意覆蓋問題連結需求、資料、限制式與成果；PDF 從 SVG 匯出，Draw.io GUI 渲染尚未驗證。編譯後共 8 頁。

2026-10-12 全圖 skill 修訂：所有三張主圖已更換；概念圖提供原生 Draw.io／SVG，結果圖依保存的實測紀錄重畫，顯示各條件的三次實測點與平均。PDF 仍為 7 頁。

使用者指定的精細 verification-reference 圖已直接嵌入第 4 頁；圖說明確標示數字與通過標記皆為示意。先前的簡化向量版本保留為編輯來源，並非目前刊登圖的逐元素重建。最新稿件 8 頁。

架構圖也改為使用者指定的精細構圖（只校正覆蓋公式、QUBO 變數與 checker 描述），已直接嵌入 main.pdf 第 3 頁。全文維持 8 頁；沒有新增論文 PDF 副本。
