# Verification Report — GCI World 2026 Final Assignment (Company A)

**Verifier:** `@verifier`  
**Overall Status:** **PASS** (All 5 audit checks passed; zero blocking discrepancies found)

---

## 1. PDF Constraints (`JoseAlonsoYanezMejia.pdf`)
- **Page Count:** `14` pages (`<= 15` page limit per `Final Assignment/README.docx`) — **PASS**
- **File Size:** `2,791,805` bytes (`2.66 MB`, `<= 10 MB` limit) — **PASS**

## 2. Explicit Model Reporting on Slides (`JoseAlonsoYanezMejia.pdf`)
Extracted text via `pypdf` confirms all model names, evaluation metrics, and exact scores are explicitly written in slide text (Slides 7–8), not solely embedded in figures:
- **Models Named:** `XGBoost`, `LightGBM`, `Random Forest`, `Logistic Regression` — **PASS**
- **Metrics Named:** `ROC-AUC`, `PR-AUC`, `Recall`, `F1`, `Accuracy` (plus `Precision` and `F2`) — **PASS**
- **Exact Scores Reported in Text:**
  - `XGBoost`: CV ROC-AUC `0.6977 ± 0.0042`, Test ROC-AUC `0.6981`, Test PR-AUC `0.6864`, Test F1 `0.6451`, Test Accuracy `0.6438` (Slide 7); Threshold `0.40` Test Recall `0.8395`, Precision `0.5769`, F2 `0.7695` (Slide 8).
  - `LightGBM`: CV ROC-AUC `0.6978 ± 0.0055`, Test ROC-AUC `0.6974`, Test PR-AUC `0.6858`, Test F1 `0.6384` (Slide 7).
  - `Random Forest`: CV ROC-AUC `0.6772 ± 0.0054`, Test ROC-AUC `0.6779`, Test PR-AUC `0.6598`, Test F1 `0.6296` (Slide 7).
  - `Logistic Regression`: CV ROC-AUC `0.6380 ± 0.0023`, Test ROC-AUC `0.6409`, Test PR-AUC `0.6226`, Test F1 `0.5961` (Slide 7).

## 3. Notebook Execution & Integrity (`JoseAlonsoYanezMejia.ipynb`)
- **Structure:** `30` cells total (`19` code cells, `11` markdown cells) — **PASS**
- **Execution Errors:** `0` cells with `"output_type": "error"` — **PASS**
- **Populated Outputs:** All `19` code cells have populated outputs (`0` empty code-cell outputs) — **PASS**
- **Relative Paths:** Uses `os.path.join("Final Assignment", "telecom")` for `Client.csv` and `Record.csv` (`0` hardcoded local/absolute paths) — **PASS**

## 4. Cross-Artifact Numerical Consistency
Verified consistency across `outputs/eda_summary.md`, `outputs/model_summary.md`, `outputs/business_proposal.md`, `JoseAlonsoYanezMejia.ipynb`, and `JoseAlonsoYanezMejia.pdf`:

| Metric | `outputs/*.md` | `JoseAlonsoYanezMejia.ipynb` | `JoseAlonsoYanezMejia.pdf` | Status |
|---|---|---|---|---|
| Dataset Rows | `100,000` | `100,000` | `100,000` | **PASS** |
| Observed Churn Rate | `49.56%` | `49.56%` | `49.56%` | **PASS** |
| Churned Monthly Revenue | `$2,870,795.53` (`$2.87M/mo`) | `$2,870,795.53` (`$2.87M/month`) | `$2.87M / mo` | **PASS** |
| XGBoost Test ROC-AUC | `0.6981` | `0.6981` | `0.6981` | **PASS** |
| Targeted Subscribers | `19,340` | `19,340` | `19,340` | **PASS** |
| Total Program Cost | `$468,700` | `$468,700` | `$468,700` (`$468.7K`) | **PASS** |
| 6-Month Net Impact / ROI | `+$704,451` / `150.3%` | `704451.0` / `150.3%` | `+$704.5K` / `150.3%` | **PASS** |
| 12-Month Net Impact / ROI | `+$1,877,602` / `400.6%` | `+$1,877,602` / `400.6%` | `+$1.88M` / `400.6%` | **PASS** |
| 6-Month Break-Even Win-Back | `7.99%` | `7.99%` | `7.99%` | **PASS** |

## 5. Citations & Submission Note
- **Citations `[1]`–`[9]`:** All 9 citations appear on Slide 3 (Market Analysis) and are listed in full on Slide 14 (References & Reproducibility), matching `outputs/references.txt` 1-to-1 — **PASS**
- **Submission Note (Non-Blocking):** Slide 14 contains `[Placeholder: Insert GenAI Chat/Transcript URL Here]` for the user to insert their GenAI chat URL before uploading to Omnicampus (per `Final Assignment/README.docx` Section 3).
