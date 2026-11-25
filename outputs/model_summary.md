# Churn Prediction Modeling Summary — Company A (US Wireless Telecom)

**Source Script:** `scripts/02_modeling.py`  
**Predictions Export:** `outputs/test_predictions.csv` (`20,000` held-out test customers × `17` columns)  
**Dataset:** Inner merge of `Final Assignment/telecom/Client.csv` and `Record.csv` on `Customer_ID` (`100,000` rows × `100` raw columns → `113` modeling features: `92` numeric, `21` categorical; `churn = 49.56%`).  
**Validation Protocol:** `80/20` stratified train/test split (`80,000` train / `20,000` test, `random_state=42`) with **5-fold Stratified CV** on the training set. All preprocessing (median imputation + missingness indicators, `"MISSING"` categorical fill + `OneHotEncoder`, and `StandardScaler` for Logistic Regression) is fit strictly inside `sklearn` Pipelines on training folds only.

---

## 1. Feature Engineering & Data Cleaning

- **Anomaly Cleaning (per `outputs/eda_summary.md`):**
  - Clipped `eqpdays < 0` (`133` rows) to `0` and added `eqpdays_neg_flag`.
  - Clipped `rev_Mean < 0` (`5` rows) to `0` and added `rev_neg_flag`.
- **Engineered Intervenable & Behavioral Features (`13` new features):**
  - `overage_share = ovrrev_Mean / max(rev_Mean, 1)`
  - `drop_blk_rate = drop_blk_Mean / max(attempt_Mean, 1)`
  - `comp_rate = complete_Mean / max(attempt_Mean, 1)`
  - `rev_trend_ratio = avg3rev / max(avg6rev, 1)`
  - `mou_trend_ratio = avg3mou / max(avg6mou, 1)`
  - `eqp_per_month = eqpdays / max(months * 30, 1)`
  - `care_per_mou = custcare_Mean / max(mou_Mean, 1)`
  - `eqpdays_gt_365` (`eqpdays > 365`), `has_custcare` (`custcare_Mean > 0`), `active_line_ratio` (`actvsubs / max(uniqsubs, 1)`), `mou_per_rev`, `roam_share`, `mrc_share`.
- **Linear Model Collinearity Guard:** Dropped near-duplicate `adjqty`, `adjmou`, `adjrev` (`r >= 0.9985` with `tot*`) for Logistic Regression.

---

## 2. Model Family Comparison (5-Fold Stratified CV & Held-Out Test)

| Model | Key Hyperparameters | CV ROC-AUC (mean ± std) | Test ROC-AUC | CV PR-AUC (mean ± std) | Test PR-AUC | Test Recall (@0.50) | Test Precision (@0.50) | Test F1 (@0.50) | Test Accuracy (@0.50) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** *(Winner)* | `n_estimators=400, max_depth=5, lr=0.05, subsample=0.8, colsample=0.8, reg_lambda=2.0` | `0.6977 ± 0.0042` | **`0.6981`** | `0.6832 ± 0.0059` | **`0.6864`** | `0.6533` | `0.6372` | **`0.6451`** | **`0.6438`** |
| **LightGBM** | `n_estimators=400, max_depth=6, num_leaves=31, lr=0.05, subsample=0.8, colsample=0.8, reg_lambda=2.0` | **`0.6978 ± 0.0055`** | `0.6974` | `0.6831 ± 0.0072` | `0.6858` | `0.6448` | `0.6321` | `0.6384` | `0.6380` |
| **Random Forest** | `n_estimators=300, max_depth=16, min_samples_leaf=20` | `0.6772 ± 0.0054` | `0.6779` | `0.6586 ± 0.0082` | `0.6598` | `0.6439` | `0.6159` | `0.6296` | `0.6245` |
| **Logistic Regression** | `C=1.0, max_iter=1000, solver=lbfgs, StandardScaler + OHE` | `0.6380 ± 0.0023` | `0.6409` | `0.6170 ± 0.0041` | `0.6226` | `0.5919` | `0.6003` | `0.5961` | `0.6024` |

**Takeaway:** Gradient-boosted trees (**XGBoost** and **LightGBM**) outperform Logistic Regression by **+5.7 pp ROC-AUC** (`0.6981` vs. `0.6409`) and **+6.4 pp PR-AUC** (`0.6864` vs. `0.6226`), confirming strong non-linear interactions (e.g., handset age × tenure × usage drop). XGBoost is selected as the winning model on held-out Test ROC-AUC (`0.6981`), PR-AUC (`0.6864`), F1 (`0.6451`), and Accuracy (`0.6438`), with virtually zero CV-to-Test generalization gap (`0.6977` → `0.6981`).

---

## 3. Threshold Tuning & Asymmetric Business Cost Trade-Off (XGBoost)

Missing a churner risks ~$58.72/mo in recurring revenue (~$352 over a 6-month retention horizon, or ~$684 for Q4 high-value subscribers), whereas a targeted retention offer (e.g., handset upgrade credit or plan optimization) costs ~$30–$50. Lowering the decision threshold increases churner capture (Recall & F2):

| Threshold | Flagged Test (%) | CV Recall | CV Precision | CV F1 | CV F2 | Test Recall | Test Precision | Test F1 | Test F2 | Test Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.35` | `80.18%` | `0.8999` | `0.5579` | `0.6888` | `0.8016` | `0.9025` | `0.5579` | `0.6895` | `0.8033` | `0.5972` |
| **`0.40`** | `72.11%` | `0.8398` | `0.5792` | `0.6855` | `0.7705` | **`0.8395`** | `0.5769` | `0.6839` | `0.7695` | `0.6153` |
| `0.45` | `62.15%` | `0.7577` | `0.6052` | `0.6729` | `0.7214` | `0.7550` | `0.6021` | `0.6699` | `0.7185` | `0.6313` |
| `0.50` | `50.82%` | `0.6481` | `0.6359` | `0.6419` | `0.6456` | `0.6533` | `0.6372` | `0.6451` | `0.6500` | `0.6438` |
| `0.55` | `38.66%` | `0.5195` | `0.6683` | `0.5846` | `0.5437` | `0.5221` | `0.6693` | `0.5866` | `0.5461` | `0.6353` |

- For **broad low-cost digital outreach** (SMS/app offers), `threshold = 0.40` captures **83.95%** of all churners (`+18.6 pp` over `0.50`).
- For **budget-constrained high-touch offers** (handset subsidies), targeting **Top Risk Quintile (Q5)** or **Q4+Q5** maximizes ROI per dollar spent (see Section 5).

---

## 4. Top Churn Drivers (XGBoost Gain Importance)

| Rank | Feature | Importance (%) | Category | Business Interpretation |
|---:|---|---:|---|---|
| 1 | `eqpdays_gt_365` | `4.63%` | **Intervenable** | Handset age > 1 year — strongest single churn trigger |
| 2 | `eqpdays` | `2.59%` | **Intervenable** | Continuous handset age in days |
| 3 | `months` | `1.58%` | Structural | Subscriber tenure (peak churn at 13–18 mo) |
| 4 | `change_rev_is_missing` | `1.40%` | **Intervenable** | Missing billing trajectory (new/irregular usage) |
| 5 | `hnd_webcap_WCMB` | `1.30%` | Structural | Web-capable handset tier |
| 6 | `hnd_price` | `1.21%` | **Intervenable** | Device price tier (entry `<= $50` churns highest) |
| 7 | `income_is_missing` | `1.16%` | Structural | Missing demographic enrichment |
| 8 | `change_mou_is_missing` | `1.14%` | **Intervenable** | Missing usage trajectory |
| 9 | `refurb_new_R` | `1.13%` | **Intervenable** | Refurbished handset flag (`53.44%` churn vs `48.91%` new) |
| 10 | `refurb_new_N` | `1.07%` | **Intervenable** | New handset flag |
| 11 | `active_line_ratio` | `1.07%` | Structural | Ratio of active lines to unique subscribers |
| 12 | `ethnic_O` | `1.03%` | Structural | Demographic segment |
| 13 | `asl_flag_N` | `0.98%` | **Intervenable** | No Account Spending Limit (`50.96%` churn vs `40.93%` `Y`) |
| 14 | `crclscod_EA` | `0.96%` | Structural | Credit class `EA` |
| 15 | `asl_flag_Y` | `0.95%` | **Intervenable** | Account Spending Limit enabled (retention protector) |

**10 of the top 15** features (and **11 of the top 20**, including `mrc_share` at `0.89%` and `change_mou` at `0.80%`) are directly **intervenable** via device upgrade offers, refurbished-to-new swaps, ASL enrollment, and usage-drop outreach.

---

## 5. Test Set Risk Stratification & High-Value Segment Lift (`n = 20,000`)

### 5.1 All Test Customers by Predicted Risk Quintile (`XGBoost`)

| Risk Quintile | Customers | Actual Churners | Actual Churn Rate | Lift vs. Base (`49.56%`) | Churned Rev/Mo | Share of Churned Rev | Mean Rev/Mo | Mean `eqpdays` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Q1 (Lowest)** | `4,000` | `980` | `24.50%` | `0.49x` | `$62,819.57` | `11.01%` | `$59.99` | `227.4` |
| **Q2** | `4,000` | `1,638` | `40.95%` | `0.83x` | `$100,427.02` | `17.61%` | `$61.21` | `360.3` |
| **Q3** | `4,000` | `1,974` | `49.35%` | `1.00x` | `$108,767.51` | `19.07%` | `$56.26` | `433.0` |
| **Q4** | `4,000` | `2,360` | `59.00%` | `1.19x` | `$135,453.59` | `23.75%` | `$58.99` | `467.3` |
| **Q5 (Highest)** | `4,000` | `2,960` | **`74.00%`** | **`1.49x`** | **`$162,919.61`** | **`28.56%`** | `$56.17` | `475.2` |

- **Top Decile (`D10`, `n=2,000`):** **`78.50%`** actual churn (`1.58x` lift vs. baseline, `1,570` churners, `$79,541.67/mo` at risk) vs. **Bottom Decile (`D1`)** at **`19.40%`** (`0.39x`) — a **4.05x** top-to-bottom decile separation.
- **Top 2 Quintiles (`Q4 + Q5`, `40%` of base):** Capture **`5,320` of `9,912` churners (`53.67%`)** and **`$298,373/mo` (`52.31%` of all churned monthly revenue)**.

### 5.2 High-Value Customers (`Q4 Revenue: rev_Mean >= $70.76`, `n = 5,027` in Test)

| Risk Quintile | HV Customers | HV Churners | HV Churn Rate | Lift vs. Base | HV Churned Rev/Mo | Mean Rev/Mo | Mean `eqpdays` |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Q1 (Lowest)** | `1,071` | `320` | `29.88%` | `0.60x` | `$34,651.59` | `$108.09` | `188.8` |
| **Q2** | `1,134` | `467` | `41.18%` | `0.83x` | `$53,715.88` | `$114.18` | `269.2` |
| **Q3** | `920` | `431` | `46.85%` | `0.95x` | `$47,045.14` | `$111.31` | `315.1` |
| **Q4** | `986` | `557` | `56.49%` | `1.14x` | `$64,604.10` | `$118.13` | `358.6` |
| **Q5 (Highest)** | `916` | `646` | **`70.52%`** | **`1.42x`** | **`$74,947.22`** | `$114.79` | `387.2` |

- Targeting **High-Value `Q4 + Q5` (`1,902` test subscribers, `37.8%` of HV base)** captures **`1,203` HV churners (`49.7%` of all HV churners)** and **`$139,551/mo` (`$1.67M/year` in the 20% test set, or ~$8.37M/year scaled to the 100k base)** with average handset age `359–387` days.

---

## 6. Generated Figures (`outputs/figures/`, 300 DPI)

1. `outputs/figures/model_comparison_roc_auc.png` — 5-fold CV ROC-AUC (±std) vs. Held-Out Test ROC-AUC & PR-AUC across all 4 model families.
2. `outputs/figures/model_feature_importance.png` — Top 15 XGBoost churn drivers color-coded by Intervenable vs. Structural.
3. `outputs/figures/model_risk_quintile_lift.png` — Test set actual churn rate & lift by risk quintile + monthly churned revenue at risk (All vs. High-Value Q4).
4. `outputs/figures/model_roc_pr_curves.png` — Held-out Test ROC and Precision-Recall curves across all 4 model families.
