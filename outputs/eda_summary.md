# Exploratory Data Analysis Summary — Company A (US Wireless Telecom)

**Source Script:** `scripts/01_eda.py`  
**Inputs:** `Final Assignment/telecom/Client.csv` (100,000 × 50) & `Final Assignment/telecom/Record.csv` (100,000 × 51)  
**Merged Dataset:** 100,000 rows × 100 columns (1:1 inner join on `Customer_ID`; 0 unmatched IDs, 0 duplicate rows, `Customer_ID` is the only overlapping column).

---

## 1. Data Quality Audit

### 1.1 Schema & Types
- **100 columns total:** 69 `float64`, 21 `str` (categorical/object), 10 `int64` (including `Customer_ID` and `churn`).
- **Zero constant columns** (all 100 columns have >1 unique value).

### 1.2 Missingness (43 of 100 columns have missing values)
Missingness is heavily concentrated in third-party/demographic enrichment fields, while core billing and call-record metrics are >97% complete:

| Severity | Columns | Missing % Range | Notes |
|---|---|---|---|
| **High (>20%)** | `numbcars` (49.37%), `dwllsize` (38.31%), `HHstatin` (37.92%), `ownrent` (33.71%), `dwlltype` (31.91%), `lor` (30.19%), `income` (25.44%), `adults` (23.02%), `infobase` (22.08%) | 22.08% – 49.37% | Demographic/InfoBase enrichment; treat missing as explicit `"MISSING"` category or add missingness flags |
| **Moderate (2%–15%)** | `hnd_webcap` (10.19%), `prizm_social_one` (7.39%), `avg6mou` (2.84%), `avg6rev` (2.84%), `avg6qty` (2.84%) | 2.84% – 10.19% | `avg6*` missing likely reflects tenure <6 months or incomplete 6-mo history |
| **Low (<2%)** | `kid0_2`..`kid16_17`, `ethnic`, `marital`, `creditcd`, `forgntvl`, `rv`, `truck` (1.73% each); `change_rev`, `change_mou` (0.89%); `hnd_price` (0.85%); `rev_Mean`, `mou_Mean`, `totmrc_Mean`, `ovrrev_Mean`, `ovrmou_Mean`, `vceovr_Mean`, `datovr_Mean`, `da_Mean`, `roam_Mean` (0.36%); `area` (0.04%); `eqpdays`, `refurb_new`, `dualband`, `models`, `phones` (0.001%, 1 row) | 0.001% – 1.73% | Safe for median/mode imputation + missingness indicator |

### 1.3 Anomalies, Outliers & Collinearity
- **Negative `eqpdays`:** 133 rows have `eqpdays < 0` (min `-5`), plus 33 rows with `eqpdays == 0` and 1 missing. Negative equipment age is a data-entry/timing anomaly and should be clipped to `0` or flagged.
- **Negative `rev_Mean` / `totmrc_Mean` / `avg6rev`:** 5 rows have `rev_Mean < 0` (min `-$6.17`), likely billing credits/adjustments; `totmrc_Mean` min is `-$26.92`; `avg6rev` min is `-$2.00`.
- **Extreme right-skew in usage & revenue:** `rev_Mean` p99 = `$227.89` vs. max = `$3,843.26`; `custcare_Mean` p99 = `20.33` vs. max = `675.33`; `ovrrev_Mean` p99 = `$139.27` vs. max = `$1,102.40`; `change_mou` ranges from `-3,875` to `+31,219`. Tree models handle skew well; linear models require winsorization (e.g., 1st/99th percentile) or `log1p` transforms.
- **Near-duplicate / Collinear pairs (`|r| >= 0.95`):**
  - `totcalls` ↔ `adjqty` (`r = 0.9998`), `totmou` ↔ `adjmou` (`r = 0.9997`), `totrev` ↔ `adjrev` (`r = 0.9985`)
  - `plcd_vce_Mean` ↔ `attempt_Mean` (`r = 0.9984`), `comp_vce_Mean` ↔ `complete_Mean` (`r = 0.9977`)
  - `ovrrev_Mean` ↔ `vceovr_Mean` (`r = 0.9948`), `ccrndmou_Mean` ↔ `cc_mou_Mean` (`r = 0.9889`), `avg3mou` ↔ `mou_Mean` (`r = 0.9811`)
- **Target Leakage Check:** `churn` is defined as churn occurring 31–60 days *after* the observation date. No single feature has suspiciously high linear correlation with `churn` (max `|r| = 0.1127` for `eqpdays`, followed by `hnd_price` at `-0.1032`, `totmrc_Mean` at `-0.0686`, `mou_Mean` at `-0.0570`). No post-outcome leakage columns were detected, though `Customer_ID` must be excluded from modeling.

---

## 2. Target Balance & Revenue / CLV Distribution

- **Class Balance (`churn`):** Balanced (likely case-control / stratified sample for PoC modeling):
  - **Retained (`churn = 0`):** `50,438` (`50.44%`)
  - **Churned (`churn = 1`):** `49,562` (`49.56%`)
- **Revenue & Tenure Summary (`n = 100,000`, `rev_Mean` non-null `n = 99,643`):**
  - `rev_Mean`: mean `$58.72`, median `$48.20`, IQR `$33.26–$70.75`, p90 `$103.98`.
  - `totmrc_Mean` (recurring base charge): mean `$46.18`, median `$44.99`.
  - `ovrrev_Mean` (overage): mean `$13.56`, median `$1.00`, p75 `$14.44`, p90 `$40.06`.
  - `totrev` (lifetime revenue): mean `$1,031.92`, median `$804.53`, IQR `$518.98–$1,263.77`, p90 `$1,915.00`.
  - `months` (tenure): mean `18.83` mo, median `16.0` mo, range `6–61` mo.
- **Revenue at Risk:**
  - Total monthly revenue across observed base: **`$5,851,035.45`**.
  - Monthly revenue from churned customers: **`$2,870,795.53` (`49.06%`)**, or **`$34,449,546.40` annualized**.

---

## 3. Intervenable Churn Drivers & High-Value Segments

### 3.1 Handset Age (`eqpdays`) — Strongest Operational Driver
Churn rises monotonically from `41.81%` (1–180 days) to `57.89%` (>730 days) — a **+16.1 percentage-point (pp)** spread:
- `1–180d`: `18,161` (`18.16%`), churn **`41.81%`**, mean rev `$78.68`
- `181–365d`: `36,356` (`36.36%`), churn **`46.97%`**, mean rev `$59.31`
- `366–540d`: `21,138` (`21.14%`), churn **`53.71%`**, mean rev `$54.11`
- `541–730d`: `13,367` (`13.37%`), churn **`54.04%`**, mean rev `$49.40`
- `>730d`: `10,811` (`10.81%`), churn **`57.89%`**, mean rev `$43.15`
- *(166 rows `<=0` anomaly have `33.73%` churn, mean rev `$83.90`.)*

### 3.2 Handset Price (`hnd_price`) & Refurbished Devices (`refurb_new`)
- **Handset Price:** Customers on entry/subsidized devices (`<=$50`, `26.75%` of base) churn at **`57.46%`**, vs. **`41.33%`** (`$151–$200`, `10.33%`) and **`36.02%`** (`>$200`, `0.63%`).
- **Refurbished vs. New (`refurb_new`):** Refurbished handset users (`R`, `14,406` = `14.41%`) churn at **`53.44%`** (mean rev `$63.50`) vs. **`48.91%`** for new handsets (`N`, `85,593` = `85.59%`).

### 3.3 Service Quality (`drop_blk_Mean`) & Customer Care (`custcare_Mean`)
- **Silent Churners (`custcare_Mean == 0`):** `55,692` customers (`55.69%`) made **zero** customer care calls and churn at **`51.76%`**, whereas customers who contacted care (`>5` calls: `10,291` = `10.29%`) churn at **`44.15%`** (and spend `$81.69`/mo). Proactive outreach is critical because most churners leave without calling care.
- **Dropped/Blocked Calls (`drop_blk_Mean`):** `0` dropped/blocked calls (`11,865` = `11.86%`) has `55.73%` churn (driven by low-usage/inactive lines, mean rev `$39.53`), while heavy users with `>20` dropped/blocked calls (`13,270` = `13.27%`) spend `$100.52`/mo and churn at `48.39%`.

### 3.4 Overage Burden (`ovrrev_Mean`) & Usage Trajectory (`change_mou`, `change_rev`)
- **Usage Decline (`change_mou`):** Subscribers whose monthly minutes dropped by `>200` vs. their 3-month average (`12,199` = `12.20%`) churn at **`54.99%`** (and have high mean rev `$87.74`), compared to **`45.39%`** for stable/growing users (`0 to +200` minutes) — a **+9.6 pp** early-warning signal.
- **Overage Share (`ovr_ratio = ovrrev_Mean / rev_Mean`):** When overage exceeds `50%` of the bill (`9,333` = `9.33%`, mean rev `$115.45`), churn jumps to **`55.31%`** vs. **`46.51%`** for `(0%, 10%]` overage. Similarly, `ovrrev_Mean > $30` (`13,860` = `13.86%`, mean rev `$126.15`) churns at **`52.62%`**.

### 3.5 Account Spending Limit (`asl_flag`), Credit Class (`crclscod`), Tenure (`months`) & Geography (`area`)
- **Account Spending Limit (`asl_flag`):** `Y` (`13,936` = `13.94%`) churns at **`40.93%`** (mean rev `$67.36`) vs. `N` (`86,064` = `86.06%`) at **`50.96%`** — a **10.0 pp** retention advantage.
- **Credit Class (`crclscod`):** Top classes `AA` (`36.51%`, `50.57%` churn), `A` (`17.00%`, `51.73%`), `BA` (`12.67%`, `52.13%`), `CA` (`8.33%`, `46.87%`), `EA` (`6.42%`, `40.97%`, `$72.58` rev). Highest churn in `A2` (`61.56%`, `n=861`); lowest in `E4` (`30.61%`, `n=833`, `$79.39` rev).
- **Tenure (`months`):** Churn peaks in the **13–18 month** window (`52.72%`, `n=24,163`), coinciding with post-1-year contract/device aging, vs. `38.38%` in `1–6m` and `47.29%` in `>36m`.
- **Geography (`area`):** Highest churn in **Northwest/Rocky Mountain** (`56.91%`, `n=4,328`) and **South Florida** (`53.36%`, `n=3,332`), lowest in **Midwest** (`45.90%`, `n=6,346`) and **DC/Maryland/Virginia** (`46.03%`, `n=6,567`).

### 3.6 High-Value At-Risk Segment Sizing (Top Revenue Quartile `Q4`, `rev_Mean >= $70.76`)
- **Q4 Total:** `24,910` subscribers, mean rev **`$114.03`/mo**, churn **`48.61%`**.
- **Q4 + Aging Handset (`eqpdays > 365`):** **`7,763` subscribers** (`31.2%` of Q4), churning at **`55.21%`** (and **`59.50%`** for `>730d`), representing **`$832,098/mo` (`$9.99M/year`)** in high-value revenue with an immediate device-upgrade intervention lever.

---

## 4. Recommended Data-Cleaning Rules & Candidate Features for Modeling

### Data-Cleaning Rules
1. **Merge & ID:** Inner join `Client.csv` and `Record.csv` on `Customer_ID`; drop `Customer_ID` before training.
2. **Anomaly Handling:** Clip `eqpdays < 0` to `0` and add `eqpdays_neg_flag`; keep `rev_Mean < 0` or clip at `0` with `rev_neg_flag`.
3. **Missingness:** Impute numeric features with median + binary `_is_missing` indicators (especially for `avg6*`, `change_mou`, `change_rev`, `hnd_price`, `income`, `lor`); fill categorical `NaN` with `"MISSING"`.
4. **Collinearity & Skew:** Drop redundant `adj*` duplicates (`adjqty`, `adjmou`, `adjrev`) if using linear/logistic models; apply `log1p` or 1%–99% winsorization for linear models (tree ensembles can use raw + engineered features).

### High-Signal Candidate Features
- **Equipment & Upgrade:** `eqpdays`, `eqpdays_gt_365` (`eqpdays > 365`), `hnd_price`, `refurb_new`, `phones`, `models`, `dualband`.
- **Behavioral Trajectory:** `change_mou`, `change_rev`, `mou_3m_vs_6m = avg3mou / (avg6mou + 1)`, `rev_3m_vs_6m = avg3rev / (avg6rev + 1)`.
- **Billing & Overage:** `ovr_ratio = ovrrev_Mean / max(rev_Mean, 1)`, `ovrrev_Mean`, `ovrmou_Mean`, `totmrc_Mean`, `rev_Mean`.
- **Quality & Care Friction:** `drop_blk_Mean`, `drop_rate = drop_blk_Mean / (attempt_Mean + 1)`, `comp_rate = complete_Mean / (attempt_Mean + 1)`, `custcare_Mean`, `has_custcare = (custcare_Mean > 0)`.
- **Contract / Account / Geo:** `months`, `asl_flag`, `crclscod`, `area`, `uniqsubs`, `actvsubs`.

---

## 5. Generated Figures (`outputs/figures/`, 300 DPI)
1. `outputs/figures/eda_churn_by_eqpdays.png` — Subscriber volume & churn rate across handset age bins.
2. `outputs/figures/eda_churn_by_quality_care.png` — Churn rate by dropped/blocked calls & customer care contacts.
3. `outputs/figures/eda_value_vs_churn_segments.png` — Churn rate (%) heatmap across Monthly Revenue Quartiles × Handset Age.
4. `outputs/figures/eda_overage_and_usage_change.png` — Churn rate by `change_mou` (leading indicator) and `ovrrev_Mean`.
5. `outputs/figures/eda_missingness_and_tenure.png` — Top 15 missing columns & lifecycle churn/revenue by tenure.
