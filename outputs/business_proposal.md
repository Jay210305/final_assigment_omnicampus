# Churn Prevention & High-Value Retention Proposal — Company A (US Wireless Telecom)

**Prepared by:** Proposal Quantifier | **Source Script:** [`scripts/03_business_impact.py`](file:///c:/Users/jay21/Documents/Universidad/Omnicampus/Final_Assignment/scripts/03_business_impact.py)  
**Supporting Analyses:** [`outputs/market_research.md`](file:///c:/Users/jay21/Documents/Universidad/Omnicampus/Final_Assignment/outputs/market_research.md) · [`outputs/eda_summary.md`](file:///c:/Users/jay21/Documents/Universidad/Omnicampus/Final_Assignment/outputs/eda_summary.md) · [`outputs/model_summary.md`](file:///c:/Users/jay21/Documents/Universidad/Omnicampus/Final_Assignment/outputs/model_summary.md) · [`outputs/test_predictions.csv`](file:///c:/Users/jay21/Documents/Universidad/Omnicampus/Final_Assignment/outputs/test_predictions.csv)

---

## 1. Executive Summary

Company A faces severe revenue leakage across its 100,000-subscriber base: **49.56%** of observed subscribers churn (`$2.87M/month` or `$34.45M/year` in lost recurring revenue), far exceeding the **0.7%–1.1% monthly** Big Three US postpaid benchmark. Rather than deploying a costly, untargeted blanket discount across all 100,000 subscribers (`$2.50M` budget, `37.4%` 6-month ROI), we propose a **Two-Tier Predictive Retention Program** powered by our winning **XGBoost churn model** (`0.6981` held-out Test ROC-AUC) and grounded in intervenable operational drivers (`eqpdays > 365`, overage shock, and silent usage drops).

Scaling the 20,000-customer held-out test set by **5x** to Company A's 100,000-customer base under a conservative **20% intervention win-back rate**:
- **Target Population:** **19,340 subscribers** (**19.34%** of base; **68.56%** actual churn rate), capturing **13,260 actual churners** and **$1.173M/month** in at-risk revenue.
- **Total Program Cost:** **$468,700** (`$24.23` blended unit cost per targeted customer) — **81.3% lower spend** than blanket retention (`$2.50M`).
- **Gross Revenue Protected:** **$1,173,151** (6-month horizon) | **$2,346,302** (12-month horizon).
- **Net Financial Impact:** **+$704,451** (6-month) | **+$1,877,602** (12-month).
- **Program ROI:** **150.3%** (6-month) | **400.6%** (12-month) — **4.0x higher 6-month ROI** than blanket retention (`37.4%`).
- **Churn Reduction:** **-2.65 pp** across the entire 100,000-customer base (**2,652 subscribers retained**; **-13.71 pp** within the targeted cohort).
- **Break-Even Win-Back Rate:** **7.99%** (6-month) | **4.00%** (12-month) — meaning the program remains profitable even if win-back is less than half our 20% base case.
- **Secondary Upside (CAC Avoidance):** Retaining 2,652 subscribers avoids **$663,000** in replacement acquisition spend at the conservative **$250/user** US telecom CAC floor (or **$1.06M+** at the $400+ postpaid benchmark), excluded from service-revenue ROI above to prevent double-counting.

---

## 2. Targeted Intervenable Segments & Intervention Playbook

Both EDA and XGBoost gain importance confirm that churn is concentrated around **equipment aging** (`eqpdays_gt_365` #1 at `4.63%`, `eqpdays` #2 at `2.59%`, `hnd_price` #6 at `1.21%`, `refurb_new` #9–10), **account spending limit protection** (`asl_flag` #13/#15), **overage shocks** (`ovrrev_Mean > $30` or `>50%` of bill), and **silent usage drops** (`custcare_Mean == 0` with `change_mou < -200`).

We partition the 100,000-customer base into three mutually exclusive, actionable tiers:

| Tier | Segment Definition (Exact Rules) | 100k Base N (% Base) | Actual Churn Rate | Mean Rev/Mo | Concrete Intervention & Channel | Unit Cost | Total Cost (100k) |
|---|---|---:|---:|---:|---|---:|---:|
| **Tier 1A: HV + High-Risk + Ageing Handset** | `rev_Mean >= $70.76` (Top Revenue Q4) AND `XGBoost Risk Q4/Q5` AND `eqpdays > 365` | **4,175** (`4.18%`) | `62.04%` | `$108.53` | **Proactive EIP / Handset Upgrade Trade-In Credit** (`$35` amortized promo credit) + **Priority Care Check-In** (`$10` outbound concierge, <10-min resolution) | **$45.00** | **$187,875** |
| **Tier 1B: HV + High-Risk + Newer Handset** | `rev_Mean >= $70.76` (Top Revenue Q4) AND `XGBoost Risk Q4/Q5` AND `eqpdays <= 365` | **5,335** (`5.34%`) | `64.20%` | `$122.40` | **Plan-Rightsizing / Overage Relief Credit + ASL Enrollment** (`$15` bill credit) + **Proactive Care Outreach** (`$10`) | **$25.00** | **$133,375** |
| **Tier 2: Mass-Market High-Risk Intervenable** | `rev_Mean < $70.76` AND `XGBoost Risk Q5 (Highest)` AND (`eqpdays > 365` OR `ovrrev_Mean > $30` OR `ovr_ratio > 50%` OR (`custcare == 0` & `change_mou < -200`)) | **9,830** (`9.83%`) | `73.70%` | `$46.29` | **Automated Digital/SMS Plan-Rightsizing & Loyalty Perk** (zero-transfer app self-service overage cap or refurb-to-new swap voucher) | **$15.00** | **$147,450** |
| **Total Recommended Program (T1 + T2)** | Union of Tier 1A, Tier 1B, and Tier 2 | **19,340** (`19.34%`) | **`68.56%`** | **`$79.64`** | **Tiered Value-Matched Retention** | **$24.23** *(avg)* | **$468,700** |

*Unit Cost Benchmarking:* Tier 2 (`$15`) and Tier 1B (`$25`) sit squarely within the **$15–$35/user** US telecom retention benchmark (`market_research.md`), while Tier 1A (`$45`) reflects the **$30–$50** handset/EIP credit range (`model_summary.md`) and remains **5.6x–13.3x cheaper** than **$250–$600+** US wireless CAC.

---

## 3. Financial Quantification & Formulas (100,000-Customer Base)

### Exact Methodology & Formulas
All metrics are computed on `outputs/test_predictions.csv` (`n = 20,000`, `rev_Mean` median-imputed for `64` missing rows and clipped at `>= $0`) and scaled by `5x`:
- $\text{Total Cost} = 5 \times \sum_{i \in \text{Target}} c_i$ (applied to **all** targeted subscribers—both true churners and false positives—explicitly pricing in deadweight subsidies).
- $\text{Gross Revenue Protected}(H) = 5 \times w \times H \times \sum_{i \in \text{Target} \cap \{\text{churn}=1\}} \text{rev}_i$, where $w = 0.20$ (base win-back rate) and $H \in \{6, 12\}$ months.
- $\text{Net Financial Impact}(H) = \text{Gross Revenue Protected}(H) - \text{Total Cost}$.
- $\text{ROI}(H) = \frac{\text{Net Financial Impact}(H)}{\text{Total Cost}} \times 100\%$.
- $\text{Break-Even Win-Back Rate } w^*(H) = \frac{\text{Total Cost}}{5 \times H \times \sum_{i \in \text{Target} \cap \{\text{churn}=1\}} \text{rev}_i} \times 100\%$.

### 3.1 Tier-Level Financial Breakdown (20% Win-Back Base Case)

| Segment | Targeted N | Actual Churners | Retained (20%) | Total Cost | Gross Rev (6-Mo) | Net Impact (6-Mo) | ROI (6-Mo) | Gross Rev (12-Mo) | Net Impact (12-Mo) | ROI (12-Mo) | Break-Even (6-Mo / 12-Mo) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Tier 1A (HV + Ageing)** | `4,175` | `2,590` | `518` | `$187,875` | `$334,161` | `$146,286` | `77.9%` | `$668,321` | `$480,446` | `255.7%` | `11.2% / 5.6%` |
| **Tier 1B (HV + Newer)** | `5,335` | `3,425` | `685` | `$133,375` | `$503,147` | `$369,772` | `277.2%` | `$1,006,295` | `$872,920` | `654.5%` | `5.3% / 2.7%` |
| **Tier 1 Subtotal (All HV)** | `9,510` | `6,015` | `1,203` | `$321,250` | `$837,308` | `$516,058` | `160.6%` | `$1,674,616` | `$1,353,366` | `421.3%` | `7.7% / 3.8%` |
| **Tier 2 (Mass Q5 Intv.)** | `9,830` | `7,245` | `1,449` | `$147,450` | `$335,843` | `$188,393` | `127.8%` | `$671,686` | `$524,236` | `355.5%` | `8.8% / 4.4%` |
| **Total Recommended (T1+T2)** | **`19,340`** | **`13,260`** | **`2,652`** | **`$468,700`** | **`$1,173,151`** | **`$704,451`** | **`150.3%`** | **`$2,346,302`** | **`$1,877,602`** | **`400.6%`** | **`7.99% / 4.00%`** |

### 3.2 Why Targeted Retention Beats Blanket Retention

| Strategy (100k Base, 20% Win-Back) | Targeted N (% Base) | Seg. Churn Rate | Total Cost | Net Impact (6-Mo) | ROI (6-Mo) | Net Impact (12-Mo) | ROI (12-Mo) | Break-Even (6-Mo) | Base Churn Red. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1. Blanket Retention (All 100k)** | `100,000` (`100.0%`) | `49.56%` | `$2,500,000` | `$934,679` | `37.4%` | `$4,369,359` | `174.8%` | `14.56%` | `-9.91 pp` |
| **2. Model-Targeted (`p >= 0.40`)** | `72,115` (`72.1%`) | `57.69%` | `$1,802,875` | `$1,024,764` | `56.8%` | `$3,852,404` | `213.7%` | `12.75%` | `-8.32 pp` |
| **3. Model-Targeted (`p >= 0.50`)** | `50,820` (`50.8%`) | `63.72%` | `$1,270,500` | `$921,950` | `72.6%` | `$3,114,401` | `245.1%` | `11.59%` | `-6.48 pp` |
| **4. Model Top Quintile (`Q5`)** | `20,000` (`20.0%`) | `74.00%` | `$500,000` | `$486,138` | `97.2%` | `$1,472,276` | `294.5%` | `10.14%` | `-2.96 pp` |
| **5. Recommended Tiered (T1+T2)** | **`19,340`** (`19.3%`) | **`68.56%`** | **`$468,700`** | **`$704,451`** | **`150.3%`** | **`$1,877,602`** | **`400.6%`** | **`7.99%`** | **`-2.65 pp`** |
| **6. Laser HV + Ageing (T1A Only)** | `4,175` (`4.2%`) | `62.04%` | `$187,875` | `$146,286` | `77.9%` | `$480,446` | `255.7%` | `11.24%` | `-0.52 pp` |

**Key Insight:** Blanket retention wastes `$1.26M` subsidizing `50,440` non-churners, requiring a steep **14.56%** 6-month win-back rate just to break even. By combining **XGBoost risk stratification** with **customer value (`rev_Mean >= $70.76`)** and **intervenable triggers**, the Recommended Tiered Program achieves **4.0x the 6-month ROI** (`150.3%` vs. `37.4%`), **+45% higher net impact than untargeted `Q5`** (`$704.5K` vs. `$486.1K` at 6 months on a smaller budget), and cuts break-even risk nearly in half (**7.99%** vs. **14.56%**).

![Targeting Strategy Comparison](file:///c:/Users/jay21/Documents/Universidad/Omnicampus/Final_Assignment/outputs/figures/impact_targeting_comparison.png)

---

## 4. Sensitivity Analysis (Win-Back Rates & Model Thresholds)

### 4.1 Win-Back Rate Sensitivity — Recommended Tiered Program (`$468,700` Fixed Cost)

| Win-Back Rate | Retained Subs | Gross Rev (6-Mo) | Net Impact (6-Mo) | ROI (6-Mo) | Gross Rev (12-Mo) | Net Impact (12-Mo) | ROI (12-Mo) | Base Churn Red. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **10% (Pessimistic)** | `1,326` | `$586,576` | **`+$117,876`** | `25.1%` | `$1,173,151` | **`+$704,451`** | `150.3%` | `-1.33 pp` |
| **15% (Conservative)** | `1,989` | `$879,863` | **`+$411,163`** | `87.7%` | `$1,759,727` | **`+$1,291,027`** | `275.4%` | `-1.99 pp` |
| **20% (Base Case)** | `2,652` | `$1,173,151` | **`+$704,451`** | `150.3%` | `$2,346,302` | **`+$1,877,602`** | `400.6%` | `-2.65 pp` |
| **25% (Upside)** | `3,315` | `$1,466,439` | **`+$997,739`** | `212.9%` | `$2,932,878` | **`+$2,464,178`** | `525.7%` | `-3.32 pp` |
| **30% (Stretch)** | `3,978` | `$1,759,727` | **`+$1,291,027`** | `275.4%` | `$3,519,453` | **`+$3,050,753`** | `650.9%` | `-3.98 pp` |

### 4.2 Threshold Choice Sensitivity (`p >= 0.40` vs. `p >= 0.50` vs. `Q5` vs. `Tiered`)

| Threshold / Rule | Targeted N | Total Cost | 6-Mo Net @ 10% WB | 6-Mo Net @ 20% WB | 6-Mo Net @ 30% WB | 6-Mo Break-Even WB |
|---|---:|---:|---:|---:|---:|---:|
| **`p >= 0.40` (High Recall)** | `72,115` | `$1,802,875` | `-$389,055` *(loss)* | `+$1,024,764` | `+$2,438,584` | `12.75%` |
| **`p >= 0.50` (Balanced)** | `50,820` | `$1,270,500` | `-$174,275` *(loss)* | `+$921,950` | `+$2,018,176` | `11.59%` |
| **`Top Quintile Q5`** | `20,000` | `$500,000` | `-$6,931` *(loss)* | `+$486,138` | `+$979,207` | `10.14%` |
| **`HV (rev>=$70.76) & Q4+Q5`** | `9,510` | `$321,250` | `+$97,404` | `+$516,058` | `+$934,712` | **`7.67%`** |
| **`Recommended Tiered (T1+T2)`** | **`19,340`** | **`$468,700`** | **`+$117,876`** | **`+$704,451`** | **`+$1,291,027`** | **`7.99%`** |

Notably, **only** the High-Value & Recommended Tiered strategies remain **cash-flow positive at 6 months even under a 10% win-back rate**, whereas broad thresholds (`p >= 0.40`, `p >= 0.50`, `Q5`) lose money at 10% win-back.

![Break-Even & Sensitivity Analysis](file:///c:/Users/jay21/Documents/Universidad/Omnicampus/Final_Assignment/outputs/figures/impact_sensitivity_breakeven.png)

---

## 5. Three-Month A/B Pilot Design

Before full 100k rollout, we recommend a controlled **90-day (3-month) randomized A/B pilot** on **20% of the eligible Tier 1 + Tier 2 population** (`~3,868` subscribers, stratified by Tier 1A / 1B / 2 and region):
- **Randomization Split:** **50% Treatment (`n = 1,934`)** receiving the tiered retention offers vs. **50% Holdout Control (`n = 1,934`)** receiving business-as-usual care (no proactive offer).
- **Pilot Budget:** **`$46,870`** (`10%` of full-scale annual program cost, capped risk).
- **Primary Pass/Fail Criteria (Tied to Break-Even):**
  1. **Go (Full Rollout):** Treatment churn is **>= 8.0 percentage points lower** than Control among predicted churners (exceeding the **7.99%** 6-month break-even win-back rate with 95% confidence; minimum detectable effect at 80% power is ~4.1 pp) AND net 90-day ARPU minus offer cost is positive.
  2. **Iterate (Refine Offers):** Win-back between **4.0% and 7.99%** (breaks even at 12 months, not 6 months) — tighten Tier 1A trade-in cost or restrict Tier 2 to `D10` (`78.5%` churn).
  3. **No-Go (Kill/Pivot):** Win-back **< 4.0%** or negative ARPU lift due to plan-downgrade cannibalization.
- **Timeline & Owners:**
  - *Weeks 1–2 (Setup):* **Data Science & CRM Ops** deploy weekly XGBoost scoring, clean `eqpdays`/`rev_Mean` anomalies, and lock stratified randomization.
  - *Weeks 3–10 (Execution):* **Retention Marketing & Priority Care** launch Tier 1A/1B outbound/EIP offers and Tier 2 automated SMS/app flows (tracking <10-min resolution & zero-transfer SLA).
  - *Weeks 11–12 (Readout):* **Finance & Analytics** audit 31–60 day post-offer churn, net ARPU, and offer redemption cost; present Go/No-Go to **VP of Retention**.

---

## 6. Risk & Mitigation Matrix

| Risk Category | Specific Failure Mode | Impact | Mitigation Strategy |
|---|---|---|---|
| **Deadweight Loss (Subsidizing Stayers)** | Offering `$15–$45` incentives to false positives (`31.4%` of targeted cohort who would not have churned). | Dilutes ROI; inflates cost. | Already priced into all ROI/break-even math; further mitigate via **triggered delivery** (send Tier 1A EIP credit only upon engagement) and restricting Tier 2 to `Q5` + behavioral trigger. |
| **ARPU Cannibalization** | Plan-rightsizing or overage relief lowers monthly recurring revenue from heavy overage payers (`>$115/mo`). | Reduces net revenue protected. | Cap bill credits at `15%` of monthly ARPU (`$15–$25` one-time or 3-mo promo) rather than permanent rate-plan downgrades; track **Net ARPU** in the A/B pilot. |
| **Data Quality & Missingness** | `22%–49%` missingness in demographic enrichment (`numbcars`, `income`), `133` negative `eqpdays`, `5` negative `rev_Mean` rows. | Scoring drift or mis-tiering. | Modeling pipeline relies primarily on >97% complete billing/usage/equipment fields (`eqpdays`, `rev_Mean`, `change_mou`) and clips `eqpdays < 0` / `rev_Mean < 0` with explicit missingness flags. |
| **Seasonality & Competitor Promos** | Q4 flagship device launches or aggressive Cable/MVNO promos spike churn above static test-set baseline. | Win-back rate drops below `8%`. | Refresh XGBoost scores monthly; pre-stage Tier 1A upgrade offers 30 days ahead of Q4 device launches. |
| **Care Center Capacity** | Outbound Tier 1 calls overwhelm care reps, causing `>10-min` wait times or `20%–25%` transfers (known churn accelerant). | Backfires on satisfaction. | Route Tier 2 (`50.8%` of volume) to **100% automated SMS/app self-service**, reserving live reps strictly for Tier 1 (`9,510` HV accounts over 90 days ≈ ~150 calls/day). |
