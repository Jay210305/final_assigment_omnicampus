"""
03_business_impact.py
=====================
Quantifies the business impact, ROI, break-even win-back rate, sensitivity
analysis, and targeting strategy comparison for Company A's Churn Prevention &
High-Value Retention Program.

Inputs:
    - outputs/test_predictions.csv (20,000 held-out test customers)

Outputs:
    - outputs/figures/impact_sensitivity_breakeven.png (300 DPI)
    - outputs/figures/impact_targeting_comparison.png (300 DPI)
    - stdout summary tables used in outputs/business_proposal.md
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "test_predictions.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCALE_FACTOR = 5  # 20,000 test set -> 100,000 Company A base
HV_THRESHOLD = 70.76  # Top revenue quartile (Q4) cutoff from EDA
BASE_WIN_BACK = 0.20  # 20% base-case win-back assumption
WIN_BACK_RATES = [0.10, 0.15, 0.20, 0.25, 0.30]
CAC_CONSERVATIVE = 250.0  # Lower bound of $250-$400 US wireless CAC (market_research.md)
CAC_POSTPAID_MID = 400.0  # Lower bound of $400-$600+ fully-loaded postpaid CAC

# Unit intervention costs ($/targeted customer), benchmarked to market_research.md ($15-$35)
# and model_summary.md ($30-$50 for handset upgrade credits)
COST_BLANKET = 25.0
COST_TIER1A_HV_AGING = 45.0     # EIP/handset trade-in bill credit ($35) + proactive care ($10)
COST_TIER1B_HV_OTHER = 25.0     # Plan-rightsizing / bill credit / ASL enrollment + priority care
COST_TIER2_MASS_Q5 = 15.0       # Automated SMS/app overage-relief or loyalty perk

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
})


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PRED_PATH)
    # Clean revenue for financial calculations (64 missing in test -> median; clip <0 to 0)
    rev_median = df["rev_Mean"].median()
    df["rev_clean"] = df["rev_Mean"].fillna(rev_median).clip(lower=0.0)
    df["ovr_clean"] = df["ovrrev_Mean"].fillna(0.0).clip(lower=0.0)
    df["ovr_ratio"] = df["ovr_clean"] / df["rev_clean"].clip(lower=1.0)

    # Core flags grounded in EDA + XGBoost feature importance
    df["is_hv"] = df["rev_Mean"] >= HV_THRESHOLD
    df["is_q5"] = df["risk_quintile"] == "Q5 (Highest)"
    df["is_q4_q5"] = df["risk_quintile"].isin(["Q4", "Q5 (Highest)"])
    df["is_aging_handset"] = df["eqpdays"] > 365
    df["is_ovr_shock"] = (df["ovr_clean"] > 30.0) | (df["ovr_ratio"] > 0.50)
    df["is_silent_drop"] = (df["custcare_Mean"] == 0) & (df["change_mou"] < -200)
    df["is_intervenable_mass"] = (
        df["is_aging_handset"] | df["is_ovr_shock"] | df["is_silent_drop"]
    )
    return df


def evaluate_strategy(
    df: pd.DataFrame,
    name: str,
    mask: pd.Series,
    unit_cost: pd.Series | float,
    win_back: float = BASE_WIN_BACK,
) -> dict:
    """Compute exact 100k-scaled financial & churn metrics for a targeting mask."""
    sub = df[mask]
    n_test = len(sub)
    n_100k = n_test * SCALE_FACTOR

    actual_churners_test = int(sub["churn"].sum())
    actual_churners_100k = actual_churners_test * SCALE_FACTOR
    seg_churn_rate = sub["churn"].mean() if n_test > 0 else 0.0

    # Exact monthly revenue of actual churners in this segment (no global-mean approximation)
    churned_rev_mo_test = sub.loc[sub["churn"] == 1, "rev_clean"].sum()
    churned_rev_mo_100k = churned_rev_mo_test * SCALE_FACTOR
    mean_rev_seg = sub["rev_clean"].mean() if n_test > 0 else 0.0

    if isinstance(unit_cost, pd.Series):
        total_cost_test = unit_cost.loc[mask].sum()
        avg_unit_cost = total_cost_test / n_test if n_test > 0 else 0.0
    else:
        total_cost_test = n_test * float(unit_cost)
        avg_unit_cost = float(unit_cost)
    total_cost_100k = total_cost_test * SCALE_FACTOR

    retained_100k = actual_churners_100k * win_back
    gross_rev_6m = churned_rev_mo_100k * 6.0 * win_back
    gross_rev_12m = churned_rev_mo_100k * 12.0 * win_back

    net_6m = gross_rev_6m - total_cost_100k
    net_12m = gross_rev_12m - total_cost_100k

    roi_6m = (net_6m / total_cost_100k * 100.0) if total_cost_100k > 0 else 0.0
    roi_12m = (net_12m / total_cost_100k * 100.0) if total_cost_100k > 0 else 0.0

    # Break-even win-back rate w* where Gross_Rev_Protected(w*) == Total_Cost
    be_6m = (total_cost_100k / (churned_rev_mo_100k * 6.0) * 100.0) if churned_rev_mo_100k > 0 else np.inf
    be_12m = (total_cost_100k / (churned_rev_mo_100k * 12.0) * 100.0) if churned_rev_mo_100k > 0 else np.inf

    # Churn reduction in percentage points across the full 100k base
    total_base_100k = len(df) * SCALE_FACTOR
    churn_red_base_pp = retained_100k / total_base_100k * 100.0
    churn_red_seg_pp = seg_churn_rate * win_back * 100.0

    cac_avoided_conservative = retained_100k * CAC_CONSERVATIVE

    return {
        "Strategy": name,
        "Test_N": n_test,
        "Base_100k_N": n_100k,
        "Pct_Base": n_test / len(df) * 100.0,
        "Actual_Churners_100k": actual_churners_100k,
        "Seg_Churn_Rate_Pct": seg_churn_rate * 100.0,
        "Mean_Rev_Mo": mean_rev_seg,
        "Churned_Rev_Mo_100k": churned_rev_mo_100k,
        "Avg_Unit_Cost": avg_unit_cost,
        "Total_Cost_100k": total_cost_100k,
        "Win_Back_Pct": win_back * 100.0,
        "Retained_100k": retained_100k,
        "Gross_Rev_6m": gross_rev_6m,
        "Net_Impact_6m": net_6m,
        "ROI_6m_Pct": roi_6m,
        "Gross_Rev_12m": gross_rev_12m,
        "Net_Impact_12m": net_12m,
        "ROI_12m_Pct": roi_12m,
        "BreakEven_6m_Pct": be_6m,
        "BreakEven_12m_Pct": be_12m,
        "Base_Churn_Red_pp": churn_red_base_pp,
        "Seg_Churn_Red_pp": churn_red_seg_pp,
        "CAC_Avoided_250": cac_avoided_conservative,
    }


def main() -> None:
    df = load_data()

    # Define Tiered Program segments (mutually exclusive)
    tier1a_mask = df["is_hv"] & df["is_q4_q5"] & df["is_aging_handset"]
    tier1b_mask = df["is_hv"] & df["is_q4_q5"] & (~df["is_aging_handset"])
    tier1_mask = tier1a_mask | tier1b_mask
    tier2_mask = (~df["is_hv"]) & df["is_q5"] & df["is_intervenable_mass"]
    recommended_mask = tier1_mask | tier2_mask

    # Per-customer cost vector for the Recommended Tiered Program
    tiered_cost = pd.Series(0.0, index=df.index)
    tiered_cost[tier1a_mask] = COST_TIER1A_HV_AGING
    tiered_cost[tier1b_mask] = COST_TIER1B_HV_OTHER
    tiered_cost[tier2_mask] = COST_TIER2_MASS_Q5

    # Strategy definitions
    strategies = [
        ("1. Blanket Retention (All Base)", pd.Series(True, index=df.index), COST_BLANKET),
        ("2. Model-Targeted (p >= 0.40)", df["pred_prob"] >= 0.40, COST_BLANKET),
        ("3. Model-Targeted (p >= 0.50)", df["pred_prob"] >= 0.50, COST_BLANKET),
        ("4. Model Top Quintile (Q5)", df["is_q5"], COST_BLANKET),
        ("5. Recommended Tiered Program (T1+T2)", recommended_mask, tiered_cost),
        ("6. Laser HV + Ageing Handset (T1A)", tier1a_mask, COST_TIER1A_HV_AGING),
    ]

    # 1. Main Strategy Comparison (Base Case: 20% win-back)
    comp_rows = [evaluate_strategy(df, name, mask, cost, BASE_WIN_BACK) for name, mask, cost in strategies]
    comp_df = pd.DataFrame(comp_rows)

    print("=== STRATEGY COMPARISON (100,000 BASE, 20% WIN-BACK) ===")
    cols_show = [
        "Strategy", "Base_100k_N", "Pct_Base", "Seg_Churn_Rate_Pct",
        "Total_Cost_100k", "Gross_Rev_6m", "Net_Impact_6m", "ROI_6m_Pct",
        "Gross_Rev_12m", "Net_Impact_12m", "ROI_12m_Pct",
        "BreakEven_6m_Pct", "BreakEven_12m_Pct", "Base_Churn_Red_pp", "CAC_Avoided_250"
    ]
    print(comp_df[cols_show].to_string(index=False))

    # 2. Tier Breakdown for Recommended Program
    tier_rows = [
        evaluate_strategy(df, "Tier 1A: HV + Q4/Q5 + Ageing (>365d)", tier1a_mask, COST_TIER1A_HV_AGING, BASE_WIN_BACK),
        evaluate_strategy(df, "Tier 1B: HV + Q4/Q5 + Newer (<=365d)", tier1b_mask, COST_TIER1B_HV_OTHER, BASE_WIN_BACK),
        evaluate_strategy(df, "Tier 1 Subtotal (All HV Q4+Q5)", tier1_mask, tiered_cost, BASE_WIN_BACK),
        evaluate_strategy(df, "Tier 2: Mass Q5 Intervenable", tier2_mask, COST_TIER2_MASS_Q5, BASE_WIN_BACK),
        evaluate_strategy(df, "Total Recommended Program (T1 + T2)", recommended_mask, tiered_cost, BASE_WIN_BACK),
    ]
    tier_df = pd.DataFrame(tier_rows)
    print("\n=== RECOMMENDED PROGRAM TIER BREAKDOWN (100,000 BASE, 20% WIN-BACK) ===")
    print(tier_df[cols_show].to_string(index=False))

    # 3. Sensitivity Analysis across Win-Back Rates (10% - 30%)
    sens_rows = []
    for wb in WIN_BACK_RATES:
        sens_rows.append(evaluate_strategy(df, "Recommended Tiered (T1+T2)", recommended_mask, tiered_cost, wb))
    sens_df = pd.DataFrame(sens_rows)
    print("\n=== SENSITIVITY ANALYSIS: RECOMMENDED PROGRAM (10% - 30% WIN-BACK) ===")
    print(sens_df[["Win_Back_Pct", "Retained_100k", "Total_Cost_100k", "Gross_Rev_6m", "Net_Impact_6m", "ROI_6m_Pct", "Gross_Rev_12m", "Net_Impact_12m", "ROI_12m_Pct", "Base_Churn_Red_pp"]].to_string(index=False))

    # 4. Threshold Sensitivity (at 10%, 20%, 30% win-back)
    thresh_rows = []
    for name, mask, cost in [
        ("Model p >= 0.40", df["pred_prob"] >= 0.40, COST_BLANKET),
        ("Model p >= 0.50", df["pred_prob"] >= 0.50, COST_BLANKET),
        ("Model Top Quintile Q5", df["is_q5"], COST_BLANKET),
        ("HV (rev>=70.76) & p>=0.40", df["is_hv"] & (df["pred_prob"] >= 0.40), 35.0),
        ("HV (rev>=70.76) & Q4+Q5", tier1_mask, tiered_cost),
        ("Recommended Tiered (T1+T2)", recommended_mask, tiered_cost),
    ]:
        for wb in [0.10, 0.20, 0.30]:
            thresh_rows.append(evaluate_strategy(df, name, mask, cost, wb))
    thresh_df = pd.DataFrame(thresh_rows)
    print("\n=== THRESHOLD SENSITIVITY (10%, 20%, 30% WIN-BACK) ===")
    print(thresh_df[["Strategy", "Win_Back_Pct", "Base_100k_N", "Total_Cost_100k", "Net_Impact_6m", "ROI_6m_Pct", "Net_Impact_12m", "ROI_12m_Pct", "BreakEven_6m_Pct"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # Chart 1: impact_sensitivity_breakeven.png
    # -----------------------------------------------------------------------
    wb_grid = np.linspace(0.02, 0.35, 67)
    rec_net_6m = [evaluate_strategy(df, "Rec", recommended_mask, tiered_cost, w)["Net_Impact_6m"] / 1e6 for w in wb_grid]
    rec_net_12m = [evaluate_strategy(df, "Rec", recommended_mask, tiered_cost, w)["Net_Impact_12m"] / 1e6 for w in wb_grid]
    blk_net_6m = [evaluate_strategy(df, "Blk", pd.Series(True, index=df.index), COST_BLANKET, w)["Net_Impact_6m"] / 1e6 for w in wb_grid]
    q5_net_6m = [evaluate_strategy(df, "Q5", df["is_q5"], COST_BLANKET, w)["Net_Impact_6m"] / 1e6 for w in wb_grid]

    be_rec_6m = evaluate_strategy(df, "Rec", recommended_mask, tiered_cost, BASE_WIN_BACK)["BreakEven_6m_Pct"]
    be_rec_12m = evaluate_strategy(df, "Rec", recommended_mask, tiered_cost, BASE_WIN_BACK)["BreakEven_12m_Pct"]
    be_blk_6m = evaluate_strategy(df, "Blk", pd.Series(True, index=df.index), COST_BLANKET, BASE_WIN_BACK)["BreakEven_6m_Pct"]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)
    ax.plot(wb_grid * 100, rec_net_12m, color="#0B5394", lw=2.6, label=f"Recommended Tiered (12-Mo Horizon) | Break-Even: {be_rec_12m:.1f}%")
    ax.plot(wb_grid * 100, rec_net_6m, color="#1F77B4", lw=2.4, linestyle="-", label=f"Recommended Tiered (6-Mo Horizon) | Break-Even: {be_rec_6m:.1f}%")
    ax.plot(wb_grid * 100, q5_net_6m, color="#2CA02C", lw=1.8, linestyle="--", label="Model Top Quintile Q5 (6-Mo Horizon)")
    ax.plot(wb_grid * 100, blk_net_6m, color="#D62728", lw=1.8, linestyle=":", label=f"Blanket Retention All 100k (6-Mo) | Break-Even: {be_blk_6m:.1f}%")

    ax.axhline(0, color="#333333", lw=1.1, linestyle="-")
    ax.axvline(20.0, color="#666666", lw=1.0, linestyle="--", alpha=0.7)
    ax.text(20.4, max(rec_net_12m) * 0.90, "Base Case\n(20% Win-Back)", fontsize=9, color="#444444", fontweight="bold")

    # Annotate break-even points on y=0
    ax.scatter([be_rec_12m, be_rec_6m, be_blk_6m], [0, 0, 0], color=["#0B5394", "#1F77B4", "#D62728"], zorder=5, s=45)
    ax.annotate(f"{be_rec_12m:.1f}%", xy=(be_rec_12m, 0), xytext=(be_rec_12m - 2.2, -1.6), fontsize=8.5, color="#0B5394", fontweight="bold")
    ax.annotate(f"{be_rec_6m:.1f}%", xy=(be_rec_6m, 0), xytext=(be_rec_6m + 0.6, -1.6), fontsize=8.5, color="#1F77B4", fontweight="bold")
    ax.annotate(f"{be_blk_6m:.1f}%", xy=(be_blk_6m, 0), xytext=(be_blk_6m + 0.5, -1.6), fontsize=8.5, color="#D62728", fontweight="bold")

    ax.fill_between(wb_grid * 100, 0, rec_net_6m, where=np.array(rec_net_6m) > 0, color="#1F77B4", alpha=0.08)
    ax.set_title("Net Financial Impact & Break-Even Sensitivity (Company A, 100k Base)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Intervention Win-Back Rate (% of Targeted Churners Retained)", fontsize=10.5)
    ax.set_ylabel("Net Financial Impact ($ Millions)", fontsize=10.5)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.1fM"))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fontsize=8.8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "impact_sensitivity_breakeven.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Chart 2: impact_targeting_comparison.png
    # -----------------------------------------------------------------------
    short_labels = [
        "Blanket\n(All 100k)",
        "Model\np >= 0.40",
        "Model\np >= 0.50",
        "Model\nTop Q5",
        "Recommended\nTiered (T1+T2)",
        "Laser HV +\nAgeing (T1A)",
    ]
    net_6m_vals = comp_df["Net_Impact_6m"].values / 1e6
    net_12m_vals = comp_df["Net_Impact_12m"].values / 1e6
    roi_6m_vals = comp_df["ROI_6m_Pct"].values
    roi_12m_vals = comp_df["ROI_12m_Pct"].values

    x = np.arange(len(short_labels))
    w = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=300)

    # Left panel: Net Financial Impact ($M)
    b1 = ax1.bar(x - w/2, net_6m_vals, width=w, color="#4C78A8", label="6-Month Horizon")
    b2 = ax1.bar(x + w/2, net_12m_vals, width=w, color="#0B5394", label="12-Month Horizon")
    ax1.set_title("Net Financial Impact ($M) at 20% Win-Back", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=8.5)
    ax1.set_ylabel("Net Impact ($ Millions)", fontsize=10)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.1fM"))
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.8)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"${h:.2f}M", ha="center", va="bottom", fontsize=7.5)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"${h:.2f}M", ha="center", va="bottom", fontsize=7.5)

    # Right panel: ROI (%)
    r1 = ax2.bar(x - w/2, roi_6m_vals, width=w, color="#54A24B", label="6-Month ROI (%)")
    r2 = ax2.bar(x + w/2, roi_12m_vals, width=w, color="#2E6B27", label="12-Month ROI (%)")
    ax2.set_title("Program ROI (%) at 20% Win-Back", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=8.5)
    ax2.set_ylabel("Return on Investment (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8.8)
    for bar in r1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 6, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)
    for bar in r2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 6, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Targeting Efficiency: Blanket vs. Model-Targeted vs. High-Value Tiered Retention", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "impact_targeting_comparison.png", dpi=300)
    plt.close(fig)

    print("\nSaved figures to:", FIG_DIR)


if __name__ == "__main__":
    main()
