"""
01_eda.py — Exploratory Data Analysis for Company A (US Wireless Telecom)
GCI World 2026 Final Assignment

Loads Client.csv and Record.csv, inner-merges on Customer_ID, audits data quality,
analyzes churn balance, revenue/CLV, intervenable churn drivers, and high-value segments,
and saves 5 publication-quality 300-DPI PNG charts to outputs/figures/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration & Styling
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join("Final Assignment", "telecom")
CLIENT_PATH = os.path.join(DATA_DIR, "Client.csv")
RECORD_PATH = os.path.join(DATA_DIR, "Record.csv")
FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

NAVY = "#1B365D"
TEAL = "#008080"
CORAL = "#D9534F"
AMBER = "#E69F00"
SLATE = "#5C6B73"
LIGHT_BLUE = "#4A90D9"


def load_and_merge():
    client = pd.read_csv(CLIENT_PATH)
    record = pd.read_csv(RECORD_PATH)
    print(f"Client shape: {client.shape}, unique Customer_ID: {client['Customer_ID'].nunique()}")
    print(f"Record shape: {record.shape}, unique Customer_ID: {record['Customer_ID'].nunique()}")
    overlap = sorted(set(client.columns) & set(record.columns))
    print(f"Overlapping columns: {overlap}")
    df = client.merge(record, on="Customer_ID", how="inner")
    print(f"Merged shape: {df.shape}, exact duplicate rows: {df.duplicated().sum()}")
    return df


def audit_data_quality(df):
    print("\n=== DATA QUALITY AUDIT ===")
    print("Dtypes breakdown:")
    print(df.dtypes.value_counts())

    # Missingness
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_df = pd.DataFrame({
        "missing_count": miss,
        "missing_pct": (miss / len(df) * 100).round(3),
        "dtype": df[miss.index].dtypes.astype(str),
    })
    print(f"\nColumns with missing values ({len(miss_df)} of {df.shape[1]}):")
    print(miss_df.to_string())

    # Zero-variance or near-constant check
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    print(f"\nConstant columns: {const_cols}")

    # Anomalies / negative values in key numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nAnomalies / Negative / Extreme checks on key numeric features:")
    check_cols = [
        "rev_Mean", "totrev", "avgrev", "months", "eqpdays", "hnd_price",
        "mou_Mean", "totmou", "drop_blk_Mean", "custcare_Mean", "ovrrev_Mean",
        "ovrmou_Mean", "change_mou", "change_rev", "income", "adults", "lor", "numbcars"
    ]
    rows = []
    for c in check_cols:
        s = df[c].dropna()
        rows.append({
            "col": c,
            "min": round(s.min(), 3),
            "p01": round(s.quantile(0.01), 3),
            "median": round(s.median(), 3),
            "mean": round(s.mean(), 3),
            "p99": round(s.quantile(0.99), 3),
            "max": round(s.max(), 3),
            "n_neg": int((s < 0).sum()),
            "n_zero": int((s == 0).sum()),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Collinearity / duplicate-meaning checks
    print("\nHigh-correlation pairs (|r| >= 0.95) among numeric features:")
    corr = df[num_cols].drop(columns=["Customer_ID"], errors="ignore").corr()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.95:
                pairs.append((cols[i], cols[j], round(val, 4)))
    for a, b, v in sorted(pairs, key=lambda x: -abs(x[2])):
        print(f"  {a} <-> {b}: r = {v}")

    # Target leakage check: top correlations with churn
    churn_corr = df[num_cols].drop(columns=["Customer_ID", "churn"], errors="ignore").corrwith(df["churn"]).dropna()
    top_corr = churn_corr.reindex(churn_corr.abs().sort_values(ascending=False).head(15).index)
    print("\nTop 15 linear correlations with churn:")
    for k, v in top_corr.items():
        print(f"  {k}: {v:+.4f}")

    return miss_df


def analyze_target_and_revenue(df):
    print("\n=== TARGET & REVENUE / CLV ===")
    vc = df["churn"].value_counts().sort_index()
    print(f"Churn = 0 (Retained): {vc[0]:,} ({vc[0]/len(df)*100:.2f}%)")
    print(f"Churn = 1 (Churned):  {vc[1]:,} ({vc[1]/len(df)*100:.2f}%)")

    rev_cols = ["rev_Mean", "totmrc_Mean", "ovrrev_Mean", "avgrev", "totrev", "avg3rev", "avg6rev", "months"]
    print("\nOverall Revenue & Tenure Summary:")
    summary = df[rev_cols].agg(["count", "mean", "std", "min", "median", "max"]).T
    summary["p25"] = df[rev_cols].quantile(0.25)
    summary["p75"] = df[rev_cols].quantile(0.75)
    summary["p90"] = df[rev_cols].quantile(0.90)
    summary["p99"] = df[rev_cols].quantile(0.99)
    print(summary[["count", "mean", "median", "p25", "p75", "p90", "p99", "min", "max"]].round(2).to_string())

    print("\nRevenue & Tenure by Churn Status (Mean / Median):")
    grp = df.groupby("churn")[rev_cols].agg(["mean", "median"])
    print(grp.round(2).to_string())

    # Total monthly revenue pool & churned revenue at risk
    total_monthly_rev = df["rev_Mean"].sum()
    churned_monthly_rev = df.loc[df["churn"] == 1, "rev_Mean"].sum()
    print(f"\nTotal monthly revenue (rev_Mean sum, non-null): ${total_monthly_rev:,.2f}")
    print(f"Churned monthly revenue at risk: ${churned_monthly_rev:,.2f} ({churned_monthly_rev/total_monthly_rev*100:.2f}%)")
    print(f"Annualized churned revenue at risk (x12): ${churned_monthly_rev*12:,.2f}")


def summarize_binned(df, col, bins, labels, title):
    s = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    res = df.groupby(s, observed=False).agg(
        n=("churn", "count"),
        churn_rate=("churn", "mean"),
        rev_mean=("rev_Mean", "mean"),
        totrev_median=("totrev", "median"),
    ).reset_index()
    res["share_pct"] = (res["n"] / len(df) * 100).round(2)
    res["churn_pct"] = (res["churn_rate"] * 100).round(2)
    res["rev_mean"] = res["rev_mean"].round(2)
    res["totrev_median"] = res["totrev_median"].round(2)
    print(f"\n--- {title} ({col}) ---")
    print(res[[col, "n", "share_pct", "churn_pct", "rev_mean", "totrev_median"]].to_string(index=False))
    return res


def summarize_categorical(df, col, top_n=15):
    res = df.fillna({col: "MISSING"}).groupby(col).agg(
        n=("churn", "count"),
        churn_rate=("churn", "mean"),
        rev_mean=("rev_Mean", "mean"),
    ).reset_index()
    res["share_pct"] = (res["n"] / len(df) * 100).round(2)
    res["churn_pct"] = (res["churn_rate"] * 100).round(2)
    res["rev_mean"] = res["rev_mean"].round(2)
    res = res.sort_values("n", ascending=False).head(top_n)
    print(f"\n--- Categorical: {col} (top {top_n}) ---")
    print(res[[col, "n", "share_pct", "churn_pct", "rev_mean"]].to_string(index=False))
    return res


def analyze_drivers_and_segments(df):
    print("\n=== INTERVENVABLE CHURN DRIVERS & SEGMENTS ===")

    # 1. eqpdays
    eqp_res = summarize_binned(
        df, "eqpdays",
        bins=[-10, 0, 180, 365, 540, 730, 5000],
        labels=["<=0 (Anomaly)", "1-180d", "181-365d", "366-540d", "541-730d", ">730d"],
        title="Handset Age"
    )

    # 2. drop_blk_Mean
    db_res = summarize_binned(
        df, "drop_blk_Mean",
        bins=[-0.01, 0, 2, 5, 10, 20, 1000],
        labels=["0", "(0, 2]", "(2, 5]", "(5, 10]", "(10, 20]", ">20"],
        title="Dropped + Blocked Calls"
    )

    # 3. custcare_Mean
    cc_res = summarize_binned(
        df, "custcare_Mean",
        bins=[-0.01, 0, 1, 3, 5, 1000],
        labels=["0", "(0, 1]", "(1, 3]", "(3, 5]", ">5"],
        title="Customer Care Calls"
    )

    # 4. ovrrev_Mean & overage ratio
    ovr_res = summarize_binned(
        df, "ovrrev_Mean",
        bins=[-0.01, 0, 5, 15, 30, 10000],
        labels=["$0", "$0.01-$5", "$5-$15", "$15-$30", ">$30"],
        title="Overage Revenue"
    )

    df["ovr_ratio"] = np.where(df["rev_Mean"] > 0, df["ovrrev_Mean"] / df["rev_Mean"], 0.0)
    summarize_binned(
        df, "ovr_ratio",
        bins=[-0.01, 0, 0.10, 0.25, 0.50, 100],
        labels=["0%", "(0%, 10%]", "(10%, 25%]", "(25%, 50%]", ">50%"],
        title="Overage Share of Monthly Revenue"
    )

    # 5. change_mou & change_rev
    cmou_res = summarize_binned(
        df, "change_mou",
        bins=[-100000, -200, -50, 0, 50, 200, 100000],
        labels=["<-200", "-200 to -50", "-50 to 0", "0 to +50", "+50 to +200", ">+200"],
        title="Change in Monthly Minutes (vs 3-mo avg)"
    )

    crev_res = summarize_binned(
        df, "change_rev",
        bins=[-100000, -30, -10, 0, 10, 30, 100000],
        labels=["<-30", "-30 to -10", "-10 to 0", "0 to +10", "+10 to +30", ">+30"],
        title="Change in Monthly Revenue (vs 3-mo avg)"
    )

    # 6. months (tenure)
    ten_res = summarize_binned(
        df, "months",
        bins=[0, 6, 12, 18, 24, 36, 100],
        labels=["1-6m", "7-12m", "13-18m", "19-24m", "25-36m", ">36m"],
        title="Tenure in Months"
    )

    # 7. hnd_price
    hnd_res = summarize_binned(
        df, "hnd_price",
        bins=[0, 50, 100, 150, 200, 1000],
        labels=["<=$50", "$51-$100", "$101-$150", "$151-$200", ">$200"],
        title="Current Handset Price"
    )

    # 8. Categoricals
    summarize_categorical(df, "asl_flag", top_n=5)
    summarize_categorical(df, "refurb_new", top_n=5)
    summarize_categorical(df, "area", top_n=25)
    summarize_categorical(df, "crclscod", top_n=15)
    summarize_categorical(df, "new_cell", top_n=5)
    summarize_categorical(df, "dualband", top_n=5)

    # 9. Revenue tiers & High-Value At-Risk Segments
    df["rev_tier"] = pd.qcut(df["rev_Mean"], q=4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])
    rev_tier_res = df.groupby("rev_tier", observed=False).agg(
        n=("churn", "count"),
        churn_pct=("churn", lambda x: round(x.mean() * 100, 2)),
        rev_mean=("rev_Mean", "mean"),
        rev_min=("rev_Mean", "min"),
        rev_max=("rev_Mean", "max"),
        totrev_median=("totrev", "median"),
    ).reset_index()
    print("\n--- Revenue Quartiles (rev_Mean) ---")
    print(rev_tier_res.round(2).to_string(index=False))

    # High-value (top quartile rev_Mean) x eqpdays > 365
    hv = df[df["rev_tier"] == "Q4 (Highest)"]
    hv_old_eqp = hv[hv["eqpdays"] > 365]
    hv_high_drop = hv[hv["drop_blk_Mean"] > 10]
    hv_any_risk = hv[(hv["eqpdays"] > 365) | (hv["drop_blk_Mean"] > 10) | (hv["custcare_Mean"] == 0)]
    print("\n--- High-Value (Q4 rev_Mean) Risk Subsegments ---")
    print(f"Q4 total: n={len(hv):,}, churn={hv['churn'].mean()*100:.2f}%, mean rev=${hv['rev_Mean'].mean():.2f}")
    print(f"Q4 & eqpdays > 365d: n={len(hv_old_eqp):,} ({len(hv_old_eqp)/len(hv)*100:.1f}% of Q4), churn={hv_old_eqp['churn'].mean()*100:.2f}%, monthly rev=${hv_old_eqp['rev_Mean'].sum():,.0f}")
    print(f"Q4 & drop_blk_Mean > 10: n={len(hv_high_drop):,} ({len(hv_high_drop)/len(hv)*100:.1f}% of Q4), churn={hv_high_drop['churn'].mean()*100:.2f}%, monthly rev=${hv_high_drop['rev_Mean'].sum():,.0f}")

    # Cross-tab: rev_tier x eqpdays bin
    df["eqp_bin"] = pd.cut(
        df["eqpdays"],
        bins=[-10, 180, 365, 540, 730, 5000],
        labels=["<=180d", "181-365d", "366-540d", "541-730d", ">730d"]
    )
    ct = df.groupby(["rev_tier", "eqp_bin"], observed=False)["churn"].mean().unstack() * 100
    print("\n--- Churn Rate (%) by Revenue Quartile x Handset Age ---")
    print(ct.round(2).to_string())

    return {
        "eqp": eqp_res,
        "db": db_res,
        "cc": cc_res,
        "ovr": ovr_res,
        "cmou": cmou_res,
        "ten": ten_res,
        "ct": ct,
    }


# ---------------------------------------------------------------------------
# Plotting Functions (300 DPI)
# ---------------------------------------------------------------------------
def plot_churn_by_eqpdays(eqp_res, overall_churn):
    # Exclude the tiny <=0 anomaly bin from bar visual or note it in subtitle
    sub = eqp_res[eqp_res["eqpdays"] != "<=0 (Anomaly)"].copy()
    fig, ax1 = plt.subplots(figsize=(9, 5.2))

    x = np.arange(len(sub))
    bars = ax1.bar(x, sub["n"] / 1000, width=0.55, color="#D6E4F0", edgecolor=NAVY, linewidth=0.8, label="Subscribers (thousands)")
    ax1.set_ylabel("Subscribers (Thousands)", color=NAVY, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=NAVY)
    ax1.set_ylim(0, sub["n"].max() / 1000 * 1.38)

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(x, sub["churn_pct"], color=CORAL, marker="o", linewidth=2.6, markersize=8, label="Churn Rate (%)")
    ax2.axhline(overall_churn, color=SLATE, linestyle="--", linewidth=1.2, label=f"Overall Avg ({overall_churn:.1f}%)")
    ax2.set_ylabel("Churn Rate (%)", color=CORAL, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=CORAL)
    ax2.set_ylim(30, 66)

    for i, (_, row) in enumerate(sub.iterrows()):
        ax2.annotate(
            f"{row['churn_pct']:.1f}%",
            (i, row["churn_pct"]),
            textcoords="offset points",
            xytext=(0, -16 if i == 0 else 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=CORAL,
        )
        ax1.text(
            i, row["n"] / 1000 + 0.8,
            f"n={int(row['n']):,}",
            ha="center", va="bottom", fontsize=8.5, color=NAVY
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["eqpdays"], fontsize=10)
    ax1.set_xlabel("Handset Age (eqpdays)", fontsize=11, fontweight="bold")
    plt.title("Churn Risk Climbs Sharply as Handset Age Exceeds 1 Year", fontsize=13, fontweight="bold", pad=14, color="#111111")
    fig.text(0.5, 0.01, "Source: Company A Telecom Dataset (n=99,999 valid eqpdays; 19 rows with eqpdays <= 0 omitted).", ha="center", fontsize=8.5, color="#555555")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIG_DIR, "eda_churn_by_eqpdays.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_quality_and_care(db_res, cc_res, overall_churn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Left: Dropped/Blocked Calls
    x1 = np.arange(len(db_res))
    colors1 = [TEAL if v < overall_churn else CORAL for v in db_res["churn_pct"]]
    ax1.bar(x1, db_res["churn_pct"], width=0.58, color=colors1, alpha=0.88, edgecolor="#333333", linewidth=0.6)
    ax1.axhline(overall_churn, color=SLATE, linestyle="--", linewidth=1.2, label=f"Overall Avg ({overall_churn:.1f}%)")
    for i, row in db_res.iterrows():
        ax1.text(i, row["churn_pct"] + 0.8, f"{row['churn_pct']:.1f}%\n(n={int(row['n']):,})", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(db_res["drop_blk_Mean"], fontsize=9.5)
    ax1.set_xlabel("Mean Dropped + Blocked Calls / Month (drop_blk_Mean)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Churn Rate (%)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 68)
    ax1.set_title("Network Quality: Dropped & Blocked Calls", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", fontsize=8.8)

    # Right: Customer Care Calls
    x2 = np.arange(len(cc_res))
    colors2 = [CORAL if v >= overall_churn else TEAL for v in cc_res["churn_pct"]]
    ax2.bar(x2, cc_res["churn_pct"], width=0.55, color=colors2, alpha=0.88, edgecolor="#333333", linewidth=0.6)
    ax2.axhline(overall_churn, color=SLATE, linestyle="--", linewidth=1.2, label=f"Overall Avg ({overall_churn:.1f}%)")
    for i, row in cc_res.iterrows():
        ax2.text(i, row["churn_pct"] + 0.8, f"{row['churn_pct']:.1f}%\n(n={int(row['n']):,})", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cc_res["custcare_Mean"], fontsize=9.5)
    ax2.set_xlabel("Mean Customer Care Calls / Month (custcare_Mean)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Churn Rate (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 68)
    ax2.set_title("Service Engagement: Customer Care Contacts", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", fontsize=8.8)

    fig.suptitle("Service Quality & Care Engagement Strongly Differentiate Churn Risk", fontsize=13.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01, "Source: Company A Telecom Dataset (n=100,000). Note: 0 care calls (55.7% of base) has the highest churn (51.8%) — silent churners.", ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = os.path.join(FIG_DIR, "eda_churn_by_quality_care.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_value_vs_churn_segments(ct):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    data = ct.values
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=38, vmax=60)

    ax.set_xticks(np.arange(len(ct.columns)))
    ax.set_xticklabels(ct.columns, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(ct.index)))
    ax.set_yticklabels(ct.index, fontsize=10, fontweight="bold")
    ax.set_xlabel("Handset Age (eqpdays)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Monthly Revenue Quartile (rev_Mean)", fontsize=11, fontweight="bold", labelpad=8)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt_color = "white" if val > 51 else "#111111"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Churn Rate (%)", fontsize=10, fontweight="bold")

    ax.grid(False)
    plt.title("Churn Rate (%) Matrix: Monthly Revenue Tier vs. Handset Age", fontsize=13, fontweight="bold", pad=12)
    fig.text(0.5, 0.01, "Source: Company A Telecom Dataset. Highest-revenue subscribers (Q4) with >1-year-old handsets show 54.2%-59.5% churn.", ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(FIG_DIR, "eda_value_vs_churn_segments.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_overage_and_usage_change(cmou_res, ovr_res, overall_churn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Left: change_mou
    x1 = np.arange(len(cmou_res))
    colors1 = [CORAL if v > overall_churn else LIGHT_BLUE for v in cmou_res["churn_pct"]]
    ax1.bar(x1, cmou_res["churn_pct"], width=0.58, color=colors1, alpha=0.9, edgecolor="#333333", linewidth=0.6)
    ax1.axhline(overall_churn, color=SLATE, linestyle="--", linewidth=1.2, label=f"Overall Avg ({overall_churn:.1f}%)")
    for i, row in cmou_res.iterrows():
        ax1.text(i, row["churn_pct"] + 0.8, f"{row['churn_pct']:.1f}%\n(n={int(row['n']):,})", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cmou_res["change_mou"], fontsize=8.8, rotation=15)
    ax1.set_xlabel("Change in Monthly Minutes vs. 3-Mo Avg (change_mou)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Churn Rate (%)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 68)
    ax1.set_title("Usage Drop Is a Leading Churn Indicator", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(loc="lower right", fontsize=8.8)

    # Right: ovrrev_Mean
    x2 = np.arange(len(ovr_res))
    colors2 = [TEAL if v < overall_churn else AMBER for v in ovr_res["churn_pct"]]
    ax2.bar(x2, ovr_res["churn_pct"], width=0.55, color=colors2, alpha=0.9, edgecolor="#333333", linewidth=0.6)
    ax2.axhline(overall_churn, color=SLATE, linestyle="--", linewidth=1.2, label=f"Overall Avg ({overall_churn:.1f}%)")
    for i, row in ovr_res.iterrows():
        ax2.text(i, row["churn_pct"] + 0.8, f"{row['churn_pct']:.1f}%\n(n={int(row['n']):,})", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["0", "0.01–5", "5–15", "15–30", ">30"], fontsize=9.5)
    ax2.set_xlabel("Mean Monthly Overage Revenue ($, ovrrev_Mean)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Churn Rate (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 68)
    ax2.set_title("Billing Overage vs. Churn Rate", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(loc="lower right", fontsize=8.8)

    fig.suptitle("Behavioral Signals: Sharp Usage Declines & Overage Patterns", fontsize=13.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01, "Source: Company A Telecom Dataset (n=100,000). Declining minutes (< -200) signals elevated churn (55.0% vs 45.4% for stable/growing).", ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = os.path.join(FIG_DIR, "eda_overage_and_usage_change.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_missingness_and_tenure(miss_df, ten_res, overall_churn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # Left: Top 15 missing columns
    top_miss = miss_df.head(15).iloc[::-1]
    y = np.arange(len(top_miss))
    bar_colors = [CORAL if p > 30 else (AMBER if p > 5 else LIGHT_BLUE) for p in top_miss["missing_pct"]]
    ax1.barh(y, top_miss["missing_pct"], color=bar_colors, edgecolor="#333333", linewidth=0.5)
    for i, (col, row) in enumerate(top_miss.iterrows()):
        ax1.text(row["missing_pct"] + 1.0, i, f"{row['missing_pct']:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(top_miss.index, fontsize=9)
    ax1.set_xlabel("Missing Values (%)", fontsize=10.5, fontweight="bold")
    ax1.set_xlim(0, 108)
    ax1.set_title("Top 15 Columns by Missingness (%)", fontsize=11.5, fontweight="bold", pad=10)

    # Right: Tenure vs Churn & Revenue
    x2 = np.arange(len(ten_res))
    ax2.bar(x2, ten_res["rev_mean"], width=0.52, color="#D6E4F0", edgecolor=NAVY, linewidth=0.8, label="Mean Monthly Rev ($)")
    ax2.set_ylabel("Mean Monthly Revenue ($)", color=NAVY, fontsize=10.5, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=NAVY)
    ax2.set_ylim(0, ten_res["rev_mean"].max() * 1.32)

    ax2_r = ax2.twinx()
    ax2_r.grid(False)
    ax2_r.plot(x2, ten_res["churn_pct"], color=CORAL, marker="s", linewidth=2.4, markersize=7, label="Churn Rate (%)")
    ax2_r.axhline(overall_churn, color=SLATE, linestyle="--", linewidth=1.1, label=f"Avg Churn ({overall_churn:.1f}%)")
    ax2_r.set_ylabel("Churn Rate (%)", color=CORAL, fontsize=10.5, fontweight="bold")
    ax2_r.tick_params(axis="y", labelcolor=CORAL)
    ax2_r.set_ylim(30, 68)

    for i, row in ten_res.iterrows():
        ax2_r.annotate(f"{row['churn_pct']:.1f}%", (i, row["churn_pct"]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.8, fontweight="bold", color=CORAL)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(ten_res["months"], fontsize=9.5)
    ax2.set_xlabel("Customer Tenure (months)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Churn Rate & Monthly Revenue by Tenure", fontsize=11.5, fontweight="bold", pad=10)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8.3, frameon=True, facecolor="white")

    fig.suptitle("Data Quality Audit & Customer Lifecycle Overview", fontsize=13.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01, "Source: Company A Telecom Dataset (n=100,000, 100 columns). Demographic fields (numbcars, dwllsize, income) have highest missingness.", ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out = os.path.join(FIG_DIR, "eda_missingness_and_tenure.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def main():
    df = load_and_merge()
    miss_df = audit_data_quality(df)
    analyze_target_and_revenue(df)
    results = analyze_drivers_and_segments(df)
    overall_churn = df["churn"].mean() * 100

    print("\n=== GENERATING FIGURES ===")
    plot_churn_by_eqpdays(results["eqp"], overall_churn)
    plot_quality_and_care(results["db"], results["cc"], overall_churn)
    plot_value_vs_churn_segments(results["ct"])
    plot_overage_and_usage_change(results["cmou"], results["ovr"], overall_churn)
    plot_missingness_and_tenure(miss_df, results["ten"], overall_churn)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
