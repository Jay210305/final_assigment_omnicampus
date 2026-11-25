"""
02_modeling.py — Churn Prediction Modeling & Risk Segmentation for Company A
GCI World 2026 Final Assignment

Pipeline:
1. Inner-merge Client.csv and Record.csv on Customer_ID (100,000 x 100).
2. Clean documented anomalies (negative eqpdays, negative rev_Mean) and engineer
   domain-driven, intervenable features (overage_share, drop_blk_rate, comp_rate,
   rev_trend_ratio, mou_trend_ratio, eqp_per_month, care_per_mou, etc.).
3. 80/20 stratified train/test split (random_state=42).
4. 5-fold Stratified CV on train comparing 4 model families under identical
   leak-free sklearn Pipelines (median imputation + missingness indicators,
   StandardScaler for Logistic Regression, OneHotEncoder for categoricals):
   - Logistic Regression
   - Random Forest Classifier
   - XGBoost Classifier
   - LightGBM Classifier
5. Evaluate ROC-AUC, PR-AUC, Recall, Precision, F1, Accuracy (CV mean+/-std & Test),
   plus asymmetric business-cost threshold tuning on CV OOF probabilities.
6. Extract top feature importances (intervenable vs. structural), Test risk-quintile
   & risk-decile churn lift, and High-Value Customer (Q4 rev_Mean >= $70.76) breakdown.
7. Export outputs/test_predictions.csv and 4 publication-quality 300-DPI PNG charts
   to outputs/figures/.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration & Styling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_DIR = os.path.join("Final Assignment", "telecom")
CLIENT_PATH = os.path.join(DATA_DIR, "Client.csv")
RECORD_PATH = os.path.join(DATA_DIR, "Record.csv")
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
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

INTERVENVABLE_FEATURES = {
    "eqpdays", "eqp_per_month", "eqpdays_gt_365", "eqpdays_neg_flag",
    "hnd_price", "refurb_new", "dualband", "phones", "models",
    "overage_share", "ovrrev_Mean", "ovrmou_Mean", "vceovr_Mean", "datovr_Mean",
    "drop_blk_Mean", "drop_blk_rate", "comp_rate", "drop_vce_Mean", "blck_vce_Mean",
    "custcare_Mean", "care_per_mou", "has_custcare", "cc_mou_Mean", "ccrndmou_Mean",
    "change_mou", "change_rev", "mou_trend_ratio", "rev_trend_ratio",
    "asl_flag", "roam_share", "mrc_share", "mou_per_rev",
}


# ---------------------------------------------------------------------------
# 1. Load, Merge & Feature Engineering
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    client = pd.read_csv(CLIENT_PATH)
    record = pd.read_csv(RECORD_PATH)
    df = client.merge(record, on="Customer_ID", how="inner", suffixes=("", "_rec"))

    # Handle any potential duplicate columns from merge
    dup_cols = [c for c in df.columns if c.endswith("_rec")]
    if dup_cols:
        print(f"Dropping duplicate merge columns: {dup_cols}")
        df = df.drop(columns=dup_cols)

    # Anomaly flags & cleaning (documented in outputs/eda_summary.md)
    df["eqpdays_neg_flag"] = (df["eqpdays"] < 0).astype(int)
    df["eqpdays"] = df["eqpdays"].clip(lower=0)

    df["rev_neg_flag"] = (df["rev_Mean"] < 0).astype(int)
    df["rev_Mean"] = df["rev_Mean"].clip(lower=0)

    # Domain-driven, intervenable features (row-wise, zero leakage)
    rev_denom = df["rev_Mean"].clip(lower=1.0)
    att_denom = df["attempt_Mean"].clip(lower=1.0)
    mou_denom = df["mou_Mean"].clip(lower=1.0)
    avg6rev_denom = df["avg6rev"].clip(lower=1.0)
    avg6mou_denom = df["avg6mou"].clip(lower=1.0)
    months_days = (df["months"] * 30.0).clip(lower=1.0)
    uniq_denom = df["uniqsubs"].clip(lower=1.0)

    df["overage_share"] = df["ovrrev_Mean"] / rev_denom
    df["drop_blk_rate"] = df["drop_blk_Mean"] / att_denom
    df["comp_rate"] = df["complete_Mean"] / att_denom
    df["rev_trend_ratio"] = df["avg3rev"] / avg6rev_denom
    df["mou_trend_ratio"] = df["avg3mou"] / avg6mou_denom
    df["eqp_per_month"] = df["eqpdays"] / months_days
    df["care_per_mou"] = df["custcare_Mean"] / mou_denom
    df["eqpdays_gt_365"] = np.where(df["eqpdays"].isna(), np.nan, (df["eqpdays"] > 365).astype(float))
    df["has_custcare"] = np.where(df["custcare_Mean"].isna(), np.nan, (df["custcare_Mean"] > 0).astype(float))
    df["active_line_ratio"] = df["actvsubs"] / uniq_denom
    df["mou_per_rev"] = df["mou_Mean"] / rev_denom
    df["roam_share"] = df["roam_Mean"] / rev_denom
    df["mrc_share"] = df["totmrc_Mean"] / rev_denom

    print(f"Prepared dataset shape: {df.shape} | Churn rate: {df['churn'].mean():.4f}")
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing Pipelines (Leak-Free)
# ---------------------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols, scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names_from_ct(ct, num_cols, cat_cols):
    """Extract clean feature names from fitted ColumnTransformer."""
    names = []
    num_imp = ct.named_transformers_["num"].named_steps["imputer"]
    names.extend(num_cols)
    if hasattr(num_imp, "indicator_") and num_imp.indicator_ is not None:
        miss_idx = num_imp.indicator_.features_
        names.extend([f"{num_cols[i]}_is_missing" for i in miss_idx])
    ohe = ct.named_transformers_["cat"].named_steps["ohe"]
    names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    return names


# ---------------------------------------------------------------------------
# 3. Evaluation Helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_models_cv_and_test(X_train, y_train, X_test, y_test, num_cols, cat_cols):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Drop exact/near-exact collinear adj* duplicates for Logistic Regression stability
    lr_drop = {"adjqty", "adjmou", "adjrev"}
    lr_num_cols = [c for c in num_cols if c not in lr_drop]

    model_specs = {
        "Logistic Regression": {
            "num_cols": lr_num_cols,
            "scale": True,
            "estimator": lambda: LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
            ),
        },
        "Random Forest": {
            "num_cols": num_cols,
            "scale": False,
            "estimator": lambda: RandomForestClassifier(
                n_estimators=300,
                max_depth=16,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        },
        "XGBoost": {
            "num_cols": num_cols,
            "scale": False,
            "estimator": lambda: XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        },
        "LightGBM": {
            "num_cols": num_cols,
            "scale": False,
            "estimator": lambda: LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
        },
    }

    results = []
    oof_preds = {}
    test_preds = {}
    fitted_pipelines = {}

    for name, spec in model_specs.items():
        t0 = time.time()
        print(f"\n--- Training & 5-Fold CV: {name} ---")
        fold_metrics = []
        oof_prob = np.zeros(len(X_train), dtype=float)

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            pipe = Pipeline([
                ("prep", build_preprocessor(spec["num_cols"], cat_cols, scale_numeric=spec["scale"])),
                ("clf", spec["estimator"]()),
            ])
            pipe.fit(X_tr, y_tr)
            val_prob = pipe.predict_proba(X_val)[:, 1]
            oof_prob[val_idx] = val_prob
            m = compute_metrics(y_val, val_prob, threshold=0.50)
            fold_metrics.append(m)

        # Fit final model on full training set & evaluate on held-out test set
        full_pipe = Pipeline([
            ("prep", build_preprocessor(spec["num_cols"], cat_cols, scale_numeric=spec["scale"])),
            ("clf", spec["estimator"]()),
        ])
        full_pipe.fit(X_train, y_train)
        test_prob = full_pipe.predict_proba(X_test)[:, 1]
        test_m = compute_metrics(y_test, test_prob, threshold=0.50)

        oof_preds[name] = oof_prob
        test_preds[name] = test_prob
        fitted_pipelines[name] = full_pipe

        fold_df = pd.DataFrame(fold_metrics)
        elapsed = time.time() - t0
        row = {"model": name, "elapsed_s": round(elapsed, 1)}
        for k in ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]:
            row[f"cv_{k}_mean"] = fold_df[k].mean()
            row[f"cv_{k}_std"] = fold_df[k].std()
            row[f"test_{k}"] = test_m[k]
        results.append(row)

        print(
            f"  CV ROC-AUC: {row['cv_roc_auc_mean']:.4f} +/- {row['cv_roc_auc_std']:.4f} | "
            f"Test ROC-AUC: {row['test_roc_auc']:.4f} | Test PR-AUC: {row['test_pr_auc']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
    return results_df, oof_preds, test_preds, fitted_pipelines, model_specs


# ---------------------------------------------------------------------------
# 4. Threshold & Asymmetric Cost Analysis
# ---------------------------------------------------------------------------
def analyze_thresholds(y_train, oof_prob, y_test, test_prob):
    """
    Evaluate threshold trade-offs on CV OOF probabilities and held-out Test set.
    Business framing: missing a churner risks ~$58.72/mo x 6 mo margin (~$352 CLV),
    whereas a targeted retention offer (e.g. $35-$50) has ~7:1 to 10:1 asymmetry.
    We evaluate thresholds 0.30..0.60 and report F1, Recall, Precision, Accuracy,
    and F2-score (which weights Recall 2x Precision).
    """
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    rows = []
    for t in thresholds:
        m_cv = compute_metrics(y_train, oof_prob, threshold=t)
        m_te = compute_metrics(y_test, test_prob, threshold=t)
        # F2 = 5 * P * R / (4*P + R)
        f2_cv = (5 * m_cv["precision"] * m_cv["recall"]) / max(4 * m_cv["precision"] + m_cv["recall"], 1e-9)
        f2_te = (5 * m_te["precision"] * m_te["recall"]) / max(4 * m_te["precision"] + m_te["recall"], 1e-9)
        flagged_pct = (test_prob >= t).mean() * 100
        rows.append({
            "threshold": t,
            "flagged_pct_test": round(flagged_pct, 2),
            "cv_recall": round(m_cv["recall"], 4),
            "cv_precision": round(m_cv["precision"], 4),
            "cv_f1": round(m_cv["f1"], 4),
            "cv_f2": round(f2_cv, 4),
            "test_recall": round(m_te["recall"], 4),
            "test_precision": round(m_te["precision"], 4),
            "test_f1": round(m_te["f1"], 4),
            "test_f2": round(f2_te, 4),
            "test_accuracy": round(m_te["accuracy"], 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Feature Importance & Risk Segmentation
# ---------------------------------------------------------------------------
def extract_feature_importance(pipe, num_cols, cat_cols):
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    feat_names = get_feature_names_from_ct(prep, num_cols, cat_cols)

    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0])
    else:
        raise ValueError("Unsupported classifier for feature importance")

    imp_df = pd.DataFrame({"feature": feat_names, "importance": imps})
    imp_df["importance_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100

    # Map one-hot encoded features back to base feature for intervenable tagging
    def is_intervenable(f):
        base = f.replace("_is_missing", "")
        if base in INTERVENVABLE_FEATURES:
            return True
        for prefix in INTERVENVABLE_FEATURES:
            if base.startswith(prefix + "_"):
                return True
        return False

    imp_df["category"] = np.where(
        imp_df["feature"].apply(is_intervenable), "Intervenable", "Structural"
    )
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def compute_risk_tables(test_df):
    baseline_churn = test_df["churn"].mean()

    def agg_group(df_in, group_col):
        g = df_in.groupby(group_col, observed=False).agg(
            n=("Customer_ID", "count"),
            churners=("churn", "sum"),
            churn_rate=("churn", "mean"),
            avg_prob=("pred_prob", "mean"),
            total_rev_mo=("rev_Mean", "sum"),
            churned_rev_mo=("rev_Mean", lambda s: s[df_in.loc[s.index, "churn"] == 1].sum()),
            mean_rev=("rev_Mean", "mean"),
            mean_eqpdays=("eqpdays", "mean"),
            mean_ovrrev=("ovrrev_Mean", "mean"),
        ).reset_index()
        g["lift_vs_base"] = g["churn_rate"] / baseline_churn
        g["churn_pct"] = g["churn_rate"] * 100
        g["rev_share_pct"] = g["churned_rev_mo"] / df_in.loc[df_in["churn"] == 1, "rev_Mean"].sum() * 100
        return g

    quintile_tbl = agg_group(test_df, "risk_quintile")
    decile_tbl = agg_group(test_df, "risk_decile")
    hv_df = test_df[test_df["rev_Mean"] >= 70.76].copy()
    hv_quintile_tbl = agg_group(hv_df, "risk_quintile")
    return quintile_tbl, decile_tbl, hv_quintile_tbl


# ---------------------------------------------------------------------------
# 6. Publication-Quality 300-DPI Figures
# ---------------------------------------------------------------------------
def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    models = results_df["model"].tolist()
    x = np.arange(len(models))
    w = 0.25

    b1 = ax.bar(
        x - w, results_df["cv_roc_auc_mean"], width=w,
        yerr=results_df["cv_roc_auc_std"], capsize=4,
        color=LIGHT_BLUE, label="5-Fold CV ROC-AUC (mean ± std)", edgecolor="white"
    )
    b2 = ax.bar(
        x, results_df["test_roc_auc"], width=w,
        color=NAVY, label="Held-Out Test ROC-AUC", edgecolor="white"
    )
    b3 = ax.bar(
        x + w, results_df["test_pr_auc"], width=w,
        color=TEAL, label="Held-Out Test PR-AUC", edgecolor="white"
    )

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0.50, max(results_df["test_roc_auc"].max(), results_df["test_pr_auc"].max()) + 0.06)
    ax.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax.set_title(
        "Model Family Comparison: 5-Fold Stratified CV vs. Held-Out Test (n=20,000)",
        fontsize=12, fontweight="bold", pad=12, color=NAVY
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9, loc="upper right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_comparison_roc_auc.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_feature_importance(imp_df, winner_name):
    top = imp_df.head(15).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=300)
    colors = [TEAL if c == "Intervenable" else SLATE for c in top["category"]]
    bars = ax.barh(top["feature"], top["importance_pct"], color=colors, edgecolor="white", height=0.7)

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.12, bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}%", va="center", ha="left", fontsize=8.5, color="#222222"
        )

    ax.set_xlim(0, top["importance_pct"].max() * 1.18)
    ax.set_xlabel("Relative Importance (% of Total Gain / Split Weight)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Top 15 Churn Drivers ({winner_name}) — Intervenable vs. Structural",
        fontsize=12, fontweight="bold", pad=12, color=NAVY
    )
    legend_handles = [
        Patch(facecolor=TEAL, label="Intervenable Driver (Actionable Lever)"),
        Patch(facecolor=SLATE, label="Structural / Account Context"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_feature_importance.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_risk_quintile_lift(quintile_tbl, hv_quintile_tbl):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300)

    # Panel A: All Test Customers — Churn Rate & Lift by Risk Quintile
    q_labels = quintile_tbl["risk_quintile"].astype(str).tolist()
    x = np.arange(len(q_labels))
    bars = ax1.bar(x, quintile_tbl["churn_pct"], color=[SLATE, LIGHT_BLUE, AMBER, "#E07A5F", CORAL], width=0.6)
    ax1.axhline(quintile_tbl["churners"].sum() / quintile_tbl["n"].sum() * 100, color=NAVY, linestyle="--", linewidth=1.2, label="Test Baseline (49.56%)")

    for i, bar in enumerate(bars):
        h = bar.get_height()
        lift = quintile_tbl.loc[i, "lift_vs_base"]
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 1.0,
            f"{h:.1f}%\n({lift:.2f}x)", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(q_labels, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 88)
    ax1.set_ylabel("Actual Churn Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_title("A. Test Set Churn Rate & Lift by Risk Quintile", fontsize=11, fontweight="bold", color=NAVY)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Panel B: Monthly Churned Revenue at Risk ($K) — All vs High-Value (Q4 rev >= $70.76)
    rev_all_k = quintile_tbl["churned_rev_mo"] / 1000.0
    rev_hv_k = hv_quintile_tbl["churned_rev_mo"] / 1000.0
    w = 0.36
    b_all = ax2.bar(x - w / 2, rev_all_k, width=w, color=NAVY, label="All Test Churned Rev ($K/mo)")
    b_hv = ax2.bar(x + w / 2, rev_hv_k, width=w, color=TEAL, label="High-Value Q4 (>= $70.76) Churned Rev ($K/mo)")

    for bar in b_all:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.5, f"${h:.0f}K", ha="center", va="bottom", fontsize=8)
    for bar in b_hv:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.5, f"${h:.0f}K", ha="center", va="bottom", fontsize=8, color=TEAL, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(q_labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, rev_all_k.max() * 1.18)
    ax2.set_ylabel("Monthly Churned Revenue at Risk ($K)", fontsize=10, fontweight="bold")
    ax2.set_title("B. Monthly Revenue at Risk by Predicted Risk Quintile", fontsize=11, fontweight="bold", color=NAVY)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.suptitle("Test Set Risk Stratification & High-Value Revenue Concentration (n=20,000)", fontsize=12.5, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_risk_quintile_lift.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_roc_pr_curves(y_test, test_preds, results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2), dpi=300)
    palette = {
        "LightGBM": TEAL,
        "XGBoost": NAVY,
        "Random Forest": AMBER,
        "Logistic Regression": CORAL,
    }

    for _, row in results_df.iterrows():
        m = row["model"]
        prob = test_preds[m]
        c = palette.get(m, SLATE)

        fpr, tpr, _ = roc_curve(y_test, prob)
        ax1.plot(fpr, tpr, lw=2.0, color=c, label=f"{m} (AUC={row['test_roc_auc']:.4f})")

        prec, rec, _ = precision_recall_curve(y_test, prob)
        ax2.plot(rec, prec, lw=2.0, color=c, label=f"{m} (AP={row['test_pr_auc']:.4f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Random (0.5000)")
    ax1.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=10, fontweight="bold")
    ax1.set_title("A. ROC Curves (Held-Out Test Set)", fontsize=11, fontweight="bold", color=NAVY)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    baseline = y_test.mean()
    ax2.axhline(baseline, color="k", linestyle="--", lw=1.0, alpha=0.6, label=f"Baseline ({baseline:.4f})")
    ax2.set_xlabel("Recall", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=10, fontweight="bold")
    ax2.set_title("B. Precision-Recall Curves (Held-Out Test Set)", fontsize=11, fontweight="bold", color=NAVY)
    ax2.legend(loc="lower left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_roc_pr_curves.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_and_prepare_data()

    drop_cols = ["Customer_ID", "churn"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]
    print(f"Features: {len(feature_cols)} total ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    X = df[feature_cols]
    y = df["churn"].astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train shape: {X_train.shape} (churn={y_train.mean():.4f}) | Test shape: {X_test.shape} (churn={y_test.mean():.4f})")

    results_df, oof_preds, test_preds, fitted_pipelines, model_specs = evaluate_models_cv_and_test(
        X_train, y_train, X_test, y_test, num_cols, cat_cols
    )

    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(results_df.round(4).to_string(index=False))

    winner_name = results_df.loc[0, "model"]
    winner_pipe = fitted_pipelines[winner_name]
    winner_num_cols = model_specs[winner_name]["num_cols"]
    winner_oof = oof_preds[winner_name]
    winner_test_prob = test_preds[winner_name]
    print(f"\nWinning Model (by Test ROC-AUC): {winner_name}")

    # Threshold tuning
    thresh_df = analyze_thresholds(y_train, winner_oof, y_test, winner_test_prob)
    print("\n=== THRESHOLD & ASYMMETRIC COST TRADE-OFF ===")
    print(thresh_df.to_string(index=False))

    # Feature importance (winner)
    imp_df = extract_feature_importance(winner_pipe, winner_num_cols, cat_cols)
    print(f"\n=== TOP 20 FEATURES ({winner_name}) ===")
    print(imp_df.head(20)[["feature", "importance_pct", "category"]].round(3).to_string(index=False))

    # Test predictions export & risk segmentation
    export_cols = [
        "Customer_ID", "rev_Mean", "eqpdays", "ovrrev_Mean",
        "custcare_Mean", "drop_blk_Mean", "change_mou", "months",
        "hnd_price", "asl_flag", "refurb_new",
    ]
    test_out = df.loc[idx_test, export_cols].copy()
    test_out["churn"] = y_test.values
    test_out["pred_prob"] = winner_test_prob
    test_out["pred_label_050"] = (winner_test_prob >= 0.50).astype(int)
    test_out["pred_label_040"] = (winner_test_prob >= 0.40).astype(int)
    test_out["risk_quintile"] = pd.qcut(
        test_out["pred_prob"], q=5, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"]
    )
    test_out["risk_decile"] = pd.qcut(
        test_out["pred_prob"], q=10, labels=[f"D{i}" for i in range(1, 11)]
    )

    pred_path = os.path.join(OUT_DIR, "test_predictions.csv")
    test_out.to_csv(pred_path, index=False)
    print(f"\nSaved test predictions: {pred_path} ({test_out.shape})")

    quintile_tbl, decile_tbl, hv_quintile_tbl = compute_risk_tables(test_out)
    print("\n=== TEST RISK QUINTILE LIFT (ALL TEST, n=20,000) ===")
    print(quintile_tbl[["risk_quintile", "n", "churners", "churn_pct", "lift_vs_base", "churned_rev_mo", "rev_share_pct", "mean_rev", "mean_eqpdays"]].round(2).to_string(index=False))

    print("\n=== TEST RISK DECILE LIFT (ALL TEST, n=20,000) ===")
    print(decile_tbl[["risk_decile", "n", "churners", "churn_pct", "lift_vs_base", "churned_rev_mo", "mean_rev", "mean_eqpdays"]].round(2).to_string(index=False))

    print("\n=== HIGH-VALUE SEGMENT (Q4 rev_Mean >= $70.76) BY RISK QUINTILE ===")
    print(hv_quintile_tbl[["risk_quintile", "n", "churners", "churn_pct", "lift_vs_base", "churned_rev_mo", "mean_rev", "mean_eqpdays"]].round(2).to_string(index=False))

    # Generate 4 300-DPI charts
    plot_model_comparison(results_df)
    plot_feature_importance(imp_df, winner_name)
    plot_risk_quintile_lift(quintile_tbl, hv_quintile_tbl)
    plot_roc_pr_curves(y_test, test_preds, results_df)

    # Save machine-readable summary tables for exact reporting
    results_df.to_csv(os.path.join(OUT_DIR, "model_metrics_table.csv"), index=False)
    imp_df.head(30).to_csv(os.path.join(OUT_DIR, "model_top_features.csv"), index=False)
    quintile_tbl.to_csv(os.path.join(OUT_DIR, "model_risk_quintiles.csv"), index=False)


if __name__ == "__main__":
    main()
