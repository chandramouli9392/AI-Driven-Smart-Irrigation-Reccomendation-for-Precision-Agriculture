"""
evaluate_models.py
==================
Generates all visualisations and evaluation artefacts.
Run AFTER train_classification.py and train_regression.py.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              classification_report, r2_score,
                              mean_absolute_error)

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_and_preprocess
from feature_engineering import add_engineered_features

RANDOM_STATE = 42
CSV_PATH = "AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ============================================================
# 1. EDA VISUALISATIONS
# ============================================================

def plot_eda(df):
    print("[VIZ] Generating EDA plots ...")

    # 1a. Target distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc = df["Irrigation_Required"].value_counts()
    axes[0].bar(vc.index, vc.values, color=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Irrigation_Required Class Distribution")
    axes[0].set_ylabel("Count")
    for i, (v, c) in enumerate(zip(vc.index, vc.values)):
        axes[0].text(i, c + 80, f"{c:,}\n({c/len(df)*100:.1f}%)", ha="center", fontsize=9)
    axes[1].hist(df["Water_Required_m3"], bins=50, color="#3498db", edgecolor="white")
    axes[1].set_title("Water_Required_m3 Distribution")
    axes[1].set_xlabel("Water Required (m3)")
    axes[1].set_ylabel("Count")
    med = df["Water_Required_m3"].median()
    axes[1].axvline(med, color="red", linestyle="--", label=f"Median = {med:,.0f}")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_target_distributions.png", dpi=150)
    plt.close()

    # 1b. Spectral indices across Crop_Stage
    indices = ["NDVI", "NDWI", "MNDWI", "NDMI", "EVI"]
    fig, axes = plt.subplots(1, len(indices), figsize=(20, 4), sharey=False)
    stage_order = ["Initial", "Development", "Mid", "Late"]
    for ax, idx in zip(axes, indices):
        sns.boxplot(data=df, x="Crop_Stage", y=idx, order=stage_order,
                    palette="Set2", ax=ax)
        ax.set_title(idx)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    plt.suptitle("Spectral Indices by Crop Stage", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_spectral_indices_by_stage.png", dpi=150)
    plt.close()

    # 1c. Correlation heatmap
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    excl = ["Grid_ID", "Valid_Pixel_Count"]
    num_cols = [c for c in num_cols if c not in excl]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_correlation_heatmap.png", dpi=150)
    plt.close()

    # 1d. ET0 and ETc over time
    df_date = df.copy()
    df_date["Date_parsed"] = pd.to_datetime(df_date["Date"], dayfirst=True)
    daily = df_date.groupby("Date_parsed")[["ET0_mm", "ETc_mm",
                                            "Effective_Rainfall_mm"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily["Date_parsed"], daily["ET0_mm"], label="ET0 (mm)", color="#e67e22")
    ax.plot(daily["Date_parsed"], daily["ETc_mm"], label="ETc (mm)", color="#2980b9")
    ax.fill_between(daily["Date_parsed"], 0, daily["Effective_Rainfall_mm"],
                    alpha=0.4, color="#27ae60", label="Eff. Rainfall (mm)")
    ax.set_title("Daily Mean ET0, ETc & Effective Rainfall Over Observation Period")
    ax.set_ylabel("mm")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_et_rainfall_timeseries.png", dpi=150)
    plt.close()

    # 1e. NDVI distribution by Irrigation_Required
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, grp in df.groupby("Irrigation_Required"):
        ax.hist(grp["NDVI"], bins=40, alpha=0.6, label=label)
    ax.set_title("NDVI Distribution by Irrigation Requirement")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_ndvi_by_irrigation.png", dpi=150)
    plt.close()

    # 1f. Scatter ETc vs Water_Required_m3
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(df["ETc_mm"], df["Water_Required_m3"],
                    c=df["Effective_Rainfall_mm"], cmap="viridis", alpha=0.3, s=5)
    plt.colorbar(sc, ax=ax, label="Eff. Rainfall (mm)")
    ax.set_xlabel("ETc (mm)")
    ax.set_ylabel("Water Required (m3)")
    ax.set_title("ETc vs Water Required (coloured by Rainfall)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_etc_vs_water_scatter.png", dpi=150)
    plt.close()

    print(f"[VIZ] EDA plots saved to {FIG_DIR}/")


# ============================================================
# 2. CLASSIFICATION EVALUATION PLOTS
# ============================================================

def plot_classification(cls_bundle):
    print("[VIZ] Generating classification evaluation plots ...")
    model    = cls_bundle["best_model"]
    X_test   = cls_bundle["X_test"]
    y_test   = cls_bundle["y_cls_test"]
    feat_imp = cls_bundle["feature_importance"]

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2a. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NO", "YES"],
                yticklabels=["NO", "YES"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {cls_bundle['best_name']} (Test Set)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/07_classification_confusion_matrix.png", dpi=150)
    plt.close()

    # 2b. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2c3e50", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve -- {cls_bundle['best_name']} (Test Set)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/08_classification_roc_curve.png", dpi=150)
    plt.close()

    # 2c. Feature importance
    top20 = feat_imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20["Feature"][::-1].tolist(), top20["Importance"][::-1].tolist(),
            color=plt.cm.viridis(np.linspace(0.2, 0.9, len(top20))))
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 20 Feature Importances -- {cls_bundle['best_name']} Classifier")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/09_classification_feature_importance.png", dpi=150)
    plt.close()

    # 2d. Probability calibration histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_proba[y_test == 0], bins=40, alpha=0.7, color="#e74c3c", label="Actual: NO")
    ax.hist(y_proba[y_test == 1], bins=40, alpha=0.7, color="#2ecc71", label="Actual: YES")
    ax.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")
    ax.set_xlabel("Predicted Probability of YES")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Probability Distribution (Test Set)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/10_classification_probability_hist.png", dpi=150)
    plt.close()

    print("[VIZ] Classification plots saved.")


# ============================================================
# 3. REGRESSION EVALUATION PLOTS
# ============================================================

def plot_regression(reg_bundle):
    print("[VIZ] Generating regression evaluation plots ...")
    model    = reg_bundle["best_model_B"]
    X_test   = reg_bundle["X_test_B"]
    y_test   = reg_bundle["y_reg_test"]
    feat_imp = reg_bundle["feature_importance"]

    y_pred    = model.predict(X_test)
    residuals = y_test.values - y_pred

    # 3a. Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=8, color="#2980b9")
    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ax.set_xlabel("Actual Water Required (m3)")
    ax.set_ylabel("Predicted Water Required (m3)")
    ax.set_title(
        f"Actual vs Predicted -- {reg_bundle['best_name_B']} (Setting B, Test)\n"
        f"R2={r2:.4f}  MAE={mae:,.0f} m3"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/11_regression_actual_vs_predicted.png", dpi=150)
    plt.close()

    # 3b. Residual analysis
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=6, color="#8e44ad")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Water Required (m3)")
    axes[0].set_ylabel("Residual (m3)")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(residuals, bins=50, color="#16a085", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (m3)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    plt.suptitle(
        f"{reg_bundle['best_name_B']} Regressor -- Residual Analysis (Test Set)",
        fontsize=12
    )
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/12_regression_residuals.png", dpi=150)
    plt.close()

    # 3c. Feature importance
    top20 = feat_imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20["Feature"][::-1].tolist(), top20["Importance"][::-1].tolist(),
            color=plt.cm.plasma(np.linspace(0.2, 0.9, len(top20))))
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top 20 Feature Importances -- {reg_bundle['best_name_B']} Regressor (Setting B)")
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/13_regression_feature_importance.png", dpi=150)
    plt.close()

    # 3d. Model comparison bar chart
    comp_df = reg_bundle["comparison"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    settings = comp_df["Setting"].unique()
    colors = ["#3498db", "#e67e22"]
    for i, metric in enumerate(["Val_R2", "Val_MAE_m3", "Val_MAPE%"]):
        x = np.arange(len(comp_df["Model"].unique()))
        width = 0.35
        for j, (setting, color) in enumerate(zip(settings, colors)):
            grp = comp_df[comp_df["Setting"] == setting].reset_index(drop=True)
            axes[i].bar(x + j * width, grp[metric], width, label=setting, color=color, alpha=0.8)
        axes[i].set_xticks(x + width / 2)
        axes[i].set_xticklabels(comp_df["Model"].unique(), rotation=25, ha="right")
        axes[i].set_title(metric)
        axes[i].legend(fontsize=8)
    plt.suptitle("Regression Model Comparison: Setting A (Full) vs Setting B (Sat+Crop)", fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{FIG_DIR}/14_regression_model_comparison.png", dpi=150)
    plt.close()

    print("[VIZ] Regression plots saved.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    plot_eda(df)

    with open("models/best_irrigation_classifier.pkl", "rb") as f:
        cls_pkl = pickle.load(f)
    with open("models/best_water_requirement_regressor.pkl", "rb") as f:
        reg_pkl = pickle.load(f)

    data = load_and_preprocess(CSV_PATH, RANDOM_STATE)
    X_tr = add_engineered_features(data["X_train"])
    X_vl = add_engineered_features(data["X_val"])
    X_te = add_engineered_features(data["X_test"])

    excl_B = ["Effective_Rainfall_mm", "ETc_mm", "ET0_Rain_Ratio"]
    B_cols = [c for c in X_te.columns if c not in excl_B]

    fi_cls = pd.read_csv("reports/classification_feature_importance.csv")
    fi_reg = pd.read_csv("reports/regression_feature_importance.csv")

    cls_bundle = {
        "best_model":        cls_pkl["model"],
        "best_name":         cls_pkl["tuned_model_name"],
        "X_test":            X_te,
        "y_cls_test":        data["y_cls_test"],
        "feature_importance": fi_cls,
    }
    reg_bundle = {
        "best_model_B":      reg_pkl["model_B"],
        "best_name_B":       reg_pkl["best_name_B"],
        "X_test_B":          X_te[B_cols],
        "y_reg_test":        data["y_reg_test"],
        "feature_importance": fi_reg,
        "comparison":        pd.read_csv("reports/regression_comparison.csv"),
    }

    plot_classification(cls_bundle)
    plot_regression(reg_bundle)

    print("\n[DONE] All visualisations saved to figures/")
