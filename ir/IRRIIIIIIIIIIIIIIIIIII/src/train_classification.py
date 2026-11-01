"""
train_classification.py
=======================
Task A: Machine Learning Irrigation Requirement Classification
Target: Irrigation_Required (YES / NO)

SCIENTIFIC & RESEARCH INTEGRITY NOTE:
  The target 'Irrigation_Required' in this dataset is deterministically derived from
  the FAO-56 water balance equation:
      ETc_mm = Kc * ET0_mm
      Effective_Rainfall_mm = 0.80 * PRECTOTCORR
      Irrigation_Requirement_mm = max(0, ETc_mm - Effective_Rainfall_mm)
      Irrigation_Required = 'YES' if Irrigation_Requirement_mm > 0 else 'NO'

  To maintain strict scientific integrity, two experimental paradigms are evaluated:
  - Setting A (Full Features / Formula Reproduction Upper Bound):
    Includes ETc_mm and Effective_Rainfall_mm. Demonstrates the upper bound of how
    accurately tree ensembles can learn the non-linear max(0, x-y) threshold rule.
  - Setting B (Independent Remote Sensing Predictive Setting):
    Strictly excludes ETc_mm and Effective_Rainfall_mm. Evaluates whether Sentinel-2
    multispectral bands, vegetation indices, ET0, and crop phenology can independently
    predict the necessity of irrigation without direct rainfall knowledge.

CLASS IMBALANCE:
  - YES: 13,487 (94.99%)
  - NO : 712    (5.01%)
  Techniques applied: Class weighting, threshold tuning on cross-validated OOF predictions,
  balanced accuracy, and explicit comparison against a Majority Class Baseline.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    RandomizedSearchCV,
    cross_validate,
    cross_val_predict,
)
from xgboost import XGBClassifier

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import load_and_preprocess, CAT_FEATURES
from feature_engineering import add_engineered_features

RANDOM_STATE = 42


def evaluate_cls_comprehensive(model, X, y, threshold=0.5, label=""):
    """
    Computes a full suite of classification metrics with class-specific precision/recall,
    balanced accuracy, PR-AUC, ROC-AUC, and custom decision threshold.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
    else:
        preds = model.predict(X)
        proba = preds.astype(float)

    acc = accuracy_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, preds, average="weighted", zero_division=0)

    # Per-class metrics: class 0 (NO - minority), class 1 (YES - majority)
    prec_no = precision_score(y, preds, pos_label=0, zero_division=0)
    rec_no = recall_score(y, preds, pos_label=0, zero_division=0)
    f1_no = f1_score(y, preds, pos_label=0, zero_division=0)

    prec_yes = precision_score(y, preds, pos_label=1, zero_division=0)
    rec_yes = recall_score(y, preds, pos_label=1, zero_division=0)
    f1_yes = f1_score(y, preds, pos_label=1, zero_division=0)

    try:
        roc_auc = roc_auc_score(y, proba)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y, proba)
    except Exception:
        pr_auc = float("nan")

    cm = confusion_matrix(y, preds, labels=[0, 1])

    if label:
        print(f"  {label:24s} -> Acc={acc:.4f}  BalAcc={bal_acc:.4f}  MacF1={macro_f1:.4f}  "
              f"NO_Rec={rec_no:.4f}  NO_F1={f1_no:.4f}  ROC_AUC={roc_auc:.4f}  PR_AUC={pr_auc:.4f}")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_no": prec_no,
        "recall_no": rec_no,
        "f1_no": f1_no,
        "precision_yes": prec_yes,
        "recall_yes": rec_yes,
        "f1_yes": f1_yes,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "proba": proba,
        "preds": preds,
    }


def find_optimal_threshold_oof(model, X_trainval, y_trainval, cv=None):
    """
    Finds optimal decision threshold using Out-Of-Fold (OOF) cross-validation predictions.
    Maximizes balanced accuracy to guarantee robust sensitivity to the 5% minority class.
    When a plateau of maximum balanced accuracy exists, selects the 75th percentile of the
    optimal plateau to counteract the 95:5 class imbalance and provide margin against distribution shift.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    try:
        oof_proba = cross_val_predict(model, X_trainval, y_trainval, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thresholds = np.linspace(0.10, 0.95, 86)
        scores = np.array([balanced_accuracy_score(y_trainval, (oof_proba >= th).astype(int)) for th in thresholds])
        max_score = float(np.max(scores))
        optimal_indices = np.where(scores >= max_score - 1e-4)[0]
        optimal_thresholds = thresholds[optimal_indices]
        best_th = float(np.percentile(optimal_thresholds, 75))
        return best_th, max_score
    except Exception:
        return 0.50, 0.50


def train_classifiers(csv_path="AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"):
    if not os.path.exists(csv_path):
        csv_path = os.path.join("IRRIIIIIIIIIIIIIIIIIII", csv_path)

    print("=" * 75)
    print("TASK A: IRRIGATION REQUIREMENT CLASSIFICATION BENCHMARK")
    print("=" * 75)

    data = load_and_preprocess(csv_path, RANDOM_STATE)

    # Engineered feature sets
    X_tr_full = add_engineered_features(data["X_train"], include_rainfall_interactions=True)
    X_vl_full = add_engineered_features(data["X_val"], include_rainfall_interactions=True)
    X_te_full = add_engineered_features(data["X_test"], include_rainfall_interactions=True)

    y_tr, y_vl, y_te = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]

    # Setting B: Independent remote sensing features (no ETc, no Effective_Rainfall_mm, no ET0_Rain_Ratio)
    excl_B = ["Effective_Rainfall_mm", "ETc_mm", "ET0_Rain_Ratio"]
    B_cols = [c for c in X_tr_full.columns if c not in excl_B]

    X_tr_B = X_tr_full[B_cols].copy()
    X_vl_B = X_vl_full[B_cols].copy()
    X_te_B = X_te_full[B_cols].copy()

    X_tv_B = pd.concat([X_tr_B, X_vl_B], axis=0)
    X_tv_full = pd.concat([X_tr_full, X_vl_full], axis=0)
    y_tv = pd.concat([y_tr, y_vl], axis=0)

    n_yes = int((y_tr == 1).sum())
    n_no = int((y_tr == 0).sum())
    cw_ratio = n_yes / max(n_no, 1)

    print(f"\n[CLASS IMBALANCE] Train counts   : NO={n_no} ({n_no/len(y_tr)*100:.2f}%), YES={n_yes} ({n_yes/len(y_tr)*100:.2f}%)")
    print(f"[CLASS IMBALANCE] Train+Val counts: NO={(y_tv==0).sum()}, YES={(y_tv==1).sum()}")
    print(f"[CLASS IMBALANCE] Test counts    : NO={(y_te==0).sum()}, YES={(y_te==1).sum()}")

    # 1. Majority Class Baseline
    class MajorityBaseline:
        def fit(self, X, y): pass
        def predict(self, X): return np.ones(len(X), dtype=int)
        def predict_proba(self, X):
            p1 = np.ones(len(X))
            p0 = np.zeros(len(X))
            return np.column_stack([p0, p1])

    base_dummy = MajorityBaseline()
    base_te = evaluate_cls_comprehensive(base_dummy, X_te_full, y_te, label="Majority Class Baseline")

    # 2. Classifier Definitions
    def build_models():
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=16, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                scale_pos_weight=1.0 / cw_ratio,
                eval_metric="logloss", random_state=RANDOM_STATE,
                verbosity=0, n_jobs=-1
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=300, max_depth=16, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=4, random_state=RANDOM_STATE
            ),
        }

    # =========================================================================
    # SETTING B: INDEPENDENT REMOTE SENSING PREDICTIVE CHALLENGE
    # =========================================================================
    print(f"\n" + "-" * 75)
    print(f"EXPERIMENT 1: SETTING B -- INDEPENDENT SATELLITE + WEATHER ({len(B_cols)} features)")
    print(f"Excludes direct mathematical precursors: {excl_B}")
    print("-" * 75)

    models_B = build_models()
    results_B = {}
    thresholds_B = {}

    for name, mdl in models_B.items():
        print(f"\n  -- Training {name} (Setting B) --")
        # Calibrate optimal decision threshold using OOF predictions on Train+Val
        opt_th, oof_bal_acc = find_optimal_threshold_oof(mdl, X_tv_B, y_tv)
        thresholds_B[name] = opt_th
        print(f"     [OOF Threshold Tuning] Optimal BalAcc={oof_bal_acc:.4f} at threshold={opt_th:.2f}")

        # Train on training split
        mdl.fit(X_tr_B, y_tr)
        tr_eval = evaluate_cls_comprehensive(mdl, X_tr_B, y_tr, threshold=opt_th, label=f"Train ({name})")
        vl_eval = evaluate_cls_comprehensive(mdl, X_vl_B, y_vl, threshold=opt_th, label=f"Val   ({name})")

        # Refit on Train+Val for final independent Test set evaluation
        mdl_tv = build_models()[name]
        mdl_tv.fit(X_tv_B, y_tv)
        te_eval = evaluate_cls_comprehensive(mdl_tv, X_te_B, y_te, threshold=opt_th, label=f"Test  ({name})")

        results_B[name] = {
            "model": mdl_tv,
            "threshold": opt_th,
            "oof_bal_acc": oof_bal_acc,
            "train": tr_eval,
            "val": vl_eval,
            "test": te_eval,
        }

    # Best model selected strictly on OOF and Validation performance (no test set leakage)
    best_name_B = max(results_B, key=lambda n: (results_B[n]["oof_bal_acc"], results_B[n]["val"]["macro_f1"], 1 if n == "ExtraTrees" else 0))
    print(f"\n>>> Best Setting B model selected: {best_name_B}")

    # =========================================================================
    # SETTING A: FULL FEATURE UPPER BOUND (FORMULA REPRODUCTION)
    # =========================================================================
    print(f"\n" + "-" * 75)
    print(f"EXPERIMENT 2: SETTING A -- FULL FEATURES (Upper Bound Reference, {X_tr_full.shape[1]} features)")
    print(f"Includes ETc_mm and Effective_Rainfall_mm.")
    print("-" * 75)

    models_A = build_models()
    results_A = {}

    for name, mdl in models_A.items():
        print(f"\n  -- Training {name} (Setting A) --")
        mdl.fit(X_tv_full, y_tv)
        tr_eval = evaluate_cls_comprehensive(mdl, X_tr_full, y_tr, threshold=0.50, label=f"Train ({name})")
        vl_eval = evaluate_cls_comprehensive(mdl, X_vl_full, y_vl, threshold=0.50, label=f"Val   ({name})")
        te_eval = evaluate_cls_comprehensive(mdl, X_te_full, y_te, threshold=0.50, label=f"Test  ({name})")
        results_A[name] = {"model": mdl, "threshold": 0.50, "train": tr_eval, "val": vl_eval, "test": te_eval}

    best_name_A = max(results_A, key=lambda n: results_A[n]["test"]["macro_f1"])

    # =========================================================================
    # HYPERPARAMETER OPTIMIZATION (ON BEST SETTING B MODEL)
    # =========================================================================
    print(f"\n" + "-" * 75)
    print(f"HYPERPARAMETER TUNING: RandomizedSearchCV on {best_name_B} (Setting B)")
    print("-" * 75)

    param_spaces = {
        "RandomForest": {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 16, 22, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "XGBoost": {
            "n_estimators": [200, 300, 400],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.75, 0.85, 1.0],
            "min_child_weight": [2, 5, 8],
        },
        "ExtraTrees": {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 16, 22, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "GradientBoosting": {
            "n_estimators": [150, 250, 350],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.75, 0.85],
            "min_samples_leaf": [2, 4, 8],
        },
    }

    base_estimators = {
        "RandomForest": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(scale_pos_weight=1.0/cw_ratio, eval_metric="logloss",
                                  random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rscv = RandomizedSearchCV(
        base_estimators[best_name_B],
        param_spaces[best_name_B],
        n_iter=15,
        cv=skf,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    rscv.fit(X_tv_B, y_tv)
    best_tuned_model_B = rscv.best_estimator_
    best_params_B = rscv.best_params_
    print(f"  Best Parameters: {best_params_B}")

    opt_th_tuned, _ = find_optimal_threshold_oof(best_tuned_model_B, X_tv_B, y_tv)
    print(f"  Tuned Decision Threshold: {opt_th_tuned:.2f}")

    te_tuned_eval = evaluate_cls_comprehensive(
        best_tuned_model_B, X_te_B, y_te, threshold=opt_th_tuned, label=f"Test (Tuned {best_name_B})"
    )

    # =========================================================================
    # ADVANCED VALIDATION STRATEGY COMPARISON: SPATIAL VS TEMPORAL VS CV
    # =========================================================================
    print(f"\n" + "-" * 75)
    print("VALIDATION STRATEGY COMPARISON (Setting B - Best Model):")
    print("-" * 75)

    df_full = data["df_raw"].copy()
    X_all_raw = df_full[data["feature_names"]].copy()
    X_all_raw[CAT_FEATURES] = data["cat_encoder"].transform(X_all_raw[CAT_FEATURES])
    X_all_full = add_engineered_features(X_all_raw, include_rainfall_interactions=True)
    X_all_B = X_all_full[B_cols].copy()
    y_all = (df_full["Irrigation_Required"] == "YES").astype(int)

    # 1. 5-Fold Stratified Group K-Fold (Spatial Grouping by Grid_ID)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    spatial_acc, spatial_bal_acc, spatial_f1 = [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(sgkf.split(X_all_B, y_all, groups=df_full["Grid_ID"])):
        mdl_fold = base_estimators[best_name_B].set_params(**best_params_B)
        mdl_fold.fit(X_all_B.iloc[tr_idx], y_all.iloc[tr_idx])
        preds = mdl_fold.predict(X_all_B.iloc[val_idx])
        spatial_acc.append(accuracy_score(y_all.iloc[val_idx], preds))
        spatial_bal_acc.append(balanced_accuracy_score(y_all.iloc[val_idx], preds))
        spatial_f1.append(f1_score(y_all.iloc[val_idx], preds, average="macro", zero_division=0))

    print(f"  1. Spatial Holdout (5-Fold Stratified Group by Grid_ID):")
    print(f"     Accuracy        : {np.mean(spatial_acc):.4f} +/- {np.std(spatial_acc):.4f}")
    print(f"     Balanced Accuracy: {np.mean(spatial_bal_acc):.4f} +/- {np.std(spatial_bal_acc):.4f}")
    print(f"     Macro F1        : {np.mean(spatial_f1):.4f} +/- {np.std(spatial_f1):.4f}")

    # 2. Standard 5-Fold Stratified Cross-Validation (Naive Random Split)
    cv_res = cross_validate(
        best_tuned_model_B, X_all_B, y_all, cv=skf,
        scoring=["accuracy", "balanced_accuracy", "f1_macro"], n_jobs=-1
    )
    print(f"  2. Naive Stratified CV (Random split - prone to spatial autocorrelation):")
    print(f"     Accuracy        : {cv_res['test_accuracy'].mean():.4f} +/- {cv_res['test_accuracy'].std():.4f}")
    print(f"     Balanced Accuracy: {cv_res['test_balanced_accuracy'].mean():.4f} +/- {cv_res['test_balanced_accuracy'].std():.4f}")
    print(f"     Macro F1        : {cv_res['test_f1_macro'].mean():.4f} +/- {cv_res['test_f1_macro'].std():.4f}")

    print(f"  3. Strict Chronological Holdout (Test Set Performance):")
    print(f"     Accuracy        : {te_tuned_eval['accuracy']:.4f}")
    print(f"     Balanced Accuracy: {te_tuned_eval['balanced_accuracy']:.4f}")
    print(f"     Macro F1        : {te_tuned_eval['macro_f1']:.4f}")

    # =========================================================================
    # COMPILE FINAL COMPARISON TABLE
    # =========================================================================
    comparison_rows = []

    # Majority baseline
    comparison_rows.append({
        "Setting": "Baseline",
        "Model": "Majority_Class_YES",
        "Accuracy": round(base_te["accuracy"], 4),
        "Balanced_Accuracy": round(base_te["balanced_accuracy"], 4),
        "Macro_F1": round(base_te["macro_f1"], 4),
        "Weighted_F1": round(base_te["weighted_f1"], 4),
        "NO_Precision": round(base_te["precision_no"], 4),
        "NO_Recall": round(base_te["recall_no"], 4),
        "NO_F1": round(base_te["f1_no"], 4),
        "YES_Precision": round(base_te["precision_yes"], 4),
        "YES_Recall": round(base_te["recall_yes"], 4),
        "YES_F1": round(base_te["f1_yes"], 4),
        "ROC_AUC": round(base_te["roc_auc"], 4),
        "PR_AUC": round(base_te["pr_auc"], 4),
    })

    # Setting B models
    for name, res in results_B.items():
        te = res["test"]
        comparison_rows.append({
            "Setting": "B_Independent",
            "Model": name,
            "Accuracy": round(te["accuracy"], 4),
            "Balanced_Accuracy": round(te["balanced_accuracy"], 4),
            "Macro_F1": round(te["macro_f1"], 4),
            "Weighted_F1": round(te["weighted_f1"], 4),
            "NO_Precision": round(te["precision_no"], 4),
            "NO_Recall": round(te["recall_no"], 4),
            "NO_F1": round(te["f1_no"], 4),
            "YES_Precision": round(te["precision_yes"], 4),
            "YES_Recall": round(te["recall_yes"], 4),
            "YES_F1": round(te["f1_yes"], 4),
            "ROC_AUC": round(te["roc_auc"], 4),
            "PR_AUC": round(te["pr_auc"], 4),
        })

    # Setting B Tuned
    comparison_rows.append({
        "Setting": "B_Independent",
        "Model": f"{best_name_B}_Tuned",
        "Accuracy": round(te_tuned_eval["accuracy"], 4),
        "Balanced_Accuracy": round(te_tuned_eval["balanced_accuracy"], 4),
        "Macro_F1": round(te_tuned_eval["macro_f1"], 4),
        "Weighted_F1": round(te_tuned_eval["weighted_f1"], 4),
        "NO_Precision": round(te_tuned_eval["precision_no"], 4),
        "NO_Recall": round(te_tuned_eval["recall_no"], 4),
        "NO_F1": round(te_tuned_eval["f1_no"], 4),
        "YES_Precision": round(te_tuned_eval["precision_yes"], 4),
        "YES_Recall": round(te_tuned_eval["recall_yes"], 4),
        "YES_F1": round(te_tuned_eval["f1_yes"], 4),
        "ROC_AUC": round(te_tuned_eval["roc_auc"], 4),
        "PR_AUC": round(te_tuned_eval["pr_auc"], 4),
    })

    # Setting A models
    for name, res in results_A.items():
        te = res["test"]
        comparison_rows.append({
            "Setting": "A_FormulaRef",
            "Model": name,
            "Accuracy": round(te["accuracy"], 4),
            "Balanced_Accuracy": round(te["balanced_accuracy"], 4),
            "Macro_F1": round(te["macro_f1"], 4),
            "Weighted_F1": round(te["weighted_f1"], 4),
            "NO_Precision": round(te["precision_no"], 4),
            "NO_Recall": round(te["recall_no"], 4),
            "NO_F1": round(te["f1_no"], 4),
            "YES_Precision": round(te["precision_yes"], 4),
            "YES_Recall": round(te["recall_yes"], 4),
            "YES_F1": round(te["f1_yes"], 4),
            "ROC_AUC": round(te["roc_auc"], 4),
            "PR_AUC": round(te["pr_auc"], 4),
        })

    comp_df = pd.DataFrame(comparison_rows)
    print("\n[CLASSIFICATION BENCHMARK TABLE]")
    print(comp_df.to_string(index=False))

    # Feature Importance (Tuned Best Setting B Model)
    feat_names = B_cols
    fi_vals = best_tuned_model_B.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": fi_vals}).sort_values("Importance", ascending=False)
    print("\n[TOP 15 FEATURE IMPORTANCES (Setting B)]")
    print(fi_df.head(15).to_string(index=False))

    # Save Artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    with open("models/best_irrigation_classifier.pkl", "wb") as f:
        pickle.dump({
            "model_B": best_tuned_model_B,
            "model_A": results_A[best_name_A]["model"],
            "feature_names_B": B_cols,
            "feature_names_A": X_tr_full.columns.tolist(),
            "best_params_B": best_params_B,
            "best_name_B": best_name_B,
            "best_name_A": best_name_A,
            "optimal_threshold_B": opt_th_tuned,
            "label_map": data["label_map"],
            "cat_encoder": data["cat_encoder"],
        }, f)
    print("\nSaved: models/best_irrigation_classifier.pkl")

    comp_df.to_csv("reports/classification_comparison.csv", index=False)
    fi_df.to_csv("reports/classification_feature_importance.csv", index=False)

    # Detailed text report
    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("AI-DRIVEN SMART IRRIGATION RECOMMENDATION SYSTEM\n")
        f.write("TASK A: IRRIGATION REQUIREMENT CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("RESEARCH INTEGRITY & METHODOLOGICAL AUDIT:\n")
        f.write("- Dataset records: 14,199 across 514 grid cells in Godavari Delta, AP.\n")
        f.write("- Target derivation: Deterministic FAO-56 hydrological water-balance formula.\n")
        f.write("  NO ground-truth soil moisture or flowmeter sensor readings are present.\n")
        f.write("- Class Imbalance: YES=94.99% (13,487), NO=5.01% (712).\n")
        f.write("  Crucial Finding: NO cases exist ONLY on 2 specific dates (2024-10-11 & 2025-10-11)\n")
        f.write("  where precipitation exceeded crop evapotranspiration.\n\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"BEST INDEPENDENT MODEL (Setting B): {best_name_B} (Tuned)\n")
        f.write(f"Optimal Hyperparameters: {best_params_B}\n")
        f.write(f"Tuned Decision Threshold: {opt_th_tuned:.2f}\n\n")
        f.write("MODEL COMPARISON SUMMARY:\n")
        f.write(comp_df.to_string(index=False) + "\n\n")
        f.write("TEST SET CLASSIFICATION METRICS (Tuned Setting B):\n")
        f.write(f"  Accuracy           : {te_tuned_eval['accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy  : {te_tuned_eval['balanced_accuracy']:.4f}\n")
        f.write(f"  Macro F1           : {te_tuned_eval['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1        : {te_tuned_eval['weighted_f1']:.4f}\n")
        f.write(f"  Minority NO Recall : {te_tuned_eval['recall_no']:.4f}\n")
        f.write(f"  Minority NO F1     : {te_tuned_eval['f1_no']:.4f}\n")
        f.write(f"  PR-AUC             : {te_tuned_eval['pr_auc']:.4f}\n")
        f.write(f"  ROC-AUC            : {te_tuned_eval['roc_auc']:.4f}\n\n")
        f.write("CONFUSION MATRIX (Test Set):\n")
        f.write(str(te_tuned_eval['confusion_matrix']) + "\n\n")
        f.write("TOP 15 FEATURE IMPORTANCES (Setting B):\n")
        f.write(fi_df.head(15).to_string(index=False) + "\n")

    return {
        "best_model_B": best_tuned_model_B,
        "best_model_A": results_A[best_name_A]["model"],
        "best_name_B": best_name_B,
        "optimal_threshold_B": opt_th_tuned,
        "comparison": comp_df,
        "feature_importance": fi_df,
        "X_test_B": X_te_B,
        "X_test_full": X_te_full,
        "y_cls_test": y_te,
        "results_B": results_B,
        "results_A": results_A,
    }


if __name__ == "__main__":
    train_classifiers()
