"""
train_classification.py
=======================
Task A: Irrigation Requirement Classification
Target: Irrigation_Required (YES / NO)

SCIENTIFIC NOTE -- TARGET DERIVATION:
  The Irrigation_Required column is NOT an independently verified field
  observation. It is deterministically derived from the FAO-56 water
  balance formula:
      Irrigation_Requirement_mm = max(0, ETc_mm - Effective_Rainfall_mm)
      Irrigation_Required = 'YES' if Irrigation_Requirement_mm > 0 else 'NO'
  This model learns to reproduce the FAO-56 decision rule from satellite
  and weather inputs. It does NOT validate against measured field events.

  Class imbalance: YES ~95.0%, NO ~5.0%
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_and_preprocess
from feature_engineering import add_engineered_features

RANDOM_STATE = 42
CSV_PATH = "AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"


def evaluate_cls(model, X, y, label=""):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    acc   = accuracy_score(y, preds)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    f1    = f1_score(y, preds, zero_division=0)
    auc   = roc_auc_score(y, proba) if proba is not None else float("nan")
    if label:
        print(f"  {label:20s} -> Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc, "preds": preds}


def train_classifiers():
    print("=" * 70)
    print("TASK A: IRRIGATION REQUIREMENT CLASSIFICATION")
    print("=" * 70)

    data = load_and_preprocess(CSV_PATH, RANDOM_STATE)
    X_tr = add_engineered_features(data["X_train"])
    X_vl = add_engineered_features(data["X_val"])
    X_te = add_engineered_features(data["X_test"])

    y_tr, y_vl, y_te = data["y_cls_train"], data["y_cls_val"], data["y_cls_test"]

    n_yes = int((y_tr == 1).sum())
    n_no  = int((y_tr == 0).sum())
    cw_ratio = n_yes / max(n_no, 1)
    print(f"\n[CLS] Train class counts: NO={n_no}, YES={n_yes}  (ratio {cw_ratio:.1f}:1)")
    print(f"[CLS] Val class counts  : NO={(y_vl==0).sum()}, YES={(y_vl==1).sum()}")
    print(f"[CLS] Test class counts : NO={(y_te==0).sum()}, YES={(y_te==1).sum()}")

    # -- Model definitions ---------------------------------------------------
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=cw_ratio,
            eval_metric="logloss", random_state=RANDOM_STATE,
            verbosity=0, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
    }

    # -- Initial training & validation comparison ----------------------------
    results = {}
    print("\n[CLS] Training all classifiers ...")
    for name, mdl in models.items():
        print(f"\n  -- {name} --")
        mdl.fit(X_tr, y_tr)
        tr_res = evaluate_cls(mdl, X_tr, y_tr, "Train")
        vl_res = evaluate_cls(mdl, X_vl, y_vl, "Val  ")
        results[name] = {"model": mdl, "train": tr_res, "val": vl_res}

    # -- Hyperparameter tuning on best val-F1 model -------------------------
    best_name = max(results, key=lambda n: results[n]["val"]["f1"])
    print(f"\n[CLS] Best by Val F1: {best_name} -- running RandomizedSearchCV ...")

    param_spaces = {
        "RandomForest": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "XGBoost": {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 5, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [3, 5, 10],
        },
        "ExtraTrees": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "min_samples_leaf": [2, 4, 8],
        },
    }

    base_map = {
        "RandomForest": RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(scale_pos_weight=cw_ratio, eval_metric="logloss",
                                  random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_tv = pd.concat([X_tr, X_vl], axis=0)
    y_tv = pd.concat([y_tr, y_vl], axis=0)

    rscv = RandomizedSearchCV(
        base_map[best_name], param_spaces[best_name],
        n_iter=30, cv=cv, scoring="f1", n_jobs=-1,
        random_state=RANDOM_STATE, verbose=0
    )
    rscv.fit(X_tv, y_tv)

    best_params = rscv.best_params_
    print(f"[CLS] Best params: {best_params}")
    best_model = rscv.best_estimator_

    # -- Final evaluation on held-out test set ------------------------------
    print("\n[CLS] FINAL model evaluation on independent test set:")
    te_res = evaluate_cls(best_model, X_te, y_te, "Test ")
    cv_f1  = cross_val_score(best_model, X_tv, y_tv, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  CV F1 (5-fold): {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")

    print("\n[CLS] Full classification report (test set):")
    print(classification_report(y_te, best_model.predict(X_te),
                                 target_names=["NO (0)", "YES (1)"]))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_te, best_model.predict(X_te)))

    # -- Compile comparison table -------------------------------------------
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":     name,
            "Train_Acc": round(res["train"]["accuracy"], 4),
            "Val_Acc":   round(res["val"]["accuracy"], 4),
            "Val_F1":    round(res["val"]["f1"], 4),
            "Val_AUC":   round(res["val"]["roc_auc"], 4),
            "Val_Recall":round(res["val"]["recall"], 4),
        })
    vl_tuned = evaluate_cls(best_model, X_vl, y_vl)
    rows.append({
        "Model":     f"{best_name}_Tuned",
        "Train_Acc": round(evaluate_cls(best_model, X_tr, y_tr)["accuracy"], 4),
        "Val_Acc":   round(vl_tuned["accuracy"], 4),
        "Val_F1":    round(vl_tuned["f1"], 4),
        "Val_AUC":   round(vl_tuned["roc_auc"], 4),
        "Val_Recall":round(vl_tuned["recall"], 4),
    })
    comp_df = pd.DataFrame(rows)
    print("\n[CLS] Model Comparison Table:")
    print(comp_df.to_string(index=False))

    # -- Feature importance -------------------------------------------------
    feat_names = X_tr.columns.tolist()
    fi = best_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": fi}).sort_values("Importance", ascending=False)
    print("\n[CLS] Top 15 Feature Importances:")
    print(fi_df.head(15).to_string(index=False))

    # -- Save artefacts -----------------------------------------------------
    os.makedirs("models", exist_ok=True)
    with open("models/best_irrigation_classifier.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "feature_names": feat_names,
            "label_map": data["label_map"],
            "cat_encoder": data["cat_encoder"],
            "best_params": best_params,
            "tuned_model_name": best_name,
        }, f)
    print("\n[CLS] Saved: models/best_irrigation_classifier.pkl")

    os.makedirs("reports", exist_ok=True)
    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("AI-DRIVEN SMART IRRIGATION RECOMMENDATION SYSTEM\n")
        f.write("TASK A: IRRIGATION REQUIREMENT CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")
        f.write("SCIENTIFIC DISCLAIMER:\n")
        f.write("The target label 'Irrigation_Required' is deterministically\n")
        f.write("derived from the FAO-56 water balance:\n")
        f.write("  Irrigation_Required = 'YES' if (ETc_mm - Eff_Rain_mm) > 0\n")
        f.write("This model reproduces that decision rule from satellite inputs.\n")
        f.write("It does NOT validate against observed field irrigation events.\n\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Model: {best_name} (after tuning)\n")
        f.write(f"Best Hyperparameters: {best_params}\n\n")
        f.write("MODEL COMPARISON TABLE:\n")
        f.write(comp_df.to_string(index=False) + "\n\n")
        f.write(f"CV F1 (5-fold, train+val combined): {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}\n\n")
        f.write("TEST SET PERFORMANCE:\n")
        f.write(classification_report(y_te, best_model.predict(X_te),
                                      target_names=["NO (0)", "YES (1)"]) + "\n")
        f.write("Confusion Matrix (Test):\n")
        f.write(str(confusion_matrix(y_te, best_model.predict(X_te))) + "\n\n")
        f.write("TOP 15 FEATURE IMPORTANCES:\n")
        f.write(fi_df.head(15).to_string(index=False) + "\n")

    comp_df.to_csv("reports/classification_comparison.csv", index=False)
    fi_df.to_csv("reports/classification_feature_importance.csv", index=False)

    return {
        "best_model": best_model,
        "best_name": best_name,
        "comparison": comp_df,
        "feature_importance": fi_df,
        "data": data,
        "X_train": X_tr, "X_val": X_vl, "X_test": X_te,
        "y_cls_train": y_tr, "y_cls_val": y_vl, "y_cls_test": y_te,
    }


if __name__ == "__main__":
    train_classifiers()
