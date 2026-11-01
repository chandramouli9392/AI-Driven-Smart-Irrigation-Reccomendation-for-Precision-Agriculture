"""
train_regression.py
===================
Task B: Water Requirement Prediction (Regression)
Target: Water_Required_m3

SCIENTIFIC NOTE -- TARGET DERIVATION:
  Water_Required_m3 is mathematically derived as:
      Irrigation_Requirement_mm = max(0, ETc_mm - Effective_Rainfall_mm)
      Water_Required_m3 = Irrigation_Requirement_mm * Field_Area_ha * 10
  Field_Area_ha is CONSTANT (605.672381 ha) for all records.

  Two regression settings are evaluated:
  (A) Full feature set (includes ETc_mm + Effective_Rainfall_mm)
      -- near-perfect due to formula structure; serves as upper-bound reference.
  (B) Satellite + ET0 + Kc + Crop metadata only
      (excludes Effective_Rainfall_mm and ETc_mm)
      -- operationally realistic challenge setting.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_and_preprocess
from feature_engineering import add_engineered_features

RANDOM_STATE = 42
CSV_PATH = "AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"


def mape(y_true, y_pred, eps=1.0):
    """MAPE -- skip rows where |true| < eps to avoid division by zero."""
    mask = np.abs(y_true) >= eps
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_reg(model, X, y, label=""):
    preds = model.predict(X)
    r2    = r2_score(y, preds)
    mae   = mean_absolute_error(y, preds)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    mp    = mape(y.values, preds)
    if label:
        print(f"  {label:20s} -> R2={r2:.4f}  MAE={mae:,.0f} m3  RMSE={rmse:,.0f} m3  MAPE={mp:.2f}%")
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mp, "preds": preds}


def train_regressors():
    print("=" * 70)
    print("TASK B: WATER REQUIREMENT REGRESSION")
    print("=" * 70)

    data = load_and_preprocess(CSV_PATH, RANDOM_STATE)
    X_tr_raw = add_engineered_features(data["X_train"])
    X_vl_raw = add_engineered_features(data["X_val"])
    X_te_raw = add_engineered_features(data["X_test"])

    y_tr, y_vl, y_te = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]

    # -- SETTING A: full features (ETc + Eff_Rain included) -----------------
    print("\n-- SETTING A: Full features (ETc_mm + Effective_Rainfall_mm included) --")
    print("   (Near-perfect because target ~ (ETc-Rain)*Area*10)")

    models_A = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
    }

    results_A = {}
    for name, mdl in models_A.items():
        print(f"\n  -- {name} --")
        mdl.fit(X_tr_raw, y_tr)
        tr_res = evaluate_reg(mdl, X_tr_raw, y_tr, "Train")
        vl_res = evaluate_reg(mdl, X_vl_raw, y_vl, "Val  ")
        results_A[name] = {"model": mdl, "train": tr_res, "val": vl_res}

    # -- SETTING B: satellite + ET0 + Kc + crop meta (no Eff_Rain / no ETc) -
    excl_B = ["Effective_Rainfall_mm", "ETc_mm", "ET0_Rain_Ratio"]
    B_cols_tr = [c for c in X_tr_raw.columns if c not in excl_B]
    X_tr_B = X_tr_raw[B_cols_tr].copy()
    X_vl_B = X_vl_raw[B_cols_tr].copy()
    X_te_B = X_te_raw[B_cols_tr].copy()

    print(f"\n-- SETTING B: Satellite + crop/weather only ({len(B_cols_tr)} features) --")
    print(f"   Excluded: {excl_B}")

    models_B = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
    }

    results_B = {}
    for name, mdl in models_B.items():
        print(f"\n  -- {name} --")
        mdl.fit(X_tr_B, y_tr)
        tr_res = evaluate_reg(mdl, X_tr_B, y_tr, "Train")
        vl_res = evaluate_reg(mdl, X_vl_B, y_vl, "Val  ")
        results_B[name] = {"model": mdl, "train": tr_res, "val": vl_res}

    # -- Hyperparameter tuning on best Setting-B model (val R2) -------------
    best_name_B = max(results_B, key=lambda n: results_B[n]["val"]["r2"])
    print(f"\n[REG] Best Setting-B model by Val R2: {best_name_B}")

    param_spaces = {
        "RandomForest": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.5, "sqrt", "log2"],
        },
        "XGBoost": {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 5, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 1.0],
        },
        "ExtraTrees": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
        },
    }

    base_map = {
        "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(verbosity=0, n_jobs=-1, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    X_tv_B = pd.concat([X_tr_B, X_vl_B], axis=0)
    y_tv   = pd.concat([y_tr, y_vl], axis=0)
    cv5    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rscv = RandomizedSearchCV(
        base_map[best_name_B], param_spaces[best_name_B],
        n_iter=30, cv=cv5, scoring="r2", n_jobs=-1,
        random_state=RANDOM_STATE, verbose=0
    )
    rscv.fit(X_tv_B, y_tv)
    best_params_B = rscv.best_params_
    print(f"[REG] Best params (Setting B): {best_params_B}")
    best_model_B = rscv.best_estimator_

    print("\n[REG] FINAL evaluation (Setting B, tuned) on independent test set:")
    te_res_B = evaluate_reg(best_model_B, X_te_B, y_te, "Test ")
    cv_r2 = cross_val_score(best_model_B, X_tv_B, y_tv, cv=cv5, scoring="r2", n_jobs=-1)
    print(f"  CV R2 (5-fold): {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")

    best_name_A = max(results_A, key=lambda n: results_A[n]["val"]["r2"])
    best_model_A = results_A[best_name_A]["model"]
    print(f"\n[REG] Setting A best ({best_name_A}) on test set:")
    te_res_A = evaluate_reg(best_model_A, X_te_raw, y_te, "Test ")

    # -- Compile comparison table -------------------------------------------
    rows = []
    for setting, res_dict in [("A_Full", results_A), ("B_SatOnly", results_B)]:
        for name, res in res_dict.items():
            rows.append({
                "Setting": setting,
                "Model": name,
                "Train_R2":   round(res["train"]["r2"], 4),
                "Val_R2":     round(res["val"]["r2"], 4),
                "Val_MAE_m3": round(res["val"]["mae"]),
                "Val_RMSE_m3":round(res["val"]["rmse"]),
                "Val_MAPE%":  round(res["val"]["mape"], 2),
            })

    comp_df = pd.DataFrame(rows)
    print("\n[REG] Model Comparison Table:")
    print(comp_df.to_string(index=False))

    # -- Feature importance (Setting B tuned) --------------------------------
    fi = best_model_B.feature_importances_
    fi_df = pd.DataFrame({"Feature": B_cols_tr, "Importance": fi}).sort_values("Importance", ascending=False)
    print("\n[REG] Top 15 Feature Importances (Setting B):")
    print(fi_df.head(15).to_string(index=False))

    # -- Save artefacts ------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    with open("models/best_water_requirement_regressor.pkl", "wb") as f:
        pickle.dump({
            "model_B": best_model_B,
            "model_A": best_model_A,
            "feature_names_B": B_cols_tr,
            "feature_names_A": X_tr_raw.columns.tolist(),
            "best_params_B": best_params_B,
            "best_name_B": best_name_B,
            "best_name_A": best_name_A,
        }, f)
    print("[REG] Saved: models/best_water_requirement_regressor.pkl")

    os.makedirs("reports", exist_ok=True)
    with open("reports/regression_report.txt", "w", encoding="utf-8") as f:
        f.write("AI-DRIVEN SMART IRRIGATION RECOMMENDATION SYSTEM\n")
        f.write("TASK B: WATER REQUIREMENT REGRESSION\n")
        f.write("=" * 60 + "\n\n")
        f.write("SCIENTIFIC DISCLAIMER:\n")
        f.write("Water_Required_m3 = Irrigation_Requirement_mm x Field_Area_ha x 10\n")
        f.write("Field_Area_ha is CONSTANT (605.672381 ha) for all records.\n")
        f.write("Setting A (full features) measures formula reproduction accuracy.\n")
        f.write("Setting B (satellite+ET0+Kc) measures true predictive challenge.\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("MODEL COMPARISON TABLE:\n")
        f.write(comp_df.to_string(index=False) + "\n\n")
        f.write(f"\nBest Setting-B Model: {best_name_B} (tuned)\n")
        f.write(f"Best Params: {best_params_B}\n")
        f.write(f"CV R2 (5-fold): {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}\n\n")
        f.write("TEST SET (Setting B, tuned):\n")
        f.write(f"  R2   = {te_res_B['r2']:.4f}\n")
        f.write(f"  MAE  = {te_res_B['mae']:,.0f} m3\n")
        f.write(f"  RMSE = {te_res_B['rmse']:,.0f} m3\n")
        f.write(f"  MAPE = {te_res_B['mape']:.2f}%\n\n")
        f.write("TOP 15 FEATURE IMPORTANCES (Setting B):\n")
        f.write(fi_df.head(15).to_string(index=False) + "\n")

    comp_df.to_csv("reports/regression_comparison.csv", index=False)
    fi_df.to_csv("reports/regression_feature_importance.csv", index=False)

    return {
        "best_model_B": best_model_B,
        "best_model_A": best_model_A,
        "best_name_B": best_name_B,
        "comparison": comp_df,
        "feature_importance": fi_df,
        "B_cols": B_cols_tr,
        "data": data,
        "X_train_B": X_tr_B, "X_val_B": X_vl_B, "X_test_B": X_te_B,
        "X_train_A": X_tr_raw, "X_val_A": X_vl_raw, "X_test_A": X_te_raw,
        "y_reg_train": y_tr, "y_reg_val": y_vl, "y_reg_test": y_te,
    }


if __name__ == "__main__":
    train_regressors()
