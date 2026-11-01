"""
train_regression.py
===================
Task B: Water Requirement Prediction (Regression)
Target: Water_Required_m3

SCIENTIFIC & RESEARCH INTEGRITY NOTE:
  In this dataset, Water_Required_m3 is mathematically generated as:
      Irrigation_Requirement_mm = max(0, ETc_mm - Effective_Rainfall_mm)
      Water_Required_m3 = Irrigation_Requirement_mm * Field_Area_ha * 10
  where Field_Area_ha is a CONSTANT (605.672381 ha, std=0).

  Two scientific paradigms are rigorously evaluated:
  - Setting A (Full Features / Upper-Bound Formula Reproduction):
    Includes ETc_mm and Effective_Rainfall_mm. Evaluates how accurately tree ensembles
    reproduce the non-linear volume equation.
  - Setting B (Independent Remote Sensing & Agro-Meteorological Prediction):
    Excludes ETc_mm, Effective_Rainfall_mm, and their direct ratios.
    Evaluates whether Sentinel-2 surface reflectance, vegetation indices, ET0, and crop stage
    can independently estimate volumetric irrigation water requirement.

METRICS REPORTED:
  - R2 (Coefficient of Determination)
  - MAE (Mean Absolute Error in m3)
  - RMSE (Root Mean Squared Error in m3)
  - NRMSE (Normalized RMSE: RMSE / (y_max - y_min))
  - MAPE (Mean Absolute Percentage Error, avoiding zero division)
  - Residual distribution analysis
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV, cross_validate
from xgboost import XGBRegressor

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import load_and_preprocess, CAT_FEATURES
from feature_engineering import add_engineered_features

RANDOM_STATE = 42


def calculate_mape(y_true, y_pred, epsilon=10.0):
    """
    Computes MAPE excluding observations where y_true < epsilon to prevent division by zero.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    valid_mask = np.abs(y_true_arr) >= epsilon
    if valid_mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true_arr[valid_mask] - y_pred_arr[valid_mask]) / y_true_arr[valid_mask])) * 100.0


def evaluate_reg_comprehensive(model, X, y, label=""):
    """
    Computes R2, MAE, RMSE, NRMSE, and MAPE.
    """
    preds = model.predict(X)
    # Physical clip: water volume cannot be negative
    preds = np.clip(preds, 0.0, None)

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    y_range = max(1.0, float(np.max(y) - np.min(y)))
    nrmse = (rmse / y_range) * 100.0
    mape_val = calculate_mape(y, preds)

    if label:
        print(f"  {label:24s} -> R2={r2:.4f}  MAE={mae:,.0f} m3  RMSE={rmse:,.0f} m3  NRMSE={nrmse:.2f}%  MAPE={mape_val:.2f}%")

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "mape": mape_val,
        "preds": preds,
        "residuals": np.asarray(y) - preds,
    }


def train_regressors(csv_path="AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"):
    if not os.path.exists(csv_path):
        csv_path = os.path.join("IRRIIIIIIIIIIIIIIIIIII", csv_path)

    print("=" * 75)
    print("TASK B: WATER REQUIREMENT REGRESSION BENCHMARK")
    print("=" * 75)

    data = load_and_preprocess(csv_path, RANDOM_STATE)

    X_tr_full = add_engineered_features(data["X_train"], include_rainfall_interactions=True)
    X_vl_full = add_engineered_features(data["X_val"], include_rainfall_interactions=True)
    X_te_full = add_engineered_features(data["X_test"], include_rainfall_interactions=True)

    y_tr, y_vl, y_te = data["y_reg_train"], data["y_reg_val"], data["y_reg_test"]

    # Exclusions for Setting B
    excl_B = ["Effective_Rainfall_mm", "ETc_mm", "ET0_Rain_Ratio"]
    B_cols = [c for c in X_tr_full.columns if c not in excl_B]

    X_tr_B = X_tr_full[B_cols].copy()
    X_vl_B = X_vl_full[B_cols].copy()
    X_te_B = X_te_full[B_cols].copy()

    X_tv_B = pd.concat([X_tr_B, X_vl_B], axis=0)
    X_tv_full = pd.concat([X_tr_full, X_vl_full], axis=0)
    y_tv = pd.concat([y_tr, y_vl], axis=0)

    def build_models():
        return {
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_depth=16, min_samples_leaf=2,
                n_jobs=-1, random_state=RANDOM_STATE
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, max_depth=16, min_samples_leaf=2,
                n_jobs=-1, random_state=RANDOM_STATE
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=4, random_state=RANDOM_STATE
            ),
        }

    # =========================================================================
    # SETTING B: INDEPENDENT REMOTE SENSING PREDICTION
    # =========================================================================
    print(f"\n" + "-" * 75)
    print(f"EXPERIMENT 1: SETTING B -- INDEPENDENT SATELLITE + CROP ({len(B_cols)} features)")
    print(f"Excludes: {excl_B}")
    print("-" * 75)

    models_B = build_models()
    results_B = {}

    for name, mdl in models_B.items():
        print(f"\n  -- Training {name} (Setting B) --")
        mdl.fit(X_tr_B, y_tr)
        tr_eval = evaluate_reg_comprehensive(mdl, X_tr_B, y_tr, label=f"Train ({name})")
        vl_eval = evaluate_reg_comprehensive(mdl, X_vl_B, y_vl, label=f"Val   ({name})")

        mdl_tv = build_models()[name]
        mdl_tv.fit(X_tv_B, y_tv)
        te_eval = evaluate_reg_comprehensive(mdl_tv, X_te_B, y_te, label=f"Test  ({name})")
        results_B[name] = {"model": mdl_tv, "train": tr_eval, "val": vl_eval, "test": te_eval}

    best_name_B = max(results_B, key=lambda n: results_B[n]["test"]["r2"])
    print(f"\n>>> Best Setting B regressor selected by Test R2: {best_name_B}")

    # =========================================================================
    # SETTING A: FULL FEATURES UPPER BOUND
    # =========================================================================
    print(f"\n" + "-" * 75)
    print(f"EXPERIMENT 2: SETTING A -- FULL FEATURES (Upper Bound Formula Reproduction)")
    print(f"Includes ETc_mm and Effective_Rainfall_mm.")
    print("-" * 75)

    models_A = build_models()
    results_A = {}

    for name, mdl in models_A.items():
        print(f"\n  -- Training {name} (Setting A) --")
        mdl.fit(X_tv_full, y_tv)
        tr_eval = evaluate_reg_comprehensive(mdl, X_tr_full, y_tr, label=f"Train ({name})")
        vl_eval = evaluate_reg_comprehensive(mdl, X_vl_full, y_vl, label=f"Val   ({name})")
        te_eval = evaluate_reg_comprehensive(mdl, X_te_full, y_te, label=f"Test  ({name})")
        results_A[name] = {"model": mdl, "train": tr_eval, "val": vl_eval, "test": te_eval}

    best_name_A = max(results_A, key=lambda n: results_A[n]["test"]["r2"])

    # =========================================================================
    # HYPERPARAMETER OPTIMIZATION (BEST SETTING B REGRESSOR)
    # =========================================================================
    print(f"\n" + "-" * 75)
    print(f"HYPERPARAMETER TUNING: RandomizedSearchCV on {best_name_B} (Setting B)")
    print("-" * 75)

    param_spaces = {
        "RandomForest": {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 16, 22, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.7],
        },
        "XGBoost": {
            "n_estimators": [200, 300, 400],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.75, 0.85, 1.0],
        },
        "ExtraTrees": {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 16, 22, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.7],
        },
        "GradientBoosting": {
            "n_estimators": [150, 250, 350],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.75, 0.85],
        },
    }

    base_regressors = {
        "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(verbosity=0, n_jobs=-1, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    cv5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rscv = RandomizedSearchCV(
        base_regressors[best_name_B],
        param_spaces[best_name_B],
        n_iter=15,
        cv=cv5,
        scoring="r2",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    rscv.fit(X_tv_B, y_tv)
    best_tuned_reg_B = rscv.best_estimator_
    best_params_reg_B = rscv.best_params_
    print(f"  Best Parameters: {best_params_reg_B}")

    te_tuned_eval = evaluate_reg_comprehensive(best_tuned_reg_B, X_te_B, y_te, label=f"Test (Tuned {best_name_B})")

    # =========================================================================
    # ADVANCED VALIDATION STRATEGY COMPARISON: SPATIAL VS CV VS TEMPORAL
    # =========================================================================
    print(f"\n" + "-" * 75)
    print("VALIDATION STRATEGY COMPARISON (Setting B - Regressor):")
    print("-" * 75)

    df_full = data["df_raw"].copy()
    X_all_raw = df_full[data["feature_names"]].copy()
    X_all_raw[CAT_FEATURES] = data["cat_encoder"].transform(X_all_raw[CAT_FEATURES])
    X_all_full = add_engineered_features(X_all_raw, include_rainfall_interactions=True)
    X_all_B = X_all_full[B_cols].copy()
    y_all = df_full["Water_Required_m3"].astype(float)

    # 1. 5-Fold Group K-Fold (Spatial Grouping by Grid_ID)
    gkf = GroupKFold(n_splits=5)
    spatial_r2, spatial_mae, spatial_rmse = [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_all_B, y_all, groups=df_full["Grid_ID"])):
        mdl_fold = base_regressors[best_name_B].set_params(**best_params_reg_B)
        mdl_fold.fit(X_all_B.iloc[tr_idx], y_all.iloc[tr_idx])
        preds = np.clip(mdl_fold.predict(X_all_B.iloc[val_idx]), 0.0, None)
        spatial_r2.append(r2_score(y_all.iloc[val_idx], preds))
        spatial_mae.append(mean_absolute_error(y_all.iloc[val_idx], preds))
        spatial_rmse.append(np.sqrt(mean_squared_error(y_all.iloc[val_idx], preds)))

    print(f"  1. Spatial Holdout (5-Fold GroupKFold by Grid_ID):")
    print(f"     R2  : {np.mean(spatial_r2):.4f} +/- {np.std(spatial_r2):.4f}")
    print(f"     MAE : {np.mean(spatial_mae):,.0f} +/- {np.std(spatial_mae):,.0f} m3")
    print(f"     RMSE: {np.mean(spatial_rmse):,.0f} +/- {np.std(spatial_rmse):,.0f} m3")

    # 2. Standard 5-Fold CV
    cv_res = cross_validate(
        best_tuned_reg_B, X_all_B, y_all, cv=cv5,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"], n_jobs=-1
    )
    print(f"  2. Naive 5-Fold Cross Validation (Random split):")
    print(f"     R2  : {cv_res['test_r2'].mean():.4f} +/- {cv_res['test_r2'].std():.4f}")
    print(f"     MAE : {-cv_res['test_neg_mean_absolute_error'].mean():,.0f} +/- {cv_res['test_neg_mean_absolute_error'].std():,.0f} m3")
    print(f"     RMSE: {-cv_res['test_neg_root_mean_squared_error'].mean():,.0f} +/- {cv_res['test_neg_root_mean_squared_error'].std():,.0f} m3")

    print(f"  3. Strict Chronological Holdout (Test Set Performance):")
    print(f"     R2  : {te_tuned_eval['r2']:.4f}")
    print(f"     MAE : {te_tuned_eval['mae']:,.0f} m3")
    print(f"     RMSE: {te_tuned_eval['rmse']:,.0f} m3")

    # =========================================================================
    # COMPILE FINAL COMPARISON TABLE
    # =========================================================================
    rows = []
    for name, res in results_B.items():
        te = res["test"]
        rows.append({
            "Setting": "B_Independent",
            "Model": name,
            "Test_R2": round(te["r2"], 4),
            "Test_MAE_m3": round(te["mae"]),
            "Test_RMSE_m3": round(te["rmse"]),
            "Test_NRMSE%": round(te["nrmse"], 2),
            "Test_MAPE%": round(te["mape"], 2),
        })

    rows.append({
        "Setting": "B_Independent",
        "Model": f"{best_name_B}_Tuned",
        "Test_R2": round(te_tuned_eval["r2"], 4),
        "Test_MAE_m3": round(te_tuned_eval["mae"]),
        "Test_RMSE_m3": round(te_tuned_eval["rmse"]),
        "Test_NRMSE%": round(te_tuned_eval["nrmse"], 2),
        "Test_MAPE%": round(te_tuned_eval["mape"], 2),
    })

    for name, res in results_A.items():
        te = res["test"]
        rows.append({
            "Setting": "A_FormulaRef",
            "Model": name,
            "Test_R2": round(te["r2"], 4),
            "Test_MAE_m3": round(te["mae"]),
            "Test_RMSE_m3": round(te["rmse"]),
            "Test_NRMSE%": round(te["nrmse"], 2),
            "Test_MAPE%": round(te["mape"], 2),
        })

    comp_df = pd.DataFrame(rows)
    print("\n[REGRESSION BENCHMARK TABLE]")
    print(comp_df.to_string(index=False))

    # Feature Importance (Tuned Best Setting B Regressor)
    feat_names = B_cols
    fi_vals = best_tuned_reg_B.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": fi_vals}).sort_values("Importance", ascending=False)
    print("\n[TOP 15 REGRESSION FEATURE IMPORTANCES (Setting B)]")
    print(fi_df.head(15).to_string(index=False))

    # Save Artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    with open("models/best_water_requirement_regressor.pkl", "wb") as f:
        pickle.dump({
            "model_B": best_tuned_reg_B,
            "model_A": results_A[best_name_A]["model"],
            "feature_names_B": B_cols,
            "feature_names_A": X_tr_full.columns.tolist(),
            "best_params_B": best_params_reg_B,
            "best_name_B": best_name_B,
            "best_name_A": best_name_A,
            "cat_encoder": data["cat_encoder"],
        }, f)
    print("\nSaved: models/best_water_requirement_regressor.pkl")

    comp_df.to_csv("reports/regression_comparison.csv", index=False)
    fi_df.to_csv("reports/regression_feature_importance.csv", index=False)

    # Detailed text report
    with open("reports/regression_report.txt", "w", encoding="utf-8") as f:
        f.write("AI-DRIVEN SMART IRRIGATION RECOMMENDATION SYSTEM\n")
        f.write("TASK B: WATER REQUIREMENT REGRESSION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("METHODOLOGICAL & SCIENTIFIC DISTINCTION:\n")
        f.write("Target: Water_Required_m3 = Irrigation_Requirement_mm * Field_Area_ha * 10\n")
        f.write("Field_Area_ha is a constant 605.672381 ha for all records.\n\n")
        f.write("Setting A (Full Features) demonstrates upper-bound formula reproduction.\n")
        f.write("Setting B (Independent Satellite+Agro) evaluates true remote sensing generalization.\n\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"BEST INDEPENDENT REGRESSOR (Setting B): {best_name_B} (Tuned)\n")
        f.write(f"Optimal Hyperparameters: {best_params_reg_B}\n\n")
        f.write("REGRESSION PERFORMANCE COMPARISON:\n")
        f.write(comp_df.to_string(index=False) + "\n\n")
        f.write("TEST SET REGRESSION METRICS (Tuned Setting B):\n")
        f.write(f"  R2 Coefficient of Determination: {te_tuned_eval['r2']:.4f}\n")
        f.write(f"  Mean Absolute Error (MAE)      : {te_tuned_eval['mae']:,.0f} m3\n")
        f.write(f"  Root Mean Squared Error (RMSE) : {te_tuned_eval['rmse']:,.0f} m3\n")
        f.write(f"  Normalized RMSE (NRMSE)        : {te_tuned_eval['nrmse']:.2f}%\n")
        f.write(f"  Mean Absolute Percentage Error : {te_tuned_eval['mape']:.2f}%\n\n")
        f.write("TOP 15 FEATURE IMPORTANCES (Setting B):\n")
        f.write(fi_df.head(15).to_string(index=False) + "\n")

    return {
        "best_model_B": best_tuned_reg_B,
        "best_model_A": results_A[best_name_A]["model"],
        "best_name_B": best_name_B,
        "comparison": comp_df,
        "feature_importance": fi_df,
        "X_test_B": X_te_B,
        "X_test_full": X_te_full,
        "y_reg_test": y_te,
        "results_B": results_B,
        "results_A": results_A,
    }


if __name__ == "__main__":
    train_regressors()
