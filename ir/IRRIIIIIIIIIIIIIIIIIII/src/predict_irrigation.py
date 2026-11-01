"""
predict_irrigation.py
=====================
Production-ready, application-level prediction engine for the AI-Driven Smart Irrigation System.

Combines:
1. Machine Learning Classification (Optimal Threshold applied)
2. Machine Learning Regression (Water Volume Estimation in m3)
3. FAO-56 Agronomic Recommendation Engine (Advisory, percolation, water layer management)
4. Rigorous Input Validation & Sensor Data-Quality Warnings
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import add_engineered_features
from data_preprocessing import harmonise_crop_cycle, CAT_FEATURES
from irrigation_recommendation import calculate_irrigation_recommendation

# Strict domain-valid physical ranges
VALID_RANGES = {
    "B02_Blue":              (0.01, 0.70),
    "B03_Green":             (0.01, 0.70),
    "B04_Red":               (0.01, 0.75),
    "B08_NIR":               (0.01, 0.90),
    "B11_SWIR1":             (0.01, 0.80),
    "B12_SWIR2":             (0.01, 0.70),
    "NDVI":                  (-0.20, 0.95),
    "NDWI":                  (-0.80, 0.40),
    "MNDWI":                 (-0.80, 0.50),
    "NDMI":                  (-0.50, 0.60),
    "EVI":                   (-0.20, 1.00),
    "ET0_mm":                (1.0, 12.0),
    "PRECTOTCORR":           (0.0, 80.0),
    "Kc":                    (0.3, 1.8),
    "Crop_Progress":         (0.0, 1.0),
    "Effective_Rainfall_mm": (0.0, 50.0),
    "Cloud_Cover":           (0.0, 100.0),
    "Valid_Pixel_Count":     (1, 1000),
}

REQUIRED_FIELDS = [
    "B02_Blue", "B03_Green", "B04_Red", "B08_NIR", "B11_SWIR1", "B12_SWIR2",
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI",
    "ET0_mm", "PRECTOTCORR", "Kc", "Crop_Progress",
    "Crop_Stage", "Crop_Cycle",
    "Cloud_Cover", "Valid_Pixel_Count",
]


def load_models(model_dir="models") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads trained classifier and regressor bundles from disk."""
    if not os.path.isabs(model_dir):
        # Check current working directory, then relative to script
        if not os.path.exists(model_dir):
            model_dir = os.path.join(os.path.dirname(SRC_DIR), "models")

    cls_path = os.path.join(model_dir, "best_irrigation_classifier.pkl")
    reg_path = os.path.join(model_dir, "best_water_requirement_regressor.pkl")

    if not os.path.exists(cls_path) or not os.path.exists(reg_path):
        raise FileNotFoundError(
            f"Trained models not found in {model_dir}. Please run run_pipeline.py first."
        )

    with open(cls_path, "rb") as f:
        cls_bundle = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg_bundle = pickle.load(f)

    return cls_bundle, reg_bundle


def validate_observation(obs: Dict[str, Any]) -> Tuple[List[str], bool]:
    """
    Validates input schema and physical sensor boundaries.
    Returns (warnings, is_valid).
    """
    warns = []
    missing = [k for k in REQUIRED_FIELDS if k not in obs or obs[k] is None]
    if missing:
        warns.append(f"CRITICAL: Missing mandatory fields: {missing}")
        return warns, False

    for field, (lo, hi) in VALID_RANGES.items():
        if field in obs and obs[field] is not None:
            val = float(obs[field])
            if not (lo <= val <= hi):
                warns.append(f"OUT_OF_RANGE: {field}={val:.4f} is outside normal bounds [{lo}, {hi}].")

    if obs.get("Cloud_Cover", 0) > 60.0:
        warns.append(f"HIGH_CLOUD_COVER: Cloud cover ({obs['Cloud_Cover']:.1f}%) exceeds 60%. Optical indices may be degraded.")

    if obs.get("Valid_Pixel_Count", 500) < 50:
        warns.append(f"LOW_PIXEL_COUNT: Valid pixel count ({obs['Valid_Pixel_Count']}) is low. Scene quality degraded.")

    return warns, True


def predict_single(
    obs: Dict[str, Any],
    cls_bundle: Dict[str, Any],
    reg_bundle: Dict[str, Any],
    rainfall_mm: float = 0.0,
    field_area_ha: float = 605.672381,
) -> Dict[str, Any]:
    """
    End-to-end prediction for an individual field observation.
    """
    warnings, is_valid = validate_observation(obs)

    if not is_valid:
        return {
            "irrigation_required": "UNKNOWN",
            "confidence": 0.0,
            "decision_threshold": 0.50,
            "ml_water_required_m3": 0.0,
            "fao56_water_volume_m3": 0.0,
            "agronomic_advisory": "VALIDATION FAILED — check warnings",
            "stage_name": "N/A",
            "warnings": warnings,
            "top_features": {},
            "is_safe": False,
        }

    # Prepare DataFrame row
    row_data = {k: [obs[k]] for k in REQUIRED_FIELDS}
    # PRECTOTCORR: use directly if provided, otherwise derive from rainfall_mm parameter
    if "PRECTOTCORR" not in obs or obs["PRECTOTCORR"] is None:
        row_data["PRECTOTCORR"] = [rainfall_mm]
    if "Effective_Rainfall_mm" in obs:
        row_data["Effective_Rainfall_mm"] = [obs["Effective_Rainfall_mm"]]
    else:
        row_data["Effective_Rainfall_mm"] = [rainfall_mm * 0.80]

    if "ETc_mm" in obs:
        row_data["ETc_mm"] = [obs["ETc_mm"]]
    else:
        row_data["ETc_mm"] = [obs["ET0_mm"] * obs["Kc"]]

    df = pd.DataFrame(row_data)
    df["Crop_Cycle"] = harmonise_crop_cycle(df["Crop_Cycle"])

    # Categorical transformation using saved fitted encoder
    cat_enc = cls_bundle["cat_encoder"]
    df[CAT_FEATURES] = cat_enc.transform(df[CAT_FEATURES])

    # Feature engineering
    df_eng = add_engineered_features(df, include_rainfall_interactions=True)

    # 1. Classification Inference (Setting B Independent Model)
    cls_model = cls_bundle["model_B"]
    feat_cols_cls = cls_bundle["feature_names_B"]
    X_cls = df_eng[[c for c in feat_cols_cls if c in df_eng.columns]]

    prob_yes = float(cls_model.predict_proba(X_cls)[0, 1])
    th = cls_bundle.get("optimal_threshold_B", 0.50)
    irr_req = "YES" if prob_yes >= th else "NO"

    # 2. Regression Inference (Setting B Regressor)
    reg_model = reg_bundle["model_B"]
    feat_cols_reg = reg_bundle["feature_names_B"]
    X_reg = df_eng[[c for c in feat_cols_reg if c in df_eng.columns]]

    ml_water_m3 = float(np.clip(reg_model.predict(X_reg)[0], 0.0, None))
    # If classifier firmly predicts NO, water volume is zeroed out
    if irr_req == "NO":
        ml_water_m3 = 0.0

    # 3. FAO-56 Agronomic Decision Support System
    fao_rec = calculate_irrigation_recommendation(
        et0_mm=float(obs["ET0_mm"]),
        rainfall_mm=float(rainfall_mm),
        crop_stage=str(obs["Crop_Stage"]),
        kc=float(obs["Kc"]),
        field_area_ha=field_area_ha,
    )

    # 4. Feature Importance Attribution (Top 5 features)
    if hasattr(cls_model, "feature_importances_"):
        fi_series = pd.Series(cls_model.feature_importances_, index=feat_cols_cls)
        top5 = fi_series.nlargest(5).to_dict()
    else:
        top5 = {}

    return {
        "irrigation_required": irr_req,
        "confidence": round(prob_yes, 4),
        "decision_threshold": round(th, 2),
        "ml_water_required_m3": round(ml_water_m3, 2),
        "fao56_water_volume_m3": fao_rec["water_volume_m3"],
        "agronomic_advisory": fao_rec["agronomic_advisory"],
        "stage_name": fao_rec["stage_name"],
        "warnings": warnings,
        "top_features": {k: round(v, 4) for k, v in top5.items()},
        "is_safe": len(warnings) == 0,
    }


def predict_batch(df_input: pd.DataFrame, cls_bundle: Dict[str, Any], reg_bundle: Dict[str, Any]) -> pd.DataFrame:
    """Batch prediction on an entire pandas DataFrame."""
    results = []
    for _, row in df_input.iterrows():
        obs = row.to_dict()
        res = predict_single(obs, cls_bundle, reg_bundle, rainfall_mm=obs.get("PRECTOTCORR", 0.0))
        results.append({
            "Irrigation_Required": res["irrigation_required"],
            "Confidence_YES": res["confidence"],
            "ML_Water_Required_m3": res["ml_water_required_m3"],
            "FAO56_Water_m3": res["fao56_water_volume_m3"],
            "Warning_Count": len(res["warnings"]),
            "Is_Safe": res["is_safe"],
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Testing production prediction pipeline...")
    try:
        cls_b, reg_b = load_models()
        sample = {
            "B02_Blue": 0.08, "B03_Green": 0.10, "B04_Red": 0.09,
            "B08_NIR": 0.45, "B11_SWIR1": 0.28, "B12_SWIR2": 0.18,
            "NDVI": 0.66, "NDWI": -0.22, "MNDWI": -0.30, "NDMI": 0.12, "EVI": 0.38,
            "ET0_mm": 5.8, "Kc": 1.2, "Crop_Progress": 0.55,
            "Crop_Stage": "Mid", "Crop_Cycle": "2",
            "Cloud_Cover": 15.0, "Valid_Pixel_Count": 400,
        }
        res = predict_single(sample, cls_b, reg_b, rainfall_mm=0.2)
        print("Prediction Output:")
        for k, v in res.items():
            print(f"  {k:24s}: {v}")
    except Exception as e:
        print(f"Prediction test note: {e}")
