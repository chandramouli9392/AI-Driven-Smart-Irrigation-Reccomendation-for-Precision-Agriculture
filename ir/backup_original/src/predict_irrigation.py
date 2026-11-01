"""
predict_irrigation.py
=====================
Production prediction pipeline for the AI-Driven Smart Irrigation System.
"""

import os
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import add_engineered_features
from data_preprocessing import harmonise_crop_cycle, CAT_FEATURES

# -- Validated input ranges derived from training dataset --
VALID_RANGES = {
    "B02_Blue":             (0.05, 0.50),
    "B03_Green":            (0.05, 0.55),
    "B04_Red":              (0.05, 0.60),
    "B08_NIR":              (0.05, 0.80),
    "B11_SWIR1":            (0.05, 0.70),
    "B12_SWIR2":            (0.01, 0.55),
    "NDVI":                 (-0.1, 0.80),
    "NDWI":                 (-0.55, 0.10),
    "MNDWI":                (-0.50, 0.40),
    "NDMI":                 (-0.25, 0.40),
    "EVI":                  (-0.10, 0.80),
    "ET0_mm":               (3.5,  7.5),
    "ETc_mm":               (3.0,  9.0),
    "Kc":                   (0.5,  1.5),
    "Crop_Progress":        (0.0,  1.0),
    "Effective_Rainfall_mm":(0.0,  8.0),
    "Cloud_Cover":          (0.0,  100.0),
    "Valid_Pixel_Count":    (1,    1000),
}

REQUIRED_FIELDS = [
    "B02_Blue", "B03_Green", "B04_Red", "B08_NIR", "B11_SWIR1", "B12_SWIR2",
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI",
    "ET0_mm", "ETc_mm", "Kc", "Crop_Progress", "Effective_Rainfall_mm",
    "Crop_Stage", "Crop_Cycle",
    "Cloud_Cover", "Valid_Pixel_Count",
]


def load_models(model_dir="models"):
    """Load classifier and regressor from disk."""
    cls_path = os.path.join(model_dir, "best_irrigation_classifier.pkl")
    reg_path = os.path.join(model_dir, "best_water_requirement_regressor.pkl")
    if not os.path.exists(cls_path) or not os.path.exists(reg_path):
        raise FileNotFoundError(
            "Model files not found. Run train_classification.py and "
            "train_regression.py first."
        )
    with open(cls_path, "rb") as f:
        cls_bundle = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg_bundle = pickle.load(f)
    return cls_bundle, reg_bundle


def validate_inputs(obs: dict):
    """Returns (warnings_list, is_safe)."""
    warns = []
    missing = [k for k in REQUIRED_FIELDS if k not in obs or obs[k] is None]
    if missing:
        warns.append(f"CRITICAL: Missing required fields: {missing}")
        return warns, False
    for field, (lo, hi) in VALID_RANGES.items():
        if field in obs:
            val = obs[field]
            if not (lo <= val <= hi):
                warns.append(
                    f"WARNING: {field}={val:.4f} is outside trained range "
                    f"[{lo}, {hi}]. Prediction may be unreliable."
                )
    if obs.get("Cloud_Cover", 0) > 70:
        warns.append(
            f"WARNING: Cloud_Cover={obs['Cloud_Cover']:.1f}% is very high. "
            "Spectral indices may be unreliable."
        )
    return warns, True


def predict_single(obs: dict, cls_bundle: dict, reg_bundle: dict):
    """
    Predict irrigation requirement for a single paddy field observation.

    Returns dict with keys:
        irrigation_required, confidence, water_required_m3,
        warnings, top_features, is_safe
    """
    warns, is_safe = validate_inputs(obs)

    if not is_safe:
        return {
            "irrigation_required": "UNKNOWN",
            "confidence": None,
            "water_required_m3": None,
            "warnings": warns,
            "top_features": {},
            "is_safe": False,
        }

    row = {k: [obs[k]] for k in REQUIRED_FIELDS}
    df = pd.DataFrame(row)
    df["Crop_Cycle"] = harmonise_crop_cycle(df["Crop_Cycle"])

    cat_enc = cls_bundle["cat_encoder"]
    df[CAT_FEATURES] = cat_enc.transform(df[CAT_FEATURES])

    df_eng = add_engineered_features(df)

    # Classification
    cls_model = cls_bundle["model"]
    train_cols_cls = list(cls_model.feature_names_in_) \
        if hasattr(cls_model, "feature_names_in_") else list(df_eng.columns)
    available = [c for c in train_cols_cls if c in df_eng.columns]
    X_cls = df_eng[available]
    prob_yes = float(cls_model.predict_proba(X_cls)[0, 1])
    irr_req  = "YES" if prob_yes >= 0.5 else "NO"

    # Regression (Setting B)
    reg_model      = reg_bundle["model_B"]
    reg_features_B = reg_bundle["feature_names_B"]
    X_reg = df_eng[[c for c in reg_features_B if c in df_eng.columns]]
    water_m3 = max(0.0, float(reg_model.predict(X_reg)[0]))

    # Top feature importances
    fi = pd.Series(cls_model.feature_importances_, index=available)
    top_feats = fi.nlargest(5).to_dict()

    return {
        "irrigation_required": irr_req,
        "confidence": round(prob_yes, 4),
        "water_required_m3": round(water_m3, 2),
        "warnings": warns,
        "top_features": {k: round(v, 4) for k, v in top_feats.items()},
        "is_safe": True,
    }


def predict_batch(df_input: pd.DataFrame, cls_bundle: dict, reg_bundle: dict):
    """Predict for a batch DataFrame. Returns DataFrame with prediction columns."""
    results = []
    for _, row in df_input.iterrows():
        obs = row.to_dict()
        r = predict_single(obs, cls_bundle, reg_bundle)
        results.append({
            "irrigation_required": r["irrigation_required"],
            "confidence_yes":      r["confidence"],
            "water_required_m3":   r["water_required_m3"],
            "n_warnings":          len(r["warnings"]),
            "is_safe":             r["is_safe"],
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Loading models ...")
    cls_bundle, reg_bundle = load_models()

    sample = {
        "B02_Blue": 0.08, "B03_Green": 0.10, "B04_Red": 0.09,
        "B08_NIR": 0.45, "B11_SWIR1": 0.28, "B12_SWIR2": 0.18,
        "NDVI": 0.66, "NDWI": -0.22, "MNDWI": -0.30, "NDMI": 0.12, "EVI": 0.38,
        "ET0_mm": 5.8, "ETc_mm": 6.96, "Kc": 1.2,
        "Crop_Progress": 0.55, "Effective_Rainfall_mm": 0.40,
        "Crop_Stage": "Mid", "Crop_Cycle": "2",
        "Cloud_Cover": 15.0, "Valid_Pixel_Count": 400,
    }

    print("\n-- Sample Prediction --")
    result = predict_single(sample, cls_bundle, reg_bundle)
    print(f"  Irrigation Required : {result['irrigation_required']}")
    print(f"  Confidence (P=YES)  : {result['confidence']:.4f}")
    print(f"  Water Required (m3) : {result['water_required_m3']:,.0f}")
    print(f"  Warnings            : {result['warnings'] or 'None'}")
    print(f"  Top driving features: {result['top_features']}")
