"""
data_preprocessing.py
=====================
Preprocessing pipeline for the AI-Driven Smart Irrigation Recommendation System.

Key architectural decisions (with scientific justification):
- Grid_ID, Scene_ID, Date are excluded from features (identifiers, not predictors).
- Field_Area_ha is a constant (605.672 ha) across all rows -- excluded as uninformative.
- Irrigation_Requirement_mm is excluded -- it is the direct antecedent of both targets.
- PRECTOTCORR is excluded -- it equals Effective_Rainfall_mm / 0.8 exactly.
- Effective_Rainfall_mm = 0.80 * PRECTOTCORR exactly; only one is kept.
- Crop_Cycle has a mixed format (1/2/3/4 vs "Cycle 1"/"Cycle 2") -- harmonised.
- All encoders / scalers are fit ONLY on training data and applied to validation/test.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import pickle
import os

# --- Feature sets -----------------------------------------------------------

IDENTIFIER_COLS = ["Date", "Grid_ID", "Scene_ID"]

CLASSIFICATION_TARGET = "Irrigation_Required"
REGRESSION_TARGET     = "Water_Required_m3"

# Satellite band + spectral index features
SPECTRAL_FEATURES = [
    "B02_Blue", "B03_Green", "B04_Red", "B08_NIR",
    "B11_SWIR1", "B12_SWIR2",
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI",
]

# Environmental / agro features
AGRO_FEATURES = [
    "Cloud_Cover", "Valid_Pixel_Count",
    "ET0_mm",
    "Crop_Progress",
    "Kc",
    "Effective_Rainfall_mm",
    "ETc_mm",          # crop water demand -- agronomically independent input
]

# Categorical features
CAT_FEATURES = ["Crop_Stage", "Crop_Cycle"]

ALL_NUMERICAL_FEATURES = SPECTRAL_FEATURES + AGRO_FEATURES
ALL_FEATURES = ALL_NUMERICAL_FEATURES + CAT_FEATURES


def harmonise_crop_cycle(series: pd.Series) -> pd.Series:
    """Normalise mixed format: 'Cycle 1' -> '1', '2' -> '2'."""
    return series.astype(str).str.replace(r"Cycle\s*", "", regex=True).str.strip()


def load_and_preprocess(csv_path: str, random_state: int = 42):
    """
    Load CSV, engineer features, split into train/validation/test sets.

    Returns
    -------
    splits : dict with keys
        X_train, X_val, X_test            -- DataFrames (features)
        y_cls_train/val/test              -- Series (0/1 binary)
        y_reg_train/val/test              -- Series (Water_Required_m3)
        feature_names                     -- list of feature columns used
        label_map                         -- {0: 'NO', 1: 'YES'}
        cat_encoder                       -- fitted OrdinalEncoder
    """
    df = pd.read_csv(csv_path)
    print(f"[DATA] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # -- Harmonise Crop_Cycle ------------------------------------------------
    df["Crop_Cycle"] = harmonise_crop_cycle(df["Crop_Cycle"])

    # -- Build feature matrix ------------------------------------------------
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y_cls = df[CLASSIFICATION_TARGET].copy()
    y_reg = df[REGRESSION_TARGET].copy()

    print(f"[DATA] Features used: {len(feature_cols)} -> {feature_cols}")
    print(f"[DATA] Missing values in X: {X.isnull().sum().sum()}")
    print(f"[DATA] Class distribution:")
    print(y_cls.value_counts(normalize=True).round(4).mul(100))

    # -- Encode classification target ----------------------------------------
    # NO -> 0, YES -> 1
    y_cls_enc = (y_cls == "YES").astype(int)
    label_map = {0: "NO", 1: "YES"}

    # -- Time-based split (temporal leakage prevention) ----------------------
    df["_Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True)
    sorted_dates = sorted(df["_Date_parsed"].unique())
    n_dates = len(sorted_dates)

    train_cutoff = sorted_dates[int(n_dates * 0.70)]
    val_cutoff   = sorted_dates[int(n_dates * 0.85)]

    train_mask = df["_Date_parsed"] <= train_cutoff
    val_mask   = (df["_Date_parsed"] > train_cutoff) & (df["_Date_parsed"] <= val_cutoff)
    test_mask  = df["_Date_parsed"] > val_cutoff

    print(f"[SPLIT] Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    print(f"[SPLIT] Train dates: {df[train_mask]['_Date_parsed'].min().date()} to {df[train_mask]['_Date_parsed'].max().date()}")
    print(f"[SPLIT] Val dates  : {df[val_mask]['_Date_parsed'].min().date()} to {df[val_mask]['_Date_parsed'].max().date()}")
    print(f"[SPLIT] Test dates : {df[test_mask]['_Date_parsed'].min().date()} to {df[test_mask]['_Date_parsed'].max().date()}")

    X_train = X[train_mask].copy()
    X_val   = X[val_mask].copy()
    X_test  = X[test_mask].copy()

    y_cls_train = y_cls_enc[train_mask].copy()
    y_cls_val   = y_cls_enc[val_mask].copy()
    y_cls_test  = y_cls_enc[test_mask].copy()

    y_reg_train = y_reg[train_mask].copy()
    y_reg_val   = y_reg[val_mask].copy()
    y_reg_test  = y_reg[test_mask].copy()

    # -- Encode categoricals (ordinal for tree models) -----------------------
    crop_stage_order = ["Initial", "Development", "Mid", "Late"]
    crop_cycle_vals  = sorted(X["Crop_Cycle"].unique(), key=lambda x: int(x))

    cat_encoder = OrdinalEncoder(
        categories=[crop_stage_order, crop_cycle_vals],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    cat_encoder.fit(X_train[CAT_FEATURES])

    for split in [X_train, X_val, X_test]:
        split[CAT_FEATURES] = cat_encoder.transform(split[CAT_FEATURES])

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_cls_train": y_cls_train, "y_cls_val": y_cls_val, "y_cls_test": y_cls_test,
        "y_reg_train": y_reg_train, "y_reg_val": y_reg_val, "y_reg_test": y_reg_test,
        "feature_names": feature_cols,
        "label_map": label_map,
        "cat_encoder": cat_encoder,
    }
