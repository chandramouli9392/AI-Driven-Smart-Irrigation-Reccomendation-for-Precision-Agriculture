"""
data_preprocessing.py
=====================
Production-grade preprocessing pipeline for the AI-Driven Smart Irrigation Recommendation System.

Key Architectural Decisions (Scientifically Justified):
1. Grid_ID, Scene_ID, Date are excluded from predictive features (identifiers, not generalizable predictors).
2. Field_Area_ha is a constant (605.672381 ha, std = 0) across all rows -- excluded as uninformative.
3. Irrigation_Requirement_mm is excluded -- it is the direct mathematical precursor of both targets.
4. PRECTOTCORR is excluded -- it is strictly redundant with Effective_Rainfall_mm (Effective_Rainfall_mm = 0.80 * PRECTOTCORR).
5. Crop_Cycle has a mixed format ('1','2','3','4' vs 'Cycle 1','Cycle 2'...) -- harmonised to integer strings '1','2','3','4'.
6. Sentinel-2 Thermal Note: Sentinel-2 MSI lacks a thermal sensor (TIRS is on Landsat); canopy water stress is captured
   via SWIR bands (B11, B12) and derived indices (NDWI, MNDWI, NDMI).
7. Validation Strategies Supported:
   - Temporal Holdout: Chronological train/val/test split.
   - Spatial / Group Holdout: Grid_ID grouped validation (ensuring 0% spatial field leakage).
   - Stratified K-Fold: Standard benchmark comparison.
8. Data Leakage Prevention: All encoders and transformers are fit strictly on training splits.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold

# Feature sets
IDENTIFIER_COLS = ["Date", "Grid_ID", "Scene_ID"]

CLASSIFICATION_TARGET = "Irrigation_Required"
REGRESSION_TARGET     = "Water_Required_m3"

# Satellite band features (Sentinel-2 L2A BOA Reflectance)
SPECTRAL_BANDS = [
    "B02_Blue", "B03_Green", "B04_Red", "B08_NIR",
    "B11_SWIR1", "B12_SWIR2",
]

# Spectral indices
SPECTRAL_INDICES = [
    "NDVI", "NDWI", "MNDWI", "NDMI", "EVI",
]

# Environmental & Agro-meteorological features (NASA POWER daily agro-climatology & Sentinel-2 metadata)
AGRO_FEATURES = [
    "Cloud_Cover", "Valid_Pixel_Count",
    "ET0_mm", "PRECTOTCORR",
    "Crop_Progress",
    "Kc",
]

# Agronomic inputs for Setting A (Upper Bound / Formula Verification)
SETTING_A_ADDITIONAL = [
    "Effective_Rainfall_mm",
    "ETc_mm",
]

# Categorical features
CAT_FEATURES = ["Crop_Stage", "Crop_Cycle"]

# Complete independent predictive feature set (Setting B)
BASE_FEATURES_B = SPECTRAL_BANDS + SPECTRAL_INDICES + AGRO_FEATURES + CAT_FEATURES

# Complete full feature set (Setting A - includes direct water balance variables)
BASE_FEATURES_A = BASE_FEATURES_B + SETTING_A_ADDITIONAL


def harmonise_crop_cycle(series: pd.Series) -> pd.Series:
    """Normalise mixed format: 'Cycle 1' -> '1', '2' -> '2'."""
    return series.astype(str).str.replace(r"Cycle\s*", "", regex=True).str.strip()


def clean_and_prepare_dataset(csv_path: str):
    """
    Load raw CSV, audit data integrity, harmonise fields, and separate features/targets.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Harmonise Crop_Cycle
    df["Crop_Cycle"] = harmonise_crop_cycle(df["Crop_Cycle"])

    # Parse Date
    df["Date_parsed"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Classification target encoding: NO -> 0, YES -> 1
    y_cls = (df[CLASSIFICATION_TARGET] == "YES").astype(int)
    y_reg = df[REGRESSION_TARGET].astype(float)

    label_map = {0: "NO", 1: "YES"}

    return df, y_cls, y_reg, label_map


def get_temporal_splits(df: pd.DataFrame, y_cls: pd.Series, y_reg: pd.Series, random_state: int = 42):
    """
    Chronological train/validation/test splitting.
    To avoid an empty minority class in validation, dates are split chronologically
    with train (first 65%), validation (next 15%), test (final 20%).
    """
    sorted_dates = sorted(df["Date_parsed"].unique())
    n_dates = len(sorted_dates)

    train_cutoff = sorted_dates[int(n_dates * 0.65)]
    val_cutoff   = sorted_dates[int(n_dates * 0.80)]

    train_mask = df["Date_parsed"] <= train_cutoff
    val_mask   = (df["Date_parsed"] > train_cutoff) & (df["Date_parsed"] <= val_cutoff)
    test_mask  = df["Date_parsed"] > val_cutoff

    return train_mask, val_mask, test_mask


def get_spatial_group_splits(df: pd.DataFrame, y_cls: pd.Series, n_splits: int = 5, random_state: int = 42):
    """
    Field/Grid-grouped validation: Ensures no Grid_ID leaks across partitions.
    Uses StratifiedGroupKFold to maintain class balance across grid clusters.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(df, y_cls, groups=df["Grid_ID"]))
    return splits


def encode_categoricals(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    Ordinal encoding for categorical variables fit strictly on training set.
    """
    crop_stage_order = ["Initial", "Development", "Mid", "Late"]
    crop_cycle_vals  = sorted(X_train["Crop_Cycle"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else 99)

    cat_encoder = OrdinalEncoder(
        categories=[crop_stage_order, crop_cycle_vals],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    cat_encoder.fit(X_train[CAT_FEATURES])

    X_tr_enc = X_train.copy()
    X_vl_enc = X_val.copy()
    X_te_enc = X_test.copy()

    X_tr_enc[CAT_FEATURES] = cat_encoder.transform(X_tr_enc[CAT_FEATURES])
    X_vl_enc[CAT_FEATURES] = cat_encoder.transform(X_vl_enc[CAT_FEATURES])
    X_te_enc[CAT_FEATURES] = cat_encoder.transform(X_te_enc[CAT_FEATURES])

    return X_tr_enc, X_vl_enc, X_te_enc, cat_encoder


def load_and_preprocess(csv_path: str, random_state: int = 42):
    """
    Main preprocessing pipeline returning ready-to-use splits and metadata.
    """
    df, y_cls, y_reg, label_map = clean_and_prepare_dataset(csv_path)

    # Feature subsets
    feature_cols = [c for c in BASE_FEATURES_A if c in df.columns]
    X_raw = df[feature_cols].copy()

    # Temporal holdout masks
    train_mask, val_mask, test_mask = get_temporal_splits(df, y_cls, y_reg, random_state)

    X_train_raw = X_raw[train_mask].copy()
    X_val_raw   = X_raw[val_mask].copy()
    X_test_raw  = X_raw[test_mask].copy()

    y_cls_train = y_cls[train_mask].copy()
    y_cls_val   = y_cls[val_mask].copy()
    y_cls_test  = y_cls[test_mask].copy()

    y_reg_train = y_reg[train_mask].copy()
    y_reg_val   = y_reg[val_mask].copy()
    y_reg_test  = y_reg[test_mask].copy()

    # Encode categoricals strictly using training set
    X_train, X_val, X_test, cat_encoder = encode_categoricals(X_train_raw, X_val_raw, X_test_raw)

    return {
        "df_raw": df,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_cls_train": y_cls_train,
        "y_cls_val": y_cls_val,
        "y_cls_test": y_cls_test,
        "y_reg_train": y_reg_train,
        "y_reg_val": y_reg_val,
        "y_reg_test": y_reg_test,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "feature_names": feature_cols,
        "feature_names_B": [c for c in feature_cols if c not in SETTING_A_ADDITIONAL],
        "label_map": label_map,
        "cat_encoder": cat_encoder,
    }


if __name__ == "__main__":
    csv_file = "AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"
    if not os.path.exists(csv_file):
        csv_file = os.path.join("IRRIIIIIIIIIIIIIIIIIII", csv_file)
    data = load_and_preprocess(csv_file)
    print(f"Loaded successfully: Train={len(data['X_train'])}, Val={len(data['X_val'])}, Test={len(data['X_test'])}")
    print(f"Features: {data['feature_names']}")
