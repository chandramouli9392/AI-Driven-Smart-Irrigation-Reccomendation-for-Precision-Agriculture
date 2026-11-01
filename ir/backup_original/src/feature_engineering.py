"""
feature_engineering.py
=======================
Additional derived features for the irrigation models.
All features here are computed from independent inputs only.
No target-derived quantities.
"""

import numpy as np
import pandas as pd


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add agronomically and remotely sensed derived features.
    Input X must already contain the base feature columns.
    Returns a new DataFrame with extra columns appended.
    """
    Xf = X.copy()

    # 1. Simple Ratio (SR): NIR/Red
    if "B08_NIR" in Xf.columns and "B04_Red" in Xf.columns:
        denom = Xf["B04_Red"].replace(0, np.nan)
        Xf["SR"] = Xf["B08_NIR"] / denom

    # 2. Bare Soil Index (BSI)
    if all(c in Xf.columns for c in ["B11_SWIR1", "B04_Red", "B08_NIR", "B02_Blue"]):
        num = Xf["B11_SWIR1"] + Xf["B04_Red"] - Xf["B08_NIR"] - Xf["B02_Blue"]
        den = Xf["B11_SWIR1"] + Xf["B04_Red"] + Xf["B08_NIR"] + Xf["B02_Blue"]
        den = den.replace(0, np.nan)
        Xf["BSI"] = num / den

    # 3. Kc x Crop_Progress interaction
    if "Kc" in Xf.columns and "Crop_Progress" in Xf.columns:
        Xf["Kc_Progress"] = Xf["Kc"] * Xf["Crop_Progress"]

    # 4. NDVI x Crop_Progress
    if "NDVI" in Xf.columns and "Crop_Progress" in Xf.columns:
        Xf["NDVI_Progress"] = Xf["NDVI"] * Xf["Crop_Progress"]

    # 5. ET0 to Rainfall ratio (proxy for water deficit severity)
    if "ET0_mm" in Xf.columns and "Effective_Rainfall_mm" in Xf.columns:
        Xf["ET0_Rain_Ratio"] = Xf["ET0_mm"] / (Xf["Effective_Rainfall_mm"] + 0.1)

    # Fill any NaN introduced by division
    Xf = Xf.fillna(Xf.median(numeric_only=True))

    return Xf


def get_feature_groups():
    """Return a dict mapping group names to feature lists for importance analysis."""
    return {
        "Spectral_Bands":   ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR", "B11_SWIR1", "B12_SWIR2"],
        "Spectral_Indices": ["NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SR", "BSI"],
        "ET_Crop":          ["ET0_mm", "ETc_mm", "Kc", "Crop_Progress", "Kc_Progress", "NDVI_Progress"],
        "Water_Balance":    ["Effective_Rainfall_mm", "ET0_Rain_Ratio"],
        "Crop_Meta":        ["Crop_Stage", "Crop_Cycle"],
        "Image_Quality":    ["Cloud_Cover", "Valid_Pixel_Count"],
    }
