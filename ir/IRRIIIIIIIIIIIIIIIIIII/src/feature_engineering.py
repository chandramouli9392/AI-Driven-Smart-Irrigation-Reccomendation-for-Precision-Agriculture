"""
feature_engineering.py
=======================
Advanced agronomic and remote-sensing feature engineering for precision agriculture.

Engineered Features:
1. SR (Simple Ratio): NIR / Red (vigour index).
2. BSI (Bare Soil Index): ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)).
3. MSI (Moisture Stress Index): SWIR1 / NIR (higher values indicate severe canopy moisture deficit).
4. NDRE_proxy (Red-Edge/NIR proxy): (NIR - Red) / (NIR + Red + 0.01).
5. Kc_Progress Interaction: Kc * Crop_Progress (represents biological water demand adjusted for phenology).
6. NDVI_Progress Interaction: NDVI * Crop_Progress (canopy development adjusted for phenology).
7. NDWI_Progress Interaction: NDWI * Crop_Progress (canopy water content adjusted for phenology).
8. ET0_Rain_Ratio: ET0_mm / (Effective_Rainfall_mm + 0.1) [Included in Setting A as water deficit index].
"""

import sys
import os
import numpy as np
import pandas as pd


def add_engineered_features(X: pd.DataFrame, include_rainfall_interactions: bool = True) -> pd.DataFrame:
    """
    Compute domain-specific agronomic and spectral features.
    Guarantees no target leakage.
    """
    Xf = X.copy()

    # 1. Simple Ratio (SR): NIR / Red
    if "B08_NIR" in Xf.columns and "B04_Red" in Xf.columns:
        denom = Xf["B04_Red"].replace(0, np.nan)
        Xf["SR"] = (Xf["B08_NIR"] / denom).clip(0, 50.0)

    # 2. Bare Soil Index (BSI)
    if all(c in Xf.columns for c in ["B11_SWIR1", "B04_Red", "B08_NIR", "B02_Blue"]):
        num = (Xf["B11_SWIR1"] + Xf["B04_Red"]) - (Xf["B08_NIR"] + Xf["B02_Blue"])
        den = (Xf["B11_SWIR1"] + Xf["B04_Red"]) + (Xf["B08_NIR"] + Xf["B02_Blue"])
        den = den.replace(0, np.nan)
        Xf["BSI"] = (num / den).clip(-1.0, 1.0)

    # 3. Moisture Stress Index (MSI): SWIR1 / NIR
    if "B11_SWIR1" in Xf.columns and "B08_NIR" in Xf.columns:
        den = Xf["B08_NIR"].replace(0, np.nan)
        Xf["MSI"] = (Xf["B11_SWIR1"] / den).clip(0, 10.0)

    # 4. Phenology & Crop Growth Interactions
    if "Kc" in Xf.columns and "Crop_Progress" in Xf.columns:
        Xf["Kc_Progress"] = Xf["Kc"] * Xf["Crop_Progress"]

    if "NDVI" in Xf.columns and "Crop_Progress" in Xf.columns:
        Xf["NDVI_Progress"] = Xf["NDVI"] * Xf["Crop_Progress"]

        Xf["NDWI_Progress"] = Xf["NDWI"] * Xf["Crop_Progress"]

    # 5. Agro-climatic precipitation to atmospheric demand ratio (Setting B: using raw PRECTOTCORR)
    if "PRECTOTCORR" in Xf.columns and "ET0_mm" in Xf.columns:
        Xf["Rain_to_ET0_Ratio"] = (Xf["PRECTOTCORR"] / (Xf["ET0_mm"] + 0.1)).clip(0, 100.0)

    # 6. Agro-climatic deficit ratio (Setting A only: using Effective_Rainfall_mm)
    if include_rainfall_interactions and "ET0_mm" in Xf.columns and "Effective_Rainfall_mm" in Xf.columns:
        Xf["ET0_Rain_Ratio"] = (Xf["ET0_mm"] / (Xf["Effective_Rainfall_mm"] + 0.1)).clip(0, 100.0)

    # Handle any NaN values introduced by division
    Xf = Xf.fillna(Xf.median(numeric_only=True))

    return Xf


def get_feature_groups():
    """Mapping of feature domain categories for interpretability and ablation."""
    return {
        "Spectral_Bands":   ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR", "B11_SWIR1", "B12_SWIR2"],
        "Spectral_Indices": ["NDVI", "NDWI", "MNDWI", "NDMI", "EVI", "SR", "BSI", "MSI"],
        "ET_Crop":          ["ET0_mm", "ETc_mm", "Kc", "Crop_Progress", "Kc_Progress", "NDVI_Progress", "NDWI_Progress"],
        "Water_Balance":    ["Effective_Rainfall_mm", "ET0_Rain_Ratio"],
        "Crop_Meta":        ["Crop_Stage", "Crop_Cycle"],
        "Image_Quality":    ["Cloud_Cover", "Valid_Pixel_Count"],
    }
