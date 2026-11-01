"""
run_pipeline.py -- Master script for the AI-Driven Smart Irrigation System.
Execute from the project root: python run_pipeline.py
"""
import sys, os, time

# Reconfigure stdout/stderr ONCE here; modules must NOT do this themselves.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "src")

CSV_PATH = "AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"

print("=" * 70)
print("  AI-DRIVEN SMART IRRIGATION RECOMMENDATION SYSTEM")
print("  Precision Agriculture - Paddy Fields - Sentinel-2")
print("=" * 70)

# ---------------------------------------------------------------------------
# PHASE 1: EDA
# ---------------------------------------------------------------------------
print("\n[PIPELINE] PHASE 1: Exploratory Data Analysis")
import pandas as pd, numpy as np
df = pd.read_csv(CSV_PATH)
print(f"  Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicate rows: {df.duplicated().sum()}")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  Grid cells (unique Grid_IDs): {df['Grid_ID'].nunique()}")
print(f"  Unique scene dates: {df['Date'].nunique()}")

vc = df["Irrigation_Required"].value_counts()
print(f"\n  -- Target: Irrigation_Required --")
print(f"  YES: {vc.get('YES',0):,} ({vc.get('YES',0)/len(df)*100:.2f}%)")
print(f"  NO : {vc.get('NO',0):,}  ({vc.get('NO',0)/len(df)*100:.2f}%)")

print(f"\n  -- Target: Water_Required_m3 --")
print(f"  Min: {df['Water_Required_m3'].min():,.0f} m3  Max: {df['Water_Required_m3'].max():,.0f} m3  Mean: {df['Water_Required_m3'].mean():,.0f} m3")

print(f"\n  -- DATA LEAKAGE ANALYSIS --")
print(f"  ETc_mm = Kc x ET0_mm? Max abs diff: {(df['Kc']*df['ET0_mm']-df['ETc_mm']).abs().max():.2e}")
net_irr  = (df['ETc_mm'] - df['Effective_Rainfall_mm']).clip(lower=0)
vol_calc = net_irr * df['Field_Area_ha'] * 10
print(f"  Water_Required_m3 ~ clip(ETc-Rain,0)*Area*10? Max diff: {(vol_calc-df['Water_Required_m3']).abs().max():.2e}")
print(f"  Irrigation_Required == (Water>0)? All match: {((df['Irrigation_Required']=='YES')==(df['Water_Required_m3']>0)).all()}")
rr = df[df['PRECTOTCORR']>0]['Effective_Rainfall_mm'] / df[df['PRECTOTCORR']>0]['PRECTOTCORR']
print(f"  Effective_Rainfall = 0.8 x PRECTOTCORR? Unique ratio: {rr.unique()}")
print(f"  Field_Area_ha constant: {df['Field_Area_ha'].std():.2e} std")

print("\n  -- CONCLUSIONS --")
for line in [
    "ETc_mm = Kc x ET0_mm                            [EXACT FORMULA]",
    "Eff_Rain_mm = 0.8 x PRECTOTCORR                 [EXACT FORMULA]",
    "Irr_Req_mm  = max(0, ETc - Eff_Rain)            [EXACT FORMULA]",
    "Water_m3    = Irr_Req_mm x 605.672 x 10         [EXACT FORMULA]",
    "Irr_Req'd   = YES if Water_m3 > 0               [DETERMINISTIC RULE]",
    "Field_Area_ha = 605.672 (CONSTANT) => EXCLUDED",
    "PRECTOTCORR = Eff_Rain/0.8        => EXCLUDED",
    "Irrigation_Requirement_mm (target precursor)     => EXCLUDED",
    "Models learn FAO-56 rule from satellite+weather inputs.",
]:
    print(f"  [!] {line}")

# ---------------------------------------------------------------------------
# PHASES 2-5: Classification Training
# ---------------------------------------------------------------------------
print("\n[PIPELINE] PHASES 2-5: Classification Training")
t0 = time.time()
from train_classification import train_classifiers
cls_results = train_classifiers()
print(f"  Classification done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# PHASES 2-5: Regression Training
# ---------------------------------------------------------------------------
print("\n[PIPELINE] PHASES 2-5: Regression Training")
t0 = time.time()
from train_regression import train_regressors
reg_results = train_regressors()
print(f"  Regression done in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# PHASE 6: Visualisations
# ---------------------------------------------------------------------------
print("\n[PIPELINE] PHASE 6: Generating Visualisations")
import matplotlib
matplotlib.use("Agg")
from evaluate_models import plot_eda, plot_classification, plot_regression
import pickle

plot_eda(df)

with open("models/best_irrigation_classifier.pkl", "rb") as f:
    cls_pkl = pickle.load(f)
with open("models/best_water_requirement_regressor.pkl", "rb") as f:
    reg_pkl = pickle.load(f)

fi_cls = pd.read_csv("reports/classification_feature_importance.csv")
fi_reg = pd.read_csv("reports/regression_feature_importance.csv")

X_te   = cls_results["X_test"]
excl_B = ["Effective_Rainfall_mm", "ETc_mm", "ET0_Rain_Ratio"]
B_cols = [c for c in X_te.columns if c not in excl_B]

plot_classification({
    "best_model":         cls_pkl["model"],
    "best_name":          cls_pkl["tuned_model_name"],
    "X_test":             X_te,
    "y_cls_test":         cls_results["y_cls_test"],
    "feature_importance": fi_cls,
})
plot_regression({
    "best_model_B":       reg_pkl["model_B"],
    "best_name_B":        reg_pkl["best_name_B"],
    "X_test_B":           X_te[B_cols],
    "y_reg_test":         reg_results["y_reg_test"],
    "feature_importance": fi_reg,
    "comparison":         pd.read_csv("reports/regression_comparison.csv"),
})

# ---------------------------------------------------------------------------
# PHASE 8: Demo Prediction
# ---------------------------------------------------------------------------
print("\n[PIPELINE] PHASE 8: Demo Prediction")
from predict_irrigation import load_models, predict_single
cls_b, reg_b = load_models()
sample = {
    "B02_Blue":0.08,"B03_Green":0.10,"B04_Red":0.09,
    "B08_NIR":0.45,"B11_SWIR1":0.28,"B12_SWIR2":0.18,
    "NDVI":0.66,"NDWI":-0.22,"MNDWI":-0.30,"NDMI":0.12,"EVI":0.38,
    "ET0_mm":5.8,"ETc_mm":6.96,"Kc":1.2,
    "Crop_Progress":0.55,"Effective_Rainfall_mm":0.40,
    "Crop_Stage":"Mid","Crop_Cycle":"2",
    "Cloud_Cover":15.0,"Valid_Pixel_Count":400,
}
res = predict_single(sample, cls_b, reg_b)
print(f"  Irrigation Required : {res['irrigation_required']}")
print(f"  Confidence (P=YES)  : {res['confidence']:.4f}")
print(f"  Water Required (m3) : {res['water_required_m3']:,.0f}")
print(f"  Warnings            : {res['warnings'] or 'None'}")
print(f"  Top driving features: {res['top_features']}")

# ---------------------------------------------------------------------------
# PHASE 9: Save preprocessing bundle & consolidated report
# ---------------------------------------------------------------------------
print("\n[PIPELINE] PHASE 9: Saving artefacts")
from data_preprocessing import load_and_preprocess
data_meta = load_and_preprocess(CSV_PATH)
with open("models/preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump({
        "cat_encoder":    data_meta["cat_encoder"],
        "feature_names":  data_meta["feature_names"],
        "label_map":      data_meta["label_map"],
    }, f)
print("  Saved: models/preprocessing_pipeline.pkl")

cls_comp = pd.read_csv("reports/classification_comparison.csv")
reg_comp = pd.read_csv("reports/regression_comparison.csv")
cls_comp.insert(0, "Task", "Classification")
reg_comp.insert(0, "Task", "Regression")
pd.concat([cls_comp, reg_comp], ignore_index=True).to_csv("reports/model_comparison.csv", index=False)
print("  Saved: reports/model_comparison.csv")

print("\n[PIPELINE] ALL PHASES COMPLETE")
print("  models/  -> classifier + regressor + preprocessing pkl")
print("  reports/ -> classification_report.txt, regression_report.txt, model_comparison.csv")
print("  figures/ -> 14 PNG visualisation files")
