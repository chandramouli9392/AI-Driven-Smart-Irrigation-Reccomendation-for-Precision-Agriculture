import pandas as pd
import numpy as np

csv_path = r"IRRIIIIIIIIIIIIIIIIIII/AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv"
df = pd.read_csv(csv_path)

print("=== DATASET SUMMARY ===")
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(f"Total missing values: {df.isnull().sum().sum()}")
nulls = df.isnull().sum()
if nulls.sum() > 0:
    print(f"Columns with nulls:\n{nulls[nulls > 0]}")
else:
    print("Zero missing values across all columns.")
print(f"Duplicate rows: {df.duplicated().sum()}")

print("\n=== TEMPORAL & SPATIAL ===")
df["Date_dt"] = pd.to_datetime(df["Date"], dayfirst=True)
print(f"Date range: {df['Date_dt'].min().strftime('%Y-%m-%d')} to {df['Date_dt'].max().strftime('%Y-%m-%d')}")
print(f"Unique dates: {df['Date'].nunique()}")
print(f"Unique Grid_IDs: {df['Grid_ID'].nunique()}")
if "Scene_ID" in df.columns:
    print(f"Unique Scene_IDs: {df['Scene_ID'].nunique()}")

print("\n=== TARGETS & FORMULA CHECKS ===")
print("Irrigation_Required value counts:")
print(df["Irrigation_Required"].value_counts(dropna=False))
print(df["Irrigation_Required"].value_counts(normalize=True).mul(100).round(2))

if "Kc" in df.columns and "ET0_mm" in df.columns and "ETc_mm" in df.columns:
    diff_etc = (df["Kc"] * df["ET0_mm"] - df["ETc_mm"]).abs()
    print(f"ETc_mm == Kc * ET0_mm max abs diff: {diff_etc.max():.6e}, mean: {diff_etc.mean():.6e}")

if "PRECTOTCORR" in df.columns and "Effective_Rainfall_mm" in df.columns:
    diff_rain = (df["PRECTOTCORR"] * 0.8 - df["Effective_Rainfall_mm"]).abs()
    print(f"Eff_Rain == 0.8 * PRECTOTCORR max abs diff: {diff_rain.max():.6e}")

if "ETc_mm" in df.columns and "Effective_Rainfall_mm" in df.columns:
    calc_req = (df["ETc_mm"] - df["Effective_Rainfall_mm"]).clip(lower=0)
    if "Irrigation_Requirement_mm" in df.columns:
        diff_req = (calc_req - df["Irrigation_Requirement_mm"]).abs()
        print(f"Irr_Req_mm == max(0, ETc - Rain) max abs diff: {diff_req.max():.6e}")
    if "Field_Area_ha" in df.columns and "Water_Required_m3" in df.columns:
        calc_vol = calc_req * df["Field_Area_ha"] * 10
        diff_vol = (calc_vol - df["Water_Required_m3"]).abs()
        print(f"Water_m3 == Req_mm * Area * 10 max abs diff: {diff_vol.max():.6e}")
    match_cls = ((df["Irrigation_Required"] == "YES") == (df["Water_Required_m3"] > 0)).all()
    print(f"Irrigation_Required == (Water_Required_m3 > 0): {match_cls}")

print("\n=== SENSOR & FIELD ATTRIBUTES ===")
if "Field_Area_ha" in df.columns:
    print(f"Field_Area_ha min: {df['Field_Area_ha'].min()}, max: {df['Field_Area_ha'].max()}, std: {df['Field_Area_ha'].std()}")
print("Crop_Stage counts:", df["Crop_Stage"].value_counts().to_dict() if "Crop_Stage" in df.columns else "N/A")
print("Crop_Cycle counts:", df["Crop_Cycle"].value_counts().to_dict() if "Crop_Cycle" in df.columns else "N/A")
print("Crop_Progress summary:", df["Crop_Progress"].describe().to_dict() if "Crop_Progress" in df.columns else "N/A")

print("\n=== SENSOR/SATELLITE CHECK ===")
# Check if thermal or landsat columns exist
thermal_cols = [c for c in df.columns if any(k in c.lower() for k in ["therm", "b10", "lst", "temp", "st_"])]
print(f"Thermal/LST columns found: {thermal_cols}")
landsat_cols = [c for c in df.columns if "landsat" in c.lower()]
print(f"Landsat columns found: {landsat_cols}")
