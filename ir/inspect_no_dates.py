import pandas as pd
import numpy as np

df = pd.read_csv("AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv")
df["Date_dt"] = pd.to_datetime(df["Date"], dayfirst=True)

no_dates = df[df["Irrigation_Required"] == "NO"]["Date_dt"].unique()
print("NO dates:", [d.strftime('%Y-%m-%d') for d in no_dates])

# Look at date by date stats for each date
date_stats = df.groupby("Date_dt").agg(
    n=("Grid_ID", "count"),
    target_no=("Irrigation_Required", lambda s: (s == "NO").sum()),
    rain=("Effective_Rainfall_mm", "mean"),
    prec=("PRECTOTCORR", "mean"),
    et0=("ET0_mm", "mean"),
    etc=("ETc_mm", "mean"),
    kc=("Kc", "mean"),
    ndvi=("NDVI", "mean"),
    ndwi=("NDWI", "mean"),
    mndwi=("MNDWI", "mean"),
    ndmi=("NDMI", "mean"),
    b02=("B02_Blue", "mean"),
    b04=("B04_Red", "mean"),
    b08=("B08_NIR", "mean"),
    b11=("B11_SWIR1", "mean"),
    b12=("B12_SWIR2", "mean"),
    cloud=("Cloud_Cover", "mean"),
).reset_index()

print("\n--- All 31 dates summary ---")
print(date_stats[["Date_dt", "n", "target_no", "rain", "et0", "etc", "cloud", "ndvi", "ndwi", "ndmi"]].to_string(index=False))

print("\n--- Compare NO rows vs YES rows across all features ---")
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ["Grid_ID", "Valid_Pixel_Count", "Water_Required_m3", "Irrigation_Requirement_mm"]]
comp = []
for c in num_cols:
    no_mean = df[df["Irrigation_Required"] == "NO"][c].mean()
    no_std = df[df["Irrigation_Required"] == "NO"][c].std()
    yes_mean = df[df["Irrigation_Required"] == "YES"][c].mean()
    yes_std = df[df["Irrigation_Required"] == "YES"][c].std()
    comp.append({"Feature": c, "NO_mean": no_mean, "NO_std": no_std, "YES_mean": yes_mean, "YES_std": yes_std, "Diff": no_mean - yes_mean})

comp_df = pd.DataFrame(comp)
print(comp_df.sort_values(by="Diff", key=abs, ascending=False).to_string(index=False))
