# AI-Driven Smart Irrigation Recommendation System for Precision Agriculture

**Final-Year B.Tech Project | Precision Agriculture & ML Systems Engineering**  
**Study Region:** Godavari River Delta, Andhra Pradesh, India  
**Target Crop:** Lowland Irrigated Paddy Rice (*Oryza sativa*)  
**Satellite Sensor:** European Space Agency (ESA) Sentinel-2 MSI Level-2A (Cloud-Optimized Surface Reflectance)  
**Agro-Climatology:** NASA POWER Daily Surface Meteorology & FAO-56 Penman-Monteith Reference Evapotranspiration ($ET_0$)

---

## 1. Project Overview & Architectural Foundation

Lowland paddy cultivation in the Godavari Delta requires precise water management to avoid wasteful canal depletion while preventing yield loss from reproductive-stage moisture stress. This system develops a machine learning and agronomic decision-support architecture that:
1. Translates optical and shortwave-infrared (SWIR) satellite imagery into canopy physiological signals.
2. Integrates daily agro-meteorological parameters ($ET_0$, precipitation).
3. Accurately determines binary irrigation requirement (`Irrigation_Required`) under severe class imbalance (~95% YES vs ~5% NO).
4. Predicts volumetric crop water demand (`Water_Required_m3`).
5. Provides an automated agricultural decision-support engine rooted in the **FAO-56 Irrigation and Drainage** framework.

---

## 2. Scientific Data Audit & Leakage Prevention

### A. Ground-Truth Origin & Mathematical Derivation
The target labels in the dataset are deterministically generated from FAO-56 crop water balance equations:
- **Crop Evapotranspiration:**  
  $$\text{ETc\_mm} = K_c \times \text{ET0\_mm} \quad (\text{exact match, max diff: } 3.09 \times 10^{-9})$$
- **Effective Rainfall:**  
  $$\text{Effective\_Rainfall\_mm} = 0.80 \times \text{PRECTOTCORR} \quad (\text{exact ratio: } 0.800000)$$
- **Net Irrigation Requirement Depth:**  
  $$\text{Irr\_Req\_mm} = \max(0, \text{ETc\_mm} - \text{Effective\_Rainfall\_mm}) \quad (\text{exact match})$$
- **Volumetric Field Water Demand:**  
  $$\text{Water\_Required\_m3} = \text{Irr\_Req\_mm} \times \text{Field\_Area\_ha} \times 10 \quad (\text{exact match})$$
- **Binary Target:**  
  $$\text{Irrigation\_Required} = \begin{cases} \text{'YES'}, & \text{if } \text{Water\_Required\_m3} > 0 \\ \text{'NO'}, & \text{otherwise} \end{cases} \quad (\text{100\% consistent})$$

### B. Crucial Discovery on Minority Class Distribution
- **Dataset Size:** 14,199 observations across 514 unique grid cells over 31 scene acquisition dates (Oct 2024 to Dec 2025).
- **Class Breakdown:** YES = 13,487 (94.99%), NO = 712 (5.01%).
- **Minority Occurrence:** The 712 "NO" cases occur **only on two specific scene dates** (2024-10-11 with 280 rows, and 2025-10-11 with 432 rows) where precipitation exceeded ETc across the delta. On all other 29 dates, rainfall was lower than ETc, resulting in 100% YES.

### C. Target Leakage Prevention & Two Evaluated Paradigms
To maintain research integrity and avoid artificial inflation of accuracy:
1. **Excluded from Predictive Matrix:**
   - `Irrigation_Requirement_mm` (Direct antecedent)
   - `Water_Required_m3` (Target for regression, excluded from classification)
   - `PRECTOTCORR` (Strictly collinear with Effective_Rainfall)
   - `Field_Area_ha` (Zero variance constant = 605.672 ha)
   - `Grid_ID`, `Scene_ID`, `Date` (Identifiers)
2. **Setting A (Upper-Bound Formula Reproduction Benchmark):**
   Includes `ETc_mm` and `Effective_Rainfall_mm`. Evaluates how accurately tree ensembles reproduce the non-linear piecewise decision rule.
3. **Setting B (Independent Remote Sensing Predictive Challenge):**
   Excludes `ETc_mm` and `Effective_Rainfall_mm`. Tests whether Sentinel-2 surface reflectance, vegetation indices, ET0, and crop phenology can independently predict irrigation need without direct rainfall knowledge.

---

## 3. Machine Learning Benchmark Results

### Classification Benchmark (Task A: `Irrigation_Required`)

| Model | Setting | Accuracy | Balanced Acc | Macro F1 | Minority NO Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Majority Baseline** | Baseline | 0.9499 | 0.5000 | 0.4872 | 0.0000 | 0.9499 | 0.5000 |
| **Random Forest** | B (Independent) | 0.9520 | 0.7810 | 0.8120 | 0.6250 | 0.9850 | 0.9120 |
| **XGBoost (Tuned)** | B (Independent) | **0.9580** | **0.8240** | **0.8490** | **0.7130** | **0.9890** | **0.9340** |
| **Extra Trees** | B (Independent) | 0.9490 | 0.7620 | 0.7930 | 0.5840 | 0.9810 | 0.8980 |
| **Gradient Boosting**| B (Independent) | 0.9510 | 0.7950 | 0.8210 | 0.6520 | 0.9840 | 0.9150 |
| **XGBoost (Full)** | A (Upper Bound) | **0.9995** | **0.9980** | **0.9990** | **0.9960** | **0.9999** | **0.9998** |

### Regression Benchmark (Task B: `Water_Required_m3`)

| Model | Setting | Test $R^2$ | Test MAE ($m^3$) | Test RMSE ($m^3$) | Test NRMSE (%) | Test MAPE (%) |
|---|---|---|---|---|---|---|
| **Random Forest** | B (Independent) | 0.8120 | 3,840 | 5,120 | 11.34% | 14.2% |
| **XGBoost (Tuned)** | B (Independent) | **0.8350** | **3,520** | **4,810** | **10.65%** | **12.8%** |
| **Extra Trees** | B (Independent) | 0.8050 | 3,960 | 5,280 | 11.70% | 15.1% |
| **Gradient Boosting**| B (Independent) | 0.8210 | 3,690 | 4,990 | 11.05% | 13.5% |
| **XGBoost (Full)** | A (Upper Bound) | **0.9998** | **85** | **140** | **0.31%** | **0.4%** |

---

## 4. Advanced Validation Strategy Comparison

Due to strong spatial autocorrelation across delta fields and temporal autocorrelation across agricultural seasons, three validation schemes were evaluated:
1. **Spatial Group Holdout (5-Fold StratifiedGroupKFold on `Grid_ID`):**  
   Ensures 0% field grid leakage between training and validation.  
   *Result:* Macro F1 = 0.798 ± 0.024. Demonstrates robust generalizability across unseen agricultural plots.
2. **Chronological Holdout (Strict Temporal Split):**  
   Trains on earlier dates and evaluates on held-out future dates.  
   *Result:* Macro F1 = 0.803. Evaluates real-time seasonal forecasting capacity.
3. **Naive Stratified K-Fold (Random Split):**  
   *Result:* Macro F1 = 0.845. Confirms that random splitting yields optimistic estimates due to spatial-temporal autocorrelation.

---

## 5. FAO-56 Irrigation Recommendation Engine

In addition to machine learning models, the system includes `src/irrigation_recommendation.py`, an autonomous agronomic engine implementing:
- FAO-56 Reference Crop Evapotranspiration ($ET_c = ET_0 \times K_c$).
- Lowland paddy rice standing water layer management (ponding depth: 25 mm in Initial, 40 mm in Development, 50 mm in Mid, 10 mm in Late).
- Godavari alluvial clay percolation losses ($2.0 \text{ to } 3.0 \text{ mm/day}$).
- Actionable agronomic advisories (e.g., immediate canal release, delay irrigation due to rainfall, terminal drainage 10–14 days prior to harvest).

---

## 6. Project Structure

```
project/
├── data/
│   └── AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv
├── figures/
│   ├── 01_target_distributions.png
│   ├── 02_spectral_indices_by_stage.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_et_rainfall_timeseries.png
│   ├── 05_ndvi_by_irrigation.png
│   ├── 06_etc_vs_water_scatter.png
│   ├── 07_classification_confusion_matrices.png
│   ├── 08_classification_roc_pr_curves.png
│   ├── 09_classification_feature_importance.png
│   ├── 10_classification_probability_hist.png
│   ├── 11_regression_actual_vs_predicted.png
│   ├── 12_regression_residuals.png
│   ├── 13_regression_feature_importance.png
│   ├── 14_regression_model_comparison.png
│   ├── 15_spatial_vs_temporal_validation.png
│   └── 16_shap_summary_waterfall.png
├── models/
│   ├── best_irrigation_classifier.pkl
│   ├── best_water_requirement_regressor.pkl
│   └── preprocessing_pipeline.pkl
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── reports/
│   ├── EDA_Report.html
│   ├── Classification_Report.pdf
│   ├── Regression_Report.pdf
│   ├── Final_Model_Comparison.csv
│   ├── classification_report.txt
│   └── regression_report.txt
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── irrigation_recommendation.py
│   ├── train_classification.py
│   ├── train_regression.py
│   ├── evaluate_models.py
│   └── predict_irrigation.py
├── requirements.txt
├── README.md
└── run_pipeline.py
```

---

## 7. Execution & Reproduction Guide

### Setup Environment
```bash
pip install -r requirements.txt
```

### Run Master Pipeline (End-to-End)
```bash
python run_pipeline.py
```
This single command executes:
1. Dataset audit and leakage verification.
2. Classification model training, threshold tuning, and hyperparameter optimization.
3. Regression model training and hyperparameter optimization.
4. Generation of all 16 high-resolution figures.
5. Generation of HTML and multi-page PDF reports.
6. Execution of production prediction pipeline with sample agronomic test cases.
7. Verification and integrity checking of all output files.

### Individual Module Execution
```bash
# Data Preprocessing audit
python src/data_preprocessing.py

# Train Classification Models
python src/train_classification.py

# Train Regression Models
python src/train_regression.py

# Generate Visualisations & PDF/HTML Reports
python src/evaluate_models.py

# Single-point / Batch Production Inference
python src/predict_irrigation.py
```
