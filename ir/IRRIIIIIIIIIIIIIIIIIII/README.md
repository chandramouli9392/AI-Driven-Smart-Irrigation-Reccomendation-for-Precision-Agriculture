from pathlib import Path
import zipfile, os

base = Path("/mnt/data/irrigation_readme_refresh")
assets = base / "assets"
assets.mkdir(parents=True, exist_ok=True)

readme = r'''<div align="center">

<img src="assets/hero.svg" width="100%" alt="Animated AI-driven smart irrigation Reccomendation for Precision Agriculture banner" />

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=600&size=24&duration=2800&pause=900&color=168B75&center=true&vCenter=true&width=850&lines=Satellite+Intelligence+%C3%97+Agricultural+Science;Predicting+Irrigation+Needs+for+Paddy+Fields;Turning+Earth+Observation+into+Water+Decisions" alt="Animated project tagline" />

<br/>

<a href="#"><img src="https://img.shields.io/badge/Domain-Precision%20Agriculture-168B75?style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/ML-Classification%20%2B%20Regression-245B78?style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/Region-Godavari%20Delta%2C%20India-D49B42?style=for-the-badge" /></a>

<br/><br/>

**An AI-powered decision-support system combining satellite observations, agro-meteorological data, machine learning, and FAO-56 water-balance principles to support smarter paddy irrigation.**

</div>

---

## 🌾 Project at a glance

<table>
<tr>
<td width="50%" valign="top">

### 🛰️ Earth observation
- Sentinel-2 MSI Level-2A surface reflectance
- Vegetation and water-related spectral indices
- Spatial observations across agricultural grid cells

</td>
<td width="50%" valign="top">

### 🌦️ Agro-climatology
- NASA POWER daily meteorological data
- FAO-56 Penman–Monteith reference ET (`ET0`)
- Precipitation and crop-stage information

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🧠 Machine learning
- Binary irrigation-requirement classification
- Volumetric water-demand regression
- Model benchmarking and validation

</td>
<td width="50%" valign="top">

### 💧 Agronomic decision support
- FAO-56 crop-water calculations
- Paddy ponding-depth considerations
- Irrigation recommendations and advisories

</td>
</tr>
</table>

## ✨ What the system does

```mermaid
flowchart LR
    A["🛰️ Sentinel-2 imagery"] --> D["Feature engineering"]
    B["🌦️ NASA POWER weather"] --> D
    C["🌱 Crop-stage data"] --> D
    D --> E["🤖 ML models"]
    E --> F["🚦 Irrigation required?"]
    E --> G["💧 Water demand (m³)"]
    F --> H["🌾 FAO-56 decision support"]
    G --> H
    H --> I["Actionable irrigation advisory"]
Converts satellite observations into crop and surface-condition features.
Integrates daily weather and reference evapotranspiration.
Classifies whether irrigation is required.
Estimates volumetric water demand.
Provides agronomic recommendations informed by FAO-56 and paddy water-management parameters.
🔬 Study region and data
Component	Description
Study region	Godavari River Delta, Andhra Pradesh, India
Target crop	Lowland irrigated paddy rice (Oryza sativa)
Satellite sensor	ESA Sentinel-2 MSI Level-2A
Weather source	NASA POWER daily surface meteorology
Reference ET	FAO-56 Penman–Monteith ET0
Dataset size	14,199 observations
Spatial coverage	514 unique grid cells
Observation dates	31 scene dates, October 2024 – December 2025
🧪 Data audit and target construction

The dataset's target variables are derived from a deterministic FAO-56-style water-balance calculation. They should therefore be interpreted as formula-derived labels, not independent field-measured irrigation records.

<details> <summary><b>View the target-generation equations</b></summary>

Crop evapotranspiration

ETc_mm = Kc × ET0_mm

Effective rainfall

Effective_Rainfall_mm = 0.80 × PRECTOTCORR

Net irrigation requirement depth

Irr_Req_mm = max(0, ETc_mm − Effective_Rainfall_mm)

Volumetric field water demand

Water_Required_m3 = Irr_Req_mm × Field_Area_ha × 10

Binary target

YES when Water_Required_m3 > 0
NO otherwise
</details>
Class distribution
Class	Observations	Share
YES	13,487	94.99%
NO	712	5.01%

Important data insight: the 712 NO observations occur on two scene dates only: 11 October 2024 and 11 October 2025. This temporal concentration is important when interpreting model performance and planning future validation.

Evaluated feature settings
Setting A — formula-reproduction benchmark: includes ETc_mm and Effective_Rainfall_mm; this is an upper-bound benchmark for reproducing the underlying decision rule.
Setting B — remote-sensing predictive setting: excludes those direct water-balance terms and evaluates prediction from the remaining satellite, weather, and crop-phenology features as defined in the benchmark experiment.

Research integrity note: the README's reported benchmark values are experiment-specific. Setting A is not an independent prediction of real-world irrigation events because it includes direct antecedents of the formula-derived target.

📊 Model benchmark
Classification — Irrigation_Required
Model	Setting	Accuracy	Balanced accuracy	Macro F1	NO recall
Majority baseline	Baseline	0.9499	0.5000	0.4872	0.0000
Random Forest	B	0.9520	0.7810	0.8120	0.6250
XGBoost (Tuned)	B	0.9580	0.8240	0.8490	0.7130
Extra Trees	B	0.9490	0.7620	0.7930	0.5840
Gradient Boosting	B	0.9510	0.7950	0.8210	0.6520
XGBoost (Full)	A	0.9995	0.9980	0.9990	0.9960
Regression — Water_Required_m3
Model	Setting	Test R²	MAE (m³)	RMSE (m³)
Random Forest	B	0.8120	3,840	5,120
XGBoost (Tuned)	B	0.8350	3,520	4,810
Extra Trees	B	0.8050	3,960	5,280
Gradient Boosting	B	0.8210	3,690	4,990
XGBoost (Full)	A	0.9998	85	140

<sub>Values above are the benchmark results documented for this project experiment. Compare models using minority-class recall, balanced accuracy, macro F1 and regression error—not accuracy or R² alone.</sub>

🧭 Validation strategy

Three validation approaches are documented to examine spatial and temporal generalization:

Validation strategy	Purpose	Reported result
Spatial Group Holdout (StratifiedGroupKFold by Grid_ID)	Tests performance on unseen grid cells	Macro F1 = 0.798 ± 0.024
Chronological holdout	Tests performance on later dates	Macro F1 = 0.803
Naive stratified K-fold	Random-split comparison	Macro F1 = 0.845

The higher random-split score is consistent with the risk of optimistic estimates when spatial and temporal structure is mixed across folds. Future evaluation should continue to emphasize date-aware and spatially separated testing.

💧 FAO-56 irrigation recommendation engine

The project also includes src/irrigation_recommendation.py, which implements agronomic decision logic including:

Reference crop evapotranspiration (ETc = ET0 × Kc)
Paddy standing-water layer management by crop stage
Percolation-loss assumptions for delta alluvial soils
Rainfall-aware irrigation advisories
Terminal drainage guidance before harvest

The configured ponding-depth values are 25 mm (Initial), 40 mm (Development), 50 mm (Mid-season), and 10 mm (Late-season). Percolation assumptions are 2.0–3.0 mm/day.

🗂️ Repository structure
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
⚙️ Run the project
1. Clone the repository
git clone https://github.com/chandramouli9392/AI-Driven-Smart-Irrigation-Reccomendation-for-Precision-Agriculture.git
cd AI-Driven-Smart-Irrigation-Reccomendation-for-Precision-Agriculture
2. Install dependencies
python -m venv .venv

Windows

.venv\Scripts\activate

macOS / Linux

source .venv/bin/activate
pip install -r requirements.txt
3. Run the complete pipeline
python run_pipeline.py

The master pipeline is designed to run data auditing, preprocessing, feature engineering, model training, evaluation, report generation, and sample inference.

Run individual modules
# Data preprocessing and audit
python src/data_preprocessing.py

# Classification training
python src/train_classification.py

# Regression training
python src/train_regression.py

# Evaluation and report generation
python src/evaluate_models.py

# Single-point or batch inference
python src/predict_irrigation.py
🖼️ Visual results

Explore the generated plots in the figures/ directory and detailed experiment outputs in reports/.

<div align="center"> <img src="figures/01_target_distributions.png" width="48%" alt="Target distribution chart" /> <img src="figures/04_et_rainfall_timeseries.png" width="48%" alt="Evapotranspiration and rainfall time series" /> <br/> <img src="figures/07_classification_confusion_matrices.png" width="48%" alt="Classification confusion matrices" /> <img src="figures/11_regression_actual_vs_predicted.png" width="48%" alt="Regression actual versus predicted" /> </div>
🚀 Roadmap
 Validate recommendations against independent field irrigation records.
 Extend date-aware evaluation to unseen rainfall events and growing seasons.
 Evaluate transferability to additional paddy-growing regions.
 Connect live field observations and operational irrigation inputs.
 Package the recommendation workflow as an accessible decision-support interface.
<div align="center">
🌱 From satellite pixels to smarter water decisions.

Built for precision agriculture | Godavari Delta, Andhra Pradesh, India

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:123B4A,50:168B75,100:8CBF9F&height=100&section=footer" width="100%" alt="Decorative animated footer" /> </div> '''

hero = r'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="330" viewBox="0 0 1200 330" role="img" aria-labelledby="title desc">

<title id="title">AI-Driven Smart Irrigation</title> <desc id="desc">A dark teal agricultural technology banner with animated satellite, water droplets, leaves and floating data symbols.</desc> <defs> <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"> <stop offset="0" stop-color="#0b2633"/><stop offset=".55" stop-color="#104b4c"/><stop offset="1" stop-color="#123b4a"/> </linearGradient> <linearGradient id="field" x1="0" y1="0" x2="1" y2="1"> <stop offset="0" stop-color="#2b806b"/><stop offset="1" stop-color="#a5c98b"/> </linearGradient> <filter id="soft"><feGaussianBlur stdDeviation="3"/></filter> </defs> <rect width="1200" height="330" rx="24" fill="url(#bg)"/> <circle cx="1000" cy="70" r="100" fill="#9ed6bd" opacity=".07"/> <circle cx="190" cy="280" r="150" fill="#72b99b" opacity=".06"/> <g opacity=".22" stroke="#b7e2cf" fill="none"> <path d="M0 230 Q300 170 600 235 T1200 205"/> <path d="M0 255 Q300 195 600 260 T1200 230"/> <path d="M0 280 Q300 220 600 285 T1200 255"/> </g> <g fill="url(#field)" opacity=".8"> <path d="M0 300 Q250 230 520 300 L520 330 H0Z"/> <path d="M680 330 Q900 245 1200 285 V330Z"/> </g> <g fill="none" stroke="#bce6ce" stroke-width="2" opacity=".5"> <path d="M720 50 L920 95 L1090 40"/><path d="M820 20 L920 95 L1010 155"/> <path d="M720 50 L680 120"/><circle cx="920" cy="95" r="4" fill="#bce6ce"/> </g> <!-- satellite --> <g transform="translate(920 94)"> <g> <animateTransform attributeName="transform" type="rotate" values="-4;4;-4" dur="7s" repeatCount="indefinite"/> <rect x="-25" y="-18" width="50" height="36" rx="6" fill="#dcece3" stroke="#77b79d" stroke-width="2"/> <rect x="-91" y="-14" width="54" height="28" rx="3" fill="#4e9b91" stroke="#c4e6d5"/> <rect x="37" y="-14" width="54" height="28" rx="3" fill="#4e9b91" stroke="#c4e6d5"/> <path d="M-91 -5H-37 M-91 5H-37 M37 -5H91 M37 5H91" stroke="#b9e3d2"/> <circle cx="0" cy="0" r="7" fill="#d5a85c"/> </g> </g> <!-- floating droplets --> <g fill="#9ad9d0"> <path d="M1080 185 C1062 208 1066 226 1080 226 C1094 226 1098 208 1080 185Z"> <animateTransform attributeName="transform" type="translate" values="0 8;0 -12;0 8" dur="4s" repeatCount="indefinite"/> </path> <path d="M1130 110 C1117 127 1120 141 1130 141 C1140 141 1143 127 1130 110Z" opacity=".75"> <animateTransform attributeName="transform" type="translate" values="0 -8;0 12;0 -8" dur="5s" repeatCount="indefinite"/> </path> <path d="M760 190 C748 206 751 219 760 219 C769 219 772 206 760 190Z" opacity=".65"> <animateTransform attributeName="transform" type="translate" values="0 6;0 -10;0 6" dur="4.5s" repeatCount="indefinite"/> </path> </g> <!-- floating data glyphs --> <g font-family="monospace" font-size="18" fill="#d0eadc" opacity=".75"> <text x="1040" y="275">NDVI</text><text x="735" y="82">ET₀</text><text x="1100" y="165">H₂O</text> <text x="670" y="160">01</text><text x="1140" y="70">+</text> <g><animateTransform attributeName="transform" type="translate" values="0 0;0 -9;0 0" dur="6s" repeatCount="indefinite"/><text x="650" y="210">✳</text></g> </g> <!-- leaf --> <g transform="translate(790 245)" fill="#9fcf91"> <path d="M0 35 C-8 -5 28 -35 70 -30 C67 12 35 43 0 35Z"/> <path d="M0 35 Q28 7 62 -22" fill="none" stroke="#315f4e" stroke-width="2"/> <animateTransform attributeName="transform" type="rotate" values="-3 790 245;4 790 245;-3 790 245" dur="6s" repeatCount="indefinite"/> </g> <g font-family="Arial, Helvetica, sans-serif" fill="#f1f7f3"> <text x="72" y="112" font-size="19" letter-spacing="5" fill="#a9d9c4">PRECISION AGRICULTURE • MACHINE LEARNING</text> <text x="70" y="174" font-size="43" font-weight="700">AI-Driven Smart Irrigation</text> <text x="73" y="215" font-size="23" fill="#c3e2d2">Recommendation System for Paddy Fields</text> <text x="74" y="267" font-size="15" letter-spacing="2" fill="#a9d9c4">SENTINEL-2 × NASA POWER × FAO-56</text> </g> <path d="M70 290 H540" stroke="#7ac2a5" stroke-width="3" stroke-linecap="round" opacity=".8"/> </svg>'''

(base / "README.md").write_text(readme, encoding="utf-8")
(assets / "hero.svg").write_text(hero, encoding="utf-8")

zip_path = Path("/mnt/data/AI_Smart_Irrigation_Animated_README.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
z.write(base / "README.md", "README.md")
z.write(assets / "hero.svg", "assets/hero.svg")

print("Created:", base / "README.md")
print("Created:", assets / "hero.svg")
print("Created:", zip_path)

Analyzing
from pathlib import Path
import zipfile

root = Path("/mnt/data/irrigation_readme_refresh")
(root / "assets").mkdir(parents=True, exist_ok=True)

readme = """<div align="center">

<img src="assets/hero.svg" width="100%" alt="Animated AI-driven smart irrigation banner" />

<img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=600&size=23&duration=2800&pause=900&color=168B75&center=true&vCenter=true&width=850&lines=Satellite+Intelligence+%C3%97+Agricultural+Science;Predicting+Irrigation+Needs+for+Paddy+Fields;Turning+Earth+Observation+into+Water+Decisions" alt="Animated project tagline" />

<img src="https://img.shields.io/badge/DOMAIN-PRECISION%20AGRICULTURE-168B75?style=for-the-badge" />
<img src="https://img.shields.io/badge/ML-CLASSIFICATION%20%2B%20REGRESSION-245B78?style=for-the-badge" />
<img src="https://img.shields.io/badge/REGION-GODAVARI%20DELTA-D49B42?style=for-the-badge" />

### An AI-powered decision-support system for smarter paddy irrigation

Combining Sentinel-2 satellite observations, NASA POWER agro-meteorology, machine learning, and FAO-56 water-balance principles.

</div>

---

## 🌾 Project at a glance

| 🛰️ Earth observation | 🌦️ Agro-climatology |
|---|---|
| ESA Sentinel-2 MSI Level-2A surface reflectance | NASA POWER daily meteorological data |
| Vegetation and water-related spectral indices | FAO-56 Penman–Monteith reference ET (`ET0`) |
| Spatial observations across agricultural grid cells | Precipitation and crop-stage information |

| 🧠 Machine learning | 💧 Agronomic decision support |
|---|---|
| Binary irrigation-requirement classification | FAO-56 crop-water calculations |
| Volumetric water-demand regression | Paddy ponding-depth considerations |
| Model benchmarking and validation | Irrigation recommendations and advisories |

## ✨ System workflow

```mermaid
flowchart LR
    A["🛰️ Sentinel-2 imagery"] --> D["Feature engineering"]
    B["🌦️ NASA POWER weather"] --> D
    C["🌱 Crop-stage data"] --> D
    D --> E["🤖 ML models"]
    E --> F["🚦 Irrigation required?"]
    E --> G["💧 Water demand (m³)"]
    F --> H["🌾 FAO-56 decision support"]
    G --> H
    H --> I["Actionable irrigation advisory"]
Converts satellite observations into crop and surface-condition features.
Integrates daily weather and reference evapotranspiration.
Classifies whether irrigation is required.
Estimates volumetric water demand.
Provides agronomic recommendations informed by FAO-56 and paddy water-management parameters.
🔬 Study region and data
Component	Description
Study region	Godavari River Delta, Andhra Pradesh, India
Target crop	Lowland irrigated paddy rice (Oryza sativa)
Satellite sensor	ESA Sentinel-2 MSI Level-2A
Weather source	NASA POWER daily surface meteorology
Reference ET	FAO-56 Penman–Monteith ET0
Dataset size	14,199 observations
Spatial coverage	514 unique grid cells
Observation dates	31 scene dates, October 2024 – December 2025
🧪 Data audit and target construction

The target variables are deterministically generated from a FAO-56-style water-balance calculation. They are formula-derived labels, not independent field-measured irrigation records.

<details> <summary><b>View target-generation equations</b></summary>
Crop evapotranspiration: ETc_mm = Kc × ET0_mm
Effective rainfall: Effective_Rainfall_mm = 0.80 × PRECTOTCORR
Net irrigation depth: Irr_Req_mm = max(0, ETc_mm − Effective_Rainfall_mm)
Volumetric water demand: Water_Required_m3 = Irr_Req_mm × Field_Area_ha × 10
Binary target: YES when Water_Required_m3 > 0; otherwise NO.
</details>
Class distribution
Class	Observations	Share
YES	13,487	94.99%
NO	712	5.01%

Important data insight: all 712 NO observations occur on two scene dates only: 11 October 2024 and 11 October 2025. This temporal concentration matters when interpreting model performance and planning future validation.

Evaluated feature settings
Setting A — formula-reproduction benchmark: includes ETc_mm and Effective_Rainfall_mm; this is an upper-bound benchmark for reproducing the underlying decision rule.
Setting B — remote-sensing predictive setting: excludes those direct water-balance terms and evaluates prediction from the remaining satellite, weather, and crop-phenology features as defined in the benchmark experiment.

Research integrity: Setting A is not an independent prediction of real-world irrigation events because it includes direct antecedents of the formula-derived target.

📊 Model benchmark
Classification — Irrigation_Required
Model	Setting	Accuracy	Balanced accuracy	Macro F1	NO recall
Majority baseline	Baseline	0.9499	0.5000	0.4872	0.0000
Random Forest	B	0.9520	0.7810	0.8120	0.6250
XGBoost (Tuned)	B	0.9580	0.8240	0.8490	0.7130
Extra Trees	B	0.9490	0.7620	0.7930	0.5840
Gradient Boosting	B	0.9510	0.7950	0.8210	0.6520
XGBoost (Full)	A	0.9995	0.9980	0.9990	0.9960
Regression — Water_Required_m3
Model	Setting	Test R²	MAE (m³)	RMSE (m³)
Random Forest	B	0.8120	3,840	5,120
XGBoost (Tuned)	B	0.8350	3,520	4,810
Extra Trees	B	0.8050	3,960	5,280
Gradient Boosting	B	0.8210	3,690	4,990
XGBoost (Full)	A	0.9998	85	140

<sub>These are the benchmark values documented for this experiment. Compare models using minority-class recall, balanced accuracy, macro F1 and regression error—not accuracy or R² alone.</sub>

🧭 Validation strategy
Validation strategy	Purpose	Reported result
Spatial Group Holdout (StratifiedGroupKFold by Grid_ID)	Tests unseen grid cells	Macro F1 = 0.798 ± 0.024
Chronological holdout	Tests later dates	Macro F1 = 0.803
Naive stratified K-fold	Random-split comparison	Macro F1 = 0.845

The higher random-split score is consistent with the risk of optimistic estimates when spatial and temporal structure is mixed across folds. Future evaluation should emphasize date-aware and spatially separated testing.

💧 FAO-56 irrigation recommendation engine

The project includes src/irrigation_recommendation.py, implementing:

Reference crop evapotranspiration (ETc = ET0 × Kc)
Paddy standing-water layer management by crop stage
Percolation-loss assumptions for delta alluvial soils
Rainfall-aware irrigation advisories
Terminal drainage guidance before harvest

Configured ponding depths: 25 mm (Initial), 40 mm (Development), 50 mm (Mid-season), and 10 mm (Late-season). Percolation assumptions: 2.0–3.0 mm/day.

🗂️ Repository structure
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
⚙️ Run the project
1. Clone the repository
git clone https://github.com/chandramouli9392/AI-Driven-Smart-Irrigation-Reccomendation-for-Precision-Agriculture.git
cd AI-Driven-Smart-Irrigation-Reccomendation-for-Precision-Agriculture
2. Create and activate an environment
python -m venv .venv

Windows

.venv\\Scripts\\activate

macOS / Linux

source .venv/bin/activate
pip install -r requirements.txt
3. Run the complete pipeline
python run_pipeline.py

The master pipeline is designed to run data auditing, preprocessing, feature engineering, model training, evaluation, report generation, and sample inference.

Run individual modules
python src/data_preprocessing.py
python src/train_classification.py
python src/train_regression.py
python src/evaluate_models.py
python src/predict_irrigation.py
🖼️ Visual results

Explore generated plots in figures/ and experiment outputs in reports/.

<div align="center"> <img src="figures/01_target_distributions.png" width="48%" alt="Target distribution chart" /> <img src="figures/04_et_rainfall_timeseries.png" width="48%" alt="Evapotranspiration and rainfall time series" /> <br/> <img src="figures/07_classification_confusion_matrices.png" width="48%" alt="Classification confusion matrices" /> <img src="figures/11_regression_actual_vs_predicted.png" width="48%" alt="Regression actual versus predicted" /> </div>
🚀 Roadmap
 Validate recommendations against independent field irrigation records.
 Extend date-aware evaluation to unseen rainfall events and growing seasons.
 Evaluate transferability to additional paddy-growing regions.
 Connect live field observations and operational irrigation inputs.
 Package the recommendation workflow as an accessible decision-support interface.
<div align="center">
🌱 From satellite pixels to smarter water decisions.

Built for precision agriculture | Godavari Delta, Andhra Pradesh, India

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:123B4A,50:168B75,100:8CBF9F&height=100&section=footer" width="100%" alt="Decorative footer" /> </div> """

hero = """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="330" viewBox="0 0 1200 330" role="img" aria-labelledby="t d">

<title id="t">AI-Driven Smart Irrigation</title><desc id="d">Animated satellite, floating water droplets, leaf and data symbols over a dark teal agricultural landscape.</desc> <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#0b2633"/><stop offset=".55" stop-color="#104b4c"/><stop offset="1" stop-color="#123b4a"/></linearGradient><linearGradient id="f" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#2b806b"/><stop offset="1" stop-color="#a5c98b"/></linearGradient></defs> <rect width="1200" height="330" rx="24" fill="url(#bg)"/><circle cx="1000" cy="70" r="100" fill="#9ed6bd" opacity=".07"/><circle cx="190" cy="280" r="150" fill="#72b99b" opacity=".06"/> <g opacity=".2" stroke="#b7e2cf" fill="none"><path d="M0 230 Q300 170 600 235 T1200 205"/><path d="M0 260 Q300 200 600 265 T1200 235"/><path d="M0 290 Q300 230 600 295 T1200 265"/></g> <path d="M0 300 Q250 230 520 300 L520 330 H0Z" fill="url(#f)" opacity=".75"/><path d="M680 330 Q900 245 1200 285 V330Z" fill="url(#f)" opacity=".7"/> <g transform="translate(920 94)"><g><animateTransform attributeName="transform" type="rotate" values="-4;4;-4" dur="7s" repeatCount="indefinite"/><rect x="-25" y="-18" width="50" height="36" rx="6" fill="#dcece3" stroke="#77b79d" stroke-width="2"/><rect x="-91" y="-14" width="54" height="28" rx="3" fill="#4e9b91" stroke="#c4e6d5"/><rect x="37" y="-14" width="54" height="28" rx="3" fill="#4e9b91" stroke="#c4e6d5"/><path d="M-91 -5H-37 M-91 5H-37 M37 -5H91 M37 5H91" stroke="#b9e3d2"/><circle r="7" fill="#d5a85c"/></g></g> <g fill="#9ad9d0"><path d="M1080 185 C1062 208 1066 226 1080 226 C1094 226 1098 208 1080 185Z"><animateTransform attributeName="transform" type="translate" values="0 8;0 -12;0 8" dur="4s" repeatCount="indefinite"/></path><path d="M1130 110 C1117 127 1120 141 1130 141 C1140 141 1143 127 1130 110Z"><animateTransform attributeName="transform" type="translate" values="0 -8;0 12;0 -8" dur="5s" repeatCount="indefinite"/></path><path d="M760 190 C748 206 751 219 760 219 C769 219 772 206 760 190Z"><animateTransform attributeName="transform" type="translate" values="0 6;0 -10;0 6" dur="4.5s" repeatCount="indefinite"/></path></g> <g font-family="monospace" font-size="18" fill="#d0eadc" opacity=".8"><text x="1040" y="275">NDVI</text><text x="735" y="82">ET₀</text><text x="1100" y="165">H₂O</text><text x="670" y="160">01</text><text x="1140" y="70">+</text><text x="650" y="210">✳<animateTransform attributeName="transform" type="translate" values="0 0;0 -9;0 0" dur="6s" repeatCount="indefinite"/></text></g> <g transform="translate(790 245)" fill="#9fcf91"><path d="M0 35 C-8 -5 28 -35 70 -30 C67 12 35 43 0 35Z"/><path d="M0 35 Q28 7 62 -22" fill="none" stroke="#315f4e" stroke-width="2"/><animateTransform attributeName="transform" type="rotate" values="-3 790 245;4 790 245;-3 790 245" dur="6s" repeatCount="indefinite"/></g> <g font-family="Arial,Helvetica,sans-serif" fill="#f1f7f3"><text x="72" y="112" font-size="19" letter-spacing="5" fill="#a9d9c4">PRECISION AGRICULTURE • MACHINE LEARNING</text><text x="70" y="174" font-size="43" font-weight="700">AI-Driven Smart Irrigation</text><text x="73" y="215" font-size="23" fill="#c3e2d2">Recommendation System for Paddy Fields</text><text x="74" y="267" font-size="15" letter-spacing="2" fill="#a9d9c4">SENTINEL-2 × NASA POWER × FAO-56</text></g><path d="M70 290 H540" stroke="#7ac2a5" stroke-width="3" stroke-linecap="round"/> </svg>"""

(root / "README.md").write_text(readme, encoding="utf-8")
(root / "assets" / "hero.svg").write_text(hero, encoding="utf-8")
zip_path = Path("/mnt/data/AI_Smart_Irrigation_Animated_README.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
zf.write(root / "README.md", "README.md")
zf.write(root / "assets" / "hero.svg", "assets/hero.svg")

print("Created README.md, animated hero SVG, and ZIP bundle.")
