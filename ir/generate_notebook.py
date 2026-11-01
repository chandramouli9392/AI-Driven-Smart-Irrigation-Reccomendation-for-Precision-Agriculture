import json
import os

def create_eda_notebook(output_path):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI-Driven Smart Irrigation Recommendation System for Precision Agriculture\n",
                "## Final-Year B.Tech Project | Exploratory Data Analysis & Hydrological Integrity Audit\n",
                "\n",
                "**Agricultural Study Area:** Godavari Delta, Andhra Pradesh, India  \n",
                "**Target Crop:** Lowland Irrigated Paddy Rice (*Oryza sativa*)  \n",
                "**Remote Sensing Data:** Sentinel-2 Level-2A Cloud-Optimized Surface Reflectance (BOA)  \n",
                "**Agro-Meteorological Data:** NASA POWER Daily Agroclimatology & FAO-56 Reference Evapotranspiration ($ET_0$)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "sns.set_theme(style='whitegrid', font_scale=1.1)\n",
                "%matplotlib inline\n",
                "\n",
                "CSV_PATH = 'AI DRIVEN SMART IRRIGATION RECCOMENDATION FOR REAL_SENTINEL2_DATASET.csv'\n",
                "df = pd.read_csv(CSV_PATH)\n",
                "print(f'Dataset Loaded: {df.shape[0]:,} observations x {df.shape[1]} attributes')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Data Integrity & Missingness Audit\n",
                "We verify data completeness, missingness, duplicates, and unique spatial-temporal units."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Missing values count:', df.isnull().sum().sum())\n",
                "print('Duplicate rows count :', df.duplicated().sum())\n",
                "print('Unique Grid Cells     :', df['Grid_ID'].nunique())\n",
                "print('Unique Scene Dates    :', df['Date'].nunique())\n",
                "\n",
                "df['Date_parsed'] = pd.to_datetime(df['Date'], dayfirst=True)\n",
                "print(f\"Date Range: {df['Date_parsed'].min().date()} to {df['Date_parsed'].max().date()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Mathematical Target Derivation & Data Leakage Investigation\n",
                "The target variable `Irrigation_Required` is derived directly from the FAO-56 hydrological water balance equation:\n",
                "$$ET_c = K_c \\times ET_0$$\n",
                "$$P_{eff} = 0.80 \\times PRECTOTCORR$$\n",
                "$$I_{req} = \\max(0, ET_c - P_{eff})$$\n",
                "$$V_{water} = I_{req} \\times \\text{Area} \\times 10$$\n",
                "$$\\text{Irrigation\\_Required} = \\begin{cases} \\text{YES}, & \\text{if } V_{water} > 0 \\\\ \\text{NO}, & \\text{otherwise} \\end{cases}$$\n",
                "\n",
                "Below we verify whether these equations hold exactly."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "diff_etc = (df['Kc'] * df['ET0_mm'] - df['ETc_mm']).abs().max()\n",
                "print(f'ETc = Kc * ET0 max abs diff: {diff_etc:.2e}')\n",
                "\n",
                "rr = df[df['PRECTOTCORR'] > 0]['Effective_Rainfall_mm'] / df[df['PRECTOTCORR'] > 0]['PRECTOTCORR']\n",
                "print(f'Effective_Rainfall = 0.8 * PRECTOTCORR unique ratio: {rr.unique()}')\n",
                "\n",
                "net_irr = (df['ETc_mm'] - df['Effective_Rainfall_mm']).clip(lower=0)\n",
                "vol_calc = net_irr * df['Field_Area_ha'] * 10\n",
                "diff_vol = (vol_calc - df['Water_Required_m3']).abs().max()\n",
                "print(f'Water_Required_m3 max diff from formula: {diff_vol:.2e}')\n",
                "\n",
                "match_cls = ((df['Irrigation_Required'] == 'YES') == (df['Water_Required_m3'] > 0)).all()\n",
                "print(f'Irrigation_Required == (Water > 0): {match_cls}')\n",
                "print('Field Area Standard Deviation:', df['Field_Area_ha'].std())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Class Imbalance & Temporal Distribution of Minority 'NO' Cases\n",
                "We investigate why the dataset is 94.99% YES and 5.01% NO."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(df['Irrigation_Required'].value_counts(normalize=True).mul(100).round(2))\n",
                "\n",
                "# Check dates where NO occurs\n",
                "no_by_date = df[df['Irrigation_Required'] == 'NO'].groupby('Date_parsed').size()\n",
                "print('\\nDates where Irrigation_Required == NO:')\n",
                "print(no_by_date)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
                "df['Irrigation_Required'].value_counts().plot(kind='bar', ax=axes[0], color=['#27ae60', '#e74c3c'])\n",
                "axes[0].set_title('Target Class Distribution')\n",
                "axes[0].set_ylabel('Observation Count')\n",
                "\n",
                "df['Water_Required_m3'].plot(kind='hist', bins=40, ax=axes[1], color='#2980b9')\n",
                "axes[1].set_title('Water Volume Distribution (m3)')\n",
                "axes[1].set_xlabel('Volume (m3)')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Remote Sensing & Spectral Phenology Analysis\n",
                "Sentinel-2 MSI bands and derived vegetation/water indices across phenological stages."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "indices = ['NDVI', 'NDWI', 'MNDWI', 'NDMI', 'EVI']\n",
                "stage_order = ['Initial', 'Development', 'Mid', 'Late']\n",
                "\n",
                "fig, axes = plt.subplots(1, len(indices), figsize=(18, 4))\n",
                "for ax, idx in zip(axes, indices):\n",
                "    sns.boxplot(data=df, x='Crop_Stage', y=idx, order=stage_order, ax=ax, palette='viridis')\n",
                "    ax.set_title(idx)\n",
                "    ax.set_xlabel('')\n",
                "    ax.tick_params(axis='x', rotation=25)\n",
                "plt.suptitle('Spectral Indices across Crop Stages', y=1.02)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Architectural Conclusions & Modeling Recommendations\n",
                "1. **Leakage Elimination:** Direct mathematical precursors (`Irrigation_Requirement_mm`, `PRECTOTCORR`, `Field_Area_ha`) must be excluded from feature matrices.\n",
                "2. **Dual Experimental Paradigms:**\n",
                "   - **Setting A:** Full features (Upper Bound formula reproduction).\n",
                "   - **Setting B:** Independent remote sensing (No ETc, no rainfall) to evaluate true satellite generalizability.\n",
                "3. **Validation Strategy:** Spatial grouping by `Grid_ID` and chronological splitting prevent data leakage across space and time.\n",
                "4. **Class Imbalance:** Class weights and decision threshold tuning are mandatory to overcome the 95:5 class ratio."
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook: {output_path}")

create_eda_notebook("notebooks/exploratory_data_analysis.ipynb")
create_eda_notebook("IRRIIIIIIIIIIIIIIIIIII/notebooks/exploratory_data_analysis.ipynb")
