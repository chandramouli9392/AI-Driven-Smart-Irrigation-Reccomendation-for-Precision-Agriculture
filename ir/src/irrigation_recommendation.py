"""
irrigation_recommendation.py
============================
Agricultural Decision Support Engine based on the FAO-56 Irrigation & Drainage Framework.
Specialised for Lowland Paddy Rice (Oryza sativa) in the Godavari Delta, Andhra Pradesh.

Scientific References:
1. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration:
   Guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56, Rome.
2. Doorenbos, J., & Pruitt, W. O. (1977). Guidelines for predicting crop water requirements.
   FAO Irrigation and Drainage Paper 24, Rome.
3. Bouman, B. A. M., & Tuong, T. P. (2001). Field water management to save water and
   increase its productivity in irrigated lowland rice. Agricultural Water Management, 49(1), 11-30.

Hydrological Balance Formula:
    ETc (mm/day) = ET0 (mm/day) * Kc
    P_eff (mm/day) = 0.80 * PRECTOTCORR (or USDA SCS Method)
    Percolation (mm/day) = Seepage and percolation loss through alluvial puddled layer (2.0 - 4.0 mm/day)
    Net Irrigation (mm) = max(0.0, ETc + Percolation - P_eff)
    Water Volume (m3) = Irrigation Depth (mm) * Field Area (ha) * 10
"""

import sys
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PaddyPhenologyStage:
    stage_name: str
    kc_default: float
    target_standing_water_mm: float  # Recommended standing water layer (ponding depth)
    percolation_rate_mm_day: float   # Typical vertical percolation in Godavari clay
    drainage_advisory: str


PADDY_STAGE_SPECS = {
    "Initial": PaddyPhenologyStage(
        stage_name="Initial (Transplanting to Early Tillering)",
        kc_default=1.05,
        target_standing_water_mm=25.0,  # Shallow water for seedling establishment
        percolation_rate_mm_day=3.0,
        drainage_advisory="Maintain shallow water (2-3 cm) to prevent seedling submergence and shock."
    ),
    "Development": PaddyPhenologyStage(
        stage_name="Development (Active Tillering to Stem Elongation)",
        kc_default=1.15,
        target_standing_water_mm=40.0,  # Moderate ponding for weed suppression
        percolation_rate_mm_day=2.5,
        drainage_advisory="Gradually increase standing water to 3-5 cm for weed suppression."
    ),
    "Mid": PaddyPhenologyStage(
        stage_name="Mid-Season (Panicle Initiation to Flowering/Heading)",
        kc_default=1.25,
        target_standing_water_mm=50.0,  # Critical peak water requirement
        percolation_rate_mm_day=2.0,
        drainage_advisory="Critical reproductive phase: maintain 5 cm standing water. Moisture stress causes floret sterility."
    ),
    "Late": PaddyPhenologyStage(
        stage_name="Late-Season (Milk to Dough & Grain Ripening)",
        kc_default=0.90,
        target_standing_water_mm=10.0,  # Decreasing ponding to terminal drainage
        percolation_rate_mm_day=2.0,
        drainage_advisory="Terminal drainage: drain fields 10-14 days before harvest to promote uniform ripening."
    ),
}

DEFAULT_FIELD_AREA_HA = 605.672381  # Standard Godavari delta administrative grid cell area


def calculate_fao56_etc(et0_mm: float, kc: Optional[float] = None, stage_name: str = "Mid") -> float:
    """Calculate Crop Evapotranspiration: ETc = ET0 * Kc."""
    if kc is None or np.isnan(kc):
        stage_spec = PADDY_STAGE_SPECS.get(stage_name, PADDY_STAGE_SPECS["Mid"])
        kc = stage_spec.kc_default
    return max(0.0, et0_mm * kc)


def calculate_effective_rainfall(total_rainfall_mm: float, method: str = "empirical_80") -> float:
    """
    Calculate Effective Rainfall available for crop uptake.
    Methods:
    - 'empirical_80': 0.80 * Precipitation (standard in Godavari agricultural dataset)
    - 'fao_usda': FAO/USDA Soil Conservation Service empirical formula
    """
    if total_rainfall_mm <= 0:
        return 0.0

    if method == "empirical_80":
        return max(0.0, total_rainfall_mm * 0.80)
    elif method == "fao_usda":
        if total_rainfall_mm <= 8.3:
            return max(0.0, total_rainfall_mm * 0.8)
        else:
            return max(0.0, 8.3 * 0.8 + 0.6 * (total_rainfall_mm - 8.3))
    else:
        return total_rainfall_mm * 0.80


def calculate_irrigation_recommendation(
    et0_mm: float,
    rainfall_mm: float,
    crop_stage: str = "Mid",
    kc: Optional[float] = None,
    current_standing_water_mm: Optional[float] = None,
    field_area_ha: float = DEFAULT_FIELD_AREA_HA,
    include_percolation: bool = False,
    irrigation_efficiency: float = 0.85,
) -> Dict[str, Any]:
    """
    Comprehensive FAO-56 Precision Irrigation Recommendation Engine.

    Parameters:
    -----------
    et0_mm : float
        Reference Evapotranspiration (mm/day) from NASA POWER / weather station.
    rainfall_mm : float
        Total precipitation (mm/day) or NASA POWER PRECTOTCORR.
    crop_stage : str
        One of 'Initial', 'Development', 'Mid', 'Late'.
    kc : float, optional
        Crop coefficient. If None, stage default is used.
    current_standing_water_mm : float, optional
        Observed standing water layer on paddy plot.
    field_area_ha : float
        Field or cluster area in hectares (default: 605.672 ha).
    include_percolation : bool
        Whether to incorporate soil percolation loss (for gross field demand).
    irrigation_efficiency : float
        Application & canal conveyance efficiency (default: 85%).

    Returns:
    --------
    dict containing:
        - irrigation_required: 'YES' or 'NO'
        - net_irrigation_depth_mm: float
        - gross_irrigation_depth_mm: float
        - water_volume_m3: float
        - daily_etc_mm: float
        - effective_rainfall_mm: float
        - percolation_mm: float
        - agronomic_advisory: str
        - water_deficit_mm: float
    """
    stage_spec = PADDY_STAGE_SPECS.get(crop_stage, PADDY_STAGE_SPECS["Mid"])
    used_kc = kc if (kc is not None and not np.isnan(kc)) else stage_spec.kc_default

    etc_mm = max(0.0, et0_mm * used_kc)
    eff_rainfall_mm = calculate_effective_rainfall(rainfall_mm, method="empirical_80")

    percolation_mm = stage_spec.percolation_rate_mm_day if include_percolation else 0.0

    # Theoretical FAO-56 net water deficit (identical to dataset derivation when percolation=0)
    net_irrigation_depth_mm = max(0.0, (etc_mm + percolation_mm) - eff_rainfall_mm)

    # Adjust for existing standing water layer if reported by field sensor
    if current_standing_water_mm is not None:
        target_swd = stage_spec.target_standing_water_mm
        swd_deficit = max(0.0, target_swd - current_standing_water_mm)
        # Total required is ETc deficit + water layer replenishment
        total_depth_mm = max(0.0, net_irrigation_depth_mm + swd_deficit)
    else:
        total_depth_mm = net_irrigation_depth_mm

    # Gross water accounting for distribution and application efficiency
    gross_depth_mm = total_depth_mm / max(0.1, irrigation_efficiency)

    # Volumetric conversion: Depth (mm) * Area (ha) * 10 = m3
    water_volume_m3 = total_depth_mm * field_area_ha * 10.0
    gross_volume_m3 = gross_depth_mm * field_area_ha * 10.0

    irrigation_required = "YES" if total_depth_mm > 0.001 else "NO"

    # Actionable agronomic advisory
    if irrigation_required == "NO":
        advisory = (
            f"Sufficient rainfall received ({eff_rainfall_mm:.2f} mm eff. rain vs {etc_mm:.2f} mm ETc). "
            f"No canal water application required. {stage_spec.drainage_advisory}"
        )
    else:
        advisory = (
            f"Water deficit of {total_depth_mm:.2f} mm detected. "
            f"Recommended canal discharge: {water_volume_m3:,.0f} m3 ({water_volume_m3 / (field_area_ha * 1000):.2f} mm/ha). "
            f"{stage_spec.drainage_advisory}"
        )

    return {
        "irrigation_required": irrigation_required,
        "net_irrigation_depth_mm": round(total_depth_mm, 4),
        "gross_irrigation_depth_mm": round(gross_depth_mm, 4),
        "water_volume_m3": round(water_volume_m3, 2),
        "gross_water_volume_m3": round(gross_volume_m3, 2),
        "etc_mm": round(etc_mm, 4),
        "effective_rainfall_mm": round(eff_rainfall_mm, 4),
        "percolation_mm": round(percolation_mm, 2),
        "stage_name": stage_spec.stage_name,
        "agronomic_advisory": advisory,
        "field_area_ha": field_area_ha,
    }


if __name__ == "__main__":
    import numpy as np
    # Demo recommendation
    rec = calculate_irrigation_recommendation(
        et0_mm=5.8,
        rainfall_mm=0.5,
        crop_stage="Mid",
        kc=1.2,
        field_area_ha=605.672,
    )
    print("FAO-56 Irrigation Engine Demo:")
    for k, v in rec.items():
        print(f"  {k:25s}: {v}")
