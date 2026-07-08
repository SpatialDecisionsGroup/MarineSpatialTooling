"""Shared Landsat constants and helper functions used across packages.

Covers Landsat 8 and 9 Collection 2 Level-2 surface reflectance bands, with
the same structure as common.sentinel so both can be used symmetrically.
"""

from typing import Dict
import numpy as np
import pandas as pd

from .gee_satellite import GEESatelliteManager, GEESatelliteSpec

# Band names as returned by Earth Engine for LC08/LC09 C02 T1_L2 collections.
LANDSAT_DOWNLOAD_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

# Column names in CSVs/DataFrames (lowercase, prefixed with ls_ to avoid
# collision with Sentinel-2 indices when both appear in the same frame).
LANDSAT_BAND_COLUMNS = ["ls_b1", "ls_b2", "ls_b3", "ls_b4", "ls_b5", "ls_b6", "ls_b7"]

LANDSAT_BAND_NAME_MAP = dict(zip(LANDSAT_DOWNLOAD_BANDS, LANDSAT_BAND_COLUMNS))

# Landsat C2 L2: reflectance = DN * 0.0000275 + (-0.2)
LANDSAT_SCALE = 0.0000275
LANDSAT_OFFSET = -0.2

# Wider window than Sentinel-2 to account for Landsat's longer revisit.
LANDSAT_WINDOW_DAYS = 30

LANDSAT_INDEX_COLUMNS = ["ls_ndvi", "ls_ndwi", "ls_mndwi", "ls_evi", "ls_savi", "ls_nbr"]


def compute_landsat_indices(features: Dict[str, float]) -> Dict[str, float]:
    """Add spectral indices to a dict that already has scaled ls_b* values (0–1).

    Modifies `features` in place and returns it.  Call this after populating band
    values from any source (GEE or ACOLITE) to get a consistent index set.
    """
    b2 = features.get("ls_b2", np.nan)  # Blue
    b3 = features.get("ls_b3", np.nan)  # Green
    b4 = features.get("ls_b4", np.nan)  # Red
    b5 = features.get("ls_b5", np.nan)  # NIR
    b6 = features.get("ls_b6", np.nan)  # SWIR1
    b7 = features.get("ls_b7", np.nan)  # SWIR2

    features["ls_ndvi"] = float((b5 - b4) / (b5 + b4)) if not (pd.isna(b5) or pd.isna(b4) or (b5 + b4) == 0) else np.nan
    features["ls_ndwi"] = float((b3 - b5) / (b3 + b5)) if not (pd.isna(b3) or pd.isna(b5) or (b3 + b5) == 0) else np.nan
    features["ls_mndwi"] = float((b3 - b6) / (b3 + b6)) if not (pd.isna(b3) or pd.isna(b6) or (b3 + b6) == 0) else np.nan
    features["ls_evi"] = float(2.5 * (b5 - b4) / (b5 + 6 * b4 - 7.5 * b2 + 1)) if not (pd.isna(b2) or pd.isna(b4) or pd.isna(b5) or (b5 + 6 * b4 - 7.5 * b2 + 1) == 0) else np.nan
    features["ls_savi"] = float(1.5 * (b5 - b4) / (b5 + b4 + 0.5)) if not (pd.isna(b5) or pd.isna(b4) or (b5 + b4 + 0.5) == 0) else np.nan
    features["ls_nbr"] = float((b5 - b7) / (b5 + b7)) if not (pd.isna(b5) or pd.isna(b7) or (b5 + b7) == 0) else np.nan
    return features


def build_landsat_feature_values(stats: Dict[str, float], scene_date: str) -> Dict[str, float]:
    """Convert Earth Engine reduction results into Landsat reflectance features.

    `stats` is the dict returned by an EE reduction (keys are SR_B1 … SR_B7).
    Scale/offset conversion follows Landsat Collection 2 Level-2 documentation.
    """
    features: Dict[str, float] = {}
    for band_name, col_name in LANDSAT_BAND_NAME_MAP.items():
        value = stats.get(band_name)
        if value is None:
            features[col_name] = np.nan
        else:
            features[col_name] = float(value) * LANDSAT_SCALE + LANDSAT_OFFSET

    compute_landsat_indices(features)
    features["ls_scene_date"] = scene_date
    return features


def add_landsat_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Landsat band and index columns to frame with default NaN values."""
    df = frame.copy()
    for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
        df[column] = np.nan
    df["ls_scene_date"] = ""
    df["landsat_status"] = "pending"
    return df


class LandsatManager(GEESatelliteManager):
    """Manages Landsat 8/9 (Collection 2 Level-2 SR) retrieval, turbidity estimation, and downloads via GEE."""

    SPEC = GEESatelliteSpec(
        display_name="Landsat",
        collection=["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"],
        band_names=["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"],
        cloud_cover_property="CLOUD_COVER",
        red_band="SR_B4",
        green_band="SR_B3",
        resolution_meters=30,
    )
