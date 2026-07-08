"""Shared Sentinel-2 constants and helper functions used across packages.

Place common Sentinel-2 band lists, scaling and index calculations here so
`superres` and `seagrass` can reuse the same logic.
"""

from typing import Dict
import numpy as np
import pandas as pd

# Download band names (as used by Earth Engine / Sentinel-2).
SENTINEL2_DOWNLOAD_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]

# Column names we use in CSVs / DataFrames (lowercase, prefixed with s2_).
SENTINEL2_BAND_COLUMNS = [
    "s2_b1",
    "s2_b2",
    "s2_b3",
    "s2_b4",
    "s2_b5",
    "s2_b6",
    "s2_b7",
    "s2_b8",
    "s2_b8a",
    "s2_b11",
    "s2_b12",
]

SENTINEL2_BAND_NAME_MAP = dict(zip(SENTINEL2_DOWNLOAD_BANDS, SENTINEL2_BAND_COLUMNS))

# Scale applied to L2A SR products
SENTINEL2_SCALE = 10000.0

# Default temporal window (either module may override)
SENTINEL2_WINDOW_DAYS = 15

SENTINEL2_INDEX_COLUMNS = ["ndvi", "gndvi", "ndre", "ndwi", "mndwi", "evi", "savi", "nbr"]


def compute_sentinel2_indices(features: Dict[str, float]) -> Dict[str, float]:
    """Add spectral indices to a dict that already has scaled s2_b* values (0–1).

    Modifies `features` in place and returns it.  Call this after populating band
    values from any source (GEE or ACOLITE) to get a consistent index set.
    """
    b2 = features.get("s2_b2", np.nan)
    b3 = features.get("s2_b3", np.nan)
    b4 = features.get("s2_b4", np.nan)
    b5 = features.get("s2_b5", np.nan)
    b8 = features.get("s2_b8", np.nan)
    b8a = features.get("s2_b8a", np.nan)
    b11 = features.get("s2_b11", np.nan)
    b12 = features.get("s2_b12", np.nan)

    features["ndvi"] = float((b8 - b4) / (b8 + b4)) if not (pd.isna(b8) or pd.isna(b4) or (b8 + b4) == 0) else np.nan
    features["gndvi"] = float((b8 - b3) / (b8 + b3)) if not (pd.isna(b8) or pd.isna(b3) or (b8 + b3) == 0) else np.nan
    features["ndre"] = float((b8a - b5) / (b8a + b5)) if not (pd.isna(b8a) or pd.isna(b5) or (b8a + b5) == 0) else np.nan
    features["ndwi"] = float((b3 - b8) / (b3 + b8)) if not (pd.isna(b3) or pd.isna(b8) or (b3 + b8) == 0) else np.nan
    features["mndwi"] = float((b3 - b11) / (b3 + b11)) if not (pd.isna(b3) or pd.isna(b11) or (b3 + b11) == 0) else np.nan
    features["evi"] = float(2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1)) if not (pd.isna(b2) or pd.isna(b4) or pd.isna(b8) or (b8 + 6 * b4 - 7.5 * b2 + 1) == 0) else np.nan
    features["savi"] = float(1.5 * (b8 - b4) / (b8 + b4 + 0.5)) if not (pd.isna(b8) or pd.isna(b4) or (b8 + b4 + 0.5) == 0) else np.nan
    features["nbr"] = float((b8 - b12) / (b8 + b12)) if not (pd.isna(b8) or pd.isna(b12) or (b8 + b12) == 0) else np.nan
    return features


def build_sentinel2_feature_values(stats: Dict[str, float], scene_date: str) -> Dict[str, float]:
    """Convert Earth Engine reduction results into normalized Sentinel-2 features.

    `stats` is the dictionary returned by an EE reduction (keys are download-band
    names like "B2", "B3", ...). `scene_date` is the YYYYMMDD string for the
    selected scene and is copied into the returned mapping as `scene_date`.
    """
    features: Dict[str, float] = {}
    for index, band_name in enumerate(SENTINEL2_DOWNLOAD_BANDS):
        output_name = SENTINEL2_BAND_NAME_MAP.get(band_name, SENTINEL2_BAND_COLUMNS[index])
        value = stats.get(band_name)
        if value is None:
            features[output_name] = np.nan
        else:
            features[output_name] = float(value) / SENTINEL2_SCALE

    compute_sentinel2_indices(features)
    features["scene_date"] = scene_date
    return features


def add_sentinel2_columns(frame):
    """Add Sentinel-2 band and index columns to `frame` with default values."""
    df = frame.copy()
    for column in SENTINEL2_BAND_COLUMNS + SENTINEL2_INDEX_COLUMNS:
        df[column] = np.nan
    df["scene_date"] = ""
    df["sentinel2_status"] = "pending_date"
    return df
