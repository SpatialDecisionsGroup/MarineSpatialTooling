"""Attach Landsat, Sentinel-2, and PlanetScope band values to Indonesian seagrass CSVs.

The site spreadsheets live alongside the downloaded PlanetScope folders in:
/home/ysi/Documents/Uni/MQPostdoc/projects/superres/indonesian_seagrass

Imagery sources:
  - Landsat 8/9 (30 m, low-res): ACOLITE NetCDF output if available, else GEE.
  - Sentinel-2 (10 m, high-res): ACOLITE NetCDF output if available, else GEE.
  - PlanetScope (3 m, for later comparison): read from pre-downloaded local rasters
    since PlanetScope cannot be batch-downloaded via API due to quota limits.

The 9 sites fall into two clusters ~200km apart (see BAWEAN_LIMIT / BALURAN_LIMIT),
so raw-scene download and ACOLITE processing run per cluster:

  # 1. See which dates need coverage + the bounding box, per cluster
  python seagrass/indonesia.py dates --root <root_dir>

  # 2. Download Level-1 scenes for one cluster (free accounts required — see below)
  python seagrass/indonesia.py download --root <root_dir> --cluster bawean \\
      --scenes /raw_scenes/bawean/ \\
      --cdse-user you@email.com --cdse-pass yourpass \\
      --usgs-user yourname --usgs-token yourtoken
  # repeat with --cluster baluran --scenes /raw_scenes/baluran/

  # 3. Run ACOLITE per cluster (limit is set from the cluster automatically)
  python seagrass/indonesia.py acolite --cluster bawean \\
      --scenes /raw_scenes/bawean/ --acolite-out /acolite_out/
  # repeat with --cluster baluran --scenes /raw_scenes/baluran/

  # 4. Build the CSVs (ACOLITE output covers both clusters; falls back to GEE
  #    per-row for anything ACOLITE didn't cover). --gee-project can be omitted
  #    if gee_project is set in common/credentials.json.
  python seagrass/indonesia.py build --root <root_dir> --acolite-dir /acolite_out/

Skipping steps 1-3 and running 'build' with just --root (no --acolite-dir) uses
GEE's standard L2A/SR products for everything — ACOLITE is an upgrade, not a
requirement, same as seagrass/tampa_bay.py.

Accounts
--------
Sentinel-2 : https://dataspace.copernicus.eu  (free)
Landsat     : https://ers.cr.usgs.gov          (free, generate Application Token
              in your profile under "Access" → "Create Application Token")
ACOLITE     : see common/acolite_pipeline.py's module docstring.
"""

from __future__ import annotations
import re
from datetime import timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as transform_coordinates

from common.credentials import load_credentials
from common.gee_clear_sky import (
    s2_local_clear_candidates,
    landsat_local_clear_candidates,
    select_by_local_clarity,
)
from common.sentinel import (
    SENTINEL2_BAND_COLUMNS,
    SENTINEL2_DOWNLOAD_BANDS,
    SENTINEL2_BAND_NAME_MAP,
    SENTINEL2_SCALE,
    SENTINEL2_INDEX_COLUMNS,
    build_sentinel2_feature_values,
    compute_sentinel2_indices,
)
from common.landsat import (
    LANDSAT_BAND_COLUMNS,
    LANDSAT_DOWNLOAD_BANDS,
    LANDSAT_INDEX_COLUMNS,
    build_landsat_feature_values,
    compute_landsat_indices,
)
from common.sentinel2 import Sentinel2Manager
from common.landsat import LandsatManager
from common.depth_correction import (
    S2_LYZENGA_PAIRS,
    LS_LYZENGA_PAIRS,
    S2_LYZENGA_COLUMNS,
    LS_LYZENGA_COLUMNS,
    add_lyzenga_columns,
)
from common.acolite_pipeline import (
    ACOLITE_REPO_PATH,
    ACOLITE_S2_BAND_MAP,
    ACOLITE_LS_BAND_MAP,
    merge_date_windows,
    download_sentinel2,
    download_landsat,
    run_acolite_batch,
    scan_acolite_output,
    select_acolite_scene,
    sample_acolite_nc,
)
import argparse


DEFAULT_ROOT = Path(".")
SUMMARY_FILES = ["summary.csv", "summary_pantai_bama.csv"]
PS_BAND_COLUMNS = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
PS_INDEX_COLUMNS = ["ndvi", "gndvi", "ndre", "ndwi", "evi", "savi", "cig"]
PS_BAND_SCALE = 10000.0

# Wider than common/sentinel.py's and common/landsat.py's defaults (15 / 30 days):
# some Indonesian sites (Somor-somor, Pamasaran, Cinta) sit under persistent cloud,
# so a narrow window can miss every clear scene entirely.
INDONESIA_S2_WINDOW_DAYS = 60
INDONESIA_LS_WINDOW_DAYS = 60

# Ground footprint sampled around each observation point for band extraction.
# This must be a fixed physical size shared across sensors, NOT a fixed pixel
# count — "3x3 pixels" means 900 m^2 at Sentinel-2's 10 m resolution but 8100 m^2
# at Landsat's 30 m, a 9x difference in the area actually being averaged. That
# asymmetry (previously buffer(15) for S2 vs buffer(45) for Landsat) biases any
# cross-sensor comparison, since a bigger footprint averages out more local
# noise/GPS error independent of anything about the sensor itself.
BAND_SAMPLE_BUFFER_M = 45

# The 9 sites fall into two geographically distant clusters (~200 km apart) —
# ACOLITE limit format: [south, west, north, east], ±0.05° buffer around each
# cluster's site extent (same convention as seagrass/tampa_bay.py's TAMPA_BAY_LIMIT).
BAWEAN_LIMIT = [-5.907, 112.538, -5.698, 112.787]     # Cinta, Jerat Lanjeng, Pamasaran, Pasir Putih, Somor-somor
BALURAN_LIMIT = [-7.895, 114.411, -7.793, 114.513]    # Pantai Bama (4 stations)
CLUSTER_LIMITS = {"bawean": BAWEAN_LIMIT, "baluran": BALURAN_LIMIT}

_BAWEAN_SITES = {"cinta", "jerat lanjeng", "pamasaran", "pasir putih", "somor somor"}


def site_cluster(location: str) -> str | None:
    normalized = normalise_label(location)
    if normalized.startswith("pantai bama"):
        return "baluran"
    if normalized in _BAWEAN_SITES:
        return "bawean"
    return None


def normalise_label(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = []
    for column in frame.columns:
        column_name = str(column).strip()
        if column_name == "" or column_name.startswith("Unnamed:"):
            column_name = "Date"
        renamed.append(column_name)
    frame = frame.copy()
    frame.columns = renamed
    return frame


def coverage_class(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return ""
    if numeric_value < 30:
        return "low"
    if numeric_value < 60:
        return "medium"
    return "high"


def parse_date_value(value: object) -> str:
    parsed = parse_date_object(value)
    if parsed is None:
        return ""
    return parsed.strftime("%Y%m%d")


def parse_date_object(value: object) -> pd.Timestamp | None:
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{8}", text):
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    elif "T" in text:
        parsed = pd.to_datetime(text, errors="coerce")
    else:
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def format_date_window(value: object, window_days: int) -> tuple[str, str] | tuple[None, None]:
    parsed = parse_date_object(value)
    if parsed is None:
        return None, None
    start = (parsed - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (parsed + timedelta(days=window_days)).strftime("%Y-%m-%d")
    return start, end


def discover_imagery_folders(root_dir: Path) -> dict[str, list[Path]]:
    """Collect all clipped SuperDove GeoTIFFs by normalized site name."""
    folder_map: dict[str, list[Path]] = {}

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name == "Seagrass - Baluran Indonesia":
            continue

        ps_dir = folder / "PSScene"
        if not ps_dir.exists():
            continue

        tif_files = sorted(ps_dir.glob("*_AnalyticMS_SR_8b_harmonized_clip.tif"))
        if not tif_files:
            continue

        site_name = folder.name
        site_name = site_name.replace("_Seagrass_psscene_analytic_8b_sr_udm2", "")
        site_name = site_name.replace("_psscene_analytic_8b_sr_udm2", "")
        site_name = site_name.replace("_", " ")
        site_name = re.sub(r"\s+", " ", site_name).strip()

        folder_map.setdefault(normalise_label(site_name), []).extend(tif_files)

    return folder_map


def resolve_site_images(location: str, folder_map: dict[str, list[Path]]) -> list[Path]:
    normalized_location = normalise_label(location)
    if normalized_location in folder_map:
        return folder_map[normalized_location]

    candidates: list[tuple[int, int, list[Path]]] = []
    for key, files in folder_map.items():
        if key.startswith(normalized_location) or normalized_location.startswith(key):
            candidates.append((abs(len(key) - len(normalized_location)), len(files), files))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def select_scene_by_date(files: list[Path], row: pd.Series) -> Path:
    """Pick the best raster for a row, preferring one that matches the row date."""
    row_date = parse_date_object(row.get("Date", ""))
    if row_date is not None:
        exact_matches: list[Path] = []
        nearest_candidate: tuple[int, Path] | None = None

        for file_path in files:
            file_date = parse_date_object(file_path.name[:8])
            if file_date is None:
                continue
            if file_date == row_date:
                exact_matches.append(file_path)
            else:
                distance = abs((file_date - row_date).days)
                if nearest_candidate is None or distance < nearest_candidate[0]:
                    nearest_candidate = (distance, file_path)

        if exact_matches:
            return exact_matches[0]
        if nearest_candidate is not None:
            return nearest_candidate[1]

    return files[0]



def extract_window_band_means(
    raster_path: Path,
    longitude: float,
    latitude: float,
    band_columns: list[str],
    band_scale: float,
    band_name_map: dict[str, str],
) -> dict[str, float]:
    """Sample a 3×3 raster neighborhood around a point and return normalized band means."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            x_coord, y_coord = longitude, latitude
        else:
            x_values, y_values = transform_coordinates("EPSG:4326", src.crs, [longitude], [latitude])
            x_coord, y_coord = x_values[0], y_values[0]

        row, col = src.index(x_coord, y_coord)
        window = Window(col - 1, row - 1, 3, 3)
        data = src.read(window=window, boundless=True, masked=True)
        nodata_value = src.nodata
        band_names = list(src.descriptions) if src.descriptions else []

    features: dict[str, float] = {}
    for index, band_name in enumerate(band_names):
        output_name = band_name_map.get(band_name, band_columns[index] if index < len(band_columns) else band_name)
        band_window = data[index]
        if np.ma.isMaskedArray(band_window):
            band_mean = band_window.mean()
            features[output_name] = np.nan if np.ma.is_masked(band_mean) else float(band_mean) / band_scale
        elif nodata_value is not None:
            valid_values = band_window[band_window != nodata_value]
            features[output_name] = np.nan if valid_values.size == 0 else float(np.mean(valid_values)) / band_scale
        else:
            features[output_name] = float(np.mean(band_window)) / band_scale

    return features


def extract_ps_pixel_features(raster_path: Path, longitude: float, latitude: float) -> dict[str, float]:
    """Sample PlanetScope bands and compute indices from a local raster."""
    features = extract_window_band_means(
        raster_path=raster_path,
        longitude=longitude,
        latitude=latitude,
        band_columns=PS_BAND_COLUMNS,
        band_scale=PS_BAND_SCALE,
        band_name_map={name: name for name in PS_BAND_COLUMNS},
    )

    red = features.get("red", np.nan)
    nir = features.get("nir", np.nan)
    green = features.get("green", np.nan)
    blue = features.get("blue", np.nan)
    rededge = features.get("rededge", np.nan)
    green_i = features.get("green_i", np.nan)

    features["ndvi"] = float((nir - red) / (nir + red)) if not (pd.isna(red) or pd.isna(nir) or (nir + red) == 0) else np.nan
    features["gndvi"] = float((nir - green) / (nir + green)) if not (pd.isna(green) or pd.isna(nir) or (nir + green) == 0) else np.nan
    features["ndre"] = float((nir - rededge) / (nir + rededge)) if not (pd.isna(rededge) or pd.isna(nir) or (nir + rededge) == 0) else np.nan
    features["ndwi"] = float((green - nir) / (green + nir)) if not (pd.isna(green) or pd.isna(nir) or (nir + green) == 0) else np.nan
    features["evi"] = float(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)) if not (pd.isna(blue) or pd.isna(red) or pd.isna(nir) or (nir + 6 * red - 7.5 * blue + 1) == 0) else np.nan
    features["savi"] = float(1.5 * (nir - red) / (nir + red + 0.5)) if not (pd.isna(red) or pd.isna(nir) or (nir + red + 0.5) == 0) else np.nan
    features["cig"] = float((nir / green_i) - 1.0) if not (pd.isna(green_i) or pd.isna(nir) or green_i == 0) else np.nan
    features["ps_scene_date"] = parse_date_value(raster_path.name[:8])
    return features


def _init_frame_from_csv(input_file: Path) -> pd.DataFrame:
    """Read CSV, normalise column names, and insert coverage_class."""
    frame = pd.read_csv(input_file)
    frame = clean_column_names(frame)
    if "Location" not in frame.columns:
        raise ValueError(f"{input_file.name} does not contain a Location column")
    coverage_column = "Coverage (%)"
    if coverage_column in frame.columns:
        coverage_index = frame.columns.get_loc(coverage_column)
        frame.insert(coverage_index + 1, "coverage_class", frame[coverage_column].apply(coverage_class))
    else:
        frame["coverage_class"] = ""
    return frame


def load_cluster_dates(root_dir: Path, cluster: str) -> list[str]:
    """Unique ISO observation dates for sites in one cluster, across both summary files."""
    dates: set[str] = set()
    for filename in SUMMARY_FILES:
        path = root_dir / filename
        if not path.exists():
            continue
        frame = _init_frame_from_csv(path)
        for _, row in frame.iterrows():
            if site_cluster(str(row.get("Location", ""))) != cluster:
                continue
            parsed = parse_date_object(row.get("Date", ""))
            if parsed is not None:
                dates.add(parsed.date().isoformat())
    return sorted(dates)


def augment_with_planetscope(input_file: Path, folder_map: dict[str, list[Path]]) -> pd.DataFrame:
    """Load a summary CSV, attach PlanetScope band values from local rasters."""
    frame = _init_frame_from_csv(input_file)

    for column in PS_BAND_COLUMNS + PS_INDEX_COLUMNS:
        frame[column] = np.nan
    frame["ps_scene_date"] = ""

    for row_index, row in frame.iterrows():
        location = str(row["Location"]).strip()
        files = resolve_site_images(location, folder_map)
        if not files:
            continue

        raster_path = select_scene_by_date(files, row)
        features = extract_ps_pixel_features(raster_path, float(row["X"]), float(row["Y"]))

        for column in PS_BAND_COLUMNS + PS_INDEX_COLUMNS:
            frame.at[row_index, column] = features.get(column, np.nan)
        frame.at[row_index, "ps_scene_date"] = features.get("ps_scene_date", "")

    return frame


def _group_rows_by_location_date(frame: pd.DataFrame, folder_map: dict[str, list[Path]]) -> dict[tuple[str, str], list[dict]]:
    """Group observation rows by (site, date) so we can batch GEE requests."""
    group_records: dict[tuple[str, str], list[dict]] = {}
    for row_index, row in frame.iterrows():
        location = str(row["Location"]).strip()
        if not resolve_site_images(location, folder_map):
            continue
        group_key = (normalise_label(location), str(row.get("Date", "")).strip())
        group_records.setdefault(group_key, []).append({
            "row_index": int(row_index),
            "longitude": float(row["X"]),
            "latitude": float(row["Y"]),
            "date_value": row.get("Date", ""),
        })
    return group_records


def augment_with_sentinel2(
    input_file: Path,
    folder_map: dict[str, list[Path]],
    sentinel2_manager: Sentinel2Manager,
    acolite_scenes: list[dict] | None = None,
) -> pd.DataFrame:
    """Load a summary CSV, attach Sentinel-2 band values.

    Tries ACOLITE NetCDF output first (if `acolite_scenes` is given, from
    scan_acolite_output), sampling directly at each point; falls back to GEE
    (local-clarity scene selection) for any row ACOLITE didn't cover.
    """
    frame = _init_frame_from_csv(input_file)

    feature_columns = list(SENTINEL2_BAND_COLUMNS) + list(SENTINEL2_INDEX_COLUMNS) + ["s2_scene_date"]
    for column in feature_columns:
        frame[column] = np.nan
    frame["s2_scene_date"] = ""
    frame["s2_source"] = ""

    group_records = _group_rows_by_location_date(frame, folder_map)

    for (_, date_value), records in group_records.items():
        first_row = records[0]
        remaining = records

        if acolite_scenes:
            acolite_scene = select_acolite_scene(
                acolite_scenes, first_row["date_value"], "S2", INDONESIA_S2_WINDOW_DAYS,
            )
            if acolite_scene:
                still_needed = []
                for r in remaining:
                    sampled = sample_acolite_nc(
                        acolite_scene["path"], r["longitude"], r["latitude"], ACOLITE_S2_BAND_MAP,
                    )
                    if sampled and not all(pd.isna(v) for v in sampled.values()):
                        compute_sentinel2_indices(sampled)
                        for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
                            frame.at[r["row_index"], column] = sampled.get(column, np.nan)
                        frame.at[r["row_index"], "s2_scene_date"] = parse_date_value(acolite_scene["date"])
                        frame.at[r["row_index"], "s2_source"] = "acolite"
                    else:
                        still_needed.append(r)
                remaining = still_needed

        if not remaining:
            continue

        date_start, date_end = format_date_window(first_row["date_value"], INDONESIA_S2_WINDOW_DAYS)
        if not date_start or not date_end:
            continue

        candidates = s2_local_clear_candidates(
            first_row["longitude"], first_row["latitude"], date_start, date_end,
        )
        if not candidates:
            continue

        selected = select_by_local_clarity(candidates, first_row["date_value"])
        if selected is None:
            continue

        scene_date = parse_date_value(selected.get("date", ""))
        image = sentinel2_manager.get_image_by_asset_id(selected["asset_id"])
        if image is None:
            continue

        region_features = [
            ee.Feature(
                ee.Geometry.Point([r["longitude"], r["latitude"]]).buffer(BAND_SAMPLE_BUFFER_M).bounds(),
                {"row_index": r["row_index"]},
            )
            for r in remaining
        ]
        try:
            reduced = image.select(SENTINEL2_DOWNLOAD_BANDS).reduceRegions(
                collection=ee.FeatureCollection(region_features),
                reducer=ee.Reducer.mean(),
                scale=10,
                tileScale=2,
            ).getInfo()
        except Exception:
            continue

        for feature in reduced.get("features", []):
            props = feature.get("properties", {})
            row_index = props.get("row_index")
            if row_index is None:
                continue
            row_index = int(row_index)
            feat = build_sentinel2_feature_values(props, scene_date)
            for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
                frame.at[row_index, column] = feat.get(column, np.nan)
            frame.at[row_index, "s2_scene_date"] = feat.get("scene_date", "")
            frame.at[row_index, "s2_source"] = "gee"

    return frame


def augment_with_landsat(
    input_file: Path,
    folder_map: dict[str, list[Path]],
    landsat_manager: LandsatManager,
    acolite_scenes: list[dict] | None = None,
) -> pd.DataFrame:
    """Load a summary CSV, attach Landsat band values.

    Tries ACOLITE NetCDF output first (if `acolite_scenes` is given, from
    scan_acolite_output), sampling directly at each point; falls back to GEE
    (local-clarity scene selection) for any row ACOLITE didn't cover.
    """
    frame = _init_frame_from_csv(input_file)

    for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
        frame[column] = np.nan
    frame["ls_scene_date"] = ""
    frame["ls_source"] = ""

    group_records = _group_rows_by_location_date(frame, folder_map)

    for (_, date_value), records in group_records.items():
        first_row = records[0]
        remaining = records

        if acolite_scenes:
            acolite_scene = select_acolite_scene(
                acolite_scenes, first_row["date_value"], "LS", INDONESIA_LS_WINDOW_DAYS,
            )
            if acolite_scene:
                still_needed = []
                for r in remaining:
                    sampled = sample_acolite_nc(
                        acolite_scene["path"], r["longitude"], r["latitude"], ACOLITE_LS_BAND_MAP,
                    )
                    if sampled and not all(pd.isna(v) for v in sampled.values()):
                        compute_landsat_indices(sampled)
                        for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
                            frame.at[r["row_index"], column] = sampled.get(column, np.nan)
                        frame.at[r["row_index"], "ls_scene_date"] = parse_date_value(acolite_scene["date"])
                        frame.at[r["row_index"], "ls_source"] = "acolite"
                    else:
                        still_needed.append(r)
                remaining = still_needed

        if not remaining:
            continue

        date_start, date_end = format_date_window(first_row["date_value"], INDONESIA_LS_WINDOW_DAYS)
        if not date_start or not date_end:
            continue

        candidates = landsat_local_clear_candidates(
            first_row["longitude"], first_row["latitude"], date_start, date_end,
        )
        if not candidates:
            continue

        selected = select_by_local_clarity(candidates, first_row["date_value"])
        if selected is None:
            continue

        scene_date = parse_date_value(selected.get("date", ""))
        image = landsat_manager.get_image_by_asset_id(selected["asset_id"])
        if image is None:
            continue

        region_features = [
            ee.Feature(
                ee.Geometry.Point([r["longitude"], r["latitude"]]).buffer(BAND_SAMPLE_BUFFER_M).bounds(),
                {"row_index": r["row_index"]},
            )
            for r in remaining
        ]
        try:
            reduced = image.select(LANDSAT_DOWNLOAD_BANDS).reduceRegions(
                collection=ee.FeatureCollection(region_features),
                reducer=ee.Reducer.mean(),
                scale=30,
                tileScale=2,
            ).getInfo()
        except Exception:
            continue

        for feature in reduced.get("features", []):
            props = feature.get("properties", {})
            row_index = props.get("row_index")
            if row_index is None:
                continue
            row_index = int(row_index)
            feat = build_landsat_feature_values(props, scene_date)
            for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
                frame.at[row_index, column] = feat.get(column, np.nan)
            frame.at[row_index, "ls_scene_date"] = feat.get("ls_scene_date", "")
            frame.at[row_index, "ls_source"] = "gee"

    return frame


def build_combined_frame(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Combine the enriched site tables into a single dataset."""
    combined_frames: list[pd.DataFrame] = []
    for source_name, frame in frames:
        frame_copy = frame.copy()
        if source_name == "summary_pantai_bama.csv" and "Station" in frame_copy.columns:
            station_series = frame_copy["Station"].astype(str).str.strip()
            frame_copy["Location"] = frame_copy["Location"].astype(str).str.strip() + " Station " + station_series
        combined_frames.append(frame_copy)

    if not combined_frames:
        return pd.DataFrame()

    combined = pd.concat(combined_frames, ignore_index=True, sort=False)
    if "Station" in combined.columns:
        columns = list(combined.columns)
        columns.remove("Station")
        location_index = columns.index("Location")
        columns.insert(location_index + 1, "Station")
        combined = combined[columns]

    return combined


def augment_with_lyzenga(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Lyzenga depth-invariant bottom index columns to an enriched frame.

    Indices are computed from the s2_b* / ls_b* band columns already present.
    Rows with missing or non-positive band values yield NaN for those indices.
    """
    frame = frame.copy()
    for col in S2_LYZENGA_COLUMNS + LS_LYZENGA_COLUMNS:
        frame[col] = np.nan

    for row_index, row in frame.iterrows():
        s2_vals = {c: row.get(c, np.nan) for c in ["s2_b1", "s2_b2", "s2_b3", "s2_b4"]}
        s2_lyz = add_lyzenga_columns(s2_vals, S2_LYZENGA_PAIRS)
        for col in S2_LYZENGA_COLUMNS:
            frame.at[row_index, col] = s2_lyz.get(col, np.nan)

        ls_vals = {c: row.get(c, np.nan) for c in ["ls_b1", "ls_b2", "ls_b3", "ls_b4"]}
        ls_lyz = add_lyzenga_columns(ls_vals, LS_LYZENGA_PAIRS)
        for col in LS_LYZENGA_COLUMNS:
            frame.at[row_index, col] = ls_lyz.get(col, np.nan)

    return frame


def prepare_indonesia(
    root_dir: Path | str = DEFAULT_ROOT,
    output_suffix: str = "_with_bands",
    gee_project: str | None = None,
    acolite_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare Indonesian site CSVs enriched with Landsat, Sentinel-2, and PlanetScope bands.

    If `acolite_dir` is given, Sentinel-2/Landsat band values are sampled from ACOLITE
    NetCDF output there first (water-optimised atmospheric correction), falling back
    to GEE for any row ACOLITE didn't cover. Returns
    (planetscope_combined, sentinel2_combined, landsat_combined).
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    if gee_project is None:
        try:
            creds = load_credentials()
            gee_project = creds.get("gee_project")
        except FileNotFoundError:
            gee_project = None

    if not gee_project:
        raise FileNotFoundError("No Earth Engine project found in common/credentials/credentials.json")

    sentinel2_manager = Sentinel2Manager(gee_project=gee_project)
    landsat_manager = LandsatManager(gee_project=gee_project)

    acolite_scenes: list[dict] = []
    acolite_dir = Path(acolite_dir) if acolite_dir else None
    if acolite_dir is not None and acolite_dir.exists():
        acolite_scenes = scan_acolite_output(acolite_dir)
        print(f"Indexed {len(acolite_scenes)} ACOLITE scenes in {acolite_dir}")

    folder_map = discover_imagery_folders(root_dir)
    if not folder_map:
        print(f"Warning: no PlanetScope imagery folders found under {root_dir}; skipping PlanetScope augmentation")

    ps_frames: list[tuple[str, pd.DataFrame]] = []
    s2_frames: list[tuple[str, pd.DataFrame]] = []
    ls_frames: list[tuple[str, pd.DataFrame]] = []

    for filename in SUMMARY_FILES:
        input_file = root_dir / filename
        if not input_file.exists():
            print(f"Skipping missing file: {input_file}")
            continue

        print(f"\nProcessing {filename}…")
        if folder_map:
            ps_frames.append((filename, augment_with_planetscope(input_file, folder_map)))
        s2_frames.append((filename, augment_with_sentinel2(input_file, folder_map, sentinel2_manager, acolite_scenes)))
        ls_frames.append((filename, augment_with_landsat(input_file, folder_map, landsat_manager, acolite_scenes)))

    ps_combined = build_combined_frame(ps_frames)
    s2_combined = build_combined_frame(s2_frames)
    ls_combined = build_combined_frame(ls_frames)

    if not ps_combined.empty:
        ps_combined = augment_with_lyzenga(ps_combined)
        out = root_dir / f"summary_combined_planetscope{output_suffix}.csv"
        ps_combined.to_csv(out, index=False)
        print(f"Wrote {out}")

    if not s2_combined.empty:
        s2_combined = augment_with_lyzenga(s2_combined)
        out = root_dir / f"summary_combined_sentinel2{output_suffix}.csv"
        s2_combined.to_csv(out, index=False)
        print(f"Wrote {out}")

    if not ls_combined.empty:
        ls_combined = augment_with_lyzenga(ls_combined)
        out = root_dir / f"summary_combined_landsat{output_suffix}.csv"
        ls_combined.to_csv(out, index=False)
        print(f"Wrote {out}")

    return ps_combined, s2_combined, ls_combined


def _cmd_dates(args) -> None:
    root_dir = Path(args.root)
    clusters = list(CLUSTER_LIMITS) if args.cluster == "all" else [args.cluster]
    for cluster in clusters:
        dates = load_cluster_dates(root_dir, cluster)
        print(f"\n=== {cluster} ({CLUSTER_LIMITS[cluster]}) ===")
        print(f"Unique observation dates: {len(dates)}")
        s2_wins = merge_date_windows(dates, INDONESIA_S2_WINDOW_DAYS)
        ls_wins = merge_date_windows(dates, INDONESIA_LS_WINDOW_DAYS)
        print(f"S2 merged search windows (±{INDONESIA_S2_WINDOW_DAYS} days): {len(s2_wins)}")
        print(f"LS merged search windows (±{INDONESIA_LS_WINDOW_DAYS} days): {len(ls_wins)}")


def _cmd_download(args) -> None:
    root_dir = Path(args.root)
    limit = CLUSTER_LIMITS[args.cluster]
    dates = load_cluster_dates(root_dir, args.cluster)
    scenes_dir = Path(args.scenes)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    if args.cdse_user and args.cdse_pass:
        download_sentinel2(dates, scenes_dir, args.cdse_user, args.cdse_pass,
                            limit=limit, window_days=INDONESIA_S2_WINDOW_DAYS)
    else:
        print("Skipping Sentinel-2 (no --cdse-user / --cdse-pass)")
    if args.usgs_user and args.usgs_token:
        download_landsat(dates, scenes_dir, args.usgs_user, args.usgs_token,
                          limit=limit, window_days=INDONESIA_LS_WINDOW_DAYS)
    else:
        print("Skipping Landsat (no --usgs-user / --usgs-token)")


def _cmd_acolite(args) -> None:
    run_acolite_batch(
        scenes_dir=Path(args.scenes),
        output_dir=Path(args.acolite_out),
        limit=CLUSTER_LIMITS[args.cluster],
        acolite_path=Path(args.acolite_repo),
    )


def _cmd_build(args) -> None:
    prepare_indonesia(
        root_dir=Path(args.root),
        output_suffix=args.output_suffix,
        gee_project=args.gee_project,
        acolite_dir=Path(args.acolite_dir) if args.acolite_dir else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indonesian seagrass site pipeline (Landsat, Sentinel-2, PlanetScope).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sites fall into two clusters ~200km apart:\n"
            "  bawean  : Cinta, Jerat Lanjeng, Pamasaran, Pasir Putih, Somor-somor\n"
            "  baluran : Pantai Bama (4 stations)\n"
            "Run 'python seagrass/indonesia.py <command> --help' for per-command options."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    # dates
    p = sub.add_parser("dates", help="Show observation date summary and bounding box per cluster")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--cluster", choices=["bawean", "baluran", "all"], default="all")

    # download
    p = sub.add_parser("download", help="Download Level-1 scenes from CDSE / USGS for one cluster")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--cluster", choices=["bawean", "baluran"], required=True)
    p.add_argument("--scenes", required=True, help="Directory to save downloaded scenes")
    p.add_argument("--cdse-user", metavar="EMAIL")
    p.add_argument("--cdse-pass", metavar="PASSWORD")
    p.add_argument("--usgs-user", metavar="USERNAME")
    p.add_argument("--usgs-token", metavar="TOKEN")

    # acolite
    p = sub.add_parser("acolite", help="Run ACOLITE on downloaded scenes for one cluster")
    p.add_argument("--scenes", required=True)
    p.add_argument("--acolite-out", required=True, help="Output directory for NetCDF files")
    p.add_argument("--cluster", choices=["bawean", "baluran"], required=True)
    p.add_argument("--acolite-repo", default=str(ACOLITE_REPO_PATH))

    # build
    p = sub.add_parser("build", help="Build the CSVs from summary sheets + imagery bands")
    p.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--output-suffix", type=str, default="_with_bands")
    p.add_argument("--gee-project", type=str, default=None)
    p.add_argument("--acolite-dir", metavar="DIR",
                   help="Directory of ACOLITE NetCDF output for BOTH clusters (primary imagery source)")

    args = parser.parse_args()
    dispatch = {
        "dates": _cmd_dates,
        "download": _cmd_download,
        "acolite": _cmd_acolite,
        "build": _cmd_build,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()
