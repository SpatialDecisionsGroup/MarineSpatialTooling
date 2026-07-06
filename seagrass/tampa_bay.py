"""Prepare the Tampa Bay seagrass transect dataset as CSV.

Converts the local transect shapefiles into a tabular format, then attaches
imagery band values from three sources:

  - Landsat 8/9 (30 m, low-res): sampled from Google Earth Engine.
  - Sentinel-2 (10 m, high-res): sampled from Google Earth Engine.
  - PlanetScope (3 m, for later comparison): read from pre-downloaded local
    rasters pointed to by --planetscope-raster-dir.  Not downloaded
    automatically because PlanetScope is quota-constrained.

A Random Forest classification accuracy comparison across Landsat and
Sentinel-2 (and PlanetScope if available) is saved to accuracy_comparison.csv.
"""

from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as transform_coordinates

from common.common import format_date_window, parse_date_object, parse_date_value
from common.sentinel import (
    SENTINEL2_BAND_COLUMNS,
    SENTINEL2_DOWNLOAD_BANDS,
    SENTINEL2_INDEX_COLUMNS,
    add_sentinel2_columns,
    build_sentinel2_feature_values,
)
from common.landsat import (
    LANDSAT_BAND_COLUMNS,
    LANDSAT_DOWNLOAD_BANDS,
    LANDSAT_INDEX_COLUMNS,
    LANDSAT_WINDOW_DAYS,
    add_landsat_columns,
    build_landsat_feature_values,
)
from common.sentinel2 import Sentinel2Manager
from common.landsat import LandsatManager
import argparse


DEFAULT_ROOT = Path(".")
POINTS_FILE = "seagrass_transect_points.shp"
LINES_FILE = "seagrass_transect_lines.shp"
SENTINEL2_WINDOW_DAYS = 15

# PlanetScope 8-band SuperDove columns (same format as Indonesia).
PS_BAND_COLUMNS = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
PS_INDEX_COLUMNS = ["ps_ndvi", "ps_gndvi", "ps_ndre", "ps_ndwi", "ps_evi", "ps_savi", "ps_cig"]
PS_BAND_SCALE = 10000.0


def load_points(root_dir):
    points = gpd.read_file(root_dir / POINTS_FILE)
    points = points.rename(columns={"TRAN_ID": "transect_id", "Species": "species", "Presenc": "presence", "MnAbndn": "mean_abundance"})
    points = points.to_crs("EPSG:4326")
    points["latitude"] = points.geometry.y
    points["longitude"] = points.geometry.x
    points["point_wkt"] = points.geometry.to_wkt()
    return points


def load_lines(root_dir):
    lines = gpd.read_file(root_dir / LINES_FILE)
    lines = lines.rename(columns={"Site": "transect_id", "TtlSpcs": "total_species", "TtlPrsn": "total_presence", "ActivYN": "active"})
    lines = lines.to_crs("EPSG:4326")
    lines["line_wkt"] = lines.geometry.to_wkt()
    return lines


def build_base_table(root_dir):
    points = load_points(root_dir)
    lines = load_lines(root_dir)
    line_cols = [col for col in ["transect_id", "bearing", "active", "total_species", "total_presence", "Shp_Lng", "MnAgncy", "Commnts", "line_wkt"] if col in lines.columns]
    line_frame = lines[line_cols].drop_duplicates(subset=["transect_id"]) if "transect_id" in line_cols else lines[line_cols].drop_duplicates()

    merged = points.merge(line_frame, on="transect_id", how="left", suffixes=("", "_line"))
    merged = merged.drop(columns=[col for col in ["geometry"] if col in merged.columns])
    merged["dataset"] = "Tampa Bay seagrass transects"
    merged["survey_date"] = pd.NA
    return pd.DataFrame(merged)


def _select_scene_info_by_date(images_info: list[dict], survey_date) -> tuple[int, dict]:
    row_date = parse_date_object(survey_date)
    if row_date is None:
        return 0, images_info[0]

    nearest_candidate: tuple[int, int, dict] | None = None
    for index, image_info in enumerate(images_info):
        image_date = parse_date_object(image_info.get("date", ""))
        if image_date is None:
            continue
        if image_date.date() == row_date.date():
            return index, image_info
        distance = abs((image_date - row_date).days)
        if nearest_candidate is None or distance < nearest_candidate[0]:
            nearest_candidate = (distance, index, image_info)

    if nearest_candidate is not None:
        return nearest_candidate[1], nearest_candidate[2]
    return 0, images_info[0]


def _sample_gee_point(
    manager,
    longitude: float,
    latitude: float,
    survey_date,
    window_days: int,
    download_bands: list[str],
    scale_meters: int,
    buffer_meters: int,
    build_features_fn,
) -> dict | None:
    """Sample any GEE-backed satellite at a single point and return feature dict."""
    date_start, date_end = format_date_window(survey_date, window_days)
    if not date_start or not date_end:
        return None

    images_info = manager.retrieve_images(latitude, longitude, date_start, date_end, 8)
    if not images_info:
        return None

    index, scene_info = _select_scene_info_by_date(images_info, survey_date)
    image = manager.get_image_by_index(latitude, longitude, date_start, date_end, index)
    if image is None:
        return None

    region = ee.Geometry.Point([longitude, latitude]).buffer(buffer_meters).bounds()
    try:
        stats = image.select(download_bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale_meters,
            bestEffort=True,
            maxPixels=1_000_000,
        ).getInfo()
    except Exception:
        return None

    if not stats:
        return None

    scene_date_parsed = parse_date_object(scene_info.get("date", ""))
    scene_date_str = scene_date_parsed.strftime("%Y%m%d") if scene_date_parsed is not None else ""
    return build_features_fn(stats, scene_date_str)


def _extract_raster_point(
    raster_path: Path,
    longitude: float,
    latitude: float,
    band_columns: list[str],
    band_scale: float,
) -> dict[str, float]:
    """Sample a 3×3 window from a local raster at the given coordinate."""
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            x_coord, y_coord = longitude, latitude
        else:
            xs, ys = transform_coordinates("EPSG:4326", src.crs, [longitude], [latitude])
            x_coord, y_coord = xs[0], ys[0]

        row, col = src.index(x_coord, y_coord)
        window = Window(col - 1, row - 1, 3, 3)
        data = src.read(window=window, boundless=True, masked=True)
        nodata_value = src.nodata

    features: dict[str, float] = {}
    for i, col_name in enumerate(band_columns):
        if i >= data.shape[0]:
            features[col_name] = np.nan
            continue
        band = data[i]
        if np.ma.isMaskedArray(band):
            mean = band.mean()
            features[col_name] = np.nan if np.ma.is_masked(mean) else float(mean) / band_scale
        elif nodata_value is not None:
            valid = band[band != nodata_value]
            features[col_name] = np.nan if valid.size == 0 else float(np.mean(valid)) / band_scale
        else:
            features[col_name] = float(np.mean(band)) / band_scale
    return features


def _compute_ps_indices(features: dict[str, float]) -> dict[str, float]:
    red = features.get("red", np.nan)
    nir = features.get("nir", np.nan)
    green = features.get("green", np.nan)
    blue = features.get("blue", np.nan)
    rededge = features.get("rededge", np.nan)
    green_i = features.get("green_i", np.nan)

    features["ps_ndvi"] = float((nir - red) / (nir + red)) if not (pd.isna(red) or pd.isna(nir) or (nir + red) == 0) else np.nan
    features["ps_gndvi"] = float((nir - green) / (nir + green)) if not (pd.isna(green) or pd.isna(nir) or (nir + green) == 0) else np.nan
    features["ps_ndre"] = float((nir - rededge) / (nir + rededge)) if not (pd.isna(rededge) or pd.isna(nir) or (nir + rededge) == 0) else np.nan
    features["ps_ndwi"] = float((green - nir) / (green + nir)) if not (pd.isna(green) or pd.isna(nir) or (nir + green) == 0) else np.nan
    features["ps_evi"] = float(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)) if not (pd.isna(blue) or pd.isna(red) or pd.isna(nir) or (nir + 6 * red - 7.5 * blue + 1) == 0) else np.nan
    features["ps_savi"] = float(1.5 * (nir - red) / (nir + red + 0.5)) if not (pd.isna(red) or pd.isna(nir) or (nir + red + 0.5) == 0) else np.nan
    features["ps_cig"] = float((nir / green_i) - 1.0) if not (pd.isna(green_i) or pd.isna(nir) or green_i == 0) else np.nan
    return features


def prepare_transect_csv(
    root_dir,
    output_file,
    gee_project=None,
    survey_date_column=None,
    planetscope_raster_dir=None,
):
    """Prepare the Tampa Bay transect dataset with Landsat, Sentinel-2, and PlanetScope bands."""
    root_dir = Path(root_dir)
    output_file = Path(output_file)

    frame = build_base_table(root_dir)
    frame = add_sentinel2_columns(frame)
    frame = add_landsat_columns(frame)

    # PlanetScope columns — present if raster directory provided, else marked unavailable.
    for column in PS_BAND_COLUMNS + PS_INDEX_COLUMNS:
        frame[column] = np.nan
    frame["ps_scene_date"] = ""
    frame["planetscope_status"] = "not_available" if planetscope_raster_dir is None else "pending"

    sentinel2_manager = Sentinel2Manager(gee_project=gee_project) if gee_project else None
    landsat_manager = LandsatManager(gee_project=gee_project) if gee_project else None

    # Discover PlanetScope rasters if a directory was supplied.
    ps_tif_files: list[Path] = []
    if planetscope_raster_dir is not None:
        ps_raster_path = Path(planetscope_raster_dir)
        ps_tif_files = sorted(ps_raster_path.glob("*.tif")) + sorted(ps_raster_path.glob("*.TIF"))
        if not ps_tif_files:
            print(f"Warning: no TIF files found in {ps_raster_path}; skipping PlanetScope")

    has_date_col = survey_date_column and survey_date_column in frame.columns

    for row_index, row in frame.iterrows():
        longitude = float(row["longitude"])
        latitude = float(row["latitude"])
        survey_date = row.get(survey_date_column) if has_date_col else None

        # Sentinel-2
        if sentinel2_manager and survey_date is not None and pd.notna(survey_date):
            features = _sample_gee_point(
                manager=sentinel2_manager,
                longitude=longitude,
                latitude=latitude,
                survey_date=survey_date,
                window_days=SENTINEL2_WINDOW_DAYS,
                download_bands=SENTINEL2_DOWNLOAD_BANDS,
                scale_meters=10,
                buffer_meters=15,
                build_features_fn=build_sentinel2_feature_values,
            )
            if features is not None:
                for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
                    frame.at[row_index, column] = features.get(column, np.nan)
                frame.at[row_index, "scene_date"] = features.get("scene_date", "")
                frame.at[row_index, "sentinel2_status"] = "sampled"

        # Landsat
        if landsat_manager and survey_date is not None and pd.notna(survey_date):
            features = _sample_gee_point(
                manager=landsat_manager,
                longitude=longitude,
                latitude=latitude,
                survey_date=survey_date,
                window_days=LANDSAT_WINDOW_DAYS,
                download_bands=LANDSAT_DOWNLOAD_BANDS,
                scale_meters=30,
                buffer_meters=45,
                build_features_fn=build_landsat_feature_values,
            )
            if features is not None:
                for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
                    frame.at[row_index, column] = features.get(column, np.nan)
                frame.at[row_index, "ls_scene_date"] = features.get("ls_scene_date", "")
                frame.at[row_index, "landsat_status"] = "sampled"

        # PlanetScope (local rasters only)
        if ps_tif_files:
            survey_date_obj = parse_date_object(survey_date) if survey_date is not None and pd.notna(survey_date) else None
            best_file = ps_tif_files[0]
            if survey_date_obj is not None:
                nearest: tuple[int, Path] | None = None
                for tif in ps_tif_files:
                    tif_date = parse_date_object(tif.stem[:8])
                    if tif_date is None:
                        continue
                    dist = abs((tif_date - survey_date_obj).days)
                    if nearest is None or dist < nearest[0]:
                        nearest = (dist, tif)
                if nearest is not None:
                    best_file = nearest[1]

            try:
                band_vals = _extract_raster_point(best_file, longitude, latitude, PS_BAND_COLUMNS, PS_BAND_SCALE)
                band_vals = _compute_ps_indices(band_vals)
                for column in PS_BAND_COLUMNS + PS_INDEX_COLUMNS:
                    frame.at[row_index, column] = band_vals.get(column, np.nan)
                file_date = parse_date_object(best_file.stem[:8])
                frame.at[row_index, "ps_scene_date"] = file_date.strftime("%Y%m%d") if file_date else ""
                frame.at[row_index, "planetscope_status"] = "sampled"
            except Exception as exc:
                frame.at[row_index, "planetscope_status"] = f"error: {exc}"

    frame.to_csv(output_file, index=False)
    print(f"Wrote {output_file}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the Tampa Bay transect CSV with Landsat, Sentinel-2, and PlanetScope bands."
    )
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--output", "-o", type=str, default="tampa_transects_with_bands.csv")
    parser.add_argument("--gee-project", type=str, default=None)
    parser.add_argument("--survey-date-column", type=str, default="survey_date")
    parser.add_argument("--planetscope-raster-dir", type=str, default=None,
                        help="Directory containing pre-downloaded PlanetScope TIF files")
    args = parser.parse_args()

    prepare_transect_csv(
        root_dir=Path(args.root),
        output_file=args.output,
        gee_project=args.gee_project,
        survey_date_column=args.survey_date_column,
        planetscope_raster_dir=args.planetscope_raster_dir,
    )
