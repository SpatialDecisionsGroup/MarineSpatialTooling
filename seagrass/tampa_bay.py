"""Prepare the Tampa Bay seagrass transect dataset as CSV.

This script converts the local transect shapefiles into a tabular format, joins
point and line metadata, and leaves placeholders for future SuperDove/PlanetScope
features. Sentinel-2 sampling is supported when a survey date is available.
"""

from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

from common.common import format_date_window, parse_date_object
from common.sentinel import (
    SENTINEL2_BAND_COLUMNS,
    SENTINEL2_DOWNLOAD_BANDS,
    SENTINEL2_INDEX_COLUMNS,
    add_sentinel2_columns,
    build_sentinel2_feature_values,
)
from superres.sentinel2 import Sentinel2Manager
import argparse


DEFAULT_ROOT = Path(".")
POINTS_FILE = "seagrass_transect_points.shp"
LINES_FILE = "seagrass_transect_lines.shp"
SENTINEL2_WINDOW_DAYS = 15
PLACEHOLDER_COLUMNS = [
    "superdove_status",
    "superdove_notes",
    "planetscope_status",
    "planetscope_notes",
]


def load_points(root_dir):
    """Load the transect point shapefile and normalise its columns."""
    points = gpd.read_file(root_dir / POINTS_FILE)
    points = points.rename(columns={"TRAN_ID": "transect_id", "Species": "species", "Presenc": "presence", "MnAbndn": "mean_abundance"})
    points = points.to_crs("EPSG:4326")
    points["latitude"] = points.geometry.y
    points["longitude"] = points.geometry.x
    points["point_wkt"] = points.geometry.to_wkt()
    return points


def load_lines(root_dir):
    """Load the transect line shapefile and normalise its columns."""
    lines = gpd.read_file(root_dir / LINES_FILE)
    lines = lines.rename(columns={"Site": "transect_id", "TtlSpcs": "total_species", "TtlPrsn": "total_presence", "ActivYN": "active"})
    lines = lines.to_crs("EPSG:4326")
    lines["line_wkt"] = lines.geometry.to_wkt()
    return lines


def build_base_table(root_dir):
    """Build the base transect CSV from the local shapefiles."""
    points = load_points(root_dir)
    lines = load_lines(root_dir)
    line_cols = [col for col in ["transect_id", "bearing", "active", "total_species", "total_presence", "Shp_Lng", "MnAgncy", "Commnts", "line_wkt"] if col in lines.columns]
    line_frame = lines[line_cols].drop_duplicates(subset=["transect_id"]) if "transect_id" in line_cols else lines[line_cols].drop_duplicates()

    merged = points.merge(line_frame, on="transect_id", how="left", suffixes=("", "_line"))
    merged = merged.drop(columns=[col for col in ["geometry"] if col in merged.columns])
    merged["dataset"] = "Tampa Bay seagrass transects"
    merged["survey_date"] = pd.NA

    for column in PLACEHOLDER_COLUMNS:
        merged[column] = "pending"

    return pd.DataFrame(merged)

def parse_survey_date(value):
    """Parse a survey date value, if available."""
    return parse_date_object(value)

def sample_sentinel2_features(sentinel2_manager, longitude, latitude, survey_date):
    """Sample Sentinel-2 features for a single point when a survey date is available."""
    date_start, date_end = format_date_window(survey_date, SENTINEL2_WINDOW_DAYS)
    if not date_start or not date_end:
        return None

    images_info = sentinel2_manager.retrieve_images(latitude, longitude, date_start, date_end, 8)
    if not images_info:
        return None

    # Prefer the temporally closest scene from the date window.
    selected = None
    row_date = parse_survey_date(survey_date)
    if row_date is not None:
        exact_matches: list[tuple[int, dict]] = []
        nearest_candidate: tuple[int, int, dict] | None = None
        for index, image_info in enumerate(images_info):
            image_date = parse_date_object(image_info.get("date", ""))
            if image_date is None:
                continue
            if image_date.date() == row_date.date():
                exact_matches.append((index, image_info))
            else:
                distance = abs((image_date - row_date).days)
                if nearest_candidate is None or distance < nearest_candidate[0]:
                    nearest_candidate = (distance, index, image_info)
        if exact_matches:
            selected = exact_matches[0]
        elif nearest_candidate is not None:
            selected = (nearest_candidate[1], nearest_candidate[2])
    if selected is None:
        selected = (0, images_info[0])

    index, scene_info = selected
    image = sentinel2_manager.get_image_by_index(latitude, longitude, date_start, date_end, index)
    if image is None:
        return None

    region = ee.Geometry.Point([longitude, latitude]).buffer(15).bounds()
    try:
        stats = image.select(SENTINEL2_DOWNLOAD_BANDS).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,
            bestEffort=True,
            maxPixels=1_000_000,
        ).getInfo()
    except Exception:
        return None

    if not stats:
        return None

    scene_date_parsed = parse_survey_date(scene_info.get("date", ""))
    scene_date_str = scene_date_parsed.strftime("%Y%m%d") if scene_date_parsed is not None else ""

    return build_sentinel2_feature_values(stats, scene_date_str)


def prepare_transect_csv(root_dir, output_file, gee_project=None, survey_date_column=None):
    """Prepare the Tampa Bay transect dataset as a CSV."""
    frame = build_base_table(root_dir)
    frame = add_sentinel2_columns(frame)

    sentinel2_manager = Sentinel2Manager(gee_project=gee_project) if gee_project else None

    if sentinel2_manager and survey_date_column and survey_date_column in frame.columns:
        for row_index, row in frame.iterrows():
            survey_date = row.get(survey_date_column)
            features = sample_sentinel2_features(
                sentinel2_manager=sentinel2_manager,
                longitude=float(row["longitude"]),
                latitude=float(row["latitude"]),
                survey_date=survey_date,
            )
            if features is None:
                continue
            for column in SENTINEL2_BAND_COLUMNS + SENTINEL2_INDEX_COLUMNS:
                frame.at[row_index, column] = features.get(column, np.nan)
            frame.at[row_index, "scene_date"] = features.get("scene_date", "")
            frame.at[row_index, "sentinel2_status"] = "sampled"
    else:
        frame["sentinel2_status"] = "pending_date"

    frame.to_csv(output_file, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the Tampa Bay transect CSV with optional Sentinel-2 sampling.")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="Root directory containing shapefiles")
    parser.add_argument("--output", "-o", type=str, default="tampa_transects_with_bands.csv", help="Output CSV path")
    parser.add_argument("--gee-project", type=str, default=None, help="Google Earth Engine project ID (optional)")
    parser.add_argument("--survey-date-column", type=str, default="survey_date", help="Column name with survey dates (optional)")
    args = parser.parse_args()

    out_frame = prepare_transect_csv(Path(args.root), args.output, gee_project=args.gee_project, survey_date_column=args.survey_date_column)
    print(f"Wrote {args.output}")
