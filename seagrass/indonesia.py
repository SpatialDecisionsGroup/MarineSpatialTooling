"""Attach PlanetScope and Sentinel-2 band values to the Indonesian seagrass CSVs.

The site spreadsheets live alongside the downloaded PlanetScope folders in:
/home/ysi/Documents/Uni/MQPostdoc/projects/superres/indonesian_seagrass

This script reads the two summary CSVs, matches each location to the correct
download folder, samples the relevant raster band values at each row's point,
and writes augmented CSVs with explicit numeric predictor columns.
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
from common.sentinel import (
	SENTINEL2_BAND_COLUMNS,
	SENTINEL2_DOWNLOAD_BANDS,
	SENTINEL2_BAND_NAME_MAP,
	SENTINEL2_SCALE,
	SENTINEL2_WINDOW_DAYS,
	build_sentinel2_feature_values,
)
from superres.sentinel2 import Sentinel2Manager
import argparse


DEFAULT_ROOT = Path(".")
SUMMARY_FILES = ["summary.csv", "summary_pantai_bama.csv"]
BAND_COLUMNS = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]
BAND_SCALE = 10000.0
# Sentinel-2 constants and helper functions are provided by common.sentinel


def normalise_label(value: str) -> str:
	"""Normalize folder and site labels so they can be matched reliably."""
	value = value.strip().lower()
	value = value.replace("_", " ")
	value = value.replace("-", " ")
	value = re.sub(r"\s+", " ", value)
	return value


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
	"""Strip whitespace and normalize the unnamed date column."""
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
	"""Map numeric coverage percentages into low/medium/high categories."""
	numeric_value = pd.to_numeric(value, errors="coerce")
	if pd.isna(numeric_value):
		return ""
	if numeric_value < 30:
		return "low"
	if numeric_value < 60:
		return "medium"
	return "high"


def parse_date_value(value: object) -> str:
	"""Convert a CSV date value to YYYYMMDD when possible."""
	parsed = parse_date_object(value)
	if parsed is None:
		return ""
	return parsed.strftime("%Y%m%d")


def parse_date_object(value: object) -> pd.Timestamp | None:
	"""Parse a date value into a Timestamp when possible."""
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


def format_date_window(value: object, window_days: int = SENTINEL2_WINDOW_DAYS) -> tuple[str, str] | tuple[None, None]:
	"""Build a date window around a CSV date value."""
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
	"""Resolve the imagery files for one site name."""
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


def select_scene(files: list[Path], row: pd.Series) -> Path:
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
	"""Sample a 3x3 raster neighborhood around a point and return normalized band means."""
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
		output_name = band_name_map.get(band_name, band_columns[index])
		band_window = data[index]
		if np.ma.isMaskedArray(band_window):
			band_mean = band_window.mean()
			if np.ma.is_masked(band_mean):
				features[output_name] = np.nan
			else:
				features[output_name] = float(band_mean) / band_scale
		elif nodata_value is not None:
			valid_values = band_window[band_window != nodata_value]
			if valid_values.size == 0:
				features[output_name] = np.nan
			else:
				features[output_name] = float(np.mean(valid_values)) / band_scale
		else:
			features[output_name] = float(np.mean(band_window)) / band_scale

	return features


def extract_pixel_features(raster_path: Path, longitude: float, latitude: float) -> dict[str, float]:
	"""Sample a 3x3 raster neighborhood around a point and compute indices."""
	features = extract_window_band_means(
		raster_path=raster_path,
		longitude=longitude,
		latitude=latitude,
		band_columns=BAND_COLUMNS,
		band_scale=BAND_SCALE,
		band_name_map={name: name for name in BAND_COLUMNS},
	)

	red = features.get("red", np.nan)
	nir = features.get("nir", np.nan)
	green = features.get("green", np.nan)
	blue = features.get("blue", np.nan)
	rededge = features.get("rededge", np.nan)
	green_i = features.get("green_i", np.nan)
	if pd.isna(red) or pd.isna(nir) or (nir + red) == 0:
		features["ndvi"] = np.nan
	else:
		features["ndvi"] = float((nir - red) / (nir + red))

	if pd.isna(green) or pd.isna(nir) or (nir + green) == 0:
		features["gndvi"] = np.nan
	else:
		features["gndvi"] = float((nir - green) / (nir + green))

	if pd.isna(rededge) or pd.isna(nir) or (nir + rededge) == 0:
		features["ndre"] = np.nan
	else:
		features["ndre"] = float((nir - rededge) / (nir + rededge))

	if pd.isna(green) or pd.isna(nir) or (nir + green) == 0:
		features["ndwi"] = np.nan
	else:
		features["ndwi"] = float((green - nir) / (green + nir))

	if pd.isna(blue) or pd.isna(red) or pd.isna(nir) or (nir + 6 * red - 7.5 * blue + 1) == 0:
		features["evi"] = np.nan
	else:
		features["evi"] = float(2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1))

	if pd.isna(red) or pd.isna(nir) or (nir + red + 0.5) == 0:
		features["savi"] = np.nan
	else:
		features["savi"] = float(1.5 * (nir - red) / (nir + red + 0.5))

	if pd.isna(green_i) or pd.isna(nir) or (green_i == 0):
		features["cig"] = np.nan
	else:
		features["cig"] = float((nir / green_i) - 1.0)

	features["scene_date"] = parse_date_value(raster_path.name[:8])
	return features


def extract_sentinel2_features(raster_path: Path, longitude: float, latitude: float) -> dict[str, float]:
	"""Sample a 3x3 Sentinel-2 raster neighborhood and compute indices."""
	features = extract_window_band_means(
		raster_path=raster_path,
		longitude=longitude,
		latitude=latitude,
		band_columns=SENTINEL2_BAND_COLUMNS,
		band_scale=SENTINEL2_SCALE,
		band_name_map=SENTINEL2_BAND_NAME_MAP,
	)

	b1 = features.get("s2_b1", np.nan)
	b2 = features.get("s2_b2", np.nan)
	b3 = features.get("s2_b3", np.nan)
	b4 = features.get("s2_b4", np.nan)
	b5 = features.get("s2_b5", np.nan)
	b8 = features.get("s2_b8", np.nan)
	b8a = features.get("s2_b8a", np.nan)
	b11 = features.get("s2_b11", np.nan)
	b12 = features.get("s2_b12", np.nan)

	if pd.isna(b8) or pd.isna(b4) or (b8 + b4) == 0:
		features["ndvi"] = np.nan
	else:
		features["ndvi"] = float((b8 - b4) / (b8 + b4))

	if pd.isna(b8) or pd.isna(b3) or (b8 + b3) == 0:
		features["gndvi"] = np.nan
	else:
		features["gndvi"] = float((b8 - b3) / (b8 + b3))

	if pd.isna(b8a) or pd.isna(b5) or (b8a + b5) == 0:
		features["ndre"] = np.nan
	else:
		features["ndre"] = float((b8a - b5) / (b8a + b5))

	if pd.isna(b3) or pd.isna(b8) or (b3 + b8) == 0:
		features["ndwi"] = np.nan
	else:
		features["ndwi"] = float((b3 - b8) / (b3 + b8))

	if pd.isna(b3) or pd.isna(b11) or (b3 + b11) == 0:
		features["mndwi"] = np.nan
	else:
		features["mndwi"] = float((b3 - b11) / (b3 + b11))

	if pd.isna(b2) or pd.isna(b4) or pd.isna(b8) or (b8 + 6 * b4 - 7.5 * b2 + 1) == 0:
		features["evi"] = np.nan
	else:
		features["evi"] = float(2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1))

	if pd.isna(b8) or pd.isna(b4) or (b8 + b4 + 0.5) == 0:
		features["savi"] = np.nan
	else:
		features["savi"] = float(1.5 * (b8 - b4) / (b8 + b4 + 0.5))

	if pd.isna(b8) or pd.isna(b12) or (b8 + b12) == 0:
		features["nbr"] = np.nan
	else:
		features["nbr"] = float((b8 - b12) / (b8 + b12))

	features["scene_date"] = parse_date_value(raster_path.name[:8])
	return features


# Sentinel-2 feature construction is provided by `common.sentinel.build_sentinel2_feature_values`


def augment_summary_csv(input_file: Path, root_dir: Path, output_file: Path, folder_map: dict[str, list[Path]]) -> pd.DataFrame:
	"""Load a summary CSV, attach band values, and write the enriched file."""
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

	band_columns = list(BAND_COLUMNS) + ["ndvi", "gndvi", "ndre", "ndwi", "evi", "savi", "cig", "scene_date"]
	for column in band_columns:
		frame[column] = np.nan
	frame["scene_date"] = ""

	for row_index, row in frame.iterrows():
		location = str(row["Location"]).strip()
		files = resolve_site_images(location, folder_map)

		if not files:
			continue

		raster_path = select_scene(files, row)
		longitude = float(row["X"])
		latitude = float(row["Y"])
		features = extract_pixel_features(raster_path, longitude, latitude)

		for column in BAND_COLUMNS:
			frame.at[row_index, column] = features.get(column, np.nan)
		frame.at[row_index, "ndvi"] = features.get("ndvi", np.nan)
		frame.at[row_index, "gndvi"] = features.get("gndvi", np.nan)
		frame.at[row_index, "ndre"] = features.get("ndre", np.nan)
		frame.at[row_index, "ndwi"] = features.get("ndwi", np.nan)
		frame.at[row_index, "evi"] = features.get("evi", np.nan)
		frame.at[row_index, "savi"] = features.get("savi", np.nan)
		frame.at[row_index, "cig"] = features.get("cig", np.nan)
		frame.at[row_index, "scene_date"] = features.get("scene_date", "")

	return frame


def select_sentinel2_scene(images_info: list[dict], row: pd.Series) -> tuple[int, dict] | None:
	"""Pick the best Sentinel-2 scene for a row, preferring a date match."""
	row_date = parse_date_object(row.get("Date", ""))
	if row_date is None:
		return 0, images_info[0]

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
		return exact_matches[0]
	if nearest_candidate is not None:
		return nearest_candidate[1], nearest_candidate[2]
	return 0, images_info[0]


def sample_sentinel2_features(
	sentinel2_manager: Sentinel2Manager,
	longitude: float,
	latitude: float,
	date_value: object,
) -> dict[str, float] | None:
	"""Sample Sentinel-2 bands and indices from Earth Engine for one row."""
	date_start, date_end = format_date_window(date_value)
	if not date_start or not date_end:
		return None

	images_info = sentinel2_manager.retrieve_images(
		latitude,
		longitude,
		date_start,
		date_end,
		8,
	)
	if not images_info:
		return None

	selected = select_sentinel2_scene(images_info, pd.Series({"Date": date_value}))
	if selected is None:
		return None

	index, scene_info = selected

	image = sentinel2_manager.get_image_by_index(
		latitude,
		longitude,
		date_start,
		date_end,
		index,
	)
	if image is None:
		return None

	image = image.select(SENTINEL2_DOWNLOAD_BANDS)
	region = ee.Geometry.Point([longitude, latitude]).buffer(15).bounds()
	try:
		stats = image.reduceRegion(
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

	features: dict[str, float] = {}
	for index, band_name in enumerate(SENTINEL2_DOWNLOAD_BANDS):
		output_name = SENTINEL2_BAND_NAME_MAP.get(band_name, SENTINEL2_BAND_COLUMNS[index])
		value = stats.get(band_name)
		if value is None:
			features[output_name] = np.nan
		else:
			features[output_name] = float(value) / SENTINEL2_SCALE

	b2 = features.get("s2_b2", np.nan)
	b3 = features.get("s2_b3", np.nan)
	b4 = features.get("s2_b4", np.nan)
	b5 = features.get("s2_b5", np.nan)
	b8 = features.get("s2_b8", np.nan)
	b8a = features.get("s2_b8a", np.nan)
	b11 = features.get("s2_b11", np.nan)
	b12 = features.get("s2_b12", np.nan)

	if pd.isna(b8) or pd.isna(b4) or (b8 + b4) == 0:
		features["ndvi"] = np.nan
	else:
		features["ndvi"] = float((b8 - b4) / (b8 + b4))

	if pd.isna(b8) or pd.isna(b3) or (b8 + b3) == 0:
		features["gndvi"] = np.nan
	else:
		features["gndvi"] = float((b8 - b3) / (b8 + b3))

	if pd.isna(b8a) or pd.isna(b5) or (b8a + b5) == 0:
		features["ndre"] = np.nan
	else:
		features["ndre"] = float((b8a - b5) / (b8a + b5))

	if pd.isna(b3) or pd.isna(b8) or (b3 + b8) == 0:
		features["ndwi"] = np.nan
	else:
		features["ndwi"] = float((b3 - b8) / (b3 + b8))

	if pd.isna(b3) or pd.isna(b11) or (b3 + b11) == 0:
		features["mndwi"] = np.nan
	else:
		features["mndwi"] = float((b3 - b11) / (b3 + b11))

	if pd.isna(b2) or pd.isna(b4) or pd.isna(b8) or (b8 + 6 * b4 - 7.5 * b2 + 1) == 0:
		features["evi"] = np.nan
	else:
		features["evi"] = float(2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1))

	if pd.isna(b8) or pd.isna(b4) or (b8 + b4 + 0.5) == 0:
		features["savi"] = np.nan
	else:
		features["savi"] = float(1.5 * (b8 - b4) / (b8 + b4 + 0.5))

	if pd.isna(b8) or pd.isna(b12) or (b8 + b12) == 0:
		features["nbr"] = np.nan
	else:
		features["nbr"] = float((b8 - b12) / (b8 + b12))

	features["scene_date"] = parse_date_value(scene_info.get("date", ""))
	return features


def augment_summary_csv_sentinel2(
	input_file: Path,
	output_file: Path,
	folder_map: dict[str, list[Path]],
	sentinel2_manager: Sentinel2Manager,
) -> pd.DataFrame:
	"""Load a summary CSV, attach Sentinel-2 band values, and write the enriched file."""
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

	feature_columns = list(SENTINEL2_BAND_COLUMNS) + ["ndvi", "gndvi", "ndre", "ndwi", "mndwi", "evi", "savi", "nbr", "scene_date"]
	for column in feature_columns:
		frame[column] = np.nan
	frame["scene_date"] = ""

	group_records: dict[tuple[str, str], list[dict]] = {}
	for row_index, row in frame.iterrows():
		location = str(row["Location"]).strip()
		files = resolve_site_images(location, folder_map)
		if not files:
			continue

		group_key = (normalise_label(location), str(row.get("Date", "")).strip())
		group_records.setdefault(group_key, []).append(
			{
				"row_index": int(row_index),
				"longitude": float(row["X"]),
				"latitude": float(row["Y"]),
				"date_value": row.get("Date", ""),
			}
		)

	for (normalized_location, date_value), records in group_records.items():
		first_row = records[0]
		date_start, date_end = format_date_window(first_row["date_value"])
		if not date_start or not date_end:
			continue

		images_info = sentinel2_manager.retrieve_images(
			first_row["latitude"],
			first_row["longitude"],
			date_start,
			date_end,
			8,
		)
		if not images_info:
			continue

		selected = select_sentinel2_scene(images_info, pd.Series({"Date": first_row["date_value"]}))
		if selected is None:
			continue

		index, scene_info = selected
		scene_date = parse_date_value(scene_info.get("date", ""))
		image = sentinel2_manager.get_image_by_index(
			first_row["latitude"],
			first_row["longitude"],
			date_start,
			date_end,
			index,
		)
		if image is None:
			continue

		region_features = []
		for record in records:
			region = ee.Geometry.Point([record["longitude"], record["latitude"]]).buffer(15).bounds()
			region_features.append(ee.Feature(region, {"row_index": record["row_index"]}))

		feature_collection = ee.FeatureCollection(region_features)
		try:
			reduced = image.select(SENTINEL2_DOWNLOAD_BANDS).reduceRegions(
				collection=feature_collection,
				reducer=ee.Reducer.mean(),
				scale=10,
				tileScale=2,
			).getInfo()
		except Exception:
			continue

		for feature in reduced.get("features", []):
			properties = feature.get("properties", {})
			row_index = properties.get("row_index")
			if row_index is None:
				continue
			row_index = int(row_index)
			features = build_sentinel2_feature_values(properties, scene_date)
			for column in SENTINEL2_BAND_COLUMNS:
				frame.at[row_index, column] = features.get(column, np.nan)
			frame.at[row_index, "ndvi"] = features.get("ndvi", np.nan)
			frame.at[row_index, "gndvi"] = features.get("gndvi", np.nan)
			frame.at[row_index, "ndre"] = features.get("ndre", np.nan)
			frame.at[row_index, "ndwi"] = features.get("ndwi", np.nan)
			frame.at[row_index, "mndwi"] = features.get("mndwi", np.nan)
			frame.at[row_index, "evi"] = features.get("evi", np.nan)
			frame.at[row_index, "savi"] = features.get("savi", np.nan)
			frame.at[row_index, "nbr"] = features.get("nbr", np.nan)
			frame.at[row_index, "scene_date"] = features.get("scene_date", "")

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
		# Keep station available in the combined dataset, but put it after Location for readability.
		columns = list(combined.columns)
		columns.remove("Station")
		location_index = columns.index("Location")
		columns.insert(location_index + 1, "Station")
		combined = combined[columns]

	return combined


def prepare_indonesia(root_dir: Path | str = DEFAULT_ROOT, output_suffix: str = "_with_bands", gee_project: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Prepare the Indonesian site CSVs and return the combined frames.

	This mirrors the previous CLI behaviour but is callable from a root launcher.
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

	folder_map = discover_imagery_folders(root_dir)
	if not folder_map:
		raise FileNotFoundError(f"No imagery folders found under {root_dir}")

	enriched_frames: list[tuple[str, pd.DataFrame]] = []
	sentinel2_frames: list[tuple[str, pd.DataFrame]] = []
	for filename in SUMMARY_FILES:
		input_file = root_dir / filename
		if not input_file.exists():
			print(f"Skipping missing file: {input_file}")
			continue

		frame = augment_summary_csv(input_file, root_dir, input_file, folder_map)
		enriched_frames.append((filename, frame))

	for filename in SUMMARY_FILES:
		input_file = root_dir / filename
		if not input_file.exists():
			continue

		frame = augment_summary_csv_sentinel2(
			input_file=input_file,
			output_file=input_file,
			folder_map=folder_map,
			sentinel2_manager=sentinel2_manager,
		)
		sentinel2_frames.append((filename, frame))

	combined = build_combined_frame(enriched_frames)
	if not combined.empty:
		combined_output = root_dir / f"summary_combined{output_suffix}.csv"
		combined.to_csv(combined_output, index=False)
		print(f"Wrote {combined_output}")

	sentinel2_combined = build_combined_frame(sentinel2_frames)
	if not sentinel2_combined.empty:
		sentinel2_combined_output = root_dir / f"summary_combined_sentinel2{output_suffix}.csv"
		sentinel2_combined.to_csv(sentinel2_combined_output, index=False)
		print(f"Wrote {sentinel2_combined_output}")

	return combined, sentinel2_combined


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Prepare Indonesian seagrass CSVs with PlanetScope and Sentinel-2 bands.")
	parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="Root directory containing project folders")
	parser.add_argument("--output-suffix", type=str, default="_with_bands", help="Suffix to append to output filenames")
	parser.add_argument("--gee-project", type=str, default=None, help="Google Earth Engine project ID (optional)")
	args = parser.parse_args()

	combined, sentinel2_combined = prepare_indonesia(root_dir=Path(args.root), output_suffix=args.output_suffix, gee_project=args.gee_project)
	if combined is not None and not combined.empty:
		print("Wrote combined enriched CSVs to", args.root)
	else:
		print("No enriched combined CSVs were produced.")


