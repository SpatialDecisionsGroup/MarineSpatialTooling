"""Attach Landsat and Sentinel-2 band values + indices to the Indonesian mangrove dataset.

Source data:
/home/ysi/Documents/Uni/MQPostdoc/projects/superres/indonesia_mangroves/mangrove_ML_dataframe(in).csv

That CSV is a static hex-grid dataset (~7000 planning-unit centroids, no
per-point observation date) already carrying one bespoke set of Sentinel-2-only
indices (NDRE, IRECI, MCARI, CIre1/2, ... - all need red-edge bands Landsat
doesn't have). This script drops those and rebuilds two CSVs using the same
band/index set as seagrass/indonesia.py and seagrass/tampa_bay.py
(SENTINEL2_BAND_COLUMNS + SENTINEL2_INDEX_COLUMNS, LANDSAT_BAND_COLUMNS +
LANDSAT_INDEX_COLUMNS from common/), so all three datasets in this repo are
directly comparable.

Two things the caller MUST supply because they aren't recoverable from the
CSV itself:

  --crs         pu_x/pu_y are planning-unit centroids in an unknown projected
                CRS (values ~1.5-1.7M / -213k to -565k -> metres, not degrees).
                Pass whatever EPSG code or proj string produced this hex grid,
                e.g. --crs EPSG:32749. Get it wrong and every sample point
                silently lands in the wrong place.

  --start-date / --end-date
                There's no per-row date, so instead of nearest-scene matching
                (like indonesia.py) this builds ONE cloud-masked median
                composite per sensor over this window and samples all points
                against it. Pick a window centred on whatever year the
                canopy_height / covariate layers represent, so the spectral
                bands aren't badly mismatched in time from what canopy_height
                is measuring.

Because there's no per-observation date to drive nearest-scene search, this
also differs from indonesia.py in *how* imagery is chosen: one composite
image is built per sensor (Cloud Score+ mask for S2, QA_PIXEL clear-bit mask
for Landsat) and sampled for every point in batched reduceRegions() calls,
rather than searching per point/date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

from common.credentials import load_credentials
from common.sentinel import (
    SENTINEL2_BAND_COLUMNS,
    SENTINEL2_DOWNLOAD_BANDS,
    SENTINEL2_INDEX_COLUMNS,
    compute_sentinel2_indices,
)
from common.landsat import (
    LANDSAT_BAND_COLUMNS,
    LANDSAT_DOWNLOAD_BANDS,
    LANDSAT_INDEX_COLUMNS,
    compute_landsat_indices,
)

DEFAULT_INPUT = Path(
    "/home/ysi/Documents/Uni/MQPostdoc/projects/superres/indonesia_mangroves/mangrove_ML_dataframe(in).csv"
)

# Spectral indices already baked into the source CSV (Sentinel-2-only; several
# need red-edge bands Landsat doesn't have). Dropped in favour of the
# SENTINEL2_INDEX_COLUMNS / LANDSAT_INDEX_COLUMNS sets used elsewhere in this repo.
SOURCE_INDEX_COLUMNS = [
    "NDVI", "GNDVI", "NDRE", "NDMI", "NDII2", "MSI", "NBR", "NBR2",
    "CIre1", "CIre2", "NIRv", "NDI45", "SAVI", "IRECI", "MCARI",
    "MSAVI", "MSAVI2", "DVI", "PVI", "EVI2", "RVI",
]

# Same physical footprint for both sensors (see seagrass/indonesia.py's
# BAND_SAMPLE_BUFFER_M comment) so a 30 m vs 10 m pixel doesn't bias comparisons.
BAND_SAMPLE_BUFFER_M = 45

S2_CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S2_CLEAR_THRESHOLD = 0.6

DEFAULT_CHUNK_SIZE = 1000
AOI_PAD_DEGREES = 0.05


def reproject_to_lonlat(pu_x: pd.Series, pu_y: pd.Series, source_crs: str) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(pu_x.to_numpy(), pu_y.to_numpy())
    return np.asarray(lon), np.asarray(lat)


def build_sentinel2_composite(aoi: "ee.Geometry", date_start: str, date_end: str) -> "ee.Image":
    """Cloud Score+ masked median composite over the AOI/window."""
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi).filterDate(date_start, date_end))
    cs_plus = ee.ImageCollection(S2_CLOUD_SCORE_COLLECTION)
    linked = s2.linkCollection(cs_plus, ["cs"])
    masked = linked.map(lambda img: img.updateMask(img.select("cs").gte(S2_CLEAR_THRESHOLD)))
    return masked.select(SENTINEL2_DOWNLOAD_BANDS).median().clip(aoi)


def build_landsat_composite(aoi: "ee.Geometry", date_start: str, date_end: str) -> "ee.Image":
    """QA_PIXEL clear-bit masked median composite over the AOI/window."""
    coll = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(aoi).filterDate(date_start, date_end))
    masked = coll.map(lambda img: img.updateMask(img.select("QA_PIXEL").bitwiseAnd(1 << 6).gt(0)))
    return masked.select(LANDSAT_DOWNLOAD_BANDS).median().clip(aoi)


def sample_composite(
    image: "ee.Image",
    frame: pd.DataFrame,
    download_bands: list[str],
    scale_meters: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    buffer_m: float = BAND_SAMPLE_BUFFER_M,
) -> dict[int, dict[str, float]]:
    """Batch reduceRegions() over all rows; returns {row_index: {band_name: value}}."""
    results: dict[int, dict[str, float]] = {}
    row_indices = list(frame.index)

    for start in range(0, len(row_indices), chunk_size):
        batch_indices = row_indices[start:start + chunk_size]
        features = [
            ee.Feature(
                ee.Geometry.Point([
                    float(frame.at[row_index, "longitude"]),
                    float(frame.at[row_index, "latitude"]),
                ]).buffer(buffer_m).bounds(),
                {"row_index": int(row_index)},
            )
            for row_index in batch_indices
        ]
        reduced = image.reduceRegions(
            collection=ee.FeatureCollection(features),
            reducer=ee.Reducer.mean(),
            scale=scale_meters,
            tileScale=4,
        ).getInfo()

        for feature in reduced.get("features", []):
            props = feature.get("properties", {})
            row_index = props.get("row_index")
            if row_index is None:
                continue
            results[int(row_index)] = {band: props.get(band) for band in download_bands}

    return results


def build_sentinel2_frame(
    frame: pd.DataFrame,
    aoi: "ee.Geometry",
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    out = frame.copy()
    for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
        out[column] = np.nan
    out["s2_window_start"] = date_start
    out["s2_window_end"] = date_end

    composite = build_sentinel2_composite(aoi, date_start, date_end)
    sampled = sample_composite(composite, out, SENTINEL2_DOWNLOAD_BANDS, scale_meters=10)

    for row_index, band_values in sampled.items():
        features: dict[str, float] = {}
        for ee_band, csv_col in zip(SENTINEL2_DOWNLOAD_BANDS, SENTINEL2_BAND_COLUMNS):
            value = band_values.get(ee_band)
            features[csv_col] = np.nan if value is None else float(value) / 10000.0
            out.at[row_index, csv_col] = features[csv_col]
        compute_sentinel2_indices(features)
        for column in SENTINEL2_INDEX_COLUMNS:
            out.at[row_index, column] = features.get(column, np.nan)

    return out


def build_landsat_frame(
    frame: pd.DataFrame,
    aoi: "ee.Geometry",
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    out = frame.copy()
    for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
        out[column] = np.nan
    out["ls_window_start"] = date_start
    out["ls_window_end"] = date_end

    composite = build_landsat_composite(aoi, date_start, date_end)
    sampled = sample_composite(composite, out, LANDSAT_DOWNLOAD_BANDS, scale_meters=30)

    LANDSAT_SCALE = 0.0000275
    LANDSAT_OFFSET = -0.2
    for row_index, band_values in sampled.items():
        features: dict[str, float] = {}
        for ee_band, csv_col in zip(LANDSAT_DOWNLOAD_BANDS, LANDSAT_BAND_COLUMNS):
            value = band_values.get(ee_band)
            features[csv_col] = np.nan if value is None else float(value) * LANDSAT_SCALE + LANDSAT_OFFSET
            out.at[row_index, csv_col] = features[csv_col]
        compute_landsat_indices(features)
        for column in LANDSAT_INDEX_COLUMNS:
            out.at[row_index, column] = features.get(column, np.nan)

    return out


def prepare_mangroves(
    input_csv: Path,
    source_crs: str,
    date_start: str,
    date_end: str,
    output_dir: Path | None = None,
    gee_project: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Sentinel-2 and Landsat mangrove CSVs. Returns (sentinel2_frame, landsat_frame)."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if gee_project is None:
        try:
            gee_project = load_credentials().get("gee_project")
        except FileNotFoundError:
            gee_project = None
    if not gee_project:
        raise FileNotFoundError("No Earth Engine project found in common/credentials/credentials.json")
    ee.Initialize(project=gee_project)

    frame = pd.read_csv(input_csv)
    frame = frame.drop(columns=[c for c in SOURCE_INDEX_COLUMNS if c in frame.columns])

    lon, lat = reproject_to_lonlat(frame["pu_x"], frame["pu_y"], source_crs)
    pu_x_index = frame.columns.get_loc("pu_y") + 1
    frame.insert(pu_x_index, "longitude", lon)
    frame.insert(pu_x_index + 1, "latitude", lat)

    aoi = ee.Geometry.BBox(
        float(frame["longitude"].min()) - AOI_PAD_DEGREES,
        float(frame["latitude"].min()) - AOI_PAD_DEGREES,
        float(frame["longitude"].max()) + AOI_PAD_DEGREES,
        float(frame["latitude"].max()) + AOI_PAD_DEGREES,
    )

    print(f"Loaded {len(frame)} points; AOI bounds (lon/lat): "
          f"[{frame['longitude'].min():.4f}, {frame['latitude'].min():.4f}] - "
          f"[{frame['longitude'].max():.4f}, {frame['latitude'].max():.4f}]")

    print(f"\nBuilding Sentinel-2 composite {date_start} .. {date_end} and sampling…")
    s2_frame = build_sentinel2_frame(frame, aoi, date_start, date_end)

    print(f"\nBuilding Landsat composite {date_start} .. {date_end} and sampling…")
    ls_frame = build_landsat_frame(frame, aoi, date_start, date_end)

    output_dir = output_dir or input_csv.parent
    s2_out = output_dir / "mangrove_sentinel2_with_bands.csv"
    ls_out = output_dir / "mangrove_landsat_with_bands.csv"
    s2_frame.to_csv(s2_out, index=False)
    print(f"Wrote {s2_out}")
    ls_frame.to_csv(ls_out, index=False)
    print(f"Wrote {ls_out}")

    return s2_frame, ls_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Indonesian mangrove CSVs with Landsat and Sentinel-2 bands/indices.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Path to mangrove_ML_dataframe(in).csv")
    parser.add_argument("--crs", required=True,
                         help="EPSG code or proj string that pu_x/pu_y are in, e.g. EPSG:32749")
    parser.add_argument("--start-date", required=True, help="Composite window start, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Composite window end, YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Defaults to the input CSV's directory")
    parser.add_argument("--gee-project", type=str, default=None)
    args = parser.parse_args()

    prepare_mangroves(
        input_csv=args.input,
        source_crs=args.crs,
        date_start=args.start_date,
        date_end=args.end_date,
        output_dir=args.output_dir,
        gee_project=args.gee_project,
    )
