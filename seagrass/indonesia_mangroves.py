"""Attach Landsat and Sentinel-2 band values + indices to the Indonesian mangrove plot data.

Two field-survey spreadsheets in data/, each a single tight cluster of individual
mangrove plots (own lon/lat, no per-plot date — only an approximate survey month
from the filename):

  data/indonesia_mangrove_July_2022.csv        50 plots, DMS coordinates
  data/indonesia_mangrove_July_August_2023.csv 60 plots, decimal-degree coordinates
                                                (same Baluran area as
                                                seagrass/indonesia.py's baluran cluster)

Since there's no per-plot date to match against (unlike seagrass/kenya_mangroves.py's
real survey dates), this follows seagrass/kenya_mangroves.py's module docstring's
"single composite-image approach": one locally-clearest scene is picked per cluster
for the whole padded survey window, then every plot in that cluster is sampled from
the same image via a single batched reduceRegions call — there's no reason to
re-search per plot when all plots in a cluster sit within a few hundred metres of
each other and share the same approximate survey date.

No Lyzenga / depth-invariant indices here, same reasoning as kenya_mangroves.py:
those model light attenuation through a water column and don't apply to mangrove
canopy reflectance (imaged from above the waterline, not through it).

Usage:

  # 1. See the per-cluster date windows and bounding boxes
  python seagrass/indonesia_mangroves.py dates

  # 2. Download Level-1 scenes per cluster (free accounts — see
  #    seagrass/tampa_bay.py's docstring for CDSE/USGS account setup)
  python seagrass/indonesia_mangroves.py download --cluster 2022 --scenes /raw_scenes/2022/ \\
      --cdse-user you@email.com --cdse-pass yourpass \\
      --usgs-user yourname --usgs-token yourtoken
  # repeat with --cluster 2023 --scenes /raw_scenes/2023/

  # 3. Run ACOLITE per cluster
  python seagrass/indonesia_mangroves.py acolite --cluster 2022 --scenes /raw_scenes/2022/ --acolite-out /acolite_out/
  # repeat with --cluster 2023 --scenes /raw_scenes/2023/

  # 4. Build the two combined CSVs (ACOLITE output primary; falls back to GEE
  #    for anything ACOLITE didn't cover). --gee-project can be omitted if
  #    gee_project is set in common/credentials.json.
  python seagrass/indonesia_mangroves.py build --acolite-dir /acolite_out/

Skipping steps 2-3 and running 'build' with just --gee-project uses GEE's
standard L2A/SR products for everything — ACOLITE is an upgrade, not a
requirement, same as the other seagrass/*.py scripts.

Output: two CSVs (one row per plot, both clusters combined) —
indonesia_mangroves_sentinel2_with_bands.csv and
indonesia_mangroves_landsat_with_bands.csv — each carrying the plot's own
attributes (dominant species, DBH, canopy cover, AGC, ...) plus that sensor's
bands/indices and which cluster/period it came from.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from math import cos, hypot, radians
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from common.common import parse_date_object, parse_date_value
from common.credentials import load_credentials
from common.gee_clear_sky import (
    s2_local_clear_candidates,
    landsat_local_clear_candidates,
    select_by_local_clarity,
    LOCAL_CLARITY_BUFFER_M,
)
from common.sentinel import (
    SENTINEL2_BAND_COLUMNS,
    SENTINEL2_DOWNLOAD_BANDS,
    SENTINEL2_INDEX_COLUMNS,
    SENTINEL2_SCALE,
    build_sentinel2_feature_values,
    compute_sentinel2_indices,
)
from common.landsat import (
    LANDSAT_BAND_COLUMNS,
    LANDSAT_DOWNLOAD_BANDS,
    LANDSAT_INDEX_COLUMNS,
    LANDSAT_SCALE,
    LANDSAT_OFFSET,
    build_landsat_feature_values,
    compute_landsat_indices,
)
from common.sentinel2 import Sentinel2Manager
from common.landsat import LandsatManager
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

DEFAULT_ROOT = Path("data")

# Same physical footprint for both sensors so a 30 m vs 10 m pixel doesn't bias
# comparisons (see seagrass/indonesia.py's BAND_SAMPLE_BUFFER_M comment).
BAND_SAMPLE_BUFFER_M = 45

# Mangrove canopy structure doesn't shift materially over a few weeks (unlike
# seagrass extent), so padding the named survey month(s) by this much on each
# side is safe and meaningfully improves the odds of finding a clear scene.
WINDOW_PAD_DAYS = 15

# ±0.05° buffer around each cluster's plot extent for imagery search / ACOLITE
# limit (same convention as seagrass/tampa_bay.py's TAMPA_BAY_LIMIT).
LIMIT_PAD_DEG = 0.05

# Extra context shown around the plot cluster in RGB quicklooks, beyond the
# cluster's own radius — enough to see the surrounding land/water without a
# huge (slow, large-file) export.
RGB_VIEW_PADDING_M = 1000.0
RGB_QUICKLOOK_SUBDIR = "rgb_quicklooks"
LANDSAT_RGB_BANDS = ["SR_B4", "SR_B3", "SR_B2"]     # Red, Green, Blue
SENTINEL2_RGB_BANDS = ["B4", "B3", "B2"]            # Red, Green, Blue


_DMS_RE = re.compile(r"(\d+)\s*°\s*(\d+)\s*[′']\s*([\d.]+)\s*[″\"”]")


def _dms_to_decimal(text: object) -> float:
    """Parse a DMS coordinate like 119°42′56.62″ into decimal degrees.

    Tolerant of stray whitespace and the mixed prime/quote characters
    (″ vs ” vs ") seen across rows in the source spreadsheet.
    """
    match = _DMS_RE.search(str(text).strip())
    if not match:
        raise ValueError(f"Could not parse DMS coordinate: {text!r}")
    deg, minute, sec = (float(g) for g in match.groups())
    return deg + minute / 60.0 + sec / 3600.0


def load_2022_plots(path: Path) -> pd.DataFrame:
    """Load the July 2022 plot CSV (DMS coordinates, 2-row header with units on row 2)."""
    df = pd.read_csv(path, skiprows=2, header=None, names=[
        "plot", "_longitude_dms", "_latitude_dms", "dbh_cm", "agc_mg_c_ha",
        "dominant_species", "canopy_cover_pct", "substrate", "_blank",
    ]).drop(columns=["_blank"])
    df["longitude"] = df["_longitude_dms"].apply(_dms_to_decimal)       # already East (+)
    df["latitude"] = -df["_latitude_dms"].apply(_dms_to_decimal)        # South -> negative
    return df.drop(columns=["_longitude_dms", "_latitude_dms"])


def load_2023_plots(path: Path) -> pd.DataFrame:
    """Load the July-August 2023 plot CSV (decimal-degree coordinates, 2-row header)."""
    df = pd.read_csv(path, skiprows=2, header=None, names=[
        "plot", "latitude", "longitude", "dominant_species", "dbh_cm",
        "canopy_cover_pct", "tree_height_m", "_blank1", "agc_mg_c_ha", "_blank2",
    ]).drop(columns=["_blank1", "_blank2"])
    return df


CLUSTERS: dict[str, dict] = {
    "2022": {
        "filename": "indonesia_mangrove_July_2022.csv",
        "loader": load_2022_plots,
        "period_start": "2022-07-01",
        "period_end": "2022-07-31",
        "label": "July 2022",
    },
    "2023": {
        "filename": "indonesia_mangrove_July_August_2023.csv",
        "loader": load_2023_plots,
        "period_start": "2023-07-01",
        "period_end": "2023-08-31",
        "label": "July-August 2023",
    },
}


def load_cluster(root_dir: Path, cluster_key: str) -> pd.DataFrame:
    info = CLUSTERS[cluster_key]
    path = root_dir / info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = info["loader"](path)
    df["cluster"] = cluster_key
    df["survey_period"] = info["label"]
    df["source_file"] = info["filename"]
    return df


# Plot clusters here are individual field surveys spanning well under 1 km;
# anything farther than this from the cluster's median position is treated as
# a likely data-entry error (e.g. a dropped digit in a DMS string) rather than
# a real plot location, so it doesn't blow up the search bbox / scene centroid
# for the whole cluster. The flagged row is still kept and sampled at its own
# (possibly wrong) coordinate — nothing is dropped or silently corrected.
POSITION_OUTLIER_KM = 2.0


def _inlier_mask(df: pd.DataFrame, cluster_key: str = "", max_outlier_km: float = POSITION_OUTLIER_KM) -> pd.Series:
    lon_med = df["longitude"].median()
    lat_med = df["latitude"].median()
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * cos(radians(lat_med))
    distance_km = np.hypot(
        (df["longitude"] - lon_med) * m_per_deg_lon,
        (df["latitude"] - lat_med) * m_per_deg_lat,
    ) / 1000.0
    mask = distance_km <= max_outlier_km
    if not mask.all():
        bad = df.loc[~mask].assign(distance_from_median_km=distance_km[~mask])
        print(
            f"WARNING: cluster {cluster_key!r} has {(~mask).sum()} plot(s) "
            f">{max_outlier_km} km from the rest of the cluster — likely a typo in "
            "the source spreadsheet's coordinate string (check it there; not "
            "auto-corrected here). Still sampled at their own coordinates, just "
            f"excluded from the cluster's search bbox/centroid:\n"
            f"{bad[['plot', 'longitude', 'latitude', 'distance_from_median_km']].to_string(index=False)}"
        )
    return mask


def cluster_limit(df: pd.DataFrame, cluster_key: str = "", pad_deg: float = LIMIT_PAD_DEG) -> list[float]:
    """[south, west, north, east], padded around the cluster's inlier plot extent."""
    inliers = df[_inlier_mask(df, cluster_key)]
    return [
        float(inliers["latitude"].min() - pad_deg),
        float(inliers["longitude"].min() - pad_deg),
        float(inliers["latitude"].max() + pad_deg),
        float(inliers["longitude"].max() + pad_deg),
    ]


def cluster_centroid_and_radius_m(df: pd.DataFrame, cluster_key: str = "") -> tuple[float, float, float]:
    """Returns (lon_centroid, lat_centroid, radius_m) over inlier plots only —
    flat-earth approximation, fine at the few-hundred-metre scale these
    clusters span."""
    inliers = df[_inlier_mask(df, cluster_key)]
    lon_c = float(inliers["longitude"].mean())
    lat_c = float(inliers["latitude"].mean())
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * cos(radians(lat_c))
    radius_m = 0.0
    for lon, lat in zip(inliers["longitude"], inliers["latitude"]):
        dx = (lon - lon_c) * m_per_deg_lon
        dy = (lat - lat_c) * m_per_deg_lat
        radius_m = max(radius_m, hypot(dx, dy))
    return lon_c, lat_c, radius_m


def padded_window(period_start: str, period_end: str, pad_days: int = WINDOW_PAD_DAYS) -> tuple[str, str, str]:
    """Returns (date_start, date_end, midpoint) as ISO date strings."""
    start = date.fromisoformat(period_start) - timedelta(days=pad_days)
    end = date.fromisoformat(period_end) + timedelta(days=pad_days)
    mid = start + (end - start) / 2
    return start.isoformat(), end.isoformat(), mid.isoformat()


def load_cluster_dates(root_dir: Path, cluster_key: str) -> tuple[str, str, str]:
    info = CLUSTERS[cluster_key]
    return padded_window(info["period_start"], info["period_end"])


# ===========================================================================
# GEE / ACOLITE sampling
# ===========================================================================

def _init_bands(frame: pd.DataFrame, band_columns: list[str], index_columns: list[str],
                date_col: str, source_col: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in band_columns + list(index_columns):
        frame[column] = np.nan
    frame[date_col] = ""
    frame[source_col] = ""
    return frame


def _reduce_points(image: "ee.Image", download_bands: list[str], rows: list[tuple[int, float, float]],
                    buffer_m: float, scale: int) -> dict:
    region_features = [
        ee.Feature(ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds(), {"row_index": idx})
        for idx, lon, lat in rows
    ]
    return image.select(download_bands).reduceRegions(
        collection=ee.FeatureCollection(region_features),
        reducer=ee.Reducer.mean(), scale=scale, tileScale=2,
    ).getInfo()


def sample_cluster_sentinel2(
    df: pd.DataFrame, mgr: Sentinel2Manager, cluster_key: str, acolite_scenes: list[dict],
    buffer_m: float = BAND_SAMPLE_BUFFER_M,
) -> pd.DataFrame:
    """`buffer_m` controls the radius averaged around each plot. Wider than
    BAND_SAMPLE_BUFFER_M trades the sensor-matched footprint (see that
    constant's comment) for more tolerance to GPS/geolocation error in the
    plot coordinates — useful when testing whether Sentinel-2's finer 10 m
    pixels are more exposed to positional noise than Landsat's 30 m ones."""
    info = CLUSTERS[cluster_key]
    date_start, date_end, mid_date = padded_window(info["period_start"], info["period_end"])
    lon_c, lat_c, radius_m = cluster_centroid_and_radius_m(df, cluster_key)
    clarity_buffer_m = max(radius_m + buffer_m, LOCAL_CLARITY_BUFFER_M)

    frame = _init_bands(df, SENTINEL2_BAND_COLUMNS, SENTINEL2_INDEX_COLUMNS, "s2_scene_date", "s2_source")
    remaining = list(frame.index)

    max_days = (date.fromisoformat(date_end) - date.fromisoformat(date_start)).days // 2 + 1
    acolite_scene = select_acolite_scene(acolite_scenes, mid_date, "S2", max_days) if acolite_scenes else None
    if acolite_scene:
        still_needed = []
        for idx in remaining:
            row = frame.loc[idx]
            sampled = sample_acolite_nc(acolite_scene["path"], row["longitude"], row["latitude"], ACOLITE_S2_BAND_MAP)
            if sampled and not all(pd.isna(v) for v in sampled.values()):
                compute_sentinel2_indices(sampled)
                for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
                    frame.at[idx, column] = sampled.get(column, np.nan)
                frame.at[idx, "s2_scene_date"] = parse_date_value(acolite_scene["date"])
                frame.at[idx, "s2_source"] = "acolite"
            else:
                still_needed.append(idx)
        remaining = still_needed

    if remaining:
        candidates = s2_local_clear_candidates(lon_c, lat_c, date_start, date_end, buffer_m=clarity_buffer_m)
        selected = select_by_local_clarity(candidates, mid_date)
        if selected:
            image = mgr.get_image_by_asset_id(selected["asset_id"])
            if image is not None:
                scene_date = parse_date_value(selected.get("date", ""))
                rows = [(idx, float(frame.at[idx, "longitude"]), float(frame.at[idx, "latitude"])) for idx in remaining]
                try:
                    reduced = _reduce_points(image, SENTINEL2_DOWNLOAD_BANDS, rows, buffer_m, 10)
                except Exception:
                    reduced = {}
                for feature in reduced.get("features", []):
                    props = feature.get("properties", {})
                    idx = props.get("row_index")
                    if idx is None:
                        continue
                    feat = build_sentinel2_feature_values(props, scene_date)
                    for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
                        frame.at[idx, column] = feat.get(column, np.nan)
                    frame.at[idx, "s2_scene_date"] = feat.get("scene_date", "")
                    frame.at[idx, "s2_source"] = "gee"

    return frame


def sample_cluster_landsat(
    df: pd.DataFrame, mgr: LandsatManager, cluster_key: str, acolite_scenes: list[dict],
) -> pd.DataFrame:
    info = CLUSTERS[cluster_key]
    date_start, date_end, mid_date = padded_window(info["period_start"], info["period_end"])
    lon_c, lat_c, radius_m = cluster_centroid_and_radius_m(df, cluster_key)
    clarity_buffer_m = max(radius_m + BAND_SAMPLE_BUFFER_M, LOCAL_CLARITY_BUFFER_M)

    frame = _init_bands(df, LANDSAT_BAND_COLUMNS, LANDSAT_INDEX_COLUMNS, "ls_scene_date", "ls_source")
    remaining = list(frame.index)

    max_days = (date.fromisoformat(date_end) - date.fromisoformat(date_start)).days // 2 + 1
    acolite_scene = select_acolite_scene(acolite_scenes, mid_date, "LS", max_days) if acolite_scenes else None
    if acolite_scene:
        still_needed = []
        for idx in remaining:
            row = frame.loc[idx]
            sampled = sample_acolite_nc(acolite_scene["path"], row["longitude"], row["latitude"], ACOLITE_LS_BAND_MAP)
            if sampled and not all(pd.isna(v) for v in sampled.values()):
                compute_landsat_indices(sampled)
                for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
                    frame.at[idx, column] = sampled.get(column, np.nan)
                frame.at[idx, "ls_scene_date"] = parse_date_value(acolite_scene["date"])
                frame.at[idx, "ls_source"] = "acolite"
            else:
                still_needed.append(idx)
        remaining = still_needed

    if remaining:
        candidates = landsat_local_clear_candidates(lon_c, lat_c, date_start, date_end, buffer_m=clarity_buffer_m)
        selected = select_by_local_clarity(candidates, mid_date)
        if selected:
            image = mgr.get_image_by_asset_id(selected["asset_id"])
            if image is not None:
                scene_date = parse_date_value(selected.get("date", ""))
                rows = [(idx, float(frame.at[idx, "longitude"]), float(frame.at[idx, "latitude"])) for idx in remaining]
                try:
                    reduced = _reduce_points(image, LANDSAT_DOWNLOAD_BANDS, rows, BAND_SAMPLE_BUFFER_M, 30)
                except Exception:
                    reduced = {}
                for feature in reduced.get("features", []):
                    props = feature.get("properties", {})
                    idx = props.get("row_index")
                    if idx is None:
                        continue
                    feat = build_landsat_feature_values(props, scene_date)
                    for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
                        frame.at[idx, column] = feat.get(column, np.nan)
                    frame.at[idx, "ls_scene_date"] = feat.get("ls_scene_date", "")
                    frame.at[idx, "ls_source"] = "gee"

    return frame


def prepare_indonesia_mangroves(
    root_dir: Path = DEFAULT_ROOT,
    output_dir: Path | None = None,
    gee_project: str | None = None,
    acolite_dir: Path | None = None,
    s2_buffer_m: float = BAND_SAMPLE_BUFFER_M,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the combined Sentinel-2 and Landsat CSVs across both mangrove clusters.

    `s2_buffer_m` overrides Sentinel-2's sampling radius (default matches
    Landsat's BAND_SAMPLE_BUFFER_M for a fair footprint comparison); Landsat's
    own radius is left fixed since this is meant to test whether Sentinel-2's
    finer resolution is more sensitive to plot-coordinate imprecision.

    Returns (sentinel2_frame, landsat_frame).
    """
    if gee_project is None:
        try:
            gee_project = load_credentials().get("gee_project")
        except FileNotFoundError:
            gee_project = None
    if not gee_project:
        raise FileNotFoundError("No Earth Engine project found in common/credentials/credentials.json")
    ee.Initialize(project=gee_project)

    acolite_scenes: list[dict] = []
    if acolite_dir is not None and acolite_dir.exists():
        acolite_scenes = scan_acolite_output(acolite_dir)
        print(f"Indexed {len(acolite_scenes)} ACOLITE scenes in {acolite_dir}")

    s2_mgr = Sentinel2Manager(gee_project=gee_project)
    ls_mgr = LandsatManager(gee_project=gee_project)

    s2_frames, ls_frames = [], []
    for cluster_key in CLUSTERS:
        df = load_cluster(root_dir, cluster_key)
        print(f"\nCluster {cluster_key} ({CLUSTERS[cluster_key]['label']}): {len(df)} plots")
        s2_frames.append(sample_cluster_sentinel2(df, s2_mgr, cluster_key, acolite_scenes, buffer_m=s2_buffer_m))
        ls_frames.append(sample_cluster_landsat(df, ls_mgr, cluster_key, acolite_scenes))

    s2_combined = pd.concat(s2_frames, ignore_index=True, sort=False)
    ls_combined = pd.concat(ls_frames, ignore_index=True, sort=False)

    output_dir = output_dir or root_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        (s2_combined, "indonesia_mangroves_sentinel2_with_bands.csv"),
        (ls_combined, "indonesia_mangroves_landsat_with_bands.csv"),
    ]
    for frame, filename in outputs:
        out_path = output_dir / filename
        frame.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    return s2_combined, ls_combined


# ===========================================================================
# RGB quicklooks — visualize the exact scene each cluster's bands came from
# ===========================================================================

def _percentile_stretch_to_uint8(array: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile stretch to 0-255 for a natural-looking true-colour quicklook.

    Uses ONE shared low/high cutoff across all bands (not a separate stretch
    per band) — stretching each RGB channel independently maps every band's
    median to roughly the same output level regardless of its true relative
    brightness, which destroys the inter-channel brightness ratios that
    "colour" actually is and produces a false magenta/cyan cast.
    """
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros_like(array, dtype=np.uint8)
    lo, hi = np.percentile(valid, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(array, dtype=np.uint8)
    stretched = np.clip((array - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def export_rgb_quicklook(
    image: "ee.Image", lon: float, lat: float, radius_m: float,
    bands: list[str], scale_m: int, to_reflectance,
    output_stem: Path, plot_points: list[tuple[float, float]] | None = None,
) -> tuple[Path, Path]:
    """Download `bands` (R, G, B order) around (lon, lat), convert to reflectance,
    percentile-stretch to 8-bit, and write both a georeferenced GeoTIFF (for GIS
    use — clean, no annotations) and a quicklook PNG (with plot markers, if
    `plot_points` [(lon, lat), ...] is given — handy for a quick visual sanity
    check of where the plots actually fall on the imagery). Returns (tif_path, png_path).
    """
    import rasterio
    from rasterio.warp import transform as transform_coordinates
    import requests

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    region = ee.Geometry.Point([lon, lat]).buffer(radius_m).bounds()
    url = image.select(bands).getDownloadUrl({
        "scale": scale_m, "region": region, "format": "GeoTIFF", "filePerBand": False,
    })
    raw_path = output_stem.with_name(output_stem.name + "_raw.tif")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with open(raw_path, "wb") as fh:
        fh.write(response.content)

    with rasterio.open(raw_path) as src:
        raw = src.read().astype(np.float64)
        profile = src.profile
        crs = src.crs
        raster_transform = src.transform
    raw_path.unlink(missing_ok=True)

    rgb_uint8 = _percentile_stretch_to_uint8(to_reflectance(raw))

    tif_path = output_stem.with_suffix(".tif")
    tif_profile = profile.copy()
    tif_profile.update(dtype="uint8", count=3, nodata=None)
    with rasterio.open(tif_path, "w", **tif_profile) as dst:
        dst.write(rgb_uint8)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = output_stem.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.moveaxis(rgb_uint8, 0, -1))
    if plot_points:
        lons, lats = zip(*plot_points)
        xs, ys = transform_coordinates("EPSG:4326", crs, list(lons), list(lats))
        inv = ~raster_transform
        cols, rows = zip(*(inv * (x, y) for x, y in zip(xs, ys)))
        ax.scatter(cols, rows, s=16, facecolors="none", edgecolors="red", linewidths=1.2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return tif_path, png_path


def export_cluster_rgb(
    root_dir: Path, cluster_key: str, gee_project: str, output_dir: Path, sensor: str,
) -> tuple[Path, Path] | None:
    """Fetch the same locally-clearest scene `build` would pick for this cluster
    and sensor, and write it out as an RGB GeoTIFF + annotated PNG quicklook."""
    df = load_cluster(root_dir, cluster_key)
    info = CLUSTERS[cluster_key]
    date_start, date_end, mid_date = padded_window(info["period_start"], info["period_end"])
    lon_c, lat_c, radius_m = cluster_centroid_and_radius_m(df, cluster_key)
    clarity_buffer_m = max(radius_m + BAND_SAMPLE_BUFFER_M, LOCAL_CLARITY_BUFFER_M)
    view_radius_m = radius_m + RGB_VIEW_PADDING_M
    plot_points = list(zip(df["longitude"], df["latitude"]))

    if sensor == "landsat":
        candidates = landsat_local_clear_candidates(lon_c, lat_c, date_start, date_end, buffer_m=clarity_buffer_m)
        selected = select_by_local_clarity(candidates, mid_date)
        if not selected:
            print(f"No usable Landsat scene found for cluster {cluster_key}")
            return None
        image = LandsatManager(gee_project=gee_project).get_image_by_asset_id(selected["asset_id"])
        bands, scale_m = LANDSAT_RGB_BANDS, 30
        to_reflectance = lambda arr: arr * LANDSAT_SCALE + LANDSAT_OFFSET
        satellite = next((s for s in ("LC08", "LC09") if s in selected["asset_id"]), "landsat")
    elif sensor == "sentinel2":
        candidates = s2_local_clear_candidates(lon_c, lat_c, date_start, date_end, buffer_m=clarity_buffer_m)
        selected = select_by_local_clarity(candidates, mid_date)
        if not selected:
            print(f"No usable Sentinel-2 scene found for cluster {cluster_key}")
            return None
        image = Sentinel2Manager(gee_project=gee_project).get_image_by_asset_id(selected["asset_id"])
        bands, scale_m = SENTINEL2_RGB_BANDS, 10
        to_reflectance = lambda arr: arr / SENTINEL2_SCALE
        satellite = "sentinel2"
    else:
        raise ValueError(f"Unknown sensor: {sensor!r}")

    if image is None:
        print(f"Could not resolve the selected {sensor} scene for cluster {cluster_key}")
        return None

    scene_date = parse_date_value(selected.get("date", ""))
    stem = output_dir / f"{cluster_key}_{sensor}_{satellite}_{scene_date}_rgb"
    tif_path, png_path = export_rgb_quicklook(
        image, lon_c, lat_c, view_radius_m, bands, scale_m, to_reflectance, stem, plot_points,
    )
    print(f"Wrote {tif_path}")
    print(f"Wrote {png_path}")
    return tif_path, png_path


# ===========================================================================
# CLI
# ===========================================================================

def _cmd_dates(args) -> None:
    root_dir = Path(args.root)
    clusters = list(CLUSTERS) if args.cluster == "all" else [args.cluster]
    for cluster_key in clusters:
        df = load_cluster(root_dir, cluster_key)
        date_start, date_end, mid_date = load_cluster_dates(root_dir, cluster_key)
        limit = cluster_limit(df, cluster_key)
        print(f"\n=== {cluster_key} ({CLUSTERS[cluster_key]['label']}) ===")
        print(f"Plots: {len(df)}")
        print(f"Padded search window (±{WINDOW_PAD_DAYS} days beyond named period): {date_start} .. {date_end}")
        print(f"Bounding box (S, W, N, E): {limit}")
        windows = merge_date_windows([mid_date], 0)
        print(f"Merged search window(s): {[(date_start, date_end)]}")


def _cmd_download(args) -> None:
    root_dir = Path(args.root)
    cluster_key = args.cluster
    df = load_cluster(root_dir, cluster_key)
    limit = cluster_limit(df, cluster_key)
    date_start, date_end, _ = load_cluster_dates(root_dir, cluster_key)
    scenes_dir = Path(args.scenes)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    if args.cdse_user and args.cdse_pass:
        download_sentinel2([date_start], scenes_dir, args.cdse_user, args.cdse_pass,
                            limit=limit, window_days=(date.fromisoformat(date_end) - date.fromisoformat(date_start)).days // 2)
    else:
        print("Skipping Sentinel-2 (no --cdse-user / --cdse-pass)")
    if args.usgs_user and args.usgs_token:
        download_landsat([date_start], scenes_dir, args.usgs_user, args.usgs_token,
                          limit=limit, window_days=(date.fromisoformat(date_end) - date.fromisoformat(date_start)).days // 2)
    else:
        print("Skipping Landsat (no --usgs-user / --usgs-token)")


def _cmd_acolite(args) -> None:
    root_dir = Path(args.root)
    df = load_cluster(root_dir, args.cluster)
    run_acolite_batch(
        scenes_dir=Path(args.scenes),
        output_dir=Path(args.acolite_out),
        limit=cluster_limit(df, args.cluster),
        acolite_path=Path(args.acolite_repo),
    )


def _cmd_build(args) -> None:
    prepare_indonesia_mangroves(
        root_dir=Path(args.root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        gee_project=args.gee_project,
        acolite_dir=Path(args.acolite_dir) if args.acolite_dir else None,
        s2_buffer_m=args.s2_buffer_m,
    )


def _cmd_rgb(args) -> None:
    root_dir = Path(args.root)
    gee_project = args.gee_project
    if gee_project is None:
        try:
            gee_project = load_credentials().get("gee_project")
        except FileNotFoundError:
            gee_project = None
    if not gee_project:
        raise FileNotFoundError("No Earth Engine project found in common/credentials/credentials.json")
    ee.Initialize(project=gee_project)

    output_dir = Path(args.output_dir) if args.output_dir else root_dir / RGB_QUICKLOOK_SUBDIR
    clusters = list(CLUSTERS) if args.cluster == "all" else [args.cluster]
    sensors = ["landsat", "sentinel2"] if args.sensor == "both" else [args.sensor]
    for cluster_key in clusters:
        for sensor in sensors:
            export_cluster_rgb(root_dir, cluster_key, gee_project, output_dir, sensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indonesian mangrove plot pipeline (Landsat, Sentinel-2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Two clusters, one per survey CSV:\n"
            "  2022 : indonesia_mangrove_July_2022.csv (50 plots, DMS coords)\n"
            "  2023 : indonesia_mangrove_July_August_2023.csv (60 plots, decimal coords)\n"
            "Run 'python seagrass/indonesia_mangroves.py <command> --help' for per-command options."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    # dates
    p = sub.add_parser("dates", help="Show per-cluster date window and bounding box")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--cluster", choices=["2022", "2023", "all"], default="all")

    # download
    p = sub.add_parser("download", help="Download Level-1 scenes from CDSE / USGS for one cluster")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--cluster", choices=["2022", "2023"], required=True)
    p.add_argument("--scenes", required=True, help="Directory to save downloaded scenes")
    p.add_argument("--cdse-user", metavar="EMAIL")
    p.add_argument("--cdse-pass", metavar="PASSWORD")
    p.add_argument("--usgs-user", metavar="USERNAME")
    p.add_argument("--usgs-token", metavar="TOKEN")

    # acolite
    p = sub.add_parser("acolite", help="Run ACOLITE on downloaded scenes for one cluster")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--cluster", choices=["2022", "2023"], required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--acolite-out", required=True, help="Output directory for NetCDF files")
    p.add_argument("--acolite-repo", default=str(ACOLITE_REPO_PATH))

    # build
    p = sub.add_parser("build", help="Build the two combined CSVs (both clusters, per sensor)")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--output-dir", default=None, help="Defaults to --root")
    p.add_argument("--gee-project", type=str, default=None,
                   help="Defaults to gee_project in common/credentials.json if omitted")
    p.add_argument("--acolite-dir", metavar="DIR",
                   help="Directory of ACOLITE NetCDF output for BOTH clusters (primary imagery source)")
    p.add_argument("--s2-buffer-m", type=float, default=BAND_SAMPLE_BUFFER_M,
                   help=f"Sentinel-2 sampling radius in metres (default {BAND_SAMPLE_BUFFER_M}, "
                        "same as Landsat's fixed footprint); widen to test sensitivity to GPS/coordinate error")

    # rgb
    p = sub.add_parser("rgb", help="Save a true-colour RGB GeoTIFF + PNG quicklook of the scene used per cluster")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--cluster", choices=["2022", "2023", "all"], default="all")
    p.add_argument("--sensor", choices=["landsat", "sentinel2", "both"], default="landsat")
    p.add_argument("--output-dir", default=None, help=f"Defaults to <root>/{RGB_QUICKLOOK_SUBDIR}")
    p.add_argument("--gee-project", type=str, default=None,
                   help="Defaults to gee_project in common/credentials.json if omitted")

    args = parser.parse_args()
    dispatch = {
        "dates": _cmd_dates,
        "download": _cmd_download,
        "acolite": _cmd_acolite,
        "build": _cmd_build,
        "rgb": _cmd_rgb,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()
