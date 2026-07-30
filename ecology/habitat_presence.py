"""Presence/absence habitat classification points for coral reef, seagrass and
mangrove extent, restricted to the Coral Triangle, with Landsat and Sentinel-2
bands + indices attached.

Source data: run `bash habitat_extents.sh` first (see that script's docstring).
It downloads three independent global extent layers into data/habitat_extents/:

  coral     UNEP-WCMC Global Distribution of Coral Reefs, v4.1 (WCMC-008)
  seagrass  UNEP-WCMC Global Distribution of Seagrasses, v7.1 (WCMC-013/014)
  mangrove  Global Mangrove Watch v4.0.19, 2020 epoch (Zenodo)

Unlike every other ecology/*.py script, these are not field surveys — they are
polygon extent maps with no attached photo, transect or per-feature date. So
the shape of this problem is different: instead of collapsing/matching survey
rows to imagery, this script itself generates the point set — a balanced
presence/absence classification sample. For each habitat class: 500 points
drawn from inside that habitat's polygons (presence) and 500 drawn from
nearby, outside all three habitats' polygons (absence).

Region: the Coral Triangle
---------------------------
CORAL_TRIANGLE_BBOX is a bounding rectangle (11 S-22 N, 95-165 E) commonly
cited for the Coral Triangle (Indonesia, Malaysia, Papua New Guinea,
Philippines, Solomon Islands, Timor-Leste; e.g. Veron et al. 2009) — not the
precise CTI-CFF ecoregion boundary, which is a fine simplification for
AOI-restriction purposes. All three habitats are sampled from the same bbox.

Why "absence" is drawn from a buffer around the habitat, not the whole bbox
-----------------------------------------------------------------------------
A first cut at "absence" would be any point in the bbox that isn't inside a
habitat polygon. That's a bad negative class: it puts far more open ocean and
inland/upland area into "absence" than anything resembling a genuine
ecological alternative, and a classifier would then mostly be learning
"shallow coastal water vs. deep ocean" or "coast vs. inland", not anything
about the habitat itself — the same trap flagged when reviewing the Hawaii
CNET results (a test that isn't measuring what it claims to). Instead, each
habitat's own polygons are buffered by ABSENCE_BUFFER_M and absence points are
drawn from inside that buffer, still outside all three habitats' polygons —
i.e. "nearby but not on it", a harder and more meaningful negative.

Why sampling is done on a raster grid, not directly on the polygons
----------------------------------------------------------------------
Coral (~11k polygons) and seagrass (~21k) in the Coral Triangle bbox are
tractable for vector operations (weighted point-in-polygon, buffer, union).
Global Mangrove Watch is not: 1.2 million polygons worldwide, traced at
building-level detail. A vector buffer+union over even the bbox-clipped
subset of that (still hundreds of thousands of features) did not complete in
a reasonable time — a plain unary_union over that many complex polygons is
the classic vector-GIS blowup. Every habitat is instead rasterized onto one
shared lon/lat grid at RASTER_RESOLUTION_DEG (~550 m at the equator — finer
than either sensor's sampling footprint, so this is purely a spatial-logic
grid, not a resolution the actual band values are limited to; those still
come from an exact per-point GEE query). Presence points are drawn from
habitat==True cells with sub-cell jitter; absence points from a morphological
dilation of the habitat mask (the raster equivalent of a buffer) with the
union of all three habitats' masks subtracted out. This scales to Global
Mangrove Watch's size in seconds instead of not finishing at all, at the cost
of exact polygon boundaries — a fine trade for a signal-testing dataset.

Submerged vs. emergent habitats
---------------------------------
Coral and seagrass are imaged through a water column; mangrove is imaged
above the waterline (same distinction indonesia_mangroves.py and
kenya_mangroves.py make). This affects two things here:

- Absence points for coral/seagrass are additionally required to be open
  water (MODIS MOD44W water mask) — a submerged-habitat negative that's on
  dry land would be trivially separable by NDWI alone, which isn't a
  meaningful test of benthic signal. Mangrove absence is left unconstrained
  (land and water are both legitimate "not mangrove" neighbours of a mangrove
  stand), matching world_sampling.py's own needs_visibility distinction.
- Lyzenga depth-invariant indices (band-ratio based, no depth measurement
  needed) are computed for coral/seagrass only.

No Beer-Lambert depth correction is applied for any habitat: unlike the field
surveys, these extent polygons carry no per-feature depth attribute, and there
is no bathymetry raster in this repo to fall back on (see
superres/world_sampling.py's GEBCO dependency for what that would take). This
means the drc_* columns other scripts add are simply absent here rather than
present-but-wrong.

No survey date
---------------
WCMC's coral/seagrass layers are compiled from many sources of different
vintage, and GMW mangrove is pinned to whichever epoch habitat_extents.sh
downloaded (2020 by default) — there is no natural per-point "observation
date" to match imagery against the way the field-survey scripts do. Every
point uses the same fixed reference window (REFERENCE_DATE +/- WINDOW_DAYS),
centred on the GMW epoch and padded wide because the Coral Triangle is
heavily monsoonal/cloud-affected.

Usage
-----
  bash habitat_extents.sh                                  # 1. already run
  python ecology/habitat_presence.py points --habitat all   # 2. sanity-check point sampling (no imagery)
  python ecology/habitat_presence.py build --habitat all    # 3. sample imagery, write the 6 CSVs

Output: data/habitat_presence/<habitat>_<sensor>_with_bands.csv, one row per
point, for habitat in {coral, seagrass, mangrove} and sensor in {sentinel2,
landsat} — 6 files, up to 1000 rows each.
"""

from __future__ import annotations

import argparse
import gc
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.transform
from rasterio import features as rio_features
from scipy.ndimage import binary_dilation
from shapely.geometry import box
from tqdm import tqdm

from common.common import format_date_window, parse_date_value
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

DEFAULT_HABITAT_DIR = Path("data/habitat_extents")
DEFAULT_OUTPUT_DIR = Path("data/habitat_presence")

HABITATS = ["coral", "seagrass", "mangrove"]
SUBMERGED_HABITATS = frozenset({"coral", "seagrass"})

N_PRESENCE = 500
N_ABSENCE = 500
SAMPLE_SEED = 20150601

# Bounding rectangle commonly cited for the Coral Triangle (see module
# docstring) — (south, west, north, east) in decimal degrees.
CORAL_TRIANGLE_BBOX = dict(south=-11.0, west=95.0, north=22.0, east=165.0)

# "Nearby but not on it" radius for absence sampling (see module docstring).
ABSENCE_BUFFER_M = 20_000.0
ABSENCE_BATCH_SIZE = 400
MAX_ABSENCE_BATCHES = 40

# Shared sampling grid (see module docstring's rasterization rationale).
# ~550 m at the equator, shrinking in longitude further north — finer than
# either sensor's own sampling footprint, so this only bounds the spatial
# *logic* (which cells count as habitat/buffer/excluded); actual band values
# always come from an exact per-point GEE query, not from this grid.
RASTER_RESOLUTION_DEG = 0.005
METRES_PER_DEGREE_LAT = 110_540.0

# Only simplify a habitat's geometries before rasterizing above this many
# polygons — coral (~11k) and seagrass (~21k) in the bbox are small enough
# that simplification serves no memory purpose and just degenerates small
# patches (see load_and_rasterize_habitat); only mangrove (~440k) needs it.
SIMPLIFY_THRESHOLD_POLYGONS = 50_000

# Latest epoch in GEE's MOD44W collection (annual water mask stopped in 2015;
# fine here since it's only used as a coarse land/water screen, not tied to
# the survey date the way the imagery match is).
WATER_MASK_ASSET = "MODIS/006/MOD44W/2015_01_01"

# habitat_extents.sh defaults to GMW_YEAR=2020; centring the imagery window
# there and padding a full year each side (see module docstring) balances
# "close to the mapped epoch" against "the Coral Triangle is cloudy enough
# that a narrow window often has zero clear scenes at a given point".
REFERENCE_DATE = "2020-06-01"
WINDOW_DAYS = 365

BAND_SAMPLE_BUFFER_M = 45  # matches every other ecology/*.py script
CHECKPOINT_EVERY = 100


# ===========================================================================
# Section 1 — Habitat polygon loading + rasterization
# ===========================================================================

def _bbox_geom():
    b = CORAL_TRIANGLE_BBOX
    return box(b["west"], b["south"], b["east"], b["north"])


def grid_shape_transform() -> tuple[int, int, "rasterio.Affine"]:
    b = CORAL_TRIANGLE_BBOX
    width = round((b["east"] - b["west"]) / RASTER_RESOLUTION_DEG)
    height = round((b["north"] - b["south"]) / RASTER_RESOLUTION_DEG)
    transform = rasterio.transform.from_origin(
        b["west"], b["north"], RASTER_RESOLUTION_DEG, RASTER_RESOLUTION_DEG,
    )
    return height, width, transform


# Same file-discovery convention as WorldPatchSampler._load_habitat_gdf (WCMC:
# _Py_ = polygon, _Pt_ = point, _Ln_ = line), duplicated rather than reused
# because that method reads the whole file with no bbox support — fine for
# coral (~17k) / seagrass (~293k) global rows, but Global Mangrove Watch's
# 1.2M rows need the bbox pushed down to the OGR read itself (see
# load_habitat_polygons) rather than materializing every geometry worldwide
# first and filtering after.
_SKIP_STEMS = ("_pt_", "_pt.", "_ln_", "_ln.", "_line", "_point")
_POLY_STEMS = ("_py_", "_py.", "poly", "v4", "v7")


def _find_habitat_shapefile(sub_dir: Path, habitat: str) -> Path:
    candidates: list[Path] = []
    for pattern in ("**/*.gpkg", "**/*.shp"):
        candidates.extend(sorted(sub_dir.rglob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No shapefile or gpkg found under {sub_dir}")

    def _score(p: Path) -> tuple[int, int]:
        stem = p.stem.lower()
        is_bad = any(k in stem for k in _SKIP_STEMS)
        is_poly = any(k in stem for k in _POLY_STEMS) or habitat in stem
        return (0 if is_bad else 1, 1 if is_poly else 0)

    return sorted(candidates, key=_score, reverse=True)[0]


def load_and_rasterize_habitat(
    habitat_dir: Path, habitat: str, height: int, width: int, transform,
) -> np.ndarray:
    """Load one habitat's polygons, prefiltered to the Coral Triangle bbox,
    rasterize onto the shared grid, and return only the boolean mask.

    Everything about the polygon GeoDataFrame is local to this function so it
    can be garbage-collected the moment this returns — Global Mangrove Watch's
    ~440k Coral-Triangle polygons (of 1.2M worldwide) are traced at very fine
    (sub-cell) detail and held multiple gigabytes of Shapely geometry in
    memory; holding that alongside coral/seagrass and several boolean grids
    pushed a 16 GB machine to <650 MB free and made everything (including
    unrelated tool calls) thrash. Two things fix it: (a) simplify to roughly
    the raster's own resolution immediately after load, but ONLY above
    SIMPLIFY_THRESHOLD_POLYGONS — detail finer than one grid cell is invisible
    to rasterize anyway, so this is free precision to give up for mangrove,
    but applying it to coral/seagrass's already-small polygon counts served
    no memory purpose and degenerated hundreds of the smallest reef/seagrass
    patches into empty geometries (rasterize then warns and silently drops
    them, thousands of warnings for no benefit) — and (b) never keep more
    than one habitat's raw geometries alive at once, which this function's
    scope enforces.

    Uses geopandas/OGR's `bbox=` read filter rather than a full read +
    post-hoc `.intersects()`: a full read materializes every polygon
    worldwide before anything can be discarded, and a vector
    `unary_union`/`.clip()` over that didn't finish in a practical time even
    after prefiltering. Pushing the bbox into the read itself (~3 min
    one-time cost for mangrove) cuts it straight to the Coral-Triangle rows.
    """
    sub_dir = habitat_dir / habitat
    if not sub_dir.exists():
        raise FileNotFoundError(f"{sub_dir} not found — run habitat_extents.sh first")
    path = _find_habitat_shapefile(sub_dir, habitat)
    b = CORAL_TRIANGLE_BBOX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(str(path), bbox=(b["west"], b["south"], b["east"], b["north"]))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    n_polygons = len(gdf)

    if n_polygons == 0:
        print(f"{habitat}: 0 polygons in the Coral Triangle bbox")
        return np.zeros((height, width), dtype=bool)

    if n_polygons > SIMPLIFY_THRESHOLD_POLYGONS:
        geometries = gdf.geometry.simplify(
            RASTER_RESOLUTION_DEG / 2.0, preserve_topology=False,
        ).tolist()
    else:
        geometries = gdf.geometry.tolist()
    del gdf  # drop the GeoDataFrame before rasterizing; only the geometry list is needed now

    with warnings.catch_warnings():
        # A handful of degenerate (empty/invalid) geometries after simplify is
        # expected at mangrove's scale — rasterize's own skip-and-warn is the
        # correct behaviour, just not worth a warning per occurrence.
        warnings.simplefilter("ignore", category=rio_features.ShapeSkipWarning)
        mask = rio_features.rasterize(
            ((geom, 1) for geom in geometries), out_shape=(height, width),
            transform=transform, fill=0, dtype="uint8",
        ).astype(bool)
    del geometries
    gc.collect()  # release the simplified geometry list before the next habitat loads
    print(f"{habitat}: {n_polygons} polygons -> {int(mask.sum())} grid cells "
          f"({RASTER_RESOLUTION_DEG}-degree resolution) in the Coral Triangle bbox")
    return mask


def build_habitat_masks(habitat_dir: Path) -> dict[str, np.ndarray]:
    """Boolean habitat masks on the shared grid, plus '_union' = any habitat.

    Habitats are loaded and rasterized one at a time (see
    load_and_rasterize_habitat) rather than all-polygons-then-all-rasters, so
    peak memory is one habitat's raw geometries plus the (cheap, ~92 MB each)
    boolean grids — not all three raw polygon sets simultaneously.
    """
    height, width, transform = grid_shape_transform()
    masks = {h: load_and_rasterize_habitat(habitat_dir, h, height, width, transform) for h in HABITATS}
    masks["_union"] = masks["coral"] | masks["seagrass"] | masks["mangrove"]
    return masks


# ===========================================================================
# Section 2 — Point sampling on the raster grid
# ===========================================================================

def _cells_to_jittered_points(
    flat_indices: np.ndarray, height: int, width: int, transform, np_rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Grid-cell centres + uniform sub-cell jitter -> continuous (lon, lat) points."""
    rows, cols = np.unravel_index(flat_indices, (height, width))
    cx, cy = rasterio.transform.xy(transform, rows, cols)
    cx = np.asarray(cx) + (np_rng.random(len(rows)) - 0.5) * RASTER_RESOLUTION_DEG
    cy = np.asarray(cy) + (np_rng.random(len(rows)) - 0.5) * RASTER_RESOLUTION_DEG
    return list(zip(cx.tolist(), cy.tolist()))


def sample_presence_points(
    mask: np.ndarray, n: int, rng: random.Random, height: int, width: int, transform,
) -> list[tuple[float, float]]:
    """n points drawn from mask==True cells (sub-cell jitter for continuous
    coordinates). Uniform-over-cells approximates area-weighting automatically
    since a larger reef/seagrass/mangrove patch simply occupies more cells."""
    idx_pool = np.flatnonzero(mask)
    if idx_pool.size == 0:
        return []
    np_rng = np.random.default_rng(rng.randrange(2**31))
    chosen = np_rng.choice(idx_pool, size=n, replace=idx_pool.size < n)
    return _cells_to_jittered_points(chosen, height, width, transform, np_rng)


def _water_mask_batch(points: list[tuple[float, float]]) -> list[bool]:
    """True where MOD44W says open water, for each (lon, lat) in points."""
    if not points:
        return []
    image = ee.Image(WATER_MASK_ASSET).select("water_mask")
    point_features = [
        ee.Feature(ee.Geometry.Point([lon, lat]), {"idx": i})
        for i, (lon, lat) in enumerate(points)
    ]
    reduced = image.reduceRegions(
        collection=ee.FeatureCollection(point_features), reducer=ee.Reducer.first(), scale=250,
    ).getInfo()
    result = [False] * len(points)
    for feature in reduced.get("features", []):
        props = feature.get("properties", {})
        idx = props.get("idx")
        if idx is not None:
            result[idx] = props.get("first") == 1
    return result


def sample_absence_points(
    habitat: str, masks: dict[str, np.ndarray], n: int, rng: random.Random,
    require_water: bool, height: int, width: int, transform, buffer_m: float = ABSENCE_BUFFER_M,
) -> list[tuple[float, float]]:
    """n points within buffer_m of habitat's own cells (a morphological
    dilation — the raster equivalent of a vector buffer), excluding every
    habitat's own cells, and (if require_water) classified open water."""
    mean_lat = (CORAL_TRIANGLE_BBOX["south"] + CORAL_TRIANGLE_BBOX["north"]) / 2.0
    cell_m = RASTER_RESOLUTION_DEG * METRES_PER_DEGREE_LAT
    radius_cells = max(1, round(buffer_m / cell_m))
    # Iterating a small 8-connectivity structure grows the mask outward by
    # roughly `radius_cells` in every direction (an octagon, not a perfect
    # circle) — much faster than dilating with one large explicit disk
    # structure over a multi-million-cell grid, and precise circularity
    # doesn't matter for a "nearby but not on it" absence zone.
    dilated = binary_dilation(masks[habitat], structure=np.ones((3, 3), dtype=bool), iterations=radius_cells)
    candidate_mask = dilated & ~masks["_union"]

    pool = np.flatnonzero(candidate_mask)
    if pool.size == 0:
        print(f"  WARNING: no candidate absence cells for {habitat} — widen ABSENCE_BUFFER_M.")
        return []

    np_rng = np.random.default_rng(rng.randrange(2**31))
    accepted: list[tuple[float, float]] = []
    remaining = pool.copy()

    for _ in range(MAX_ABSENCE_BATCHES):
        if len(accepted) >= n or remaining.size == 0:
            break
        batch_size = min(ABSENCE_BATCH_SIZE, remaining.size)
        batch = np_rng.choice(remaining, size=batch_size, replace=False)
        candidates = _cells_to_jittered_points(batch, height, width, transform, np_rng)
        if require_water and candidates:
            is_water = _water_mask_batch(candidates)
            candidates = [c for c, ok in zip(candidates, is_water) if ok]
        accepted.extend(candidates)
        remaining = np.setdiff1d(remaining, batch, assume_unique=True)

    if len(accepted) < n:
        print(f"  WARNING: only found {len(accepted)}/{n} valid absence points for {habitat} "
              f"after {MAX_ABSENCE_BATCHES} batches — widen ABSENCE_BUFFER_M or raise the batch size/count.")
    rng.shuffle(accepted)
    return accepted[:n]


# ===========================================================================
# Section 3 — Build the point set for one habitat
# ===========================================================================

def build_habitat_points(
    habitat: str, masks: dict[str, np.ndarray],
    n_presence: int = N_PRESENCE, n_absence: int = N_ABSENCE, seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    rng = random.Random(seed ^ (hash(habitat) & 0xFFFFFFFF))
    height, width, transform = grid_shape_transform()

    presence_xy = sample_presence_points(masks[habitat], n_presence, rng, height, width, transform)
    print(f"{habitat}: presence {len(presence_xy)}/{n_presence}")

    absence_xy = sample_absence_points(
        habitat, masks, n_absence, rng, require_water=habitat in SUBMERGED_HABITATS,
        height=height, width=width, transform=transform,
    )
    print(f"{habitat}: absence  {len(absence_xy)}/{n_absence}")

    rows = []
    for i, (lon, lat) in enumerate(presence_xy):
        rows.append({"point_id": f"{habitat}_presence_{i:04d}", "habitat": habitat,
                     "label": 1, "longitude": lon, "latitude": lat})
    for i, (lon, lat) in enumerate(absence_xy):
        rows.append({"point_id": f"{habitat}_absence_{i:04d}", "habitat": habitat,
                     "label": 0, "longitude": lon, "latitude": lat})
    df = pd.DataFrame(rows)
    df["dataset"] = f"{habitat} presence/absence, Coral Triangle (WCMC/GMW extents)"
    return df


# ===========================================================================
# Section 4 — Per-point imagery sampling (fixed reference window)
# ===========================================================================

S2_OUTPUT_COLUMNS = SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS)
LS_OUTPUT_COLUMNS = LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS


def _init_output_columns(frame: pd.DataFrame, submerged: bool) -> pd.DataFrame:
    df = frame.copy()
    for column in S2_OUTPUT_COLUMNS + (S2_LYZENGA_COLUMNS if submerged else []):
        df[column] = np.nan
    for column in LS_OUTPUT_COLUMNS + (LS_LYZENGA_COLUMNS if submerged else []):
        df[column] = np.nan
    for column in ("s2_scene_date", "s2_source", "ls_scene_date", "ls_source"):
        df[column] = ""
    df["s2_clear_score"] = np.nan
    df["ls_clear_score"] = np.nan
    return df


def _with_retries(fn, attempts: int = 3, base_delay: float = 1.5):
    result = None
    for attempt in range(attempts):
        try:
            result = fn()
        except Exception:
            result = None
        if result:
            return result
        if attempt < attempts - 1:
            time.sleep(base_delay * (attempt + 1) + random.uniform(0, 0.5))
    return result


def _sample_gee_point(manager, candidates_fn, lon: float, lat: float, download_bands: list[str],
                      scale_m: int, build_features_fn) -> tuple[dict | None, float]:
    date_start, date_end = format_date_window(REFERENCE_DATE, WINDOW_DAYS)
    clarity_buffer_m = max(BAND_SAMPLE_BUFFER_M, LOCAL_CLARITY_BUFFER_M)
    candidates = candidates_fn(lon, lat, date_start, date_end, buffer_m=clarity_buffer_m)
    selected = select_by_local_clarity(candidates, REFERENCE_DATE)
    if selected is None:
        return None, np.nan
    image = manager.get_image_by_asset_id(selected["asset_id"])
    if image is None:
        return None, np.nan
    region = ee.Geometry.Point([lon, lat]).buffer(BAND_SAMPLE_BUFFER_M).bounds()
    stats = image.select(download_bands).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=scale_m,
        bestEffort=True, maxPixels=1_000_000,
    ).getInfo()
    if not stats:
        return None, np.nan
    clear_score = selected.get("clear_score")
    return (build_features_fn(stats, parse_date_value(selected.get("date", ""))),
            float(clear_score) if clear_score is not None else np.nan)


def _process_point(row_index: int, lon: float, lat: float, submerged: bool,
                   s2_mgr: "Sentinel2Manager | None", ls_mgr: "LandsatManager | None") -> dict:
    result: dict = {"row_index": row_index, "s2_hit": False, "ls_hit": False}

    s2_features, s2_clear = (None, np.nan)
    if s2_mgr is not None:
        s2_features, s2_clear = _with_retries(lambda: _sample_gee_point(
            s2_mgr, s2_local_clear_candidates, lon, lat,
            SENTINEL2_DOWNLOAD_BANDS, 10, build_sentinel2_feature_values,
        )) or (None, np.nan)
    if s2_features:
        for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
            result[column] = s2_features.get(column, np.nan)
        result["s2_scene_date"] = s2_features.get("scene_date", "")
        result["s2_source"] = "gee"
        result["s2_clear_score"] = s2_clear
        if submerged:
            lyz = add_lyzenga_columns(
                {c: s2_features.get(c, np.nan) for c in SENTINEL2_BAND_COLUMNS}, S2_LYZENGA_PAIRS,
            )
            for column in S2_LYZENGA_COLUMNS:
                result[column] = lyz.get(column, np.nan)
        result["s2_hit"] = True

    ls_features, ls_clear = (None, np.nan)
    if ls_mgr is not None:
        ls_features, ls_clear = _with_retries(lambda: _sample_gee_point(
            ls_mgr, landsat_local_clear_candidates, lon, lat,
            LANDSAT_DOWNLOAD_BANDS, 30, build_landsat_feature_values,
        )) or (None, np.nan)
    if ls_features:
        for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
            result[column] = ls_features.get(column, np.nan)
        result["ls_scene_date"] = ls_features.get("ls_scene_date", "")
        result["ls_source"] = "gee"
        result["ls_clear_score"] = ls_clear
        if submerged:
            lyz = add_lyzenga_columns(
                {c: ls_features.get(c, np.nan) for c in LANDSAT_BAND_COLUMNS}, LS_LYZENGA_PAIRS,
            )
            for column in LS_LYZENGA_COLUMNS:
                result[column] = lyz.get(column, np.nan)
        result["ls_hit"] = True

    return result


# ===========================================================================
# Section 5 — CSV builder
# ===========================================================================

def sample_habitat_imagery(
    points: pd.DataFrame, habitat: str, output_dir: Path,
    gee_project: str, max_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    submerged = habitat in SUBMERGED_HABITATS
    frame = _init_output_columns(points, submerged)
    frame["_done"] = False

    checkpoint_path = output_dir / f"{habitat}.checkpoint.csv"
    if checkpoint_path.exists():
        try:
            str_columns = ["s2_scene_date", "s2_source", "ls_scene_date", "ls_source"]
            checkpoint = pd.read_csv(checkpoint_path, dtype={c: str for c in str_columns})
            for column in str_columns:
                checkpoint[column] = checkpoint[column].fillna("")
            matches = (len(checkpoint) == len(frame)
                       and "point_id" in checkpoint.columns
                       and checkpoint["point_id"].astype(str).equals(frame["point_id"].astype(str)))
        except Exception:
            checkpoint, matches = None, False
        if matches:
            frame = checkpoint
            frame["_done"] = frame["_done"].fillna(False).astype(bool)
            print(f"Resuming {habitat} from checkpoint: {int(frame['_done'].sum())}/{len(frame)} points already done")
        else:
            print(f"Checkpoint at {checkpoint_path} doesn't match this data; starting fresh")

    ee.Initialize(project=gee_project)
    s2_mgr = Sentinel2Manager(gee_project=gee_project)
    ls_mgr = LandsatManager(gee_project=gee_project)

    pending = [
        (index, float(row["longitude"]), float(row["latitude"]))
        for index, row in frame.iterrows() if not bool(row["_done"])
    ]

    s2_hits = int((frame["s2_source"] == "gee").sum())
    ls_hits = int((frame["ls_source"] == "gee").sum())

    def _save_checkpoint() -> None:
        tmp = checkpoint_path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(checkpoint_path)

    pbar = tqdm(total=len(frame), initial=len(frame) - len(pending),
                desc=f"Sampling {habitat}", unit="pt")
    pbar.set_postfix(s2=s2_hits, ls=ls_hits)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_point, index, lon, lat, submerged, s2_mgr, ls_mgr): index
            for index, lon, lat in pending
        }
        for future in as_completed(futures):
            row_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\nPoint at row {row_index} failed: {exc}")
                result = {"row_index": row_index, "s2_hit": False, "ls_hit": False}

            for column, value in result.items():
                if column in ("row_index", "s2_hit", "ls_hit"):
                    continue
                frame.at[row_index, column] = value
            frame.at[row_index, "_done"] = True
            s2_hits += int(result.get("s2_hit", False))
            ls_hits += int(result.get("ls_hit", False))

            pbar.update(1)
            pbar.set_postfix(s2=s2_hits, ls=ls_hits)
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                _save_checkpoint()

    pbar.close()
    if pending:
        _save_checkpoint()

    s2_only = S2_OUTPUT_COLUMNS + (S2_LYZENGA_COLUMNS if submerged else []) + ["s2_scene_date", "s2_source", "s2_clear_score"]
    ls_only = LS_OUTPUT_COLUMNS + (LS_LYZENGA_COLUMNS if submerged else []) + ["ls_scene_date", "ls_source", "ls_clear_score"]
    shared = [c for c in frame.columns if c not in s2_only + ls_only + ["_done"]]

    s2_frame = frame[shared + s2_only]
    ls_frame = frame[shared + ls_only]

    for sensor_frame, filename in ((s2_frame, f"{habitat}_sentinel2_with_bands.csv"),
                                   (ls_frame, f"{habitat}_landsat_with_bands.csv")):
        out_path = output_dir / filename
        sensor_frame.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({len(sensor_frame)} points x {len(sensor_frame.columns)} columns)")

    print(f"{habitat} imagery coverage: Sentinel-2 {s2_hits}/{len(frame)}, Landsat {ls_hits}/{len(frame)}")
    checkpoint_path.unlink(missing_ok=True)
    return s2_frame, ls_frame


# ===========================================================================
# CLI
# ===========================================================================

def _habitats_from_arg(arg: str) -> list[str]:
    return list(HABITATS) if arg == "all" else [arg]


def _cmd_points(args) -> None:
    habitat_dir = Path(args.habitat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(h in SUBMERGED_HABITATS for h in _habitats_from_arg(args.habitat)):
        gee_project = args.gee_project
        if gee_project is None:
            try:
                gee_project = load_credentials().get("gee_project")
            except FileNotFoundError:
                gee_project = None
        if not gee_project:
            raise FileNotFoundError(
                "No Earth Engine project found — coral/seagrass absence sampling needs "
                "the MOD44W water mask. Set common/credentials.json or pass --gee-project."
            )
        ee.Initialize(project=gee_project)

    # All three habitats are always loaded and rasterized regardless of
    # --habitat: absence sampling excludes every habitat's cells, not just
    # the one being built, so the union mask needs all three either way.
    masks = build_habitat_masks(habitat_dir)
    for habitat in _habitats_from_arg(args.habitat):
        points = build_habitat_points(habitat, masks)
        print(f"\n{habitat}: {len(points)} points ({(points['label'] == 1).sum()} presence, "
              f"{(points['label'] == 0).sum()} absence)")
        if args.out_dir_points:
            out_path = output_dir / f"{habitat}_points.csv"
            points.to_csv(out_path, index=False)
            print(f"Wrote {out_path}")


def _cmd_build(args) -> None:
    habitat_dir = Path(args.habitat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gee_project = args.gee_project
    if gee_project is None:
        try:
            gee_project = load_credentials().get("gee_project")
        except FileNotFoundError:
            gee_project = None
    if not gee_project:
        raise FileNotFoundError("No Earth Engine project found in common/credentials.json")
    ee.Initialize(project=gee_project)

    habitats = _habitats_from_arg(args.habitat)
    needs_fresh_points = args.regenerate_points or any(
        not (output_dir / f"{h}_points.csv").exists() for h in habitats
    )
    # All three habitats load/rasterize together (absence needs the full
    # union regardless of which habitat(s) are requested) — skipped entirely
    # when every requested habitat already has a saved point set.
    masks = build_habitat_masks(habitat_dir) if needs_fresh_points else None

    for habitat in habitats:
        points_path = output_dir / f"{habitat}_points.csv"
        if points_path.exists() and not args.regenerate_points:
            points = pd.read_csv(points_path)
            print(f"Reusing existing point set: {points_path} ({len(points)} points)")
        else:
            points = build_habitat_points(habitat, masks)
            points.to_csv(points_path, index=False)
            print(f"Wrote {points_path}")
        sample_habitat_imagery(points, habitat, output_dir, gee_project, max_workers=args.workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coral/seagrass/mangrove presence-absence pipeline (Landsat, Sentinel-2), Coral Triangle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'python ecology/habitat_presence.py <command> --help' for per-command options.",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    p = sub.add_parser("points", help="Build & summarise presence/absence point sets (no imagery)")
    p.add_argument("--habitat-dir", default=str(DEFAULT_HABITAT_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--habitat", choices=HABITATS + ["all"], default="all")
    p.add_argument("--gee-project", type=str, default=None)
    p.add_argument("--out-dir-points", action="store_true", help="Write <habitat>_points.csv")

    p = sub.add_parser("build", help="Build point sets (if not already saved) and sample imagery")
    p.add_argument("--habitat-dir", default=str(DEFAULT_HABITAT_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--habitat", choices=HABITATS + ["all"], default="all")
    p.add_argument("--gee-project", type=str, default=None)
    p.add_argument("--regenerate-points", action="store_true", help="Resample points even if <habitat>_points.csv exists")
    p.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    dispatch = {"points": _cmd_points, "build": _cmd_build}
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()
