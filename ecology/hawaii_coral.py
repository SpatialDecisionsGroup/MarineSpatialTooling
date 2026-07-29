"""Attach Landsat and Sentinel-2 band values + indices to the NOAA Hawaii 2019
benthic cover dataset.

Source data:
  data/hawaii-coral/MV_BIA_CNET_ANALYSIS_DATA_HAWAII_2019.csv   (144,320 rows)
  data/hawaii-coral/ESD_BenthicCoverStRS_DataDictionary_HAWAII_20200729.csv

What a row in the source actually is
------------------------------------
The 144,320 rows are NOT 144,320 locations. This is a CoralNet point-intercept
dataset, and one row is one *annotated point on one photograph*:

  X_POS / Y_POS  pixel coordinates of the annotated point within the image —
                 image space, not geographic space (data dictionary: "X position
                 of the analyzed point on the image")
  IMAGE_NAME     SITE_YEAR_REP_PHOTOID — one downward photo along the transect
  LATITUDE /     recorded per SITE, not per photo or per point
  LONGITUDE

Confirmed against the file: 484 unique SITEs, 484 unique (SITE, lat, lon) triples
and 484 unique lat/lon pairs — so SITE ↔ coordinate is exactly 1:1, and SITE is
the only real spatial unit here. Each site has ~30 photos (min 17, max 33) and
exactly 10 annotated points per photo, giving the ~300 rows per site. Every site
also has exactly one survey date, one replicate (A) and one depth bin, so those
collapse without ambiguity.

That resolves the "same lon/lat, different x/y" observation: the repeated
coordinates are the same dive site, and the varying x/y are annotation positions
inside that site's photos. This script therefore collapses the point annotations
into one row per site, which is both the correct spatial unit and far below any
size concern — 484 rows, not 144k (the --max-sites cap defaults to 2000 and
never binds on this file).

How the many labels become columns on one row
----------------------------------------------
Point-intercept annotations aggregate to percent cover: for a given label,
cover% = (points with that label / points at the site) x 100. Every label at
all three tiers becomes its own column on the site's row, so nothing is lost:

  cover_<category>_pct    8 benthic CATEGORY_NAME values (coral, turf alga,
                          macroalga, coralline alga, sediment, ...) — sums to 100
  sub_<subcategory>_pct   22 benthic SUBCATEGORY_NAME values (branching/massive/
                          encrusting/foliose hard coral, sand, zoanthid, ...)
                          — also sums to 100
  genus_<genus>_pct       20 GENERA_NAME values above --min-label-share; sums to
                          <=100, the remainder being the rare-genus tail

Those three prefixes are exclusive to percent-cover columns, so
df.filter(like="genus_") is a safe feature selector. The per-site scalars are
named out of that namespace: n_categories / n_subcategories / n_genera /
n_coral_genera (richness), shannon_genus, dominant_category /
dominant_subcategory / dominant_genus.

Denominator: cover is expressed over *benthic* points, i.e. excluding the
NON_BENTHIC_CATEGORIES below ("Tape and wand" — the transect tape and the
diver's wand in frame — and "Unclassified", which is mostly "Shadow"). Those are
annotation artefacts, not seafloor, and leaving them in the denominator would
scale every real cover value down by however much of the frame the tape happened
to occupy. Both counts are kept (n_points_total, n_points_benthic) plus
pct_points_tape_wand / pct_points_unclassified, so the raw denominator is
recoverable and the discarded fraction is visible per site.

Depth and water-column correction
----------------------------------
Unlike the mangrove scripts, this benthos is imaged *through* a water column
(0-30 m), so Lyzenga depth-invariant indices and Beer-Lambert depth correction
both apply, same as seagrass/tampa_bay.py.

MIN_DEPTH/MAX_DEPTH are in feet and only recorded for coral demographic surveys
— present for 187 of the 484 sites. DEPTH_BIN is present for all 484, so the
Beer-Lambert correction falls back to the bin midpoint (Shallow 3 m, Mid 12 m,
Deep 24 m per the data dictionary's >0-6 / >6-18 / >18-30 m ranges) where a
measurement is missing; depth_source records which was used. The two agree well
where both exist (measured bin means: 4.3 / 12.1 / 21.8 m).

Caveat worth knowing before reading the results: optical bottom signal is
effectively gone below roughly 15-20 m even in Hawaii's clear water, so the
"Deep" bin (136 sites, ~22 m) should be expected to carry little or no benthic
information in either sensor. Filter on depth_bin / depth_mid_m before
concluding a variable is unlearnable.

Usage
-----
  # 1. Collapse to sites and inspect — no GEE, no network
  python ecology/hawaii_coral.py sites

  # 2. See the survey date spread and bounding box
  python ecology/hawaii_coral.py dates

  # 3. Build the two CSVs. --gee-project can be omitted if gee_project is set
  #    in common/credentials.json.
  python ecology/hawaii_coral.py build

There is no download/acolite subcommand here, unlike tampa_bay.py and
kenya_mangroves.py: the 484 sites span all eight main Hawaiian islands
(19.03-22.22 N, -160.25 to -154.80 E) over seven months, which is hundreds of
Level-1 products — not a single practical ACOLITE limit. Sites still sample from
ACOLITE NetCDFs if you point --acolite-dir at output produced elsewhere; GEE's
L2A/SR products are the default source.

Output: two CSVs, one row per site, sharing every site/cover column —
hawaii_coral_sentinel2_with_bands.csv and hawaii_coral_landsat_with_bands.csv.
"""

from __future__ import annotations

import argparse
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd
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
    S2_DRC_COLUMNS,
    LS_DRC_COLUMNS,
    S2_LYZENGA_COLUMNS,
    LS_LYZENGA_COLUMNS,
    add_lyzenga_columns,
    add_depth_corrected_columns,
)
from common.acolite_pipeline import (
    ACOLITE_S2_BAND_MAP,
    ACOLITE_LS_BAND_MAP,
    scan_acolite_output,
    select_acolite_scene,
    sample_acolite_nc,
)

DEFAULT_ROOT = Path("data/hawaii-coral")
SOURCE_FILE = "MV_BIA_CNET_ANALYSIS_DATA_HAWAII_2019.csv"
OUTPUT_STEM = "hawaii_coral"

# Same physical footprint for both sensors (see seagrass/indonesia.py's
# BAND_SAMPLE_BUFFER_M comment) so a 30 m vs 10 m pixel doesn't bias comparisons.
# A site's ~30 photos run along a transect of roughly this scale, so this also
# approximates the area the annotations actually describe.
BAND_SAMPLE_BUFFER_M = 45

# Benthic cover doesn't change measurably over a few weeks, so a wide window is
# safe and buys far better odds of an AOI-locally-clear scene than the 15/30-day
# defaults in common/sentinel.py and common/landsat.py.
HAWAII_S2_WINDOW_DAYS = 45
HAWAII_LS_WINDOW_DAYS = 45

# Annotation artefacts, not seafloor — excluded from the cover denominator (see
# module docstring). "Unclassified" is dominated by "Shadow".
NON_BENTHIC_CATEGORIES = frozenset({"Tape and wand", "Unclassified"})

# DEPTH_BIN midpoints in metres, from the data dictionary's ranges
# (Shallow >0-6 m, Mid >6-18 m, Deep >18-30 m). Used for the Beer-Lambert
# correction at the 297 sites with no measured MIN_DEPTH/MAX_DEPTH.
DEPTH_BIN_MIDPOINT_M = {"Shallow": 3.0, "Mid": 12.0, "Deep": 24.0}

FEET_TO_METRES = 0.3048

# Beer-Lambert Kd (m^-1) for clear oceanic water, overriding common/depth_correction.py's
# S2_KD / LS_KD. Those defaults (0.30 at blue, 0.15 at green) describe "moderately
# turbid coastal / estuarine water" — Tampa Bay, which is what they were written
# for. Hawaiian forereef is Jerlov type I/IA, roughly an order of magnitude
# clearer at blue-green, and using the turbid values here makes
# exp(2 x Kd x depth) blow past beer_lambert_correction's 1.0 cap at every site
# deeper than a few metres: verified against the shared constants, drc_s2_b3 came
# back as a flat 1.0 for both a 12 m and a 24 m site, i.e. a constant column with
# no information in it for the 345 of 484 sites that are Mid or Deep.
#
# Values are anchored on pure-water absorption (Pope & Fry 1997), which is the
# physical floor for Kd and the dominant term in clear water, nudged up slightly
# for backscatter. Keys match the shared maps so the drc_* column names are
# unchanged.
#
# Red and NIR still saturate the cap, and should: at 665 nm the water column
# itself absorbs essentially all bottom signal within a few metres, so a
# saturated drc_s2_b4 is a true statement about the physics, not an artefact.
# Expect only the blue/green drc_* columns (b1-b3) to carry usable information,
# and only at Shallow and Mid depth.
HAWAII_S2_KD: dict[str, float] = {
    "s2_b1": 0.040,   # 443 nm
    "s2_b2": 0.045,   # 492 nm
    "s2_b3": 0.090,   # 560 nm
    "s2_b4": 0.450,   # 665 nm
    "s2_b5": 0.800,   # 704 nm
    "s2_b6": 2.500,   # 740 nm
    "s2_b7": 3.000,   # 783 nm
    "s2_b8": 4.500,   # 842 nm
    "s2_b8a": 4.800,  # 865 nm
}

HAWAII_LS_KD: dict[str, float] = {
    "ls_b1": 0.040,   # 443 nm
    "ls_b2": 0.043,   # 482 nm
    "ls_b3": 0.090,   # 561 nm
    "ls_b4": 0.420,   # 655 nm
    "ls_b5": 4.800,   # 865 nm
}

# Genus labels below this share of all benthic points get no column of their own
# — with 58 genera the tail is mostly all-zero columns that add width without
# adding signal. Their points still count toward the category/subcategory
# columns and toward n_points_benthic, so no points are dropped.
DEFAULT_MIN_LABEL_SHARE = 0.001

# 484 sites is already well under this; it exists so the script stays bounded if
# pointed at a larger regional file from the same NOAA series.
DEFAULT_MAX_SITES = 2000
SUBSAMPLE_SEED = 20190501

CHECKPOINT_EVERY = 50


# ===========================================================================
# Section 1 — Load the point annotations and collapse them to sites
# ===========================================================================

def _column_token(label: object) -> str:
    """'Turf growing on hard substrate' -> 'turf_growing_on_hard_substrate'."""
    token = re.sub(r"[^0-9a-z]+", "_", str(label).strip().lower())
    return token.strip("_") or "unknown"


def load_annotations(source_csv: Path) -> pd.DataFrame:
    """Read the raw point-annotation CSV (one row per annotated image point)."""
    if not source_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {source_csv}")
    df = pd.read_csv(source_csv, low_memory=False)
    for column in ("SITE", "ISLAND", "CATEGORY_NAME", "SUBCATEGORY_NAME",
                   "GENERA_NAME", "TIER_1", "REEF_ZONE", "DEPTH_BIN"):
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
    # DATE_ is 'd-mmm-yy' (e.g. 4-May-19); normalise once so every downstream
    # consumer sees an unambiguous ISO date rather than re-parsing the format.
    df["survey_date"] = pd.to_datetime(df["DATE_"], format="%d-%b-%y").dt.strftime("%Y-%m-%d")
    df["is_benthic"] = ~df["CATEGORY_NAME"].isin(NON_BENTHIC_CATEGORIES)
    return df


def _mode_or_first(series: pd.Series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else (series.iloc[0] if len(series) else "")


def _shannon(counts: pd.Series) -> float:
    """Shannon diversity H' over a label-count vector; 0.0 for a single label."""
    total = float(counts.sum())
    if total <= 0:
        return np.nan
    proportions = counts[counts > 0] / total
    return float(-(proportions * np.log(proportions)).sum())


def _site_depths(group: pd.DataFrame) -> tuple[float, float, float, str]:
    """(min_m, max_m, mid_m, depth_source) for one site.

    MIN_DEPTH/MAX_DEPTH are feet and constant within a site where present
    (verified: no site carries two different values). Falls back to the
    DEPTH_BIN midpoint when the site has no measurement.
    """
    min_ft = group["MIN_DEPTH"].dropna()
    max_ft = group["MAX_DEPTH"].dropna()
    if not min_ft.empty and not max_ft.empty:
        min_m = float(min_ft.iloc[0]) * FEET_TO_METRES
        max_m = float(max_ft.iloc[0]) * FEET_TO_METRES
        return min_m, max_m, (min_m + max_m) / 2.0, "measured"
    depth_bin = str(group["DEPTH_BIN"].iloc[0])
    mid_m = DEPTH_BIN_MIDPOINT_M.get(depth_bin, np.nan)
    return np.nan, np.nan, mid_m, "depth_bin_midpoint" if not np.isnan(mid_m) else ""


def _cover_table(df: pd.DataFrame, label_column: str, prefix: str,
                 keep_labels: list[str], benthic_totals: pd.Series) -> pd.DataFrame:
    """Wide percent-cover table: index SITE, one column per kept label.

    Percentages are over `benthic_totals` (the site's benthic point count), not
    over the label column's own total, so every tier shares one denominator and
    cover_*_pct sums to 100 across categories.
    """
    benthic = df[df["is_benthic"]]
    counts = (benthic.groupby(["SITE", label_column], observed=True)
              .size().unstack(fill_value=0))
    counts = counts.reindex(columns=keep_labels, fill_value=0)
    percent = counts.div(benthic_totals, axis=0) * 100.0
    percent.columns = [f"{prefix}{_column_token(c)}_pct" for c in percent.columns]
    return percent


def build_site_frame(
    df: pd.DataFrame, min_label_share: float = DEFAULT_MIN_LABEL_SHARE,
) -> pd.DataFrame:
    """Collapse point annotations to one row per SITE with percent-cover columns."""
    benthic = df[df["is_benthic"]]
    benthic_totals = benthic.groupby("SITE", observed=True).size()

    coral_genera = set(
        benthic.loc[benthic["TIER_1"] == "CORAL", "GENERA_NAME"].dropna().unique()
    )

    genus_counts = benthic["GENERA_NAME"].value_counts()
    genus_keep = sorted(genus_counts[genus_counts >= min_label_share * len(benthic)].index)
    genus_dropped = sorted(set(genus_counts.index) - set(genus_keep))
    category_keep = sorted(benthic["CATEGORY_NAME"].dropna().unique())
    subcategory_keep = sorted(benthic["SUBCATEGORY_NAME"].dropna().unique())

    grouped = df.groupby("SITE", observed=True)
    meta = grouped.agg(
        roundid=("ROUNDID", "first"),
        missionid=("MISSIONID", "first"),
        region_name=("REGION_NAME", "first"),
        island=("ISLAND", "first"),
        latitude=("LATITUDE", "first"),
        longitude=("LONGITUDE", "first"),
        reef_zone=("REEF_ZONE", "first"),
        depth_bin=("DEPTH_BIN", "first"),
        survey_date=("survey_date", "first"),
        obs_year=("OBS_YEAR", "first"),
        replicate=("REPLICATE", "first"),
        n_images=("IMAGE_NAME", "nunique"),
        n_points_total=("SITE", "size"),
        n_analysts=("ANALYST", "nunique"),
        analyst_primary=("ANALYST", _mode_or_first),
    )

    depths = grouped.apply(_site_depths, include_groups=False)
    meta[["depth_min_m", "depth_max_m", "depth_mid_m", "depth_source"]] = pd.DataFrame(
        depths.tolist(), index=depths.index
    )

    meta["n_points_benthic"] = benthic_totals.reindex(meta.index).fillna(0).astype(int)
    non_benthic = df[~df["is_benthic"]]
    for category in sorted(NON_BENTHIC_CATEGORIES):
        counts = (non_benthic[non_benthic["CATEGORY_NAME"] == category]
                  .groupby("SITE", observed=True).size())
        meta[f"pct_points_{_column_token(category)}"] = (
            counts.reindex(meta.index).fillna(0) / meta["n_points_total"] * 100.0
        )

    # Dominance / richness are computed on benthic points only, matching the
    # cover columns' denominator.
    benthic_grouped = benthic.groupby("SITE", observed=True)
    meta["dominant_category"] = benthic_grouped["CATEGORY_NAME"].agg(_mode_or_first)
    meta["dominant_subcategory"] = benthic_grouped["SUBCATEGORY_NAME"].agg(_mode_or_first)
    meta["dominant_genus"] = benthic_grouped["GENERA_NAME"].agg(_mode_or_first)
    # Named n_*/shannon_* rather than *_richness so that the cover_/sub_/genus_
    # prefixes stay exclusive to percent-cover columns — otherwise selecting
    # features with df.filter(like="genus_") silently picks up a richness count
    # and a diversity index alongside the cover values.
    meta["n_categories"] = benthic_grouped["CATEGORY_NAME"].nunique()
    meta["n_subcategories"] = benthic_grouped["SUBCATEGORY_NAME"].nunique()
    meta["n_genera"] = benthic_grouped["GENERA_NAME"].nunique()
    meta["n_coral_genera"] = (
        benthic[benthic["GENERA_NAME"].isin(coral_genera)]
        .groupby("SITE", observed=True)["GENERA_NAME"].nunique()
        .reindex(meta.index).fillna(0).astype(int)
    )
    meta["shannon_genus"] = (
        benthic.groupby(["SITE", "GENERA_NAME"], observed=True).size()
        .groupby("SITE", observed=True).agg(_shannon)
    )

    covers = [
        _cover_table(df, "CATEGORY_NAME", "cover_", category_keep, benthic_totals),
        _cover_table(df, "SUBCATEGORY_NAME", "sub_", subcategory_keep, benthic_totals),
        _cover_table(df, "GENERA_NAME", "genus_", genus_keep, benthic_totals),
    ]

    sites = pd.concat([meta] + covers, axis=1).reset_index().rename(columns={"SITE": "site"})
    sites["dataset"] = "NOAA ESD benthic cover, main Hawaiian Islands 2019"

    if genus_dropped:
        print(f"Genus columns: kept {len(genus_keep)} of {len(genus_counts)} "
              f"(>= {min_label_share:.3%} of benthic points); the remaining "
              f"{len(genus_dropped)} rare genera still count toward the "
              "category/subcategory columns.")
    return sites.sort_values("site").reset_index(drop=True)


def subsample_sites(sites: pd.DataFrame, max_sites: int) -> pd.DataFrame:
    """Cap the site count, allocating proportionally across (island, depth_bin).

    Deterministic (fixed seed) so reruns and checkpoints line up. A plain head()
    would silently drop whole islands, since sites are alphabetical by island
    code — that would bias exactly the variables this dataset is meant to test.
    """
    if len(sites) <= max_sites:
        return sites
    rng = random.Random(SUBSAMPLE_SEED)
    keep: list[int] = []
    strata = list(sites.groupby(["island", "depth_bin"], observed=True, dropna=False))
    for _, group in strata:
        quota = max(1, round(max_sites * len(group) / len(sites)))
        indices = list(group.index)
        rng.shuffle(indices)
        keep.extend(indices[:quota])
    rng.shuffle(keep)
    keep = sorted(keep[:max_sites])
    print(f"Capped {len(sites)} sites to {len(keep)} "
          f"(proportional across {len(strata)} island x depth-bin strata, seed {SUBSAMPLE_SEED})")
    return sites.loc[keep].sort_values("site").reset_index(drop=True)


def load_sites(
    root_dir: Path = DEFAULT_ROOT,
    min_label_share: float = DEFAULT_MIN_LABEL_SHARE,
    max_sites: int = DEFAULT_MAX_SITES,
) -> pd.DataFrame:
    df = load_annotations(root_dir / SOURCE_FILE)
    print(f"Loaded {len(df):,} point annotations "
          f"({df['IMAGE_NAME'].nunique():,} images, {df['SITE'].nunique()} sites)")
    sites = build_site_frame(df, min_label_share)
    print(f"Collapsed to {len(sites)} sites across {sites['island'].nunique()} islands")
    return subsample_sites(sites, max_sites)


# ===========================================================================
# Section 2 — Per-site imagery sampling
# ===========================================================================

S2_OUTPUT_COLUMNS = (SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS)
                     + S2_DRC_COLUMNS + S2_LYZENGA_COLUMNS)
LS_OUTPUT_COLUMNS = (LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS
                     + LS_DRC_COLUMNS + LS_LYZENGA_COLUMNS)


def _init_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for column in S2_OUTPUT_COLUMNS + LS_OUTPUT_COLUMNS:
        df[column] = np.nan
    for column in ("s2_scene_date", "s2_source", "ls_scene_date", "ls_source"):
        df[column] = ""
    df["s2_clear_score"] = np.nan
    df["ls_clear_score"] = np.nan
    return df


def _compute_corrections(band_values: dict, depth_m: float, kd_map, drc_columns,
                         lyzenga_pairs, lyzenga_columns) -> dict:
    """Beer-Lambert + Lyzenga columns for one sample. Pure — safe in worker threads."""
    band_values = {c: float(v) for c, v in band_values.items()}
    out: dict = {}
    if not np.isnan(depth_m) and depth_m > 0:
        corrected = add_depth_corrected_columns(dict(band_values), depth_m, kd_map)
        out.update({c: corrected.get(c, np.nan) for c in drc_columns})
    lyzenga = add_lyzenga_columns(dict(band_values), lyzenga_pairs)
    out.update({c: lyzenga.get(c, np.nan) for c in lyzenga_columns})
    return out


def _with_retries(fn, attempts: int = 3, base_delay: float = 1.5):
    """Retry fn() with backoff — recovers from transient GEE rate-limit/network
    hiccups under parallel load (same helper as seagrass/tampa_bay.py).

    Note the interaction with _sample_gee_site, which returns a (features,
    clear_score) tuple: a tuple is always truthy, so a clean "no scene in this
    window" result ends the loop after one attempt, while a raised exception
    still retries. That is the behaviour we want here — select_by_local_clarity
    only returns None when the window genuinely holds zero scenes, so re-asking
    would burn the backoff sleeps to get the same empty answer.
    """
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


def _sample_gee_site(manager, candidates_fn, lon: float, lat: float, survey_date: str,
                     window_days: int, download_bands: list[str], scale_m: int,
                     build_features_fn) -> tuple[dict | None, float]:
    """Pick the AOI-locally-clearest scene near the survey date and reduce it
    over the site's buffer. Returns (features, clear_score)."""
    date_start, date_end = format_date_window(survey_date, window_days)
    if not date_start:
        return None, np.nan
    clarity_buffer_m = max(BAND_SAMPLE_BUFFER_M, LOCAL_CLARITY_BUFFER_M)
    candidates = candidates_fn(lon, lat, date_start, date_end, buffer_m=clarity_buffer_m)
    selected = select_by_local_clarity(candidates, survey_date)
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


def _process_site(row_index: int, lon: float, lat: float, survey_date: str, depth_m: float,
                  s2_mgr: "Sentinel2Manager | None", ls_mgr: "LandsatManager | None",
                  acolite_scenes: list[dict], s2_window_days: int,
                  ls_window_days: int) -> dict:
    """Every S2/Landsat column for one site. Touches no shared state — safe to
    run concurrently in a thread pool."""
    result: dict = {"row_index": row_index, "s2_hit": False, "ls_hit": False}

    # --- Sentinel-2 -------------------------------------------------------
    s2_features: dict | None = None
    s2_date, s2_source, s2_clear = "", "", np.nan

    scene = select_acolite_scene(acolite_scenes, survey_date, "S2", s2_window_days) if acolite_scenes else None
    if scene:
        sampled = sample_acolite_nc(scene["path"], lon, lat, ACOLITE_S2_BAND_MAP)
        if sampled and not all(pd.isna(v) for v in sampled.values()):
            s2_features = compute_sentinel2_indices(sampled)
            s2_date, s2_source = parse_date_value(scene["date"]), "acolite"

    if not s2_features and s2_mgr is not None:
        sampled = _with_retries(lambda: _sample_gee_site(
            s2_mgr, s2_local_clear_candidates, lon, lat, survey_date, s2_window_days,
            SENTINEL2_DOWNLOAD_BANDS, 10, build_sentinel2_feature_values,
        ))
        if sampled and sampled[0]:
            s2_features, s2_clear = sampled
            s2_date, s2_source = s2_features.get("scene_date", ""), "gee"

    if s2_features:
        for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
            result[column] = s2_features.get(column, np.nan)
        result["s2_scene_date"], result["s2_source"] = s2_date, s2_source
        result["s2_clear_score"] = s2_clear
        result.update(_compute_corrections(
            {c: s2_features.get(c, np.nan) for c in SENTINEL2_BAND_COLUMNS},
            depth_m, HAWAII_S2_KD, S2_DRC_COLUMNS, S2_LYZENGA_PAIRS, S2_LYZENGA_COLUMNS,
        ))
        result["s2_hit"] = True

    # --- Landsat ----------------------------------------------------------
    ls_features: dict | None = None
    ls_date, ls_source, ls_clear = "", "", np.nan

    scene = select_acolite_scene(acolite_scenes, survey_date, "LS", ls_window_days) if acolite_scenes else None
    if scene:
        sampled = sample_acolite_nc(scene["path"], lon, lat, ACOLITE_LS_BAND_MAP)
        if sampled and not all(pd.isna(v) for v in sampled.values()):
            ls_features = compute_landsat_indices(sampled)
            ls_date, ls_source = parse_date_value(scene["date"]), "acolite"

    if not ls_features and ls_mgr is not None:
        sampled = _with_retries(lambda: _sample_gee_site(
            ls_mgr, landsat_local_clear_candidates, lon, lat, survey_date, ls_window_days,
            LANDSAT_DOWNLOAD_BANDS, 30, build_landsat_feature_values,
        ))
        if sampled and sampled[0]:
            ls_features, ls_clear = sampled
            ls_date, ls_source = ls_features.get("ls_scene_date", ""), "gee"

    if ls_features:
        for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
            result[column] = ls_features.get(column, np.nan)
        result["ls_scene_date"], result["ls_source"] = ls_date, ls_source
        result["ls_clear_score"] = ls_clear
        result.update(_compute_corrections(
            {c: ls_features.get(c, np.nan) for c in LANDSAT_BAND_COLUMNS},
            depth_m, HAWAII_LS_KD, LS_DRC_COLUMNS, LS_LYZENGA_PAIRS, LS_LYZENGA_COLUMNS,
        ))
        result["ls_hit"] = True

    return result


# ===========================================================================
# Section 3 — CSV builder
# ===========================================================================

def prepare_hawaii_coral(
    root_dir: Path = DEFAULT_ROOT,
    output_dir: Path | None = None,
    gee_project: str | None = None,
    acolite_dir: Path | None = None,
    min_label_share: float = DEFAULT_MIN_LABEL_SHARE,
    max_sites: int = DEFAULT_MAX_SITES,
    s2_window_days: int = HAWAII_S2_WINDOW_DAYS,
    ls_window_days: int = HAWAII_LS_WINDOW_DAYS,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the site-level Sentinel-2 and Landsat CSVs. Returns (s2_frame, ls_frame).

    Samples `max_workers` sites concurrently (thread pool — network-bound, not
    CPU-bound) and checkpoints to <output>.checkpoint.csv every CHECKPOINT_EVERY
    sites, so an interrupted run resumes on rerun rather than restarting.
    """
    output_dir = output_dir or root_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _init_output_columns(load_sites(root_dir, min_label_share, max_sites))
    frame["_done"] = False

    checkpoint_path = output_dir / f"{OUTPUT_STEM}.checkpoint.csv"
    if checkpoint_path.exists():
        try:
            # Force dtype on the string columns: {s2,ls}_scene_date hold YYYYMMDD
            # text, which plain read_csv reloads as float64 — the next write of a
            # date string into that column then raises TypeError.
            str_columns = ["s2_scene_date", "s2_source", "ls_scene_date", "ls_source"]
            checkpoint = pd.read_csv(checkpoint_path, dtype={c: str for c in str_columns})
            for column in str_columns:
                checkpoint[column] = checkpoint[column].fillna("")
            matches = (len(checkpoint) == len(frame)
                       and "site" in checkpoint.columns
                       and checkpoint["site"].astype(str).equals(frame["site"].astype(str)))
        except Exception:
            checkpoint, matches = None, False
        if matches:
            frame = checkpoint
            frame["_done"] = frame["_done"].fillna(False).astype(bool)
            print(f"Resuming from checkpoint: {int(frame['_done'].sum())}/{len(frame)} sites already done")
        else:
            print(f"Checkpoint at {checkpoint_path} doesn't match this data; starting fresh")

    acolite_scenes: list[dict] = []
    if acolite_dir is not None and acolite_dir.exists():
        acolite_scenes = scan_acolite_output(acolite_dir)
        print(f"Indexed {len(acolite_scenes)} ACOLITE scenes in {acolite_dir}")

    if gee_project is None:
        try:
            gee_project = load_credentials().get("gee_project")
        except FileNotFoundError:
            gee_project = None
    if not gee_project and not acolite_scenes:
        raise FileNotFoundError(
            "No Earth Engine project found in common/credentials/credentials.json "
            "and no --acolite-dir given — there is no imagery source to sample."
        )

    s2_mgr = ls_mgr = None
    if gee_project:
        ee.Initialize(project=gee_project)
        s2_mgr = Sentinel2Manager(gee_project=gee_project)
        ls_mgr = LandsatManager(gee_project=gee_project)

    pending = [
        (index, float(row["longitude"]), float(row["latitude"]), str(row["survey_date"]),
         float(row["depth_mid_m"]) if pd.notna(row["depth_mid_m"]) else np.nan)
        for index, row in frame.iterrows() if not bool(row["_done"])
    ]

    s2_hits = int(frame["s2_source"].isin(["acolite", "gee"]).sum())
    ls_hits = int(frame["ls_source"].isin(["acolite", "gee"]).sum())

    def _save_checkpoint() -> None:
        tmp = checkpoint_path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.replace(checkpoint_path)

    pbar = tqdm(total=len(frame), initial=len(frame) - len(pending),
                desc="Sampling imagery", unit="site")
    pbar.set_postfix(s2=s2_hits, ls=ls_hits)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_site, index, lon, lat, survey_date, depth_m,
                            s2_mgr, ls_mgr, acolite_scenes, s2_window_days, ls_window_days): index
            for index, lon, lat, survey_date, depth_m in pending
        }
        for future in as_completed(futures):
            row_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\nSite at row {row_index} failed: {exc}")
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

    s2_only = S2_OUTPUT_COLUMNS + ["s2_scene_date", "s2_source", "s2_clear_score"]
    ls_only = LS_OUTPUT_COLUMNS + ["ls_scene_date", "ls_source", "ls_clear_score"]
    shared = [c for c in frame.columns if c not in s2_only + ls_only + ["_done"]]

    s2_frame = frame[shared + s2_only]
    ls_frame = frame[shared + ls_only]

    for sensor_frame, filename in ((s2_frame, f"{OUTPUT_STEM}_sentinel2_with_bands.csv"),
                                   (ls_frame, f"{OUTPUT_STEM}_landsat_with_bands.csv")):
        out_path = output_dir / filename
        sensor_frame.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({len(sensor_frame)} sites x {len(sensor_frame.columns)} columns)")

    print(f"Imagery coverage: Sentinel-2 {s2_hits}/{len(frame)}, Landsat {ls_hits}/{len(frame)}")
    checkpoint_path.unlink(missing_ok=True)
    return s2_frame, ls_frame


# ===========================================================================
# CLI
# ===========================================================================

def _cmd_sites(args) -> None:
    sites = load_sites(Path(args.root), args.min_label_share, args.max_sites)
    cover_columns = [c for c in sites.columns if c.endswith("_pct") and not c.startswith("pct_points_")]
    print(f"\nSite frame: {len(sites)} rows x {len(sites.columns)} columns "
          f"({len(cover_columns)} cover columns)")
    print(f"\nSites per island:\n{sites['island'].value_counts().to_string()}")
    print(f"\nSites per depth bin:\n{sites['depth_bin'].value_counts().to_string()}")
    print(f"\nDepth source:\n{sites['depth_source'].value_counts().to_string()}")
    print(f"\nDominant benthic category:\n{sites['dominant_category'].value_counts().to_string()}")
    headline = [c for c in ("cover_coral_pct", "cover_turf_alga_pct", "cover_macroalga_pct",
                            "cover_coralline_alga_pct", "cover_sediment_pct") if c in sites.columns]
    print(f"\nHeadline cover columns (% of benthic points):\n"
          f"{sites[headline].describe().round(2).to_string()}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sites.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


def _cmd_dates(args) -> None:
    sites = load_sites(Path(args.root), args.min_label_share, args.max_sites)
    dates = sorted(sites["survey_date"].unique())
    print(f"\nUnique survey dates: {len(dates)}  ({dates[0]} .. {dates[-1]})")
    by_month = pd.to_datetime(sites["survey_date"]).dt.to_period("M").value_counts().sort_index()
    print(f"\nSites per month:\n{by_month.to_string()}")
    print(f"\nBounding box (S, W, N, E): ["
          f"{sites['latitude'].min():.3f}, {sites['longitude'].min():.3f}, "
          f"{sites['latitude'].max():.3f}, {sites['longitude'].max():.3f}]")
    print(f"Search windows: Sentinel-2 ±{args.s2_window_days} days, "
          f"Landsat ±{args.ls_window_days} days around each site's own date")


def _cmd_build(args) -> None:
    prepare_hawaii_coral(
        root_dir=Path(args.root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        gee_project=args.gee_project,
        acolite_dir=Path(args.acolite_dir) if args.acolite_dir else None,
        min_label_share=args.min_label_share,
        max_sites=args.max_sites,
        s2_window_days=args.s2_window_days,
        ls_window_days=args.ls_window_days,
        max_workers=args.workers,
    )


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help=f"Directory holding {SOURCE_FILE} (default {DEFAULT_ROOT})")
    parser.add_argument("--min-label-share", type=float, default=DEFAULT_MIN_LABEL_SHARE,
                        help="Minimum share of benthic points for a genus to get its own "
                             f"column (default {DEFAULT_MIN_LABEL_SHARE:.3f}); rarer genera "
                             "still count toward the category/subcategory columns")
    parser.add_argument("--max-sites", type=int, default=DEFAULT_MAX_SITES,
                        help=f"Cap on unique sites, allocated proportionally across island x "
                             f"depth-bin strata (default {DEFAULT_MAX_SITES}; this file has 484, "
                             "so it does not bind)")
    parser.add_argument("--s2-window-days", type=int, default=HAWAII_S2_WINDOW_DAYS)
    parser.add_argument("--ls-window-days", type=int, default=HAWAII_LS_WINDOW_DAYS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hawaii 2019 benthic cover pipeline (Landsat, Sentinel-2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("The source CSV's rows are annotated points on photos, not locations — "
                "484 sites carry all 144,320 of them.\n"
                "Run 'python ecology/hawaii_coral.py <command> --help' for per-command options."),
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    p = sub.add_parser("sites", help="Collapse to sites and summarise (no GEE, no network)")
    _add_shared_arguments(p)
    p.add_argument("--out", default=None, help="Optionally write the site frame to this CSV")

    p = sub.add_parser("dates", help="Show the survey date spread and bounding box")
    _add_shared_arguments(p)

    p = sub.add_parser("build", help="Build the two site-level CSVs (Sentinel-2, Landsat)")
    _add_shared_arguments(p)
    p.add_argument("--output-dir", default=None, help="Defaults to --root")
    p.add_argument("--gee-project", type=str, default=None,
                   help="Defaults to gee_project in common/credentials.json if omitted")
    p.add_argument("--acolite-dir", metavar="DIR",
                   help="Directory of ACOLITE NetCDF output, used in preference to GEE where it covers a site")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent sites sampled from GEE (default 8; network-bound, try higher)")

    args = parser.parse_args()
    dispatch = {"sites": _cmd_sites, "dates": _cmd_dates, "build": _cmd_build}
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()
