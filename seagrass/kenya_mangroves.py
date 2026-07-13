"""Attach Landsat and Sentinel-2 band values + indices to the Kenya mangrove plot dataset.

Source data:
/home/ysi/Documents/Uni/MQPostdoc/projects/superres/kenya_mangroves/mangrove_data2018_editedv2.csv

Unlike seagrass/indonesia_mangroves.py's hex-grid dataset (no dates, arbitrary
planning-unit centroids needing a manual --crs), this is real field survey data:
5566 individual-tree measurements that collapse to 45 physical plots (site +
plot_no), each with its own real lon/lat and a survey date. That's the same
shape as seagrass/indonesia.py's per-observation nearest-clear-scene matching
(see common/gee_clear_sky.py), so this script follows that pattern rather than
indonesia_mangroves.py's single composite-image approach.

Spatial grouping: the data already carries an explicit "site" field (18 named
sites) and "plot_no", so no nearest-neighbour clustering is needed — (site,
plot_no) is a clean, already-correct key for one physical plot. Confirmed via
inspection: forest_type / percentage_cover / plot_size_m2 / date are constant
within every (site, plot_no) group (0 exceptions across all 45 plots), and
each plot's own lon/lat is distinct, so plots are the natural point unit.

No Lyzenga / depth-invariant indices here — those model light attenuation
through a water column and don't apply to mangrove canopy reflectance (the
canopy is imaged from above the waterline, not through it).

ACOLITE (all 45 plots sit in one small ~24 x 11 km area near Lamu, so unlike
Indonesia there's only one bbox, no clusters):

  # 1. See the plot date summary and bounding box
  python seagrass/kenya_mangroves.py dates

  # 2. Download Level-1 scenes (free accounts — see seagrass/tampa_bay.py's
  #    docstring for CDSE/USGS account setup, identical here)
  python seagrass/kenya_mangroves.py download --scenes /raw_scenes/ \\
      --cdse-user you@email.com --cdse-pass yourpass \\
      --usgs-user yourname --usgs-token yourtoken

  # 3. Run ACOLITE
  python seagrass/kenya_mangroves.py acolite --scenes /raw_scenes/ --acolite-out /acolite_out/

  # 4. Build the CSVs (ACOLITE output primary; falls back to GEE per-plot for
  #    anything ACOLITE didn't cover). --gee-project can be omitted if
  #    gee_project is set in common/credentials.json.
  python seagrass/kenya_mangroves.py build --acolite-dir /acolite_out/

Why ACOLITE matters here specifically: Sentinel-2 surface-reflectance (L2A) has
no coverage for this part of Kenya anywhere near the 2018 survey dates —
confirmed live: COPERNICUS/S2_SR_HARMONIZED has only 8 scenes at this location
across all of 2017-2018 (nearest cluster is mid/late December, 75-340+ days
from the actual Jan/Sept/Oct 2018 survey dates), while COPERNICUS/S2_HARMONIZED
(raw L1C) has 47 scenes in the same window — the raw acquisitions exist, ESA's
L2A archive just wasn't backfilled for this region that far back. Skipping
steps 1-3 and running 'build' with just --gee-project uses GEE's L2A/SR
products for everything, which means Sentinel-2 will come back essentially
blank for 2018 plots; Landsat is unaffected (LaSRC goes back further).

Output: two plot-level CSVs (one row per plot — 45 rows) with S2/Landsat bands
+ indices attached, plus two tree-level CSVs that repeat each plot's imagery
values onto its individual tree rows (5566 rows) for per-tree analyses —
satellite pixels can't resolve individual trees, so the bands are naturally a
plot-level attribute; the tree-level files just carry that value through.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.common import format_date_window, parse_date_object, parse_date_value
from common.credentials import load_credentials
from common.gee_clear_sky import (
    s2_local_clear_candidates,
    landsat_local_clear_candidates,
    select_by_local_clarity,
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

DEFAULT_INPUT = Path(
    "/home/ysi/Documents/Uni/MQPostdoc/projects/superres/kenya_mangroves/mangrove_data2018_editedv2.csv"
)

# Same physical footprint for both sensors (see seagrass/indonesia.py's
# BAND_SAMPLE_BUFFER_M comment) so a 30 m vs 10 m pixel doesn't bias comparisons.
BAND_SAMPLE_BUFFER_M = 45

KENYA_S2_WINDOW_DAYS = 60
KENYA_LS_WINDOW_DAYS = 60

# ±0.05° buffer around the 45 plots' extent (same convention as
# seagrass/tampa_bay.py's TAMPA_BAY_LIMIT). ACOLITE limit format: [south, west, north, east]
KENYA_LIMIT = [-2.361, 40.672, -2.158, 40.989]


def load_tree_data(input_csv: Path) -> pd.DataFrame:
    """Read the raw tree-level CSV. utf-8-sig strips the BOM; column names get
    stripped since the source has a stray " date " with trailing whitespace."""
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df["species_norm"] = df["species"].astype(str).str.strip().str.title()
    return df


def _mode_or_first(series: pd.Series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else series.iloc[0]


def build_plot_frame(trees: pd.DataFrame) -> pd.DataFrame:
    """Collapse tree-level rows to one row per physical plot (site, plot_no)."""
    plots = trees.groupby(["site", "plot_no"], as_index=False).agg(
        area=("area", "first"),
        longitude=("x_longitude", "first"),
        latitude=("y_latitude", "first"),
        survey_date=("date", "first"),
        forest_type=("forest_type", "first"),
        percentage_cover=("percentage_cover", "first"),
        plot_size_m2=("plot_size_m2", "first"),
        tree_count=("tree_no", "count"),
        mean_dbh_cm=("dbh_cm", "mean"),
        mean_height_m=("height_m", "mean"),
        max_height_m=("height_m", "max"),
        species_richness=("species_norm", "nunique"),
        dominant_species=("species_norm", _mode_or_first),
        live_fraction=("comment", lambda s: (s.fillna("Live").str.strip() == "Live").mean()),
    )
    return plots


def load_plot_dates(input_csv: Path) -> list[str]:
    """Unique ISO survey dates across all plots, for the 'dates' summary."""
    trees = load_tree_data(input_csv)
    plots = build_plot_frame(trees)
    dates: set[str] = set()
    for value in plots["survey_date"]:
        parsed = parse_date_object(value)
        if parsed is not None:
            dates.add(parsed.date().isoformat())
    return sorted(dates)


def select_target_scene_dates(input_csv: Path) -> tuple[list[str], list[str]]:
    """Find the actual AOI-locally-clearest scene date for every plot, via GEE
    metadata only (no download). Each L1C/L1 product is roughly 1 GB, and a
    plain window download would pull every scene within ±window_days of each
    survey date regardless of whether it's ever used — for this dataset that's
    ~198 GB of Sentinel-2 alone to end up using a handful. This narrows
    'download' down to just the specific dates that select_by_local_clarity
    would actually pick, so only those get fetched.

    Returns (s2_target_dates, ls_target_dates) as sorted ISO date lists.
    """
    trees = load_tree_data(input_csv)
    plots = build_plot_frame(trees)

    s2_dates: set[str] = set()
    ls_dates: set[str] = set()
    for _, row in tqdm(plots.iterrows(), total=len(plots), desc="Finding target scenes", unit="plot"):
        lon, lat, survey_date = float(row["longitude"]), float(row["latitude"]), row["survey_date"]

        s2_start, s2_end = format_date_window(survey_date, KENYA_S2_WINDOW_DAYS)
        if s2_start:
            # L1C, not L2A: this is picking which *raw* scene to download for
            # ACOLITE, and L2A has real coverage gaps here (see module docstring)
            # that L1C doesn't — querying L2A would just find nothing to target.
            candidates = s2_local_clear_candidates(
                lon, lat, s2_start, s2_end, buffer_m=BAND_SAMPLE_BUFFER_M,
                collection_id="COPERNICUS/S2_HARMONIZED",
            )
            selected = select_by_local_clarity(candidates, survey_date)
            if selected:
                d = parse_date_object(selected.get("date", ""))
                if d is not None:
                    s2_dates.add(d.date().isoformat())

        ls_start, ls_end = format_date_window(survey_date, KENYA_LS_WINDOW_DAYS)
        if ls_start:
            candidates = landsat_local_clear_candidates(lon, lat, ls_start, ls_end, buffer_m=BAND_SAMPLE_BUFFER_M)
            selected = select_by_local_clarity(candidates, survey_date)
            if selected:
                d = parse_date_object(selected.get("date", ""))
                if d is not None:
                    ls_dates.add(d.date().isoformat())

    return sorted(s2_dates), sorted(ls_dates)


def _sample_plot_s2(
    mgr: Sentinel2Manager, lon: float, lat: float, survey_date: str,
    acolite_scenes: list[dict],
) -> tuple[dict | None, str]:
    """Returns (features, source) where source is 'acolite', 'gee', or ''."""
    if acolite_scenes:
        scene = select_acolite_scene(acolite_scenes, survey_date, "S2", KENYA_S2_WINDOW_DAYS)
        if scene:
            sampled = sample_acolite_nc(scene["path"], lon, lat, ACOLITE_S2_BAND_MAP)
            if sampled and not all(pd.isna(v) for v in sampled.values()):
                compute_sentinel2_indices(sampled)
                sampled["scene_date"] = parse_date_value(scene["date"])
                return sampled, "acolite"

    date_start, date_end = format_date_window(survey_date, KENYA_S2_WINDOW_DAYS)
    if not date_start:
        return None, ""
    candidates = s2_local_clear_candidates(lon, lat, date_start, date_end, buffer_m=BAND_SAMPLE_BUFFER_M)
    selected = select_by_local_clarity(candidates, survey_date)
    if selected is None:
        return None, ""
    image = mgr.get_image_by_asset_id(selected["asset_id"])
    if image is None:
        return None, ""
    region = ee.Geometry.Point([lon, lat]).buffer(BAND_SAMPLE_BUFFER_M).bounds()
    try:
        stats = image.select(SENTINEL2_DOWNLOAD_BANDS).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=10,
            bestEffort=True, maxPixels=1_000_000,
        ).getInfo()
    except Exception:
        return None, ""
    if not stats:
        return None, ""
    scene_date = parse_date_value(selected.get("date", ""))
    return build_sentinel2_feature_values(stats, scene_date), "gee"


def _sample_plot_landsat(
    mgr: LandsatManager, lon: float, lat: float, survey_date: str,
    acolite_scenes: list[dict],
) -> tuple[dict | None, str]:
    """Returns (features, source) where source is 'acolite', 'gee', or ''."""
    if acolite_scenes:
        scene = select_acolite_scene(acolite_scenes, survey_date, "LS", KENYA_LS_WINDOW_DAYS)
        if scene:
            sampled = sample_acolite_nc(scene["path"], lon, lat, ACOLITE_LS_BAND_MAP)
            if sampled and not all(pd.isna(v) for v in sampled.values()):
                compute_landsat_indices(sampled)
                sampled["ls_scene_date"] = parse_date_value(scene["date"])
                return sampled, "acolite"

    date_start, date_end = format_date_window(survey_date, KENYA_LS_WINDOW_DAYS)
    if not date_start:
        return None, ""
    candidates = landsat_local_clear_candidates(lon, lat, date_start, date_end, buffer_m=BAND_SAMPLE_BUFFER_M)
    selected = select_by_local_clarity(candidates, survey_date)
    if selected is None:
        return None, ""
    image = mgr.get_image_by_asset_id(selected["asset_id"])
    if image is None:
        return None, ""
    region = ee.Geometry.Point([lon, lat]).buffer(BAND_SAMPLE_BUFFER_M).bounds()
    try:
        stats = image.select(LANDSAT_DOWNLOAD_BANDS).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=30,
            bestEffort=True, maxPixels=1_000_000,
        ).getInfo()
    except Exception:
        return None, ""
    if not stats:
        return None, ""
    scene_date = parse_date_value(selected.get("date", ""))
    return build_landsat_feature_values(stats, scene_date), "gee"


def sample_plots(
    plots: pd.DataFrame, gee_project: str, acolite_scenes: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample Sentinel-2 and Landsat bands/indices for every plot. Returns (s2_frame, ls_frame)."""
    acolite_scenes = acolite_scenes or []
    s2_mgr = Sentinel2Manager(gee_project=gee_project)
    ls_mgr = LandsatManager(gee_project=gee_project)

    s2_frame = plots.copy()
    for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
        s2_frame[column] = np.nan
    s2_frame["s2_scene_date"] = ""
    s2_frame["s2_source"] = ""

    ls_frame = plots.copy()
    for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
        ls_frame[column] = np.nan
    ls_frame["ls_scene_date"] = ""
    ls_frame["ls_source"] = ""

    s2_hits = ls_hits = 0
    pbar = tqdm(plots.iterrows(), total=len(plots), desc="Sampling plots", unit="plot")
    for idx, row in pbar:
        lon, lat, survey_date = float(row["longitude"]), float(row["latitude"]), row["survey_date"]

        s2_feat, s2_src = _sample_plot_s2(s2_mgr, lon, lat, survey_date, acolite_scenes)
        if s2_feat:
            for column in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
                s2_frame.at[idx, column] = s2_feat.get(column, np.nan)
            s2_frame.at[idx, "s2_scene_date"] = s2_feat.get("scene_date", "")
            s2_frame.at[idx, "s2_source"] = s2_src
            s2_hits += 1

        ls_feat, ls_src = _sample_plot_landsat(ls_mgr, lon, lat, survey_date, acolite_scenes)
        if ls_feat:
            for column in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
                ls_frame.at[idx, column] = ls_feat.get(column, np.nan)
            ls_frame.at[idx, "ls_scene_date"] = ls_feat.get("ls_scene_date", "")
            ls_frame.at[idx, "ls_source"] = ls_src
            ls_hits += 1

        pbar.set_postfix(s2=s2_hits, ls=ls_hits)

    return s2_frame, ls_frame


def prepare_kenya_mangroves(
    input_csv: Path,
    output_dir: Path | None = None,
    gee_project: str | None = None,
    acolite_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build plot-level and tree-level Sentinel-2/Landsat CSVs for the Kenya mangrove plots.

    Returns (s2_plots, ls_plots, s2_trees, ls_trees).
    """
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

    acolite_scenes: list[dict] = []
    if acolite_dir is not None and acolite_dir.exists():
        acolite_scenes = scan_acolite_output(acolite_dir)
        print(f"Indexed {len(acolite_scenes)} ACOLITE scenes in {acolite_dir}")

    trees = load_tree_data(input_csv)
    print(f"Loaded {len(trees)} tree measurements")

    plots = build_plot_frame(trees)
    print(f"Collapsed to {len(plots)} plots across {plots['site'].nunique()} sites")

    s2_plots, ls_plots = sample_plots(plots, gee_project, acolite_scenes)

    join_keys = ["site", "plot_no"]
    s2_plot_cols = [c for c in s2_plots.columns if c not in trees.columns or c in join_keys]
    ls_plot_cols = [c for c in ls_plots.columns if c not in trees.columns or c in join_keys]
    s2_trees = trees.merge(s2_plots[s2_plot_cols], on=join_keys, how="left")
    ls_trees = trees.merge(ls_plots[ls_plot_cols], on=join_keys, how="left")

    output_dir = output_dir or input_csv.parent
    outputs = [
        (s2_plots, "kenya_mangroves_plots_sentinel2_with_bands.csv"),
        (ls_plots, "kenya_mangroves_plots_landsat_with_bands.csv"),
        (s2_trees, "kenya_mangroves_trees_sentinel2_with_bands.csv"),
        (ls_trees, "kenya_mangroves_trees_landsat_with_bands.csv"),
    ]
    for frame, filename in outputs:
        out_path = output_dir / filename
        frame.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    return s2_plots, ls_plots, s2_trees, ls_trees


# ===========================================================================
# CLI
# ===========================================================================

def _cmd_dates(args) -> None:
    dates = load_plot_dates(Path(args.input))
    print(f"Unique plot survey dates: {len(dates)}")
    for d in dates:
        print(f"  {d}")
    print(f"\nBounding box (S, W, N, E): {KENYA_LIMIT}")
    s2_wins = merge_date_windows(dates, KENYA_S2_WINDOW_DAYS)
    ls_wins = merge_date_windows(dates, KENYA_LS_WINDOW_DAYS)
    print(f"S2 merged search windows (±{KENYA_S2_WINDOW_DAYS} days): {len(s2_wins)}")
    print(f"LS merged search windows (±{KENYA_LS_WINDOW_DAYS} days): {len(ls_wins)}")


# A full ±KENYA_S2_WINDOW_DAYS window pulls every scene in range regardless of
# whether it's usable; this narrow window around a pre-identified target date
# just needs to be wide enough that a date-boundary/timezone off-by-one
# doesn't miss the exact product.
TARGETED_DOWNLOAD_WINDOW_DAYS = 2


def _cmd_download(args) -> None:
    input_csv = Path(args.input)
    scenes_dir = Path(args.scenes)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    if args.full_window:
        s2_dates = ls_dates = load_plot_dates(input_csv)
        s2_window_days, ls_window_days = KENYA_S2_WINDOW_DAYS, KENYA_LS_WINDOW_DAYS
        print("Full-window mode: downloading every scene within "
              f"±{KENYA_S2_WINDOW_DAYS} days of each survey date (many GB more than needed).")
    else:
        gee_project = args.gee_project
        if gee_project is None:
            try:
                gee_project = load_credentials().get("gee_project")
            except FileNotFoundError:
                gee_project = None
        if not gee_project:
            raise FileNotFoundError("No Earth Engine project found in common/credentials/credentials.json")
        ee.Initialize(project=gee_project)
        print("Finding the actual best-candidate scene per plot via GEE metadata (no download)…")
        s2_dates, ls_dates = select_target_scene_dates(input_csv)
        s2_window_days = ls_window_days = TARGETED_DOWNLOAD_WINDOW_DAYS
        print(f"Targeting {len(s2_dates)} Sentinel-2 date(s) and {len(ls_dates)} Landsat date(s) "
              f"instead of the full ±{KENYA_S2_WINDOW_DAYS}-day window.")

    if args.cdse_user and args.cdse_pass:
        download_sentinel2(s2_dates, scenes_dir, args.cdse_user, args.cdse_pass,
                            limit=KENYA_LIMIT, window_days=s2_window_days)
    else:
        print("Skipping Sentinel-2 (no --cdse-user / --cdse-pass)")
    if args.usgs_user and args.usgs_token:
        download_landsat(ls_dates, scenes_dir, args.usgs_user, args.usgs_token,
                          limit=KENYA_LIMIT, window_days=ls_window_days)
    else:
        print("Skipping Landsat (no --usgs-user / --usgs-token)")


def _cmd_acolite(args) -> None:
    run_acolite_batch(
        scenes_dir=Path(args.scenes),
        output_dir=Path(args.acolite_out),
        limit=KENYA_LIMIT,
        acolite_path=Path(args.acolite_repo),
    )


def _cmd_build(args) -> None:
    prepare_kenya_mangroves(
        input_csv=Path(args.input),
        output_dir=args.output_dir,
        gee_project=args.gee_project,
        acolite_dir=Path(args.acolite_dir) if args.acolite_dir else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Kenya mangrove plot CSVs with Landsat and Sentinel-2 bands/indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'python seagrass/kenya_mangroves.py <command> --help' for per-command options.",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    # dates
    p = sub.add_parser("dates", help="Show plot date summary and bounding box")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)

    # download
    p = sub.add_parser("download", help="Download Level-1 scenes from CDSE / USGS")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--scenes", required=True, help="Directory to save downloaded scenes")
    p.add_argument("--cdse-user", metavar="EMAIL")
    p.add_argument("--cdse-pass", metavar="PASSWORD")
    p.add_argument("--usgs-user", metavar="USERNAME")
    p.add_argument("--usgs-token", metavar="TOKEN")
    p.add_argument("--gee-project", type=str, default=None,
                   help="Used to pre-identify target scenes via GEE metadata; "
                        "defaults to gee_project in common/credentials.json if omitted")
    p.add_argument("--full-window", action="store_true",
                   help=f"Download every scene within the full ±{KENYA_S2_WINDOW_DAYS}-day window "
                        "instead of just the pre-identified best-candidate dates (much more data)")

    # acolite
    p = sub.add_parser("acolite", help="Run ACOLITE on downloaded scenes")
    p.add_argument("--scenes", required=True)
    p.add_argument("--acolite-out", required=True, help="Output directory for NetCDF files")
    p.add_argument("--acolite-repo", default=str(ACOLITE_REPO_PATH))

    # build
    p = sub.add_parser("build", help="Build the plot-level and tree-level CSVs")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Path to mangrove_data2018_editedv2.csv")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to the input CSV's directory")
    p.add_argument("--gee-project", type=str, default=None,
                   help="Defaults to gee_project in common/credentials.json if omitted")
    p.add_argument("--acolite-dir", metavar="DIR",
                   help="Directory of ACOLITE NetCDF output (primary imagery source)")

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
