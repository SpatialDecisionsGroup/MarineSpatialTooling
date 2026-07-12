"""Prepare the Tampa Bay seagrass transect dataset as CSV.

Full pipeline (run steps in order, or combine flags to do them all at once):

  # 1. See which dates need satellite coverage and get the bounding box
  python seagrass/tampa_bay.py dates --root data/

  # 2. Download Level-1 scenes (free accounts required — see below)
  python seagrass/tampa_bay.py download --root data/ --scenes /raw_scenes/ \\
      --cdse-user you@email.com --cdse-pass yourpass   # Sentinel-2 via CDSE
      --usgs-user yourname --usgs-token yourtoken       # Landsat via USGS

  # 3. Run ACOLITE atmospheric correction on downloaded scenes
  python seagrass/tampa_bay.py acolite \\
      --scenes /raw_scenes/ --acolite-out /acolite_out/

  # 4. Build the CSVs (uses ACOLITE output; falls back to GEE if scenes are absent).
  #    Writes two files — tampa_transects_sentinel2_with_bands.csv and
  #    tampa_transects_landsat_with_bands.csv — from the -o base name.
  #    --gee-project can be omitted if gee_project is set in
  #    common/credentials.json.
  python seagrass/tampa_bay.py build --root data/ \\
      --acolite-dir /acolite_out/ --gee-project my-gee-project \\
      -o tampa_transects_with_bands.csv

  # Or run everything in one go:
  python seagrass/tampa_bay.py all --root data/ \\
      --scenes /raw_scenes/ --acolite-out /acolite_out/ \\
      --cdse-user ... --cdse-pass ... --usgs-user ... --usgs-token ... \\
      --gee-project my-gee-project -o tampa_transects_with_bands.csv

Note on GEE vs ACOLITE
----------------------
GEE is used as a fallback when no ACOLITE scenes are available (step 4 / 'build').
You can run 'build' with --gee-project only and skip steps 2-3 entirely; you will
get GEE's standard L2A atmospheric correction rather than ACOLITE's water-optimised
correction.  ACOLITE is an upgrade, not a requirement.

ACOLITE needs raw Level-1 packages (.SAFE / .tar), which must be downloaded from
Copernicus Data Space (https://dataspace.copernicus.eu — free account) or USGS
(https://ers.cr.usgs.gov — free account, then generate an Application Token).
GEE only provides processed band values, so it cannot feed ACOLITE.

Accounts
--------
Sentinel-2 : https://dataspace.copernicus.eu  (free)
Landsat     : https://ers.cr.usgs.gov          (free, generate Application Token
              in your profile under "Access" → "Create Application Token")
ACOLITE     : git clone https://github.com/acolite/acolite
              ACOLITE has no requirements.txt (only a conda environment.yml);
              its dependencies (pyresample, cartopy, gdal, h5py, pygrib,
              scikit-image, zarr, fsspec, aiohttp) are already declared in
              this project's pyproject.toml, so `uv sync` covers them.
              Set ACOLITE_REPO_PATH below to point at the clone.
"""

from __future__ import annotations

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import argparse
import ee
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.common import format_date_window, parse_date_object, parse_date_value
from common.credentials import load_credentials
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
    S2_KD,
    LS_KD,
    S2_DRC_COLUMNS,
    LS_DRC_COLUMNS,
    S2_LYZENGA_COLUMNS,
    LS_LYZENGA_COLUMNS,
    add_lyzenga_columns,
    add_depth_corrected_columns,
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

DEFAULT_ROOT = Path(".")
ENDPOINTS_FILE = "transect_endpoints.csv"
JSON_FILE = "tb_seagrass_transects.json"

# Tampa Bay bounding box (±0.05° buffer around transect extent).
# ACOLITE limit format: [south, west, north, east]
TAMPA_BAY_LIMIT = [27.448, -82.862, 28.050, -82.343]

# Wider than common/sentinel.py's and common/landsat.py's defaults (15 / 30 days):
# Tampa Bay's frequent cloud cover often leaves a narrow window with zero scenes
# under MAX_CLOUD_COVER, so search wider and take whichever scene is closest by date.
TAMPA_S2_WINDOW_DAYS = 60
TAMPA_LS_WINDOW_DAYS = 60

_ABUNDANCE_CODE_RE = re.compile(r"^([\d.]+)")


# ===========================================================================
# Section 1 — Observation data loading
# ===========================================================================

def load_transect_endpoints(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["TRAN_ID"], keep="first")
    return df.set_index("TRAN_ID")


def obs_coordinates(
    tran_id: str, site_m: float, endpoints: pd.DataFrame
) -> tuple[float, float] | tuple[None, None]:
    """Linear interpolation along the transect line → (lon, lat)."""
    if tran_id not in endpoints.index:
        return None, None
    row = endpoints.loc[tran_id]
    length_m = float(row["length_m"])
    if length_m <= 0:
        return None, None
    frac = min(float(site_m) / length_m, 1.0)
    lon = float(row["start_lon"]) + frac * (float(row["end_lon"]) - float(row["start_lon"]))
    lat = float(row["start_lat"]) + frac * (float(row["end_lat"]) - float(row["start_lat"]))
    return lon, lat


def _parse_abundance_code(text: str | None) -> float:
    if not text:
        return np.nan
    m = _ABUNDANCE_CODE_RE.match(str(text).strip())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return np.nan


def load_json_to_frame(json_path: Path, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Flatten the transect JSON into one row per observation point."""
    with open(json_path, encoding="utf-8") as fh:
        records = json.load(fh)

    rows: list[dict] = []
    for record in records:
        tran_id = str(record.get("Transect", "")).strip()
        bay_full = str(record.get("BaySegment", "")).strip()
        bay_seg = endpoints.loc[tran_id, "bay_segment"] if tran_id in endpoints.index else ""

        for obs in record.get("Observation") or []:
            site_m = obs.get("Site")
            if site_m is None:
                continue
            site_m = float(site_m)
            lon, lat = obs_coordinates(tran_id, site_m, endpoints)
            if lon is None:
                continue
            depth_raw = obs.get("Depth")
            depth_m = abs(float(depth_raw)) / 100.0 if depth_raw is not None else np.nan
            rows.append({
                "record_id": record.get("ID"),
                "transect_id": tran_id,
                "bay_segment": bay_seg,
                "bay_segment_full": bay_full,
                "assessment_year": record.get("AssessmentYear"),
                "observation_date": obs.get("ObservationDate", ""),
                "site_m": site_m,
                "longitude": lon,
                "latitude": lat,
                "depth_m": depth_m,
                "species": str(obs.get("Species", "")).strip(),
                "species_abundance": str(obs.get("SpeciesAbundance", "")).strip(),
                "abundance_code": _parse_abundance_code(obs.get("SpeciesAbundance")),
                "blade_length_avg": obs.get("BladeLength_Avg"),
                "blade_length_stddev": obs.get("BladeLength_StdDev"),
                "shoot_density_avg": obs.get("ShootDensity_Avg"),
                "shoot_density_stddev": obs.get("ShootDensity_StdDev"),
                "epiphyte_density": obs.get("EpiphyteDensity"),
                "sediment_type": obs.get("SedimentType"),
                "monitoring_agency": record.get("MonitoringAgency"),
                "dataset": "Tampa Bay seagrass transects",
            })
    return pd.DataFrame(rows)


def _load_observation_dates(json_path: Path) -> list[str]:
    with open(json_path, encoding="utf-8") as fh:
        records = json.load(fh)
    dates: set[str] = set()
    for rec in records:
        for obs in rec.get("Observation") or []:
            d = str(obs.get("ObservationDate", ""))[:10]
            if len(d) == 10 and d[4] == "-":
                dates.add(d)
    return sorted(dates)



# ===========================================================================
# Section 5 — GEE sampling (fallback)
# ===========================================================================

def _select_scene_by_date(images_info: list[dict], obs_date: str) -> tuple[int, dict]:
    obs_dt = parse_date_object(obs_date)
    if obs_dt is None or not images_info:
        return 0, images_info[0]
    best: tuple[int, int, dict] | None = None
    for idx, info in enumerate(images_info):
        scene_dt = parse_date_object(info.get("date", ""))
        if scene_dt is None:
            continue
        if scene_dt.date() == obs_dt.date():
            return idx, info
        dist = abs((scene_dt - obs_dt).days)
        if best is None or dist < best[0]:
            best = (dist, idx, info)
    return (best[1], best[2]) if best else (0, images_info[0])


def _sample_gee_point(
    manager,
    longitude: float,
    latitude: float,
    obs_date: str,
    window_days: int,
    download_bands: list[str],
    scale_meters: int,
    buffer_meters: int,
    build_features_fn,
) -> dict | None:
    date_start, date_end = format_date_window(obs_date, window_days)
    if not date_start or not date_end:
        return None
    # Wide window (60 days) can hold more than a handful of scenes; ask for enough
    # candidates that the closest-by-date one isn't cut off by the cloud-cover sort.
    images_info = manager.retrieve_images(latitude, longitude, date_start, date_end, 24)
    if not images_info:
        return None
    idx, scene_info = _select_scene_by_date(images_info, obs_date)
    image = manager.get_image_by_index(latitude, longitude, date_start, date_end, idx)
    if image is None:
        return None
    region = ee.Geometry.Point([longitude, latitude]).buffer(buffer_meters).bounds()
    try:
        stats = image.select(download_bands).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region,
            scale=scale_meters, bestEffort=True, maxPixels=1_000_000,
        ).getInfo()
    except Exception:
        return None
    if not stats:
        return None
    scene_dt = parse_date_object(scene_info.get("date", ""))
    return build_features_fn(stats, scene_dt.strftime("%Y%m%d") if scene_dt else "")


# ===========================================================================
# Section 6 — CSV builder
# ===========================================================================

def _init_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for col in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
        df[col] = np.nan
    df["s2_scene_date"] = ""
    df["s2_source"] = ""
    for col in S2_DRC_COLUMNS + S2_LYZENGA_COLUMNS:
        df[col] = np.nan
    for col in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
        df[col] = np.nan
    df["ls_scene_date"] = ""
    df["ls_source"] = ""
    for col in LS_DRC_COLUMNS + LS_LYZENGA_COLUMNS:
        df[col] = np.nan
    return df


def _compute_corrections(band_vals: dict, depth_m: float, kd_map, drc_cols,
                          lyzenga_pairs, lyzenga_cols) -> dict:
    """Pure version of the old frame-mutating _apply_corrections — safe to call from worker threads."""
    band_vals = {c: float(v) for c, v in band_vals.items()}
    out: dict = {}
    if not np.isnan(depth_m) and depth_m > 0:
        drc = add_depth_corrected_columns(dict(band_vals), depth_m, kd_map)
        out.update({col: drc.get(col, np.nan) for col in drc_cols})
    lyz = add_lyzenga_columns(dict(band_vals), lyzenga_pairs)
    out.update({col: lyz.get(col, np.nan) for col in lyzenga_cols})
    return out


def _with_retries(fn, attempts: int = 3, base_delay: float = 1.5):
    """Retry fn() a few times with backoff — cheap even on a genuine no-data result,
    but recovers from transient GEE rate-limit/network hiccups under parallel load."""
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


def _process_row(
    row_index: int,
    lon: float,
    lat: float,
    obs_date: str,
    depth_m: float,
    s2_mgr: "Sentinel2Manager | None",
    ls_mgr: "LandsatManager | None",
    acolite_scenes: list[dict],
    sentinel2_window_days: int,
) -> dict:
    """Compute every S2/Landsat column for one observation. Touches no shared
    state (no DataFrame access) — safe to run concurrently in a thread pool."""
    result: dict = {"row_index": row_index, "s2_hit": False, "ls_hit": False}

    # Sentinel-2
    s2_feat: dict | None = None
    s2_date = ""
    s2_src = ""

    s2_scene = select_acolite_scene(acolite_scenes, obs_date, "S2", sentinel2_window_days)
    if s2_scene:
        sampled = sample_acolite_nc(s2_scene["path"], lon, lat, ACOLITE_S2_BAND_MAP)
        if sampled:
            s2_feat = sampled
            compute_sentinel2_indices(s2_feat)
            s2_date = parse_date_value(s2_scene["date"])
            s2_src = "acolite"

    if not s2_feat and s2_mgr:
        s2_feat = _with_retries(lambda: _sample_gee_point(
            s2_mgr, lon, lat, obs_date, sentinel2_window_days, SENTINEL2_DOWNLOAD_BANDS,
            10, 15, build_sentinel2_feature_values,
        ))
        if s2_feat:
            s2_date = s2_feat.get("scene_date", "")
            s2_src = "gee"

    if s2_feat:
        for col in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
            result[col] = s2_feat.get(col, np.nan)
        result["s2_scene_date"] = s2_date
        result["s2_source"] = s2_src
        band_vals = {c: s2_feat.get(c, np.nan) for c in SENTINEL2_BAND_COLUMNS}
        result.update(_compute_corrections(band_vals, depth_m, S2_KD, S2_DRC_COLUMNS,
                                            S2_LYZENGA_PAIRS, S2_LYZENGA_COLUMNS))
        result["s2_hit"] = True

    # Landsat
    ls_feat: dict | None = None
    ls_date = ""
    ls_src = ""

    ls_scene = select_acolite_scene(acolite_scenes, obs_date, "LS", TAMPA_LS_WINDOW_DAYS)
    if ls_scene:
        sampled = sample_acolite_nc(ls_scene["path"], lon, lat, ACOLITE_LS_BAND_MAP)
        if sampled:
            ls_feat = sampled
            compute_landsat_indices(ls_feat)
            ls_date = parse_date_value(ls_scene["date"])
            ls_src = "acolite"

    if not ls_feat and ls_mgr:
        ls_feat = _with_retries(lambda: _sample_gee_point(
            ls_mgr, lon, lat, obs_date, TAMPA_LS_WINDOW_DAYS, LANDSAT_DOWNLOAD_BANDS,
            30, 45, build_landsat_feature_values,
        ))
        if ls_feat:
            ls_date = ls_feat.get("ls_scene_date", "")
            ls_src = "gee"

    if ls_feat:
        for col in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
            result[col] = ls_feat.get(col, np.nan)
        result["ls_scene_date"] = ls_date
        result["ls_source"] = ls_src
        band_vals = {c: ls_feat.get(c, np.nan) for c in LANDSAT_BAND_COLUMNS}
        result.update(_compute_corrections(band_vals, depth_m, LS_KD, LS_DRC_COLUMNS,
                                            LS_LYZENGA_PAIRS, LS_LYZENGA_COLUMNS))
        result["ls_hit"] = True

    return result


CHECKPOINT_EVERY = 100


def _sensor_output_path(output_file: Path, sensor: str) -> Path:
    """tampa_transects_with_bands.csv -> tampa_transects_<sensor>_with_bands.csv"""
    stem = output_file.stem
    suffix = output_file.suffix or ".csv"
    if stem.endswith("_with_bands"):
        stem = f"{stem[: -len('_with_bands')]}_{sensor}_with_bands"
    else:
        stem = f"{stem}_{sensor}"
    return output_file.with_name(stem + suffix)


def build_transect_csv(
    root_dir: Path,
    output_file: Path,
    gee_project: str | None = None,
    acolite_dir: Path | None = None,
    sentinel2_window_days: int = TAMPA_S2_WINDOW_DAYS,
    max_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Tampa Bay transect dataset.

    Samples GEE/ACOLITE for `max_workers` rows concurrently (thread pool — the
    work is network-bound, not CPU-bound), and checkpoints progress to
    <output>.checkpoint.csv every CHECKPOINT_EVERY completed rows. If this run
    is interrupted, rerunning the exact same command resumes from there instead
    of starting over.

    Writes two CSVs (Sentinel-2 and Landsat, each with the shared observation
    columns) next to `output_file`, in the same one-CSV-per-sensor style as
    seagrass/indonesia.py.  Returns (sentinel2_frame, landsat_frame).
    """
    for required in (root_dir / ENDPOINTS_FILE, root_dir / JSON_FILE):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    endpoints = load_transect_endpoints(root_dir / ENDPOINTS_FILE)
    print(f"Loaded {len(endpoints)} transect endpoints")
    frame = load_json_to_frame(root_dir / JSON_FILE, endpoints)
    print(f"Loaded {len(frame)} observations from JSON")
    frame = _init_output_columns(frame)
    frame["_done"] = False

    checkpoint_path = output_file.with_name(f"{output_file.stem}.checkpoint.csv")
    if checkpoint_path.exists():
        try:
            checkpoint = pd.read_csv(checkpoint_path)
            same_data = (
                len(checkpoint) == len(frame)
                and "record_id" in checkpoint.columns
                and checkpoint["record_id"].astype(str).equals(frame["record_id"].astype(str))
            )
        except Exception:
            checkpoint, same_data = None, False
        if same_data:
            frame = checkpoint
            frame["_done"] = frame["_done"].fillna(False).astype(bool)
            print(f"Resuming from checkpoint: {int(frame['_done'].sum())}/{len(frame)} rows already done")
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

    s2_mgr: Sentinel2Manager | None = None
    ls_mgr: LandsatManager | None = None
    if gee_project:
        s2_mgr = Sentinel2Manager(gee_project=gee_project)
        ls_mgr = LandsatManager(gee_project=gee_project)

    pending: list[tuple[int, float, float, str, float]] = []
    for row_index, row in frame.iterrows():
        if bool(row["_done"]):
            continue
        obs_date = str(row["observation_date"]).strip()
        if not obs_date or obs_date == "nan":
            frame.at[row_index, "_done"] = True
            continue
        depth_m = float(row["depth_m"]) if pd.notna(row["depth_m"]) else np.nan
        pending.append((row_index, float(row["longitude"]), float(row["latitude"]), obs_date, depth_m))

    s2_hits = int(frame["s2_source"].isin(["acolite", "gee"]).sum())
    ls_hits = int(frame["ls_source"].isin(["acolite", "gee"]).sum())

    def _save_checkpoint() -> None:
        tmp = checkpoint_path.with_suffix(".tmp")
        frame.to_csv(tmp, index=False)
        tmp.rename(checkpoint_path)

    pbar = tqdm(total=len(frame), initial=len(frame) - len(pending),
                desc="Sampling imagery", unit="obs")
    pbar.set_postfix(s2=s2_hits, ls=ls_hits)

    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_row, row_index, lon, lat, obs_date, depth_m,
                             s2_mgr, ls_mgr, acolite_scenes, sentinel2_window_days): row_index
            for row_index, lon, lat, obs_date, depth_m in pending
        }
        for future in as_completed(futures):
            row_index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\nRow {row_index} failed: {exc}")
                result = {"row_index": row_index, "s2_hit": False, "ls_hit": False}

            for col, val in result.items():
                if col in ("row_index", "s2_hit", "ls_hit"):
                    continue
                frame.at[row_index, col] = val
            frame.at[row_index, "_done"] = True
            s2_hits += int(result.get("s2_hit", False))
            ls_hits += int(result.get("ls_hit", False))

            pbar.update(1)
            pbar.set_postfix(s2=s2_hits, ls=ls_hits)

            completed_since_checkpoint += 1
            if completed_since_checkpoint >= CHECKPOINT_EVERY:
                _save_checkpoint()
                completed_since_checkpoint = 0

    pbar.close()
    if pending:
        _save_checkpoint()

    s2_only_cols = (SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS)
                    + ["s2_scene_date", "s2_source"] + S2_DRC_COLUMNS + S2_LYZENGA_COLUMNS)
    ls_only_cols = (LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS
                    + ["ls_scene_date", "ls_source"] + LS_DRC_COLUMNS + LS_LYZENGA_COLUMNS)
    shared_cols = [c for c in frame.columns if c not in s2_only_cols + ls_only_cols + ["_done"]]

    s2_frame = frame[shared_cols + s2_only_cols]
    ls_frame = frame[shared_cols + ls_only_cols]

    s2_out = _sensor_output_path(output_file, "sentinel2")
    ls_out = _sensor_output_path(output_file, "landsat")
    s2_frame.to_csv(s2_out, index=False)
    print(f"Wrote {s2_out}")
    ls_frame.to_csv(ls_out, index=False)
    print(f"Wrote {ls_out}")

    checkpoint_path.unlink(missing_ok=True)
    return s2_frame, ls_frame


# ===========================================================================
# CLI
# ===========================================================================

def _cmd_dates(args) -> None:
    dates = _load_observation_dates(Path(args.root) / JSON_FILE)
    s2_dates = [d for d in dates if d >= "2015-07-01"]
    print(f"Total unique observation dates : {len(dates)}")
    print(f"  Landsat era (all)             : {len(dates)}")
    print(f"  Sentinel-2 era (≥ 2015-07-01): {len(s2_dates)}")
    print()
    print(f"Tampa Bay bounding box (S, W, N, E): {TAMPA_BAY_LIMIT}")
    print(f"Sentinel-2 tile: T17RNH (and partial T17RMH)")
    print(f"Landsat path/row: 17/41")
    print()
    s2_wins = merge_date_windows(s2_dates, TAMPA_S2_WINDOW_DAYS)
    ls_wins = merge_date_windows(dates, TAMPA_LS_WINDOW_DAYS)
    print(f"S2 merged search windows (±{TAMPA_S2_WINDOW_DAYS} days): {len(s2_wins)}")
    print(f"LS merged search windows (±{TAMPA_LS_WINDOW_DAYS} days): {len(ls_wins)}")


def _cmd_download(args) -> None:
    dates = _load_observation_dates(Path(args.root) / JSON_FILE)
    scenes_dir = Path(args.scenes)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    if args.cdse_user and args.cdse_pass:
        download_sentinel2(dates, scenes_dir, args.cdse_user, args.cdse_pass,
                            limit=TAMPA_BAY_LIMIT, window_days=TAMPA_S2_WINDOW_DAYS)
    else:
        print("Skipping Sentinel-2 (no --cdse-user / --cdse-pass)")
    if args.usgs_user and args.usgs_token:
        download_landsat(dates, scenes_dir, args.usgs_user, args.usgs_token,
                          limit=TAMPA_BAY_LIMIT, window_days=TAMPA_LS_WINDOW_DAYS)
    else:
        print("Skipping Landsat (no --usgs-user / --usgs-token)")


def _cmd_acolite(args) -> None:
    run_acolite_batch(
        scenes_dir=Path(args.scenes),
        output_dir=Path(args.acolite_out),
        limit=TAMPA_BAY_LIMIT,
        acolite_path=Path(args.acolite_repo),
    )


def _cmd_build(args) -> None:
    build_transect_csv(
        root_dir=Path(args.root),
        output_file=Path(args.output),
        gee_project=args.gee_project,
        acolite_dir=Path(args.acolite_dir) if args.acolite_dir else None,
        sentinel2_window_days=args.s2_window_days,
        max_workers=args.workers,
    )


def _cmd_all(args) -> None:
    scenes_dir = Path(args.scenes) if args.scenes else None
    acolite_out = Path(args.acolite_out) if args.acolite_out else None

    if scenes_dir and (args.cdse_user or args.usgs_user):
        _cmd_download(args)

    if scenes_dir and acolite_out:
        run_acolite_batch(
            scenes_dir=scenes_dir,
            output_dir=acolite_out,
            limit=TAMPA_BAY_LIMIT,
            acolite_path=Path(args.acolite_repo),
        )

    build_transect_csv(
        root_dir=Path(args.root),
        output_file=Path(args.output),
        gee_project=args.gee_project,
        acolite_dir=acolite_out,
        sentinel2_window_days=args.s2_window_days,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tampa Bay seagrass transect pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'python seagrass/tampa_bay.py <command> --help' for per-command options.",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    # dates
    p = sub.add_parser("dates", help="Show observation date summary and bounding box")
    p.add_argument("--root", default="data")

    # download
    p = sub.add_parser("download", help="Download Level-1 scenes from CDSE / USGS")
    p.add_argument("--root", default="data")
    p.add_argument("--scenes", required=True, help="Directory to save downloaded scenes")
    p.add_argument("--cdse-user", metavar="EMAIL")
    p.add_argument("--cdse-pass", metavar="PASSWORD")
    p.add_argument("--usgs-user", metavar="USERNAME")
    p.add_argument("--usgs-token", metavar="TOKEN")

    # acolite
    p = sub.add_parser("acolite", help="Run ACOLITE on downloaded scenes")
    p.add_argument("--scenes", required=True)
    p.add_argument("--acolite-out", required=True, help="Output directory for NetCDF files")
    p.add_argument("--acolite-repo", default=str(ACOLITE_REPO_PATH))

    # build
    p = sub.add_parser("build", help="Build the CSV from JSON + imagery bands")
    p.add_argument("--root", default="data")
    p.add_argument("--output", "-o", default="tampa_transects_with_bands.csv",
                   help="Base filename; writes <base>_sentinel2<suffix> and <base>_landsat<suffix>")
    p.add_argument("--gee-project", metavar="PROJECT",
                   help="Defaults to gee_project in common/credentials.json if omitted")
    p.add_argument("--acolite-dir", metavar="DIR",
                   help="Directory of ACOLITE NetCDF output (primary imagery source)")
    p.add_argument("--s2-window-days", type=int, default=TAMPA_S2_WINDOW_DAYS)
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent GEE sampling threads (default 8)")

    # all
    p = sub.add_parser("all", help="Run the full pipeline in one go")
    p.add_argument("--root", default="data")
    p.add_argument("--scenes", metavar="DIR", help="Directory for downloaded scenes")
    p.add_argument("--acolite-out", metavar="DIR", help="Directory for ACOLITE output")
    p.add_argument("--acolite-repo", default=str(ACOLITE_REPO_PATH))
    p.add_argument("--cdse-user", metavar="EMAIL")
    p.add_argument("--cdse-pass", metavar="PASSWORD")
    p.add_argument("--usgs-user", metavar="USERNAME")
    p.add_argument("--usgs-token", metavar="TOKEN")
    p.add_argument("--gee-project", metavar="PROJECT",
                   help="Defaults to gee_project in common/credentials.json if omitted")
    p.add_argument("--output", "-o", default="tampa_transects_with_bands.csv",
                   help="Base filename; writes <base>_sentinel2<suffix> and <base>_landsat<suffix>")
    p.add_argument("--s2-window-days", type=int, default=TAMPA_S2_WINDOW_DAYS)
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent GEE sampling threads (default 8)")

    args = parser.parse_args()
    dispatch = {
        "dates": _cmd_dates,
        "download": _cmd_download,
        "acolite": _cmd_acolite,
        "build": _cmd_build,
        "all": _cmd_all,
    }
    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()
