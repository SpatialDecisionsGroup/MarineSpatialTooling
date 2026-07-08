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

  # 4. Build the CSV (uses ACOLITE output; falls back to GEE if scenes are absent)
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
              pip install -r acolite/requirements.txt
              Set ACOLITE_REPO_PATH below.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import argparse
import ee
import numpy as np
import pandas as pd

from common.common import format_date_window, parse_date_object, parse_date_value
from common.sentinel import (
    SENTINEL2_BAND_COLUMNS,
    SENTINEL2_DOWNLOAD_BANDS,
    SENTINEL2_INDEX_COLUMNS,
    SENTINEL2_WINDOW_DAYS,
    build_sentinel2_feature_values,
    compute_sentinel2_indices,
)
from common.landsat import (
    LANDSAT_BAND_COLUMNS,
    LANDSAT_DOWNLOAD_BANDS,
    LANDSAT_INDEX_COLUMNS,
    LANDSAT_WINDOW_DAYS,
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


# --------------------------------------------------------------------------- #
# >>>  SET THIS to the absolute path of your cloned acolite repository.       #
ACOLITE_REPO_PATH = Path.home() / "Documents" / "acolite"
# --------------------------------------------------------------------------- #

DEFAULT_ROOT = Path(".")
ENDPOINTS_FILE = "transect_endpoints.csv"
JSON_FILE = "tb_seagrass_transects.json"

# ACOLITE rhos variable → CSV column name
ACOLITE_S2_BAND_MAP: dict[str, str] = {
    "rhos_443": "s2_b1", "rhos_492": "s2_b2", "rhos_560": "s2_b3",
    "rhos_665": "s2_b4", "rhos_704": "s2_b5", "rhos_740": "s2_b6",
    "rhos_783": "s2_b7", "rhos_842": "s2_b8", "rhos_865": "s2_b8a",
    "rhos_1614": "s2_b11", "rhos_2202": "s2_b12",
}
ACOLITE_LS_BAND_MAP: dict[str, str] = {
    "rhos_443": "ls_b1", "rhos_482": "ls_b2", "rhos_561": "ls_b3",
    "rhos_655": "ls_b4", "rhos_865": "ls_b5", "rhos_1609": "ls_b6",
    "rhos_2201": "ls_b7",
}

# Tampa Bay bounding box (±0.05° buffer around transect extent).
# ACOLITE limit format: [south, west, north, east]
TAMPA_BAY_LIMIT = [27.448, -82.862, 28.050, -82.343]

_ABUNDANCE_CODE_RE = re.compile(r"^([\d.]+)")
_DATE_RE = re.compile(r"(\d{4})[_\-](\d{2})[_\-](\d{2})")


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


def _merge_date_windows(dates: list[str], window_days: int) -> list[tuple[str, str]]:
    """Collapse observation dates into merged search windows (reduces API calls)."""
    parsed = sorted({date.fromisoformat(d) for d in dates})
    if not parsed:
        return []
    delta = timedelta(days=window_days)
    windows: list[tuple[date, date]] = []
    cur_s, cur_e = parsed[0] - delta, parsed[0] + delta
    for d in parsed[1:]:
        ns, ne = d - delta, d + delta
        if ns <= cur_e:
            cur_e = max(cur_e, ne)
        else:
            windows.append((cur_s, cur_e))
            cur_s, cur_e = ns, ne
    windows.append((cur_s, cur_e))
    return [(s.isoformat(), e.isoformat()) for s, e in windows]


# ===========================================================================
# Section 2 — Scene downloads
# ===========================================================================

def _download_file(session, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r    {dest.name}: {100*downloaded/total:.0f}%",
                          end="", flush=True)
    tmp.rename(dest)
    print(f"\r    {dest.name}: done          ")


def _cdse_token(username: str, password: str) -> str:
    import requests
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        "/protocol/openid-connect/token",
        data={"grant_type": "password", "username": username,
              "password": password, "client_id": "cdse-public"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_sentinel2(
    dates: list[str],
    scenes_dir: Path,
    username: str,
    password: str,
    limit: list[float] = TAMPA_BAY_LIMIT,
) -> int:
    import requests

    print("\n=== Sentinel-2 download (Copernicus Data Space) ===")
    token = _cdse_token(username, password)
    s2_dates = [d for d in dates if d >= "2015-07-01"]
    windows = _merge_date_windows(s2_dates, SENTINEL2_WINDOW_DAYS)
    print(f"Searching {len(windows)} merged date window(s) …")

    south, west, north, east = limit
    wkt = (f"POLYGON(({west} {south},{east} {south},"
           f"{east} {north},{west} {north},{west} {south}))")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    seen: set[str] = set()
    n = 0
    for date_start, date_end in windows:
        params = {
            "$filter": (
                "Collection/Name eq 'SENTINEL-2' "
                "and Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'productType' "
                "and att/OData.CSC.StringAttribute/Value eq 'S2MSI1C') "
                f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') "
                f"and ContentDate/Start ge {date_start}T00:00:00.000Z "
                f"and ContentDate/Start le {date_end}T23:59:59.000Z"
            ),
            "$orderby": "ContentDate/Start",
            "$top": 100,
        }
        resp = session.get(
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
            params=params, timeout=60,
        )
        resp.raise_for_status()
        for product in resp.json().get("value", []):
            pid = product["Id"]
            if pid in seen:
                continue
            seen.add(pid)
            dest = scenes_dir / f"{product['Name']}.zip"
            print(f"  {product['Name']}")
            # Refresh token per download (they expire after 10 min)
            session.headers["Authorization"] = f"Bearer {_cdse_token(username, password)}"
            _download_file(
                session,
                f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({pid})/$value",
                dest,
            )
            n += 1
            time.sleep(0.5)

    print(f"Sentinel-2: {n} scene(s) → {scenes_dir}")
    return n


_M2M = "https://m2m.cr.usgs.gov/api/api/json/stable"
_LS_DATASETS = [
    "landsat_ot_c2_l1",    # Landsat 8/9
    "landsat_etm_c2_l1",   # Landsat 7
    "landsat_tm_c2_l1",    # Landsat 4/5
]


def _m2m(session, endpoint: str, payload: dict) -> dict:
    resp = session.post(f"{_M2M}/{endpoint}", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errorCode"):
        raise RuntimeError(f"M2M {data['errorCode']}: {data.get('errorMessage')}")
    return data.get("data") or {}


def download_landsat(
    dates: list[str],
    scenes_dir: Path,
    username: str,
    token: str,
    limit: list[float] = TAMPA_BAY_LIMIT,
) -> int:
    import requests

    print("\n=== Landsat download (USGS M2M) ===")
    session = requests.Session()
    api_key = _m2m(session, "login-token", {"username": username, "token": token})
    session.headers["X-Auth-Token"] = api_key
    print("  Logged in to USGS M2M")

    south, west, north, east = limit
    windows = _merge_date_windows(dates, LANDSAT_WINDOW_DAYS)
    print(f"Searching {len(windows)} merged date window(s) …")

    to_dl: list[dict] = []
    seen: set[str] = set()
    for ds_name in _LS_DATASETS:
        for date_start, date_end in windows:
            result = _m2m(session, "scene-search", {
                "datasetName": ds_name,
                "spatialFilter": {
                    "filterType": "mbr",
                    "lowerLeft": {"latitude": south, "longitude": west},
                    "upperRight": {"latitude": north, "longitude": east},
                },
                "acquisitionFilter": {"start": date_start, "end": date_end},
                "maxResults": 50,
                "useCustomization": False,
            })
            for scene in result.get("results") or []:
                eid = scene["entityId"]
                if eid in seen:
                    continue
                seen.add(eid)
                to_dl.append({"entityId": eid, "dataset": ds_name,
                               "displayId": scene.get("displayId", eid)})

    print(f"  Found {len(to_dl)} unique scene(s)")
    n = 0
    for i in range(0, len(to_dl), 20):
        batch = to_dl[i: i + 20]
        opts = _m2m(session, "download-options", {
            "datasetName": batch[0]["dataset"],
            "entityIds": [s["entityId"] for s in batch],
        })
        downloads = [
            {"entityId": p["entityId"], "productId": p["id"]}
            for p in (opts or [])
            if p.get("downloadSystem") == "dds" and "Bundle" in p.get("productName", "")
        ]
        if not downloads:
            continue
        url_result = _m2m(session, "download-request", {"downloads": downloads})
        for item in url_result.get("availableDownloads") or []:
            url = item.get("url")
            did = item.get("displayId", item.get("entityId", "scene"))
            dest = scenes_dir / f"{did}.tar"
            if dest.exists():
                print(f"    already downloaded: {did}")
                continue
            print(f"  {did}")
            _download_file(session, url, dest)
            n += 1
            time.sleep(0.5)

    _m2m(session, "logout", {})
    print(f"Landsat: {n} scene(s) → {scenes_dir}")
    return n


# ===========================================================================
# Section 3 — ACOLITE runner
# ===========================================================================

def _find_scenes(scenes_dir: Path) -> list[Path]:
    found: list[Path] = []
    for entry in sorted(scenes_dir.iterdir()):
        name = entry.name.upper()
        if entry.is_file() and entry.suffix.lower() in {".zip", ".tar", ".gz"}:
            found.append(entry)
        elif entry.is_dir() and (name.endswith(".SAFE") or name.startswith("LC0")):
            found.append(entry)
    return found


def run_acolite_batch(
    scenes_dir: Path,
    output_dir: Path,
    acolite_path: Path = ACOLITE_REPO_PATH,
    limit: list[float] = TAMPA_BAY_LIMIT,
) -> None:
    """Run ACOLITE on every scene in scenes_dir, writing NetCDF to output_dir."""
    if not acolite_path.exists():
        raise FileNotFoundError(
            f"ACOLITE repo not found at {acolite_path}.\n"
            "Clone it:  git clone https://github.com/acolite/acolite\n"
            f"Then set ACOLITE_REPO_PATH at the top of {__file__}."
        )
    if str(acolite_path) not in sys.path:
        sys.path.insert(0, str(acolite_path))
    try:
        import acolite as ac
    except ImportError as exc:
        raise ImportError(
            f"Could not import acolite from {acolite_path}: {exc}\n"
            "Install its dependencies:  pip install -r acolite/requirements.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = _find_scenes(scenes_dir)
    if not scenes:
        print(f"No scenes found in {scenes_dir}")
        return

    print(f"\n=== ACOLITE processing: {len(scenes)} scene(s) ===")
    settings_base: dict = {
        "limit": limit,
        "l2w_parameters": ["rhos"],   # write all rhos_* surface-reflectance bands
        "s2_target_res": 10,          # resample all S2 bands to 10 m
        "glint_correction": True,     # sun glint correction
        "dsf_aot_estimate": "tiled",  # dark-spectrum fitting, tile-based
        "merge_tiles": True,          # merge adjacent S2 tiles
    }
    for scene_path in scenes:
        print(f"\nProcessing {scene_path.name} …")
        try:
            ac.acolite_run(settings={
                **settings_base,
                "inputfile": str(scene_path),
                "output": str(output_dir),
            })
            print(f"  OK → {output_dir}")
        except Exception as exc:
            print(f"  FAILED: {exc}")


# ===========================================================================
# Section 4 — ACOLITE scene sampling
# ===========================================================================

def _detect_acolite_sensor(nc_path: Path) -> str | None:
    name = nc_path.stem.upper()
    if any(t in name for t in ("S2A", "S2B", "MSI", "SENTINEL2", "SENTINEL-2")):
        return "S2"
    if any(t in name for t in ("L8", "L9", "LC08", "LC09", "LANDSAT", "OLI")):
        return "LS"
    return None


def scan_acolite_output(acolite_dir: Path) -> list[dict]:
    scenes: list[dict] = []
    for nc_file in sorted(acolite_dir.glob("**/*.nc")):
        m = _DATE_RE.search(nc_file.name)
        if not m:
            continue
        sensor = _detect_acolite_sensor(nc_file)
        if sensor is None:
            continue
        scenes.append({
            "path": nc_file,
            "date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
            "sensor": sensor,
        })
    return scenes


def _select_acolite_scene(
    scenes: list[dict], obs_date: str, sensor: str, max_days: int
) -> dict | None:
    obs_dt = parse_date_object(obs_date)
    if obs_dt is None:
        return None
    best: tuple[int, dict] | None = None
    for scene in scenes:
        if scene["sensor"] != sensor:
            continue
        scene_dt = parse_date_object(scene["date"])
        if scene_dt is None:
            continue
        dist = abs((scene_dt - obs_dt).days)
        if dist <= max_days and (best is None or dist < best[0]):
            best = (dist, scene)
    return best[1] if best else None


def _sample_acolite_nc(
    nc_path: Path, lon: float, lat: float, band_map: dict[str, str]
) -> dict[str, float]:
    try:
        import xarray as xr
    except ImportError:
        print("Warning: xarray not installed; ACOLITE sampling unavailable.")
        return {}
    try:
        ds = xr.open_dataset(nc_path, mask_and_scale=True)
    except Exception as exc:
        print(f"Warning: could not open {nc_path.name}: {exc}")
        return {}

    lat_dim = next((d for d in ("lat", "y") if d in ds.coords), None)
    lon_dim = next((d for d in ("lon", "x") if d in ds.coords), None)
    result: dict[str, float] = {}
    try:
        for rhos_var, csv_col in band_map.items():
            if rhos_var not in ds:
                result[csv_col] = np.nan
                continue
            da = ds[rhos_var]
            if lat_dim and lon_dim:
                try:
                    val = float(da.sel({lat_dim: lat, lon_dim: lon},
                                       method="nearest").values)
                except Exception:
                    val = np.nan
            else:
                val = np.nan
            result[csv_col] = val if not np.isnan(val) else np.nan
    except Exception as exc:
        print(f"Warning: sampling error in {nc_path.name}: {exc}")
    finally:
        ds.close()
    return result


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
    images_info = manager.retrieve_images(latitude, longitude, date_start, date_end, 8)
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


def _apply_corrections(frame, row_index, band_cols, depth_m, kd_map, drc_cols,
                        lyzenga_pairs, lyzenga_cols) -> None:
    band_vals = {c: float(frame.at[row_index, c]) for c in band_cols}
    if not np.isnan(depth_m) and depth_m > 0:
        drc = add_depth_corrected_columns(dict(band_vals), depth_m, kd_map)
        for col in drc_cols:
            frame.at[row_index, col] = drc.get(col, np.nan)
    lyz = add_lyzenga_columns(dict(band_vals), lyzenga_pairs)
    for col in lyzenga_cols:
        frame.at[row_index, col] = lyz.get(col, np.nan)


def build_transect_csv(
    root_dir: Path,
    output_file: Path,
    gee_project: str | None = None,
    acolite_dir: Path | None = None,
    sentinel2_window_days: int = SENTINEL2_WINDOW_DAYS,
) -> pd.DataFrame:
    for required in (root_dir / ENDPOINTS_FILE, root_dir / JSON_FILE):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    endpoints = load_transect_endpoints(root_dir / ENDPOINTS_FILE)
    print(f"Loaded {len(endpoints)} transect endpoints")
    frame = load_json_to_frame(root_dir / JSON_FILE, endpoints)
    print(f"Loaded {len(frame)} observations from JSON")
    frame = _init_output_columns(frame)

    acolite_scenes: list[dict] = []
    if acolite_dir is not None and acolite_dir.exists():
        acolite_scenes = scan_acolite_output(acolite_dir)
        print(f"Indexed {len(acolite_scenes)} ACOLITE scenes in {acolite_dir}")

    s2_mgr: Sentinel2Manager | None = None
    ls_mgr: LandsatManager | None = None
    if gee_project:
        s2_mgr = Sentinel2Manager(gee_project=gee_project)
        ls_mgr = LandsatManager(gee_project=gee_project)

    for row_index, row in frame.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        obs_date = str(row["observation_date"]).strip()
        depth_m = float(row["depth_m"]) if pd.notna(row["depth_m"]) else np.nan
        if not obs_date or obs_date == "nan":
            continue

        # Sentinel-2
        s2_feat: dict | None = None
        s2_date = ""
        s2_src = ""

        s2_scene = _select_acolite_scene(acolite_scenes, obs_date, "S2", sentinel2_window_days)
        if s2_scene:
            sampled = _sample_acolite_nc(s2_scene["path"], lon, lat, ACOLITE_S2_BAND_MAP)
            if sampled:
                s2_feat = sampled
                compute_sentinel2_indices(s2_feat)
                s2_date = parse_date_value(s2_scene["date"])
                s2_src = "acolite"

        if not s2_feat and s2_mgr:
            s2_feat = _sample_gee_point(s2_mgr, lon, lat, obs_date,
                                         sentinel2_window_days, SENTINEL2_DOWNLOAD_BANDS,
                                         10, 15, build_sentinel2_feature_values)
            if s2_feat:
                s2_date = s2_feat.get("scene_date", "")
                s2_src = "gee"

        if s2_feat:
            for col in SENTINEL2_BAND_COLUMNS + list(SENTINEL2_INDEX_COLUMNS):
                frame.at[row_index, col] = s2_feat.get(col, np.nan)
            frame.at[row_index, "s2_scene_date"] = s2_date
            frame.at[row_index, "s2_source"] = s2_src
            _apply_corrections(frame, row_index, SENTINEL2_BAND_COLUMNS, depth_m,
                                S2_KD, S2_DRC_COLUMNS, S2_LYZENGA_PAIRS, S2_LYZENGA_COLUMNS)

        # Landsat
        ls_feat: dict | None = None
        ls_date = ""
        ls_src = ""

        ls_scene = _select_acolite_scene(acolite_scenes, obs_date, "LS", LANDSAT_WINDOW_DAYS)
        if ls_scene:
            sampled = _sample_acolite_nc(ls_scene["path"], lon, lat, ACOLITE_LS_BAND_MAP)
            if sampled:
                ls_feat = sampled
                compute_landsat_indices(ls_feat)
                ls_date = parse_date_value(ls_scene["date"])
                ls_src = "acolite"

        if not ls_feat and ls_mgr:
            ls_feat = _sample_gee_point(ls_mgr, lon, lat, obs_date,
                                         LANDSAT_WINDOW_DAYS, LANDSAT_DOWNLOAD_BANDS,
                                         30, 45, build_landsat_feature_values)
            if ls_feat:
                ls_date = ls_feat.get("ls_scene_date", "")
                ls_src = "gee"

        if ls_feat:
            for col in LANDSAT_BAND_COLUMNS + LANDSAT_INDEX_COLUMNS:
                frame.at[row_index, col] = ls_feat.get(col, np.nan)
            frame.at[row_index, "ls_scene_date"] = ls_date
            frame.at[row_index, "ls_source"] = ls_src
            _apply_corrections(frame, row_index, LANDSAT_BAND_COLUMNS, depth_m,
                                LS_KD, LS_DRC_COLUMNS, LS_LYZENGA_PAIRS, LS_LYZENGA_COLUMNS)

    frame.to_csv(output_file, index=False)
    print(f"Wrote {output_file}")
    return frame


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
    s2_wins = _merge_date_windows(s2_dates, SENTINEL2_WINDOW_DAYS)
    ls_wins = _merge_date_windows(dates, LANDSAT_WINDOW_DAYS)
    print(f"S2 merged search windows (±{SENTINEL2_WINDOW_DAYS} days): {len(s2_wins)}")
    print(f"LS merged search windows (±{LANDSAT_WINDOW_DAYS} days): {len(ls_wins)}")
    if args.output:
        Path(args.output).write_text("\n".join(dates) + "\n", encoding="utf-8")
        print(f"\nDate list saved to {args.output}")
    else:
        print()
        for d in dates:
            print(f"  {d}")
        print("\n(Use --output dates.txt to save to a file)")


def _cmd_download(args) -> None:
    dates = _load_observation_dates(Path(args.root) / JSON_FILE)
    scenes_dir = Path(args.scenes)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    if args.cdse_user and args.cdse_pass:
        download_sentinel2(dates, scenes_dir, args.cdse_user, args.cdse_pass)
    else:
        print("Skipping Sentinel-2 (no --cdse-user / --cdse-pass)")
    if args.usgs_user and args.usgs_token:
        download_landsat(dates, scenes_dir, args.usgs_user, args.usgs_token)
    else:
        print("Skipping Landsat (no --usgs-user / --usgs-token)")


def _cmd_acolite(args) -> None:
    run_acolite_batch(
        scenes_dir=Path(args.scenes),
        output_dir=Path(args.acolite_out),
        acolite_path=Path(args.acolite_repo),
    )


def _cmd_build(args) -> None:
    build_transect_csv(
        root_dir=Path(args.root),
        output_file=Path(args.output),
        gee_project=args.gee_project,
        acolite_dir=Path(args.acolite_dir) if args.acolite_dir else None,
        sentinel2_window_days=args.s2_window_days,
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
            acolite_path=Path(args.acolite_repo),
        )

    build_transect_csv(
        root_dir=Path(args.root),
        output_file=Path(args.output),
        gee_project=args.gee_project,
        acolite_dir=acolite_out,
        sentinel2_window_days=args.s2_window_days,
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
    p.add_argument("--output", metavar="FILE", help="Save date list to file")

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
    p.add_argument("--output", "-o", default="tampa_transects_with_bands.csv")
    p.add_argument("--gee-project", metavar="PROJECT")
    p.add_argument("--acolite-dir", metavar="DIR",
                   help="Directory of ACOLITE NetCDF output (primary imagery source)")
    p.add_argument("--s2-window-days", type=int, default=SENTINEL2_WINDOW_DAYS)

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
    p.add_argument("--gee-project", metavar="PROJECT")
    p.add_argument("--output", "-o", default="tampa_transects_with_bands.csv")
    p.add_argument("--s2-window-days", type=int, default=SENTINEL2_WINDOW_DAYS)

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
