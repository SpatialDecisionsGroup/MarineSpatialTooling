"""Shared ACOLITE + raw-scene-download pipeline for seagrass site scripts.

Site-specific scripts (seagrass/tampa_bay.py, seagrass/indonesia.py) supply their
own bounding box(es), observation dates, and ACOLITE rhos-band-name -> CSV-column
maps; everything else here is site-agnostic:

  - Merging observation dates into merged search windows (fewer API calls)
  - Downloading Level-1 scenes from Copernicus Data Space (Sentinel-2) and
    USGS M2M (Landsat)
  - Running ACOLITE atmospheric correction on downloaded scenes
  - Indexing ACOLITE NetCDF output by date/sensor and point-sampling it

ACOLITE needs raw Level-1 packages (.SAFE / .tar), which must be downloaded from
Copernicus Data Space (https://dataspace.copernicus.eu — free account) or USGS
(https://ers.cr.usgs.gov — free account, then generate an Application Token).
GEE only provides processed band values, so it cannot feed ACOLITE.

ACOLITE itself: git clone https://github.com/acolite/acolite
It has no requirements.txt (only a conda environment.yml); its dependencies
(pyresample, cartopy, gdal, h5py, pygrib, scikit-image, zarr, fsspec, aiohttp)
are declared in this project's pyproject.toml, so `uv sync` covers them.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common.common import parse_date_object

# --------------------------------------------------------------------------- #
# >>>  SET THIS to the absolute path of your cloned acolite repository.       #
ACOLITE_REPO_PATH = Path(__file__).resolve().parent.parent.parent / "acolite"
# --------------------------------------------------------------------------- #

_DATE_RE = re.compile(r"(\d{4})[_\-](\d{2})[_\-](\d{2})")

_M2M = "https://m2m.cr.usgs.gov/api/api/json/stable"
# (dataset name, operational start, operational end or None if ongoing).
# Landsat 5 TM was decommissioned 2013-06-05; querying M2M's scene-search for
# a dataset+date range years past the satellite's retirement has been observed
# to return a 500 rather than an empty result set, so windows outside a
# dataset's real operational range are skipped entirely rather than queried.
_LS_DATASETS: list[tuple[str, date, date | None]] = [
    ("landsat_ot_c2_l1", date(2013, 3, 18), None),               # Landsat 8/9
    ("landsat_etm_c2_l1", date(1999, 4, 15), None),               # Landsat 7
    ("landsat_tm_c2_l1", date(1982, 7, 16), date(2013, 6, 5)),    # Landsat 4/5
]


# ===========================================================================
# Date windows
# ===========================================================================

def merge_date_windows(dates: list[str], window_days: int) -> list[tuple[str, str]]:
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
# Scene downloads
# ===========================================================================

def _download_file(session, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name, leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


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
    limit: list[float],
    window_days: int,
) -> int:
    import requests

    print("\n=== Sentinel-2 download (Copernicus Data Space) ===")
    token = _cdse_token(username, password)
    s2_dates = [d for d in dates if d >= "2015-07-01"]
    windows = merge_date_windows(s2_dates, window_days)
    print(f"Searching {len(windows)} merged date window(s) …")

    south, west, north, east = limit
    wkt = (f"POLYGON(({west} {south},{east} {south},"
           f"{east} {north},{west} {north},{west} {south}))")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    seen: set[str] = set()
    products: list[dict] = []
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
        url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        request_params = params
        while url:
            resp = session.get(url, params=request_params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            for product in data.get("value", []):
                pid = product["Id"]
                if pid in seen:
                    continue
                seen.add(pid)
                products.append(product)
            # $top caps each page at 100; follow nextLink so a window with
            # more matches than that isn't silently truncated.
            url = data.get("@odata.nextLink")
            request_params = None

    to_fetch = [p for p in products if not (scenes_dir / f"{p['Name']}.zip").exists()]
    print(f"Found {len(products)} scene(s); "
          f"{len(products) - len(to_fetch)} already downloaded, {len(to_fetch)} to fetch")

    n = 0
    for product in tqdm(to_fetch, desc="Sentinel-2 scenes", unit="scene"):
        dest = scenes_dir / f"{product['Name']}.zip"
        # Refresh token per download (they expire after 10 min)
        session.headers["Authorization"] = f"Bearer {_cdse_token(username, password)}"
        _download_file(
            session,
            f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product['Id']})/$value",
            dest,
        )
        n += 1
        time.sleep(0.5)

    print(f"Sentinel-2: {n} scene(s) → {scenes_dir}")
    return n


def _m2m(session, endpoint: str, payload: dict, attempts: int = 4, base_delay: float = 3.0) -> dict:
    """POST to the USGS M2M API, retrying transient network/server hiccups
    (the endpoint is prone to read timeouts under load). Doesn't retry 4xx
    errors (bad credentials etc.) or M2M-reported errorCodes — those need a
    fix, not a retry."""
    import requests
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = session.post(f"{_M2M}/{endpoint}", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("errorCode"):
                raise RuntimeError(f"M2M {data['errorCode']}: {data.get('errorMessage')}")
            return data.get("data") or {}
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            last_exc = exc
        if attempt < attempts - 1:
            delay = base_delay * (2 ** attempt)
            print(f"  M2M {endpoint} request failed ({last_exc}); "
                  f"retrying in {delay:.0f}s (attempt {attempt + 2}/{attempts})…")
            time.sleep(delay)
    raise last_exc


def download_landsat(
    dates: list[str],
    scenes_dir: Path,
    username: str,
    token: str,
    limit: list[float],
    window_days: int,
) -> int:
    import requests

    print("\n=== Landsat download (USGS M2M) ===")
    session = requests.Session()
    api_key = _m2m(session, "login-token", {"username": username, "token": token})
    session.headers["X-Auth-Token"] = api_key
    print("  Logged in to USGS M2M")

    south, west, north, east = limit
    windows = merge_date_windows(dates, window_days)
    print(f"Searching {len(windows)} merged date window(s) …")

    to_dl: list[dict] = []
    seen: set[str] = set()
    for ds_name, ds_start, ds_end in _LS_DATASETS:
        for date_start, date_end in windows:
            window_start = date.fromisoformat(date_start)
            window_end = date.fromisoformat(date_end)
            if window_end < ds_start or (ds_end is not None and window_start > ds_end):
                continue
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

    total_found = len(to_dl)
    to_dl = [d for d in to_dl if not (scenes_dir / f"{d['displayId']}.tar").exists()]
    print(f"  Found {total_found} unique scene(s); "
          f"{total_found - len(to_dl)} already downloaded, {len(to_dl)} to fetch")

    n = 0
    with tqdm(total=len(to_dl), desc="Landsat scenes", unit="scene") as bar:
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
                    bar.update(1)
                    continue
                _download_file(session, url, dest)
                n += 1
                bar.update(1)
                time.sleep(0.5)

    _m2m(session, "logout", {})
    print(f"Landsat: {n} scene(s) → {scenes_dir}")
    return n


# ===========================================================================
# ACOLITE runner
# ===========================================================================

def find_scenes(scenes_dir: Path) -> list[Path]:
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
    limit: list[float],
    acolite_path: Path = ACOLITE_REPO_PATH,
) -> None:
    """Run ACOLITE on every scene in scenes_dir, writing NetCDF to output_dir."""
    if not acolite_path.exists():
        raise FileNotFoundError(
            f"ACOLITE repo not found at {acolite_path}.\n"
            "Clone it:  git clone https://github.com/acolite/acolite\n"
            "Then set ACOLITE_REPO_PATH at the top of common/acolite_pipeline.py."
        )
    if str(acolite_path) not in sys.path:
        sys.path.insert(0, str(acolite_path))
    try:
        import acolite as ac
    except ImportError as exc:
        raise ImportError(
            f"Could not import acolite from {acolite_path}: {exc}\n"
            "Its dependencies are declared in this project's pyproject.toml "
            "(pyresample, cartopy, gdal, h5py, pygrib, scikit-image, zarr, "
            "fsspec, aiohttp) — run:  uv sync"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = find_scenes(scenes_dir)
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
            ac.acolite.acolite_run(settings={
                **settings_base,
                "inputfile": str(scene_path),
                "output": str(output_dir),
            })
            print(f"  OK → {output_dir}")
        except Exception as exc:
            print(f"  FAILED: {exc}")


# ===========================================================================
# ACOLITE scene sampling
# ===========================================================================

def detect_acolite_sensor(nc_path: Path) -> str | None:
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
        sensor = detect_acolite_sensor(nc_file)
        if sensor is None:
            continue
        scenes.append({
            "path": nc_file,
            "date": f"{m.group(1)}-{m.group(2)}-{m.group(3)}",
            "sensor": sensor,
        })
    return scenes


def select_acolite_scene(
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


def sample_acolite_nc(
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


# ACOLITE rhos variable → CSV column name (shared across sites; band physics
# don't change per site, only which bands end up used downstream)
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
