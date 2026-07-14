# Marine Super-Resolution Dataset Creator

A modular Python framework for creating and downloading multi-image super-resolution datasets from satellite imagery, stratified by coral, seagrass, and mangrove habitat.

## Table of Contents

- [Dependencies](#dependencies)
- [Workflow](#workflow)
  - [0. Download Habitat Extents](#0-download-habitat-extents)
  - [1. Setup Credentials](#1-setup-credentials)
  - [2. Create Dataset Manifest](#2-create-dataset-manifest)
  - [3. Download and Preprocess Data](#3-download-and-preprocess-data)
  - [4. Postprocess](#4-postprocess)
  - [5. Review Dataset](#5-review-dataset)
  - [6. Tile the Dataset](#6-tile-the-dataset-optional)
  - [7. Check Dataset Integrity](#7-check-dataset-integrity-optional)
  - [8. Analyse Dataset](#8-analyse-dataset-optional)
- [Resuming & Growing the Dataset](#resuming--growing-the-dataset)
  - [Growing the manifest](#growing-the-manifest-create)
  - [Incremental downloads](#incremental-downloads-download)
  - [Dropping bad samples](#dropping-bad-samples-drop_samplespy)
- [Output Structure](#output-structure)
- [Data Specifications](#data-specifications)
- [Seasons](#seasons)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Core Scripts](#core-scripts)
  - [Configuration](#configuration)
  - [Data Management](#data-management)
  - [Satellite Data Modules](#satellite-data-modules)
- [Satellite Architecture](#satellite-architecture)
- [Configuration Files](#configuration-files)
- [Logging](#logging)

## Dependencies

Required packages:

- `geopandas` - GIS vector data
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `rasterio` - Raster data I/O
- `shapely` - Geometric operations
- `ee` - Google Earth Engine Python API
- `planet` - Planet Labs SDK (optional, for PlanetScope download)
- `tqdm` - Progress bars

## Workflow

All scripts are run through `uv run`, which uses the project's `uv.lock`/`pyproject.toml`
environment. Replace `uv run python` with `python` if you manage your own virtualenv.

### 0. Download Habitat Extents

Download the global coral, seagrass, and mangrove polygon datasets used for sampling:

```bash
bash habitat_extents.sh
```

This downloads three open datasets into `data/habitat_extents/`:

| Subdirectory | Dataset | Source |
| --- | --- | --- |
| `coral/` | UNEP-WCMC Global Coral Reefs v4.1 | wcmc.io/WCMC_008 |
| `seagrass/` | UNEP-WCMC Global Seagrasses v7.1 | wcmc.io/WCMC_013_014 |
| `mangrove/` | Global Mangrove Watch v4.0.19 (2020) | Zenodo 12756047 |

All three are CC BY 4.0. Requires `curl`, `unzip`, and `python3`.

### 1. Setup Credentials

```bash
uv run python setup_credentials.py
```

Optional flags: `-o, --output-dir` (default `./common`), `-p, --planet-key`
to pass the Planet API key non-interactively.

The script will:

1. Prompt for your Planet Labs API key (press Enter to skip or keep existing value)
2. Open a browser window for Earth Engine OAuth authentication

This creates:

- `./common/credentials.json` - Stores the Planet API key and GEE project ID
- Earth Engine authentication is cached locally by the `ee` library

If you skip either credential, you can re-run the setup script later to complete it.

### 2. Create Dataset Manifest

```bash
uv run python superres.py create \
  -n 1000 \
  --coastline-dir ./data/gshhg-shp-2.3.7 \
  --gebco-file ./data/gebco_2026_geotiff \
  --habitat-extents-dir ./data/habitat_extents
```

**Arguments:**

- `-o, --output`: Output directory (default: `./superres`)
- `-n, --total-samples` or `-p, --samples-per-area`: total sample count, or samples per
  habitat stratum (one is required)
- `--habitat-extents-dir`: Directory containing coral/seagrass/mangrove polygon subdirectories
  downloaded by `habitat_extents.sh` (default: `./data/habitat_extents`)
- `--coastline-dir`: Path to the GSHHG/WDBII coastline dataset (default `./data/gshhg-shp-2.3.7`)
- `--gebco-file`: Path to the GEBCO bathymetry raster used for depth metadata (default `./data/gebco_2026_geotiff`)
- `--turbidity-file`: Path to a Kd490/turbidity raster used for the bottom-visibility check
  on coral and seagrass sites (default `./data/turbidity.nc`; check is skipped if file is absent)
- `--resume`: continue from an existing `dataset_metadata.json` instead of starting over
  (see [Resuming & Growing the Dataset](#resuming--growing-the-dataset))

Samples are drawn in equal amounts from coral, seagrass, and mangrove habitat polygons.
For coral and seagrass, sites are additionally filtered by a Kd490 bottom-visibility
threshold (`BOTTOM_VISIBILITY_KD490_MAX = 0.3 m⁻¹`) so only optically shallow sites are included.
On resume, any existing samples that predate the habitat-based schema are automatically
deleted and their IDs reused.

The scripts load credentials automatically from `./common/credentials.json`.

**Outputs:**

- `./superres/metadata/dataset_metadata.json` - Complete metadata in JSON format
- `./superres/metadata/dataset_manifest.csv` - Manifest CSV for download script

### 3. Download and Preprocess Data

```bash
uv run python superres.py download superres/metadata/dataset_manifest.csv
```

**Arguments:**

- `manifest_file`: Path to CSV manifest (positional)
- `-o, --output`: Output directory (default: `./superres`)
- `--turbidity-file`: Path to a turbidity raster/NetCDF (default `./data/turbidity.nc`)
- `--resume`: skip samples that already have a `sample_metadata.json` checkpoint

The scripts load credentials automatically from `./common/credentials.json`.

For each manifest row, this creates a Planet Orders API request, clips the source item to
the sampled AOI, reprojects to the local UTM zone, and standardizes the resulting
high-res raster to an exact 512×512 grid alongside the matching low-res images. It
is safe to re-run the same command repeatedly — see
[Resuming & Growing the Dataset](#resuming--growing-the-dataset).

### 4. Postprocess

```bash
uv run python superres.py postprocess superres/data -o ./superres/processed
```

**Arguments:**

- `data_dir`: Path to the downloaded dataset's data directory (positional)
- `-o, --output`: Output directory for postprocessed rasters (default: `<data_dir>/../processed`)
- `--overwrite`: Reprocess samples even if output files already exist

For each completed sample, this:

1. Reprojects every low-res image in the stack onto the high-res grid (at the LR satellite's native resolution), so the full stack is pixel-registered against the high-res target.
2. Rescales all bands from raw sensor digital numbers to surface reflectance using documented scale/offset constants. Classification/QA bands are resampled with nearest-neighbour and left unscaled.

Output mirrors the input `sample_*/` layout under the output directory; the raw originals are not modified.

### 5. Review Dataset

Visually inspect downloaded samples and flag individual images for replacement or deletion:

```bash
uv run streamlit run superres/review_dataset.py -- --data-dir data/landsat2sentinel/data
```

The Streamlit app shows all Landsat images and the Sentinel-2 image for each sample.
Click individual Landsat thumbnails to select them for partial replacement, or use the
action buttons (Keep / Replace LR / Replace HR / Replace Both / Delete) to record a
decision. Decisions are written to `<data-dir>/../review_decisions.json` and survive
page refreshes.

### 6. Tile the Dataset (optional)

```bash
uv run python superres.py tile superres/data -o ./superres/output
```

**Arguments:**

- `data_dir`: Path to the downloaded dataset's data directory, i.e. `superres/data` (positional)
- `-o, --output`: Output directory for tiled patches (default: `./superres/output`)
- `-t, --tile-size`: Tile size in pixels (default: 512)

Writes fixed-size training tiles to `<output>/tiles/<satellite>/`.

### 7. Check Dataset Integrity (optional)

```bash
uv run python superres.py check superres/data --manifest superres/metadata/dataset_manifest.csv
```

**Arguments:**

- `data_dir`: Path to the downloaded dataset's data directory (positional)
- `--manifest`: Path to CSV manifest (for cross-referencing expected samples)
- `--verbose`: Print per-file detail for every issue

Validates every sample for: directory structure, `sample_metadata.json` presence and validity, file counts, raster integrity (openable by rasterio, correct band count), dimensions (LR non-degenerate, HR exactly `patch_size_pixels × patch_size_pixels`), georeferencing (CRS, pixel size, origin), and non-zero centre values.

### 8. Analyse Dataset (optional)

Generate a full set of analysis figures (spatial maps, spectral violins, PSD curves, alignment diagnostics, sample thumbnails):

```bash
uv run python -m superres.analyse_dataset \
    data/landsat2sentinel/metadata/dataset_manifest.csv \
    data/landsat2sentinel/data \
    --processed-dir data/landsat2sentinel/processed \
    --output figures/dataset_analysis
```

**Arguments:**

- `manifest`: Path to `dataset_manifest.csv` (positional)
- `data_dir`: Path to the `data/` directory containing `sample_*/` folders (positional)
- `--processed-dir`: Path to postprocessed rasters (default: `<data_dir>/../processed`)
- `-o, --output`: Output directory for figures (default: `figures/dataset_analysis`)
- `--n-band-samples`: Number of random samples used for spectral/PSD/histogram figures (default: 150)
- `--skip-rasters`: Skip all raster-reading figures (figs 3–4, 8–11); fast summary only
- `--images`: Only generate sample thumbnail grids (fig 11); skip everything else

## Resuming & Growing the Dataset

Both the create and download steps are designed to be re-run incrementally, so a small
test run can be grown into the full dataset without losing existing samples or
re-downloading data you already have.

### Growing the manifest (`create`)

Re-run with `--resume` and a larger `-n`/`-p`:

```bash
uv run python superres.py create -n 2000 --resume \
  --coastline-dir ./data/gshhg-shp-2.3.7 \
  --gebco-file ./data/gebco_2026_geotiff \
  --habitat-extents-dir ./data/habitat_extents
```

This loads the existing `dataset_metadata.json`, keeps every sample already collected
(their `location_id`s are preserved), counts how many samples exist per habitat class
(coral / seagrass / mangrove), and only samples the difference needed to reach the new
total. Any existing samples that predate the habitat schema are automatically deleted and
their IDs reused. `dataset_metadata.json` and `dataset_manifest.csv` are rewritten with
the full (old + new) sample list.

### Incremental downloads (`download`)

Simply re-run the same `download` command against the (possibly grown) manifest:

```bash
uv run python superres.py download superres/metadata/dataset_manifest.csv
```

- Samples with a `sample_metadata.json` checkpoint are skipped entirely.
- Low-res images already present on disk (matching the manifest's `lowres_count`)
  are not re-downloaded.
- In-progress Planet orders are tracked in `<sample>/planetscope/order_id.json`. Each run
  checks the order's current status once (without blocking): if it's still processing,
  the sample is skipped and re-checked on the next run; if it completed, the results are
  downloaded; if it failed outright, the checkpoint is cleared so the next run places a
  fresh order instead of reusing a duplicate.

### Dropping bad samples (`drop_samples.py`)

To remove specific samples so that `create --resume` can regenerate them:

```bash
python superres/drop_samples.py <metadata_dir> <data_dir> <location_id> [<location_id> ...]
```

This removes the entries from `dataset_metadata.json` / `dataset_manifest.csv` and
deletes the corresponding on-disk sample directories.

## Output Structure

```text
superres/                              # default output root (-o ./superres)
├── data/                              # Downloaded satellite images
│   ├── sample_000000/
│   │   ├── sentinel2/                 # High-res target (or low-res stack, depending on config)
│   │   │   ├── sentinel2_00_2025-02-03.tif
│   │   │   └── ...
│   │   ├── landsat/                   # Low-res input stack (default)
│   │   │   ├── landsat_00_2025-02-01.tif
│   │   │   └── ...
│   │   ├── planetscope/               # High-res target (if PlanetScope is configured)
│   │   │   ├── order_id.json          # Planet order checkpoint (resumable)
│   │   │   ├── <order_id>/            # Raw Planet order downloads
│   │   │   └── planetscope_000000.tif # Standardized 512×512 raster
│   │   └── sample_metadata.json       # Written once both downloads succeed
│   └── ...
├── processed/                         # Postprocessed rasters (output of postprocess step)
│   └── sample_000000/
│       ├── sentinel2/
│       └── landsat/
└── metadata/
    ├── dataset_metadata.json      # Complete metadata
    ├── dataset_manifest.csv       # CSV manifest for download script
    ├── creation.log               # Creation logs
    └── download.log               # Download logs
```

Credentials live outside this tree, in `./common/credentials.json` (see [Setup Credentials](#1-setup-credentials)).

A sample is only considered complete once `sample_metadata.json` exists; until then,
re-running `download` will pick up where it left off (see
[Resuming & Growing the Dataset](#resuming--growing-the-dataset)).

## Data Specifications

- **Sentinel-2**: 13 spectral bands at 10m–60m resolution
- **Landsat 8/9**: 11 bands (Collection 2 Level-2 SR) at 30m resolution
- **PlanetScope**: 8 spectral bands at 3m resolution
- **Focus**: Global shallow-water patches at coral, seagrass, and mangrove habitat sites
- **Stratification**: Equal samples per habitat class; coral/seagrass sites filtered by Kd490 bottom visibility
- **Cloud Cover**: Max 20%
- **Patch size**: 512 × 512 pixels at the high-res satellite's native resolution (5120 m at 10 m/px for Sentinel-2)

## Seasons

The dataset is sampled across 4 seasons:

1. Winter: January–February
2. Spring: April–May
3. Summer: July–August
4. Autumn: October–November

## Project Structure

```text
.
├── setup_credentials.py           # Script to setup API credentials securely
├── superres.py                    # CLI entry point: create / download / postprocess / tile / check subcommands
├── seagrass.py                    # CLI entry point for seagrass dataset preparation
├── seagrass/                      # Seagrass observation dataset preparation
│   ├── indonesia.py               # Indonesian sites: augments survey CSVs with Landsat/S2/PlanetScope bands
│   └── tampa_bay.py               # Tampa Bay transects: augments shapefile data with Landsat/S2/PlanetScope bands
├── pyproject.toml                 # Project metadata and dependencies
├── common/                        # Shared helper utilities
│   ├── dataset_utils.py           # Shared utility functions (directories, logging, CRS helpers)
│   ├── common.py                  # Label normalisation and dataframe helpers
│   ├── credentials.py             # Credential loading helpers
│   ├── gee_satellite.py           # GEESatelliteManager base class (GEE retrieval, turbidity, download)
│   ├── sentinel.py                # Sentinel-2 band constants and CSV feature helpers
│   ├── sentinel2.py               # Sentinel2Manager (subclasses GEESatelliteManager)
│   ├── landsat.py                 # Landsat band constants, CSV feature helpers, and LandsatManager
└── superres/                      # Main package
    ├── create_sr_dataset.py       # Main script to create dataset manifest
    ├── download_and_preprocess.py # Script to download and preprocess data
    ├── postprocess.py             # Align LR stack to HR grid and rescale to reflectance
    ├── tile_dataset.py            # Script to tile downloaded rasters for training
    ├── check_dataset.py           # Dataset integrity checker
    ├── analyse_dataset.py         # Dataset analysis and figure generation
    ├── drop_samples.py            # Remove samples from metadata and disk
    ├── world_sampling.py          # Global water-patch sampling logic
    ├── config.py                  # Configuration management
    ├── constants.py               # Global constants and configuration values
    ├── metadata.py                # Metadata storage and retrieval
    ├── satellites.py              # Registry of LR/HR satellite managers
    ├── planetscope.py             # PlanetScope data retrieval (Planet Labs API)
    ├── data/                      # Downloaded satellite data (created by download script)
    ├── metadata/                  # Metadata files - JSON/CSV/logs (created by create script)
    └── credentials/               # API credentials (created by setup_credentials.py)
```

By default `-o`/`--output` is `./superres`, so the generated `data/`, `metadata/`, and
`credentials/` directories sit alongside the package source under `superres/`. Pass a
different `-o` to keep generated data separate from the code.

## Modules

### Core Scripts

- **`superres.py`** - CLI entry point exposing `create`, `download`, `postprocess`, `tile`, and `check` subcommands
- **`superres/create_sr_dataset.py`** - Creates the dataset by sampling locations proportionally from ecoregions and querying satellite imagery
- **`superres/download_and_preprocess.py`** - Downloads satellite data and aligns the high-res raster to a standard grid
- **`superres/postprocess.py`** - Reprojects the low-res image stack onto the high-res grid and rescales all rasters to surface reflectance
- **`superres/tile_dataset.py`** - Tiles downloaded rasters into fixed-size training patches
- **`superres/check_dataset.py`** - Validates every sample for structural integrity, raster dimensions, georeferencing, and non-zero values
- **`superres/analyse_dataset.py`** - Generates a comprehensive set of analysis figures (spatial maps, spectral violins, temporal coverage, PSD curves, alignment diagnostics, sample thumbnails)
- **`superres/review_dataset.py`** - Streamlit GUI for visually reviewing samples; flag images for replacement or deletion; run with `uv run streamlit run superres/review_dataset.py -- --data-dir <data_dir>`
- **`superres/drop_samples.py`** - Removes specific samples from metadata and disk so they can be regenerated
- **`setup_credentials.py`** - Sets up API credentials securely in JSON format

### Configuration

- **`superres/config.py`** - `Config` class for managing dataset configuration and loading credentials
- **`superres/constants.py`** - Global constants (resolutions, bands, seasons, etc.)

### Data Management

- **`superres/metadata.py`** - `DatasetMetadata` class for managing metadata storage (JSON and CSV)
- **`superres/world_sampling.py`** - `WorldPatchSampler` for proposing globally-distributed water patch candidates

### Satellite Data Modules

- **`superres/satellites.py`** - Registry of satellite managers; defines `LOWRES_SATELLITES` (Sentinel-2, Landsat) and `HIGHRES_SATELLITES` (Sentinel-2, Landsat, PlanetScope)
- **`superres/planetscope.py`** - `PlanetScopeManager` for PlanetScope imagery via the Planet Labs Orders API
- **`common/gee_satellite.py`** - `GEESatelliteManager` base class with shared GEE retrieval, turbidity estimation, and download logic; `GEESatelliteSpec` dataclass
- **`common/sentinel2.py`** - `Sentinel2Manager` (subclasses `GEESatelliteManager`) for Sentinel-2 via Google Earth Engine
- **`common/sentinel.py`** - Sentinel-2 band column names, scale factor, spectral index calculations, and DataFrame helpers
- **`common/landsat.py`** - Landsat band column names, C2 L2 scale/offset conversion, spectral index calculations, DataFrame helpers, and `LandsatManager`
- **`common/dataset_utils.py`** - Shared utilities (UTM CRS calculation, point sampling, season date ranges, directory/logging helpers)

## Seagrass Dataset Preparation

The `seagrass/` package attaches imagery band values and spectral indices to in-situ
seagrass observation records, producing analysis-ready CSVs. Run via `seagrass.py`:

```bash
uv run python seagrass.py --site indonesia <root_dir> [--output-suffix _with_bands] [--gee-project <id>]
uv run python seagrass.py --site tampabay  <root_dir> [--output <file.csv>] [--gee-project <id>] \
  [--acolite-dir <dir>] [--s2-window-days <n>]
```

`--gee-project` is optional for both sites — if omitted, it falls back to `gee_project`
in `common/credentials.json`.

### Imagery sources

| Source | Resolution | Acquisition |
| --- | --- | --- |
| Landsat 8/9 (low-res) | 30 m | Sampled from Google Earth Engine |
| Sentinel-2 (high-res) | 10 m | Sampled from Google Earth Engine |
| PlanetScope (extra, optional) | 3 m | Read from pre-downloaded local rasters — not auto-downloaded due to quota limits |

### Indonesia (`seagrass/indonesia.py`)

Reads `summary.csv` and `summary_pantai_bama.csv` from `<root_dir>`. Discovers
PlanetScope scene folders under the same directory (the `*_psscene_analytic_8b_sr_udm2/`
layout from Planet Orders). Outputs:

- `summary_combined_landsat_with_bands.csv`
- `summary_combined_sentinel2_with_bands.csv`
- `summary_combined_planetscope_with_bands.csv` (if PlanetScope folders are present)

Sentinel-2/Landsat band sampling picks the AOI-locally-clearest scene within a
±60-day window (Cloud Score+ for S2, the QA_PIXEL "Clear" bit for Landsat) rather
than trusting whole-tile cloud-cover metadata, and samples the same physical
footprint (45m box) for both sensors so cross-sensor comparisons aren't skewed by
resolution. `seagrass.py --site indonesia` only runs this build step. The 9 sites
fall into two clusters ~200km apart (Bawean / Baluran); to download raw Level-1
scenes and run ACOLITE (water-optimised atmospheric correction) yourself, use
`seagrass/indonesia.py` directly — it has `dates`/`download`/`acolite`/`build`
subcommands per cluster; run `uv run python seagrass/indonesia.py --help` or see
the module docstring for the full pipeline.

### Tampa Bay (`seagrass/tampa_bay.py`)

Reads `transect_endpoints.csv` and `tb_seagrass_transects.json` from `<root_dir>`.
Outputs two CSVs, one per sensor (matching the Indonesia style above) — e.g. with the
default `-o tampa_transects_with_bands.csv` you get `tampa_transects_sentinel2_with_bands.csv`
and `tampa_transects_landsat_with_bands.csv` — each with that sensor's band and index
columns, plus Lyzenga depth-invariant and Beer-Lambert depth-corrected columns. Imagery
is sampled from ACOLITE NetCDF output when `--acolite-dir` is given (water-optimised
atmospheric correction), falling back to GEE's standard L2A/SR products otherwise.
Scene matching searches ±60 days around each observation date and takes whichever
scene is closest, rather than leaving a row blank — Tampa Bay's cloud cover regularly
wipes out a narrower window entirely.

`seagrass.py --site tampabay` only runs the final CSV-build step. To download raw
Level-1 scenes and run ACOLITE yourself, use `seagrass/tampa_bay.py` directly — it
has its own `dates` / `download` / `acolite` / `build` / `all` subcommands; run
`uv run python seagrass/tampa_bay.py --help` or see the module docstring for the
full pipeline and required accounts (Copernicus Data Space, USGS, ACOLITE).

## Satellite Architecture

The pipeline separates low-resolution (multi-image input stack) and high-resolution (single target) roles. The defaults are:

| Role      | Default       | Alternatives          |
|-----------|---------------|-----------------------|
| Low-res   | Landsat 8/9   | Sentinel-2            |
| High-res  | Sentinel-2    | Landsat, PlanetScope  |

GEE-backed satellites (`Sentinel2Manager`, `LandsatManager`) share retrieval, turbidity
estimation, and download logic via the `GEESatelliteManager` base class in
`common/gee_satellite.py`. To add a new GEE satellite, subclass `GEESatelliteManager`,
set a `SPEC` (`GEESatelliteSpec`), and register it in `superres/satellites.py`.

## Configuration Files

### `constants.py`

Contains all global constants:

- Satellite specifications (resolution, bands, collections)
- Dataset parameters (cloud cover threshold, images per location)
- Reflectance scale/offset per satellite
- Seasonal date ranges
- Output directory structure

### `config.py`

The `Config` class manages:

- Output directory setup
- Credential file loading
- Configuration persistence

**Example:**

```python
from superres.config import Config

config = Config(
    output_dir="./superres",
    total_samples=1000,
    ecoregion_file="ecoregions.gpkg",
)

# Load credentials
gee_creds = config.load_gee_credentials()
planet_key = config.load_planet_api_key()
```

## Logging

Both scripts produce detailed logs:

- **Creation log**: `./superres/metadata/creation.log`
- **Download log**: `./superres/metadata/download.log`
