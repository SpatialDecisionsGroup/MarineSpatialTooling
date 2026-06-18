# Super-Resolution Dataset Creator

A modular Python framework for creating and downloading multi-image super-resolution datasets from satellite imagery.

## Project Structure

```
.
├── setup_credentials.py           # Script to setup API credentials securely
├── superres.py                    # CLI entry point: create / download / tile subcommands
├── seagrass.py                    # Example / project-specific script
├── pyproject.toml                 # Project metadata and dependencies
├── common/                        # Shared helper utilities
│   └── dataset_utils.py           # Shared utility functions (directories, logging, CRS helpers)
└── superres/                      # Main package
    ├── create_sr_dataset.py       # Main script to create dataset
    ├── download_and_preprocess.py # Script to download and preprocess data
    ├── tile_dataset.py            # Script to tile downloaded rasters for training
    ├── world_sampling.py          # Global water-patch sampling logic
    ├── config.py                  # Configuration management
    ├── constants.py               # Global constants and configuration values
    ├── metadata.py                # Metadata storage and retrieval
    ├── sentinel2.py               # Sentinel-2 data retrieval
    ├── planetscope.py             # PlanetScope data retrieval
    ├── data/                      # Downloaded satellite data (created by download script)
    ├── metadata/                  # Metadata files - JSON/CSV/logs (created by create script)
    └── credentials/               # API credentials (created by setup_credentials.py)
```

By default `-o`/`--output` is `./superres`, so the generated `data/`, `metadata/`, and
`credentials/` directories sit alongside the package source under `superres/`. Pass a
different `-o` to keep generated data separate from the code.

## Modules

### Core Scripts

- **`superres.py`** - CLI entry point exposing `create`, `download`, and `tile` subcommands
- **`superres/create_sr_dataset.py`** - Creates the dataset by sampling locations proportionally from ecoregions and querying satellite imagery
- **`superres/download_and_preprocess.py`** - Downloads satellite data and aligns rasters to a common grid
- **`superres/tile_dataset.py`** - Tiles downloaded Sentinel-2/PlanetScope rasters into fixed-size training patches
- **`setup_credentials.py`** - Sets up API credentials securely in JSON format

### Configuration

- **`superres/config.py`** - `Config` class for managing dataset configuration and loading credentials
- **`superres/constants.py`** - Global constants (resolutions, bands, seasons, etc.)

### Data Management

- **`superres/metadata.py`** - `DatasetMetadata` class for managing metadata storage (JSON and CSV)
- **`superres/world_sampling.py`** - `WorldPatchSampler` for proposing globally-distributed water patch candidates

### Satellite Data Modules

- **`superres/sentinel2.py`** - `Sentinel2Manager` class for retrieving Sentinel-2 imagery via Google Earth Engine
- **`superres/planetscope.py`** - `PlanetScopeManager` class for retrieving PlanetScope imagery via Planet Labs API
- **`common/dataset_utils.py`** - Shared utilities (UTM CRS calculation, point sampling, season date ranges, directory/logging helpers)

## Workflow

All scripts are run through `uv run`, which uses the project's `uv.lock`/`pyproject.toml`
environment. Replace `uv run python` with `python` if you manage your own virtualenv.

### 1. Setup Credentials

```bash
uv run python setup_credentials.py
```

Optional flags: `-o, --output-dir` (default `./superres/credentials`), `-p, --planet-key`
to pass the Planet API key non-interactively.

The script will:
1. Prompt for your Planet Labs API key (press Enter to skip or keep existing value)
2. Open a browser window for Earth Engine OAuth authentication

This creates:

- `./superres/credentials/credentials.json` - Stores the Planet API key and GEE project ID
- Earth Engine authentication is cached locally by the `ee` library

If you skip either credential, you can re-run the setup script later to complete it.

### 2. Create Dataset Manifest

```bash
uv run python superres.py create \
  -n 1000 \
  --coastline-dir ./data/gshhg-shp-2.3.7 \
  --gebco-file ./data/gebco_2026_geotiff
```

**Arguments:**

- `-o, --output`: Output directory (default: `./superres`)
- `-n, --total-samples` or `-p, --samples-per-area`: total sample count, or samples per
  environment/depth/turbidity stratum (one is required)
- `--coastline-dir`: Path to the GSHHG/WDBII coastline dataset (default `./data/gshhg-shp-2.3.7`)
- `--gebco-file`: Path to the GEBCO bathymetry raster (default `./data/gebco_2026_geotiff`)
- `--turbidity-file`: Path to a turbidity raster/NetCDF (default `./data/turbidity.nc`)
- `--include-ecoregions` / `--exclude-ecoregions`: comma-separated MEOW province names
- `--resume`: continue from an existing `dataset_metadata.json` instead of starting over
  (see [Resuming & Growing the Dataset](#resuming--growing-the-dataset))

The scripts load credentials automatically from `./superres/credentials/credentials.json`.

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

The scripts load credentials automatically from `./superres/credentials/credentials.json`.

For each manifest row, this creates a Planet Orders API request, clips the source item to
the sampled AOI, reprojects to the local UTM zone, and standardizes the resulting
PlanetScope raster to an exact 512x512 grid alongside the matching Sentinel-2 images. It
is safe to re-run the same command repeatedly — see
[Resuming & Growing the Dataset](#resuming--growing-the-dataset).

### 4. Tile the Dataset (optional)

```bash
uv run python superres.py tile superres/data -o ./superres/output
```

**Arguments:**

- `data_dir`: Path to the downloaded dataset's data directory, i.e. `superres/data` (positional)
- `-o, --output`: Output directory for tiled patches (default: `./superres/output`)
- `-t, --tile-size`: Tile size in pixels (default: 512)

Writes fixed-size training tiles to `<output>/tiles/sentinel2/` and `<output>/tiles/planetscope/`.

## Resuming & Growing the Dataset

Both the create and download steps are designed to be re-run incrementally, so a small
test run can be grown into the full dataset without losing existing samples or
re-downloading data you already have.

### Growing the manifest (`create`)

Re-run with `--resume` and a larger `-n`/`-p`:

```bash
uv run python superres.py create -n 2000 --resume \
  --coastline-dir ./data/gshhg-shp-2.3.7 \
  --gebco-file ./data/gebco_2026_geotiff
```

This loads the existing `dataset_metadata.json`, keeps every sample already collected
(their `location_id`s are preserved), counts how many samples exist per
environment/depth/turbidity stratum, and only samples the difference needed to reach the
new total. `dataset_metadata.json` and `dataset_manifest.csv` are rewritten with the full
(old + new) sample list.

### Incremental downloads (`download`)

Simply re-run the same `download` command against the (possibly grown) manifest:

```bash
uv run python superres.py download superres/metadata/dataset_manifest.csv
```

- Samples with a `sample_metadata.json` checkpoint are skipped entirely.
- Sentinel-2 images already present on disk (matching the manifest's `sentinel2_count`)
  are not re-downloaded.
- In-progress Planet orders are tracked in `<sample>/planetscope/order_id.json`. Each run
  checks the order's current status once (without blocking): if it's still processing,
  the sample is skipped and re-checked on the next run; if it completed, the results are
  downloaded; if it failed outright, the checkpoint is cleared so the next run places a
  fresh order instead of reusing a duplicate.

## Configuration Files

### `constants.py`

Contains all global constants:
- Satellite specifications (resolution, bands, collections)
- Dataset parameters (cloud cover threshold, images per location)
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

## Data Specifications

- **Sentinel-2**: 13 spectral bands at 10m-60m resolution
- **PlanetScope**: 8 spectral bands at 3m resolution
- **Focus**: Global shallow-water patches stratified by environment, depth, and turbidity
- **Cloud Cover**: Max 20%
- **Images per Location**: 9 Sentinel-2 images per PlanetScope image
- **Sampling**: Proportional to ecoregion area, diverse seasons

## Seasons

The dataset is sampled across 4 seasons:
1. Winter: January-February
2. Spring: April-May
3. Summer: July-August
4. Autumn: October-November

## Output Structure

```
superres/                              # default output root (-o ./superres)
├── data/                              # Downloaded satellite images
│   ├── sample_000000/
│   │   ├── sentinel2/
│   │   │   ├── sentinel2_00_2025-02-03.tif
│   │   │   └── ...
│   │   ├── planetscope/
│   │   │   ├── order_id.json          # Planet order checkpoint (resumable)
│   │   │   ├── <order_id>/            # Raw Planet order downloads
│   │   │   └── planetscope_000000.tif # Standardized 512x512 raster
│   │   └── sample_metadata.json       # Written once both downloads succeed
│   └── ...
├── metadata/
│   ├── dataset_metadata.json      # Complete metadata
│   ├── dataset_manifest.csv       # CSV manifest for download script
│   ├── creation.log               # Creation logs
│   └── download.log               # Download logs
└── credentials/
    └── credentials.json           # Planet API key + GEE project (EE auth is cached locally)
```

A sample is only considered complete once `sample_metadata.json` exists; until then,
re-running `download` will pick up where it left off (see
[Resuming & Growing the Dataset](#resuming--growing-the-dataset)).

## Logging

Both scripts produce detailed logs:

- **Creation log**: `./superres/metadata/creation.log`
- **Download log**: `./superres/metadata/download.log`

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

## Notes

- Credentials are stored securely with restricted file permissions (0o600)
- All API keys and credentials should be stored in the credentials directory
- The CSV manifest file enables reproducible downloads without re-querying APIs
- Ecoregion sampling ensures balanced geographical coverage
- Seasonal sampling provides temporal diversity
