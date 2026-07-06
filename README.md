# Super-Resolution Dataset Creator

A modular Python framework for creating and downloading multi-image super-resolution datasets from satellite imagery.

## Project Structure

```
.
├── setup_credentials.py           # Script to setup API credentials securely
├── superres.py                    # CLI entry point: create / download / postprocess / tile / check subcommands
├── seagrass.py                    # Example / project-specific script
├── seagrass/                      # Project-specific seagrass analysis scripts
│   ├── indonesia.py
│   └── tampa_bay.py
├── pyproject.toml                 # Project metadata and dependencies
├── common/                        # Shared helper utilities
│   ├── dataset_utils.py           # Shared utility functions (directories, logging, CRS helpers)
│   ├── common.py                  # Label normalisation and dataframe helpers
│   ├── credentials.py             # Credential loading helpers
│   └── sentinel.py                # Shared Sentinel helpers
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
    ├── gee_satellite.py           # Generic GEE satellite base class
    ├── sentinel2.py               # Sentinel-2 data retrieval (GEE)
    ├── landsat.py                 # Landsat 8/9 data retrieval (GEE)
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
- **`superres/analyse_dataset.py`** - Generates a comprehensive set of analysis figures (spatial maps, spectral violins, temporal coverage, PSD curves, alignment diagnostics, etc.)
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
- **`superres/gee_satellite.py`** - `GEESatelliteManager` base class with shared GEE retrieval, turbidity estimation, and download logic
- **`superres/sentinel2.py`** - `Sentinel2Manager` (subclasses `GEESatelliteManager`) for Sentinel-2 via Google Earth Engine
- **`superres/landsat.py`** - `LandsatManager` (subclasses `GEESatelliteManager`) for Landsat 8/9 Collection 2 via Google Earth Engine
- **`superres/planetscope.py`** - `PlanetScopeManager` for PlanetScope imagery via the Planet Labs Orders API
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

### 5. Tile the Dataset (optional)

```bash
uv run python superres.py tile superres/data -o ./superres/output
```

**Arguments:**

- `data_dir`: Path to the downloaded dataset's data directory, i.e. `superres/data` (positional)
- `-o, --output`: Output directory for tiled patches (default: `./superres/output`)
- `-t, --tile-size`: Tile size in pixels (default: 512)

Writes fixed-size training tiles to `<output>/tiles/<satellite>/`.

### 6. Check Dataset Integrity (optional)

```bash
uv run python superres.py check superres/data --manifest superres/metadata/dataset_manifest.csv
```

**Arguments:**

- `data_dir`: Path to the downloaded dataset's data directory (positional)
- `--manifest`: Path to CSV manifest (for cross-referencing expected samples)
- `--verbose`: Print per-file detail for every issue

Validates every sample for: directory structure, `sample_metadata.json` presence and validity, file counts, raster integrity (openable by rasterio, correct band count), dimensions (LR non-degenerate, HR exactly `patch_size_pixels × patch_size_pixels`), georeferencing (CRS, pixel size, origin), and non-zero centre values.

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

## Satellite Architecture

The pipeline separates low-resolution (multi-image input stack) and high-resolution (single target) roles. The defaults are:

| Role      | Default       | Alternatives          |
|-----------|---------------|-----------------------|
| Low-res   | Landsat 8/9   | Sentinel-2            |
| High-res  | Sentinel-2    | Landsat, PlanetScope  |

GEE-backed satellites (`Sentinel2Manager`, `LandsatManager`) share retrieval, turbidity
estimation, and download logic via the `GEESatelliteManager` base class in
`gee_satellite.py`. To add a new GEE satellite, subclass `GEESatelliteManager`, set a
`SPEC` (`GEESatelliteSpec`), and register it in `satellites.py`.

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

## Data Specifications

- **Sentinel-2**: 13 spectral bands at 10m–60m resolution
- **Landsat 8/9**: 11 bands (Collection 2 Level-2 SR) at 30m resolution
- **PlanetScope**: 8 spectral bands at 3m resolution
- **Focus**: Global shallow-water patches stratified by environment, depth, and turbidity
- **Cloud Cover**: Max 20%
- **Sampling**: Proportional to ecoregion area, diverse seasons

## Seasons

The dataset is sampled across 4 seasons:
1. Winter: January–February
2. Spring: April–May
3. Summer: July–August
4. Autumn: October–November

## Output Structure

```
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
