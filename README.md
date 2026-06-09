# Super-Resolution Dataset Creator

A modular Python framework for creating and downloading multi-image super-resolution datasets from satellite imagery.

## Project Structure

```
.
├── setup_credentials.py           # Script to setup API credentials securely
├── superres.py                    # Convenience entry / legacy wrapper
├── seagrass.py                    # Example / project-specific script
├── pyproject.toml                 # Project metadata and dependencies
├── common/                        # Shared helper utilities
└── superres/                      # Main package
    ├── create_sr_dataset.py       # Main script to create dataset
    ├── download_and_preprocess.py # Script to download and preprocess data
    ├── config.py                  # Configuration management
    ├── constants.py               # Global constants and configuration values
    ├── metadata.py                # Metadata storage and retrieval
    ├── sentinel2.py               # Sentinel-2 data retrieval
    ├── planetscope.py             # PlanetScope data retrieval
    ├── dataset_utils.py           # Shared utility functions
    └── output/                    # Output directory (created by scripts)
        ├── data/                  # Downloaded satellite data
        ├── metadata/              # Metadata files (JSON and CSV)
        └── credentials/           # API credentials (created by setup_credentials.py)
```

## Modules

### Core Scripts

- **`superres/create_sr_dataset.py`** - Creates the dataset by sampling locations proportionally from ecoregions and querying satellite imagery
- **`superres/download_and_preprocess.py`** - Downloads satellite data and aligns rasters to a common grid
- **`setup_credentials.py`** - Sets up API credentials securely in JSON format

### Configuration

- **`superres/config.py`** - `Config` class for managing dataset configuration and loading credentials
- **`superres/constants.py`** - Global constants (resolutions, bands, seasons, etc.)

### Data Management

- **`superres/metadata.py`** - `DatasetMetadata` class for managing metadata storage (JSON and CSV)

### Satellite Data Modules

- **`superres/sentinel2.py`** - `Sentinel2Manager` class for retrieving Sentinel-2 imagery via Google Earth Engine
- **`superres/planetscope.py`** - `PlanetScopeManager` class for retrieving PlanetScope imagery via Planet Labs API
- **`superres/dataset_utils.py`** - Shared utilities (UTM CRS calculation, point sampling, season date ranges)

## Workflow

### 1. Setup Credentials

```bash
python setup_credentials.py
```

The script will:
1. Prompt for your Planet Labs API key (press Enter to skip or keep existing value)
2. Open a browser window for Earth Engine OAuth authentication

This creates:
- `./superres/output/credentials/credentials.json` - Stores the Planet API key
- Earth Engine authentication is cached locally by the `ee` library

If you skip either credential, you can re-run the setup script later to complete it.

### 2. Create Dataset Manifest

```bash
python superres/create_sr_dataset.py \
  -o ./superres/output \
  -n 1000 \
  --coastline-dir ./data/gshhg-shp-2.3.7 \
  --gebco-file /path/to/gebco.tif
```

**Arguments:**
- `-o, --output`: Output directory (default: `./superres/output`)
- `-n, --total-samples`: Total number of samples (default: 1000)
- `--coastline-dir`: Path to the GSHHG/WDBII coastline dataset
- `--gebco-file`: Path to the GEBCO bathymetry raster

The scripts load credentials automatically from `./superres/output/credentials/credentials.json`.

**Outputs:**
- `./superres/output/metadata/dataset_metadata.json` - Complete metadata in JSON format
- `./superres/output/metadata/dataset_manifest.csv` - Manifest CSV for download script

### 3. Download and Preprocess Data

```bash
python superres/download_and_preprocess.py \
  ./superres/output/metadata/dataset_manifest.csv \
  -o ./superres/output
```

**Arguments:**
- `manifest_file`: Path to CSV manifest (positional)
- `-o, --output`: Output directory

The scripts load credentials automatically from `./superres/output/credentials/credentials.json`.

The download stage now creates Planet Orders API requests for each patch, clips the source item to the sampled AOI, reprojects to the local UTM zone, and standardizes the resulting PlanetScope raster to an exact 512x512 grid.

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
    output_dir="./superres/output",
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
superres/output/
├── data/                          # Downloaded satellite images
│   ├── sample_000000/
│   │   ├── s2_image_00.tif
│   │   ├── s2_image_01.tif
│   │   ├── ps_image_00.tif
│   │   └── sample_metadata.json
│   └── ...
├── metadata/
│   ├── dataset_metadata.json      # Complete metadata
│   ├── dataset_manifest.csv       # CSV manifest for download script
│   └── creation.log               # Creation logs
├── credentials/
│   └── credentials.json           # Planet API key only (EE auth is cached locally)
└── download.log                   # Download script logs
```

## Logging

Both scripts produce detailed logs:
- **Creation log**: `./superres/output/metadata/creation.log`
- **Download log**: `./superres/output/download.log`

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
