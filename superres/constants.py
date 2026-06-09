"""
Global constants for SR dataset creation and processing.
"""

# Sentinel-2 specifications
SENTINEL2_RESOLUTION = 10  # meters
SENTINEL2_BANDS = 13
SENTINEL2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL2_BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
    "B8A", "B11", "B12", "SCL", "AOT"
]

# PlanetScope specifications
PLANETSCOPE_RESOLUTION = 3  # meters
PLANETSCOPE_BANDS = 8
PLANETSCOPE_ITEM_TYPE = "PSScene"
PLANETSCOPE_PRODUCT_BUNDLE = "analytic_8b_sr_udm2"
PLANETSCOPE_ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"

# Dataset configuration
SENTINEL2_MIN_IMAGES = 6  # Minimum S2 images per sample (reject if fewer)
SENTINEL2_MAX_IMAGES = 8  # Target max S2 images per location
IMAGES_PER_LOCATION = SENTINEL2_MAX_IMAGES  # Sentinel-2 images per PlanetScope image
MAX_CLOUD_COVER = 20  # percent
TARGET_CRS = "EPSG:4326"  # WGS84 - will be reprojected to UTM per location
PATCH_SIZE_PIXELS = 512
PATCH_SIZE_METERS = PATCH_SIZE_PIXELS * PLANETSCOPE_RESOLUTION

ENVIRONMENT_CLASSES = ["river", "estuary", "offshore"]
DEPTH_CLASSES = ["shallow_1", "shallow_2", "shallow_3"]
TURBIDITY_CLASSES = ["clear", "moderate", "turbid"]

DEPTH_BINS_METERS = [
    (0.0, 10.0),
    (10.0, 30.0),
    (30.0, 60.0),
]

TURBIDITY_BINS = [
    ("clear", -1.0, 0.03),
    ("moderate", 0.03, 0.08),
    ("turbid", 0.08, None),
]

RIVER_DISTANCE_METERS = 5000.0
ESTUARY_DISTANCE_METERS = 30000.0
OFFSHORE_DISTANCE_METERS = 50000.0
SENTINEL2_TURBIDITY_BUFFER_METERS = 1500.0

# Tiling configuration
TILE_SIZE_PIXELS = PATCH_SIZE_PIXELS  # 512×512 pixel tiles for training
PLANETSCOPE_TILE_STRIDE = 512  # Non-overlapping tiles from full scenes

# Seasonal date ranges (start_month, end_month); winter wraps across year-end.
SEASONS = [
    (12, 2),  # Winter (Northern hemisphere)
    (3, 5),   # Spring
    (6, 8),   # Summer
    (9, 11),  # Autumn
]

# Directory structure
DEFAULT_OUTPUT_DIR = "./superres"
DATA_SUBDIR = "data"
METADATA_SUBDIR = "metadata"
CREDENTIALS_SUBDIR = "credentials"
CREDENTIALS_CONFIG_FILENAME = "credentials.json"

# Manifest CSV columns
MANIFEST_COLUMNS = [
    "location_id",
    "latitude",
    "longitude",
    "season_id",
    "province",
    "environment_class",
    "depth_class",
    "depth_m",
    "turbidity_class",
    "turbidity_index",
    "date_range_start",
    "date_range_end",
    "alignment_crs",
    "patch_size_pixels",
    "patch_size_meters",
    "target_origin_x",
    "target_origin_y",
    "planet_item_ids",
    "planet_order_id",
    "planet_product_bundle",
    "planet_aoi_geojson",
    "sentinel2_count",
    "planetscope_count",
]
