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
SENTINEL2_CLOUD_COVER_PROPERTY = "CLOUDY_PIXEL_PERCENTAGE"
SENTINEL2_RED_BAND = "B4"
SENTINEL2_GREEN_BAND = "B3"

# Landsat specifications (Collection 2, Level-2 surface reflectance).
# Landsat 8 and 9 are merged into a single collection so the combined revisit
# cadence (~8 days) is fast enough to fill a multi-image low-res stack.
LANDSAT_RESOLUTION = 30  # meters
LANDSAT_BANDS = 7
LANDSAT_COLLECTIONS = [
    "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT/LC09/C02/T1_L2",
]
LANDSAT_BAND_NAMES = [
    "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL",
]
LANDSAT_CLOUD_COVER_PROPERTY = "CLOUD_COVER"
LANDSAT_RED_BAND = "SR_B4"
LANDSAT_GREEN_BAND = "SR_B3"

# PlanetScope specifications
PLANETSCOPE_RESOLUTION = 3  # meters
PLANETSCOPE_BANDS = 8
PLANETSCOPE_ITEM_TYPE = "PSScene"
PLANETSCOPE_PRODUCT_BUNDLE = "analytic_8b_sr_udm2"
PLANETSCOPE_ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"

# Dataset configuration
LOWRES_MIN_IMAGES = 6  # Minimum low-res images per sample (reject if fewer)
LOWRES_MAX_IMAGES = 8  # Target max low-res images per location
IMAGES_PER_LOCATION = LOWRES_MAX_IMAGES  # Low-res images per high-res image
MAX_CLOUD_COVER = 20  # percent
TARGET_CRS = "EPSG:4326"  # WGS84 - will be reprojected to UTM per location
PATCH_SIZE_PIXELS = 512
# Ground footprint of a patch depends on which satellite is used as the
# high-resolution target (chosen at runtime via --highres-satellite). This
# default matches the current default high-res satellite (Sentinel-2, 10m)
# and is only used as a fallback when a manifest row has no patch_size_meters.
DEFAULT_PATCH_SIZE_METERS = PATCH_SIZE_PIXELS * SENTINEL2_RESOLUTION

ENVIRONMENT_CLASSES = ["estuary", "offshore"]
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
TURBIDITY_BUFFER_METERS = 1500.0

# Tiling configuration
TILE_SIZE_PIXELS = PATCH_SIZE_PIXELS  # 512×512 pixel tiles for training

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
    "lowres_satellite",
    "lowres_count",
    "highres_satellite",
    "highres_item_ids",
    "highres_order_id",
    "highres_product_bundle",
    "highres_aoi_geojson",
    "highres_count",
]
