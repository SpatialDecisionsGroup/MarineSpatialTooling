"""
Sentinel-2 data retrieval and processing.
"""

from .constants import (
    SENTINEL2_BAND_NAMES,
    SENTINEL2_CLOUD_COVER_PROPERTY,
    SENTINEL2_COLLECTION,
    SENTINEL2_GREEN_BAND,
    SENTINEL2_RED_BAND,
    SENTINEL2_RESOLUTION,
)
from .gee_satellite import GEESatelliteManager, GEESatelliteSpec


class Sentinel2Manager(GEESatelliteManager):
    """Manages Sentinel-2 image retrieval, turbidity estimation, and downloads via Google Earth Engine."""

    SPEC = GEESatelliteSpec(
        display_name="Sentinel-2",
        collection=SENTINEL2_COLLECTION,
        band_names=SENTINEL2_BAND_NAMES,
        cloud_cover_property=SENTINEL2_CLOUD_COVER_PROPERTY,
        red_band=SENTINEL2_RED_BAND,
        green_band=SENTINEL2_GREEN_BAND,
        resolution_meters=SENTINEL2_RESOLUTION,
        max_nodata_pct=5.0,
    )
