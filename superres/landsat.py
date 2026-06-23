"""
Landsat data retrieval and processing.
"""

from .constants import (
    LANDSAT_BAND_NAMES,
    LANDSAT_CLOUD_COVER_PROPERTY,
    LANDSAT_COLLECTIONS,
    LANDSAT_GREEN_BAND,
    LANDSAT_RED_BAND,
    LANDSAT_RESOLUTION,
)
from .gee_satellite import GEESatelliteManager, GEESatelliteSpec


class LandsatManager(GEESatelliteManager):
    """Manages Landsat (8+9, Collection 2 Level-2 SR) image retrieval, turbidity
    estimation, and downloads via Google Earth Engine."""

    SPEC = GEESatelliteSpec(
        display_name="Landsat",
        collection=LANDSAT_COLLECTIONS,
        band_names=LANDSAT_BAND_NAMES,
        cloud_cover_property=LANDSAT_CLOUD_COVER_PROPERTY,
        red_band=LANDSAT_RED_BAND,
        green_band=LANDSAT_GREEN_BAND,
        resolution_meters=LANDSAT_RESOLUTION,
    )
