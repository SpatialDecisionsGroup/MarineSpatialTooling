"""Sentinel-2 GEE satellite manager."""

from .gee_satellite import GEESatelliteManager, GEESatelliteSpec


class Sentinel2Manager(GEESatelliteManager):
    """Manages Sentinel-2 image retrieval, turbidity estimation, and downloads via Google Earth Engine."""

    SPEC = GEESatelliteSpec(
        display_name="Sentinel-2",
        collection="COPERNICUS/S2_SR_HARMONIZED",
        band_names=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL", "AOT"],
        cloud_cover_property="CLOUDY_PIXEL_PERCENTAGE",
        red_band="B4",
        green_band="B3",
        resolution_meters=10,
        max_nodata_pct=5.0,
    )
