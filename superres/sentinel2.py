"""
Sentinel-2 data retrieval and processing.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import List, Optional

import ee
import requests
import rasterio
from rasterio.errors import RasterioIOError
from pyproj import Transformer

from .constants import (
    SENTINEL2_COLLECTION,
    SENTINEL2_BAND_NAMES,
    IMAGES_PER_LOCATION,
    MAX_CLOUD_COVER,
    SENTINEL2_TURBIDITY_BUFFER_METERS,
    TURBIDITY_BINS,
)


class Sentinel2Manager:
    """Manages Sentinel-2 image retrieval and processing."""
    
    def __init__(
        self,
        gee_credentials: Optional[str] = None,
        gee_project: Optional[str] = None,
        turbidity_raster: Optional[str] = None,
    ):
        self.gee_project = gee_project
        self.gee_credentials = gee_credentials
        self.logger = logging.getLogger("Sentinel2Manager")
        self.turbidity_raster_path = turbidity_raster
        self.turbidity_dataset = None
        self._turbidity_transformer: Optional[Transformer] = None
        if self.turbidity_raster_path:
            try:
                path = self.turbidity_raster_path
                # For NetCDF files, find the georeferenced subdataset (e.g. Kd_490)
                if path.endswith(".nc"):
                    try:
                        with rasterio.open(path) as src:
                            subs = src.subdatasets
                        sub_path = next((s for s in subs if "Kd_490" in s), subs[0] if subs else None)
                        if sub_path:
                            path = sub_path
                    except Exception as sub_err:
                        self.logger.warning(f"Could not resolve NetCDF subdatasets: {sub_err}")

                self.turbidity_dataset = rasterio.open(path)
                # Transformer from WGS84 to raster CRS
                if self.turbidity_dataset.crs:
                    self._turbidity_transformer = Transformer.from_crs("EPSG:4326", self.turbidity_dataset.crs, always_xy=True)
                self.logger.info(f"Loaded turbidity raster subdataset: {path}")
            except RasterioIOError as re:
                self.logger.warning(f"Could not open turbidity raster {self.turbidity_raster_path}: {re}")
            except Exception as e:
                self.logger.warning(f"Error loading turbidity raster {self.turbidity_raster_path}: {e}")

        # Earth Engine authentication is handled via cached OAuth
        self._initialise_ee()
    
    def _initialise_ee(self) -> None:
        """Initialise Earth Engine. Authentication is handled via cached OAuth."""
        try:
            if self.gee_project:
                ee.Initialize(project=self.gee_project)
            else:
                ee.Initialize()
            self.logger.info("Earth Engine initialised successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialise Earth Engine: {e}")
    
    def retrieve_images(self, lat, long, date_start, date_end, max_images = IMAGES_PER_LOCATION, aoi_geometry=None):
        """
        Retrieve Sentinel-2 images for a location and date range.
        
        Args:
            lat: Latitude in degrees
            long: Longitude in degrees
            date_start: Start date in YYYY-MM-DD format
            date_end: End date in YYYY-MM-DD format
            max_images: Maximum number of images to retrieve
            
        Returns:
            List of image metadata dictionaries
        """
        try:
            # Create point geometry
            point = ee.Geometry.Point([long, lat])
            search_geometry = ee.Geometry(aoi_geometry) if aoi_geometry else point
            
            # Query Sentinel-2 collection
            collection = ee.ImageCollection(SENTINEL2_COLLECTION) \
                .filterBounds(search_geometry) \
                .filterDate(date_start, date_end) \
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER)) \
                .sort("system:time_start") \
                .limit(max_images)
            
            # Get image information
            images_info = collection.toList(max_images).getInfo()
            
            if not images_info:
                self.logger.debug(
                    f"No Sentinel-2 images found for ({lat}, {long}) "
                    f"between {date_start} and {date_end}"
                )
                return []
            
            results = []
            for img_info in images_info:
                img_dict = {
                    "system_index": img_info.get("id", "").split("/")[-1],
                    "date": self._parse_timestamp(
                        img_info.get("properties", {}).get("system:time_start", 0)
                    ),
                    "cloud_cover": img_info.get("properties", {}).get(
                        "CLOUDY_PIXEL_PERCENTAGE", -1
                    ),
                    "source": "Sentinel-2 via GEE",
                }
                results.append(img_dict)
            
            self.logger.info(
                f"Retrieved {len(results)} Sentinel-2 images for "
                f"({lat}, {long})"
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Failed to retrieve Sentinel-2 images: {e}")
            return []

    def estimate_turbidity_index(
        self,
        lat: float,
        long: float,
        date_start: str,
        date_end: str,
        buffer_meters: float = SENTINEL2_TURBIDITY_BUFFER_METERS,
    ) -> Optional[float]:
        # Prefer local turbidity raster if available
        try:
            if self.turbidity_dataset is not None:
                try:
                    # sample from raster
                    x, y = (long, lat)
                    if self._turbidity_transformer is not None:
                        x, y = self._turbidity_transformer.transform(long, lat)
                    
                    scale = self.turbidity_dataset.scales[0] if self.turbidity_dataset.scales else 1.0
                    offset = self.turbidity_dataset.offsets[0] if self.turbidity_dataset.offsets else 0.0
                    nodata = self.turbidity_dataset.nodatavals[0] if self.turbidity_dataset.nodatavals else None

                    for val in self.turbidity_dataset.sample([(x, y)]):
                        if val is None:
                            return None
                        # val may be an array for multiple bands; take first non-nodata
                        if hasattr(val, "tolist"):
                            arr = list(val)
                        else:
                            arr = [val]
                        for v in arr:
                            try:
                                if v is None:
                                    continue
                                if nodata is not None and v == nodata:
                                    continue
                                return float(v) * scale + offset
                            except Exception:
                                continue
                    return None
                except Exception as rexc:
                    self.logger.debug(f"Turbidity raster sampling failed for ({lat}, {long}): {rexc}")
        except Exception:
            # fall back to GEE below
            pass

        try:
            region = ee.Geometry.Point([long, lat]).buffer(buffer_meters)
            collection = (
                ee.ImageCollection(SENTINEL2_COLLECTION)
                .filterBounds(region)
                .filterDate(date_start, date_end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
                .select(["B3", "B4"])
            )

            if collection.size().getInfo() == 0:
                return None

            composite = collection.median()
            turbidity = composite.expression(
                "(red - green) / (red + green)",
                {
                    "red": composite.select("B4"),
                    "green": composite.select("B3"),
                },
                ).rename("turbidity")
            value = turbidity.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=20,
                bestEffort=True,
                maxPixels=1e8,
                ).get("turbidity").getInfo()
            if value is None:
                return None
            return float(value)
        except Exception as exc:
            self.logger.debug(f"Failed to estimate turbidity index for ({lat}, {long}): {exc}")
            return None

    @staticmethod
    def classify_turbidity(turbidity_index: float) -> Optional[str]:
        if turbidity_index is None:
            return None
        for label, lower, upper in TURBIDITY_BINS:
            if upper is None and turbidity_index >= lower:
                return label
            if upper is not None and lower <= turbidity_index < upper:
                return label
        return None

    def estimate_turbidity_class(
        self,
        lat: float,
        long: float,
        date_start: str,
        date_end: str,
        buffer_meters: float = SENTINEL2_TURBIDITY_BUFFER_METERS,
    ) -> Optional[dict]:
        turbidity_index = self.estimate_turbidity_index(
            lat,
            long,
            date_start,
            date_end,
            buffer_meters=buffer_meters,
        )
        turbidity_class = self.classify_turbidity(turbidity_index) if turbidity_index is not None else None
        if turbidity_class is None:
            return None
        return {
            "turbidity_index": turbidity_index,
            "turbidity_class": turbidity_class,
        }
    
    @staticmethod
    def _parse_timestamp(timestamp_ms):
        """
        Convert timestamp in milliseconds to ISO format date string.
        """
        return datetime.fromtimestamp(timestamp_ms / 1000).isoformat()

    def get_image_by_index(self, lat, long, date_start, date_end, index=0):
        """Get a specific Sentinel-2 image by index.
        
        Args:
            lat: Latitude in degrees
            long: Longitude in degrees
            date_start: Start date in YYYY-MM-DD format
            date_end: End date in YYYY-MM-DD format
            index: Index of image to retrieve (0-based)
        
        Returns:
            ee.Image object, or None if not found
        """
        try:
            point = ee.Geometry.Point([long, lat])
            collection = ee.ImageCollection(SENTINEL2_COLLECTION) \
                .filterBounds(point) \
                .filterDate(date_start, date_end) \
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER)) \
                .sort("system:time_start")
            
            image = ee.Image(collection.toList(index + 1).get(index))
            return image
        except Exception as exc:
            self.logger.error(f"Failed to get Sentinel-2 image at index {index}: {exc}")
            return None

    def download_image(self, image: "ee.Image", lat: float, long: float, 
                       output_path: Path, bands: Optional[List[str]] = None) -> bool:
        """Download a Sentinel-2 image from Earth Engine.
        
        Args:
            image: ee.Image object to download
            lat: Latitude for bounding box
            long: Longitude for bounding box
            output_path: Local file path to save to
            bands: List of band names to export (defaults to all SR bands)
        
        Returns:
            True if download succeeded, False otherwise
        """
        if not bands:
            bands = SENTINEL2_BAND_NAMES
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Select bands and create small region for download
            image = image.select(bands)
            
            # Create small bounding box (0.05 degrees ~= 5.5 km at equator)
            region = ee.Geometry.Point([long, lat]).buffer(0.05)

            # Generate download URL. Do NOT force a geographic CRS together with a
            # meter-based scale; passing 'crs': 'EPSG:4326' with scale=10 causes the
            # scale to be interpreted in degrees and results in 1x1 images. Let
            # Earth Engine choose an appropriate CRS for the requested meter scale.
            url = image.getDownloadUrl({
                'scale': 10,  # 10 m resolution
                'region': region,
                'format': 'GeoTIFF',
                'filePerBand': False,
            })
            
            self.logger.debug(f"Downloading Sentinel-2 image from {url[:50]}...")
            
            # Download the file
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Downloaded Sentinel-2 image to {output_path}")
            return True
            
        except Exception as exc:
            self.logger.error(f"Failed to download Sentinel-2 image: {exc}")
            if output_path.exists():
                output_path.unlink()
            return False
