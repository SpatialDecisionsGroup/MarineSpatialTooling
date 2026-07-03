"""
Generic Google Earth Engine satellite collection manager.

Holds the retrieval, turbidity estimation, and download logic shared by any
satellite backed by a GEE ImageCollection (Sentinel-2, Landsat, ...). Concrete
satellites subclass `GEESatelliteManager` and set a class-level `SPEC`
describing their collection, bands, and cloud-cover property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import ee
import requests
import rasterio
from rasterio.errors import RasterioIOError
from pyproj import Transformer

from .constants import MAX_CLOUD_COVER, TURBIDITY_BINS, TURBIDITY_BUFFER_METERS


@dataclass(frozen=True)
class GEESatelliteSpec:
    """Static parameters that distinguish one GEE-backed satellite from another."""

    display_name: str
    collection: Union[str, Sequence[str]]
    band_names: List[str] = field(default_factory=list)
    cloud_cover_property: str = "CLOUD_COVER"
    red_band: str = ""
    green_band: str = ""
    resolution_meters: float = 10.0
    download_buffer_meters: float = 5566.0
    # Data-coverage quality filters applied in retrieve_images() alongside the
    # cloud-cover filter, to reject scenes with fill/nodata at the patch location.
    # filterBounds() finds scenes that intersect the AOI but the specific patch
    # can still land in a fill region (Landsat swath gaps, partial S2 granules).
    min_data_coverage_pct: Optional[float] = None   # e.g. DATA_COVERAGE_PERCENT >= 90
    max_nodata_pct: Optional[float] = None          # e.g. NODATA_PIXEL_PERCENTAGE <= 5


class GEESatelliteManager:
    """Manages image retrieval, turbidity estimation, and download for a GEE-backed satellite."""

    SPEC: GEESatelliteSpec

    def __init__(
        self,
        gee_credentials: Optional[str] = None,
        gee_project: Optional[str] = None,
        turbidity_raster: Optional[str] = None,
    ):
        self.gee_project = gee_project
        self.gee_credentials = gee_credentials
        self.logger = logging.getLogger(self.__class__.__name__)
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

    @classmethod
    def resolution_meters(cls) -> float:
        return cls.SPEC.resolution_meters

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

    def _collection(self) -> "ee.ImageCollection":
        """Build the (possibly merged) base ImageCollection for this satellite."""
        collection_ids = self.SPEC.collection
        if isinstance(collection_ids, str):
            collection_ids = [collection_ids]

        collection = ee.ImageCollection(collection_ids[0])
        for extra_id in collection_ids[1:]:
            collection = collection.merge(ee.ImageCollection(extra_id))
        return collection

    def retrieve_images(self, lat, long, date_start, date_end, max_images=8, aoi_geometry=None):
        """
        Retrieve images for a location and date range.

        Args:
            lat: Latitude in degrees
            long: Longitude in degrees
            date_start: Start date in YYYY-MM-DD format
            date_end: End date in YYYY-MM-DD format
            max_images: Maximum number of images to retrieve
            aoi_geometry: Optional GeoJSON-like geometry to search within instead of a point

        Returns:
            List of image metadata dictionaries
        """
        try:
            point = ee.Geometry.Point([long, lat])
            search_geometry = ee.Geometry(aoi_geometry) if aoi_geometry else point

            collection = self._collection() \
                .filterBounds(search_geometry) \
                .filterDate(date_start, date_end) \
                .filter(ee.Filter.lt(self.SPEC.cloud_cover_property, MAX_CLOUD_COVER))

            if self.SPEC.min_data_coverage_pct is not None:
                collection = collection.filter(
                    ee.Filter.gte("DATA_COVERAGE_PERCENT", self.SPEC.min_data_coverage_pct)
                )
            if self.SPEC.max_nodata_pct is not None:
                collection = collection.filter(
                    ee.Filter.lte("NODATA_PIXEL_PERCENTAGE", self.SPEC.max_nodata_pct)
                )

            collection = collection.sort(self.SPEC.cloud_cover_property).limit(max_images)

            images_info = collection.toList(max_images).getInfo()

            if not images_info:
                self.logger.debug(
                    f"No {self.SPEC.display_name} images found for ({lat}, {long}) "
                    f"between {date_start} and {date_end}"
                )
                return []

            results = []
            for img_info in images_info:
                asset_id = img_info.get("id", "")
                img_dict = {
                    "asset_id": asset_id,
                    "system_index": asset_id.split("/")[-1],
                    "date": self._parse_timestamp(
                        img_info.get("properties", {}).get("system:time_start", 0)
                    ),
                    "cloud_cover": img_info.get("properties", {}).get(
                        self.SPEC.cloud_cover_property, -1
                    ),
                    "source": f"{self.SPEC.display_name} via GEE",
                }
                results.append(img_dict)

            self.logger.info(
                f"Retrieved {len(results)} {self.SPEC.display_name} images for "
                f"({lat}, {long})"
            )

            return results

        except Exception as e:
            self.logger.error(f"Failed to retrieve {self.SPEC.display_name} images: {e}")
            return []

    def estimate_turbidity_index(
        self,
        lat: float,
        long: float,
        date_start: str,
        date_end: str,
        buffer_meters: float = TURBIDITY_BUFFER_METERS,
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
                self._collection()
                .filterBounds(region)
                .filterDate(date_start, date_end)
                .filter(ee.Filter.lt(self.SPEC.cloud_cover_property, MAX_CLOUD_COVER))
                .select([self.SPEC.red_band, self.SPEC.green_band])
            )

            if collection.size().getInfo() == 0:
                return None

            composite = collection.median()
            turbidity = composite.expression(
                "(red - green) / (red + green)",
                {
                    "red": composite.select(self.SPEC.red_band),
                    "green": composite.select(self.SPEC.green_band),
                },
                ).rename("turbidity")
            value = turbidity.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=self.SPEC.resolution_meters,
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
        buffer_meters: float = TURBIDITY_BUFFER_METERS,
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
        """Get a specific image by index within a re-queried, sorted window.

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
            collection = self._collection() \
                .filterBounds(point) \
                .filterDate(date_start, date_end) \
                .filter(ee.Filter.lt(self.SPEC.cloud_cover_property, MAX_CLOUD_COVER)) \
                .sort("system:time_start")

            image = ee.Image(collection.toList(index + 1).get(index))
            return image
        except Exception as exc:
            self.logger.error(f"Failed to get {self.SPEC.display_name} image at index {index}: {exc}")
            return None

    def get_image_by_asset_id(self, asset_id: str) -> Optional["ee.Image"]:
        """Get a specific image directly by its GEE asset id (returned as `asset_id` by retrieve_images)."""
        try:
            return ee.Image(asset_id)
        except Exception as exc:
            self.logger.error(f"Failed to resolve {self.SPEC.display_name} image {asset_id}: {exc}")
            return None

    def download_image(
        self,
        image: "ee.Image",
        lat: float,
        long: float,
        output_path: Path,
        bands: Optional[List[str]] = None,
        buffer_meters: Optional[float] = None,
    ) -> bool:
        """Download an image from Earth Engine.

        Args:
            image: ee.Image object to download
            lat: Latitude for bounding box
            long: Longitude for bounding box
            output_path: Local file path to save to
            bands: List of band names to export (defaults to all bands in SPEC)
            buffer_meters: Radius (in meters) of the download region around (lat, long).
                Defaults to SPEC.download_buffer_meters.

        Returns:
            True if download succeeded, False otherwise
        """
        if not bands:
            bands = self.SPEC.band_names
        if buffer_meters is None:
            buffer_meters = self.SPEC.download_buffer_meters

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Select bands and create small region for download
            image = image.select(bands)

            # ee.Geometry.buffer() takes its distance in true geodesic meters
            # (latitude-independent), regardless of the geometry's coordinate units.
            region = ee.Geometry.Point([long, lat]).buffer(buffer_meters)

            # Generate download URL. Do NOT force a geographic CRS together with a
            # meter-based scale; passing 'crs': 'EPSG:4326' with a meter scale causes
            # the scale to be interpreted in degrees and results in 1x1 images. Let
            # Earth Engine choose an appropriate CRS for the requested meter scale.
            url = image.getDownloadUrl({
                'scale': self.SPEC.resolution_meters,
                'region': region,
                'format': 'GeoTIFF',
                'filePerBand': False,
            })

            self.logger.debug(f"Downloading {self.SPEC.display_name} image from {url[:50]}...")

            response = requests.get(url, timeout=300)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            self.logger.info(f"Downloaded {self.SPEC.display_name} image to {output_path}")
            return True

        except Exception as exc:
            self.logger.error(f"Failed to download {self.SPEC.display_name} image: {exc}")
            if output_path.exists():
                output_path.unlink()
            return False
