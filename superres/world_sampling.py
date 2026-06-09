"""
World-scale patch sampling for the super-resolution dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Point, box, mapping
from shapely.ops import transform, unary_union

from common.dataset_utils import get_utm_crs

from .constants import (
    DEPTH_BINS_METERS,
    DEPTH_CLASSES,
    ENVIRONMENT_CLASSES,
    ESTUARY_DISTANCE_METERS,
    OFFSHORE_DISTANCE_METERS,
    PATCH_SIZE_METERS,
    PATCH_SIZE_PIXELS,
    RIVER_DISTANCE_METERS,
    SENTINEL2_TURBIDITY_BUFFER_METERS,
    TURBIDITY_CLASSES,
)


@dataclass(frozen=True)
class SampleTarget:
    environment_class: str
    depth_class: str
    turbidity_class: str


class WorldPatchSampler:
    """Sample 512x512 PlanetScope patches from a global water stratification grid."""

    def __init__(self, coastline_dir: str, gebco_file: Optional[str], logger):
        self.coastline_dir = Path(coastline_dir)
        self.gebco_file = self._resolve_gebco_file(gebco_file)
        self.logger = logger
        self._rng = np.random.default_rng()

        self.land_geometry = self._load_geometry(
            self.coastline_dir,
            ["GSHHS_*_L1.shp"],
            description="GSHHG land polygons",
        )
        self.river_geometry = self._load_geometry(
            self.coastline_dir,
            ["WDBII_river_*_L01.shp", "WDBII_river_*_L1.shp"],
            description="WDBII river network",
        )
        self.land_union = unary_union(self.land_geometry.geometry)
        self.river_union = unary_union(self.river_geometry.geometry)
        self.coastline_union = self.land_union.boundary

        self._gebco_dataset = rasterio.open(self.gebco_file)
        self._candidate_bank: Dict[Tuple[str, str], List[Dict]] = {}
        self.logger.info(
            "Loaded coastline and bathymetry sources: land=%s, rivers=%s, gebco=%s",
            len(self.land_geometry),
            len(self.river_geometry),
            self.gebco_file,
        )

    def close(self) -> None:
        if getattr(self, "_gebco_dataset", None) is not None:
            self._gebco_dataset.close()
            self._gebco_dataset = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _resolve_gebco_file(gebco_file: Optional[str]) -> str:
        data_dir = Path("./data")

        if gebco_file:
            path = Path(gebco_file)
            if not path.exists():
                raise FileNotFoundError(f"GEBCO file not found: {gebco_file}")

            # If a directory is provided, search inside for GeoTIFF tiles and merge if needed
            if path.is_dir():
                candidates = list(path.rglob("*gebco*.tif")) or list(path.rglob("*GEBCO*.tif")) or list(path.rglob("*.tif"))
                if not candidates:
                    raise FileNotFoundError(f"No GEBCO GeoTIFFs found under directory: {gebco_file}")
                if len(candidates) == 1:
                    return str(candidates[0])

                # Multiple tiles -> merge into a single GeoTIFF in ./data
                from rasterio.merge import merge

                src_files = [rasterio.open(str(p)) for p in sorted(candidates)]
                mosaic, out_trans = merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                })
                out_path = data_dir / "gebco_merged.tif"
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)

                for src in src_files:
                    src.close()
                return str(out_path)

            # If a file is provided, return it
            return str(path)

        # Auto-discover under ./data
        candidates = []
        for pattern in ("**/*gebco*.tif", "**/*gebco*.tiff", "**/*gebco*.nc", "**/*GEBCO*.tif", "**/*GEBCO*.tiff", "**/*GEBCO*.nc"):
            candidates.extend(data_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                "Could not auto-discover a GEBCO file under ./data. Pass --gebco-file explicitly."
            )
        return str(sorted(candidates)[0])

    @staticmethod
    def _load_geometry(base_dir: Path, patterns: Iterable[str], description: str) -> gpd.GeoDataFrame:
        candidates = []
        for pattern in patterns:
            candidates.extend(base_dir.rglob(pattern))
        if not candidates:
            raise FileNotFoundError(f"Could not find {description} under {base_dir}")
        gdf = gpd.read_file(sorted(candidates)[0])
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf

    @staticmethod
    def _degree_distance_to_meters(distance_degrees: float, latitude: float) -> float:
        km_per_degree = 111.320 * max(0.15, np.cos(np.deg2rad(latitude)))
        return distance_degrees * km_per_degree * 1000.0

    def _distance_to_geometry(self, point: Point, geometry) -> float:
        return float(point.distance(geometry))

    def _classify_environment(self, point: Point) -> Optional[str]:
        coastline_distance = self._degree_distance_to_meters(
            self._distance_to_geometry(point, self.coastline_union),
            point.y,
        )
        river_distance = self._degree_distance_to_meters(
            self._distance_to_geometry(point, self.river_union),
            point.y,
        )

        if river_distance <= RIVER_DISTANCE_METERS:
            return "river"
        if coastline_distance <= ESTUARY_DISTANCE_METERS:
            return "estuary"
        # Treat any location beyond the estuary buffer as offshore.
        # Previously the code only classified as 'offshore' when distance >= OFFSHORE_DISTANCE_METERS,
        # leaving a gap between ESTUARY_DISTANCE_METERS and OFFSHORE_DISTANCE_METERS that returned None.
        if coastline_distance > ESTUARY_DISTANCE_METERS:
            return "offshore"
        return None

    def _classify_depth(self, depth_meters: float) -> Optional[str]:
        for index, (lower, upper) in enumerate(DEPTH_BINS_METERS):
            if lower <= depth_meters < upper:
                return DEPTH_CLASSES[index]
        return None

    def _sample_depth(self, longitude: float, latitude: float) -> Optional[float]:
        try:
            sample = next(self._gebco_dataset.sample([(longitude, latitude)]), None)
            if sample is None:
                return None
            value = sample[0]
            if value is None:
                return None
            if np.isnan(value):
                return None
            # GEBCO is usually negative below sea level. Treat positive elevation as land.
            depth_meters = float(max(0.0, -value))
            return depth_meters
        except Exception as exc:
            self.logger.debug(f"Failed to sample GEBCO depth at ({latitude}, {longitude}): {exc}")
            return None

    def _build_patch_geometry(self, latitude: float, longitude: float) -> Dict:
        alignment_crs = get_utm_crs(latitude, longitude)
        forward = Transformer.from_crs("EPSG:4326", alignment_crs, always_xy=True)
        backward = Transformer.from_crs(alignment_crs, "EPSG:4326", always_xy=True)

        center_x, center_y = forward.transform(longitude, latitude)
        half_size = PATCH_SIZE_METERS / 2.0
        min_x = center_x - half_size
        min_y = center_y - half_size
        max_x = center_x + half_size
        max_y = center_y + half_size
        patch_utm = box(min_x, min_y, max_x, max_y)
        patch_wgs84 = transform(backward.transform, patch_utm)

        return {
            "alignment_crs": alignment_crs,
            "patch_polygon": patch_wgs84,
            "target_origin_x": min_x,
            "target_origin_y": min_y,
            "patch_size_pixels": PATCH_SIZE_PIXELS,
            "patch_size_meters": PATCH_SIZE_METERS,
        }

    @staticmethod
    def patch_geojson(patch_polygon) -> Dict:
        return mapping(patch_polygon)

    def target_bins(self) -> Iterable[SampleTarget]:
        for environment_class in ENVIRONMENT_CLASSES:
            for depth_class in DEPTH_CLASSES:
                for turbidity_class in TURBIDITY_CLASSES:
                    yield SampleTarget(environment_class, depth_class, turbidity_class)

    def random_point(self) -> Point:
        longitude = float(self._rng.uniform(-180.0, 180.0))
        latitude = float(self._rng.uniform(-60.0, 60.0))
        return Point(longitude, latitude)

    def _sample_point_near_geometry(self, geometry, buffer_meters: float, max_attempts: int = 500) -> Optional[Point]:
        """Sample a point inside a rough geometry buffer in lon/lat space.

        This is intentionally approximate. It is used only to bias rejection sampling
        toward regions that are likely to satisfy the requested environment class.
        """
        buffer_degrees = buffer_meters / 111_320.0
        buffered = geometry.buffer(buffer_degrees)
        minx, miny, maxx, maxy = buffered.bounds

        for _ in range(max_attempts):
            longitude = float(self._rng.uniform(minx, maxx))
            latitude = float(self._rng.uniform(miny, maxy))
            point = Point(longitude, latitude)
            if buffered.contains(point):
                return point
        return None

    def _sample_target_point(self, target: SampleTarget) -> Point:
        if target.environment_class == "river":
            point = self._sample_point_near_geometry(self.river_union, RIVER_DISTANCE_METERS * 2.5)
            if point is not None:
                return point
        elif target.environment_class == "estuary":
            point = self._sample_point_near_geometry(self.coastline_union, ESTUARY_DISTANCE_METERS * 1.25)
            if point is not None:
                return point

        # Offshore is the most abundant class, and any fallback target also ends up here.
        return self.random_point()

    def _environment_depth_key(self, environment_class: str, depth_class: str) -> Tuple[str, str]:
        return environment_class, depth_class

    def propose_candidate_for_environment_depth(
        self,
        environment_class: str,
        depth_class: str,
        max_attempts: int = 500,
    ) -> Optional[Dict]:
        """Find a candidate matching only environment and depth.

        Turbidity is intentionally ignored here so the creator can reuse the point
        for any open turbidity bin after the Planet/Sentinel-2 lookups run.
        """
        for _ in range(max_attempts):
            point = self._sample_target_point(SampleTarget(environment_class, depth_class, TURBIDITY_CLASSES[0]))
            if self.land_union.contains(point):
                continue

            point_environment = self._classify_environment(point)
            if point_environment != environment_class:
                continue

            depth_meters = self._sample_depth(point.x, point.y)
            if depth_meters is None:
                continue

            point_depth_class = self._classify_depth(depth_meters)
            if point_depth_class != depth_class:
                continue

            patch_info = self._build_patch_geometry(point.y, point.x)
            return {
                "latitude": point.y,
                "longitude": point.x,
                "environment_class": point_environment,
                "depth_class": point_depth_class,
                "depth_m": depth_meters,
                **patch_info,
            }
        return None

    def build_candidate_bank(self, per_environment_depth: int = 16, max_attempts_per_candidate: int = 500) -> Dict[Tuple[str, str], List[Dict]]:
        """Precompute a bank of candidates for each environment/depth pair.

        This trades memory for speed by doing the expensive local search once up front.
        The creator can then pop candidates from the bank instead of re-running random search.
        """
        bank: Dict[Tuple[str, str], List[Dict]] = {}
        for environment_class in ENVIRONMENT_CLASSES:
            for depth_class in DEPTH_CLASSES:
                key = self._environment_depth_key(environment_class, depth_class)
                bank[key] = []
                for _ in range(per_environment_depth):
                    candidate = self.propose_candidate_for_environment_depth(
                        environment_class,
                        depth_class,
                        max_attempts=max_attempts_per_candidate,
                    )
                    if candidate is None:
                        break
                    bank[key].append(candidate)
                self.logger.info(
                    "Built candidate bank for %s/%s with %s candidates",
                    environment_class,
                    depth_class,
                    len(bank[key]),
                )
        self._candidate_bank = bank
        return bank

    def pop_candidate_for_environment_depth(self, environment_class: str, depth_class: str) -> Optional[Dict]:
        key = self._environment_depth_key(environment_class, depth_class)
        bank = self._candidate_bank.get(key)
        if bank:
            return bank.pop()
        return None

    def propose_candidate(self) -> Optional[Dict]:
        point = self.random_point()
        if self.land_union.contains(point):
            return None

        environment_class = self._classify_environment(point)
        if environment_class is None:
            return None

        depth_meters = self._sample_depth(point.x, point.y)
        if depth_meters is None:
            return None

        depth_class = self._classify_depth(depth_meters)
        if depth_class is None:
            return None

        patch_info = self._build_patch_geometry(point.y, point.x)

        return {
            "latitude": point.y,
            "longitude": point.x,
            "environment_class": environment_class,
            "depth_class": depth_class,
            "depth_m": depth_meters,
            **patch_info,
        }

    def propose_candidate_for_target(self, target: SampleTarget, max_attempts: int = 500) -> Optional[Dict]:
        # Prefer a precomputed candidate bank if it exists.
        bank_candidate = self.pop_candidate_for_environment_depth(target.environment_class, target.depth_class)
        if bank_candidate is not None:
            return bank_candidate

        rejection_counts = {
            "land": 0,
            "environment": 0,
            "depth_missing": 0,
            "depth_class": 0,
        }
        for _ in range(max_attempts):
            point = self._sample_target_point(target)
            if self.land_union.contains(point):
                rejection_counts["land"] += 1
                continue

            environment_class = self._classify_environment(point)
            if environment_class != target.environment_class:
                rejection_counts["environment"] += 1
                continue

            depth_meters = self._sample_depth(point.x, point.y)
            if depth_meters is None:
                rejection_counts["depth_missing"] += 1
                continue

            depth_class = self._classify_depth(depth_meters)
            if depth_class != target.depth_class:
                rejection_counts["depth_class"] += 1
                continue

            patch_info = self._build_patch_geometry(point.y, point.x)
            return {
                "latitude": point.y,
                "longitude": point.x,
                "environment_class": environment_class,
                "depth_class": depth_class,
                "depth_m": depth_meters,
                **patch_info,
            }
        self.logger.debug(
            "No candidate found for target %s/%s after %s attempts (land=%s env=%s depth_missing=%s depth_class=%s)",
            target.environment_class,
            target.depth_class,
            max_attempts,
            rejection_counts["land"],
            rejection_counts["environment"],
            rejection_counts["depth_missing"],
            rejection_counts["depth_class"],
        )
        return None

    def estimate_turbidity_window(self, latitude: float, longitude: float) -> Tuple[str, str]:
        center_date = np.datetime64("2023-07-01")
        start_date = str(center_date - np.timedelta64(45, "D"))[:10]
        end_date = str(center_date + np.timedelta64(45, "D"))[:10]
        return start_date, end_date

    def add_turbidity_to_candidate(self, candidate: Dict, turbidity_index: float, turbidity_class: str) -> Dict:
        updated = dict(candidate)
        updated["turbidity_index"] = turbidity_index
        updated["turbidity_class"] = turbidity_class
        return updated

    def close_window(self) -> None:
        self.close()
