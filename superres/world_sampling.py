"""
World-scale patch sampling for the super-resolution dataset.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
from pyproj import Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform, unary_union

from common.dataset_utils import get_utm_crs

from .constants import (
    DEPTH_BINS_METERS,
    DEPTH_CLASSES,
    ENVIRONMENT_CLASSES,
    ESTUARY_DISTANCE_METERS,
    PATCH_SIZE_METERS,
    PATCH_SIZE_PIXELS,
    RIVER_DISTANCE_METERS,
    TURBIDITY_CLASSES,
    TURBIDITY_BINS,
)

# Resolution to downsample GEBCO to before vectorising (in degrees).
# 1/60° ≈ 1 arc-minute ≈ 1.85 km at the equator — more than adequate for
# the coarse depth bins we use (0–10 m, 10–30 m, 30–60 m).
_GEBCO_VECTORISE_RES_DEG = 1.0 / 60.0

# Output GeoPackage filename (written next to the GEBCO source file's directory).
_DEPTH_BINS_GPKG_NAME = "gebco_depth_bins.gpkg"


# ---------------------------------------------------------------------------
# Depth-bin polygon builder
# ---------------------------------------------------------------------------


def build_gebco_depth_polygons(
    gebco_path,
    land_union,
    output_path=None,
    force_rebuild=False,
    logger=None,
):
    """Vectorise a GEBCO raster into per-depth-bin ocean polygons.

    The result is cached as a GeoPackage.  On subsequent calls the file is
    loaded directly unless *force_rebuild* is True.

    Parameters
    ----------
    gebco_path:
        Path to the GEBCO GeoTIFF (may be merged or a single tile).
    land_union:
        A Shapely geometry covering all land areas (used to subtract land from
        the depth-bin polygons so only ocean pixels remain).
    output_path:
        Where to write the GeoPackage.  Defaults to a file called
        ``gebco_depth_bins.gpkg`` in the same directory as *gebco_path*.
    force_rebuild:
        If True, ignore any existing cached file and rebuild from scratch.
    logger:
        Optional standard-library logger.

    Returns
    -------
    GeoDataFrame with columns ``depth_class``, ``depth_bin_lower``,
    ``depth_bin_upper``, and ``geometry`` (WGS-84 MultiPolygon), one row per
    depth bin.
    """

    def _log(msg, *args):
        if logger:
            logger.info(msg, *args)

    # Resolve output path
    gpkg_path = (
        Path(output_path) if output_path else Path(gebco_path).parent / _DEPTH_BINS_GPKG_NAME
    )

    if gpkg_path.exists() and not force_rebuild:
        _log("Loading cached GEBCO depth-bin polygons from %s", gpkg_path)
        gdf = gpd.read_file(gpkg_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf

    _log("Building GEBCO depth-bin polygons (this runs once and is then cached) …")

    rows = []

    with rasterio.open(gebco_path) as src:
        src_res_x = abs(src.transform.a)
        src_res_y = abs(src.transform.e)
        target_res = _GEBCO_VECTORISE_RES_DEG

        # Compute downscale factor (at least 1)
        scale_x = max(1, int(round(target_res / src_res_x)))
        scale_y = max(1, int(round(target_res / src_res_y)))

        _log(
            "GEBCO native resolution %.4f°×%.4f°; downsampling by %d×%d to ~%.4f°",
            src_res_x,
            src_res_y,
            scale_x,
            scale_y,
            target_res,
        )

        out_width = max(1, src.width // scale_x)
        out_height = max(1, src.height // scale_y)

        # Read and downsample in one step using rasterio's resampling
        data = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=rasterio.enums.Resampling.average,
        )

        # Rebuild affine transform for the downsampled grid
        ds_transform = rasterio.transform.from_bounds(
            *src.bounds, width=out_width, height=out_height
        )

        nodata = src.nodata  # typically None or a sentinel like -32768

    _log(
        "Downsampled GEBCO to %d×%d pixels; vectorising %d depth bins …",
        out_width,
        out_height,
        len(DEPTH_BINS_METERS),
    )

    for idx, (lower, upper) in enumerate(DEPTH_BINS_METERS):
        depth_class = DEPTH_CLASSES[idx]

        # GEBCO convention: negative = below sea level, positive = above.
        # Convert to depth_meters = -elevation (so depths are positive numbers).
        # We only want ocean pixels (elevation < 0) within the bin range.
        elev_upper = -lower  # e.g. depth 0 m  → elevation  0 m
        elev_lower = -upper  # e.g. depth 10 m → elevation -10 m

        if nodata is not None:
            valid_mask = data != nodata
        else:
            valid_mask = np.ones(data.shape, dtype=bool)

        # Ocean pixels in this depth bin (elevation is negative, between elev_lower and elev_upper)
        bin_mask = valid_mask & (data >= elev_lower) & (data < elev_upper)

        # rasterio.features.shapes needs uint8.
        # Suppress NotGeoreferencedWarning: shapes() creates an internal in-memory
        # dataset from the raw array — the warning is benign since we supply the
        # transform explicitly for coordinate projection.
        bin_uint8 = bin_mask.astype(np.uint8)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
            geoms = [
                shape(geom)
                for geom, val in rasterio.features.shapes(bin_uint8, transform=ds_transform)
                if val == 1
            ]

        if not geoms:
            _log("  %s: no pixels found in bin [%.0f, %.0f) m — skipping", depth_class, lower, upper)
            continue

        _log("  %s: dissolving %d raw polygons …", depth_class, len(geoms))
        merged = unary_union(geoms)

        # Subtract land to keep only ocean areas
        _log("  %s: subtracting land …", depth_class)
        ocean_only = merged.difference(land_union)

        if ocean_only.is_empty:
            _log("  %s: polygon is empty after land subtraction — skipping", depth_class)
            continue

        rows.append(
            {
                "depth_class": depth_class,
                "depth_bin_lower": float(lower),
                "depth_bin_upper": float(upper),
                "geometry": ocean_only,
            }
        )
        _log("  %s: done (geom type=%s)", depth_class, ocean_only.geom_type)

    if not rows:
        raise RuntimeError("build_gebco_depth_polygons: no depth-bin polygons were produced")

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(gpkg_path), driver="GPKG")
    _log("Saved depth-bin polygons to %s", gpkg_path)

    return gdf


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleTarget:
    environment_class: str
    depth_class: str
    turbidity_class: str


# ---------------------------------------------------------------------------
# WorldPatchSampler
# ---------------------------------------------------------------------------


class WorldPatchSampler:
    """Sample 512×512 PlanetScope patches from a global water stratification grid."""

    def __init__(self, coastline_dir, gebco_file, logger, turbidity_raster=None):
        self.coastline_dir = Path(coastline_dir)
        self.gebco_file = self._resolve_gebco_file(gebco_file)
        self.logger = logger
        self._rng = np.random.default_rng()

        self.turbidity_raster_path = turbidity_raster
        self.turbidity_dataset = None
        self._turbidity_transformer = None
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
                        self.logger.warning(f"WorldPatchSampler could not resolve NetCDF subdatasets: {sub_err}")

                self.turbidity_dataset = rasterio.open(path)
                if self.turbidity_dataset.crs:
                    self._turbidity_transformer = Transformer.from_crs("EPSG:4326", self.turbidity_dataset.crs, always_xy=True)
                self.logger.info(f"WorldPatchSampler loaded turbidity raster subdataset: {path}")
            except Exception as e:
                self.logger.warning(f"WorldPatchSampler error loading turbidity raster: {e}")

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

        # Build (or load cached) depth-bin polygons from GEBCO
        self.depth_bin_gdf = build_gebco_depth_polygons(
            gebco_path=self.gebco_file,
            land_union=self.land_union,
            logger=logger,
        )
        # Index by depth_class for fast lookup
        self._depth_bin_polygons = {
            row["depth_class"]: row["geometry"]
            for _, row in self.depth_bin_gdf.iterrows()
        }

        # Keep GEBCO raster open only for precise depth_m metadata lookups
        self._gebco_dataset = rasterio.open(self.gebco_file)
        self._candidate_bank = {}

        # Per-session cache of environment-intersected depth polygons
        # Key: (depth_class, environment_class) → Shapely geometry or None
        self._env_depth_polygon_cache = {}

        self.logger.info(
            "Loaded coastline and bathymetry sources: land=%s, rivers=%s, gebco=%s, depth_bins=%s",
            len(self.land_geometry),
            len(self.river_geometry),
            self.gebco_file,
            list(self._depth_bin_polygons.keys()),
        )

    def close(self):
        if getattr(self, "_gebco_dataset", None) is not None:
            self._gebco_dataset.close()
            self._gebco_dataset = None
        if getattr(self, "turbidity_dataset", None) is not None:
            self.turbidity_dataset.close()
            self.turbidity_dataset = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # GEBCO file resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gebco_file(gebco_file):
        data_dir = Path("./data")

        if gebco_file:
            path = Path(gebco_file)
            if not path.exists():
                raise FileNotFoundError(f"GEBCO file not found: {gebco_file}")

            if path.is_dir():
                candidates = (
                    list(path.rglob("*gebco*.tif"))
                    or list(path.rglob("*GEBCO*.tif"))
                    or list(path.rglob("*.tif"))
                )
                if not candidates:
                    raise FileNotFoundError(
                        f"No GEBCO GeoTIFFs found under directory: {gebco_file}"
                    )
                if len(candidates) == 1:
                    return str(candidates[0])

                from rasterio.merge import merge

                src_files = [rasterio.open(str(p)) for p in sorted(candidates)]
                mosaic, out_trans = merge(src_files)
                out_meta = src_files[0].meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                    }
                )
                out_path = data_dir / "gebco_merged.tif"
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                for src in src_files:
                    src.close()
                return str(out_path)

            return str(path)

        candidates = []
        for pattern in (
            "**/*gebco*.tif",
            "**/*gebco*.tiff",
            "**/*gebco*.nc",
            "**/*GEBCO*.tif",
            "**/*GEBCO*.tiff",
            "**/*GEBCO*.nc",
        ):
            candidates.extend(data_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                "Could not auto-discover a GEBCO file under ./data. Pass --gebco-file explicitly."
            )
        return str(sorted(candidates)[0])

    # ------------------------------------------------------------------
    # Geometry loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _load_geometry(base_dir, patterns, description):
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

    # ------------------------------------------------------------------
    # Environment classification
    # ------------------------------------------------------------------

    @staticmethod
    def _degree_distance_to_meters(distance_degrees, latitude):
        km_per_degree = 111.320 * max(0.15, np.cos(np.deg2rad(latitude)))
        return distance_degrees * km_per_degree * 1000.0

    def _distance_to_geometry(self, point, geometry):
        return float(point.distance(geometry))

    def _classify_environment(self, point):
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
        if coastline_distance > ESTUARY_DISTANCE_METERS:
            return "offshore"
        return None

    # ------------------------------------------------------------------
    # Depth classification + raster lookup (metadata only)
    # ------------------------------------------------------------------

    def _classify_depth(self, depth_meters):
        for index, (lower, upper) in enumerate(DEPTH_BINS_METERS):
            if lower <= depth_meters < upper:
                return DEPTH_CLASSES[index]
        return None

    def _sample_depth(self, longitude, latitude):
        """Return precise depth in metres at a point (used for metadata, not sampling)."""
        try:
            sample = next(self._gebco_dataset.sample([(longitude, latitude)]), None)
            if sample is None:
                return None
            value = sample[0]
            if value is None or np.isnan(value):
                return None
            # GEBCO: negative = below sea level; positive = land/above sea level.
            return float(max(0.0, -value))
        except Exception as exc:
            self.logger.debug(
                "Failed to sample GEBCO depth at (%.4f, %.4f): %s", latitude, longitude, exc
            )
            return None

    # ------------------------------------------------------------------
    # Polygon-based point sampling (new fast path)
    # ------------------------------------------------------------------

    def _get_env_depth_polygon(self, depth_class, environment_class):
        """Return (and cache) the intersection of a depth-bin polygon with an environment zone.

        For 'offshore' this is just the depth-bin polygon itself (no extra intersection
        needed — land has already been subtracted during the build step).
        For 'estuary' and 'river' we intersect with the relevant coastline / river buffer.
        """
        cache_key = (depth_class, environment_class)
        if cache_key in self._env_depth_polygon_cache:
            return self._env_depth_polygon_cache[cache_key]

        base_poly = self._depth_bin_polygons.get(depth_class)
        if base_poly is None:
            self._env_depth_polygon_cache[cache_key] = None
            return None

        if environment_class == "offshore":
            # Subtract coastal buffer so we stay truly offshore
            estuary_buffer_deg = ESTUARY_DISTANCE_METERS / 111_320.0
            coast_buffer = self.coastline_union.buffer(estuary_buffer_deg)
            result = base_poly.difference(coast_buffer)
        elif environment_class == "estuary":
            estuary_buffer_deg = ESTUARY_DISTANCE_METERS / 111_320.0
            river_buffer_deg = RIVER_DISTANCE_METERS / 111_320.0
            coast_zone = self.coastline_union.buffer(estuary_buffer_deg)
            river_zone = self.river_union.buffer(river_buffer_deg)
            # Estuary = within coastal buffer but not within river buffer
            estuary_zone = coast_zone.difference(river_zone)
            result = base_poly.intersection(estuary_zone)
        elif environment_class == "river":
            river_buffer_deg = RIVER_DISTANCE_METERS / 111_320.0
            river_zone = self.river_union.buffer(river_buffer_deg)
            result = base_poly.intersection(river_zone)
        else:
            result = base_poly

        if result is None or result.is_empty:
            result = None

        self._env_depth_polygon_cache[cache_key] = result
        return result

    def _sample_point_in_polygon(self, polygon, max_attempts=500, batch_size=64):
        """Draw a uniformly random point from inside *polygon*.

        For MultiPolygons, picks a component at random weighted by area before
        doing bounding-box rejection sampling.  This dramatically improves hit
        rate for sparse geometries (e.g. thin shelf bands) whose global bounding
        box is mostly empty ocean.
        """
        if polygon is None or polygon.is_empty:
            return None

        # Decompose MultiPolygon into components for area-weighted selection
        if polygon.geom_type == "MultiPolygon":
            components = list(polygon.geoms)
        else:
            components = [polygon]

        areas = np.array([c.area for c in components], dtype=float)
        total = areas.sum()
        if total == 0:
            return None
        weights = areas / total

        for _ in range(max_attempts):
            # Pick a component proportional to its area
            idx = int(self._rng.choice(len(components), p=weights))
            comp = components[idx]
            minx, miny, maxx, maxy = comp.bounds

            xs = self._rng.uniform(minx, maxx, batch_size)
            ys = self._rng.uniform(miny, maxy, batch_size)
            for x, y in zip(xs, ys):
                pt = Point(x, y)
                if comp.contains(pt):
                    return pt

        return None

    # ------------------------------------------------------------------
    # Patch geometry builder
    # ------------------------------------------------------------------

    def _build_patch_geometry(self, latitude, longitude):
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
    def patch_geojson(patch_polygon):
        return mapping(patch_polygon)

    # ------------------------------------------------------------------
    # Target enumeration
    # ------------------------------------------------------------------

    def target_bins(self):
        for environment_class in ENVIRONMENT_CLASSES:
            for depth_class in DEPTH_CLASSES:
                for turbidity_class in TURBIDITY_CLASSES:
                    yield SampleTarget(environment_class, depth_class, turbidity_class)

    # ------------------------------------------------------------------
    # Random point helpers (unrestricted — global ocean)
    # ------------------------------------------------------------------

    def random_point(self):
        longitude = float(self._rng.uniform(-180.0, 180.0))
        latitude = float(self._rng.uniform(-90.0, 90.0))
        return Point(longitude, latitude)

    # ------------------------------------------------------------------
    # Candidate proposals
    # ------------------------------------------------------------------

    def propose_candidate_for_environment_depth(self, environment_class, depth_class, max_attempts=500):
        """Find a candidate matching environment and depth using polygon sampling.

        Draws random points directly from the pre-built depth-bin polygon
        intersected with the environment zone — no raster rejection loop.
        Turbidity is intentionally ignored here; the caller handles it after
        Planet/Sentinel-2 lookups.
        """
        poly = self._get_env_depth_polygon(depth_class, environment_class)

        if poly is not None and not poly.is_empty:
            # Fast path: sample from the pre-built polygon
            for _ in range(max_attempts):
                point = self._sample_point_in_polygon(poly, max_attempts=1, batch_size=64)
                if point is None:
                    continue

                # Verify environment classification (the polygon intersection is
                # approximate due to degree-based buffering)
                env = self._classify_environment(point)
                if env != environment_class:
                    continue

                depth_meters = self._sample_depth(point.x, point.y)
                if depth_meters is None:
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
        else:
            self.logger.debug(
                "No depth-bin polygon for %s/%s; falling back to rejection sampling",
                environment_class,
                depth_class,
            )

        # Fallback: original rejection sampling (kept for robustness)
        return self._propose_candidate_rejection(environment_class, depth_class, max_attempts)

    def _propose_candidate_rejection(self, environment_class, depth_class, max_attempts=500):
        """Original rejection-sampling fallback (used when polygon is unavailable)."""
        target = SampleTarget(environment_class, depth_class, TURBIDITY_CLASSES[0])
        for _ in range(max_attempts):
            point = self._sample_target_point(target)
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
                "environment_class": environment_class,
                "depth_class": depth_class,
                "depth_m": depth_meters,
                **patch_info,
            }
        return None

    def _sample_target_point(self, target):
        if target.environment_class == "river":
            point = self._sample_point_near_geometry(self.river_union, RIVER_DISTANCE_METERS * 2.5)
            if point is not None:
                return point
        elif target.environment_class == "estuary":
            point = self._sample_point_near_geometry(
                self.coastline_union, ESTUARY_DISTANCE_METERS * 1.25
            )
            if point is not None:
                return point
        return self.random_point()

    def _sample_point_near_geometry(self, geometry, buffer_meters, max_attempts=500):
        """Approximate biased rejection sampling toward a geometry buffer."""
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

    # ------------------------------------------------------------------
    # Candidate bank (pre-computed pool)
    # ------------------------------------------------------------------

    def build_candidate_bank(self, per_target=24, per_environment_depth=None, max_attempts_per_candidate=1000):
        """Precompute a bank of candidates for each environment/depth/turbidity target."""
        if per_environment_depth is not None:
            per_target = per_environment_depth
        bank = {}
        for target in self.target_bins():
            key = (target.environment_class, target.depth_class, target.turbidity_class)
            bank[key] = []
            for _ in range(per_target):
                candidate = self.propose_candidate_for_target_stratum(
                    target.environment_class,
                    target.depth_class,
                    target.turbidity_class,
                    max_attempts=max_attempts_per_candidate,
                )
                if candidate is None:
                    break
                bank[key].append(candidate)
            self.logger.info(
                "Built candidate bank for %s/%s/%s with %s candidates",
                target.environment_class,
                target.depth_class,
                target.turbidity_class,
                len(bank[key]),
            )
        self._candidate_bank = bank
        return bank

    def pop_candidate_for_target(self, target):
        key = (target.environment_class, target.depth_class, target.turbidity_class)
        bank = self._candidate_bank.get(key)
        if bank:
            return bank.pop()
        return None

    def propose_candidate(self):
        """Propose a random candidate from anywhere in the ocean."""
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

    def propose_candidate_for_target(self, target, max_attempts=500):
        # Prefer a precomputed candidate bank if it exists.
        bank_candidate = self.pop_candidate_for_target(target)
        if bank_candidate is not None:
            return bank_candidate

        return self.propose_candidate_for_target_stratum(
            target.environment_class,
            target.depth_class,
            target.turbidity_class,
            max_attempts=max_attempts,
        )

    def _classify_turbidity_from_raster(self, longitude, latitude):
        if self.turbidity_dataset is None:
            return None
        try:
            x, y = (longitude, latitude)
            if self._turbidity_transformer is not None:
                x, y = self._turbidity_transformer.transform(longitude, latitude)
            
            scale = self.turbidity_dataset.scales[0] if self.turbidity_dataset.scales else 1.0
            offset = self.turbidity_dataset.offsets[0] if self.turbidity_dataset.offsets else 0.0
            nodata = self.turbidity_dataset.nodatavals[0] if self.turbidity_dataset.nodatavals else None

            for val in self.turbidity_dataset.sample([(x, y)]):
                if val is None:
                    continue
                if hasattr(val, "tolist"):
                    arr = list(val)
                else:
                    arr = [val]
                for v in arr:
                    if v is None:
                        continue
                    try:
                        if nodata is not None and v == nodata:
                            continue
                        idx_val = float(v) * scale + offset
                        # classify using TURBIDITY_BINS
                        for label, lower, upper in TURBIDITY_BINS:
                            if upper is None and idx_val >= lower:
                                return label
                            if upper is not None and lower <= idx_val < upper:
                                return label
                    except Exception:
                        continue
            return None
        except Exception as e:
            self.logger.debug(f"Failed to sample turbidity raster at ({latitude}, {longitude}): {e}")
            return None

    def propose_candidate_for_target_stratum(self, environment_class, depth_class, turbidity_class, max_attempts=1000):
        poly = self._get_env_depth_polygon(depth_class, environment_class)
        if poly is not None and not poly.is_empty:
            for _ in range(max_attempts):
                point = self._sample_point_in_polygon(poly, max_attempts=1, batch_size=64)
                if point is None:
                    continue

                # Check environment
                env = self._classify_environment(point)
                if env != environment_class:
                    continue

                # Check depth
                depth_meters = self._sample_depth(point.x, point.y)
                if depth_meters is None:
                    continue

                # Check turbidity from local raster if available
                if self.turbidity_dataset is not None:
                    t_class = self._classify_turbidity_from_raster(point.x, point.y)
                    if t_class != turbidity_class:
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

        # Fallback to rejection sampling if polygon is unavailable
        target = SampleTarget(environment_class, depth_class, turbidity_class)
        for _ in range(max_attempts):
            point = self._sample_target_point(target)
            if point is None or self.land_union.contains(point):
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

            if self.turbidity_dataset is not None:
                t_class = self._classify_turbidity_from_raster(point.x, point.y)
                if t_class != turbidity_class:
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
        return None

    # ------------------------------------------------------------------
    # Turbidity helpers
    # ------------------------------------------------------------------

    def estimate_turbidity_window(self, latitude, longitude):
        center_date = np.datetime64("2023-07-01")
        start_date = str(center_date - np.timedelta64(45, "D"))[:10]
        end_date = str(center_date + np.timedelta64(45, "D"))[:10]
        return start_date, end_date

    def add_turbidity_to_candidate(self, candidate, turbidity_index, turbidity_class):
        updated = dict(candidate)
        updated["turbidity_index"] = turbidity_index
        updated["turbidity_class"] = turbidity_class
        return updated

    def close_window(self):
        self.close()
