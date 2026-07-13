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
from shapely.geometry import Point, box, mapping
from shapely.ops import transform, unary_union
from shapely.prepared import prep

from common.dataset_utils import get_utm_crs

from .constants import (
    BOTTOM_VISIBILITY_KD490_MAX,
    DEFAULT_PATCH_SIZE_METERS,
    HABITAT_CLASSES,
    HABITAT_EXTENTS_DIR,
    PATCH_SIZE_PIXELS,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleTarget:
    habitat_class: str


# ---------------------------------------------------------------------------
# WorldPatchSampler
# ---------------------------------------------------------------------------


class WorldPatchSampler:
    """Sample 512×512 patches from global coral/seagrass/mangrove habitat polygons."""

    def __init__(
        self,
        coastline_dir,
        gebco_file,
        logger,
        turbidity_raster=None,
        habitat_extents_dir=None,
        patch_size_meters=DEFAULT_PATCH_SIZE_METERS,
        patch_size_pixels=PATCH_SIZE_PIXELS,
    ):
        self.coastline_dir = Path(coastline_dir)
        self.gebco_file = self._resolve_gebco_file(gebco_file)
        self.logger = logger
        self.patch_size_meters = patch_size_meters
        self.patch_size_pixels = patch_size_pixels
        self._rng = np.random.default_rng()

        habitat_dir = Path(habitat_extents_dir) if habitat_extents_dir else Path(HABITAT_EXTENTS_DIR)

        # Kd490 raster for bottom-visibility check (coral/seagrass only)
        self.turbidity_raster_path = turbidity_raster
        self.turbidity_dataset = None
        self._turbidity_transformer = None
        if self.turbidity_raster_path:
            try:
                path = self.turbidity_raster_path
                if path.endswith(".nc"):
                    try:
                        with rasterio.open(path) as src:
                            subs = src.subdatasets
                        sub_path = next((s for s in subs if "Kd_490" in s), subs[0] if subs else None)
                        if sub_path:
                            path = sub_path
                    except Exception as sub_err:
                        self.logger.warning("Could not resolve NetCDF subdatasets: %s", sub_err)
                self.turbidity_dataset = rasterio.open(path)
                if self.turbidity_dataset.crs:
                    self._turbidity_transformer = Transformer.from_crs(
                        "EPSG:4326", self.turbidity_dataset.crs, always_xy=True
                    )
                self.logger.info("Loaded Kd490/turbidity raster: %s", path)
            except Exception as e:
                self.logger.warning("Error loading turbidity raster: %s", e)

        # Land geometry for ocean point validation
        self.land_geometry = self._load_geometry(
            self.coastline_dir,
            ["GSHHS_*_L1.shp"],
            description="GSHHG land polygons",
        )
        self.land_union = unary_union(self.land_geometry.geometry)
        self._land_union_prepared = prep(self.land_union)

        # Keep GEBCO raster open for depth_m metadata at accepted points
        self._gebco_dataset = rasterio.open(self.gebco_file)

        # Load habitat polygon GeoDataFrames, one per class
        self._habitat_gdfs: dict[str, gpd.GeoDataFrame] = {}
        self._habitat_components: dict[str, tuple] = {}  # class → (components, prepared, weights)
        for habitat_class in HABITAT_CLASSES:
            sub_dir = habitat_dir / habitat_class
            if not sub_dir.exists():
                self.logger.warning(
                    "Habitat directory not found: %s — run habitat_extents.sh first", sub_dir
                )
                continue
            try:
                gdf = self._load_habitat_gdf(sub_dir, habitat_class)
                self._habitat_gdfs[habitat_class] = gdf
                self._habitat_components[habitat_class] = self._build_components(gdf)
                self.logger.info(
                    "Loaded %s habitat: %d polygons",
                    habitat_class, len(gdf),
                )
            except Exception as exc:
                self.logger.warning("Failed to load %s habitat: %s", habitat_class, exc)

        if not self._habitat_gdfs:
            raise RuntimeError(
                f"No habitat polygon data loaded from {habitat_dir}. "
                "Run habitat_extents.sh to download coral/seagrass/mangrove extents."
            )

        self.logger.info(
            "WorldPatchSampler ready: habitats=%s, land=%d polygons, gebco=%s",
            list(self._habitat_gdfs.keys()),
            len(self.land_geometry),
            self.gebco_file,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

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

    @staticmethod
    def _load_habitat_gdf(sub_dir: Path, habitat_class: str) -> gpd.GeoDataFrame:
        """Find and load the polygon shapefile or gpkg from a habitat subdirectory.

        Tries all candidates in preference order: gpkg before shp, polygon-named
        files before point/line files.  Falls through to any file that actually
        contains Polygon/MultiPolygon rows so that naming variations don't break it.
        """
        # Collect all vector files (gpkg first — single file, no sidecar issues)
        candidates: list[Path] = []
        for pattern in ("**/*.gpkg", "**/*.shp"):
            candidates.extend(sorted(sub_dir.rglob(pattern)))

        if not candidates:
            raise FileNotFoundError(f"No shapefile or gpkg found under {sub_dir}")

        # Score: penalise obvious point/line files, reward polygon-named files.
        # WCMC convention: _Py_ = polygon, _Pt_ = point, _Ln_ = line.
        _SKIP = ("_pt_", "_pt.", "_ln_", "_ln.", "_line", "_point")
        _POLY = ("_py_", "_py.", "poly", "v4", "v7", habitat_class)

        def _score(p: Path) -> tuple[int, int]:
            stem = p.stem.lower()
            is_bad  = any(k in stem for k in _SKIP)
            is_poly = any(k in stem for k in _POLY)
            return (0 if is_bad else 1, 1 if is_poly else 0)

        ordered = sorted(candidates, key=_score, reverse=True)

        for path in ordered:
            try:
                gdf = gpd.read_file(str(path))
            except Exception:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
            if len(gdf) == 0:
                continue
            return gdf.reset_index(drop=True)

        raise FileNotFoundError(
            f"No polygon shapefile or gpkg with usable polygon rows found under {sub_dir}"
        )

    def _build_components(self, gdf: gpd.GeoDataFrame):
        """Pre-compute per-row area weights and bounds for random polygon selection."""
        # Project to equal-area CRS so weights reflect true surface area, not degrees²
        areas = gdf.geometry.to_crs("EPSG:6933").area.values.astype(float)
        total = areas.sum()
        weights = areas / total if total > 0 else np.ones(len(gdf)) / len(gdf)
        # Cumulative weights let _sample_point_in_habitat use searchsorted (O(log n))
        # instead of rng.choice(n, p=weights) which recomputes cumsum on every call.
        cumweights = np.cumsum(weights)
        cumweights[-1] = 1.0  # clamp floating-point drift
        # Pre-extract bounds so the hot path avoids per-attempt geometry attribute access
        bounds = np.array([geom.bounds for geom in gdf.geometry], dtype=np.float64)
        return gdf, cumweights, bounds

    # ------------------------------------------------------------------
    # Depth lookup (metadata only)
    # ------------------------------------------------------------------

    def _sample_depth(self, longitude, latitude):
        try:
            sample = next(self._gebco_dataset.sample([(longitude, latitude)]), None)
            if sample is None:
                return None
            value = sample[0]
            if value is None or np.isnan(value):
                return None
            return float(max(0.0, -value))
        except Exception as exc:
            self.logger.debug(
                "Failed to sample GEBCO depth at (%.4f, %.4f): %s", latitude, longitude, exc
            )
            return None

    # ------------------------------------------------------------------
    # Bottom-visibility check via Kd490
    # ------------------------------------------------------------------

    def _check_bottom_visibility(self, longitude, latitude) -> bool:
        """Return True if the water is clear enough to see benthic habitat.

        Uses the Kd490 (diffuse attenuation at 490 nm) raster.  If the raster
        is unavailable the check is skipped (returns True so sampling continues).
        """
        if self.turbidity_dataset is None:
            return True
        try:
            x, y = longitude, latitude
            if self._turbidity_transformer is not None:
                x, y = self._turbidity_transformer.transform(longitude, latitude)

            scale = self.turbidity_dataset.scales[0] if self.turbidity_dataset.scales else 1.0
            offset = self.turbidity_dataset.offsets[0] if self.turbidity_dataset.offsets else 0.0
            nodata = self.turbidity_dataset.nodatavals[0] if self.turbidity_dataset.nodatavals else None

            for val in self.turbidity_dataset.sample([(x, y)]):
                if val is None:
                    continue
                arr = list(val) if hasattr(val, "tolist") else [val]
                for v in arr:
                    if v is None:
                        continue
                    if nodata is not None and v == nodata:
                        return True  # no data → don't reject
                    kd490 = float(v) * scale + offset
                    return kd490 <= BOTTOM_VISIBILITY_KD490_MAX
        except Exception as exc:
            self.logger.debug(
                "Failed to sample Kd490 at (%.4f, %.4f): %s", latitude, longitude, exc
            )
        return True  # default to accepting when the check fails

    # ------------------------------------------------------------------
    # Habitat polygon point sampling
    # ------------------------------------------------------------------

    def _sample_point_in_habitat(self, habitat_class: str, max_attempts=500) -> Point | None:
        """Draw a uniformly random point from a random polygon in the habitat GDF.

        Picks a polygon weighted by area, then does bounding-box rejection
        sampling within that polygon.
        """
        entry = self._habitat_components.get(habitat_class)
        if entry is None:
            return None
        gdf, cumweights, bounds = entry
        prep_cache: dict[int, object] = {}

        for _ in range(max_attempts):
            # O(log n) weighted selection via precomputed cumulative weights
            row_idx = int(np.searchsorted(cumweights, self._rng.random()))
            row_idx = min(row_idx, len(gdf) - 1)

            minx, miny, maxx, maxy = bounds[row_idx]

            # Cache prepared geometry — same polygon often selected multiple times
            if row_idx not in prep_cache:
                prep_cache[row_idx] = prep(gdf.geometry.iloc[row_idx])
            prepared_geom = prep_cache[row_idx]

            # Rejection sampling within the bounding box
            for _ in range(64):
                x = float(self._rng.uniform(minx, maxx))
                y = float(self._rng.uniform(miny, maxy))
                pt = Point(x, y)
                if prepared_geom.contains(pt):
                    return pt

        return None

    # ------------------------------------------------------------------
    # Patch geometry builder
    # ------------------------------------------------------------------

    def _build_patch_geometry(self, latitude, longitude):
        alignment_crs = get_utm_crs(latitude, longitude)
        # Cache transformers — Transformer.from_crs is expensive and UTM zones repeat
        if not hasattr(self, "_transformer_cache"):
            self._transformer_cache: dict[str, tuple] = {}
        if alignment_crs not in self._transformer_cache:
            self._transformer_cache[alignment_crs] = (
                Transformer.from_crs("EPSG:4326", alignment_crs, always_xy=True),
                Transformer.from_crs(alignment_crs, "EPSG:4326", always_xy=True),
            )
        forward, backward = self._transformer_cache[alignment_crs]

        center_x, center_y = forward.transform(longitude, latitude)
        half_size = self.patch_size_meters / 2.0
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
            "patch_size_pixels": self.patch_size_pixels,
            "patch_size_meters": self.patch_size_meters,
        }

    @staticmethod
    def patch_geojson(patch_polygon):
        return mapping(patch_polygon)

    # ------------------------------------------------------------------
    # Target enumeration
    # ------------------------------------------------------------------

    def target_bins(self):
        for habitat_class in HABITAT_CLASSES:
            if habitat_class in self._habitat_gdfs:
                yield SampleTarget(habitat_class)

    # ------------------------------------------------------------------
    # Candidate proposals
    # ------------------------------------------------------------------

    def propose_candidate_for_target(self, target: SampleTarget, max_attempts=500):
        """Find a candidate patch inside the given habitat class.

        Points are drawn from the habitat polygon, then validated against land
        and (for submerged habitats) a Kd490 bottom-visibility threshold.
        """
        habitat_class = target.habitat_class
        needs_visibility = habitat_class in ("coral", "seagrass")

        for _ in range(max_attempts):
            point = self._sample_point_in_habitat(habitat_class)
            if point is None:
                continue

            # Reject land points (habitat polygons sometimes clip coastal land)
            if self._land_union_prepared.contains(point):
                continue

            # Reject turbid water for submerged habitats
            if needs_visibility and not self._check_bottom_visibility(point.x, point.y):
                continue

            depth_m = self._sample_depth(point.x, point.y)

            patch_info = self._build_patch_geometry(point.y, point.x)
            return {
                "latitude": point.y,
                "longitude": point.x,
                "habitat_class": habitat_class,
                "depth_m": depth_m,
                **patch_info,
            }

        self.logger.debug(
            "propose_candidate_for_target: gave up after %d attempts for habitat %s",
            max_attempts,
            habitat_class,
        )
        return None

    # ------------------------------------------------------------------
    # Candidate bank (kept for API compatibility, simplified)
    # ------------------------------------------------------------------

    def build_candidate_bank(self, per_target=24, per_environment_depth=None, max_attempts_per_candidate=1000):
        """Pre-compute a small bank of candidates per habitat class."""
        if per_environment_depth is not None:
            per_target = per_environment_depth
        bank = {}
        for target in self.target_bins():
            key = target.habitat_class
            bank[key] = []
            for _ in range(per_target):
                candidate = self.propose_candidate_for_target(target, max_attempts=max_attempts_per_candidate)
                if candidate is None:
                    break
                bank[key].append(candidate)
            self.logger.info(
                "Built candidate bank for %s with %d candidates",
                target.habitat_class,
                len(bank[key]),
            )
        self._candidate_bank = bank
        return bank

    def pop_candidate_for_target(self, target: SampleTarget):
        key = target.habitat_class
        bank = getattr(self, "_candidate_bank", {}).get(key)
        if bank:
            return bank.pop()
        return None

    # ------------------------------------------------------------------
    # Turbidity helpers (kept for downstream compatibility)
    # ------------------------------------------------------------------

    def estimate_turbidity_window(self, latitude, longitude):
        center_date = np.datetime64("2023-07-01")
        start_date = str(center_date - np.timedelta64(45, "D"))[:10]
        end_date = str(center_date + np.timedelta64(45, "D"))[:10]
        return start_date, end_date

    def close_window(self):
        self.close()
