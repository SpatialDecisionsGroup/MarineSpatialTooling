"""
Multi-image super-resolution dataset creator for global water patches.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from time import perf_counter

import numpy as np
from tqdm import tqdm
import logging

from common.dataset_utils import setup_logger

from .config import config_create
from .constants import (
    DEPTH_CLASSES,
    ENVIRONMENT_CLASSES,
    IMAGES_PER_LOCATION,
    PATCH_SIZE_METERS,
    PATCH_SIZE_PIXELS,
    PLANETSCOPE_PRODUCT_BUNDLE,
    PLANETSCOPE_RESOLUTION,
    SENTINEL2_MIN_IMAGES,
    SENTINEL2_MAX_IMAGES,
    SENTINEL2_TURBIDITY_BUFFER_METERS,
    TURBIDITY_CLASSES,
)
from .metadata import DatasetMetadata
from .planetscope import PlanetScopeManager
from .sentinel2 import Sentinel2Manager
from .world_sampling import SampleTarget, WorldPatchSampler


PLANETSCOPE_ARCHIVE_START = "2016-01-01"
PLANETSCOPE_ARCHIVE_END = date.today().isoformat()
SENTINEL2_WINDOW_DAYS = 45


def _season_id_from_month(month: int) -> int:
    return (month - 1) // 3


def _split_total_across_targets(total_samples: int, targets: List[SampleTarget]) -> Dict[SampleTarget, int]:
    base = total_samples // len(targets)
    remainder = total_samples % len(targets)
    counts = {target: base for target in targets}
    if remainder > 0:
        rng = np.random.default_rng()
        chosen_indices = rng.choice(len(targets), size=remainder, replace=False)
        for index in chosen_indices:
            counts[targets[int(index)]] += 1
    return counts


def _existing_target_counts(metadata: DatasetMetadata) -> Dict[Tuple[str, str, str], int]:
    counts: Dict[Tuple[str, str, str], int] = {}
    for sample in metadata.metadata.get("samples", []):
        key = (
            str(sample.get("environment_class", "")),
            str(sample.get("depth_class", "")),
            str(sample.get("turbidity_class", "")),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _remaining_target_counts(target_counts: Dict[SampleTarget, int], existing_counts: Dict[Tuple[str, str, str], int], targets: List[SampleTarget]) -> Dict[SampleTarget, int]:
    remaining: Dict[SampleTarget, int] = {}
    for target in targets:
        current_key = (target.environment_class, target.depth_class, target.turbidity_class)
        completed = existing_counts.get(current_key, 0)
        remaining[target] = max(0, target_counts[target] - completed)
    return remaining


def _open_targets(remaining_counts: Dict[SampleTarget, int]) -> List[SampleTarget]:
    return [target for target, count in remaining_counts.items() if count > 0]


def _find_matching_target(
    environment_class: str,
    depth_class: str,
    turbidity_class: str,
    remaining_counts: Dict[SampleTarget, int],
) -> Optional[SampleTarget]:
    for target, count in remaining_counts.items():
        if count <= 0:
            continue
        if (
            target.environment_class == environment_class
            and target.depth_class == depth_class
            and target.turbidity_class == turbidity_class
        ):
            return target
    return None


def create_dataset():
    """Create the complete world-scale super-resolution dataset."""
    config = config_create()
    metadata_path = config.metadata_dir / "dataset_metadata.json"
    if getattr(config, "resume", False) and metadata_path.exists():
        metadata = DatasetMetadata.load_json(config.metadata_dir)
    else:
        metadata = DatasetMetadata(config.metadata_dir)

    logger = setup_logger("SRDatasetCreator", config.metadata_dir / "creation.log", level=logging.DEBUG)

    sentinel2_manager = Sentinel2Manager(
        config.load_gee_credentials(), config.load_gee_project(), turbidity_raster=getattr(config, "turbidity_file", None)
    )
    planetscope_manager = PlanetScopeManager(config.load_planet_api_key())
    sampler = WorldPatchSampler(config.coastline_dir, config.gebco_file, logger)

    print("=" * 60)
    print("Creating Super-Resolution Dataset")
    print("=" * 60)
    print(f"Patch size: {PATCH_SIZE_PIXELS} x {PATCH_SIZE_PIXELS} pixels")
    print(f"Ground size: {PATCH_SIZE_METERS:.0f} m x {PATCH_SIZE_METERS:.0f} m")
    print(f"Planet bundle: {PLANETSCOPE_PRODUCT_BUNDLE}")

    bank_start = perf_counter()
    sampler.build_candidate_bank(per_environment_depth=24)
    logger.info("Candidate bank precompute completed in %.2fs", perf_counter() - bank_start)

    targets = list(sampler.target_bins())
    if not targets:
        raise ValueError("No sampling targets configured")

    if getattr(config, "samples_per_area", None):
        target_counts = {target: int(config.samples_per_area) for target in targets}
        total_samples = int(config.samples_per_area) * len(targets)
        logger.info(
            "Sampling mode: %s per stratum across %s strata = %s total",
            config.samples_per_area,
            len(targets),
            total_samples,
        )
    else:
        total_samples = int(config.total_samples)
        target_counts = _split_total_across_targets(total_samples, targets)
        logger.info(
            "Sampling mode: %s total samples distributed across %s strata",
            total_samples,
            len(targets),
        )

    if getattr(config, "resume", False):
        existing_counts = _existing_target_counts(metadata)
    else:
        existing_counts = {}

    logger.info("Step 1: Sampling global patch targets...")
    print("\nSampling water patches across the world")

    current_total = len(metadata.metadata.get("samples", []))
    attempts = 0
    max_attempts = max(10000, total_samples * 200)
    remaining_counts = _remaining_target_counts(target_counts, existing_counts, targets)
    rng = np.random.default_rng()
    run_start = perf_counter()

    with tqdm(total=total_samples, initial=current_total, desc="Acquiring paired samples") as pbar:
        while current_total < total_samples and attempts < max_attempts:
            open_targets = _open_targets(remaining_counts)
            if not open_targets:
                break

            target = open_targets[int(rng.integers(0, len(open_targets)))]
            logger.info(
                "Targeting %s/%s/%s: need %s samples",
                target.environment_class,
                target.depth_class,
                target.turbidity_class,
                remaining_counts[target],
            )

            attempts += 1
            attempt_start = perf_counter()
            candidate = sampler.propose_candidate_for_target(target)
            candidate_seconds = perf_counter() - attempt_start
            if candidate is None:
                logger.debug(
                    "Sampler returned no candidate for target %s/%s/%s after %.2fs",
                    target.environment_class,
                    target.depth_class,
                    target.turbidity_class,
                    candidate_seconds,
                )
                continue

            logger.debug(
                "Candidate search for target %s/%s/%s took %.2fs at (%.6f, %.6f)",
                target.environment_class,
                target.depth_class,
                target.turbidity_class,
                candidate_seconds,
                candidate["latitude"],
                candidate["longitude"],
            )

            planet_start = perf_counter()
            planetary_candidates = planetscope_manager.retrieve_images(
                candidate["latitude"],
                candidate["longitude"],
                PLANETSCOPE_ARCHIVE_START,
                PLANETSCOPE_ARCHIVE_END,
                8,
            )
            planet_seconds = perf_counter() - planet_start
            if not planetary_candidates:
                logger.debug(
                    "No PlanetScope catalog candidates at (%.6f, %.6f) for target %s/%s/%s after %.2fs",
                    candidate["latitude"],
                    candidate["longitude"],
                    target.environment_class,
                    target.depth_class,
                    target.turbidity_class,
                    planet_seconds,
                )
                continue

            accepted = False
            for ps_image in planetary_candidates:
                ps_date = str(ps_image.get("date", ""))[:10]
                if len(ps_date) != 10:
                    continue

                ps_day = date.fromisoformat(ps_date)
                s2_window_start = (ps_day - timedelta(days=SENTINEL2_WINDOW_DAYS)).isoformat()
                s2_window_end = (ps_day + timedelta(days=SENTINEL2_WINDOW_DAYS)).isoformat()

                s2_start = perf_counter()
                s2_images = sentinel2_manager.retrieve_images(
                    candidate["latitude"],
                    candidate["longitude"],
                    s2_window_start,
                    s2_window_end,
                    IMAGES_PER_LOCATION,
                    aoi_geometry=sampler.patch_geojson(candidate["patch_polygon"]),
                )
                s2_seconds = perf_counter() - s2_start
                if len(s2_images) < SENTINEL2_MIN_IMAGES:
                    logger.debug(
                        "Insufficient Sentinel-2 images (%s) for location (%.6f, %.6f); need %s after %.2fs",
                        len(s2_images),
                        candidate["latitude"],
                        candidate["longitude"],
                        SENTINEL2_MIN_IMAGES,
                        s2_seconds,
                    )
                    continue

                turbidity_start = perf_counter()
                turbidity_info = sentinel2_manager.estimate_turbidity_class(
                    candidate["latitude"],
                    candidate["longitude"],
                    s2_window_start,
                    s2_window_end,
                    buffer_meters=SENTINEL2_TURBIDITY_BUFFER_METERS,
                )
                turbidity_seconds = perf_counter() - turbidity_start
                if not turbidity_info:
                    logger.debug(
                        "Turbidity estimation failed for location (%.6f, %.6f) after %.2fs",
                        candidate["latitude"],
                        candidate["longitude"],
                        turbidity_seconds,
                    )
                    continue

                matched_target = _find_matching_target(
                    candidate["environment_class"],
                    candidate["depth_class"],
                    turbidity_info["turbidity_class"],
                    remaining_counts,
                )
                if matched_target is None:
                    logger.debug(
                        "Candidate at (%.6f, %.6f) fits no open stratum: env=%s depth=%s turbidity=%s",
                        candidate["latitude"],
                        candidate["longitude"],
                        candidate["environment_class"],
                        candidate["depth_class"],
                        turbidity_info.get("turbidity_class"),
                    )
                    continue

                sample = {
                    "location_id": current_total,
                    "latitude": candidate["latitude"],
                    "longitude": candidate["longitude"],
                    "season_id": _season_id_from_month(ps_day.month),
                    "province": "global",
                    "environment_class": matched_target.environment_class,
                    "depth_class": matched_target.depth_class,
                    "depth_m": candidate["depth_m"],
                    "turbidity_class": matched_target.turbidity_class,
                    "turbidity_index": turbidity_info["turbidity_index"],
                    "date_range": (s2_window_start, s2_window_end),
                    "sentinel2_images": s2_images,
                    "planetscope_images": [ps_image],
                    "alignment_crs": candidate["alignment_crs"],
                    "patch_size_pixels": candidate["patch_size_pixels"],
                    "patch_size_meters": candidate["patch_size_meters"],
                    "target_origin_x": candidate["target_origin_x"],
                    "target_origin_y": candidate["target_origin_y"],
                    "planet_item_ids": [ps_image["id"]],
                    "planet_order_id": "",
                    "planet_product_bundle": PLANETSCOPE_PRODUCT_BUNDLE,
                    "planet_aoi_geojson": sampler.patch_geojson(candidate["patch_polygon"]),
                }
                metadata.add_sample(sample)
                current_total += 1
                remaining_counts[matched_target] -= 1
                accepted = True
                pbar.update(1)
                metadata.update_dataset_sample_count(current_total)
                metadata.save_checkpoint()
                logger.info(
                    "Accepted candidate at (%.6f, %.6f) into %s/%s/%s; remaining=%s; timings candidate=%.2fs planet=%.2fs s2=%.2fs turbidity=%.2fs total_run=%.2fs",
                    candidate["latitude"],
                    candidate["longitude"],
                    matched_target.environment_class,
                    matched_target.depth_class,
                    matched_target.turbidity_class,
                    remaining_counts[matched_target],
                    candidate_seconds,
                    planet_seconds,
                    s2_seconds,
                    turbidity_seconds,
                    perf_counter() - run_start,
                )
                break

            if not accepted:
                continue

    if current_total < total_samples:
        raise ValueError(
            f"Could only obtain {current_total}/{total_samples} valid paired samples after {attempts} attempts."
        )

    metadata.update_dataset_sample_count(current_total)

    logger.info("Step 2: Saving metadata...")
    json_file, csv_file = metadata.save_checkpoint()

    print("=" * 60)
    print("Dataset creation complete!")
    print(f"Output directory: {config.output_dir}")
    print(f"Paired location samples: {current_total}")
    print(f"  Environment classes: {', '.join(ENVIRONMENT_CLASSES)}")
    print(f"  Depth classes: {', '.join(DEPTH_CLASSES)}")
    print(f"  Turbidity classes: {', '.join(TURBIDITY_CLASSES)}")
    print(f"  Patch size: {PATCH_SIZE_PIXELS}x{PATCH_SIZE_PIXELS} pixels")
    print(f"  Planet product bundle: {PLANETSCOPE_PRODUCT_BUNDLE}")
    print(f"\nOutputs:")
    print(f"  Metadata JSON: {json_file}")
    print(f"  Manifest CSV: {csv_file}")
    print("=" * 60)

    logger.info("Dataset creation complete: %s samples", current_total)
    logger.info("Total dataset creation runtime: %.2fs", perf_counter() - run_start)


if __name__ == "__main__":
    create_dataset()
