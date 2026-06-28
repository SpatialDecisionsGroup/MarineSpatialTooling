"""
Multi-image super-resolution dataset creator for global water patches.
"""

from __future__ import annotations

import logging
import shutil
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from time import perf_counter

import numpy as np
from tqdm import tqdm

from common.dataset_utils import setup_logger

from .config import config_create
from .constants import (
    DEPTH_CLASSES,
    ENVIRONMENT_CLASSES,
    LOWRES_MAX_IMAGES,
    LOWRES_MIN_IMAGES,
    PATCH_SIZE_PIXELS,
    PLANETSCOPE_PRODUCT_BUNDLE,
    TURBIDITY_BUFFER_METERS,
    TURBIDITY_CLASSES,
)
from .gee_satellite import GEESatelliteManager
from .metadata import DatasetMetadata
from .satellites import build_highres_manager, build_lowres_manager
from .world_sampling import SampleTarget, WorldPatchSampler


HIGHRES_SEARCH_START = "2016-01-01"
HIGHRES_SEARCH_END = date.today().isoformat()


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


def _sort_by_quality(images: List[Dict]) -> List[Dict]:
    """Sort image candidates best-first (lowest cloud cover, then earliest date)."""
    def sort_key(image):
        try:
            cloud_cover = float(image.get("cloud_cover", 100) or 100)
        except (TypeError, ValueError):
            cloud_cover = 100.0
        return (cloud_cover, str(image.get("date", "")))

    return sorted(images, key=sort_key)


def _build_highres_identifier_fields(highres_manager, highres_image: Dict, aoi_geojson: Dict) -> Dict:
    """Build the highres-specific sample fields, branching on the provider's download mechanism."""
    if isinstance(highres_manager, GEESatelliteManager):
        return {
            "highres_item_ids": [highres_image["asset_id"]],
            "highres_order_id": "",
            "highres_product_bundle": "",
            "highres_aoi_geojson": aoi_geojson,
        }
    return {
        "highres_item_ids": [highres_image["id"]],
        "highres_order_id": "",
        "highres_product_bundle": PLANETSCOPE_PRODUCT_BUNDLE,
        "highres_aoi_geojson": aoi_geojson,
    }


def _repair_broken_samples(metadata: DatasetMetadata, data_dir: Path, logger) -> List[int]:
    """Drop samples whose download partially failed and return their freed location_ids.

    A sample is "broken" if download_and_preprocess.py already created its data_dir/sample_<id>
    folder (i.e. a download was attempted) but never wrote sample_metadata.json - the checkpoint
    it only writes once both the low-res stack and high-res image succeed. The most common cause
    is the low-res catalog query at download time turning up fewer images than the one that
    validated this sample at creation time (see download_lowres_images' aoi_geometry handling).
    Samples that were never attempted (no data_dir folder yet) are left untouched.
    """
    samples = metadata.metadata.get("samples", [])
    broken, good = [], []
    for sample in samples:
        sample_dir = data_dir / f"sample_{int(sample['location_id']):06d}"
        if sample_dir.exists() and not (sample_dir / "sample_metadata.json").exists():
            broken.append(sample)
        else:
            good.append(sample)

    if not broken:
        logger.info("Check: no incomplete samples found among %s existing samples", len(samples))
        print("Check: no incomplete samples found.")
        return []

    for sample in broken:
        location_id = int(sample["location_id"])
        sample_dir = data_dir / f"sample_{location_id:06d}"
        logger.info(
            "Check: sample %s (%s/%s/%s) has a partial download with no sample_metadata.json "
            "checkpoint - deleting %s and regenerating a replacement",
            location_id,
            sample.get("environment_class"),
            sample.get("depth_class"),
            sample.get("turbidity_class"),
            sample_dir,
        )
        shutil.rmtree(sample_dir, ignore_errors=True)

    metadata.metadata["samples"] = good
    freed_ids = sorted(int(sample["location_id"]) for sample in broken)
    logger.info("Check: removed %s incomplete sample(s); will regenerate ids %s", len(freed_ids), freed_ids)
    print(f"Check: found {len(freed_ids)} incomplete sample(s) on disk; regenerating: {freed_ids}")
    return freed_ids


def create_dataset():
    """Create the complete world-scale super-resolution dataset."""
    config = config_create()
    metadata_path = config.metadata_dir / "dataset_metadata.json"
    if getattr(config, "resume", False) and metadata_path.exists():
        metadata = DatasetMetadata.load_json(config.metadata_dir)
    else:
        metadata = DatasetMetadata(config.metadata_dir)

    logger = setup_logger("SRDatasetCreator", config.metadata_dir / "creation.log", level=logging.DEBUG)

    freed_location_ids: List[int] = []
    if getattr(config, "check_existing", False):
        freed_location_ids = _repair_broken_samples(metadata, config.data_dir, logger)

    lowres_manager = build_lowres_manager(config)
    highres_manager = build_highres_manager(config)

    patch_size_meters = PATCH_SIZE_PIXELS * highres_manager.resolution_meters()
    sampler = WorldPatchSampler(
        config.coastline_dir,
        config.gebco_file,
        logger,
        turbidity_raster=getattr(config, "turbidity_file", None),
        patch_size_meters=patch_size_meters,
    )

    metadata.add_dataset_info(
        {
            "lowres_satellite": config.lowres_satellite,
            "lowres_resolution_meters": lowres_manager.resolution_meters(),
            "highres_satellite": config.highres_satellite,
            "highres_resolution_meters": highres_manager.resolution_meters(),
        }
    )

    print("=" * 60)
    print("Creating Super-Resolution Dataset")
    print("=" * 60)
    print(f"Low-res satellite: {config.lowres_satellite} ({lowres_manager.resolution_meters():.0f} m)")
    print(f"High-res satellite: {config.highres_satellite} ({highres_manager.resolution_meters():.0f} m)")
    print(f"Patch size: {PATCH_SIZE_PIXELS} x {PATCH_SIZE_PIXELS} pixels")
    print(f"Ground size: {patch_size_meters:.0f} m x {patch_size_meters:.0f} m")

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

    # New samples normally get the next sequential id, but ids freed by _repair_broken_samples
    # are reused first so a regenerated sample lands back in its original sample_<id> slot
    # instead of colliding with (or appending after) the ids of untouched existing samples.
    existing_location_ids = {int(sample["location_id"]) for sample in metadata.metadata.get("samples", [])}
    free_location_ids = deque(sorted(set(freed_location_ids) - existing_location_ids))
    next_new_location_id = max(existing_location_ids | set(freed_location_ids), default=-1) + 1

    def _allocate_location_id() -> int:
        nonlocal next_new_location_id
        if free_location_ids:
            return free_location_ids.popleft()
        value = next_new_location_id
        next_new_location_id += 1
        return value

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

            aoi_geojson = sampler.patch_geojson(candidate["patch_polygon"])

            highres_start = perf_counter()
            highres_candidates = _sort_by_quality(
                highres_manager.retrieve_images(
                    candidate["latitude"],
                    candidate["longitude"],
                    HIGHRES_SEARCH_START,
                    HIGHRES_SEARCH_END,
                    8,
                )
            )
            highres_seconds = perf_counter() - highres_start
            if not highres_candidates:
                logger.debug(
                    "No high-res catalog candidates at (%.6f, %.6f) for target %s/%s/%s after %.2fs",
                    candidate["latitude"],
                    candidate["longitude"],
                    target.environment_class,
                    target.depth_class,
                    target.turbidity_class,
                    highres_seconds,
                )
                continue

            accepted = False
            for highres_image in highres_candidates:
                highres_date_str = str(highres_image.get("date", ""))[:10]
                if len(highres_date_str) != 10:
                    continue

                highres_day = date.fromisoformat(highres_date_str)
                lowres_window_start = (highres_day - timedelta(days=config.lowres_window_days)).isoformat()
                lowres_window_end = (highres_day + timedelta(days=config.lowres_window_days)).isoformat()

                lowres_start = perf_counter()
                lowres_images = lowres_manager.retrieve_images(
                    candidate["latitude"],
                    candidate["longitude"],
                    lowres_window_start,
                    lowres_window_end,
                    LOWRES_MAX_IMAGES,
                    aoi_geometry=aoi_geojson,
                )
                lowres_seconds = perf_counter() - lowres_start
                if len(lowres_images) < LOWRES_MIN_IMAGES:
                    logger.debug(
                        "Insufficient low-res images (%s) for location (%.6f, %.6f); need %s after %.2fs",
                        len(lowres_images),
                        candidate["latitude"],
                        candidate["longitude"],
                        LOWRES_MIN_IMAGES,
                        lowres_seconds,
                    )
                    break

                turbidity_start = perf_counter()
                turbidity_info = lowres_manager.estimate_turbidity_class(
                    candidate["latitude"],
                    candidate["longitude"],
                    lowres_window_start,
                    lowres_window_end,
                    buffer_meters=TURBIDITY_BUFFER_METERS,
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
                    break

                sample = {
                    "location_id": _allocate_location_id(),
                    "latitude": candidate["latitude"],
                    "longitude": candidate["longitude"],
                    "season_id": _season_id_from_month(highres_day.month),
                    "province": "global",
                    "environment_class": matched_target.environment_class,
                    "depth_class": matched_target.depth_class,
                    "depth_m": candidate["depth_m"],
                    "turbidity_class": matched_target.turbidity_class,
                    "turbidity_index": turbidity_info["turbidity_index"],
                    "date_range": (lowres_window_start, lowres_window_end),
                    "lowres_satellite": config.lowres_satellite,
                    "lowres_images": lowres_images,
                    "highres_satellite": config.highres_satellite,
                    "highres_images": [highres_image],
                    "alignment_crs": candidate["alignment_crs"],
                    "patch_size_pixels": candidate["patch_size_pixels"],
                    "patch_size_meters": candidate["patch_size_meters"],
                    "target_origin_x": candidate["target_origin_x"],
                    "target_origin_y": candidate["target_origin_y"],
                    **_build_highres_identifier_fields(highres_manager, highres_image, aoi_geojson),
                }
                metadata.add_sample(sample)
                current_total += 1
                remaining_counts[matched_target] -= 1
                accepted = True
                pbar.update(1)
                metadata.update_dataset_sample_count(current_total)
                metadata.save_checkpoint()
                logger.info(
                    "Accepted candidate at (%.6f, %.6f) into %s/%s/%s; remaining=%s; timings candidate=%.2fs highres=%.2fs lowres=%.2fs turbidity=%.2fs total_run=%.2fs",
                    candidate["latitude"],
                    candidate["longitude"],
                    matched_target.environment_class,
                    matched_target.depth_class,
                    matched_target.turbidity_class,
                    remaining_counts[matched_target],
                    candidate_seconds,
                    highres_seconds,
                    lowres_seconds,
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
    print(f"  Low-res satellite: {config.lowres_satellite}")
    print(f"  High-res satellite: {config.highres_satellite}")
    print(f"\nOutputs:")
    print(f"  Metadata JSON: {json_file}")
    print(f"  Manifest CSV: {csv_file}")
    print("=" * 60)

    logger.info("Dataset creation complete: %s samples", current_total)
    logger.info("Total dataset creation runtime: %.2fs", perf_counter() - run_start)


if __name__ == "__main__":
    create_dataset()
