"""
Post-process downloaded super-resolution samples.

Run this after download_and_preprocess.py and before tile_dataset.py. For each
sample it:

1. Reprojects every low-res image in the stack onto a grid anchored at the
   same CRS/origin as the sample's high-res target (at the low-res
   satellite's own native resolution), so the stack is pixel-registered
   against the high-res target. The high-res raster is already aligned to
   this grid by download_and_preprocess.py's standardize_highres_patch().
2. Rescales every band from raw sensor digital numbers to physical surface
   reflectance using each satellite's documented scale/offset
   (REFLECTANCE_SCALE_OFFSET / PLANETSCOPE_REFLECTANCE_SCALE in constants.py),
   so low-res and high-res values are on a common, comparable scale.
   Classification/QA bands (e.g. SCL, QA_PIXEL) aren't reflectance, so they
   are left unscaled and resampled with nearest-neighbor instead of bilinear.

Output mirrors the input sample_*/<role>/ layout under a separate output
directory, leaving the raw/standardized originals untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from .constants import PLANETSCOPE_REFLECTANCE_SCALE, REFLECTANCE_SCALE_OFFSET
from .download_and_preprocess import _highres_standardized_file
from .satellites import HIGHRES_SATELLITES, LOWRES_SATELLITES
from common.dataset_utils import ensure_directory, setup_logger

PASSTHROUGH_SCALE_OFFSET = (1.0, 0.0)


def _band_scale_offsets(satellite_key: str, n_bands: int, logger) -> List[Tuple[float, float]]:
    """Per-band (scale, offset) to convert raw DN to surface reflectance for *satellite_key*.

    Bands without a documented reflectance scale (classification/QA masks) pass
    through unscaled via PASSTHROUGH_SCALE_OFFSET.
    """
    if satellite_key == "planetscope":
        return [(PLANETSCOPE_REFLECTANCE_SCALE, 0.0)] * n_bands

    manager_cls = LOWRES_SATELLITES.get(satellite_key) or HIGHRES_SATELLITES.get(satellite_key)
    band_names = manager_cls.SPEC.band_names if manager_cls else []
    if len(band_names) != n_bands:
        logger.warning(
            f"Band count mismatch for '{satellite_key}': raster has {n_bands} bands, "
            f"SPEC declares {len(band_names)}. Leaving all bands unscaled."
        )
        return [PASSTHROUGH_SCALE_OFFSET] * n_bands

    return [REFLECTANCE_SCALE_OFFSET.get(name, PASSTHROUGH_SCALE_OFFSET) for name in band_names]


def _lowres_grid_transform(origin_x: float, origin_y: float, patch_size_meters: float, pixel_size: float):
    """Build a pixel grid, anchored at (origin_x, origin_y), that fully covers
    patch_size_meters at the low-res satellite's native pixel_size.

    patch_size_meters need not be an exact multiple of pixel_size (e.g. a 5120m
    patch at 30m Landsat resolution), so the grid is rounded up to the next
    whole pixel - it may extend slightly beyond the high-res patch footprint,
    never short of it.
    """
    n_pixels = int(np.ceil(patch_size_meters / pixel_size))
    transform = from_origin(origin_x, origin_y + n_pixels * pixel_size, pixel_size, pixel_size)
    return transform, n_pixels


def _reproject_bands(source_data, indices, src_transform, src_crs, dst_transform, dst_crs, n_pixels, resampling):
    if not indices:
        return None
    destination = np.full((len(indices), n_pixels, n_pixels), np.nan, dtype=np.float32)
    reproject(
        source_data[indices],
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return destination


def postprocess_lowres_image(
    source_file: Path,
    output_file: Path,
    satellite_key: str,
    dst_transform,
    dst_crs: str,
    n_pixels: int,
    logger,
) -> bool:
    """Reproject one low-res raster onto the shared low-res grid and rescale it to reflectance."""
    try:
        with rasterio.open(source_file) as src:
            scale_offsets = _band_scale_offsets(satellite_key, src.count, logger)
            source_data = src.read().astype(np.float32)
            for band_index, (scale, offset) in enumerate(scale_offsets):
                source_data[band_index] = source_data[band_index] * scale + offset

            # Classification/QA bands (passthrough scale) shouldn't be bilinearly
            # blended with neighboring pixels - their values are codes, not continuous
            # quantities - so they're reprojected separately with nearest-neighbor.
            reflectance_bands = [i for i, so in enumerate(scale_offsets) if so != PASSTHROUGH_SCALE_OFFSET]
            categorical_bands = [i for i, so in enumerate(scale_offsets) if so == PASSTHROUGH_SCALE_OFFSET]

            dst_array = np.full((src.count, n_pixels, n_pixels), np.nan, dtype=np.float32)
            reflectance_out = _reproject_bands(
                source_data, reflectance_bands, src.transform, src.crs,
                dst_transform, dst_crs, n_pixels, Resampling.bilinear,
            )
            if reflectance_out is not None:
                dst_array[reflectance_bands] = reflectance_out
            categorical_out = _reproject_bands(
                source_data, categorical_bands, src.transform, src.crs,
                dst_transform, dst_crs, n_pixels, Resampling.nearest,
            )
            if categorical_out is not None:
                dst_array[categorical_bands] = categorical_out

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                dtype="float32",
                height=n_pixels,
                width=n_pixels,
                transform=dst_transform,
                crs=dst_crs,
                nodata=np.nan,
                compress="lzw",
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(dst_array)

        return True
    except Exception as exc:
        logger.error(f"Failed to postprocess low-res image {source_file}: {exc}")
        return False


def postprocess_highres_image(source_file: Path, output_file: Path, satellite_key: str, logger) -> bool:
    """Rescale an already grid-aligned high-res raster to surface reflectance."""
    try:
        with rasterio.open(source_file) as src:
            scale_offsets = _band_scale_offsets(satellite_key, src.count, logger)
            data = src.read().astype(np.float32)
            for band_index, (scale, offset) in enumerate(scale_offsets):
                data[band_index] = data[band_index] * scale + offset

            profile = src.profile.copy()
            profile.update(driver="GTiff", dtype="float32", compress="lzw")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(data)

        return True
    except Exception as exc:
        logger.error(f"Failed to postprocess high-res image {source_file}: {exc}")
        return False


def postprocess_sample(sample_dir: Path, output_dir: Path, logger, overwrite: bool = False) -> bool:
    """Postprocess the low-res stack and high-res target for one sample directory."""
    metadata_file = sample_dir / "sample_metadata.json"
    if not metadata_file.exists():
        logger.warning(f"Skipping {sample_dir.name}: no sample_metadata.json (incomplete sample)")
        return False

    with open(metadata_file) as file_handle:
        sample_metadata = json.load(file_handle)

    lowres_key = sample_metadata["lowres_satellite"]
    highres_key = sample_metadata["highres_satellite"]
    alignment_crs = sample_metadata["alignment_crs"]
    target_origin_x = float(sample_metadata["target_origin_x"])
    target_origin_y = float(sample_metadata["target_origin_y"])
    patch_size_meters = float(sample_metadata["patch_size_meters"])
    location_id = int(sample_metadata["location_id"])

    sample_out_dir = output_dir / sample_dir.name
    ok = True

    lowres_dir = sample_dir / lowres_key
    lowres_files = sorted(lowres_dir.glob(f"{lowres_key}_*.tif")) if lowres_dir.exists() else []
    if lowres_files:
        manager_cls = LOWRES_SATELLITES[lowres_key]
        pixel_size = manager_cls.resolution_meters()
        dst_transform, n_pixels = _lowres_grid_transform(
            target_origin_x, target_origin_y, patch_size_meters, pixel_size
        )
        lowres_out_dir = ensure_directory(sample_out_dir / lowres_key)
        for source_file in lowres_files:
            output_file = lowres_out_dir / source_file.name
            if output_file.exists() and not overwrite:
                continue
            if not postprocess_lowres_image(
                source_file, output_file, lowres_key, dst_transform, alignment_crs, n_pixels, logger
            ):
                ok = False
    else:
        logger.warning(f"Sample {sample_dir.name}: no {lowres_key} images found, skipping low-res postprocessing")

    highres_file = _highres_standardized_file(sample_dir, location_id, highres_key)
    if highres_file.exists():
        highres_out_dir = ensure_directory(sample_out_dir / highres_key)
        output_file = highres_out_dir / highres_file.name
        if not (output_file.exists() and not overwrite):
            if not postprocess_highres_image(highres_file, output_file, highres_key, logger):
                ok = False
    else:
        logger.warning(f"Sample {sample_dir.name}: no standardized {highres_key} target found")
        ok = False

    return ok


def postprocess_dataset():
    parser = argparse.ArgumentParser(
        description="Align low-res image stacks to the high-res grid and rescale all rasters to surface reflectance"
    )
    parser.add_argument("data_dir", help="Path to dataset data directory (contains sample_* folders)")
    parser.add_argument("-o", "--output", default=None, help="Output directory for postprocessed rasters (default: <data_dir>/../processed)")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess samples even if output files already exist")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        exit(1)

    output_dir = ensure_directory(Path(args.output) if args.output else data_dir.parent / "processed")
    logger = setup_logger("PostprocessDataset", output_dir / "postprocess.log")
    logger.info(f"Starting postprocessing from {data_dir}")

    sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("sample_"))

    successes = 0
    for sample_dir in tqdm(sample_dirs, desc="Postprocessing samples"):
        if postprocess_sample(sample_dir, output_dir, logger, overwrite=args.overwrite):
            successes += 1

    logger.info("Postprocessed %s/%s samples", successes, len(sample_dirs))
    print(f"Postprocessed {successes}/{len(sample_dirs)} samples")


if __name__ == "__main__":
    postprocess_dataset()
