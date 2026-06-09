"""
Tile downloaded dataset scenes into smaller patches for training.

This script reads downloaded GeoTIFF files and tiles them into 512x512 patches.
Run this after download_and_preprocess.py has finished downloading and aligning all scenes.
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from .constants import TILE_SIZE_PIXELS
from common.dataset_utils import ensure_directory, setup_logger


def tile_raster(input_file, output_dir, tile_size=TILE_SIZE_PIXELS, prefix="tile"):
    """Tile a GeoTIFF into non-overlapping patches.
    
    Args:
        input_file: Path to source GeoTIFF
        output_dir: Directory to save tiles
        tile_size: Tile size in pixels
        prefix: Prefix for output filenames
    
    Returns:
        Number of tiles created
    """
    try:
        with rasterio.open(input_file) as src:
            width, height = src.width, src.height
            profile = src.profile.copy()
            
            tile_count = 0
            for row_start in range(0, height, tile_size):
                for col_start in range(0, width, tile_size):
                    # Determine tile dimensions (handle edges)
                    row_end = min(row_start + tile_size, height)
                    col_end = min(col_start + tile_size, width)
                    tile_h = row_end - row_start
                    tile_w = col_end - col_start
                    
                    # Skip incomplete tiles at edges
                    if tile_h < tile_size or tile_w < tile_size:
                        continue
                    
                    # Read tile
                    window = Window(col_start, row_start, tile_w, tile_h)
                    data = src.read(window=window)
                    
                    # Skip tiles that are mostly NoData
                    if np.all(data == src.nodata):
                        continue
                    
                    # Write tile
                    output_file = output_dir / f"{prefix}_{row_start}_{col_start}.tif"
                    profile.update(
                        width=tile_w,
                        height=tile_h,
                        transform=rasterio.windows.transform(window, src.transform),
					)
                    with rasterio.open(output_file, "w", **profile) as dst:
                        dst.write(data)

                    tile_count += 1
            
            return tile_count
    
    except Exception as exc:
        print(f"Error tiling {input_file}: {exc}")
        return 0


def tile_dataset():
    parser = argparse.ArgumentParser(description="Tile downloaded dataset into training patches")
    parser.add_argument("data_dir", help="Path to dataset data directory (contains sample_* folders)")
    parser.add_argument("-o", "--output", default="./superres/output", help="Output directory for tiled patches")
    parser.add_argument("-t", "--tile-size", type=int, default=TILE_SIZE_PIXELS, help=f"Tile size in pixels (default {TILE_SIZE_PIXELS})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        exit(1)

    output_dir = Path(args.output)

    logger = setup_logger("TileDataset", output_dir / "tiling.log")
    logger.info(f"Starting dataset tiling from {data_dir}")
    
    tile_output = ensure_directory(output_dir / "tiles")
    sentinel2_tile_dir = ensure_directory(tile_output / "sentinel2")
    planetscope_tile_dir = ensure_directory(tile_output / "planetscope")
    
    s2_tile_count = 0
    ps_tile_count = 0
    
    sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    with tqdm(sample_dirs, desc="Tiling samples") as pbar:
        for sample_dir in pbar:
            location_id = sample_dir.name.replace("sample_", "")
            
            # Tile Sentinel-2 images
            s2_dir = sample_dir / "sentinel2"
            if s2_dir.exists():
                s2_files = sorted(s2_dir.glob("*.tif"))
                for s2_file in s2_files:
                    img_name = s2_file.stem
                    s2_sample_dir = ensure_directory(sentinel2_tile_dir / f"location_{location_id}")
                    count = tile_raster(s2_file, s2_sample_dir, args.tile_size, f"s2_{img_name}")
                    s2_tile_count += count
            
            # Tile PlanetScope images
            ps_dir = sample_dir / "planetscope"
            if ps_dir.exists():
                ps_files = sorted(ps_dir.glob("*.tif"))
                for ps_file in ps_files:
                    img_name = ps_file.stem
                    ps_sample_dir = ensure_directory(planetscope_tile_dir / f"location_{location_id}")
                    count = tile_raster(ps_file, ps_sample_dir, args.tile_size, f"ps_{img_name}")
                    ps_tile_count += count
            
            pbar.update(1)
    
    print("\n" + "=" * 60)
    print("Tiling complete!")
    print(f"Sentinel-2 tiles: {s2_tile_count}")
    print(f"PlanetScope tiles: {ps_tile_count}")
    print(f"Output directory: {tile_output}")
    print("=" * 60)
    
    logger.info(f"Tiling complete: {s2_tile_count} S2 tiles, {ps_tile_count} PS tiles")
