"""
Rebuild dataset_metadata.json and dataset_manifest.csv from individual
sample_metadata.json files on disk.

Use this when the dataset-level metadata has been overwritten but the
sample directories are still intact.

Usage:
    python -m superres.rebuild_metadata <metadata_dir> <data_dir>

Example:
    python -m superres.rebuild_metadata data/landsat2sentinel/metadata \
        data/landsat2sentinel/data
"""

from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

from superres.constants import MANIFEST_COLUMNS
from superres.metadata import DatasetMetadata


def rebuild_metadata(metadata_dir: Path, data_dir: Path, dry_run: bool = False):
    sample_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("sample_")
    )
    print(f"Found {len(sample_dirs)} sample directories in {data_dir}")

    samples = []
    skipped = []
    for sample_dir in sample_dirs:
        meta_file = sample_dir / "sample_metadata.json"
        if not meta_file.exists():
            skipped.append(sample_dir.name)
            continue
        try:
            samples.append(json.loads(meta_file.read_text()))
        except Exception as exc:
            skipped.append(f"{sample_dir.name} ({exc})")

    samples.sort(key=lambda s: int(s["location_id"]))
    print(f"Loaded {len(samples)} sample metadata files ({len(skipped)} skipped)")
    if skipped:
        print(f"  Skipped: {skipped}")

    if dry_run:
        print("\nDry run — nothing written.")
        return

    # Preserve existing dataset_info if available, just fix the sample count.
    meta_file = metadata_dir / "dataset_metadata.json"
    if meta_file.exists():
        existing = DatasetMetadata.load_json(metadata_dir)
        dataset_info = existing.metadata.get("dataset_info", {})
    else:
        dataset_info = {}

    dataset_info["total_samples"] = len(samples)

    dm = DatasetMetadata(metadata_dir)
    dm.metadata["dataset_info"] = dataset_info
    dm.metadata["samples"] = samples
    dm.save_json()
    print(f"Wrote {meta_file} ({len(samples)} samples)")

    # Write CSV directly from sample fields so column names match exactly.
    csv_path = metadata_dir / "dataset_manifest.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for s in samples:
            writer.writerow({col: s.get(col, "") for col in MANIFEST_COLUMNS})
    print(f"Wrote {csv_path} ({len(samples)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("metadata_dir", help="Path to metadata directory")
    parser.add_argument("data_dir", help="Path to data directory (contains sample_* folders)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written without writing")
    args = parser.parse_args()

    rebuild_metadata(Path(args.metadata_dir), Path(args.data_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
