"""
Remove specific sample entries from the dataset metadata and delete their
on-disk data directories, so that `create --resume` can regenerate them.

Usage:
    python superres/drop_samples.py <metadata_dir> <data_dir> <location_id> [<location_id> ...]
    python superres/drop_samples.py <metadata_dir> <data_dir> --from-report <check_report.json>

Example:
    python superres/drop_samples.py data/landsat2sentinel/metadata \
        data/landsat2sentinel/data  31 285 516 610 722 798 1112 1342 1462 1521 1559
    python superres/drop_samples.py superres/sentinel2planet superres/sentinel2planet/data \
        --from-report /tmp/issues.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from superres.metadata import DatasetMetadata


def drop_samples(metadata_dir: Path, data_dir: Path, location_ids: list[int], dry_run: bool = False):
    meta_file = metadata_dir / "dataset_metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Not found: {meta_file}")

    metadata = DatasetMetadata.load_json(metadata_dir)
    samples = metadata.metadata.get("samples", [])

    drop_set = set(location_ids)
    kept = [s for s in samples if int(s["location_id"]) not in drop_set]
    dropped = [s for s in samples if int(s["location_id"]) in drop_set]

    missing = drop_set - {int(s["location_id"]) for s in samples}
    if missing:
        print(f"Warning: these location_ids not found in metadata and will be ignored: {sorted(missing)}")

    print(f"Dropping {len(dropped)} sample(s) from metadata ({len(kept)} remaining):")
    for s in sorted(dropped, key=lambda x: int(x["location_id"])):
        loc = int(s["location_id"])
        sample_dir = data_dir / f"sample_{loc:06d}"
        print(f"  location_id={loc:6d}  env={s.get('environment_class','?')}/"
              f"{s.get('depth_class','?')}/{s.get('turbidity_class','?')}  "
              f"dir_exists={sample_dir.exists()}")

    if dry_run:
        print("\nDry run — nothing changed. Re-run without --dry-run to apply.")
        return

    # Update metadata and regenerate CSV
    metadata.metadata["samples"] = kept
    metadata.metadata["dataset_info"]["total_samples"] = len(kept)
    metadata.save_checkpoint()
    print(f"\nUpdated {meta_file.name}: {len(kept)} samples, regenerated manifest CSV.")

    # Delete sample directories so create can reuse those ids cleanly
    deleted = []
    for s in dropped:
        loc = int(s["location_id"])
        sample_dir = data_dir / f"sample_{loc:06d}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
            deleted.append(sample_dir.name)

    if deleted:
        print(f"Deleted {len(deleted)} sample director(ies): {deleted}")
    else:
        print("No sample directories found on disk to delete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata_dir", help="Path to metadata directory (contains dataset_metadata.json)")
    parser.add_argument("data_dir", help="Path to data directory (contains sample_* folders)")
    parser.add_argument("location_ids", nargs="*", type=int, help="location_id values to remove")
    parser.add_argument("--from-report", metavar="JSON", help="JSON report produced by check_dataset --output; drops all samples with issues")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without making changes")
    args = parser.parse_args()

    location_ids = list(args.location_ids)

    if args.from_report:
        report = json.loads(Path(args.from_report).read_text())
        from_report = {
            int(i["sample"].removeprefix("sample_"))
            for i in report.get("issues", [])
        }
        print(f"Found {len(from_report)} location_id(s) with issues in {args.from_report}")
        location_ids = sorted(set(location_ids) | from_report)

    if not location_ids:
        parser.error("provide at least one location_id or --from-report")

    drop_samples(
        Path(args.metadata_dir),
        Path(args.data_dir),
        location_ids,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
