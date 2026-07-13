"""
Apply review decisions from the Streamlit reviewer to the dataset.

Reads review_decisions.json (written by review_dataset.py) and acts on each
flagged sample:

  keep         — no change
  delete       — remove sample dir + remove from metadata so create --resume
                 can allocate a replacement location
  replace_lr   — delete the Landsat sub-directory (or specific files if the
                 decision names them), then remove the sample_metadata.json
                 checkpoint so download will re-acquire LR images
  replace_hr   — delete the sentinel2 sub-directory and checkpoint
  replace_both — delete the entire sample (both dirs + metadata entry), the same
                 location would yield the same bad imagery so create --resume must
                 pick a new one

After running this script:
  • Re-run create --resume  to fill slots freed by delete/replace_both with new locations
  • Re-run download         to re-try replace_lr/replace_hr at their existing locations

Usage
-----
    uv run python superres/apply_decisions.py \\
        data/landsat2sentinel/review_decisions.json \\
        data/landsat2sentinel/metadata \\
        data/landsat2sentinel/data

Options
-------
    --dry-run   Print what would happen without changing anything
    --keep-decisions   Don't clear processed entries from review_decisions.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from superres.metadata import DatasetMetadata


def _get_action(dec) -> str:
    if isinstance(dec, dict):
        return dec.get("action", "")
    return dec or ""


def _get_files(dec) -> list[str]:
    if isinstance(dec, dict) and dec.get("action") == "replace_lr":
        return dec.get("files") or []
    return []


def apply_decisions(
    decisions_path: Path,
    metadata_dir: Path,
    data_dir: Path,
    dry_run: bool = False,
    keep_decisions: bool = False,
) -> None:
    if not decisions_path.exists():
        raise FileNotFoundError(f"No decisions file found: {decisions_path}")

    decisions: dict = json.loads(decisions_path.read_text())
    if not decisions:
        print("No decisions recorded yet.")
        return

    metadata = DatasetMetadata.load_json(metadata_dir)
    samples_by_id = {int(s["location_id"]): s for s in metadata.metadata.get("samples", [])}

    counts = {"delete": 0, "replace_lr": 0, "replace_hr": 0, "replace_both": 0, "keep": 0, "unknown": 0}
    to_delete_from_meta: list[int] = []
    processed_keys: list[str] = []

    for key, dec in sorted(decisions.items(), key=lambda x: int(x[0])):
        loc_id = int(key)
        action = _get_action(dec)
        sample_dir = data_dir / f"sample_{loc_id:06d}"

        if action == "keep":
            counts["keep"] += 1
            # "keep" is a review record, not an action — leave it in the JSON
            continue

        if action not in ("delete", "replace_lr", "replace_hr", "replace_both"):
            print(f"  #{loc_id}: unknown action {action!r} — skipping")
            counts["unknown"] += 1
            continue

        if action == "delete":
            print(f"  #{loc_id}: DELETE — remove from metadata + disk")
            if not dry_run:
                if sample_dir.exists():
                    shutil.rmtree(sample_dir)
                to_delete_from_meta.append(loc_id)
            counts["delete"] += 1
            processed_keys.append(key)

        elif action == "replace_lr":
            specific_files = _get_files(dec)
            lr_dir = sample_dir / "landsat"
            if specific_files:
                print(f"  #{loc_id}: REPLACE LR ({len(specific_files)} specific files)")
                if not dry_run:
                    for fname in specific_files:
                        fpath = lr_dir / fname
                        if fpath.exists():
                            fpath.unlink()
            else:
                print(f"  #{loc_id}: REPLACE LR (full stack)")
                if not dry_run and lr_dir.exists():
                    shutil.rmtree(lr_dir)
            # Clear checkpoint so download re-attempts this sample
            if not dry_run:
                checkpoint = sample_dir / "sample_metadata.json"
                if checkpoint.exists():
                    checkpoint.unlink()
            counts["replace_lr"] += 1
            processed_keys.append(key)

        elif action == "replace_hr":
            print(f"  #{loc_id}: REPLACE HR")
            if not dry_run:
                hr_dir = sample_dir / "sentinel2"
                if hr_dir.exists():
                    shutil.rmtree(hr_dir)
                checkpoint = sample_dir / "sample_metadata.json"
                if checkpoint.exists():
                    checkpoint.unlink()
            counts["replace_hr"] += 1
            processed_keys.append(key)

        elif action == "replace_both":
            # Remove the entire sample dir and metadata entry — same location would
            # yield the same bad imagery, so create --resume must pick a new location.
            print(f"  #{loc_id}: REPLACE BOTH → remove from metadata + disk (new location needed)")
            if not dry_run:
                if sample_dir.exists():
                    shutil.rmtree(sample_dir)
                to_delete_from_meta.append(loc_id)
            counts["replace_both"] += 1
            processed_keys.append(key)

    # Remove deleted samples from metadata
    if not dry_run and to_delete_from_meta:
        delete_set = set(to_delete_from_meta)
        metadata.metadata["samples"] = [
            s for s in metadata.metadata["samples"]
            if int(s["location_id"]) not in delete_set
        ]
        metadata.metadata["dataset_info"]["total_samples"] = len(metadata.metadata["samples"])
        metadata.save_checkpoint()
        print(f"\nMetadata updated: {len(metadata.metadata['samples'])} samples remaining.")

    # Clear processed decisions from the file
    if not dry_run and not keep_decisions and processed_keys:
        remaining = {k: v for k, v in decisions.items() if k not in processed_keys}
        decisions_path.write_text(json.dumps(remaining, indent=2, sort_keys=True))
        print(f"Cleared {len(processed_keys)} processed decision(s) from {decisions_path.name}.")

    total_acted = sum(v for k, v in counts.items() if k != "keep")
    print(f"\nSummary: {counts['delete']} deleted, {counts['replace_lr']} LR replace, "
          f"{counts['replace_hr']} HR replace, {counts['replace_both']} both replace, "
          f"{counts['keep']} kept, {counts['unknown']} skipped")

    if dry_run:
        print("\nDry run — nothing changed. Re-run without --dry-run to apply.")
        return

    if total_acted > 0:
        print("\nNext steps:")
        n_new_location = counts["delete"] + counts["replace_both"]
        n_redownload = counts["replace_lr"] + counts["replace_hr"]
        step = 1
        if n_new_location:
            print(f"  {step}. Fill {n_new_location} slot(s) with new locations (create will download):")
            print(f"       uv run python superres.py create -n <total> --resume \\")
            print(f"           --coastline-dir ./data/gshhg-shp-2.3.7 \\")
            print(f"           --gebco-file ./data/gebco_2026_geotiff \\")
            print(f"           --habitat-extents-dir ./data/habitat_extents")
            step += 1
        if n_redownload:
            print(f"  {step}. Re-download {n_redownload} sample(s) at existing locations:")
            print(f"       uv run python superres.py download <manifest.csv>")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "decisions",
        help="Path to review_decisions.json (written by review_dataset.py)",
    )
    parser.add_argument(
        "metadata_dir",
        help="Path to metadata directory containing dataset_metadata.json",
    )
    parser.add_argument(
        "data_dir",
        help="Path to data directory containing sample_* folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any changes",
    )
    parser.add_argument(
        "--keep-decisions",
        action="store_true",
        help="Don't remove processed entries from review_decisions.json",
    )
    args = parser.parse_args()

    apply_decisions(
        decisions_path=Path(args.decisions),
        metadata_dir=Path(args.metadata_dir),
        data_dir=Path(args.data_dir),
        dry_run=args.dry_run,
        keep_decisions=args.keep_decisions,
    )


if __name__ == "__main__":
    main()
