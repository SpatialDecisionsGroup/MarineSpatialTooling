"""
Dataset integrity checker for the super-resolution dataset.

Validates every downloaded sample against the dataset manifest and
reports issues as a structured summary so that failed samples can be
re-queued for download.

Checks performed
----------------
Structure
  - Every manifest location_id has a corresponding sample directory
  - sample_metadata.json exists (missing = download never completed)
  - sample_metadata.json is valid JSON with required fields

File counts
  - Number of low-res .tif files matches lowres_count in metadata

Raster integrity
  - Every .tif can be opened by rasterio (corrupt / truncated files,
    e.g. an HTTP 50x error page saved as a .tif, will fail here)
  - Band count matches the satellite specification

Dimensions
  - Low-res images: not degenerate (width > 1 and height > 1)
  - High-res standardized image: exactly patch_size_pixels × patch_size_pixels

Georeferencing (high-res only; already enforced by standardize_highres_patch)
  - CRS matches alignment_crs from metadata
  - Pixel size matches patch_size_meters / patch_size_pixels (within 1mm)
  - Top-left origin matches target_origin_x / target_origin_y (within 1mm)

Values
  - Not entirely zero / nodata (sampled from a 10×10 centre window)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from tqdm import tqdm

from .constants import (
    DEFAULT_PATCH_SIZE_METERS,
    PATCH_SIZE_PIXELS,
    PLANETSCOPE_BANDS,
)
from .download_and_preprocess import _highres_standardized_file
from .satellites import HIGHRES_SATELLITES, LOWRES_SATELLITES
from common.dataset_utils import setup_logger

# GEE-downloaded GeoTIFFs have a non-standard photometric header tag that
# generates a harmless "Sum of Photometric type-related color channels"
# warning from libTIFF via GDAL's C-level stderr. It doesn't affect
# readability. Redirect stderr to suppress it: run with "2>/dev/null".


# ─── helpers ────────────────────────────────────────────────────────────────

def _expected_band_count(satellite_key: str) -> Optional[int]:
    manager_cls = LOWRES_SATELLITES.get(satellite_key) or HIGHRES_SATELLITES.get(satellite_key)
    if manager_cls is None:
        return None
    if hasattr(manager_cls, "SPEC"):
        return len(manager_cls.SPEC.band_names)
    return PLANETSCOPE_BANDS


def _all_zero_or_nodata(src: rasterio.DatasetReader) -> bool:
    """Return True if the raster is entirely zero / nodata.

    Reads one full band so sparse coastal patches (where only a small fraction
    of pixels are non-zero) aren't mistakenly flagged. LZW-compressed GeoTIFFs
    at typical SR patch sizes (~256-512 px) read fast enough for this to be OK.
    """
    try:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            return bool(np.all(data == nodata))
        return bool(np.all(data == 0))
    except Exception:
        return True


# ─── issue tracking ─────────────────────────────────────────────────────────

@dataclass
class Issue:
    code: str       # short machine-readable key, e.g. "missing_metadata"
    sample: str     # "sample_000802"
    detail: str     # human-readable description


@dataclass
class CheckReport:
    data_dir: Path
    manifest_path: Optional[Path]
    samples_checked: int = 0
    issues: List[Issue] = field(default_factory=list)

    def add(self, code: str, sample: str, detail: str):
        self.issues.append(Issue(code, sample, detail))

    def samples_with_issues(self) -> List[str]:
        return sorted(set(i.sample for i in self.issues))

    def by_code(self) -> Dict[str, List[Issue]]:
        result: Dict[str, List[Issue]] = defaultdict(list)
        for issue in self.issues:
            result[issue.code].append(issue)
        return dict(sorted(result.items()))

    def print_summary(self, verbose: bool = False):
        by_code = self.by_code()
        n_issues = len(self.issues)
        n_bad_samples = len(self.samples_with_issues())

        print(f"\n{'='*60}")
        print(f"Dataset Check Report")
        print(f"  Data dir : {self.data_dir}")
        if self.manifest_path:
            print(f"  Manifest : {self.manifest_path}")
        print(f"  Samples  : {self.samples_checked} checked")
        print(f"{'='*60}")

        if not self.issues:
            print("\nAll checks passed.")
            print(f"{'='*60}")
            return

        SECTION = {
            "missing_sample_dir":      "STRUCTURE",
            "missing_metadata":        "STRUCTURE",
            "invalid_metadata":        "STRUCTURE",
            "lowres_count_mismatch":   "FILE COUNTS",
            "missing_highres_file":    "FILE COUNTS",
            "raster_open_failed":      "RASTER INTEGRITY",
            "band_count_mismatch":     "RASTER INTEGRITY",
            "degenerate_raster":       "DIMENSIONS",
            "hr_wrong_dimensions":     "DIMENSIONS",
            "hr_crs_mismatch":         "GEOREFERENCING",
            "hr_pixel_size_mismatch":  "GEOREFERENCING",
            "hr_origin_mismatch":      "GEOREFERENCING",
            "all_nodata":              "VALUES",
        }
        LABEL = {
            "missing_sample_dir":      "manifest location_id has no sample directory",
            "missing_metadata":        "sample_metadata.json missing (incomplete download)",
            "invalid_metadata":        "sample_metadata.json invalid or missing required fields",
            "lowres_count_mismatch":   "low-res file count doesn't match metadata",
            "missing_highres_file":    "standardised high-res file missing",
            "raster_open_failed":      "raster file failed to open (corrupt / truncated)",
            "band_count_mismatch":     "unexpected band count",
            "degenerate_raster":       "degenerate raster dimensions (e.g. 1×1 pixel)",
            "hr_wrong_dimensions":     "high-res dimensions don't match patch_size_pixels",
            "hr_crs_mismatch":         "high-res CRS doesn't match alignment_crs",
            "hr_pixel_size_mismatch":  "high-res pixel size inconsistent with patch metadata",
            "hr_origin_mismatch":      "high-res origin inconsistent with target_origin",
            "all_nodata":              "raster appears to contain only zero / nodata values",
        }

        current_section = None
        for code, issues in by_code.items():
            section = SECTION.get(code, "OTHER")
            if section != current_section:
                print(f"\n{section}")
                current_section = section
            print(f"  {len(issues):4d}  {LABEL.get(code, code)}")
            if verbose:
                for issue in issues:
                    print(f"          {issue.sample}: {issue.detail}")

        print(f"\n{'='*60}")
        print(f"TOTAL: {n_issues} issues across {n_bad_samples} samples")
        if not verbose and n_bad_samples > 0:
            print(f"       Run with --verbose to list affected files.")
        print(f"{'='*60}")

    def to_json(self, path: Path):
        out = {
            "data_dir": str(self.data_dir),
            "manifest": str(self.manifest_path) if self.manifest_path else None,
            "samples_checked": self.samples_checked,
            "total_issues": len(self.issues),
            "issues": [
                {"code": i.code, "sample": i.sample, "detail": i.detail}
                for i in self.issues
            ],
        }
        path.write_text(json.dumps(out, indent=2))


# ─── per-sample checks ───────────────────────────────────────────────────────

REQUIRED_META_FIELDS = [
    "location_id", "lowres_satellite", "highres_satellite", "lowres_count",
    "alignment_crs", "patch_size_pixels", "patch_size_meters",
    "target_origin_x", "target_origin_y",
]


def _check_raster(path: Path, role: str, satellite_key: str, report: CheckReport, sample_name: str):
    """Validate a single raster file. Returns the open DatasetReader or None."""
    try:
        src = rasterio.open(path)
    except Exception as exc:
        report.add("raster_open_failed", sample_name, f"{role}/{path.name}: {exc}")
        return None

    # Band count
    expected_bands = _expected_band_count(satellite_key)
    if expected_bands is not None and src.count != expected_bands:
        report.add(
            "band_count_mismatch", sample_name,
            f"{role}/{path.name}: expected {expected_bands} bands, got {src.count}",
        )

    # Degenerate dimensions
    if src.width <= 1 or src.height <= 1:
        report.add(
            "degenerate_raster", sample_name,
            f"{role}/{path.name}: {src.width}×{src.height} pixels",
        )

    # All-nodata / all-zero
    if _all_zero_or_nodata(src):
        report.add(
            "all_nodata", sample_name,
            f"{role}/{path.name}: band 1 is entirely zero/nodata",
        )

    return src


def check_sample(sample_dir: Path, report: CheckReport, check_values: bool = True):
    sample_name = sample_dir.name
    meta_file = sample_dir / "sample_metadata.json"

    if not meta_file.exists():
        report.add("missing_metadata", sample_name, "sample_metadata.json not found")
        return

    try:
        meta = json.loads(meta_file.read_text())
        for field_name in REQUIRED_META_FIELDS:
            if field_name not in meta:
                raise KeyError(field_name)
    except Exception as exc:
        report.add("invalid_metadata", sample_name, str(exc))
        return

    lowres_key = meta["lowres_satellite"]
    highres_key = meta["highres_satellite"]
    location_id = int(meta["location_id"])
    expected_lr_count = int(meta["lowres_count"])
    patch_size_pixels = int(meta.get("patch_size_pixels", PATCH_SIZE_PIXELS))
    patch_size_meters = float(meta.get("patch_size_meters", DEFAULT_PATCH_SIZE_METERS))
    alignment_crs = meta["alignment_crs"]
    target_origin_x = float(meta["target_origin_x"])
    target_origin_y = float(meta["target_origin_y"])
    expected_pixel_size = patch_size_meters / patch_size_pixels

    # ── Low-res ──────────────────────────────────────────────────────────────
    lowres_dir = sample_dir / lowres_key
    lowres_files = sorted(lowres_dir.glob(f"{lowres_key}_*.tif")) if lowres_dir.exists() else []
    actual_lr_count = len(lowres_files)

    if actual_lr_count != expected_lr_count:
        report.add(
            "lowres_count_mismatch", sample_name,
            f"expected {expected_lr_count} {lowres_key} images, found {actual_lr_count}",
        )

    for lr_file in lowres_files:
        src = _check_raster(lr_file, lowres_key, lowres_key, report, sample_name)
        if src is not None:
            src.close()

    # ── High-res ─────────────────────────────────────────────────────────────
    hr_file = _highres_standardized_file(sample_dir, location_id, highres_key)
    if not hr_file.exists():
        report.add("missing_highres_file", sample_name, f"{hr_file.name} not found")
        return

    src = _check_raster(hr_file, highres_key, highres_key, report, sample_name)
    if src is None:
        return

    # Exact grid match — standardize_highres_patch is supposed to guarantee this.
    if src.width != patch_size_pixels or src.height != patch_size_pixels:
        report.add(
            "hr_wrong_dimensions", sample_name,
            f"{hr_file.name}: expected {patch_size_pixels}×{patch_size_pixels}, "
            f"got {src.width}×{src.height}",
        )

    # CRS
    if src.crs is None or str(src.crs).replace("EPSG:", "") not in alignment_crs:
        try:
            crs_match = src.crs and src.crs.to_epsg() == int(alignment_crs.split(":")[-1])
        except Exception:
            crs_match = False
        if not crs_match:
            report.add(
                "hr_crs_mismatch", sample_name,
                f"{hr_file.name}: CRS is {src.crs}, expected {alignment_crs}",
            )

    # Pixel size and origin (1 mm tolerance)
    tol = 1e-3
    actual_pixel = abs(src.transform.a)  # transform.a = col pixel size
    if abs(actual_pixel - expected_pixel_size) > tol:
        report.add(
            "hr_pixel_size_mismatch", sample_name,
            f"{hr_file.name}: pixel size {actual_pixel:.6f} m, expected {expected_pixel_size:.6f} m",
        )

    actual_origin_x = src.transform.c
    actual_origin_y = src.transform.f - src.height * abs(src.transform.e)
    if abs(actual_origin_x - target_origin_x) > tol or abs(actual_origin_y - target_origin_y) > tol:
        report.add(
            "hr_origin_mismatch", sample_name,
            f"{hr_file.name}: origin ({actual_origin_x:.4f}, {actual_origin_y:.4f}), "
            f"expected ({target_origin_x:.4f}, {target_origin_y:.4f})",
        )

    src.close()


# ─── driver ─────────────────────────────────────────────────────────────────

def check_dataset():
    parser = argparse.ArgumentParser(
        description="Check dataset integrity: validates file presence, raster dimensions, "
                    "band counts, georeferencing, and value ranges. "
                    "GEE-downloaded files produce harmless libTIFF warnings on stderr; "
                    "run with '2>/dev/null' to suppress them."
    )
    parser.add_argument("data_dir", help="Path to dataset data directory (contains sample_* folders)")
    parser.add_argument(
        "--manifest", default=None,
        help="Path to dataset_manifest.csv; if provided, cross-checks that every "
             "manifest location_id has a sample directory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write a JSON report of all findings to this path",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-file detail for every issue")
    parser.add_argument(
        "--no-values", action="store_true",
        help="Skip the all-zero/nodata value check (faster for large datasets)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        raise SystemExit(1)

    manifest_path = Path(args.manifest) if args.manifest else None
    report = CheckReport(data_dir=data_dir, manifest_path=manifest_path)

    # ── Manifest cross-check ─────────────────────────────────────────────────
    if manifest_path:
        try:
            import pandas as pd
            manifest = pd.read_csv(manifest_path)
            for loc_id in manifest["location_id"].astype(int):
                sample_dir = data_dir / f"sample_{loc_id:06d}"
                if not sample_dir.exists():
                    report.add(
                        "missing_sample_dir", f"sample_{loc_id:06d}",
                        f"location_id {loc_id} is in the manifest but has no sample directory",
                    )
        except Exception as exc:
            print(f"Warning: could not read manifest ({exc}), skipping manifest cross-check")

    # ── Per-sample checks ─────────────────────────────────────────────────────
    sample_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("sample_"))
    report.samples_checked = len(sample_dirs)

    for sample_dir in tqdm(sample_dirs, desc="Checking samples"):
        check_sample(sample_dir, report, check_values=not args.no_values)

    # ── Output ────────────────────────────────────────────────────────────────
    report.print_summary(verbose=args.verbose)

    if args.output:
        out_path = Path(args.output)
        report.to_json(out_path)
        print(f"\nJSON report written to {out_path}")

    raise SystemExit(0 if not report.issues else 1)


if __name__ == "__main__":
    check_dataset()
