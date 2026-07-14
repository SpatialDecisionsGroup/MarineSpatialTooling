"""
Configuration and argument parsing for SR dataset creation and processing.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from .constants import (
    DEFAULT_OUTPUT_DIR,
    HABITAT_EXTENTS_DIR,
    METADATA_SUBDIR,
    CREDENTIALS_SUBDIR,
    CREDENTIALS_CONFIG_FILENAME,
)
from .satellites import (
    DEFAULT_HIGHRES_SATELLITE,
    DEFAULT_LOWRES_SATELLITE,
    HIGHRES_SATELLITES,
    LOWRES_SATELLITES,
)

class Config:
    """Runtime configuration object for dataset creation and download."""

    def __init__(
        self,
        output_dir: str = "./superres",
        total_samples: int = 1000,
        ecoregion_file: Optional[str] = None,
        manifest_file: Optional[str] = None,
        coastline_dir: Optional[str] = None,
        gebco_file: Optional[str] = None,
        turbidity_file: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.total_samples = total_samples
        self.ecoregion_file = ecoregion_file
        self.manifest_file = manifest_file
        self.coastline_dir = coastline_dir
        self.gebco_file = gebco_file
        self.turbidity_file = turbidity_file
        # Auto-discover turbidity .nc in ./data if not explicitly provided
        if self.turbidity_file is None:
            data_dir = Path("./data")
            if data_dir.exists():
                nc_candidates = list(data_dir.rglob("*.nc"))
                if nc_candidates:
                    # Prefer filenames containing 'turbid' or 'kd' if present
                    preferred = [p for p in nc_candidates if any(k in p.name.lower() for k in ("turbid", "kd", "k_d"))]
                    pick = preferred[0] if preferred else nc_candidates[0]
                    self.turbidity_file = str(pick)
                else:
                    self.turbidity_file = None
        self.resume = False

        self.data_dir = self.output_dir / "data"
        self.metadata_dir = self.output_dir / "metadata"
        self.credentials_dir = self._resolve_credentials_dir()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_credentials_dir(self) -> Path:
        """Resolve which credentials directory to use.

        Priority is existing credentials in:
        1) output_dir/credentials
        2) output_dir.parent/credentials (useful for nested outputs like superres/output)
        3) DEFAULT_OUTPUT_DIR/credentials

        If none exist, default to DEFAULT_OUTPUT_DIR/credentials.
        """
        candidates = [
            self.output_dir / CREDENTIALS_SUBDIR,
            self.output_dir.parent / CREDENTIALS_SUBDIR,
            Path(DEFAULT_OUTPUT_DIR) / CREDENTIALS_SUBDIR,
        ]

        for candidate in candidates:
            credentials_file = candidate / CREDENTIALS_CONFIG_FILENAME
            if credentials_file.exists():
                return candidate

        return Path(DEFAULT_OUTPUT_DIR) / CREDENTIALS_SUBDIR

    def _load_json_file(self, file_path: Path):
        try:
            with open(file_path) as file_handle:
                return json.load(file_handle)
        except (json.JSONDecodeError, OSError):
            return None

    def _load_credentials_config(self) -> dict:
        combined_file = self.credentials_dir / CREDENTIALS_CONFIG_FILENAME
        if combined_file.exists():
            credentials = self._load_json_file(combined_file)
            if isinstance(credentials, dict):
                return credentials
        # Fallback: look for common/credentials.json or repo-level credentials.json
        common_file = Path("./common") / CREDENTIALS_CONFIG_FILENAME
        if common_file.exists():
            credentials = self._load_json_file(common_file)
            if isinstance(credentials, dict):
                return credentials

        repo_level = Path("./") / CREDENTIALS_CONFIG_FILENAME
        if repo_level.exists():
            credentials = self._load_json_file(repo_level)
            if isinstance(credentials, dict):
                return credentials

        # Last resort: environment variable
        import os

        api_key = os.environ.get("PLANET_API_KEY")
        gee_project = os.environ.get("GEE_PROJECT")
        result = {}
        if api_key:
            result["api_key"] = api_key
        if gee_project:
            result["gee_project"] = gee_project
        return result

    def load_gee_credentials(self) -> Optional[str]:
        """Earth Engine uses cached OAuth authentication, no credentials file needed."""
        return None

    def load_planet_api_key(self) -> Optional[str]:
        """Load the Planet API key from credentials config."""
        credentials = self._load_credentials_config()
        return credentials.get("api_key")

    def load_gee_project(self) -> Optional[str]:
        """Load the Google Earth Engine project ID from credentials config."""
        credentials = self._load_credentials_config()
        return credentials.get("gee_project")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "total_samples": self.total_samples,
            "ecoregion_file": self.ecoregion_file,
            "manifest_file": self.manifest_file,
        }


def config_create():
    parser = argparse.ArgumentParser(description="Create multi-image super-resolution dataset")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for dataset")
    
    # Sampling mode: either total or per-area
    sampling_group = parser.add_mutually_exclusive_group(required=True)
    sampling_group.add_argument("-n", "--total-samples", type=int, help="Total number of samples to create (distributed across strata)")
    sampling_group.add_argument("-p", "--samples-per-area", type=int, help="Number of samples per environment/depth/turbidity stratum")
    
    parser.add_argument("-e", "--ecoregions", help="Path to Marine Ecoregions GeoPackage or dataset folder")
    parser.add_argument(
        "--lowres-satellite",
        choices=sorted(LOWRES_SATELLITES),
        default=DEFAULT_LOWRES_SATELLITE,
        help=f"Satellite to use for the multi-image low-resolution stack (default: {DEFAULT_LOWRES_SATELLITE})",
    )
    parser.add_argument(
        "--highres-satellite",
        choices=sorted(HIGHRES_SATELLITES),
        default=DEFAULT_HIGHRES_SATELLITE,
        help=f"Satellite to use for the high-resolution target image (default: {DEFAULT_HIGHRES_SATELLITE})",
    )
    parser.add_argument(
        "--lowres-window-days",
        type=int,
        default=90,
        help="Half-width in days of the date window searched for low-res images around each high-res date "
             "(default: 90, i.e. a 180-day window). Wider windows find more cloud-free low-res passes in "
             "cloudy regions at the cost of more temporal drift between the low-res stack and the high-res target.",
    )
    parser.add_argument("--coastline-dir", default="./data/gshhg-shp-2.3.7", help="Path to the GSHHG/WDBII coastline dataset")
    parser.add_argument("--gebco-file", default="./data/gebco_2026_geotiff", help="Path to the GEBCO bathymetry GeoTIFF (defaults to auto-discovery under ./data)")
    parser.add_argument("--turbidity-file", default="./data/turbidity.nc", help="Path to Kd490/turbidity raster (.nc or GeoTIFF) used for bottom-visibility check")
    parser.add_argument("--habitat-extents-dir", default=HABITAT_EXTENTS_DIR, help="Directory containing coral/seagrass/mangrove habitat polygon subdirectories (from habitat_extents.sh)")
    parser.add_argument("--include-ecoregions", help="Comma-separated province names to include (prioritized). Use 'indonesia' for Indonesia regions.")
    parser.add_argument("--exclude-ecoregions", help="Comma-separated province names to exclude")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from an existing dataset_metadata.json (default: on; use --no-resume to start fresh)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Before resuming, scan existing samples for ones whose download partially failed "
             "(a sample_<id> directory exists under --output/data but has no sample_metadata.json "
             "checkpoint - e.g. no low-res images were actually available for that location/date "
             "window) and regenerate replacements for the same slot/stratum. Implies --resume.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Generate the manifest CSV only; skip downloading images. "
             "Use when you want to inspect the sampling plan before committing to downloads.",
    )

    args = parser.parse_args()

    # Determine total samples
    total_samples = None
    samples_per_area = None
    if args.total_samples:
        total_samples = args.total_samples
    else:
        samples_per_area = args.samples_per_area

    # Parse include/exclude lists
    include_list = []
    exclude_list = []
    
    if args.include_ecoregions:
        if args.include_ecoregions.lower() == "indonesia":
            # Default Indonesia provinces in MEOW
            include_list = [
                "Eastern Coral Triangle",
                "Western Coral Triangle",
                "Java Transitional",
                "Sahul Shelf",
                "Sunda Shelf",
            ]
        else:
            include_list = [e.strip() for e in args.include_ecoregions.split(",")]
    
    if args.exclude_ecoregions:
        exclude_list = [e.strip() for e in args.exclude_ecoregions.split(",")]

    config = Config(
        output_dir=args.output,
        total_samples=total_samples,
        ecoregion_file=getattr(args, "ecoregions", None),
        coastline_dir=args.coastline_dir,
        gebco_file=args.gebco_file,
        turbidity_file=getattr(args, "turbidity_file", None),
    )
    config.samples_per_area = samples_per_area
    config.include_ecoregions = include_list
    config.exclude_ecoregions = exclude_list
    config.habitat_extents_dir = args.habitat_extents_dir
    config.resume = args.resume or args.check
    config.check_existing = args.check
    config.manifest_only = args.manifest_only
    config.lowres_satellite = args.lowres_satellite
    config.highres_satellite = args.highres_satellite
    config.lowres_window_days = args.lowres_window_days

    return config

def _infer_output_dir_from_manifest(manifest_file: str) -> str:
    """Infer the dataset output directory from a manifest path.

    Manifests are written to <output_dir>/metadata/, so a manifest at
    dataset/metadata/manifest.csv implies an output dir of dataset/ (and thus
    data downloads to dataset/data/). Falls back to the manifest's own
    directory if it isn't sitting in a "metadata" subfolder.
    """
    manifest_parent = Path(manifest_file).parent
    if manifest_parent.name == METADATA_SUBDIR:
        return str(manifest_parent.parent)
    return str(manifest_parent)


def config_download():
    parser = argparse.ArgumentParser(description="Download and preprocess SR dataset from manifest")
    parser.add_argument("manifest_file", help="Path to dataset_manifest.csv")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (defaults to the manifest's dataset directory, e.g. "
             "dataset/metadata/manifest.csv -> dataset/, so data lands in dataset/data/)",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip samples that already have sample_metadata.json (default: on; use --no-resume to start fresh)")
    parser.add_argument("--turbidity-file", default="./data/turbidity.nc", help="Path to turbidity raster (.nc or GeoTIFF) to use instead of API calls")
    parser.add_argument(
        "--lowres-satellite",
        choices=sorted(LOWRES_SATELLITES),
        default=None,
        help="Override the low-res satellite for rows missing it (older manifests); normally read per-row from the manifest",
    )
    parser.add_argument(
        "--highres-satellite",
        choices=sorted(HIGHRES_SATELLITES),
        default=None,
        help="Override the high-res satellite for rows missing it (older manifests); normally read per-row from the manifest",
    )
    args = parser.parse_args()
    output_dir = args.output or _infer_output_dir_from_manifest(args.manifest_file)
    config = Config(
        output_dir=output_dir,
        manifest_file=args.manifest_file,
        turbidity_file=getattr(args, "turbidity_file", None),
    )
    config.resume = args.resume
    config.lowres_satellite = args.lowres_satellite
    config.highres_satellite = args.highres_satellite
    return config

def config_credentials():
    parser = argparse.ArgumentParser(description="Setup credentials for dataset creation")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for credentials")
    args = parser.parse_args()
    return Config(output_dir=str(Path(args.output_dir).parent))


def build_credentials_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Setup credentials for dataset creation")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for credentials")
    parser.add_argument("-p", "--planet-key", help="Planet Labs API key")
    return parser
