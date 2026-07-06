"""Root-level CLI for seagrass dataset tasks.

This script chooses the dataset (`tampabay` or `indonesia`) and calls the
appropriate preparation function from the `seagrass` package.
"""

import argparse
from pathlib import Path
import importlib.util
import sys


def _load_module_from_pkg(module_name: str):
    pkg_dir = Path(__file__).resolve().parent / "seagrass"
    module_path = pkg_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"seagrass_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

parser = argparse.ArgumentParser(description="Seagrass dataset entry")
parser.add_argument("root_dir", nargs="?", default=None, help="Root directory for the chosen dataset")
parser.add_argument("--site", choices=["tampabay", "indonesia"], required=True)
parser.add_argument("-o", "--output", default=None, help="Output file or suffix (site-specific meaning)")
parser.add_argument("--gee-project", default=None, help="Optional Earth Engine project id; defaults to shared credentials if available")
parser.add_argument("--survey-date-column", default=None, help="(tampabay) Column containing survey dates for Sentinel-2/Landsat sampling")
parser.add_argument("--planetscope-raster-dir", default=None, help="(tampabay) Directory containing pre-downloaded PlanetScope TIF files")
args = parser.parse_args()

if args.site == "indonesia":
    mod = _load_module_from_pkg("indonesia")
    root = Path(args.root_dir) if args.root_dir else mod.DEFAULT_ROOT
    output_suffix = args.output if args.output is not None else "_with_bands"
    mod.prepare_indonesia(root_dir=root, output_suffix=output_suffix, gee_project=args.gee_project)
else:
    mod = _load_module_from_pkg("tampa_bay")
    root = Path(args.root_dir) if args.root_dir else mod.DEFAULT_ROOT
    output_file = Path(args.output) if args.output else root / "tampa_bay_transects_prepared.csv"
    mod.prepare_transect_csv(
        root_dir=root,
        output_file=output_file,
        gee_project=args.gee_project,
        survey_date_column=args.survey_date_column,
        planetscope_raster_dir=args.planetscope_raster_dir,
    )