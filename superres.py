import sys
import argparse

from superres import apply_decisions as _apply_decisions
from superres import check_dataset
from superres import create_sr_dataset
from superres import download_and_preprocess
from superres import postprocess
from superres import tile_dataset

parser = argparse.ArgumentParser(description="Superres dataset entry")
subparsers = parser.add_subparsers(dest="action", required=True)

# Apply review decisions
subparsers.add_parser("apply", help="Apply review decisions from review_dataset.py (see apply subcommand args)")

# Check subcommand
subparsers.add_parser("check", help="Check dataset integrity (see check subcommand args)")

# Create subcommand (arguments handled by create_sr_dataset.config_create)
subparsers.add_parser("create", help="Create dataset (see create subcommand args)")

# Download subcommand (arguments handled by download_and_preprocess.config_download)
subparsers.add_parser("download", help="Download dataset (see download subcommand args)")

# Postprocess subcommand
subparsers.add_parser("postprocess", help="Align low-res to HR grid and rescale to reflectance (see postprocess subcommand args)")

# Tile subcommand
subparsers.add_parser("tile", help="Tile dataset (see tile subcommand args)")

args, remaining = parser.parse_known_args()

# Forward remaining argv to the module-level parsers by shifting sys.argv
if args.action == "apply":
	sys.argv = [sys.argv[0]] + remaining
	_apply_decisions.main()
elif args.action == "check":
	sys.argv = [sys.argv[0]] + remaining
	check_dataset.check_dataset()
elif args.action == "create":
	sys.argv = [sys.argv[0]] + remaining
	create_sr_dataset.create_dataset()
elif args.action == "download":
	sys.argv = [sys.argv[0]] + remaining
	download_and_preprocess.download_dataset()
elif args.action == "postprocess":
	sys.argv = [sys.argv[0]] + remaining
	postprocess.postprocess_dataset()
elif args.action == "tile":
	sys.argv = [sys.argv[0]] + remaining
	tile_dataset.tile_dataset()
