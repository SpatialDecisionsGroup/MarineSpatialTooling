import sys
import argparse

import superres.create_sr_dataset as create_sr_dataset
import superres.download_and_preprocess as download_and_preprocess
import superres.tile_dataset as tile_dataset

parser = argparse.ArgumentParser(description="Superres dataset entry")
subparsers = parser.add_subparsers(dest="action", required=True)

# Create subcommand (arguments handled by create_sr_dataset.config_create)
subparsers.add_parser("create", help="Create dataset (see create subcommand args)")

# Download subcommand (arguments handled by download_and_preprocess.config_download)
subparsers.add_parser("download", help="Download dataset (see download subcommand args)")

# Tile subcommand
subparsers.add_parser("tile", help="Tile dataset (see tile subcommand args)")

args, remaining = parser.parse_known_args()

# Forward remaining argv to the module-level parsers by shifting sys.argv
if args.action == "create":
	sys.argv = [sys.argv[0]] + remaining
	create_sr_dataset.create_dataset()
elif args.action == "download":
	sys.argv = [sys.argv[0]] + remaining
	download_and_preprocess.download_dataset()
elif args.action == "tile":
	sys.argv = [sys.argv[0]] + remaining
	tile_dataset.tile_dataset()
