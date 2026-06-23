"""
Metadata handling for SR dataset.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from .constants import MANIFEST_COLUMNS


class DatasetMetadata:
    """Handles metadata storage and retrieval."""
    
    def __init__(self, metadata_dir: Path):
        """
        Initialise metadata handler.
        
        Args:
            metadata_dir: Directory to store metadata files
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            "dataset_info": {
                "created_date": datetime.now().isoformat(),
                "total_samples": 0,
                "lowres_satellite": "",
                "lowres_resolution_meters": None,
                "highres_satellite": "",
                "highres_resolution_meters": None,
                "max_cloud_cover": 20,
                "crs": "EPSG:4326",
            },
            "samples": [],
            "provinces": {},
        }
    
    def add_dataset_info(self, info):
        self.metadata["dataset_info"].update(info)
    
    def add_sample(self, sample):
        self.metadata["samples"].append(sample)
    
    def add_province(self, province_id, province_info):
        self.metadata["provinces"][province_id] = province_info

    def add_ecoregion(self, ecoregion_id, ecoregion_info):
        self.add_province(ecoregion_id, ecoregion_info)

    @classmethod
    def load_json(cls, metadata_dir: Path, filename: str = "dataset_metadata.json"):
        filepath = Path(metadata_dir) / filename
        with open(filepath) as file_handle:
            loaded = json.load(file_handle)

        instance = cls(metadata_dir)
        instance.metadata = loaded
        instance.metadata.setdefault("dataset_info", {})
        instance.metadata.setdefault("samples", [])
        instance.metadata.setdefault("provinces", {})
        return instance
    
    def update_dataset_sample_count(self, count):
        self.metadata["dataset_info"]["total_samples"] = count
    
    def save_json(self, filename = "dataset_metadata.json"):
        filepath = self.metadata_dir / filename
        with open(filepath, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        return filepath

    def save_checkpoint(self):
        json_file = self.save_json()
        csv_file = self.save_manifest_csv()
        return json_file, csv_file
    
    def save_manifest_csv(self, filename = "dataset_manifest.csv"):
        filepath = self.metadata_dir / filename

        def serialise(value):
            if value is None:
                return ""
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, default=str)
            return value
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            
            for sample in self.metadata["samples"]:
                date_range = sample.get("date_range", ("", ""))
                writer.writerow({
                    "location_id": sample["location_id"],
                    "latitude": sample["latitude"],
                    "longitude": sample["longitude"],
                    "season_id": sample.get("season_id", ""),
                    "province": sample.get("province", sample.get("ecoregion", "")),
                    "environment_class": sample.get("environment_class", ""),
                    "depth_class": sample.get("depth_class", ""),
                    "depth_m": sample.get("depth_m", ""),
                    "turbidity_class": sample.get("turbidity_class", ""),
                    "turbidity_index": sample.get("turbidity_index", ""),
                    "date_range_start": date_range[0],
                    "date_range_end": date_range[1],
                    "alignment_crs": sample.get("alignment_crs", ""),
                    "patch_size_pixels": sample.get("patch_size_pixels", ""),
                    "patch_size_meters": sample.get("patch_size_meters", ""),
                    "target_origin_x": sample.get("target_origin_x", ""),
                    "target_origin_y": sample.get("target_origin_y", ""),
                    "lowres_satellite": sample.get("lowres_satellite", ""),
                    "lowres_count": len(sample.get("lowres_images", [])),
                    "highres_satellite": sample.get("highres_satellite", ""),
                    "highres_item_ids": serialise(sample.get("highres_item_ids", [])),
                    "highres_order_id": sample.get("highres_order_id", ""),
                    "highres_product_bundle": sample.get("highres_product_bundle", ""),
                    "highres_aoi_geojson": serialise(sample.get("highres_aoi_geojson", {})),
                    "highres_count": len(sample.get("highres_images", [])),
                })
        
        return filepath
    
    @staticmethod
    def load_from_csv(csv_file):
        def parse_json_field(value):
            if value in (None, ""):
                return value
            if isinstance(value, str) and value[:1] in "[{":
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        samples = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row["location_id"] = int(row["location_id"])
                row["latitude"] = float(row["latitude"])
                row["longitude"] = float(row["longitude"])
                if row.get("season_id") not in (None, ""):
                    row["season_id"] = int(row["season_id"])
                for key in ("depth_m", "turbidity_index", "patch_size_meters", "target_origin_x", "target_origin_y"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                for key in ("patch_size_pixels", "lowres_count", "highres_count"):
                    if row.get(key) not in (None, ""):
                        row[key] = int(row[key])
                row["highres_item_ids"] = parse_json_field(row.get("highres_item_ids"))
                row["highres_aoi_geojson"] = parse_json_field(row.get("highres_aoi_geojson"))
                samples.append(row)
        return samples
