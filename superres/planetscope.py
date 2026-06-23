"""
PlanetScope data retrieval and processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .constants import (
    MAX_CLOUD_COVER,
    PLANETSCOPE_ITEM_TYPE,
    PLANETSCOPE_ORDERS_URL,
    PLANETSCOPE_PRODUCT_BUNDLE,
    PLANETSCOPE_RESOLUTION,
)


class PlanetScopeManager:
    """Manages PlanetScope search, ordering, and downloads."""

    SEARCH_URL = "https://api.planet.com/data/v1/quick-search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def resolution_meters(cls) -> float:
        return PLANETSCOPE_RESOLUTION

    def _auth(self) -> tuple[str, str]:
        return self.api_key or "", ""

    def _order_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError("No Planet API key provided")
        return {
            "Authorization": f"api-key {self.api_key}",
            "Content-Type": "application/json",
        }

    def retrieve_images(
        self,
        lat: float,
        long: float,
        date_start: str,
        date_end: str,
        max_images: int = 8,
    ) -> List[Dict]:
        """Retrieve PlanetScope catalog candidates for a location and date range."""
        if not self.api_key:
            self.logger.warning("No Planet API key provided")
            return []

        try:
            aoi = {
                "type": "Polygon",
                "coordinates": [[
                    [long - 0.01, lat - 0.01],
                    [long + 0.01, lat - 0.01],
                    [long + 0.01, lat + 0.01],
                    [long - 0.01, lat + 0.01],
                    [long - 0.01, lat - 0.01],
                ]],
            }

            payload = {
                "item_types": [PLANETSCOPE_ITEM_TYPE],
                "filter": {
                    "type": "AndFilter",
                    "config": [
                        {
                            "type": "GeometryFilter",
                            "field_name": "geometry",
                            "config": aoi,
                        },
                        {
                            "type": "DateRangeFilter",
                            "field_name": "acquired",
                            "config": {
                                "gte": f"{date_start}T00:00:00Z",
                                "lte": f"{date_end}T23:59:59Z",
                            },
                        },
                        {
                            "type": "RangeFilter",
                            "field_name": "cloud_cover",
                            "config": {"lte": MAX_CLOUD_COVER},
                        },
                    ],
                },
            }

            response = requests.post(
                self.SEARCH_URL,
                json=payload,
                auth=self._auth(),
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            if not features:
                return []

            candidates = []
            for feature in features:
                props = feature.get("properties", {})
                cc = props.get("cloud_cover")
                try:
                    cc_val = float(cc) if cc is not None else 100.0
                except Exception:
                    cc_val = 100.0
                if cc_val <= 1.0:
                    cc_val *= 100.0
                candidates.append((cc_val, feature))

            candidates.sort(key=lambda item: (item[0], item[1].get("properties", {}).get("acquired", "")))
            chosen = [feature for cc_val, feature in candidates if cc_val <= MAX_CLOUD_COVER][:max_images]

            results = []
            for feature in chosen:
                properties = feature.get("properties", {})
                results.append(
                    {
                        "id": feature.get("id"),
                        "date": properties.get("acquired", ""),
                        "cloud_cover": properties.get("cloud_cover", -1),
                        "item_type": properties.get("item_type", PLANETSCOPE_ITEM_TYPE),
                        "geometry": feature.get("geometry"),
                        "source": "PlanetScope via Planet API",
                    }
                )

            self.logger.info(f"Retrieved {len(results)} PlanetScope images for ({lat}, {long})")
            return results
        except Exception as exc:
            self.logger.error(f"Failed to retrieve PlanetScope images: {exc}")
            return []

    def build_order_request(
        self,
        item_ids: List[str],
        aoi_geometry: Dict,
        alignment_crs: str,
        order_name: str,
    ) -> Dict:
        return {
            "name": order_name,
            "source_type": "scenes",
            "products": [
                {
                    "item_ids": item_ids,
                    "item_type": PLANETSCOPE_ITEM_TYPE,
                    "product_bundle": PLANETSCOPE_PRODUCT_BUNDLE,
                }
            ],
            "tools": [
                {
                    "harmonize": {
                        "target_sensor": "Sentinel-2",
                    }
                },
                {
                    "clip": {
                        "aoi": aoi_geometry,
                    }
                },
                {
                    "reproject": {
                        "projection": alignment_crs,
                        "resolution": PLANETSCOPE_RESOLUTION,
                        "kernel": "bilinear",
                    }
                },
                {
                    "file_format": {
                        "format": "COG",
                    }
                },
            ],
        }

    def create_order(
        self,
        item_ids: List[str],
        aoi_geometry: Dict,
        alignment_crs: str,
        order_name: str,
    ) -> Dict:
        response = requests.post(
            PLANETSCOPE_ORDERS_URL,
            headers=self._order_headers(),
            json=self.build_order_request(item_ids, aoi_geometry, alignment_crs, order_name),
            timeout=60,
        )
        if not response.ok:
            self.logger.error(
                "Planet order creation failed (HTTP %s): %s", response.status_code, response.text
            )
        response.raise_for_status()
        return response.json()

    def get_order(self, order_id: str) -> Dict:
        response = requests.get(
            f"{PLANETSCOPE_ORDERS_URL}/{order_id}",
            headers=self._order_headers(),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def download_result(self, url: str, output_path: Path) -> bool:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)
            return True
        except Exception as exc:
            self.logger.error(f"Failed to download order result to {output_path}: {exc}")
            if output_path.exists():
                output_path.unlink()
            return False

    def download_order_results(self, order: Dict, output_dir: Path) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths: List[Path] = []
        results = order.get("_links", {}).get("results") or []
        for result in results:
            url = result.get("location")
            name = result.get("name")
            if not url or not name:
                continue

            output_path = output_dir / name
            if self.download_result(url, output_path):
                downloaded_paths.append(output_path)

        return downloaded_paths


class QuotaExhausted(Exception):
    """Planet API quota exhausted or rate-limited."""

    def __init__(
        self,
        item_id: str,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        self.item_id = item_id
        self.status_code = status_code
        self.response_text = response_text
        detail = message
        if status_code is not None:
            detail = f"{detail} (HTTP {status_code})"
        if response_text:
            detail = f"{detail}: {response_text}"
        super().__init__(detail)


class PlanetActivationError(Exception):
    """Planet asset activation failed for a non-quota reason."""

    def __init__(
        self,
        item_id: str,
        message: str,
        status_code: Optional[int] = None,
        response_text: Optional[str] = None,
    ):
        self.item_id = item_id
        self.status_code = status_code
        self.response_text = response_text
        detail = message
        if status_code is not None:
            detail = f"{detail} (HTTP {status_code})"
        if response_text:
            detail = f"{detail}: {response_text}"
        super().__init__(detail)
