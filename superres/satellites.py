"""
Registry of satellite managers usable as the low-resolution multi-image
source or the high-resolution target in the super-resolution dataset
pipeline.

To add a new satellite, write a manager (a `GEESatelliteManager` subclass for
anything backed by a GEE collection, see `sentinel2.py`/`landsat.py`; or a
standalone class like `PlanetScopeManager` for other providers) and register
it in `LOWRES_SATELLITES` and/or `HIGHRES_SATELLITES` below.
"""

from __future__ import annotations

from typing import Dict, Type, Union

from .landsat import LandsatManager
from .planetscope import PlanetScopeManager
from .sentinel2 import Sentinel2Manager

ManagerClass = Type[Union[Sentinel2Manager, LandsatManager, PlanetScopeManager]]

# Satellites that can supply the multi-image, low-resolution input stack.
LOWRES_SATELLITES: Dict[str, ManagerClass] = {
    "sentinel2": Sentinel2Manager,
    "landsat": LandsatManager,
}

# Satellites that can supply the single high-resolution target image.
HIGHRES_SATELLITES: Dict[str, ManagerClass] = {
    "sentinel2": Sentinel2Manager,
    "landsat": LandsatManager,
    "planetscope": PlanetScopeManager,
}

DEFAULT_LOWRES_SATELLITE = "landsat"
DEFAULT_HIGHRES_SATELLITE = "sentinel2"


def build_manager(key: str, registry: Dict[str, ManagerClass], config) -> object:
    """Instantiate the manager registered under *key*, wiring up credentials from *config*."""
    try:
        manager_cls = registry[key]
    except KeyError:
        raise ValueError(f"Unknown satellite '{key}'. Available: {sorted(registry)}")

    if manager_cls is PlanetScopeManager:
        return manager_cls(config.load_planet_api_key())

    return manager_cls(
        config.load_gee_credentials(),
        config.load_gee_project(),
        turbidity_raster=getattr(config, "turbidity_file", None),
    )


def build_lowres_manager(config) -> object:
    return build_manager(config.lowres_satellite, LOWRES_SATELLITES, config)


def build_highres_manager(config) -> object:
    return build_manager(config.highres_satellite, HIGHRES_SATELLITES, config)
