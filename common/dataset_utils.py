"""
Generic dataset utilities reused across packages.
"""

import logging
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon


def get_utm_crs(latitude: float, longitude: float) -> str:
    zone = int((longitude + 180) / 6) + 1
    is_north = latitude >= 0
    epsg = 32600 + zone if is_north else 32700 + zone
    return f"EPSG:{epsg}"


def get_date_range_for_season(season_months, year: int = 2023):
    month_start, month_end = season_months
    start_date = f"{year}-{month_start:02d}-01"
    if month_start <= month_end:
        end_year = year
        end_month = month_end + 1
    else:
        end_year = year + 1
        end_month = month_end + 1

    if end_month == 13:
        end_year += 1
        end_month = 1

    end_date = f"{end_year}-{end_month:02d}-01"
    return (start_date, end_date)


def sample_point_in_polygon(polygon: Polygon, max_attempts: int = 100):
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(max_attempts):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        if polygon.contains(point):
            return point
    return None


def ensure_directory(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Create a file logger for the dataset pipeline.

    The default level is INFO; pass `level=logging.DEBUG` to capture debug messages
    such as why candidates were rejected during sampling.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
