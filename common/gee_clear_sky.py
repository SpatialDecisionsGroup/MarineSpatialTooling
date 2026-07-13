"""AOI-local clear-sky scene selection, shared across seagrass/mangrove site scripts.

Scene-level CLOUDY_PIXEL_PERCENTAGE / CLOUD_COVER is a whole-tile average, which
can look fine while the small AOI itself sits under a cloud. This scores cloud
cover directly over the sample point instead: Sentinel-2 via Cloud Score+'s "cs"
band (0=cloud, 1=clear), Landsat via the QA_PIXEL "Clear" bit (bit 6).

Originally written for seagrass/indonesia.py; extracted here once
seagrass/kenya_mangroves.py needed the same logic, since importing one
seagrass/*.py script from another risks colliding with the root-level
seagrass.py wrapper (same package name, different file).
"""

from __future__ import annotations

from datetime import datetime

import ee

from common.common import parse_date_object

S2_CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
LOCAL_CLARITY_BUFFER_M = 45

# Local clear-score tiers to try in order (0-1, higher = clearer); the first tier
# with any candidate wins, and the closest-by-date scene within that tier is used.
CLEAR_SCORE_TIERS = (0.75, 0.6, 0.4)


def s2_local_clear_candidates(
    lon: float, lat: float, date_start: str, date_end: str,
    buffer_m: float = LOCAL_CLARITY_BUFFER_M,
    collection_id: str = "COPERNICUS/S2_SR_HARMONIZED",
) -> list[dict]:
    """Sentinel-2 candidates in the window, scored by Cloud Score+ AOI-local clarity.

    Defaults to the L2A (surface reflectance) collection, for sampling band
    values directly from GEE. Pass collection_id="COPERNICUS/S2_HARMONIZED" (L1C)
    when the goal is picking which *raw* scene to download for ACOLITE instead —
    L2A coverage can have real gaps (ESA's archive isn't backfilled everywhere)
    where L1C still has the acquisition; Cloud Score+ links to either.
    """
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()
    s2 = (ee.ImageCollection(collection_id)
          .filterBounds(point).filterDate(date_start, date_end))
    cs_plus = ee.ImageCollection(S2_CLOUD_SCORE_COLLECTION)
    linked = s2.linkCollection(cs_plus, ["cs"])

    def _tag(img):
        local_cs = img.select("cs").reduceRegion(ee.Reducer.mean(), point, 10).get("cs")
        return img.set("local_clear_score", local_cs).set("full_asset_id", img.get("system:id"))

    info = linked.map(_tag).select([]).getInfo()
    return _candidates_from_feature_info(info)


def landsat_local_clear_candidates(
    lon: float, lat: float, date_start: str, date_end: str,
    buffer_m: float = LOCAL_CLARITY_BUFFER_M,
) -> list[dict]:
    """Landsat candidates in the window, scored by fraction of QA_PIXEL "Clear" pixels over the AOI."""
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_m).bounds()
    coll = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(point).filterDate(date_start, date_end))

    def _tag(img):
        clear_bit = img.select("QA_PIXEL").bitwiseAnd(1 << 6).gt(0).rename("clear")
        local_clear = clear_bit.reduceRegion(ee.Reducer.mean(), point, 30).get("clear")
        return img.set("local_clear_score", local_clear).set("full_asset_id", img.get("system:id"))

    info = coll.map(_tag).select([]).getInfo()
    return _candidates_from_feature_info(info)


def _candidates_from_feature_info(info: dict) -> list[dict]:
    candidates = []
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        timestamp = props.get("system:time_start")
        asset_id = props.get("full_asset_id")
        if timestamp is None or asset_id is None:
            continue
        candidates.append({
            "asset_id": asset_id,
            "date": datetime.fromtimestamp(timestamp / 1000).isoformat(),
            "clear_score": props.get("local_clear_score"),
        })
    return candidates


def select_by_local_clarity(candidates: list[dict], target_date: str) -> dict | None:
    """Pick the AOI-locally-clearest scene, breaking ties by date proximity.

    Tries CLEAR_SCORE_TIERS in order (0.75 / 0.6 / 0.4) and returns the
    closest-by-date candidate within the first tier that has any match. Falls
    back to closest-by-date regardless of clarity if nothing ever clears 0.4.
    """
    if not candidates:
        return None
    row_date = parse_date_object(target_date)

    def _closest(pool: list[dict]) -> dict | None:
        if row_date is None:
            return pool[0]
        best: tuple[int, dict] | None = None
        for c in pool:
            c_date = parse_date_object(c.get("date", ""))
            if c_date is None:
                continue
            distance = abs((c_date - row_date).days)
            if best is None or distance < best[0]:
                best = (distance, c)
        return best[1] if best else None

    for min_score in CLEAR_SCORE_TIERS:
        pool = [c for c in candidates if c.get("clear_score") is not None and c["clear_score"] >= min_score]
        if not pool:
            continue
        picked = _closest(pool)
        if picked is not None:
            return picked

    return _closest([c for c in candidates if c.get("clear_score") is not None]) or (candidates[0] if candidates else None)
