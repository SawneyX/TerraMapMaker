"""
Swiss orthophoto (swisstopo) fetcher for TerraMapMaker.

Fetches an exact bbox image from swisstopo WMS (ch.swisstopo.swissimage) using LV95 (EPSG:2056).

Steps:
- Convert ENU rectangle (meters) relative to GNSS reference (lat, lon, alt) to WGS84 lat/lon corners.
- Project corners to LV95 (EPSG:2056) and build a WMS 1.3.0 GetMap request with that BBOX.
- Return a PIL.Image of the requested region.

Requirements: pyproj, Pillow
"""

from __future__ import annotations

import io
import math
import urllib.parse
import urllib.request
from typing import Tuple

from PIL import Image
from pyproj import Transformer

EARTH_RADIUS_M = 6371008.8


def enu_to_wgs84_small_angle(ref_lat_deg: float, ref_lon_deg: float, east_m: float, north_m: float) -> Tuple[float, float]:
    lat0_rad = math.radians(ref_lat_deg)
    dlat_deg = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon_deg = (east_m / (EARTH_RADIUS_M * math.cos(lat0_rad))) * (180.0 / math.pi)
    return ref_lat_deg + dlat_deg, ref_lon_deg + dlon_deg


def enu_rect_to_lv95_bbox(
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
    x_min_m: float,
    y_min_m: float,
    x_max_m: float,
    y_max_m: float,
) -> Tuple[float, float, float, float]:
    """Compute LV95 (EPSG:2056) bbox (xmin, ymin, xmax, ymax) from ENU rect.

    We approximate ENU->WGS84 using small-angle formulas and then project to LV95.
    """
    # ENU corners to WGS84
    lat_bl, lon_bl = enu_to_wgs84_small_angle(ref_lat_deg, ref_lon_deg, x_min_m, y_min_m)
    lat_tr, lon_tr = enu_to_wgs84_small_angle(ref_lat_deg, ref_lon_deg, x_max_m, y_max_m)

    # Project to LV95
    to_lv95 = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    # always_xy=True => input order lon,lat
    x_bl, y_bl = to_lv95.transform(lon_bl, lat_bl)
    x_tr, y_tr = to_lv95.transform(lon_tr, lat_tr)

    xmin = min(x_bl, x_tr)
    xmax = max(x_bl, x_tr)
    ymin = min(y_bl, y_tr)
    ymax = max(y_bl, y_tr)
    return (xmin, ymin, xmax, ymax)


def fetch_swissimage_by_bbox_lv95(
    bbox_lv95: Tuple[float, float, float, float],
    size_px: Tuple[int, int] = (1024, 1024),
    format_str: str = "image/jpeg",
) -> Image.Image:
    """Fetch swisstopo swissimage for an exact LV95 bbox via WMS 1.3.0.

    bbox_lv95: (xmin, ymin, xmax, ymax) in meters (EPSG:2056)
    size_px: (width, height)
    """
    xmin, ymin, xmax, ymax = bbox_lv95

    base_url = "https://wms.geo.admin.ch/"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": "ch.swisstopo.swissimage",
        "CRS": "EPSG:2056",
        "BBOX": f"{xmin:.3f},{ymin:.3f},{xmax:.3f},{ymax:.3f}",
        "WIDTH": str(max(1, int(size_px[0]))),
        "HEIGHT": str(max(1, int(size_px[1]))),
        "FORMAT": format_str,
        "STYLES": "",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return img


__all__ = [
    "enu_rect_to_lv95_bbox",
    "fetch_swissimage_by_bbox_lv95",
]


def fetch_swissimage_by_bbox_wgs84(
    bbox_wgs84: Tuple[float, float, float, float],
    size_px: Tuple[int, int] = (1024, 1024),
    format_str: str = "image/jpeg",
) -> Image.Image:
    """Fetch swisstopo swissimage for an exact WGS84 bbox via WMS 1.3.0.

    bbox_wgs84: (min_lat, min_lon, max_lat, max_lon) in degrees
    For WMS 1.3.0 with EPSG:4326, axis order is (lat, lon): BBOX=min_lat,min_lon,max_lat,max_lon
    size_px: (width, height)
    """
    min_lat, min_lon, max_lat, max_lon = bbox_wgs84

    base_url = "https://wms.geo.admin.ch/"
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": "ch.swisstopo.swissimage",
        "CRS": "EPSG:4326",
        "BBOX": f"{min_lat:.8f},{min_lon:.8f},{max_lat:.8f},{max_lon:.8f}",
        "WIDTH": str(max(1, int(size_px[0]))),
        "HEIGHT": str(max(1, int(size_px[1]))),
        "FORMAT": format_str,
        "STYLES": "",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return img


