"""NSI source footprint BID decoder; bounding boxes are source evidence."""

from __future__ import annotations

import math
from shapely.geometry import box

ALPHABET = "23456789CFGHJMPQRVWX"


def _decode_olc(code):
    """Return a full OLC cell's W/S/E/N bounds and angular grid dimensions."""
    if code.count("+") != 1 or code.index("+") != 8:
        return None
    before, after = code.split("+")
    if "0" in before:
        first = before.index("0")
        if first not in (2, 4, 6) or before[first:] != "0" * (8 - first) or after:
            return None
        digits = before[:first]
    else:
        digits = before + after
    length = len(digits)
    if length not in (2, 4, 6, 8, 10, 11, 12, 13, 14, 15):
        return None
    if any(char not in ALPHABET for char in digits):
        return None
    if ALPHABET.index(digits[0]) >= 9 or ALPHABET.index(digits[1]) >= 18:
        return None
    latitude, longitude = -90.0, -180.0
    lat_step = lon_step = 20.0
    for offset in range(0, min(length, 10), 2):
        lat_step = lon_step = 20.0 ** (1 - offset // 2)
        latitude += ALPHABET.index(digits[offset]) * lat_step
        longitude += ALPHABET.index(digits[offset + 1]) * lon_step
    for char in digits[10:]:
        lat_step /= 5.0
        lon_step /= 4.0
        value = ALPHABET.index(char)
        latitude += (value // 4) * lat_step
        longitude += (value % 4) * lon_step
    return (
        longitude,
        latitude,
        longitude + lon_step,
        latitude + lat_step,
        lat_step,
        lon_step,
        length,
    )


def decode_bid_details(value):
    """Return bbox plus OLC quantization information, or None for invalid bid.

    extent_order is north/east/south/west. Extents count OLC cells, not metres.
    The bbox encloses quantized corner cells; it preserves no footprint shape.
    """
    if not isinstance(value, str):
        return None
    parts = value.strip().upper().split("-")
    if len(parts) != 5 or any(not v.isascii() or not v.isdigit() for v in parts[1:]):
        return None
    cell = _decode_olc(parts[0])
    if cell is None:
        return None
    west, south, east, north, dy, dx, length = cell
    n, e, s, w = (int(v) for v in parts[1:])
    bounds = (west - w * dx, south - s * dy, east + e * dx, north + n * dy)
    if (
        not all(math.isfinite(v) for v in bounds)
        or bounds[0] < -180 - 1e-9
        or bounds[2] > 180 + 1e-9
        or bounds[1] < -90 - 1e-9
        or bounds[3] > 90 + 1e-9
    ):
        return None
    return {
        "geometry": box(*bounds),
        "bounds": bounds,
        "centroid_cell_bounds": (west, south, east, north),
        "latitude_cell_degrees": dy,
        "longitude_cell_degrees": dx,
        "olc_code_length": length,
        "extent_order": "north,east,south,west",
    }


def decode_bid(value):
    """Return a WGS84 Shapely Polygon for a valid NSI/UBID bid, else None."""
    detail = decode_bid_details(value)
    return detail["geometry"] if detail else None
