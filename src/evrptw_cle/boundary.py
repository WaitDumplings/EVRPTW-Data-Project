from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from .util import sha256_file


def _polygonal_only(geometry):
    geometry = make_valid(geometry)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        areas = [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon))]
        return unary_union(areas) if areas else Polygon()
    return Polygon()


def _normalize_boundary(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        raise ValueError("Boundary source returned no features")
    if frame.crs is None:
        raise ValueError("Boundary source has no CRS")
    frame = frame.to_crs("EPSG:4326")
    geometry = _polygonal_only(unary_union(frame.geometry.tolist()))
    if geometry.is_empty:
        raise ValueError("Boundary contains no polygonal geometry")
    return gpd.GeoDataFrame(
        [{"geometry_role": "administrative_clip"}], geometry=[geometry], crs="EPSG:4326"
    )


def resolve_boundary(
    city_query: str,
    boundary_file: Path | None = None,
    which_result: int = 1,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Resolve a polygon from a local override or OSM Nominatim."""
    if boundary_file is not None:
        frame = gpd.read_file(boundary_file)
        boundary = _normalize_boundary(frame)
        return boundary, {
            "provider": "user_supplied_geojson",
            "city_query": city_query,
            "source_file": str(boundary_file),
            "source_sha256": sha256_file(boundary_file),
        }

    result = ox.geocoder.geocode_to_gdf(city_query, which_result=which_result)
    boundary = _normalize_boundary(result)
    row = result.iloc[0]
    metadata = {
        "provider": "openstreetmap_nominatim",
        "city_query": city_query,
        "which_result": which_result,
        "resolved_display_name": str(row.get("display_name", "")),
        "resolved_osm_type": str(row.get("osm_type", "")),
        "resolved_osm_id": str(row.get("osm_id", "")),
        "resolved_class": str(row.get("class", "")),
        "resolved_type": str(row.get("type", "")),
    }
    return boundary, metadata


def load_optional_query_mask(
    path: Path | None, fallback: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict]:
    if path is None:
        return fallback.copy(), {"provider": "administrative_boundary_fallback"}
    mask = _normalize_boundary(gpd.read_file(path))
    return mask, {
        "provider": "user_supplied_query_mask",
        "source_file": str(path),
        "source_sha256": sha256_file(path),
    }
