"""Loose building-scale screen for representative NSI household counts.

This is a benchmark consistency proxy, not a legal minimum or exact capacity.
Height must be independently retained Microsoft/OSM building evidence, not an
NSI story count potentially inferred from the same housing-unit estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScaleConfig:
    floor_to_floor_m: float = 3.048
    minimum_unit_area_m2: float = 10.0


def loose_capacity(area_m2, independent_height_m=None, osm_levels=None,
                   config: ScaleConfig | None = None) -> pd.DataFrame:
    """Return one-sided scale diagnostics, preserving missing-assessment cases.

    Where both height and OSM levels exist, use the larger floor proxy.  Ceil
    rather than round/floor intentionally gives roofs and height uncertainty a
    retention-oriented allowance.  All floor area is treated as if residential;
    no residential-area fraction or arbitrary absolute unit cap is introduced.
    """
    cfg = config or ScaleConfig()
    if cfg.floor_to_floor_m <= 0 or cfg.minimum_unit_area_m2 <= 0:
        raise ValueError("scale reference values must be positive")
    area = np.asarray(area_m2, dtype=float)
    n = len(area)
    height = (np.full(n, np.nan) if independent_height_m is None
              else np.asarray(independent_height_m, dtype=float))
    levels = (np.full(n, np.nan) if osm_levels is None
              else np.asarray(osm_levels, dtype=float))
    if len(height) != n or len(levels) != n:
        raise ValueError("area, height and levels must have matching lengths")
    av = np.isfinite(area) & (area > 0)
    hv = np.isfinite(height) & (height > 0)
    lv = np.isfinite(levels) & (levels > 0)
    height_floors = np.where(hv, np.ceil(height/cfg.floor_to_floor_m - 1e-10), np.nan)
    observed_floors = np.where(lv, np.ceil(levels - 1e-10), np.nan)
    floors = np.fmax(height_floors, observed_floors)
    assessed = av & (hv | lv)
    gross = np.where(assessed, area * floors, np.nan)
    # User-selected policy retains a source-supported single-unit building.
    # Thus the benchmark screen has a minimum allowed capacity of one, rather
    # than treating a very small polygon as proof that its one-unit source is
    # impossible.  This is another reason not to call the proxy a strict bound.
    capacity = np.where(assessed, np.maximum(1, np.floor(
        gross/cfg.minimum_unit_area_m2 + 1e-10)), np.nan)
    index = area_m2.index if isinstance(area_m2, pd.Series) else None
    return pd.DataFrame({
        "floor_proxy_from_height": height_floors,
        "floor_proxy_from_osm_levels": observed_floors,
        "floors_upper_proxy": floors,
        "gross_floor_area_proxy_m2": gross,
        "unit_capacity_upper": capacity,
        "scale_assessed": assessed,
        "scale_height_available": hv,
        "scale_levels_available": lv,
    }, index=index)


def annotate_capacity(frame: pd.DataFrame, min_unit_area_m2: float = 10.0,
                      floor_to_floor_m: float = 3.048) -> pd.DataFrame:
    """Add capacity fields using area_m2/height_m/levels.

    ``height_m`` is the same-building Microsoft/OSM fused height. Missing height and
    missing OSM levels leave capacity unassessed. ``capacity_height_missing``
    denotes that combined unassessed height/storey condition.
    """
    f = frame.copy()
    def column(name):
        if name not in f:
            return pd.Series(np.nan, index=f.index)
        return pd.to_numeric(f[name], errors="coerce")
    diagnostics = loose_capacity(column("area_m2"), column("height_m"),
                                 column("levels"), ScaleConfig(
                                     floor_to_floor_m=floor_to_floor_m,
                                     minimum_unit_area_m2=min_unit_area_m2))
    f["capacity_units"] = diagnostics.unit_capacity_upper
    f["gross_floor_upper"] = diagnostics.gross_floor_area_proxy_m2
    f["storeys_upper"] = diagnostics.floors_upper_proxy
    f["capacity_height_missing"] = ~diagnostics.scale_assessed
    f["capacity_height_available"] = diagnostics.scale_height_available
    f["capacity_levels_available"] = diagnostics.scale_levels_available
    return f


def within_capacity(units, capacity) -> np.ndarray:
    """Do not label missing scale evidence as a capacity failure."""
    units = np.asarray(units, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    return ~np.isfinite(capacity) | (units <= capacity)
