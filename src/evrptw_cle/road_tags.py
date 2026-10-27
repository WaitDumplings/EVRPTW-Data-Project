"""Unchanged road-class and tag primitives extracted from the legacy NSI module."""

from __future__ import annotations

import json
from typing import Any

DELIVERY_ROAD_CLASSES = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
}

EXCLUDED_ANCHOR_ROAD_CLASSES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "construction",
    "proposed",
    "raceway",
}


def _parse_multivalue(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    text = str(value or "")
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return tuple(str(item) for item in parsed)
        except json.JSONDecodeError:
            pass
    return (text,) if text else ()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
