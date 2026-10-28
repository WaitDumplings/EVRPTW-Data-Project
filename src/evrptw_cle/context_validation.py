"""Reject cross-city or cross-version component inputs before construction."""

from pathlib import Path

from pyproj import CRS

from .util import sha256_file


def validate_generation_dependencies(manifest):
    """A topology version alone cannot authorize reuse after code changes."""
    dependencies = manifest.get("generation_dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("Road context lacks generation dependency hashes; rebuild it")
    required = {"local_road_access.py", "terminal_access.py", "road_tags.py"}
    if manifest.get("policy") == "delivery_terminal_access_v2":
        required |= {"delivery_access.py", "public_locations.py", "frontage.py"}
    if not required <= dependencies.keys():
        raise ValueError("Road context omits required generation dependencies; rebuild it")
    for name, digest in dependencies.items():
        if Path(name).name != name:
            raise ValueError("Road context generation dependency must be a module filename")
        source = Path(__file__).with_name(name)
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError("Road context generation code changed; rebuild it: " + name)


def validate_context_bindings(
    manifest, *, city_slug=None, boundary_path=None, mask_path=None, area_crs=None
):
    """Bind location inputs to the same service domain and exclusions as roads.

    A hash-valid file from a different city or policy version is not a valid
    input for this context. Boundary here means the land/service boundary,
    rather than the larger administrative routing envelope.
    """
    if manifest.get("policy") != "delivery_terminal_access_v2":
        raise ValueError("The reviewed delivery-purpose access context is required")
    if manifest.get("topology_contract") != "source_direction_and_access_node_segmentation_v2":
        raise ValueError("Road context predates reviewed direction/barrier topology; rebuild it")
    validate_generation_dependencies(manifest)
    if city_slug is not None and manifest.get("city_slug") != city_slug:
        raise ValueError("Location city differs from road context city")
    for path, field, label in [
        (boundary_path, "service_boundary_sha256", "service boundary"),
        (mask_path, "mask_sha256", "exclusion masks"),
    ]:
        if path is not None and sha256_file(Path(path)) != manifest.get(field):
            raise ValueError(f"Location {label} differ from road context")
    if area_crs is not None and (
        "area_crs" not in manifest
        or CRS.from_user_input(area_crs) != CRS.from_user_input(manifest["area_crs"])
    ):
        raise ValueError("Location metric CRS differs from road context")
