"""Isolated, restartable city components for Stage 1 only."""

import argparse
import gc
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from evrptw_cle.source_preparation import source_path
from evrptw_cle.util import sha256_file, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("component", choices=["seed", "roads", "customers", "depots", "charging", "speeds"])
p.add_argument("--city", required=True)
p.add_argument("--inventory", type=Path, required=True)
p.add_argument("--source-root", type=Path, required=True)
p.add_argument("--road-sources", type=Path, required=True)
p.add_argument("--customer-registry", type=Path, required=True)
p.add_argument("--output-root", type=Path, required=True)
a = p.parse_args()
i = json.loads(a.inventory.read_text())
c = i["cities"][a.city]
base = a.output_root.resolve() / "cities" / a.city
base.mkdir(parents=True, exist_ok=True)
def resolve(key):
    return source_path(i, a.source_root, key)
boundary = resolve(c["boundaries"]["service_boundary"])
admin = resolve(c["boundaries"]["admin_boundary"])
masks = resolve(c["sources"]["public_masks"])
options = dict(
    source_dir=a.road_sources.resolve() / a.city,
    admin_boundary_path=admin, service_boundary_path=boundary,
    mask_path=masks, area_crs=c["area_crs"],
)
if a.component == "seed":
    from evrptw_cle.local_road_access import build_access_context
    if (base / "_physical_seed").exists():
        raise FileExistsError("A seed already exists; verify before reuse")
    _, report = build_access_context(**options, output_dir=base / "_physical_seed")
elif a.component == "roads":
    from evrptw_cle.delivery_access import build_delivery_access_context
    if (base / "roads").exists():
        raise FileExistsError("A road component already exists; verify before reuse")
    context, report = build_delivery_access_context(
        **options, seed_dir=base / "_physical_seed", output_dir=base / "roads",
    )
    # Preserve the seed manifest, but discard reproducible heavy intermediates.
    shutil.copyfile(base / "_physical_seed/access_context_manifest.json", base / "roads/physical_seed_manifest.json")
    del context
    gc.collect()
    shutil.rmtree(base / "_physical_seed")
elif a.component == "customers":
    from evrptw_cle.residential_sites import build_registered_site_layer
    report, _ = build_registered_site_layer(
        registry_path=a.customer_registry, city_slug=a.city, boundary_path=boundary,
        graph_path=base / "roads/graph_operational.graphml", output_dir=base / "customers",
    )
elif a.component == "depots":
    from evrptw_cle.cli import depots
    raw = resolve(c["depot"]["raw"])
    manifest = base / "depot_source_manifest.json"
    write_json(manifest, dict(
        pbf_sha256=c["depot"]["pbf_sha256"], raw_depot_sha256=sha256_file(raw),
        extraction=dict(path=str(resolve(c["depot"]["extraction"])),
                        sha256=sha256_file(resolve(c["depot"]["extraction"]))),
    ))
    depots(SimpleNamespace(
        city=a.city, area_crs=c["area_crs"], context=base / "roads", boundary=boundary,
        masks=masks, raw_depots=raw, source_manifest=manifest, output=base / "depots",
        footprints=Path(json.loads(a.customer_registry.read_text())["cities"][a.city]["sources"]["microsoft_footprints"]["path"]),
    ))
    report = json.loads((base / "depots/depot_component_manifest.json").read_text())
elif a.component == "charging":
    from evrptw_cle.charging_component import SOURCE_SCHEMA, build_charging_component
    ch = i["charging"]
    manifest = base / "charging_source_manifest.json"
    write_json(manifest, dict(
        schema=SOURCE_SCHEMA,
        **{name + "_sha256": i["files"][ch[name]]["sha256"] for name in ["station", "units", "census"]},
        census_input_station_sha256=i["files"][ch["station"]]["sha256"],
        provenance={name: dict(path=str(resolve(ch[name + "_provenance"])), sha256=i["files"][ch[name + "_provenance"]]["sha256"])
                    for name in ["station", "units", "census"]},
        source_vintages=i["source_vintages"],
    ))
    registry = json.loads(a.customer_registry.read_text())
    footprints = registry["cities"][a.city]["sources"]["microsoft_footprints"]["path"]
    report = build_charging_component(
        city_slug=a.city, station_path=resolve(ch["station"]), units_path=resolve(ch["units"]),
        census_path=resolve(ch["census"]), source_manifest_path=manifest,
        boundary_path=boundary, mask_path=masks, context_dir=base / "roads",
        area_crs=c["area_crs"], output_dir=base / "charging", footprints_path=footprints,
    )
else:
    from evrptw_cle.common_speed_binding import build_common_speed_component
    report = build_common_speed_component(
        city_slug=a.city, context_dir=base / "roads", output_dir=base / "speeds",
        boundary_path=boundary,
        state_fips=c.get("state_fips"),
    )
print(json.dumps(dict(city=a.city, component=a.component, completed=True), sort_keys=True))
