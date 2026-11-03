# Stage 1 inputs

Initialize a project with `evrptw-cle stage1 init <directory>`, edit `stage1.json`, then run `evrptw-cle stage1 plan <config>`. Relative paths resolve against the configuration file. The [run guide](stage1/SERVER_IMPLEMENTATION.zh-CN.md) gives a complete example.

## Project configuration

| Field | Content |
|---|---|
| `schema` | `evrptw_building_first_stage1_v1` |
| `output` | New run directory |
| `cities.<city>.microsoft` | Microsoft Global footprint table |
| `cities.<city>.osm` | Complete OSM building table, including unknown/nonresidential competitors |
| `cities.<city>.nsi` | Original NSI point GeoParquet with native attributes |
| `cities.<city>.area_crs` | City metric CRS for geometry and neighbor distances |
| `cities.<city>.service_boundary` | WGS84 land service surface |
| `cities.<city>.admin_boundary` | WGS84 administrative polygon |
| `components.inventory` | Road/facility source-inventory JSON |
| `components.source_root` | Root for inventory paths |
| `components.road_sources` | Per-city complete OSM extraction directories |
| `source_snapshots` | Source index, snapshot and acquisition metadata |
| `units` | Reference split, RMS fitting, support calibration and scale-screen parameters |
| `customer.active_profile` | `unrestricted` for the current positive-unit population |
| `customer.include_model_imputed` | Include separately labeled neighbor estimates |

## Source tables

| Input | Required identity and evidence |
|---|---|
| Microsoft Global | WGS84 Polygon/MultiPolygon GeoParquet; stable `building_id`, `source_kind=global`, index/source metadata, original geometry and available independent height |
| OSM buildings | Namespaced native `osm_id`, complete polygon, building/building:use and functional tags, all raw tags, independent height/levels, native version/time |
| NSI | WGS84 points; `fd_id`, `occtype`, `resunits`, available `bid`, `ftprntsrc`, `ftprntid`, `ftprntsqft`, `bldheight`, native coordinates and archived request provenance |
| Boundaries | TIGER administrative polygon and modeled land service polygon |
| OSM logistics | Native point/area identity, full geometry/tags, source provenance and explicit park identity/membership when available |
| AFDC stations | IDs, original coordinates and precision, address, public/status/access evidence and archived response metadata |
| AFDC equipment | Native station/member binding, device row, connectors, ports and mode power |
| Census address auxiliary | Archived responses bound to exact AFDC input addresses |
| OSM roads | Complete selected way geometry and node sequence, raw tags, restriction/barrier nodes and extraction provenance |

Native NSI zero-unit records are retained as source evidence for unknown-use exclusions and zero/positive conflicts. The positive residential matching pool is formed separately from ordinary RES1/RES2/RES3 records with valid positive units. A precomputed MS–NSI crosswalk is not required.

Longitude/latitude and full source polygons are retained. Areas, matching residuals, spatial blocks and neighbor distances use `area_crs`. An NSI BID-derived bounding box is source evidence; it is not the original building polygon. NSI modeled heights/storeys are not used as independent height features for estimating that record's units.

## Fixed source vintages

| Source | Version/acquisition | Meaning |
|---|---|---|
| Census Places / Area Hydrography | TIGER/Line 2025 | City and water boundaries |
| Microsoft GlobalMLBuildingFootprints | 2026-08-13 index snapshot | Index date; imagery dates remain source-specific |
| OSM regional PBF | 2026-10-25T21:21:21Z | Common replication snapshot for roads/buildings/facilities |
| NSI 2026 | Archived 2026-10-26 | Mixed upstream/model vintages; not a 2026 building census |
| AFDC station | 2026-10-26T23:28:25.956900Z | US public operating electric station inventory |
| AFDC equipment | 2026-10-27T00:03:57.499877Z | Separate equipment response; local Los Angeles date 10/26 |
| Census address auxiliary | 2026-10-26T23:47:41.254276Z | Public_AR_Current address responses |
| EPA 2023 NEI MOVES | 2023 activity/model inputs | Source-type-32 speeds; StreetLight observations 2022-01–2023-05 and EPA fills |
| EPA 2022v2 commercial temporal | Effective observations 2022-11–2023-05 | Hour/day VMT weights |
| NEI monthly VMT | Activity year 2023 | Annual representative weekday/weekend aggregation |

Keep observation, model, release and acquisition dates separately, together with file hashes, URLs and query parameters. An index date does not replace a missing imagery date.

## Road and facility bundle

```text
inputs/
  inventory.json
  boundaries/<city>/{admin,service}.geojson
  customer/<city>/{microsoft_footprints,osm_buildings,nsi}.parquet
  roads/<city>/
    osm_ways.parquet
    osm_access_nodes.parquet
    extraction.json
  afdc/
  pbf/
```

The source inventory uses `evrptw_frozen_source_archive_v1`: stable file keys map to paths/hashes; city entries hold `area_crs`, state, boundaries, depot evidence, geometry and source provenance; charging entries bind station/equipment/address responses. Raw tags include directional speed, one-way, motorcar/motor_vehicle/vehicle/access, delivery, locked and barrier values.

The selected road envelope may cross a regional PBF boundary. Prepare the union from the same snapshot, deduplicate native OSM objects, then extract complete ways and referenced nodes. Preserve parent PBF and derived extraction hashes.

## Reuse and migration

A configured `geometry_cache` is an explicit import of prepared geometry and source ownership. Its manifest records that mode and available original-source bindings. Raw rebuilding uses the three city input files directly. Keep raw and cached modes distinct in run provenance.

When moving to a server, update local paths in the project and nested inventory; copying a top-level JSON does not rewrite embedded paths. Use a new output directory for changed inputs or construction parameters. The package supplies code and profiles; the source bundle is supplied locally.
