# Local input bundle

The generator consumes versioned local inputs. This repository contains the
implementation, policies and compact speed statistics; geographic data bundles
are not distributed in this version. It does not fetch rolling “latest” data.

## Project configuration

Create a project with `evrptw-data init my-city`. Only edit project.json to select
the bundle, output directory and cities. All paths in that file are relative to
its own directory; absolute paths and `~` are also supported.

| Field | Purpose |
| --- | --- |
| sources.inventory | Source inventory JSON supplied with the local bundle |
| sources.root | Root against which inventory file paths are resolved |
| sources.roads | Complete OSM road extractions, one folder per city |
| sources.customers | Optional prepared customer registry; null prepares it from raw inputs |
| sources.osm_cohort | Optional PBF cohort description, used when road extractions are missing |
| output | A fresh output directory, outside the read-only input directories |
| cities | City slugs to build; null builds every city in the inventory |

Run `evrptw-data plan project.json` to see missing files before building. Plan
checks structure and file presence; construction checks source hashes and
compatibility. It cannot turn an arbitrary folder of provider downloads into a
complete input bundle.

## Bundle layout

```text
inputs/
  inventory.json
  boundaries/dallas/{admin,service}.geojson
  cities/dallas/                 # normalized source tables and provenance
  nsi/dallas/                    # native responses, requests, retrieval records
  afdc/                          # station, unit and Census auxiliary snapshots
  roads/dallas/
    osm_ways.parquet
    osm_access_nodes.parquet
    extraction.json
```

The bundle inventory uses schema `evrptw_frozen_source_archive_v1`. It contains:

- files: a map from stable source keys to relative path, sha256 and optional
  private_build_input. archive_path is optional for ordinary local bundles;
  historical archive imports may supply it separately.
- cities: each city's area_crs, state_fips, label, boundaries, sources, depot,
  geometry, nsi_tiles and provenance mappings.
- charging: keys for station, units, census and their corresponding
  station_provenance / units_provenance / census_provenance files.
- source_vintages: provider, release, observation and acquisition information.

For each city, boundaries maps admin_boundary and service_boundary to source
keys. sources maps osm_buildings, osm_complexes and public_masks to source keys.
depot contains raw, extraction and pbf_sha256. These provenance records must bind
the same OSM snapshot as the roads. Source keys refer to entries in files.

Raw customer preparation additionally needs geometry.global and geometry.legacy
lists of Microsoft table keys, and nsi_tiles entries containing tile_id,
response, source, request and request_sha256. The NSI retrieval record must bind
the original request and response. A historical nsi_crosswalk is optional and
is never a prerequisite for constructing a new city.

## Source table contracts

| Input | Format and required content |
| --- | --- |
| Administrative / service boundaries | GeoJSON with a valid CRS; service geometry is the modeled land service domain |
| Microsoft global / legacy extracts | GeoParquet, valid polygon geometry and stable building IDs; see microsoft_composite.py |
| Native NSI tiles | GeoJSON sequence responses with original fd_id, occtype, resunits and point coordinates; request/response provenance |
| OSM buildings | GeoParquet with osm_id, geometry and original building tags |
| OSM residential complexes | GeoParquet with osm_id, geometry, membership / residential tags |
| Exclusion masks | GeoParquet with geometry and exclusion_category; retain privately |
| Depot candidates | GeoParquet with source_osm_id, geometry and original OSM tags |
| AFDC stations / equipment | Original CSV snapshots, stable station/unit IDs and retrieval provenance |
| Census auxiliary matches | The existing address-match snapshot, bound to the AFDC input addresses |
| Road ways / access nodes | Complete source ways and referenced restriction/barrier nodes, with extraction.json and file hashes |

The exact field handling is implemented in microsoft_composite.py,
nsi_evidence.py, residential_sites.py, depot_candidates.py and
charging_component.py. Preserve native attributes; do not manufacture a
different source version by editing checksums in a previous run.

## Prepared customer registry

Set sources.customers to a prepared registry to reuse Microsoft composition and
NSI matching. The registry contains policy `identified_residential_site_v3`,
source_vintages, and a cities map. Each city provides area_crs,
boundary_sha256, sources and provenance. Required source names are
microsoft_footprints, nsi_crosswalk, osm_buildings, osm_complexes and public_masks;
each record contains path and sha256.

Paths in prepared registries follow the existing registry convention: absolute,
or relative to the registry's grandparent directory. The project runner resolves
them to absolute paths in its private working registry. Keep original matching
and retrieval provenance with the prepared tables so independent validation can
check native evidence.

## Optional road extraction

Install the osm extra with `pip install -e '.[osm]'`. An OSM cohort JSON contains
a regions map. Each entry specifies path (PBF), metadata (JSON containing its
sha256), frozen_url and cities. PBF and metadata paths are relative to the cohort
JSON file. For example:

```json
{
  "regions": {
    "texas": {
      "path": "texas-261025.osm.pbf",
      "metadata": "texas.json",
      "frozen_url": "https://download.geofabrik.de/north-america/us/texas-261025.osm.pbf",
      "cities": ["dallas", "fort-worth"]
    }
  }
}
```

Version 0.2 targets the reviewed OSM replication timestamp
`2026-10-25T21:21:21Z` and the corresponding AFDC snapshots. Provider releases,
image years and acquisition dates are different fields. Updating the input
versions requires a compatible versioned policy, rather than a silent download.

## Resuming

Repeat the same build command after interruption. Completed components are
reused only when their input, construction-code and output hashes match.
Changed or incomplete component files fail explicitly. Keep the failed run for
diagnosis and choose a new output directory when inputs or implementation have
changed. The runner does not overwrite a frozen run.
