# External inputs

Only this README belongs in Git. The `.gitignore` excludes the rest of `data/`.
This code project is not the distribution of the large frozen source snapshots.

```text
data/
  external/
    osm/<region>-261025.osm.pbf
    boundaries/2025/<city>/{admin_boundary,land_boundary}.geojson
  derived/<city>/
    microsoft_footprints.parquet
    nsi_crosswalk.parquet
    osm_buildings.parquet
    osm_complexes.parquet
  provenance/<city>/<input-manifest>.json
  private/<city>/public_masks.parquet
  sources/afdc/20261027/afdc_us_public_available_charging_units.csv
```

Use `configs/osm_sources.json`, `city_boundaries.json`, and
`customer_sources.template.json` for expected paths and checksums. The original
snapshot files must be obtained or restored separately. Live redownloads may
differ; a changed hash creates a new source version instead of silently
replacing a frozen file. Do not edit a provenance document merely to make its
hash match a new file.

`extract_frozen_access_roads.py` consumes a caller-supplied cohort manifest
linking regional PBFs, their hash/header metadata, city boundaries and output
locations. It requires the exact accepted OSM replication timestamp. The
current CLI then constructs the road-access component from those extracted
ways and nodes.

Depot construction consumes raw OSM candidates and an explicit source manifest:
`pbf_sha256` must match the road context and `raw_depot_sha256` must match the
raw candidate file. It also verifies the shared exclusion-mask hash. The
candidate kernel preserves cross-operator grouping; available counts use
`facility_group_id` rather than source-member rows.

Do not upload exclusion masks, raw blocked-road inventories, credentials or
unreviewed source caches. The future CLE distribution will need its own
portable input/output manifests and data-license review.
