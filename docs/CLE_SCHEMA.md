# CLE output and Python interface

Each city is a persistent Stage 1 environment. It has no activated orders or
operating-day vehicle fleet.

```text
outputs/cle/
  build_status.json
  stage1_verification.json
  logs/
  cities/dallas/
    cle_manifest.json
    cle_verification.json
    locations.parquet
    location_access_costs.parquet
    boundaries/
    roads/
    customers/
    depots/
    charging/
    speeds/
    private_sources/
```

| Layer | Contents |
| --- | --- |
| boundaries | Administrative and modeled service geometry |
| roads | Public graph, delivery terminal graph, physical roads and directed access evidence |
| customers | Residential locations, native NSI members, source-scope ledger and connectors |
| depots | Source candidates, cross-operator spatial groups and access rows |
| charging | Native source members, equipment configurations and charging-purpose access |
| speeds | Native directed-edge legal/reference weekday/weekend costs and graph memberships |
| private_sources | Local raw evidence needed for independent checks; not a public export |

locations.parquet is the canonical index. location_id is type-prefixed and
unique inside a city. Across cities use (city_slug, location_id). Fields include
location_type, source_location_id, geometry, candidate_eligible, pending_reason,
nullable residential_units and source-table/row foreign keys. A depot group
selects a deterministic usable source row as its access representative; all
members remain in the depot layer.

candidate_eligible is a static Stage 1 qualification. Pending locations remain
in the CLE but are excluded from the default eligible pool. A historical native
diagnostic column does not override the canonical eligibility and pending reason.

location_access_costs.parquet binds usable locations to connector length and
native terminal ingress/egress costs. Connector speed, orders, service times,
vehicle selection and dense OD matrices are deferred to Stage 2.

## Reading

```python
from evrptw_cle.cle_reader import CityLogisticsEnvironment

cle = CityLogisticsEnvironment("outputs/cle/cities/dallas")
locations = cle.locations()
costs = cle.access_costs()
public = cle.graph("public")
delivery = cle.graph("delivery_terminal")
charging = cle.graph("charging_terminal")
```

The reader requires an independent certificate matching the city manifest and
checks saved file hashes by default. Each graph receives its matching native
edge weights. Never compose terminal graphs into an unrestricted public graph.

## Validation and versioning

build performs one independent data-validation pass by default. Use verify to
run it separately. This validates saved files and never runs unit tests or
compares against the developer's historical production results.

inspect displays saved counts and certificate/manifest binding; it is a quick
metadata view and does not recheck every data byte. The reader and verify perform
the stronger checks.

A frozen historical CLE keeps its saved code and input version. Use that version
for re-auditing; this repository's newer implementation is not a substitute for
the historical verifier. Builds never overwrite a stage1_freeze.json run.
