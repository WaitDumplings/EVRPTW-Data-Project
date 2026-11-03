# CLE output and Python interface

The building-first CLE schema is `evrptw_building_first_cle_v1`. Construct it with `evrptw-cle stage1 run <config>`; inputs and parameters are documented in the [run guide](stage1/SERVER_IMPLEMENTATION.zh-CN.md).

Each city is a persistent geographic environment. Its core identity is `(city_slug, location_id)`.

```text
<output>/
  customer_coverage.csv
  cities/<city>/
    customer_inputs/
    boundaries/
    roads/
    speeds/
    customers/
    depots/
    charging/
    locations.parquet
    location_access_costs.parquet
    cle_manifest.json
    stage1_validation.json
```

| Layer | Contents |
|---|---|
| `boundaries` | Administrative and modeled land service surfaces |
| `roads` | Native physical/directed edges, delivery route graph, reference SCC and remaining purpose-terminal access |
| `speeds` | Edge limits, imputation source, state profile, weekday/weekend reference costs and graph membership |
| `customer_inputs` | Canonical population, service scope, original NSI pool/ownership, reference split, fitted RMS scales/support gates, representative assignment and unresolved reasons |
| `customers` | Building locations, unit provenance and full-footprint road attachment |
| `depots` | Canonical facility/park locations, original members and directed access |
| `charging` | Canonical AFDC locations, members, all equipment configurations and selected reference configuration |

## Customer attributes

The canonical building polygon and `entity_id` identify the location. Microsoft composite parents retired during OSM subdivision remain in provenance; the child polygon supplies the final area and access geometry.

| Field | Meaning |
|---|---|
| `units_raw_nsi` | Available original value; all individual observations remain in the NSI ledger |
| `units_final` / `residential_units` | Positive integer used for the retained building, otherwise null; layer-specific names map to the same final attribute |
| `units_source` | `nsi`, `neighbor` or `unresolved` |
| `units_method` | `nsi_representative` or `neighbor_estimated` for an assigned value; unresolved rows retain an unresolved status |
| `selected_nsi_fd_id`, `selected_nsi_resunits` | Selected original NSI record/count; retained as evidence even if the final scale gate rejects the assignment |
| `units_scope` | Building scope of the final modeled attribute |
| `residential_use` | Native/derived use evidence, separate from final units |
| `use_type` | OSM house/apartments/generic branch used by the neighbor model |
| `height_original_m`, `levels` | Original independent Microsoft/OSM height and OSM storeys |
| `capacity_units`, `capacity_height_missing` | Loose unit-capacity proxy and missing-assessment flag; a known passing proxy is required for multi-unit admission |
| `height_neighbor_estimated` | Whether a missing height was estimated for the loose scale screen |
| `neighbor_*` | Metric cost, neighbor counts, unit dispersion, reference IDs and geographic distances |
| `quality_*` | Calibration stratum, cost/dispersion limits and support decisions |
| `result_reason` | NSI representative, neighbor estimate or the mutually exclusive unresolved reason |

One NSI record and one genuine source-footprint group supply at most one selected building. `selected_nsi_fd_id` points to that representative. Multiple source units are not summed, and a neighbor estimate does not overwrite the original NSI observation. Fitted scales and support gates bind to the source/reference/configuration hashes.

## Eligibility and access

Building attribute coverage counts assigned positive units in the fixed candidate population. It is measured before road attachment. `candidate_eligible` in the canonical location table additionally reflects the component's static rules and access state. Pending records retain their source and reason.

`location_access_costs.parquet` stores geometry connector lengths, native directional offsets, ingress/egress gateways and any terminal paths to the common route graph. Road portions use the bound edge speeds. Connector lengths and conversion coefficients let the downstream scenario choose its connector speed.

A depot location references one stable usable access member and retains all facility/park members. Charging locations retain AFDC native members, public/access conditions and compatible configurations; native, assigned and vehicle-capped power fields remain distinct.

## Reading

```python
from evrptw_cle.cle_reader import CityLogisticsEnvironment

cle = CityLogisticsEnvironment("outputs/stage1-building-first/cities/dallas")
locations = cle.locations()
access = cle.access_costs()
route = cle.graph("public")
delivery_access = cle.graph("delivery_terminal")
charging_access = cle.graph("charging_terminal")
```

The graph name `public` is the reader's route-graph key; its saved policy specifies modeled ordinary-private through-routing. Purpose-terminal edges keep their own directional access semantics. The reader binds each graph to its matching native edge weights and checks saved file hashes.

Assembly records inputs, method/configuration, output hashes and common-road bindings in `cle_manifest.json`. `stage1_validation.json` records saved-data validation; `stage1 verify` runs it separately. Historical CLEs retain their own source/code/validation receipts.

Stage 2 reads stable building/depot/charging identities and final unit provenance, then generates delivery-day activation, packages, time windows, service times and instance matrices.
