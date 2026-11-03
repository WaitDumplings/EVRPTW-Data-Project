# Stage 1 methods

Stage 1 constructs a persistent City Logistics Environment in five layers. The [full specification](STAGE1_SPEC.zh-CN.md) gives equations, source vintages and implementation details; the [run guide](stage1/SERVER_IMPLEMENTATION.zh-CN.md) provides the executable workflow.

## 1. Roads

Census TIGER/Line 2025 city polygons define administrative membership; the modeled service surface removes mapped water. The road selection envelope extends 20 km beyond the administrative boundary. Complete OSM ways, native node sequences, tags and restriction nodes come from the common 2026-10-25 snapshot.

Native junctions and restriction nodes define directed segments. One-way direction, vehicle/delivery prohibitions and physical barriers are preserved. Ordinary private roads participate in delivery through-routing under the configured modeling assumption. Locations attach to eligible local roads through a geometry connector and, where needed, real directed terminal paths to the reference strongly connected component. Customer connectors use the complete canonical footprint.

## 2. Reference speeds

OSM supplies edge-specific speed limits and road classes. Valid directional limits precede generic limits; missing limits use length-weighted city medians through the documented highway/group/parent/global hierarchy.

State EPA 2023 NEI MOVES source-type-32 speed distributions, EPA StreetLight commercial-vehicle hourly profiles, and 2023 monthly VMT provide reference operating ratios. OSM motorway/trunk classes use MOVES urban restricted-access road type 4; the remaining supported roads use urban unrestricted-access type 5. Each edge receives weekday and weekend speeds by multiplying its legal limit by the corresponding state/road-type ratio, capped at that limit. The default effective-speed window is 08:00–24:00; the low-flow reference is the pooled weekend 06:00–10:00 speed-bin Q85. The specification records these reference-profile assumptions and citations.

## 3. Residential buildings with units

Microsoft Global 2026-08-13-index footprints define building geometry. Complete OSM buildings provide uses, aliases, and explicit child footprints for subdividing composite Microsoft outlines. The candidate population contains clear Microsoft–OSM entities inside the service domain, after explicit nonresidential/mixed and clear unknown-use zero-unit exclusions. Unresolved candidates remain in the coverage denominator.

NSI 2026 ordinary residential positive-unit records supply the original units evidence. The spatial matcher retains containment, source footprint identifiers, source bounding-box/area evidence and axis-normalized displacement. A source record, genuine source-footprint group and building may each participate in at most one selected assignment. Multiple NSI records are resolved by a representative assignment; their units are never summed.

Clear original matches form the reference pool. A 2 km spatial-block split keeps related source groups together and separates training, calibration and test data. The same training reference supplies calibration and final inference. Neighbor inputs are location, polygon area, independent building height/OSM levels and OSM typology. Target units and NSI occupancy bands are not predictor features.

For each city/type/available-height branch, geographic reference pairs fit RMS scales for one radial location difference, log-area difference and the independent log-height/level difference. Each available group has equal mean squared contribution on that fitting pair sample. Eight neighbors in the resulting metric produce a distance-weighted median of log units. Calibration chooses joint similarity/dispersion limits retaining the configured share of held-out references; the default is 95% reference support. A loose polygon-area/floor proxy screens implausible unit counts. Multi-unit assignments require an assessed and passing scale screen; source-supported single-unit buildings retain the configured exception. Unassessed multi-unit candidates retain a reason and stay in the denominator.

The output records `units_source=nsi|neighbor|unresolved`, the selected source ID, `units_method`, original and final values, independent/estimated height provenance, and support/scale diagnostics. See [customer algorithm](stage1/CUSTOMER_METHOD.zh-CN.md) and [current 11-city statistics](stage1/STATISTICS.zh-CN.md).

## 4. Depots

OSM logistics and warehouse evidence identifies facility candidates. Dispatch/carrier facilities and logistics proxies, including ordinary logistics offices, retain their evidence tier. Native duplicates and supported facility aliases are resolved. Explicit logistics-park identities or boundaries merge their facilities into one depot location across operators; generic industrial land does not merge unrelated facilities. Source members and their individual directed road access remain available.

## 5. Charging locations and equipment

AFDC supplies public operating stations, native coordinate precision and device configurations from the archived 2026-10-26/27 responses. GPS/Point coordinates qualify independently; address-level records use the archived Census address response as auxiliary consistency evidence. Same-coordinate, same-complete-address records share a location; conflicts retain their status and members.

Compatible configurations remain separate. A named reference van uses AC J1772 and DC CCS1 with 11/100 kW input caps. Valid reported powers precede mode-specific national port-weighted median fills, and one reference configuration is selected per location without summing ports. Native power, assigned power and vehicle-capped power remain distinct. Directed access uses the common road context. Details and exact acquisition times appear in the [facility specification](stage1/DEPOT_CHARGING.zh-CN.md).

## CLE and downstream use

Assembly binds the five layers through stable location IDs, source IDs and common native directed edges. Building attribute coverage and road-available location counts are reported separately. Stage 2 selects locations and creates demand, time windows, service times and vehicle instances. The current Stage 1 construction is configured and recorded independently from that activation step.
