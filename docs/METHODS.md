# Stage 1 modeling rules

One national source combination and the same algorithms are used across cities.
The model represents potential delivery locations and road access, not observed
orders or a survey of property entrances.

## Cities, boundaries and roads

The reference cohort comprises New York, Los Angeles, Chicago, Houston, Phoenix,
Philadelphia, San Antonio, San Diego, Dallas, Jacksonville and Fort Worth.
Administrative geometry and the modeled land service domain are separate inputs.
The road envelope may extend outside the city to preserve connectivity.

Complete OSM ways retain native direction and access-node evidence. Restrictions
and barriers split source segments. A buffer ladder selects the smallest public
envelope recovering the city nodes reachable at the maximum envelope. The
reference routing component is a deterministic directed SCC.

The public graph is separate from purpose-specific terminal access. Ordinary
private access may be modeled as invited delivery, with explicit assumption
flags; this does not establish observed owner consent. It cannot become public
through-routing or automatically grant permission to visit a charger. Explicit
prohibitions, hard barriers and identified sensitive facilities are excluded.

## Residential locations

Microsoft Global footprints are primary; legacy footprints fill supported gaps
without duplicating accepted global shapes. NSI native occupancy codes determine
residential classification; original modeled attributes and point positions are
preserved. OSM buildings and residential complexes support site aggregation.

A house is one potential location. A source-identified apartment complex is one
location even if it contains multiple buildings. Unidentified complex extent
remains explicit uncertainty. residential_units is nullable auxiliary evidence,
independent of location count and building count. It is not realized demand.

Components have unique ownership. Competing site/complex scopes or shared roofs
without sufficient evidence remain pending and have no operational unit total;
native evidence is retained. Missing units alone do not reject a residential
location. Housing classification does not use local parcel standards.

## Depot locations

Explicit dispatch facilities and warehouse proxies form the source pool.
Ordinary logistics offices remain Tier B. Industrial-only evidence is not
promoted to a confirmed departure facility. Spatial grouping allows multiple
operators at one location, using co-location, overlap and supported address or
campus evidence. Generated display names such as “Unnamed warehouse” cannot
justify a same-name merge. Source rows remain traceable beneath the group.

## Charging locations and parameters

AFDC native GPS/Point coordinates have independent admission; address-level
coordinates retain auxiliary corroboration. OSM address agreement does not
replace valid native coordinates. Same-place grouping preserves original
members, equipment and access conditions. Conflicting source identities remain
pending; coordinates are not corrected with per-ID override tables.

Compatibility is evaluated before power. A compatible configuration can be
selected from safe co-located members, with native source/unit keys retained.
Reported power, missing-value inference and the reference vehicle's AC/DC caps
are separate fields. Power and port counts are not summed across aliases.
The derating coefficient is an explicit model assumption and is not applied
twice; operating-day charging behavior belongs to Stage 2.

## Connecting locations to roads

Customers use their footprint/site extent. Depots use an internal representative
point, retaining source extent for local search. Anchors exclude motorway/trunk
frontage and check direction, other buildings, water and excluded facilities.
Local alternative candidates retain the same permission and obstacle checks.
Closer unsupported motor frontage (track/construction) triggers a shared
pending rule; no city or location-ID exception is used.

A geometric connector and its native directed terminal ingress/egress chains
are different objects. Partial source edges use metric directional offsets,
including closed ways; geometry lengths and proportional edge costs agree.
One kilometre is a diagnostic reference, not a source-deletion threshold.

## Speeds and costs

Legal/reference costs share the actual native directed edges across public,
delivery-terminal and charging-terminal memberships. The private calculation
union is not an additional public routing graph. Reference weekday/weekend
factors use the same computation with state statistics for AZ, CA, FL, IL, NY,
PA and TX: 2023 NEI speed distributions and StreetLight 2022v2 hour/day weights.
The compact source SQL and statistical provenance are included in configs/.
HPMS matching is optional; no observation is invented when no HPMS input exists.

Native edge time is 3.6 × length_m / speed_kph. Partial-edge and terminal costs
sum the actual source fractions. Connector timing stores 3.6 × length_m as a
coefficient; the operating scenario supplies its speed in Stage 2.

## Source dates and model scope

| Source | Reference version / interpretation |
| --- | --- |
| Census | 2025 city ranking/boundary reference |
| OSM | 2026-10-25T21:21:21Z |
| Microsoft | Global 2026-08-13 plus frozen legacy gap-fill; imagery years mixed or unknown |
| NSI | Acquired 2026-10-26; modeled attributes have mixed underlying vintages |
| AFDC | Station snapshot 2026-10-26; equipment retrieved 2026-10-27 UTC / 2026-10-26 local |
| EPA / StreetLight | 2023 NEI and 2022v2 statistics with 2022–2023 observations |

These layers are a versioned reference model, not simultaneous same-year
observations. Turn-restriction relations, conditional traffic rules, vehicle
dimensions, dynamic congestion, surveyed entrances and charging queues are not
modeled in this static stage. Fixed gateway choices do not prove globally
optimal routing over all real entrances. Source uncertainty is retained rather
than repaired manually. Stage 2 will define activation, demand, TW, service
times, vehicle scenarios and instance-level routing costs.
