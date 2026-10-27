# EVRPTW Data Project

Reviewed geographic components for a reproducible model of potential urban
delivery locations. This is the foundation for a later City Logistics
Environment (CLE) and EVRPTW instance generator.

The current scope is city/road access, residential customer locations, spatial
depot groups, charging-location/equipment rules, and state speed statistics.
Full CLE assembly and instance activation are not included in this version.

| Component | Current status |
| --- | --- |
| Road/customer/depot access rules | Implemented; upstream 11-city production independently verified |
| Customer/depot location components | Frozen: 4,214,699 customer locations and 7,555 depot groups |
| Pending source-safe locations | Retained: 42,638 customer locations and 232 depot groups |
| Charging identities/equipment and speed parameters | Reviewed kernels included; rebinding to the current road graph remains pending |
| Complete CLE schema and construction | Not frozen; no assembled CLE release supplied |
| Instance activation, demand, TW, service time, solvers | Not included; reviewed after the CLE freeze |

The production counts describe the verified upstream run. This new project
contains a selected and tested code port, compact statistics, policies, and
summary evidence. It does not distribute the upstream raw snapshots or claim
that a complete 11-city dataset has been rebuilt from this checkout.

The source kernels include new-primary/legacy-gap footprint composition,
archived native NSI evidence parsing, and unique geometric containment matching.
The superseded unit-count-to-housing-type classifier and building-only pilot
constructor are excluded. NSI occupancy bounds remain raw evidence; the site
builder derives nullable units independently and classifies housing from native
NSI occupancy codes.

## Install and inspect

```bash
python -m pip install -e '.[dev]'
python -m evrptw_cle status
pytest
ruff check src tests scripts
python scripts/verify_state_speed_profiles.py
```

Use `pip install -e '.[dev,osm]'` when running the optional PBF extractor.
Some source acquisition helpers also require the system `osmium` tool.
`requirements-tested.txt` records the local validation environment; it is not
a cross-platform lock file.

The public commands are `status`, `validate-inputs`, `roads`, `customers`, and
`depots`. Run `python -m evrptw_cle <command> --help` for required inputs.
Road builds use a fresh output directory. Customer/depot commands require the
reviewed `delivery_terminal_access_v2` context; they do not silently substitute
the historical road pipeline.

## Core modeling rules

- A customer location is one independent house or one source-identified
  apartment complex. A complex can contain several buildings. Nullable
  `residential_units` is auxiliary evidence; it is not a building count,
  package count, realized demand, or an activation rule.
- Depot identity is a spatial group, allowing multiple logistics operators
  at the same location. Ordinary logistics offices remain Tier B. Neither
  company count nor area is used as a hard eligibility gate or sampling weight.
- Customer access uses the full footprint/site extent. Depot access uses its
  internal representative point; native extent only scales local search.
- Ordinary private ways/gates are modeled as invited-delivery terminals.
  `delivery_authorization_assumed=true` does not imply observed owner consent.
  Private, destination/customers, and service roads stay outside public
  through-routing. Explicit prohibitions, unsupported barriers and identified
  sensitive facilities remain excluded.
- Soft building/connectivity failures may check up to eight strict-permission
  local ways. All direction and obstacle checks remain active. A distance of
  1 km is a QA reference, never an automatic source-location deletion rule.
- Terminal ingress/egress are real directed OSM edge chains to public-SCC
  gateways. They are separate from the geometric source-to-road connector.
  Pair materialization includes only source egress and destination ingress.

These are explicit benchmark assumptions, not a survey of entrances, owner
permits, company operations, or every real residential complex. Conditional
access, turn-restriction relations and unmapped barriers remain outside the
current model. Fixed gateway choices do not prove global OD optimality, and
purpose-specific directed pair costs must not be assumed symmetric or metric.

## Data, years and reproduction

The same national source combination is used across the eleven cities:
Microsoft Global 2026-08-13 plus frozen legacy gap-fill, NSI acquired
2026-10-26, and OSM frozen at `2026-10-25T21:21:21Z`. Microsoft imagery years
and NSI attribute years are mixed/partly unknown. Release, observation and
acquisition dates are different fields.

Charging rules use the frozen AFDC station and charging-unit sources. State
speed factors use 2023 NEI speed distributions and EPA 2022v2 StreetLight
hour/day statistics, covering 2022–2023 observations. The compact seven-state
canonical SQL inputs are included, with checksums and source archive references,
so the coefficients can be reproduced without the large geographic snapshots.

See [methods and assumptions](docs/METHODS_ZH.md),
[input preparation](data/README.md), and
[remaining integration gates](docs/INTEGRATION_GATES.md).
`configs/customer_sources.template.json` is an immutable-input inventory,
not supplied customer data. Copy it to `configs/customer_sources.json`, stage
the exact archived inputs, and run:

```bash
python -m evrptw_cle validate-inputs configs/customer_sources.json
```

Missing or changed files fail explicitly. A fresh live API download cannot
be assumed to reproduce a frozen, unversioned source snapshot. Large data,
raw API caches, exclusion masks and blocked-road inventories stay outside Git.
The supplied source inventory preserves the upstream archived file hashes.
Regenerating derived files with the new portable wrappers creates new artifacts
and requires an explicitly versioned source registry and verification, rather
than changing the old inventory to make it pass. Source acquisition/hydration
and a single complete cohort runner still need integration.

## Evidence and release boundary

- [11-city count comparison](evidence/city_access_comparison.csv)
- [Source/model audit](evidence/model_source_audit.csv)
- [Production verification reference](evidence/production_reference.json)
- [Porting provenance](evidence/upstream_code_provenance.json)
- [Local validation](evidence/local_validation.json)
- [Installed-wheel smoke test](evidence/wheel_installation_smoke.json)

The selected project passed 131 component tests and lint checks. Nineteen
extracted core functions match their upstream ASTs, all seven state statistics
were reproduced, and two Dallas OD paths were replayed against the external
verified graph. An installed wheel was checked outside the source tree using
the recorded local dependency environment.

[GitHub Actions template](ci/github-actions-tests.yml) is supplied but not
enabled. The current GitHub credential lacks the `workflow` scope, so the
initial push cannot install a workflow. Once workflow-writing access is
available, place the template at `.github/workflows/tests.yml` to enable it.
Remote CI has not run; the validation above was performed locally.

The verified upstream run checked 1,742,450 terminal paths. Its evidence
includes preserved source geometry, identities, NSI ownership and nullable
units. The original full repository still has six historical environment or
launcher checks outside this selected project's scope; they are recorded in
the reference evidence, not silently described as passing.

The initial repository is private at
[WaitDumplings/EVRPTW-Data-Project](https://github.com/WaitDumplings/EVRPTW-Data-Project).
Code licensing remains a decision before public release; see [NOTICE.md](NOTICE.md).
No complete CLE/instance archive, Amazon-derived operating-day data, or solver
experiment result is released here.
