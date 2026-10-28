# EVRPTW Data Project

Build **Stage 1 City Logistics Environments (CLEs)** from a shared national
source combination and consistent rules across US cities.

[中文说明](README.zh-CN.md) · [Input bundle](docs/INPUTS.md) · [Methods](docs/METHODS.md) · [CLE output](docs/CLE_SCHEMA.md)

A CLE contains the city road network, potential residential delivery locations,
spatial depot groups, charging locations/configurations, and reference travel
costs on the same native directed roads. A customer is a **location**, not an
order: one house or one source-identified apartment complex. Residential units
are nullable auxiliary information.

**This repository ships implementation and policies, not city source data or
prebuilt CLE archives.** Full data distribution and Stage 2 instance activation
are deferred. The following workflow requires a local versioned source bundle.

## Install

Use Python 3.11+ on Linux or macOS; Windows users can use WSL. A separate virtual
environment is recommended. Large cities run sequentially; 32 GB RAM and ample
working disk space are recommended.

```bash
git clone https://github.com/WaitDumplings/EVRPTW-Data-Project.git
cd EVRPTW-Data-Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For extraction from a local OSM PBF, install the optional dependency:

```bash
python -m pip install -e '.[osm]'
```

## Start with one city

```bash
evrptw-data init my-city
```

Place your local source bundle in my-city/inputs/. The generated project.json
selects Dallas by default:

```json
{
  "schema": "evrptw_stage1_project_v1",
  "sources": {
    "inventory": "inputs/inventory.json",
    "root": "inputs",
    "roads": "inputs/roads",
    "customers": null,
    "osm_cohort": null
  },
  "output": "outputs/cle",
  "cities": ["dallas"]
}
```

Paths are relative to project.json, regardless of your current working directory.
A null customer registry prepares Microsoft/NSI sources from the bundle; a
provided registry reuses prepared inputs. Existing complete road extractions
are reused; sources.osm_cohort optionally enables extraction from local PBFs.
See [input formats](docs/INPUTS.md) before preparing your own bundle.

```bash
# List missing inputs before starting expensive computation.
evrptw-data plan my-city/project.json

# Build the complete CLE and perform one final data-validation pass.
evrptw-data build my-city/project.json

# View eligible location counts and saved certificate status.
evrptw-data inspect my-city/outputs/cle
```

Without a source bundle, plan reports the missing inputs. It never substitutes
a rolling latest download for a specified snapshot.

## Multiple cities, resuming and logs

Select several cities in project.json or override them for a run:

```bash
evrptw-data build my-city/project.json --cities dallas fort-worth
```

The reference cities are New York, Los Angeles, Chicago, Houston, Phoenix,
Philadelphia, San Antonio, San Diego, Dallas, Jacksonville and Fort Worth.
Selected cities must exist in your input bundle. State speed profiles currently
cover AZ, CA, FL, IL, NY, PA and TX.

Repeat the same command to resume. Completed components are reused only when
inputs, construction code and output hashes agree. Interrupted partial folders
are retained for diagnosis and that component is restarted. Choose a new output
directory after changing inputs or implementation. Frozen runs are not overwritten.

Progress is printed per city/component. Detailed logs are in output/logs/ and
output/cities/<city>/. The normal pipeline is:

```text
Source preparation (if needed) -> Roads -> Locations and speeds -> CLE assembly -> Data validation
```

Normal builds do not run unit tests, historical comparisons or paper-audit jobs.
Independent data validation can also be deferred and run separately:

```bash
evrptw-data build my-city/project.json --no-verify
evrptw-data verify my-city/outputs/cle
```

The reader requires a certificate bound to the city manifest. An intermediate
output built with --no-verify must be validated before using the reader.

## Read a CLE in Python

```python
from evrptw_cle.cle_reader import CityLogisticsEnvironment

cle = CityLogisticsEnvironment("my-city/outputs/cle/cities/dallas")
locations = cle.locations()
eligible = locations.loc[locations.candidate_eligible]
public_roads = cle.graph("public")
delivery_terminal = cle.graph("delivery_terminal")
charging_terminal = cle.graph("charging_terminal")
access_costs = cle.access_costs()
```

The three graphs remain separate: private terminal access must not become public
through-routing. Retained, eligible and pending locations are distinct counts.
No orders, demands, TW, vehicle fleet or instance activation are generated here.

## Repository layout

```text
src/evrptw_cle/   Construction, access, parameters, assembly and reading
configs/          Shared policies and seven-state speed statistics
scripts/          Internal stage workers used by the CLI
examples/         Project configuration and reading example
docs/             Input contracts, methods and CLE schema
tests/            Development regression tests
```

For development, install `python -m pip install -e '.[dev,osm]'`, then run
`pytest` and `ruff check src scripts tests`. See [NOTICE.md](NOTICE.md) for source
attribution and distribution boundaries.
