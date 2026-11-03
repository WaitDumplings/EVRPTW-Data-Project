# EVRPTW Data Project

Build a City Logistics Environment (CLE) from five linked layers: **roads, reference speeds, residential buildings with units, depots, and charging locations**.

[中文](README.zh-CN.md) · [Methods](docs/METHODS.md) · [Full Stage 1 specification](docs/STAGE1_SPEC.zh-CN.md) · [Run guide](docs/stage1/SERVER_IMPLEMENTATION.zh-CN.md) · [Current building statistics](docs/stage1/STATISTICS.zh-CN.md)

Stage 1 uses Microsoft Global building footprints, OSM building and access evidence, NSI residential records, AFDC equipment, and state EPA speed profiles. Customer units come from one selected NSI record or a calibrated nearest-neighbor estimate. Each value keeps its source and method. The pipeline runs locally or on a server from the same configuration.

For the frozen 11-city server build, use the [Stage 1 server-agent handoff](docs/stage1/SERVER_AGENT_HANDOFF.zh-CN.md). It keeps one joint units calibration for the full cohort and records full-CLE results separately from the local building-attribute baseline.

## Install

Python 3.11+ on Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use `python -m pip install -e '.[osm]'` to extract local OSM PBF files.

## Build a city

```bash
evrptw-cle stage1 init my-city
# Edit my-city/stage1.json with your local source paths.
evrptw-cle stage1 plan my-city/stage1.json
evrptw-cle stage1 run my-city/stage1.json --cities dallas
```

`evrptw-data stage1` and `evrptw-stage1` expose the same commands. Relative paths resolve against `stage1.json`. Supply the source files described in [Inputs](docs/INPUTS.md); the repository contains the implementation and configuration.

`run` prepares and scopes buildings, calibrates the units model, estimates missing units, finalizes customer attributes, and builds the five layers and CLE. To run only building attributes:

```bash
evrptw-cle stage1 run my-city/stage1.json --cities dallas --customer-only
```

Each stage is also available separately through `prepare`, `scope`, `calibrate-units`, `estimate-units`, `finalize`, `components`, and `verify`. The [run guide](docs/stage1/SERVER_IMPLEMENTATION.zh-CN.md) explains inputs, configurable parameters and outputs.

## What the CLE contains

Roads retain native direction, vehicle restrictions and geometry. The road envelope extends 20 km beyond the city boundary; ordinary private roads are included under the delivery-routing assumption. State NEI/StreetLight statistics scale each edge's speed limit. Customers are canonical Microsoft–OSM buildings; explicit logistics parks form depot identities; AFDC supplies charging locations and equipment. All locations use the same road context.

The current 11-city offline attribute run assigns usable residential units to **3,922,741 of 4,012,462 candidate buildings (97.7639%)**: 3,194,252 NSI representatives and 728,489 neighbor estimates. This measures building attribute coverage before road attachment. City totals, unresolved reasons and maximum unit counts are in [Statistics](docs/stage1/STATISTICS.zh-CN.md).

Each assembled city includes `roads/`, `speeds/`, `customers/`, `depots/`, `charging/`, `locations.parquet`, `location_access_costs.parquet`, and manifests. See [CLE schema](docs/CLE_SCHEMA.md).

```python
from evrptw_cle.cle_reader import CityLogisticsEnvironment

cle = CityLogisticsEnvironment("my-city/outputs/stage1-building-first/cities/dallas")
locations = cle.locations()
```

## Stage 2 delivery instances

Stage 2 reads a verified CLE to sample a depot and a fixed number of customer locations, generate delivery orders, select charging stations, and build directed road costs. Start with the [Dallas Cus100 review configuration](examples/stage2/server_review.json) and the [Stage 2 run guide](docs/stage2/REVIEW_PACKAGE.zh-CN.md). The [combined server handoff](docs/SERVER_HANDOFF.zh-CN.md) describes both stages and the evidence required for each. Historical data and entrypoints are documented in [Historical CLE](docs/HISTORICAL_CLE.md).
