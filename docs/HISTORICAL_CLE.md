# Reading and re-auditing a frozen CLE

A frozen CLE keeps its original receipt, city files, certificates and
`code_snapshot/`. Current construction code can evolve without changing that
historical version. These are three separate workflows.

## Use a CLE for Stage 2

Read it normally:

```python
from evrptw_cle.cle_reader import CityLogisticsEnvironment

cle = CityLogisticsEnvironment("outputs/stage1_complete_20261027/cities/dallas")
locations = cle.locations()
```

The reader checks the saved city files and certificate binding. This does not
require the original PBF, Microsoft or NSI source paths. The existing Stage 2
reader uses this path rather than reconstructing a historical access context.

## Re-audit the original construction

First inspect the receipt-bound implementation and source inventory location:

```bash
evrptw-data history outputs/stage1_complete_20261027
```

Then run its original verifier and policies:

```bash
evrptw-data verify outputs/stage1_complete_20261027 \
  --historical --report-dir outputs/stage1_reaudit_20261031
```

The command verifies every saved implementation file against the freeze receipt,
then runs one city subprocess at a time with the saved package on `PYTHONPATH`.
It preserves the original policy bytes and hash comparisons; it does not treat
JSON reformatting as byte identity. All new reports and logs go to the fresh
report directory outside the frozen cohort. The old receipt, certificates and
city files are not overwritten. Without `--report-dir`, a new timestamped sibling
directory under `<cohort>_reaudits/` is used.

Some original provenance records contain absolute construction-time policy paths
in addition to the saved package's policy root. A small, explicitly recorded
resource resolver redirects only those declared policy references to their
receipt-bound `code_snapshot/configs/` copies. The expected SHA must match both
the saved declaration and the actual replacement bytes. The audit records the
original-to-saved resource map, its hash, and the resolver implementation hash
separately from the original verifier hash. No verifier comparisons are skipped.
PBF, CSV, residential inputs and source inventories remain at their original
paths and retain their original existence and content-hash checks.

To audit only selected cities while developing, add `--cities dallas fort-worth`.
The report records the selected city count; a partial audit is not an all-city
acceptance. Running without `--cities` audits the entire saved cohort serially.

The original raw source files must still resolve at the paths recorded in the
frozen manifests. Missing sources are reported rather than replaced with latest
downloads. Relocating a raw source bundle is a separate operation from moving a
CLE for ordinary Stage 2 consumption.

Running `verify` on a frozen cohort without `--historical` fails before starting
the current verifier and prints the correct command. This avoids confusing a
new package's policy hashes with the old construction policy.

## Construct or revise data with new code

Use a fresh output directory and bind its implementation, inputs and policies.
The old freeze remains a historical version. When a rule correction changes
actual locations or costs, rebuild the affected dependencies into a new version
and record the differences; never rewrite the old receipt to accept new bytes.

`LocalAccessContext.load` is a construction-context API. A historical context may
reject newer policy resources, even if the change was only JSON formatting.
Use the corresponding saved implementation for historical context work, or
build an explicitly versioned new context. Do not bypass its policy hash check.
To run an original Python context operation, select the saved package explicitly
in a separate process, for example:

```bash
PYTHONPATH=outputs/stage1_complete_20261027/code_snapshot/src \
  python -B -c 'from evrptw_cle.local_road_access import LocalAccessContext; c = LocalAccessContext.load("outputs/stage1_complete_20261027/cities/dallas/roads"); print(len(c.core))'
```

The scientific environment must supply the recorded dependencies; the entrypoint
does not install packages or silently change the Python environment.
