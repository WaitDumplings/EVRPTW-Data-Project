# Remaining integration gates

This project prepares reviewed components before the full CLE structure is
settled. It deliberately provides no complete-CLE or instance-activation CLI.

The source processing kernels are present, but large-snapshot distribution,
portable acquisition/hydration and one coherent cohort orchestration entry point
remain integration work. The provided archived-input hashes must not be reused
as if newly regenerated derived files were byte-identical without checking them.

1. Rebind AFDC charging locations/equipment to the current October OSM graph
   and verify all eleven cities. Historical charging counts are not proof of
   current-graph access.
2. Rebind direction-applicable legal/reference speeds, HPMS evidence and
   terminal operating speeds to that same graph. The supplied seven-state
   factors remain statewide priors, not observed edge/EDV speeds.
3. Review the CLE data structure and construction contract. Preserve public
   versus terminal edges, source and group identities, independent in/out
   ledgers and costs, nullable units, uncertainties and provenance hashes.
   Establish one portable assembler and independent verifier.
4. Independently build and freeze the complete CLE. Verify source recovery,
   all layer bindings and exports. Internal masks/blocked inventories must not
   leak into the public candidate data.
5. Audit instance activation, unique-location counts, household-to-order
   assumptions, demand, time windows, service time and vehicle constraints.
   Only then add and validate the operating-day generator.

The initial repository is private at `WaitDumplings/EVRPTW-Data-Project`.
Before public release, select the code license. Full source/data publication has additional attribution and
snapshot-distribution requirements listed in `NOTICE.md` and `data/README.md`.

The GitHub Actions workflow is retained as `ci/github-actions-tests.yml`.
The current credential lacks workflow-writing permission; remote CI is not
enabled. Moving the template to `.github/workflows/tests.yml` requires a
credential with that permission. Local test/build evidence is supplied separately.

The upstream 11-city production evidence is a reference. Ported-code tests and
packaging checks validate this new codebase separately; they do not mean the
complete source cohort has already been regenerated from the new repository.
