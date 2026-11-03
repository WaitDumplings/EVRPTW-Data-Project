# Server integration reference code

These are small reusable source fragments, not a complete customer matcher and
not production imports. They contain no historical experiments, city paths or
coverage results.

- [bid_geometry.py](bid_geometry.py): OLC/BID decoding into a quantized WGS84
  bbox and centroid cell. Migrate these pure functions into the proposed
  src/evrptw_cle/bid_geometry.py.
- [geometry_metrics.py](geometry_metrics.py): symmetric source/target area ratio.
- [osm_use_vocabulary.py](osm_use_vocabulary.py): existing OSM use vocabulary;
  apply the latest conflict-preserving semantics, not a historical classifier.

The complete direct/inference, alias, Pareto, grouping and unit algorithms are
specified in [the customer implementation document](../CUSTOMER_MATCHING_PROPOSAL.zh-CN.md).
All files use dependencies already declared by the project. They have no runner
and do not read any local source data.
