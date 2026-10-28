#!/usr/bin/env python3
"""Regenerate one city's frozen Microsoft/NSI derived inputs, or assemble the registry."""

import argparse
import json
from pathlib import Path

from evrptw_cle.source_preparation import assemble_customer_source_registry, prepare_customer_city

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--inventory", type=Path, required=True)
p.add_argument("--source-root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--city")
p.add_argument("--prepared-root", type=Path)
a = p.parse_args()
if a.city:
    result = prepare_customer_city(
        inventory_path=a.inventory, source_root=a.source_root, city_slug=a.city, output_dir=a.output
    )
    print(
        json.dumps(
            dict(
                city=a.city,
                nsi_parse=result["nsi_parse"],
                matching_counts=result["matching_counts"],
            ),
            indent=2,
        )
    )
else:
    if a.prepared_root is None:
        p.error("--prepared-root is required when assembling the source registry")
    result = assemble_customer_source_registry(
        inventory_path=a.inventory,
        source_root=a.source_root,
        prepared_root=a.prepared_root,
        output_path=a.output,
    )
    print(json.dumps(dict(cities=list(result["cities"]))))
