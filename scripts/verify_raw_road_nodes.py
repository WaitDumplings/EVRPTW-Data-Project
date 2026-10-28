import argparse
from pathlib import Path

from evrptw_cle.raw_road_validation import verify_raw_road_nodes

p=argparse.ArgumentParser(description="Read frozen PBFs independently to verify referenced restriction nodes")
p.add_argument("--road-inputs",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
verify_raw_road_nodes(a.road_inputs,a.output)
