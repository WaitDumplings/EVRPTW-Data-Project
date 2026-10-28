"""Read one already constructed CLE: python examples/read_cle.py OUTPUT/cities/dallas."""

import argparse

from evrptw_cle.cle_reader import CityLogisticsEnvironment

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("city_directory")
a = p.parse_args()
cle = CityLogisticsEnvironment(a.city_directory)
locations = cle.locations()
print(locations.groupby(["location_type", "candidate_eligible"]).size())
print("Use cle.graph('public') for the public graph; terminal graphs are separate.")
