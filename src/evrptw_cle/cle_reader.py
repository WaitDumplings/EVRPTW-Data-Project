"""Read a verified CLE without choosing customers, vehicles or operating days."""

import json
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

from .util import sha256_file


class CityLogisticsEnvironment:
    def __init__(self, path, *, verify_hashes=True):
        self.path=Path(path).resolve()
        self.manifest=json.loads((self.path / "cle_manifest.json").read_text())
        if self.manifest["schema"]!="evrptw_city_logistics_environment_v1":
            raise ValueError("Unsupported CLE schema")
        certificate=json.loads((self.path / "cle_verification.json").read_text())
        if not certificate["passed"] or certificate["cle_manifest_sha256"]!=sha256_file(self.path / "cle_manifest.json"):
            raise ValueError("CLE lacks an independent verification bound to this manifest")
        self.frozen=False
        freeze_path=self.path / self.manifest["freeze_authority"]
        if freeze_path.is_file():
            freeze=json.loads(freeze_path.read_text())
            city=freeze.get("cities",{}).get(self.manifest["city_slug"],{})
            if city.get("cle_manifest_sha256")!=certificate["cle_manifest_sha256"] or city.get("verification_sha256")!=sha256_file(self.path / "cle_verification.json"):
                raise ValueError("CLE freeze certificate does not bind this city/version")
            self.frozen=freeze.get("frozen") is True
        for name,item in self.manifest["files"].items():
            file=(self.path / name).resolve()
            if not file.is_relative_to(self.path) or not file.is_file():
                raise ValueError("Missing or external CLE data file: "+name)
            if verify_hashes and sha256_file(file)!=item["sha256"]:
                raise ValueError("CLE data changed after verification: "+name)

    def locations(self):
        return gpd.read_parquet(self.path / "locations.parquet")

    def access_costs(self):
        return pd.read_parquet(self.path / "location_access_costs.parquet")

    def graph(self, layer="public"):
        """Return exactly one source graph with its joined reference costs.

        Returning separate graphs prevents an innocent graph composition from
        making terminal/private roads available for public through-routing.
        Connector speeds and an operating-day instance are not inferred.
        """
        choices={
            "public":("roads/graph_operational.graphml","in_public_graph"),
            "delivery_terminal":("roads/graph_terminal_access.graphml","in_delivery_terminal_graph"),
            "charging_terminal":("charging/private_access/graph_terminal_access.graphml","in_charging_terminal_graph"),
        }
        if layer not in choices:
            raise ValueError("Choose public, delivery_terminal or charging_terminal")
        path,flag=choices[layer]
        graph=ox.load_graphml(self.path / path,node_dtypes={"osmid":str},edge_dtypes={"osmid":str})
        frame=pd.read_parquet(self.path / "speeds/directed_speeds.parquet")
        columns=["legal_speed_kph","legal_travel_time_s","reference_speed_weekday_kph",
                 "reference_speed_weekend_kph","reference_travel_time_weekday_s","reference_travel_time_weekend_s"]
        selected=frame.loc[frame[flag]]
        weights={tuple(key):values for key,values in zip(selected[["edge_u","edge_v","edge_key"]].astype(str).to_numpy(),selected[columns].to_numpy())}
        if graph.number_of_edges()!=len(weights):
            raise ValueError("CLE graph and source-edge weight coverage differ")
        for u,v,key,data in graph.edges(keys=True,data=True):
            identity=(str(u),str(v),str(key))
            if identity not in weights:
                raise ValueError("Unbound native edge in saved CLE graph")
            data.update(zip(columns,weights[identity]))
        graph.graph["cle_layer"]=layer
        graph.graph["terminal_through_routing_allowed"]=False
        return graph
