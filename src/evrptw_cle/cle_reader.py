"""Read a verified CLE without choosing customers, vehicles or operating days."""

import json
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

from .util import sha256_file


class CityLogisticsEnvironment:
    def __init__(self, path, *, verify_hashes=True, verify=True):
        self.path=Path(path).resolve()
        self.manifest=json.loads((self.path / "cle_manifest.json").read_text())
        schema = self.manifest["schema"]
        checked_hashes = False
        if schema == "evrptw_building_first_cle_v1":
            checked_hashes = self._building_first_verification(verify, verify_hashes)
        elif schema == "evrptw_city_logistics_environment_v1":
            # Historical acceptance and freeze receipts keep their original
            # meaning; the new verify keyword does not bypass this branch.
            certificate=json.loads((self.path / "cle_verification.json").read_text())
            if not certificate["passed"] or certificate["cle_manifest_sha256"]!=sha256_file(self.path / "cle_manifest.json"):
                raise ValueError("CLE lacks an independent verification bound to this manifest")
            self.verification = certificate
            self.verified = True
            self.frozen=False
            freeze_path=self.path / self.manifest["freeze_authority"]
            if freeze_path.is_file():
                freeze=json.loads(freeze_path.read_text())
                city=freeze.get("cities",{}).get(self.manifest["city_slug"],{})
                if city.get("cle_manifest_sha256")!=certificate["cle_manifest_sha256"] or city.get("verification_sha256")!=sha256_file(self.path / "cle_verification.json"):
                    raise ValueError("CLE freeze certificate does not bind this city/version")
                self.frozen=freeze.get("frozen") is True
        else:
            raise ValueError("Unsupported CLE schema")
        for name,item in self.manifest["files"].items():
            file=(self.path / name).resolve()
            if not file.is_relative_to(self.path) or not file.is_file():
                raise ValueError("Missing or external CLE data file: "+name)
            if verify_hashes and not checked_hashes and sha256_file(file)!=item["sha256"]:
                raise ValueError("CLE data changed after verification: "+name)

    def _building_first_verification(self, verify, verify_hashes):
        """Use a new receipt or the building validator, without legacy contracts."""
        self.frozen = False
        self.verified = False
        self.verification = None
        if not verify:
            return False
        receipt = self.path / self.manifest.get("independent_verification", "stage1_validation.json")
        if receipt.is_file():
            result = json.loads(receipt.read_text())
            if (result.get("passed") is not True or result.get("cle_manifest_sha256")
                    != sha256_file(self.path / "cle_manifest.json")):
                raise ValueError("Building CLE lacks validation bound to this manifest")
            self.verification = result
            self.verified = True
            return False
        from .stage1_validation import validate_building_stage1

        result = validate_building_stage1(self.path, verify_hashes=verify_hashes)
        if not result["passed"]:
            failed = ", ".join(item["check"] for item in result["errors"][:5])
            raise ValueError("Building CLE validation failed: " + failed)
        self.verification = result
        self.verified = True
        return bool(verify_hashes)

    def locations(self):
        return gpd.read_parquet(self.path / "locations.parquet")

    def access_costs(self):
        return pd.read_parquet(self.path / "location_access_costs.parquet")

    def graph(self, layer="public"):
        """Return exactly one source graph with its joined reference costs.

        Each graph retains the routing membership of its construction policy;
        ordinary private-road treatment is explicit in the current CLE manifest.
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
        if self.manifest["schema"] == "evrptw_building_first_cle_v1":
            graph.graph["ordinary_private_through_routing"] = bool(
                self.manifest.get("construction_options", {}).get("allow_private_through", False))
        return graph
