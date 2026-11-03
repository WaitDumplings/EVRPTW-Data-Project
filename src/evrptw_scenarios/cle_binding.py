"""Bind a Stage 2 input to its verified Stage 1 data authority."""
import json
from pathlib import Path

from evrptw_cle.util import sha256_file


BUILDING_SCHEMA = "evrptw_building_first_cle_v1"
R1_FREEZE_ID = "stage1-20261102-r1"
R1_FREEZE_SHA256 = "9db2933e7f83ffce3b3917dbad8942111db19683066e9b69d603f455fc34cfde"


def cle_authority(city_dir, manifest):
    """Return the accepted authority and independent city validation paths."""
    root = Path(city_dir)
    digest = sha256_file(root / "cle_manifest.json")
    city = manifest["city_slug"]
    if manifest["schema"] == BUILDING_SCHEMA:
        if root.parent.name != "cities" or root.name != city:
            raise ValueError("Building CLE must be in its verified Stage 1 cohort: cities/<city>")
        authority = root.parent.parent / "stage1_server_build_receipt.json"
        validation = root / "stage1_validation.json"
        receipt = json.loads(authority.read_text())
        cohort = receipt.get("cle_manifests", {})
        if (receipt.get("schema") != "evrptw_stage1_server_build_receipt_v1"
                or receipt.get("freeze_id") != R1_FREEZE_ID
                or receipt.get("freeze_manifest_sha256") != R1_FREEZE_SHA256
                or receipt.get("complete_cle_validation_passed") is not True
                or set(cohort) != set(receipt.get("validated_cities", []))
                or len(cohort) != 11 or cohort.get(city) != digest):
            raise ValueError("Stage 1 cohort receipt does not authorize this building CLE")
    elif manifest["schema"] == "evrptw_city_logistics_environment_v1":
        authority = root / manifest["freeze_authority"]
        validation = root / "cle_verification.json"
        receipt = json.loads(authority.read_text())
        entry = receipt.get("cities", {}).get(city, {})
        if (receipt.get("frozen") is not True or entry.get("cle_manifest_sha256") != digest
                or entry.get("verification_sha256") != sha256_file(validation)):
            raise ValueError("Historical freeze does not authorize this CLE")
    else:
        raise ValueError("Unsupported input CLE schema")
    certificate = json.loads(validation.read_text())
    if certificate.get("passed") is not True or certificate.get("cle_manifest_sha256") != digest:
        raise ValueError("Stage 1 validation is missing or bound to another CLE")
    return authority, validation
