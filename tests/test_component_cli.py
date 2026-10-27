import hashlib
import json

import pytest

from evrptw_cle.cli import main, require_current_context, validate_inputs


def test_cli_stops_before_unfrozen_CLE_and_activation(capsys):
    main(["status"])
    status = json.loads(capsys.readouterr().out)
    assert status["customer_depot_location_components_frozen"]
    assert not status["cle_structure_frozen"]
    assert not status["instance_activation_included"]
    with pytest.raises(SystemExit) as error:
        main(["activate"])
    assert error.value.code == 2


def test_missing_or_changed_source_is_not_silently_reused(tmp_path):
    (tmp_path / "configs").mkdir()
    source = tmp_path / "source.bin"
    content = b"frozen-source-bytes"
    registry = tmp_path / "configs/sources.json"
    registry.write_text(
        json.dumps(
            {
                "cities": {
                    "example": {
                        "sources": {
                            "evidence": {
                                "path": "source.bin",
                                "sha256": hashlib.sha256(content).hexdigest(),
                            }
                        }
                    }
                }
            }
        )
    )
    assert not validate_inputs(registry)["ready"]
    source.write_bytes(content)
    assert validate_inputs(registry)["ready"]
    source.write_bytes(b"new-live-snapshot")
    result = validate_inputs(registry)
    assert not result["ready"] and result["checks"][0]["status"] == "hash_mismatch"


def test_legacy_context_cannot_enter_component_cli(tmp_path):
    (tmp_path / "access_context_manifest.json").write_text(
        json.dumps({"policy": "local_first_real_osm_access_v1"})
    )
    with pytest.raises(ValueError, match="delivery-purpose"):
        require_current_context(tmp_path)
