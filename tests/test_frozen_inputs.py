import json

import pytest

from evrptw_cle.frozen_inputs import hydrate_frozen_inputs
from evrptw_cle.util import sha256_file


def archive(tmp_path):
    original = tmp_path / "archive"
    original.mkdir()
    source = original / "source.bin"
    source.write_bytes(b"frozen source")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            dict(
                schema="evrptw_frozen_source_archive_v1",
                files=dict(
                    test=dict(
                        archive_path="source.bin",
                        path="data/source.bin",
                        sha256=sha256_file(source),
                    )
                ),
            )
        )
    )
    return original, source, inventory


def test_frozen_hydration_is_hash_checked_and_does_not_overwrite(tmp_path):
    original, source, inventory = archive(tmp_path)
    output = tmp_path / "output"
    report = hydrate_frozen_inputs(
        inventory_path=inventory, archive_root=original, output_root=output
    )
    assert report["file_count"] == 1 and not report["source_data_refreshed"]
    assert (output / "data/source.bin").read_bytes() == source.read_bytes()
    second = hydrate_frozen_inputs(
        inventory_path=inventory, archive_root=original, output_root=output
    )
    assert second["sources"][0]["mode"] == "hash_verified_reuse"


def test_missing_source_fails_before_creating_output(tmp_path):
    original, source, inventory = archive(tmp_path)
    source.unlink()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Missing or changed"):
        hydrate_frozen_inputs(inventory_path=inventory, archive_root=original, output_root=output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["path", "archive_path"])
def test_archive_cannot_escape_the_requested_roots(tmp_path, field):
    original, _, inventory = archive(tmp_path)
    document = json.loads(inventory.read_text())
    document["files"]["test"][field] = "../outside.bin"
    inventory.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="relative paths"):
        hydrate_frozen_inputs(
            inventory_path=inventory, archive_root=original, output_root=tmp_path / "output"
        )
