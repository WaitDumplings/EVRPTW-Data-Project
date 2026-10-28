import pytest

from evrptw_cle.code_epoch import assert_code_epoch, capture_code_epoch


def test_running_build_rejects_changed_and_added_source_files(tmp_path):
    for name in ["pyproject.toml","requirements-tested.txt"]:
        (tmp_path / name).write_text("version one")
    source=tmp_path / "src/package"
    source.mkdir(parents=True)
    module=source / "builder.py"
    module.write_text("VALUE=1")
    epoch=capture_code_epoch(tmp_path)
    assert_code_epoch(tmp_path,epoch)
    module.write_text("VALUE=2")
    with pytest.raises(RuntimeError,match="changed during the run"):
        assert_code_epoch(tmp_path,epoch)
    module.write_text("VALUE=1")
    (source / "new.py").write_text("VALUE=3")
    with pytest.raises(RuntimeError,match="changed during the run"):
        assert_code_epoch(tmp_path,epoch)


def test_policies_are_pinned_while_post_acceptance_status_can_change(tmp_path):
    for name in ["pyproject.toml","requirements-tested.txt"]:
        (tmp_path / name).write_text("version one")
    configs=tmp_path / "configs"
    configs.mkdir()
    policy=configs / "contract.json"
    policy.write_text('{"rule":1}')
    status=configs / "project_status.json"
    status.write_text('{"frozen":false}')
    epoch=capture_code_epoch(tmp_path)
    status.write_text('{"frozen":true}')
    assert_code_epoch(tmp_path,epoch)
    policy.write_text('{"rule":2}')
    with pytest.raises(RuntimeError,match="changed during the run"):
        assert_code_epoch(tmp_path,epoch)
