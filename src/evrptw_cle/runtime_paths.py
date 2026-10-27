"""Locate immutable policy resources in editable installs and wheels."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    (_REPO_ROOT / "configs")
    if (_REPO_ROOT / "pyproject.toml").is_file()
    else Path(__file__).with_name("resources")
)
