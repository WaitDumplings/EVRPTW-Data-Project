"""Locate immutable policy resources in editable installs and wheels."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    (_REPO_ROOT / "configs")
    if (_REPO_ROOT / "pyproject.toml").is_file()
    else Path(__file__).with_name("resources")
)
COMMAND_ROOT = (
    _REPO_ROOT / "scripts"
    if (_REPO_ROOT / "pyproject.toml").is_file()
    else Path(__file__).with_name("commands")
)


def policy_files():
    """Implementation resources only; local inventories are not shared policies."""
    local_names = {"project_status.json", "phase1_sources.template.json",
                   "customer_sources.template.json", "road_extraction_cohort.template.json"}
    return sorted(path for path in CONFIG_ROOT.rglob("*.json") if path.name not in local_names)
