"""Pin the construction entrypoint, source code and policies for one run."""

from pathlib import Path

from .util import sha256_file


def capture_code_epoch(project):
    project=Path(project)
    paths={project / name for name in ["pyproject.toml","requirements-tested.txt"]}
    for folder in ["src","scripts","configs","tests"]:
        paths.update(path for path in (project / folder).rglob("*")
                     if path.is_file() and path.suffix in {".py",".json",".sql"}
                     and "__pycache__" not in path.parts and path.name!="project_status.json")
    return {str(path.relative_to(project)):sha256_file(path) for path in sorted(paths)}


def assert_code_epoch(project, expected):
    if capture_code_epoch(project)!=expected:
        raise RuntimeError("Construction code or policy changed during the run; stop and restart with the current source version")
