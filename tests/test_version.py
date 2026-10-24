"""The package and the distribution version must not drift apart."""

import tomllib
from pathlib import Path

import monee


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert monee.__version__ == project["version"]
