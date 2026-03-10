"""End-to-end test that runs the full pytask pipeline."""

import shutil
from pathlib import Path

import pytask
import pytest
from _pytask.outcomes import ExitCode

from hedonic_analysis import config
from hedonic_analysis.config import ROOT


@pytest.mark.end_to_end
def test_pytask_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shutil.copytree(ROOT / "documents", tmp_path / "documents")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copy(ROOT / "myst.yml", tmp_path / "myst.yml")

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "BLD", tmp_path / "bld")
    monkeypatch.setattr(config, "BLD_DATA", tmp_path / "bld" / "data")
    monkeypatch.setattr(config, "BLD_ANALYSIS", tmp_path / "bld" / "analysis")
    monkeypatch.setattr(config, "BLD_IMAGES", tmp_path / "bld" / "images")
    monkeypatch.setattr(config, "BLD_FINAL", tmp_path / "bld" / "final")
    monkeypatch.setattr(config, "DOCUMENTS", tmp_path / "documents")
    monkeypatch.setattr(config, "SRC", tmp_path / "src" / "hedonic_analysis")

    session = pytask.build(
        config=ROOT / "pyproject.toml",
        force=True,
    )
    assert session.exit_code == ExitCode.OK
