"""Tests for the interactive EXE build helper."""

from pathlib import Path

import pytest

from scripts import build_exe
from scripts.build_exe import read_version, validate_version, write_version


def test_gui_wrapper_imports_canonical_package_namespace():
    content = (build_exe.PROJECT_ROOT / "daru_gui.py").read_text(encoding="utf-8")

    assert "from daru.gui.app import main" in content
    assert "from src.daru" not in content


def test_pyinstaller_spec_collects_daru_submodules():
    content = build_exe.SPEC_FILE.read_text(encoding="utf-8")

    assert "collect_submodules('daru')" in content
    assert "pathex=[str(project_root / 'src')]" in content
    assert "hiddenimports=daru_hiddenimports +" in content


def test_read_and_write_version(tmp_path: Path):
    version_file = tmp_path / "__init__.py"
    version_file.write_text(
        'APP_NAME = "Daru Document Translator"\n'
        '__version__ = "1.0.0"\n',
        encoding="utf-8",
    )

    assert read_version(version_file) == "1.0.0"

    write_version("2.3.4", version_file)

    assert read_version(version_file) == "2.3.4"


@pytest.mark.parametrize("version", ["1", "1.2", "v1.2.3", "1.2.3-beta"])
def test_validate_version_rejects_invalid_values(version: str):
    with pytest.raises(ValueError):
        validate_version(version)


@pytest.mark.parametrize("version", ["1.0.0", "2026.6.12", "1.2.3.4"])
def test_validate_version_accepts_numeric_versions(version: str):
    assert validate_version(version) == version


def test_empty_version_argument_uses_current_version(monkeypatch):
    built_versions = []
    monkeypatch.setattr(build_exe, "parse_args", lambda: type("Args", (), {"version": ""})())
    monkeypatch.setattr(build_exe, "read_version", lambda: "3.2.1")
    monkeypatch.setattr(
        build_exe,
        "build",
        lambda version: built_versions.append(version) or 0,
    )

    assert build_exe.main() == 0
    assert built_versions == ["3.2.1"]
