"""Cross-platform RTL-SDR tool discovery (Bijaya, 24 Aug: no hard-coded
rtl_sdr.exe - find the installed driver per OS)."""

import os
import stat
import sys

import pytest

from rtl_recorder import doctor, utils
from rtl_recorder.exceptions import ExecutableNotFoundError


def _fake_tool(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name + (".exe" if sys.platform.startswith("win") else ""))
    path.write_text("#!/bin/sh\necho fake\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def empty_path(monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: None)  # nothing on PATH


def test_found_via_rtl_sdr_home(tmp_path, monkeypatch, empty_path):
    tool = _fake_tool(tmp_path / "rtl-sdr-release" / "x64", "rtl_sdr")
    monkeypatch.setenv("RTL_SDR_HOME", str(tmp_path / "rtl-sdr-release"))
    assert utils.find_executable("rtl_sdr") == str(tool)


def test_path_wins_over_install_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: "/on/path/" + name)
    assert utils.find_executable("rtl_sdr") == "/on/path/rtl_sdr"


def test_explicit_path_must_exist(tmp_path):
    with pytest.raises(ExecutableNotFoundError, match="configured path"):
        utils.find_executable("rtl_sdr", str(tmp_path / "nope"))


@pytest.mark.parametrize("system,expected", [
    ("Windows", "Zadig"), ("Darwin", "brew install librtlsdr"), ("Linux", "apt install rtl-sdr"),
])
def test_install_hint_matches_os(monkeypatch, system, expected):
    monkeypatch.setattr(utils.platform, "system", lambda: system)
    assert expected in utils.install_hint()


@pytest.mark.parametrize("system,folder", [
    ("Darwin", "/opt/homebrew/bin"), ("Linux", "/usr/local/bin"),
])
def test_candidate_dirs_cover_os_defaults(monkeypatch, system, folder):
    monkeypatch.delenv("RTL_SDR_HOME", raising=False)
    monkeypatch.setattr(utils.platform, "system", lambda: system)
    assert any(d.as_posix() == folder for d in utils.candidate_install_dirs())


def test_missing_tool_error_contains_os_instructions(monkeypatch, empty_path):
    monkeypatch.delenv("RTL_SDR_HOME", raising=False)
    monkeypatch.setattr(utils, "candidate_install_dirs", lambda: [])
    with pytest.raises(ExecutableNotFoundError) as exc:
        utils.find_executable("rtl_sdr")
    assert utils.platform_name() in str(exc.value)
    assert "--rtl-sdr-path" in str(exc.value)


def test_doctor_reports_found_tools(tmp_path, monkeypatch, empty_path):
    _fake_tool(tmp_path / "tools", "rtl_sdr")
    _fake_tool(tmp_path / "tools", "rtl_test")
    monkeypatch.setenv("RTL_SDR_HOME", str(tmp_path / "tools"))
    checks = {c.name: c for c in doctor.run_checks(output_dir=str(tmp_path / "rec"), probe_device=False)}
    assert checks["rtl_sdr"].ok and checks["rtl_test"].ok
    assert checks["output folder"].ok
    report = doctor.format_report(list(checks.values()))
    assert "rtl_sdr" in report


def test_doctor_explains_missing_tools(tmp_path, monkeypatch, empty_path):
    monkeypatch.delenv("RTL_SDR_HOME", raising=False)
    monkeypatch.setattr(utils, "candidate_install_dirs", lambda: [])
    checks = doctor.run_checks(output_dir=str(tmp_path / "rec"), probe_device=True)
    report = doctor.format_report(checks)
    assert "To install the RTL-SDR tools" in report
    assert "simulation mode" in report
    assert doctor.main(["--output-dir", str(tmp_path / "rec"), "--no-device"]) == 1
