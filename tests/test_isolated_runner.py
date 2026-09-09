"""Resource protection and reports for the isolated test runner."""

import json
from pathlib import Path
import sys

import pytest

from run_tests_isolated import ROOT, positive_int, read_result, service_command


def test_service_command_keeps_tests_outside_editor_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":42")
    monkeypatch.setenv("UNRELATED_TEST_SECRET", "not-forwarded")
    module = ROOT / "tests" / "test_roster_importer.py"
    command = service_command(module, tmp_path, "test-unit", 1024, 180)
    assert command[:5] == ["systemd-run", "--user", "--wait", "--pipe", "--collect"]
    assert "--scope" not in command
    assert "--property=MemoryMax=1024M" in command
    assert "--property=MemorySwapMax=0" in command
    assert "--property=OOMPolicy=stop" in command
    assert "--property=RuntimeMaxSec=180" in command
    assert "--setenv=DISPLAY=:42" in command
    assert all("UNRELATED_TEST_SECRET" not in argument for argument in command)
    assert command[command.index(sys.executable):][:4] == [sys.executable, "-m", "pytest", str(module)]
    assert f"--junitxml={tmp_path / 'junit.xml'}" in command


def test_reports_peak_rss_and_junit_counts(tmp_path):
    (tmp_path / "memory.json").write_text(
        'Command exited with non-zero status 1\n{"peak_rss_kib":2048,"elapsed_seconds":1.5}\n',
        encoding="utf-8")
    (tmp_path / "junit.xml").write_text(
        '<testsuites><testsuite tests="4" failures="1" errors="0" skipped="1" /></testsuites>',
        encoding="utf-8")
    result = read_result(ROOT / "tests" / "test_roster_importer.py", tmp_path, 1)
    assert result["returncode"] == 1
    assert result["peak_rss_kib"] == 2048
    assert result["elapsed_seconds"] == 1.5
    assert result["junit"] == {"tests": 4, "failures": 1, "errors": 0, "skipped": 1}
    assert json.loads(json.dumps(result)) == result


def test_missing_reports_do_not_claim_success_or_zero_memory(tmp_path):
    result = read_result(ROOT / "tests" / "test_roster_importer.py", tmp_path, 137)
    assert result["returncode"] == 137
    assert result["peak_rss_kib"] is None
    assert "junit" not in result
    assert Path(result["log"]).parent == tmp_path


@pytest.mark.parametrize("value", ["0", "-1"])
def test_limits_must_be_positive(value):
    from argparse import ArgumentTypeError

    with pytest.raises(ArgumentTypeError):
        positive_int(value)