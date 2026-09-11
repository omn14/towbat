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


@pytest.mark.parametrize('paths,collect_only,blocked', [
    ([], False, False),
    (['first.py', 'first.py'], False, False),
    (['first.py', 'second.py'], False, True),
    (['first.py', 'second.py'], True, False),
])
def test_multi_module_guard_preserves_single_module_and_discovery(paths, collect_only, blocked):
    from types import SimpleNamespace
    from tests.conftest import pytest_runtestloop

    session = SimpleNamespace(
        items=[SimpleNamespace(path=Path(path)) for path in paths],
        config=SimpleNamespace(getoption=lambda name: collect_only if name == 'collectonly' else None))
    if blocked:
        with pytest.raises(pytest.UsageError, match='python run_tests_isolated.py'):
            pytest_runtestloop(session)
    else:
        assert pytest_runtestloop(session) is None


@pytest.mark.parametrize('available,expected_exit', [(1791, 2), (1792, 0), (2047, 0)])
def test_full_suite_memory_threshold_keeps_cap_and_smaller_margin(tmp_path, monkeypatch, available, expected_exit):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import run_tests_isolated as runner

    output = tmp_path / 'results'
    module = ROOT / 'tests' / 'test_isolated_runner.py'
    monkeypatch.setattr(sys, 'argv', ['run_tests_isolated.py', str(module), '--output', str(output)])
    monkeypatch.setattr(runner.shutil, 'which', lambda name: '/usr/bin/systemd-run')
    monkeypatch.setattr(runner, 'available_memory_mb', lambda: available)
    execute = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    monkeypatch.setattr(runner, 'read_result', lambda module, directory, code: {
        'module': str(module.relative_to(ROOT)), 'returncode': code, 'peak_rss_kib': None,
        'junit': {'tests': 1, 'failures': 0, 'errors': 0, 'skipped': 0}})

    assert runner.main() == expected_exit
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['memory_limit_mb'] == 1536
    assert summary['memory_headroom_mb'] == 256
    assert summary['swap_limit_mb'] == 0
    assert summary['complete'] is (expected_exit == 0)
    assert len(summary['results']) == int(expected_exit == 0)
    if expected_exit:
        assert 'keeping 256 MiB headroom' in summary['stopped_reason']
        execute.assert_called_once()
    else:
        command = execute.call_args.args[0]
        assert '--property=MemoryMax=1536M' in command
        assert '--property=MemorySwapMax=0' in command