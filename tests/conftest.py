"""Keep module batches out of one process with shared Panda3D global state."""

import pytest


def pytest_runtestloop(session):
    if session.config.getoption('collectonly'):
        return
    modules = {item.path for item in session.items}
    if len(modules) > 1:
        raise pytest.UsageError(
            f'{len(modules)} test modules selected in one process. '
            'Panda3D scene state and memory are not isolated between modules.\n'
            'Run the full suite with:\n'
            '  source .venv/bin/activate && python run_tests_isolated.py\n'
            'For a subset, pass test module paths to run_tests_isolated.py. '
            'Single-module pytest runs and --collect-only remain available.'
        )