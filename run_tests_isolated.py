"""Run pytest modules in separate, memory-bounded Linux user services."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ElementTree


ROOT = Path(__file__).resolve().parent
FORWARDED_ENV = ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR")


def available_memory_mb() -> int:
    with open("/proc/meminfo", encoding="utf-8") as memory_info:
        for line in memory_info:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    raise RuntimeError("Cannot determine available memory; refusing to run tests")


def service_command(module: Path, output: Path, unit: str,
                    memory_mb: int, timeout: int) -> list[str]:
    command = [
        "systemd-run", "--user", "--wait", "--pipe", "--collect",
        f"--unit={unit}", f"--working-directory={ROOT}",
        "--property=MemoryAccounting=yes", f"--property=MemoryMax={memory_mb}M",
        "--property=MemorySwapMax=0", "--property=OOMPolicy=stop",
        f"--property=RuntimeMaxSec={timeout}", "--property=TimeoutStopSec=10",
    ]
    for name in FORWARDED_ENV:
        if name in os.environ:
            command.append(f"--setenv={name}={os.environ[name]}")
    command.extend([
        "/usr/bin/time", "-f", '{"peak_rss_kib":%M,"elapsed_seconds":%e}',
        "-o", str(output / "memory.json"),
        sys.executable, "-m", "pytest", str(module), "-q", "--tb=short",
        "--show-capture=no", "--durations=3", f"--junitxml={output / 'junit.xml'}",
    ])
    return command


def read_result(module: Path, output: Path, returncode: int) -> dict:
    result = {"module": str(module.relative_to(ROOT)), "returncode": returncode,
              "log": str(output / "pytest.log"), "peak_rss_kib": None}
    memory_file = output / "memory.json"
    if memory_file.exists():
        for line in memory_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                result.update(json.loads(line))
    junit_file = output / "junit.xml"
    if junit_file.exists():
        suites = ElementTree.parse(junit_file).getroot().iter("testsuite")
        totals = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.get(key, "0"))
        result["junit"] = totals
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", help="Test module paths; default: all tests/test_*.py")
    parser.add_argument("--memory-mb", type=positive_int, default=1536)
    parser.add_argument("--timeout", type=positive_int, default=180,
                        help="Maximum seconds per module (default: 180)")
    parser.add_argument("--output", type=Path, help="New directory for results")
    args = parser.parse_args()
    if not shutil.which("systemd-run") or not Path("/usr/bin/time").is_file():
        parser.error("Linux systemd user services and /usr/bin/time are required")
    if subprocess.run(["systemctl", "--user", "show-environment"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        parser.error("No systemd user manager; refusing to run unbounded tests")

    modules = ([Path(module).resolve() for module in args.modules] if args.modules else
               sorted((ROOT / "tests").rglob("test_*.py")))
    for module in modules:
        if not module.is_file() or not module.is_relative_to(ROOT / "tests"):
            parser.error(f"Not a test module in this repository: {module}")
    modules = list(dict.fromkeys(modules))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or ROOT / ".pytest_cache" / "isolated" / f"{stamp}-{os.getpid()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"memory_limit_mb": args.memory_mb, "swap_limit_mb": 0,
               "timeout_seconds": args.timeout,
               "planned_modules": [str(module.relative_to(ROOT)) for module in modules],
               "results": [], "complete": False}

    def save_summary():
        temporary = output / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "summary.json")

    save_summary()
    print(f"Results: {output}", flush=True)
    print(f"{len(modules)} modules, sequential services, {args.memory_mb} MiB RAM each, no swap", flush=True)
    try:
        for index, module in enumerate(modules, start=1):
            available = available_memory_mb()
            if available < args.memory_mb + 512:
                summary["stopped_reason"] = f"Only {available} MiB available; keeping 512 MiB headroom"
                print(summary["stopped_reason"], flush=True)
                return 2
            module_output = output / f"{index:03d}-{module.stem}"
            module_output.mkdir()
            unit = f"towbat-tests-{os.getpid()}-{index}"
            print(f"[{index}/{len(modules)}] {module.name}", flush=True)
            with (module_output / "pytest.log").open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(
                        service_command(module, module_output, unit, args.memory_mb, args.timeout),
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                except KeyboardInterrupt:
                    subprocess.run(["systemctl", "--user", "stop", unit], check=False)
                    summary["stopped_reason"] = "Interrupted; active test service stopped"
                    return 130
            result = read_result(module, module_output, completed.returncode)
            summary["results"].append(result)
            save_summary()
            peak = result["peak_rss_kib"]
            peak_text = f"{peak / 1024:.1f} MiB" if peak is not None else "unavailable (see log)"
            print(f"  exit={completed.returncode}, peak RSS={peak_text}, JUnit={result.get('junit', {})}", flush=True)
            if completed.returncode and "junit" not in result:
                summary["stopped_reason"] = f"Service/collection failure in {module.name}; inspect log before retrying"
                return 2
        summary["complete"] = True
        return int(any(result["returncode"] for result in summary["results"]))
    finally:
        save_summary()
        measured = [result for result in summary["results"] if result["peak_rss_kib"] is not None]
        for result in sorted(measured, key=lambda result: result["peak_rss_kib"], reverse=True)[:5]:
            print(f"Peak: {result['peak_rss_kib'] / 1024:.1f} MiB {result['module']}", flush=True)
        print(f"Completed {len(summary['results'])}/{len(modules)} modules; summary: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())