# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Measure a fixed CPython workload without importing site or GPU libraries."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path


ITERATIONS = 10_000_000
REPEATS = 3
EXPECTED = 818035329


def workload(iterations: int) -> int:
    value = 1
    for _ in range(iterations):
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    return value


def select_cores(allowed: list[int]) -> list[dict[str, int]]:
    packages: dict[int, dict[int, int]] = {}
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        packages.setdefault(package, {}).setdefault(core, cpu)
    selected = []
    for package, cores in sorted(packages.items())[:2]:
        core = sorted(cores)[len(cores) // 2]
        selected.append({"cpu": cores[core], "package": package, "core": core})
    return selected


def frequency_khz(cpu: int) -> int | None:
    try:
        return int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text())
    except (OSError, ValueError):
        return None


def read_optional(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def process_status() -> dict[str, str]:
    text = read_optional("/proc/self/status") or ""
    prefixes = ("Speculation", "Seccomp", "NoNewPrivs", "Cpus_allowed_list")
    return {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if ":" in line and line.startswith(prefixes)
    }


def sibling_cpus(cpu: int) -> set[int]:
    text = read_optional(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    cpus = {cpu}
    for interval in (text or "").split(","):
        if not interval:
            continue
        first, _, last = interval.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def cpu_ticks(cpus: set[int]) -> dict[str, list[int]]:
    names = {f"cpu{cpu}" for cpu in cpus}
    lines = (read_optional("/proc/stat") or "").splitlines()
    return {parts[0]: [int(v) for v in parts[1:]] for line in lines if (parts := line.split()) and parts[0] in names}


def cpu_identity() -> dict[str, list[str]]:
    fields = {name: set() for name in ("model name", "cpu family", "model", "stepping", "microcode")}
    for line in (read_optional("/proc/cpuinfo") or "").splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() in fields:
            fields[name.strip()].add(value.strip())
    return {name: sorted(values) for name, values in fields.items()}


def main(output: Path) -> None:
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError("CPU control requires python -I -S")
    if any(name.split(".")[0] in {"torch", "triton", "numpy"} for name in sys.modules):
        raise RuntimeError("Unexpected numerical library loaded in CPU control")
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("CPU control must have one process thread")
    original_affinity = sorted(os.sched_getaffinity(0))
    selected = select_cores(original_affinity)
    trials = []
    try:
        # Specialize the same bytecode on each selected core before timed trials.
        for entry in selected:
            os.sched_setaffinity(0, {entry["cpu"]})
            workload(100_000)
        for repeat in range(REPEATS):
            # Alternate order to reduce a fixed relationship with host load drift.
            for entry in selected if repeat % 2 == 0 else reversed(selected):
                cpu = entry["cpu"]
                os.sched_setaffinity(0, {cpu})
                if os.sched_getaffinity(0) != {cpu}:
                    raise RuntimeError("Failed to pin CPU control")
                siblings = sibling_cpus(cpu)
                status_before = process_status()
                ticks_before = cpu_ticks(siblings)
                frequency_before = frequency_khz(cpu)
                usage_before = resource.getrusage(resource.RUSAGE_SELF)
                timestamp = time.time_ns()
                wall_before = time.perf_counter_ns()
                cpu_before = time.process_time_ns()
                checksum = workload(ITERATIONS)
                cpu_ns = time.process_time_ns() - cpu_before
                wall_ns = time.perf_counter_ns() - wall_before
                usage_after = resource.getrusage(resource.RUSAGE_SELF)
                if checksum != EXPECTED or cpu_ns <= 0 or wall_ns <= 0:
                    raise RuntimeError("Invalid CPU control result")
                trials.append(
                    {
                        **entry,
                        "repeat": repeat,
                        "time_ns": timestamp,
                        "wall_ns": wall_ns,
                        "process_cpu_ns": cpu_ns,
                        "checksum": checksum,
                        "frequency_before_khz": frequency_before,
                        "frequency_after_khz": frequency_khz(cpu),
                        "voluntary_switches": usage_after.ru_nvcsw - usage_before.ru_nvcsw,
                        "involuntary_switches": usage_after.ru_nivcsw - usage_before.ru_nivcsw,
                        "container_process_status_before": status_before,
                        "selected_and_sibling_ticks_before": ticks_before,
                        "selected_and_sibling_ticks_after": cpu_ticks(siblings),
                    }
                )
    finally:
        os.sched_setaffinity(0, original_affinity)
    if sorted(os.sched_getaffinity(0)) != original_affinity:
        raise RuntimeError("CPU control affinity was not restored")
    report = {
        "mode": "single_thread_fixed_integer_work_not_gpu_or_import_benchmark",
        "iterations": ITERATIONS,
        "repeats": REPEATS,
        "warmup_iterations_per_core": 100_000,
        "python": sys.version,
        "kernel": os.uname().release,
        "cpu_identity_after_trials": cpu_identity(),
        "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": "median physical core per visible package, lowest allowed sibling, at most two packages",
        "sibling_ticks_scope": "CPU-wide ticks during the workload, not task attribution or exclusive CPU ownership",
        "selected_cores": selected,
        "affinity_before": original_affinity,
        "affinity_restored": True,
        "trials": trials,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write(json.dumps({"cpu_control_trials": len(trials), "affinity_restored": True}) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
