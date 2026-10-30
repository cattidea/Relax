# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt in to SSB mitigation in one isolated CPU-control process; never opt
out."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType


PR_GET_SPECULATION_CTRL = 52
PR_SET_SPECULATION_CTRL = 53
PR_SPEC_STORE_BYPASS = 0
PR_SPEC_INDIRECT_BRANCH = 1
PR_SPEC_DISABLE = 4
INITIAL_STATE = 3  # PR_SPEC_PRCTL | PR_SPEC_ENABLE; observation only.
MITIGATED_STATE = 5  # PR_SPEC_PRCTL | PR_SPEC_DISABLE.
UNSUPPORTED = {errno.EINVAL, errno.ENODEV, errno.ENXIO, errno.ENOSYS, errno.EOPNOTSUPP}
Prctl = Callable[[int, int, int, int, int], int]


def load_helper() -> ModuleType:
    path = Path(__file__).with_name("duration_cpu_control.py")
    spec = importlib.util.spec_from_file_location("_duration_cpu_control", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the original CPU-control helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def load_prctl() -> Prctl:
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    return prctl


def query(prctl: Prctl, which: int) -> dict[str, int]:
    ctypes.set_errno(0)
    value = prctl(PR_GET_SPECULATION_CTRL, which, 0, 0, 0)
    return {"value": value, "errno": ctypes.get_errno() if value < 0 else 0}


def snapshot(prctl: Prctl, helper: ModuleType) -> dict:
    return {
        "pid": os.getpid(),
        "ssb": query(prctl, PR_SPEC_STORE_BYPASS),
        "indirect_branch": query(prctl, PR_SPEC_INDIRECT_BRANCH),
        "process_status": helper.process_status(),
    }


def check_process() -> None:
    if not sys.flags.isolated or not sys.flags.no_site:
        raise RuntimeError("SSB control requires python -I -S")
    if any(name.split(".")[0] in {"torch", "triton", "numpy"} for name in sys.modules):
        raise RuntimeError("Unexpected numerical library loaded in SSB control")
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("SSB control must have one process thread")


def check_state(state: dict, expected: int, text: str, pid: int) -> None:
    if state["pid"] != pid or state["ssb"] != {"value": expected, "errno": 0}:
        raise RuntimeError("SSB state is not the required controllable per-thread state")
    if state["process_status"].get("Speculation_Store_Bypass") != text:
        raise RuntimeError("Kernel process status does not match the required SSB state")
    if state["process_status"].get("Seccomp") != "2":
        raise RuntimeError("Expected the existing seccomp syscall filter to remain enabled")
    indirect = state["indirect_branch"]
    if indirect["value"] < 0 and indirect["errno"] not in UNSUPPORTED:
        raise RuntimeError("Unexpected error reading indirect-branch speculation state")


def write_report(output: Path, report: dict, control: dict) -> None:
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps({**report, "ssb_control": control}, indent=2) + "\n")
    temporary.replace(output)


def main(output: Path, arm: str) -> None:
    pid = os.getpid()
    control = {
        "arm": arm,
        "valid": False,
        "stage": "initializing",
        "pid": pid,
        "pid_scope": "original helper.main called directly in this interpreter; no child interpreter",
        "scope": "per_process_SSB_opt_in_only_no_global_or_other_thread_changes",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_metadata": {
            key: os.environ[key]
            for key in ("CI_HARNESS_REF", "CI_RUN_ID", "CI_RUN_ATTEMPT", "CI_TRIAL", "CI_IMAGE")
            if key in os.environ
        },
        "set_calls": [],
    }
    report: dict = {}
    helper = None
    prctl = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_report(output, report, control)
        if arm not in {"A", "B"}:
            raise ValueError("SSB control arm must be A or B")
        check_process()
        helper = load_helper()
        prctl = load_prctl()
        control["before"] = before = snapshot(prctl, helper)
        check_state(before, INITIAL_STATE, "thread vulnerable", pid)
        if arm == "B":
            # This is the only SET operation: enable mitigation for the calling thread.
            call = {"operation": PR_SET_SPECULATION_CTRL, "which": PR_SPEC_STORE_BYPASS, "control": PR_SPEC_DISABLE}
            control["set_calls"].append(call)
            ctypes.set_errno(0)
            result = prctl(PR_SET_SPECULATION_CTRL, PR_SPEC_STORE_BYPASS, PR_SPEC_DISABLE, 0, 0)
            call.update(result=result, errno=ctypes.get_errno() if result < 0 else 0)
            if result != 0:
                raise OSError(call["errno"] or errno.EIO, "Cannot enable per-thread SSB mitigation")
        expected = INITIAL_STATE if arm == "A" else MITIGATED_STATE
        expected_text = "thread vulnerable" if arm == "A" else "thread mitigated"
        control["after_selection"] = selected = snapshot(prctl, helper)
        check_state(selected, expected, expected_text, pid)
        if selected["indirect_branch"] != before["indirect_branch"]:
            raise RuntimeError("Indirect-branch state changed while selecting the SSB arm")
        control["stage"] = "original_cpu_helper_running"
        write_report(output, report, control)
        helper.main(output)
        report = json.loads(output.read_text())
        control["after_cpu"] = after = snapshot(prctl, helper)
        check_process()
        check_state(after, expected, expected_text, pid)
        if after["indirect_branch"] != before["indirect_branch"]:
            raise RuntimeError("Indirect-branch state changed during the CPU measurement")
        initial_status = {
            key: value for key, value in before["process_status"].items() if key != "Speculation_Store_Bypass"
        }
        if any(
            {key: value for key, value in state["process_status"].items() if key != "Speculation_Store_Bypass"}
            != initial_status
            for state in (selected, after)
        ):
            raise RuntimeError("Process security or restored affinity changed during the SSB control")
        trials = report["trials"]
        if (
            report["iterations"] != 10_000_000
            or report["repeats"] != 3
            or report["warmup_iterations_per_core"] != 100_000
            or len(report["selected_cores"]) != 2
            or len(trials) != 6
            or not report["affinity_restored"]
        ):
            raise RuntimeError("The original CPU-control measurement recipe was not preserved")
        if any(
            trial["checksum"] != 818035329
            or trial["container_process_status_before"].get("Speculation_Store_Bypass") != expected_text
            or trial["container_process_status_before"].get("Cpus_allowed_list") != str(trial["cpu"])
            or any(
                trial["container_process_status_before"].get(key) != before["process_status"].get(key)
                for key in ("NoNewPrivs", "Seccomp", "Seccomp_filters")
            )
            for trial in trials
        ):
            raise RuntimeError("A CPU trial did not retain its required SSB/security/affinity state")
        control.update(valid=True, stage="complete", indirect_branch_unchanged=True, cpu_trial_statuses_valid=True)
        write_report(output, report, control)
    except BaseException as error:
        control.update(valid=False, error={"type": type(error).__name__, "message": str(error)})
        if helper is not None and prctl is not None:
            try:
                control["after_failure"] = snapshot(prctl, helper)
            except BaseException as snapshot_error:
                control["after_failure_error"] = type(snapshot_error).__name__
        try:
            write_report(output, report, control)
        except BaseException as write_error:
            sys.stderr.write(f"Cannot save SSB partial report: {type(write_error).__name__}\n")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -I -S duration_ssb_control.py OUTPUT A|B")
    main(Path(sys.argv[1]), sys.argv[2])
