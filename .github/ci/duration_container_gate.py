# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Validate the temporary container-policy comparison; never changes prctl
state."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType
from typing import Any


SOURCE = "ea3f0cee94b9400cfe1591beb4f2dbfe712d362c"
IMAGE = "sha256:bb7be4d7b1543361e7b6354d16aab9550b1b7985ed0db9d390ee55493c75f93b"
TRIALS = ("01-A", "02-B", "03-B", "04-A")
EXPECTED_TEST_FILES = {
    "tests/backends/fsdp/test_weight_transaction.py": "3fdc8cd86858df6a1d758531bb482575ceaa93515c6f9324d70d7be0ad9fd1bd",
    "tests/backends/megatron/test_qsa_mixed.py": "dfca1b1e8bbd7a9144c716016c26e85665ee8ce4a73881fcfd022f8c709912a8",
    "tests/backends/megatron/test_gdn_cp_gpu.py": "2482b095802a0b0d4d24341c817f81ef1af29dc296e6d8ed0b5ec9f58b711aa6",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_sibling(name: str) -> ModuleType:
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("_container_gate_" + name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load existing helper: " + name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(output: Path, data: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def security(status: dict, arm: str, *, cpu: int | None = None) -> None:
    require(status.get("Seccomp") == ("2" if arm == "A" else "0"), "Unexpected seccomp mode")
    require(
        status.get("Speculation_Store_Bypass") == ("thread force mitigated" if arm == "A" else "thread vulnerable"),
        "Unexpected thread SSB policy",
    )
    require(status.get("NoNewPrivs") == "0", "Unexpected NoNewPrivs")
    if "Seccomp_filters" in status:
        filters = status["Seccomp_filters"]
        require(isinstance(filters, str) and filters.isdecimal(), "Invalid seccomp filter count")
        require(int(filters) > 0 if arm == "A" else int(filters) == 0, "Unexpected seccomp filter count")
    if cpu is not None:
        require(status.get("Cpus_allowed_list") == str(cpu), "CPU-control affinity differs")


def snapshot_check(state: dict, arm: str, pid: int) -> None:
    require(state["pid"] == pid, "CPU-control process changed")
    # Linux v5.10 ssb_prctl_get returns PRCTL|FORCE_DISABLE (1|8), not 1|8|4.
    require(state["ssb"] == {"value": 9 if arm == "A" else 3, "errno": 0}, "Unexpected PR_GET SSB state")
    security(state["process_status"], arm)
    branch = state["indirect_branch"]
    if branch["value"] < 0:
        require(
            branch["errno"] in load_sibling("duration_ssb_control").UNSUPPORTED, "Unexpected indirect-branch errno"
        )
    else:
        require(branch["errno"] == 0, "Unexpected successful PR_GET errno")


def cpu_check(report: dict, arm: str) -> None:
    require(
        (report["iterations"], report["repeats"], report["warmup_iterations_per_core"]) == (10_000_000, 3, 100_000),
        "CPU-control recipe changed",
    )
    cores = report["selected_cores"]
    require(len(cores) == 2 and len({(c["package"], c["core"]) for c in cores}) == 2, "Expected two physical cores")
    require(report["affinity_restored"] is True, "CPU affinity not restored")
    samples = report["trials"]
    require(len(samples) == 6, "Incomplete CPU samples")
    expected = [cores[0], cores[1], cores[1], cores[0], cores[0], cores[1]]
    for index, (sample, core) in enumerate(zip(samples, expected)):
        require(all(sample[k] == core[k] for k in ("cpu", "package", "core")), "CPU order changed")
        require(sample["repeat"] == index // 2 and sample["checksum"] == 818035329, "CPU work changed")
        require(0 < sample["process_cpu_ns"] <= sample["wall_ns"], "Invalid CPU timing")
        security(sample["container_process_status_before"], arm, cpu=sample["cpu"])


def measure(output: Path, arm: str) -> None:
    control: dict[str, Any] = {
        "arm": arm,
        "valid": False,
        "stage": "initializing",
        "pid": os.getpid(),
        "get_only": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_metadata": {
            key: os.environ[key]
            for key in ("CI_TRIAL", "CI_SOURCE_REF", "CI_IMAGE", "CI_HARNESS_REF", "CI_RUN_ID", "CI_RUN_ATTEMPT")
            if key in os.environ
        },
    }
    report: dict = {}
    observer = helper = prctl = None
    try:
        write_json(output, {"container_policy": control})
        require(arm in {"A", "B"}, "Unknown container-policy arm")
        observer = load_sibling("duration_ssb_control")
        observer.check_process()
        helper, prctl = observer.load_helper(), observer.load_prctl()
        control["before"] = before = observer.snapshot(prctl, helper)
        snapshot_check(before, arm, control["pid"])
        control["stage"] = "original_cpu_helper_running"
        write_json(output, {"container_policy": control})
        helper.main(output)
        report = json.loads(output.read_text())
        control["after_cpu"] = after = observer.snapshot(prctl, helper)
        observer.check_process()
        snapshot_check(after, arm, control["pid"])
        require(before == after, "CPU-control security or restored affinity drifted")
        cpu_check(report, arm)
        control.update(valid=True, stage="complete")
        write_json(output, {**report, "container_policy": control})
    except BaseException as error:
        control.update(valid=False, error={"type": type(error).__name__, "message": str(error)})
        if observer is not None and helper is not None and prctl is not None:
            try:
                control["after_failure"] = observer.snapshot(prctl, helper)
            except BaseException as snapshot_error:
                control["after_failure_error"] = type(snapshot_error).__name__
        try:
            write_json(output, {**report, "container_policy": control})
        except BaseException as write_error:
            sys.stderr.write(f"Cannot save container-policy partial report: {type(write_error).__name__}\n")
        raise


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def finite_nonnegative(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def phase_check(
    directory: Path, arm: str, nodes: tuple[str, ...], warm_nodes: tuple[str, ...], affinity: list[int]
) -> dict:
    files = sorted(directory.glob("phase-*.jsonl"))
    require(bool(files), "No phase records")
    ends: list[dict] = []
    points: list[dict] = []
    exceptions = []
    protocols = []
    pids = set()
    for path in files:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        require(bool(records), "Empty phase file")
        pid = records[0]["pid"]
        require(path.name == f"phase-{pid}.jsonl" and pid not in pids, "Phase PID/file mismatch")
        pids.add(pid)
        require(
            (records[0]["phase"], records[0]["event"])
            in {
                ("pytest.collection", "begin"),
                ("fsdp.module_enter", "point"),
                ("gdn.module_enter", "point"),
                ("qsa.module_enter", "point"),
            },
            "Missing the first expected PID event",
        )
        security(records[0]["process_security"], arm)
        allowed = set()
        for interval in records[0]["process_security"]["Cpus_allowed_list"].split(","):
            require(re.fullmatch(r"[0-9]+(?:-[0-9]+)?", interval) is not None, "Invalid phase affinity")
            first, _, last = interval.partition("-")
            left, right = int(first), int(last or first)
            require(0 <= left <= right <= max(affinity), "Invalid phase affinity range")
            allowed.update(range(left, right + 1))
        require(allowed == set(affinity), "Phase process affinity changed")
        stack = []
        previous = -1
        for index, record in enumerate(records):
            require(record["pid"] == pid and record["nodeid"] in (*nodes, "<collection>"), "Unexpected phase PID/node")
            require(
                type(record["perf_counter_ns"]) is int and record["perf_counter_ns"] >= previous, "Nonmonotonic phases"
            )
            previous = record["perf_counter_ns"]
            require(index == 0 or "process_security" not in record, "Repeated process-security metadata")
            event = record["event"]
            identity = (record["phase"], record["nodeid"], record["rank"], record["iteration"])
            if event == "begin":
                require(record["parent_phase"] == (stack[-1][0][0] if stack else None), "Incorrect parent span")
                stack.append((identity, record))
            elif event == "end":
                require(bool(stack) and stack[-1][0] == identity, "Unmatched phase end")
                _, begin = stack.pop()
                require(record["parent_phase"] == begin["parent_phase"], "Phase parent changed")
                require(record["cpu_measurement_available"] is True, "CPU phase measurement unavailable")
                require(record["process_cpu_scope"] == "all_threads_in_this_process", "CPU scope changed")
                require(record["rusage_children_scope"] == "reaped_children_only", "Child CPU scope changed")
                require(finite_nonnegative(record["elapsed_ns"]), "Invalid phase duration")
                require(finite_nonnegative(record["process_time_ns_delta"]), "Invalid phase CPU time")
                for group in ("rusage_self_delta", "rusage_children_delta"):
                    usage = record[group]
                    require(
                        set(usage) == {"utime_seconds", "stime_seconds", "nvcsw", "nivcsw", "minflt", "majflt"},
                        "Missing CPU counters",
                    )
                    require(all(finite_nonnegative(value) for value in usage.values()), "Invalid resource counter")
                if record["exception_type"] is not None:
                    exceptions.append((record["nodeid"], record["phase"], record["rank"], record["exception_type"]))
                if record["phase"] == "pytest.protocol":
                    protocols.append((record["perf_counter_ns"], record["nodeid"]))
                ends.append(record)
            else:
                require(event == "point", "Unknown phase event")
                points.append(record)
        require(not stack, "Unclosed phase span")
    order = (nodes[0], nodes[3], nodes[1], nodes[2])
    require(tuple(node for _, node in sorted(protocols)) == order, "Incomplete/reordered pytest protocols")
    expected_errors = [(nodes[0], "fsdp.error_collectives", rank, "WeightSyncError") for rank in (0, 1)]
    require(sorted(exceptions) == expected_errors, "Unexpected phase exception/missing expected FSDP error")
    for rank in (0, 1):
        for phase in ("fsdp.worker_body", "fsdp.gloo_init", "fsdp.gloo_new_group", "fsdp.destroy_group"):
            require(
                sum(r["nodeid"] == nodes[0] and r["rank"] == rank and r["phase"] == phase for r in ends) == 1,
                "Missing FSDP worker phase",
            )
        for iteration in ("original", "warm"):
            for phase in (
                "gdn.compute",
                "gdn.cp1_forward_backward",
                "gdn.cp2_forward_backward",
                "gdn.gradient_collectives_and_assertions",
            ):
                require(
                    sum(
                        r["nodeid"] == nodes[3]
                        and r["rank"] == rank
                        and r["iteration"] == iteration
                        and r["phase"] == phase
                        for r in ends
                    )
                    == 1,
                    "Incomplete GDN original/warm path",
                )
            require(
                sum(
                    r["nodeid"] == nodes[3]
                    and r["rank"] == rank
                    and r["iteration"] == iteration
                    and r["phase"] == "compile_counters"
                    for r in points
                )
                == 1,
                "Missing GDN compile-counter observation",
            )
    for node in nodes[1:3]:
        for iteration in ("original", "warm") if node in warm_nodes else ("original",):
            for phase in ("qsa.run_case", "qsa.forward", "qsa.backward", "qsa.assertions_and_original_report"):
                require(
                    sum(r["nodeid"] == node and r["iteration"] == iteration and r["phase"] == phase for r in ends)
                    == 1,
                    "Incomplete QSA original/warm path",
                )
            require(
                sum(
                    r["nodeid"] == node and r["iteration"] == iteration and r["phase"] == "compile_counters"
                    for r in points
                )
                == 1,
                "Missing QSA compile-counter observation",
            )
    return {"pids": sorted(pids), "closed_spans": len(ends), "expected_FSDP_exceptions": len(exceptions)}


def junit_time(path: Path, nodes: tuple[str, ...]) -> float:
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    require(len(suites) == 1, "Expected one JUnit suite")
    suite = suites[0]
    require(
        all(suite.get(k) == v for k, v in {"tests": "4", "errors": "0", "failures": "0", "skipped": "0"}.items()),
        "JUnit suite did not pass all four cases",
    )
    cases = list(suite.iter("testcase"))
    expected = (nodes[0], nodes[3], nodes[1], nodes[2])
    actual = tuple(c.attrib["classname"].replace(".", "/") + ".py::" + c.attrib["name"] for c in cases)
    require(
        actual == expected
        and all(not any(c.find(k) is not None for k in ("failure", "error", "skipped")) for c in cases),
        "JUnit nodes/order/status differ",
    )
    require(all(finite_nonnegative(float(c.attrib["time"])) for c in cases), "Invalid JUnit case time")
    value = float(suite.attrib["time"])
    require(math.isfinite(value) and value > 0, "Invalid JUnit suite time")
    return value


def gate(root: Path, output: Path) -> bool:
    result: dict[str, Any] = {
        "valid_gate": False,
        "full_allowed": False,
        "scope": "container_seccomp_and_speculation_policy_combination",
    }
    try:
        trace = load_sibling("duration_trace")
        nodes, warm_nodes = trace.TARGET_NODES, trace.WARM_NODES
        trials = {}
        common = None
        for name in TRIALS:
            directory = root / name
            arm = name[-1]
            trial, env, manifest, cpu = (
                json_file(directory / f)
                for f in ("trial.json", "environment.json", "instrumentation.json", "cpu-control.json")
            )
            require(trial["CI_TRIAL"] == name and trial["CI_ARM"] == arm, "Trial identity differs")
            require(trial["CI_SOURCE_REF"] == SOURCE and trial["CI_IMAGE"] == IMAGE, "Unfrozen source/image")
            require(re.fullmatch(r"[0-9a-f]{40}", trial["CI_HARNESS_REF"]) is not None, "Invalid harness ref")
            require(
                all(re.fullmatch(r"[1-9][0-9]*", str(trial[k])) for k in ("CI_RUN_ID", "CI_RUN_ATTEMPT")),
                "Invalid run identity",
            )
            require(
                trial["private_copy_matches"] is True and trial["relax_tree_file_count"] > 0, "Source copy mismatch"
            )
            require(re.fullmatch(r"[0-9a-f]{64}", trial["relax_tree_sha256"]) is not None, "Invalid tree SHA")
            env_sha = hashlib.sha256((directory / "environment.json").read_bytes()).hexdigest()
            require((directory / "environment.sha256").read_text().strip() == env_sha, "Environment SHA mismatch")
            require(
                set(env) == {"packages", "affinity", "threads"} and bool(env["packages"]) and bool(env["affinity"]),
                "Incomplete environment",
            )
            require(
                manifest["source_ref"] == SOURCE
                and tuple(manifest["nodes"]) == nodes
                and tuple(manifest["warm_nodes"]) == warm_nodes,
                "Instrumentation source/nodes differ",
            )
            require(
                manifest["mode"] == "original_then_same_process_warm_no_extra_sync"
                and manifest["extra_computation"] is True,
                "Unexpected instrumentation mode",
            )
            require(
                manifest["time_basis"] == "host_wall_no_added_cuda_synchronization"
                and manifest["iteration_labels"] == ["original", "warm"],
                "Instrumentation timing/iterations changed",
            )
            require(len(manifest["files"]) == len(EXPECTED_TEST_FILES), "Unexpected instrumentation file count")
            require(
                {f["path"]: f["original_sha256"] for f in manifest["files"]} == EXPECTED_TEST_FILES,
                "Original test SHA mismatch",
            )
            require(
                all(re.fullmatch(r"[0-9a-f]{64}", f["instrumented_sha256"]) for f in manifest["files"]),
                "Invalid instrumented SHA",
            )
            policy = cpu["container_policy"]
            require(
                policy["arm"] == arm
                and policy["valid"] is True
                and policy["get_only"] is True
                and policy["stage"] == "complete",
                "Invalid CPU policy observation",
            )
            snapshot_check(policy["before"], arm, policy["pid"])
            snapshot_check(policy["after_cpu"], arm, policy["pid"])
            require(policy["before"] == policy["after_cpu"], "CPU policy drift")
            require(
                policy["run_metadata"]
                == {
                    k: trial[k]
                    for k in ("CI_TRIAL", "CI_SOURCE_REF", "CI_IMAGE", "CI_HARNESS_REF", "CI_RUN_ID", "CI_RUN_ATTEMPT")
                },
                "CPU-control run identity differs",
            )
            require(
                policy["script_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "CPU observer script changed",
            )
            require(
                cpu["script_sha256"]
                == hashlib.sha256(Path(__file__).with_name("duration_cpu_control.py").read_bytes()).hexdigest(),
                "Original CPU helper changed",
            )
            cpu_check(cpu, arm)
            require(cpu["affinity_before"] == env["affinity"], "CPU/pytest affinity mismatch")
            identity = {
                "trial": {
                    k: trial[k]
                    for k in (
                        "CI_SOURCE_REF",
                        "CI_IMAGE",
                        "CI_HARNESS_REF",
                        "CI_RUN_ID",
                        "CI_RUN_ATTEMPT",
                        "relax_tree_sha256",
                        "relax_tree_file_count",
                    )
                },
                "environment": env,
                "environment_sha256": env_sha,
                "instrumentation": manifest,
                "CPU": {
                    k: cpu[k]
                    for k in (
                        "python",
                        "kernel",
                        "cpu_identity_after_trials",
                        "executable_sha256",
                        "script_sha256",
                        "selected_cores",
                        "affinity_before",
                    )
                },
                "policy_script_sha256": policy["script_sha256"],
            }
            require(common is None or identity == common, "Cross-trial source/environment/harness drift")
            common = identity
            trials[name] = {
                "arm": arm,
                "suite_seconds_original_plus_warm": junit_time(directory / "junit.xml", nodes),
                "phases": phase_check(directory, arm, nodes, warm_nodes, env["affinity"]),
            }
        a = [trials[name]["suite_seconds_original_plus_warm"] for name in ("01-A", "04-A")]
        b = [trials[name]["suite_seconds_original_plus_warm"] for name in ("02-B", "03-B")]
        median_a, median_b = statistics.median(a), statistics.median(b)
        allowed = median_b <= 0.95 * median_a and max(b) < min(a)
        result.update(
            valid_gate=True,
            full_allowed=allowed,
            trials=trials,
            A_seconds=a,
            B_seconds=b,
            median_A_seconds=median_a,
            median_B_seconds=median_b,
            ratio_B_over_A=median_b / median_a,
            threshold="median_B <= 0.95 * median_A and max_B < min_A",
            reason="improved_at_least_5_percent_with_separated_trials"
            if allowed
            else "improvement_below_5_percent_or_trial_ranges_overlap",
            common=common,
        )
        write_json(output, result)
        return allowed
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            write_json(output, result)
        except BaseException as write_error:
            sys.stderr.write(f"Cannot save gate failure: {type(write_error).__name__}\n")
        raise


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "measure":
        measure(Path(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "gate":
        sys.stdout.write("true\n" if gate(Path(sys.argv[2]), Path(sys.argv[3])) else "false\n")
    else:
        raise SystemExit("Usage: duration_container_gate.py measure OUTPUT A|B | gate ROOT OUTPUT")
