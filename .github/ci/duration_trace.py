# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Host wall-time markers for four CI tests; never imports GPU libraries.

Load with ``pytest -p duration_trace`` and set CI_DURATION_TRACE_DIR. Spawned
workers inherit CI_DURATION_NODE, while unrelated tests produce no records.
These timings include ordinary host-side waits and do not measure GPU time.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar


TARGET_NODES = (
    "tests/backends/fsdp/test_weight_transaction.py::test_non_src_serialization_failure_exits_all_ranks_without_hang",
    "tests/backends/megatron/test_qsa_mixed.py::test_qsa_mixed_output_and_gradients[lengths1-8]",
    "tests/backends/megatron/test_qsa_mixed.py::test_qsa_mixed_output_and_gradients[lengths2-8]",
    "tests/backends/megatron/test_gdn_cp_gpu.py::test_gdn_cp_layout_and_gradients[chunkwise]",
)
WARM_NODES = (TARGET_NODES[2], TARGET_NODES[3])

_T = TypeVar("_T")
_COLLECTING = False
_RANK: int | None = None
_ITERATION: str | None = None
_STACK: list[str] = []
_ERRORS: dict[int, int] = {}
_SECURITY_PIDS: set[int] = set()


def _trace_error(exc: Exception) -> None:
    pid = os.getpid()
    _ERRORS[pid] = _ERRORS.get(pid, 0) + 1
    if _ERRORS[pid] == 1:
        try:
            sys.stderr.write(f"CI duration trace unavailable ({type(exc).__name__}); measurements are incomplete.\n")
        except OSError:
            pass


def _cpu_snapshot() -> dict[str, Any] | None:
    """Process CPU covers all its threads; CHILDREN covers waited-for
    children."""
    try:
        usage = {}
        for name, who in (("self", resource.RUSAGE_SELF), ("children", resource.RUSAGE_CHILDREN)):
            value = resource.getrusage(who)
            usage[name] = {
                "utime_seconds": value.ru_utime,
                "stime_seconds": value.ru_stime,
                "nvcsw": value.ru_nvcsw,
                "nivcsw": value.ru_nivcsw,
                "minflt": value.ru_minflt,
                "majflt": value.ru_majflt,
            }
        return {"process_time_ns": time.process_time_ns(), **usage}
    except (OSError, ValueError) as exc:
        _trace_error(exc)
        return None


def _cpu_delta(start: dict[str, Any] | None, end: dict[str, Any] | None) -> dict[str, Any]:
    if start is None or end is None:
        return {"cpu_measurement_available": False}
    process_time_ns = end["process_time_ns"] - start["process_time_ns"]
    usage = {
        name: {key: end[name][key] - value for key, value in start[name].items()} for name in ("self", "children")
    }
    if process_time_ns < 0 or any(value < 0 for group in usage.values() for value in group.values()):
        _trace_error(ValueError("Nonmonotonic CPU/resource counters"))
        return {"cpu_measurement_available": False}
    return {
        "cpu_measurement_available": True,
        "process_cpu_scope": "all_threads_in_this_process",
        "process_time_ns_delta": process_time_ns,
        "rusage_self_delta": usage["self"],
        "rusage_children_scope": "reaped_children_only",
        "rusage_children_delta": usage["children"],
    }


def _node(collection: bool = False) -> str | None:
    if not os.environ.get("CI_DURATION_TRACE_DIR"):
        return None
    node = os.environ.get("CI_DURATION_NODE")
    if node in TARGET_NODES:
        return node
    return "<collection>" if collection and _COLLECTING else None


def warm_enabled() -> bool:
    return _node() in WARM_NODES


def _loaded_compile_counters() -> dict[str, Any]:
    """Read existing integer counters only; never cause a lazy Torch import."""
    result: dict[str, Any] = {"scope": "process_local_dynamo_inductor_only"}
    try:
        module = sys.modules.get("torch._dynamo.utils")
        counters = vars(module).get("counters") if module is not None else None
        result["dynamo"] = {"available": isinstance(counters, dict)}
        if isinstance(counters, dict):
            result["dynamo"]["counts"] = {
                name: {
                    key: value for key, value in tuple(group.items()) if isinstance(key, str) and type(value) is int
                }
                for name, group in tuple(counters.items())
                if isinstance(name, str) and isinstance(group, dict)
            }
        module = sys.modules.get("torch._inductor.metrics")
        generated = vars(module).get("generated_kernel_count") if module is not None else None
        result["inductor"] = {"available": type(generated) is int}
        if type(generated) is int:
            result["inductor"]["generated_kernel_count"] = generated
    except (TypeError, RuntimeError) as exc:
        _trace_error(exc)
        return {"available": False, "scope": result["scope"]}
    return result


def _emit(phase: str, event: str, *, nodeid: str, **fields: Any) -> None:
    """Logging failures must not replace an original test exception/result."""
    pid = os.getpid()
    try:
        record = {
            "perf_counter_ns": time.perf_counter_ns(),
            "time_ns": time.time_ns(),
            "pid": pid,
            "ppid": os.getppid(),
            "nodeid": nodeid,
            "rank": _RANK,
            "iteration": _ITERATION,
            "phase": phase,
            "event": event,
            **fields,
        }
        if pid not in _SECURITY_PIDS:
            prefixes = ("Speculation", "Seccomp", "NoNewPrivs", "Cpus_allowed_list")
            record["process_security"] = {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in Path("/proc/self/status").read_text().splitlines()
                if ":" in line and line.startswith(prefixes)
            }
        destination = Path(os.environ["CI_DURATION_TRACE_DIR"]) / f"phase-{pid}.jsonl"
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        _SECURITY_PIDS.add(pid)
    except (OSError, TypeError, ValueError) as exc:
        _trace_error(exc)


def mark(phase: str, *, collection: bool = False, **fields: Any) -> None:
    nodeid = _node(collection)
    if nodeid is not None:
        _emit(phase, "point", nodeid=nodeid, **fields)


def set_rank(rank: int) -> None:
    """Use the existing worker argument; never query a process group/device."""
    global _RANK
    _RANK = rank


class _Span:
    def __init__(self, phase: str, collection: bool) -> None:
        self.phase = phase
        self.nodeid = _node(collection)
        self.started_ns = 0
        self.started_cpu: dict[str, Any] | None = None
        self.parent_phase: str | None = None

    def __enter__(self) -> _Span:
        if self.nodeid is not None:
            self.parent_phase = _STACK[-1] if _STACK else None
            self.started_ns = time.perf_counter_ns()
            self.started_cpu = _cpu_snapshot()
            _emit(self.phase, "begin", nodeid=self.nodeid, parent_phase=self.parent_phase)
            _STACK.append(self.phase)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if self.nodeid is not None:
            cpu = _cpu_delta(self.started_cpu, _cpu_snapshot())
            elapsed_ns = time.perf_counter_ns() - self.started_ns
            _STACK.pop()
            _emit(
                self.phase,
                "end",
                nodeid=self.nodeid,
                parent_phase=self.parent_phase,
                elapsed_ns=elapsed_ns,
                exception_type=exc_type.__name__ if exc_type is not None else None,
                **cpu,
            )
        return False


def span(phase: str, *, collection: bool = False) -> _Span:
    return _Span(phase, collection)


def call(phase: str, operation: Callable[[], _T], *, collection: bool = False) -> _T:
    """Time one existing zero-argument availability query, preserving
    laziness."""
    with span(phase, collection=collection):
        return operation()


@contextmanager
def iteration(label: str, *, phase: str = "iteration") -> Generator[None, None, None]:
    """Label an original/warm computation; restore the label even on
    failure."""
    global _ITERATION
    if _node() is None:
        yield
        return
    previous = _ITERATION
    _ITERATION = label
    before = _loaded_compile_counters()
    try:
        with span(phase):
            yield
    finally:
        try:
            mark("compile_counters", before=before, after=_loaded_compile_counters())
        finally:
            _ITERATION = previous


def pytest_configure(config: Any) -> None:
    # Import pytest only when acting as its plugin. A worker importing this
    # module for markers must not preload pytest/Torch or initialize CUDA.
    import pytest

    class Runtime:
        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_collection(self, session: Any) -> Generator[None, None, None]:
            global _COLLECTING
            _COLLECTING = True
            try:
                with span("pytest.collection", collection=True):
                    yield
            finally:
                _COLLECTING = False

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_runtest_protocol(self, item: Any, nextitem: Any) -> Generator[None, None, None]:
            previous = os.environ.pop("CI_DURATION_NODE", None)
            if item.nodeid in TARGET_NODES:
                os.environ["CI_DURATION_NODE"] = item.nodeid
            try:
                with span("pytest.protocol"):
                    yield
            finally:
                os.environ.pop("CI_DURATION_NODE", None)
                if previous is not None:
                    os.environ["CI_DURATION_NODE"] = previous

        def pytest_terminal_summary(self, terminalreporter: Any) -> None:
            if _ERRORS:
                terminalreporter.write_line("CI duration trace had write errors; do not use incomplete phase data.")

    config.pluginmanager.register(Runtime(), "ci-duration-runtime")
