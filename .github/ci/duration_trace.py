# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Host wall-time markers for four CI tests; never imports GPU libraries.

Load with ``pytest -p duration_trace`` and set CI_DURATION_TRACE_DIR. Spawned
workers inherit CI_DURATION_NODE, while unrelated tests produce no records.
These timings include ordinary host-side waits and do not measure GPU time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, TypeVar


TARGET_NODES = (
    "tests/backends/fsdp/test_weight_transaction.py::test_non_src_serialization_failure_exits_all_ranks_without_hang",
    "tests/backends/megatron/test_qsa_mixed.py::test_qsa_mixed_output_and_gradients[lengths1-8]",
    "tests/backends/megatron/test_qsa_mixed.py::test_qsa_mixed_output_and_gradients[lengths2-8]",
    "tests/backends/megatron/test_gdn_cp_gpu.py::test_gdn_cp_layout_and_gradients[chunkwise]",
)

_T = TypeVar("_T")
_COLLECTING = False
_RANK: int | None = None
_STACK: list[str] = []
_ERRORS: dict[int, int] = {}


def _node(collection: bool = False) -> str | None:
    if not os.environ.get("CI_DURATION_TRACE_DIR"):
        return None
    node = os.environ.get("CI_DURATION_NODE")
    if node in TARGET_NODES:
        return node
    return "<collection>" if collection and _COLLECTING else None


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
            "phase": phase,
            "event": event,
            **fields,
        }
        destination = Path(os.environ["CI_DURATION_TRACE_DIR"]) / f"phase-{pid}.jsonl"
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        _ERRORS[pid] = _ERRORS.get(pid, 0) + 1
        if _ERRORS[pid] == 1:
            try:
                sys.stderr.write(
                    f"CI duration trace unavailable ({type(exc).__name__}); measurements are incomplete.\n"
                )
            except OSError:
                pass


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
        self.parent_phase: str | None = None

    def __enter__(self) -> _Span:
        if self.nodeid is not None:
            self.parent_phase = _STACK[-1] if _STACK else None
            self.started_ns = time.perf_counter_ns()
            _emit(self.phase, "begin", nodeid=self.nodeid, parent_phase=self.parent_phase)
            _STACK.append(self.phase)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> bool:
        if self.nodeid is not None:
            elapsed_ns = time.perf_counter_ns() - self.started_ns
            _STACK.pop()
            _emit(
                self.phase,
                "end",
                nodeid=self.nodeid,
                parent_phase=self.parent_phase,
                elapsed_ns=elapsed_ns,
                exception_type=exc_type.__name__ if exc_type is not None else None,
            )
        return False


def span(phase: str, *, collection: bool = False) -> _Span:
    return _Span(phase, collection)


def call(phase: str, operation: Callable[[], _T], *, collection: bool = False) -> _T:
    """Time one existing zero-argument availability query, preserving
    laziness."""
    with span(phase, collection=collection):
        return operation()


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
