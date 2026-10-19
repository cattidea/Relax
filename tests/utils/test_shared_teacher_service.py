# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from relax.utils.opd.shared_teacher_service import SharedTeacherService


class Backend:
    def __init__(self):
        self.current_teacher = None
        self.resident = False
        self.events = []
        self.fail_restore = False
        self.generate_calls = 0

    async def start(self, teacher):
        self.current_teacher = teacher
        self.resident = True
        self.events.append("start")

    async def switch(self, teacher):
        assert self.resident
        self.current_teacher = teacher

    async def generate(self, payload):
        assert self.resident
        self.generate_calls += 1
        return 200, {"teacher": self.current_teacher, "payload": payload}

    def mark_nonresident(self):
        self.events.append("nonresident")
        self.resident = False

    async def restore_after_onload(self):
        assert not self.resident
        self.events.append("disk_restore")
        if self.fail_restore:
            raise RuntimeError("restore failed")
        self.resident = True

    async def close(self):
        self.events.append("close")


def score(origin, teacher):
    request = Request(
        origin + f"/teachers/{teacher}/generate",
        json.dumps({"input_ids": [1, 2]}).encode(),
        {"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3) as response:
        return json.load(response)


def test_service_closes_admission_through_offload_and_forced_disk_restore():
    backend = Backend()
    service = SharedTeacherService(backend, ["A", "B"], "A", "127.0.0.1", timeout_seconds=5)
    origin = service.start()
    try:
        assert score(origin, "B")["teacher"] == "B"
        service.suspend()
        assert not backend.resident
        assert service.snapshot()["health"]["state"] == "suspended"
        with pytest.raises(HTTPError) as error:
            score(origin, "A")
        assert error.value.code == 503
        error.value.close()
        service.restore_after_onload()
        assert backend.current_teacher == "B"
        assert backend.events == ["start", "nonresident", "disk_restore"]
        assert score(origin, "A")["teacher"] == "A"
    finally:
        service.close()
    assert backend.events[-1] == "close"
    assert not service.thread.is_alive()


def test_service_restore_failure_keeps_admission_closed():
    backend = Backend()
    service = SharedTeacherService(backend, ["A", "B"], "A", "127.0.0.1", timeout_seconds=5)
    origin = service.start()
    try:
        service.suspend()
        backend.fail_restore = True
        with pytest.raises(RuntimeError, match="restore failed"):
            service.restore_after_onload()
        with pytest.raises(HTTPError) as error:
            score(origin, "A")
        assert error.value.code == 503
        error.value.close()
        with pytest.raises(RuntimeError, match="restart"):
            service.restore_after_onload()
    finally:
        service.close()


class BlockedBackend(Backend):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.entered = threading.Event()
        self.release = asyncio.Event()
        self.completed = threading.Event()
        self.cancelled = False

    async def wait_for_release(self):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def start(self, teacher):
        if self.operation == "start":
            await self.wait_for_release()
        await super().start(teacher)
        if self.operation == "start":
            self.completed.set()

    async def restore_after_onload(self):
        await self.wait_for_release()
        await super().restore_after_onload()
        self.completed.set()


def start_control_thread(operation):
    errors = []
    finished = threading.Event()

    def run():
        try:
            operation()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, finished, errors


def release_control(service, backend):
    service.loop.call_soon_threadsafe(backend.release.set)
    assert backend.completed.wait(3)
    # The control continuation and its guard run before this loop-thread marker.
    drained = threading.Event()
    service.loop.call_soon_threadsafe(drained.set)
    assert drained.wait(3)


def assert_http_failed(origin):
    with pytest.raises(HTTPError) as error:
        score(origin, "A")
    assert error.value.code == 503
    error.value.close()
    with pytest.raises(HTTPError) as error:
        urlopen(origin + "/health", timeout=3)
    assert error.value.code == 503
    with error.value as response:
        health = json.load(response)
    assert not health["healthy"]
    assert not health["accepting_requests"]
    assert health["state"] == "failed"


def test_manager_shutdown_closes_http_then_stops_owned_upstream_before_waiting(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import ray

    from relax.distributed.ray import shared_teacher_manager as module

    backend = BlockedBackend("restore")
    service = SharedTeacherService(backend, ["A", "B"], "A", "127.0.0.1", timeout_seconds=5)
    origin = service.start()
    service.suspend()
    control, finished, errors = start_control_thread(service.restore_after_onload)
    assert backend.entered.wait(3)
    cls = module.SharedTeacherManager.__ray_metadata__.modified_class
    manager = object.__new__(cls)
    manager.service = service
    manager._failed = False
    manager._onloaded = False
    child = SimpleNamespace(
        shutdown=SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=lambda: "owned-shutdown"))
    )
    manager.engine_manager = child
    monkeypatch.setattr(module, "SERVICE_DRAIN_WAIT_S", 0.05)
    killed = []

    def complete_owned_shutdown(ref, *, timeout):
        assert ref == "owned-shutdown"
        assert_http_failed(origin)
        # This simulates the owned server exiting and resolving its HTTP call.
        release_control(service, backend)

    monkeypatch.setattr(ray, "get", complete_owned_shutdown)
    monkeypatch.setattr(ray, "kill", killed.append)
    try:
        manager.shutdown()
        assert finished.wait(3)
        assert errors and "restart" in str(errors[0])
        assert not backend.cancelled
        assert service.closed and not service.thread.is_alive()
        assert killed == [child]
    finally:
        if service.thread.is_alive():
            service.loop.call_soon_threadsafe(backend.release.set)
            service.close()
        control.join(timeout=3)


def test_late_restore_after_control_timeout_never_reopens_http():
    backend = BlockedBackend("restore")
    service = SharedTeacherService(backend, ["A", "B"], "A", "127.0.0.1", timeout_seconds=5)
    origin = service.start()
    thread = None
    try:
        service.suspend()
        service.timeout_seconds = 0.05
        thread, finished, errors = start_control_thread(service.restore_after_onload)
        assert backend.entered.wait(3)
        assert finished.wait(3)
        assert len(errors) == 1 and "completion is unknown" in str(errors[0])
        assert service.failed
        assert_http_failed(origin)

        release_control(service, backend)
        assert backend.resident  # The actual upstream recovery did finish.
        assert not backend.cancelled
        assert_http_failed(origin)
        assert backend.generate_calls == 0
        assert service.gateway.health()["state"] == "failed"
        with pytest.raises(RuntimeError, match="restart"):
            service.restore_after_onload()
    finally:
        service.timeout_seconds = 5
        service.loop.call_soon_threadsafe(backend.release.set)
        service.close()
        if thread is not None:
            thread.join(timeout=3)
    assert not service.thread.is_alive()


def test_close_waits_for_timed_out_restore_without_cancelling_it():
    backend = BlockedBackend("restore")
    service = SharedTeacherService(backend, ["A", "B"], "A", "127.0.0.1", timeout_seconds=5)
    origin = service.start()
    thread = None
    try:
        service.suspend()
        service.timeout_seconds = 0.05
        thread, finished, errors = start_control_thread(service.restore_after_onload)
        assert backend.entered.wait(3)
        assert finished.wait(3)
        assert len(errors) == 1 and "completion is unknown" in str(errors[0])
        with pytest.raises(RuntimeError, match="Cannot close shared teacher"):
            service.close()
        assert service.thread.is_alive()
        assert "close" not in backend.events
        assert not backend.completed.is_set()
        assert not backend.cancelled
        assert_http_failed(origin)
        release_control(service, backend)
    finally:
        service.timeout_seconds = 5
        service.loop.call_soon_threadsafe(backend.release.set)
        service.close()
        if thread is not None:
            thread.join(timeout=3)
    assert backend.events.count("close") == 1
    assert backend.events.index("disk_restore") < backend.events.index("close")
    assert not service.thread.is_alive()


def test_late_start_after_timeout_cannot_open_http():
    backend = BlockedBackend("start")
    service = SharedTeacherService(backend, ["A", "B"], "A", "127.0.0.1", timeout_seconds=0.05)
    thread, finished, errors = start_control_thread(service.start)
    try:
        assert backend.entered.wait(3)
        assert finished.wait(3)
        assert len(errors) == 1 and "completion is unknown" in str(errors[0])
        assert service.failed
        assert service.thread.is_alive()  # Still draining the un-cancelled start.
        assert "close" not in backend.events
        release_control(service, backend)
        assert backend.resident
        assert not backend.cancelled
        assert service.runner is None
        assert service.origin is None
        assert service.port == 0
        assert not service.gateway.health()["accepting_requests"]
        assert backend.generate_calls == 0
    finally:
        service.timeout_seconds = 5
        service.loop.call_soon_threadsafe(backend.release.set)
        service.close()
        thread.join(timeout=3)
    assert backend.events.count("close") == 1
    assert not service.thread.is_alive()
