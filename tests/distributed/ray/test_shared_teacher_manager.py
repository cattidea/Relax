# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def manager_fixture(tmp_path):
    from relax.distributed.ray.shared_teacher_manager import SharedTeacherManager

    cls = SharedTeacherManager.__ray_metadata__.modified_class
    instance = object.__new__(cls)
    instance._onloaded = True
    instance._failed = False
    instance._control_timeout = 0.05
    instance.manifest = {"event_log": str(tmp_path / "full_reload_events.jsonl")}
    instance.service = SimpleNamespace(
        failed=False,
        suspend=Mock(),
        restore_after_onload=Mock(),
        snapshot=Mock(
            return_value={
                "health": {"healthy": True, "state": "suspended", "queued_requests": 0, "in_flight": 0},
                "metrics": {"T1": {"completed": 3, "http_success": 2, "http_error": 1, "batch_seconds": 0.5}},
            }
        ),
    )
    instance.engine_manager = SimpleNamespace(offload=Mock(), onload=Mock())
    return instance


def test_control_timeout_permanently_closes_admission_and_rejects_lifecycle_retry(monkeypatch, tmp_path):
    import ray

    instance = manager_fixture(tmp_path)
    calls = []

    def timeout(reference, *, timeout):
        calls.append(timeout)
        raise TimeoutError("engine completion is unknown")

    monkeypatch.setattr(ray, "get", timeout)
    with pytest.raises(TimeoutError):
        instance.offload()
    assert calls == [27]
    instance.service.suspend.assert_called_once()
    assert instance._failed and instance.service.failed
    assert not instance.health_check()
    with pytest.raises(RuntimeError, match="restart"):
        instance.onload()
    instance.service.restore_after_onload.assert_not_called()


def test_onload_awaits_bounded_engine_resume_before_forced_disk_restore(monkeypatch, tmp_path):
    import ray

    instance = manager_fixture(tmp_path)
    instance._onloaded = False
    events = []
    monkeypatch.setattr(ray, "get", lambda reference, *, timeout: events.append(("resume", timeout)))
    instance.service.restore_after_onload.side_effect = lambda **kwargs: events.append(("disk_restore", None))
    instance.onload()
    assert events == [("resume", 27), ("disk_restore", None)]
    assert instance._onloaded and not instance._failed


def test_offload_persists_drained_http_counts_before_engine_release(monkeypatch, tmp_path):
    import ray

    instance = manager_fixture(tmp_path)
    events = []
    instance.service.suspend.side_effect = lambda **kwargs: events.append("suspend")
    snapshot = instance.service.snapshot.return_value
    instance.service.snapshot.side_effect = lambda **kwargs: events.append("snapshot") or snapshot

    def release(reference, *, timeout):
        record = json.loads((tmp_path / "gateway_metrics.jsonl").read_text())
        assert record["phase"] == "before_offload"
        assert record["metrics"]["T1"]["http_success"] == 2
        assert record["metrics"]["T1"]["http_error"] == 1
        assert record["health"]["in_flight"] == 0
        events.append("release")

    monkeypatch.setattr(ray, "get", release)
    instance.offload()
    assert events == ["suspend", "snapshot", "release"]
    assert not instance._onloaded
    instance.offload()
    assert len((tmp_path / "gateway_metrics.jsonl").read_text().splitlines()) == 1


def test_offload_metrics_write_failure_keeps_admission_closed(monkeypatch, tmp_path):
    from pathlib import Path

    instance = manager_fixture(tmp_path)

    def fail_write(*args, **kwargs):
        raise OSError("injected metrics storage failure")

    monkeypatch.setattr(Path, "open", fail_write)
    with pytest.raises(OSError, match="metrics storage"):
        instance.offload()
    instance.service.suspend.assert_called_once()
    instance.engine_manager.offload.remote.assert_not_called()
    assert instance._failed and instance.service.failed
    assert not instance.health_check()
    with pytest.raises(RuntimeError, match="restart"):
        instance.onload()


@pytest.mark.parametrize("kind", ["sglang-disk-v1", "sglang-memory-v1"])
def test_constructor_selects_full_weight_backend_and_configures_only_one_engine(monkeypatch, kind):
    import ray

    from relax.distributed.ray import shared_teacher_manager as module
    from relax.utils.opd import disk_teacher_backend as disk
    from relax.utils.opd.memory_teacher_backend import LOADER

    manifest = {
        "backend": kind,
        "tp_size": 1,
        "initial_teacher": "A",
        "teachers": {name: f"/checkpoints/{name}" for name in "ABCDE"},
        "event_log": "/events.jsonl",
    }
    monkeypatch.setattr(disk, "read_manifest", lambda path: manifest)
    factory = Mock(return_value=object())
    monkeypatch.setattr(disk, "create_backend", factory)
    engine = SimpleNamespace(get_urls=SimpleNamespace(remote=lambda: ["http://engine:1234/generate"]))
    create_engine = Mock(return_value=engine)
    teacher_manager = SimpleNamespace(options=Mock(return_value=SimpleNamespace(remote=create_engine)))
    monkeypatch.setitem(
        sys.modules, "relax.distributed.ray.teacher_manager", SimpleNamespace(TeacherManager=teacher_manager)
    )
    monkeypatch.setattr(module, "with_control_plane_affinity", lambda args, options: options)
    monkeypatch.setattr(ray, "get", lambda reference, *, timeout: reference)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    service = SimpleNamespace(start=Mock(), origin="http://gateway")
    service_factory = Mock(return_value=service)
    monkeypatch.setattr(module, "SharedTeacherService", service_factory)
    args = SimpleNamespace(opd_teacher_timeout_s=99, teacher_sglang_custom_weight_loader="existing.loader")
    cls = module.SharedTeacherManager.__ray_metadata__.modified_class
    instance = cls(args, "manifest.json", pg=("placement-group",))

    assert create_engine.call_count == 1
    engine_args, replicas, tp = create_engine.call_args.args
    assert replicas == tp == 1
    assert engine_args.teacher_hf_checkpoint == "/checkpoints/A"
    assert engine_args.teacher_sglang_enable_weights_cpu_backup is False
    assert engine_args.teacher_sglang_disable_cuda_graph is True
    if kind == "sglang-memory-v1":
        assert engine_args.teacher_sglang_custom_weight_loader == ["existing.loader", LOADER]
    else:
        assert engine_args.teacher_sglang_custom_weight_loader == "existing.loader"
    assert args.teacher_sglang_custom_weight_loader == "existing.loader"
    factory.assert_called_once_with("http://engine:1234", manifest, timeout_seconds=99)
    service.start.assert_called_once()
    assert len(instance.get_urls()) == 5


def test_shutdown_stops_owned_engine_even_when_http_drain_is_blocked(monkeypatch, tmp_path):
    import ray

    instance = manager_fixture(tmp_path)
    events = []
    instance.service.begin_close = Mock(side_effect=lambda: events.append("admission_closed"))

    def close(*, timeout_seconds):
        events.append(("drain", timeout_seconds))
        if timeout_seconds == 2:
            raise RuntimeError("GPU completion unknown")

    instance.service.close = Mock(side_effect=close)
    instance.engine_manager.shutdown = Mock()
    child = instance.engine_manager
    monkeypatch.setattr(ray, "get", lambda ref, *, timeout: events.append(("owned_shutdown", timeout)))
    monkeypatch.setattr(ray, "kill", lambda handle: events.append(("owned_kill", handle is child)))
    instance.shutdown()
    assert events == ["admission_closed", ("drain", 2), ("owned_shutdown", 35), ("owned_kill", True), ("drain", 5)]
    assert instance._failed and not instance._onloaded and instance.engine_manager is None


def test_shutdown_does_not_report_unconfirmed_child_cleanup_as_success(monkeypatch, tmp_path):
    import ray

    instance = manager_fixture(tmp_path)
    instance.service.begin_close = Mock()
    instance.service.close = Mock()
    instance.engine_manager.shutdown = Mock()
    killed = []

    def timeout(*args, **kwargs):
        raise TimeoutError("owned engine failed to exit")

    monkeypatch.setattr(ray, "get", timeout)
    monkeypatch.setattr(ray, "kill", killed.append)
    with pytest.raises(RuntimeError, match="owned engine cleanup unconfirmed"):
        instance.shutdown()
    assert len(killed) == 1
    assert instance._failed
    assert instance.service.close.call_count == 2
    with pytest.raises(RuntimeError, match="owned engine cleanup unconfirmed"):
        instance.shutdown()
