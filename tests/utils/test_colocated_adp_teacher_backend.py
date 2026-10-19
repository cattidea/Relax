# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HTTP lifecycle tests; no SGLang, codec or CUDA worker is imported."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from relax.utils.opd import colocated_adp_teacher_backend as adp
from relax.utils.opd import disk_teacher_backend as disk


@pytest.fixture
def inputs(tmp_path):
    teachers = {}
    for teacher in ("A", "B"):
        path = tmp_path / teacher
        path.mkdir()
        (path / "config.json").write_text('{"model_type":"llama"}')
        (path / "tokenizer.json").write_text('{"model":{"vocab":{"x":0}}}')
        (path / "model.safetensors").write_bytes(b"metadata-only-test")
        teachers[teacher] = str(path)
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "manifest.json").write_text(
        json.dumps(
            {"version": 1, "runtime": "sglang-dense-v1", "tp_size": 1, "initial_teacher": "A", "teachers": teachers}
        )
    )
    return teachers, packages


def test_explicit_adp_factory_resolves_package_and_dependency_paths(inputs, tmp_path):
    teachers, packages = inputs
    dependency = tmp_path / "existing_dependency"
    dependency.mkdir()
    value = {
        "version": 1,
        "backend": "sglang-adp-v1",
        "tp_size": 1,
        "initial_teacher": "A",
        "teachers": teachers,
        "event_log": "events",
        "package_dir": packages.name,
        "dependency_pythonpath": [dependency.name],
    }
    path = tmp_path / "teacher_manifest.json"
    path.write_text(json.dumps(value))
    manifest = disk.read_manifest(path)
    backend = disk.create_backend("http://unused", manifest)
    assert isinstance(backend, adp.SGLangColocatedADPBackend)
    assert backend.package_dir == packages
    assert manifest["dependency_pythonpath"] == [str(dependency)]
    value["dependency_pythonpath"] = ["missing_dependency"]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="dependency"):
        disk.read_manifest(path)


def test_transactions_separate_resident_switch_and_forced_base_recovery(inputs, tmp_path, monkeypatch):
    teachers, packages = inputs
    monkeypatch.setattr(adp, "serialize_control", lambda operation, teacher: [f"{operation}:{teacher}"])
    commands = []
    paths = []

    async def scenario():
        async def handle(request):
            paths.append(request.path)
            if request.path == "/server_info":
                return web.json_response(
                    {
                        "tp_size": 1,
                        "model_path": teachers["A"],
                        "enable_memory_saver": True,
                        "disable_cuda_graph": True,
                        "custom_weight_loader": [adp.LOADER],
                    }
                )
            payload = await request.json()
            if request.path == "/update_weights_from_tensor":
                assert payload["load_format"] == adp.LOADER
                commands.extend(payload["serialized_named_tensors"])
            if request.path == "/flush_cache":
                return web.Response(text="flushed")
            return web.json_response({"success": True})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        async with TestServer(app) as server:
            backend = adp.SGLangColocatedADPBackend(
                str(server.make_url("")), 1, teachers, package_dir=packages, event_log=tmp_path / "events"
            )
            try:
                await backend.start("A")
                await backend.switch("B")
                await backend.switch("B")  # A resident no-op.
                backend.mark_nonresident()
                with pytest.raises(RuntimeError, match="nonresident"):
                    await backend.switch("B")
                # Simulate already-remapped memory. Backend must still issue
                # restore although the logical Teacher ID did not change.
                await backend.restore_after_onload()
                assert backend.current_teacher == "B" and backend.resident
                await backend.verify_bytes()
            finally:
                await backend.close()

    asyncio.run(scenario())
    assert commands == ["prepare_colocated:A", "switch:B", "restore:B", "verify:B"]
    assert "/update_weights_from_disk" not in paths
    events = [json.loads(line) for line in (tmp_path / "events").read_text().splitlines()]
    assert [e["event"] for e in events] == [
        "adp_prepare",
        "adp_delta_switch",
        "adp_restore_base_and_delta",
        "adp_validation_verify",
    ]
    assert [e["phase"] for e in events] == ["adp_prepare", "teacher_switch", "onload_restore", "validation_verify"]
    assert [e["full_cpu_recovery_base_used_this_transaction"] for e in events] == [False, False, True, False]
    assert [e["checkpoint_read_allowed"] for e in events] == [True, False, False, False]


def test_partial_restore_failure_poison_keeps_engine_paused(inputs, tmp_path, monkeypatch):
    teachers, packages = inputs
    monkeypatch.setattr(adp, "serialize_control", lambda operation, teacher: [operation])
    paths = []

    async def scenario():
        async def handle(request):
            if request.path == "/server_info":
                return web.json_response(
                    {
                        "tp_size": 1,
                        "model_path": teachers["A"],
                        "enable_memory_saver": True,
                        "disable_cuda_graph": True,
                        "custom_weight_loader": [adp.LOADER],
                    }
                )
            payload = await request.json()
            paths.append(request.path)
            if request.path == "/update_weights_from_tensor" and payload["serialized_named_tensors"] == ["restore"]:
                return web.json_response({"success": False})
            if request.path == "/flush_cache":
                return web.Response(text="flushed")
            return web.json_response({"success": True})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        async with TestServer(app) as server:
            backend = adp.SGLangColocatedADPBackend(
                str(server.make_url("")), 1, teachers, package_dir=packages, event_log=tmp_path / "events"
            )
            try:
                await backend.start("A")
                backend.mark_nonresident()
                paths.clear()
                with pytest.raises(RuntimeError, match="rejected"):
                    await backend.restore_after_onload()
                assert backend.failed and not backend.resident and backend.current_teacher is None
                assert "/continue_generation" not in paths
                with pytest.raises(RuntimeError, match="unhealthy"):
                    await backend.switch("B")
            finally:
                await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("field,value", [("tp_size", 2), ("runtime", "foreign"), ("teachers", {"A": {}})])
def test_preflight_rejects_incompatible_package_manifest(inputs, tmp_path, field, value):
    teachers, packages = inputs
    path = packages / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    backend = adp.SGLangColocatedADPBackend(
        "http://unused", 1, teachers, package_dir=packages, event_log=tmp_path / "events"
    )
    with pytest.raises(ValueError):
        backend.preflight()
