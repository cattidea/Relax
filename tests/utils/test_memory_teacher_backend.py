# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only tests; CUDA/model loading is replaced at the worker boundary."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from relax.utils.opd import disk_teacher_backend as disk
from relax.utils.opd import memory_teacher_backend as memory


@pytest.fixture
def checkpoints(tmp_path):
    result = {}
    for index, teacher in enumerate("ABCDE"):
        path = tmp_path / teacher
        path.mkdir()
        (path / "config.json").write_text('{"model_type":"llama","hidden_size":4,"dtype":"bfloat16"}')
        (path / "tokenizer.json").write_text('{"model":{"vocab":{"x":0}}}')
        (path / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')
        save_file(
            {"weight": torch.full((4, 4), index, dtype=torch.bfloat16), "scale": torch.ones(4)},
            str(path / "model.safetensors"),
        )
        result[teacher] = str(path)
    return result


def marker(operation, teacher="A", **extra):
    command = {"operation": operation, "teacher_id": teacher, **extra}
    return [(memory._PREFIX + json.dumps(command), torch.empty(0))]


@pytest.fixture
def cpu_worker(monkeypatch):
    calls = []
    model = SimpleNamespace()
    monkeypatch.setattr(memory, "_available_cpu_bytes", lambda: 10**12)
    monkeypatch.setattr(memory, "_runtime_device", lambda model: "test-cuda-device")
    monkeypatch.setattr(memory, "_synchronize", lambda device: calls.append("sync"))

    def apply(model, weights, device):
        calls.append("apply")
        model.live = {name: tensor.clone() for name, tensor in weights}

    monkeypatch.setattr(memory, "_apply_full_weights", apply)
    return model, calls


def test_complete_cpu_cache_survives_deleted_files_and_every_load_copies_all_tensors(
    checkpoints, tmp_path, cpu_worker
):
    model, calls = cpu_worker
    log = tmp_path / "worker.jsonl"
    memory.memory_teacher_loader(model, marker("prepare", teachers=checkpoints, event_log=str(log)))
    state = getattr(model, memory._STATE)
    assert len(state.weights) == 5 and state.prepared
    assert sum(state.bytes_by_teacher.values()) == 5 * (16 * 2 + 4 * 4)
    expected = {
        teacher: {name: tensor.clone() for name, tensor in weights} for teacher, weights in state.weights.items()
    }
    for directory in checkpoints.values():
        (Path(directory) / "model.safetensors").unlink()
    for teacher in "BCDEA":
        memory.memory_teacher_loader(model, marker("load", teacher))
        assert set(model.live) == {"weight", "scale"}
        assert all(torch.equal(model.live[n], expected[teacher][n]) for n in model.live)
        assert all(t.device.type == "cpu" for _, t in state.weights[teacher])
        model.live["weight"].zero_()
        assert all(torch.equal(t, expected[teacher][n]) for n, t in state.weights[teacher])
    assert calls == ["sync", "apply", "sync"] * 5
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[0]["event"] == "memory_cache_prepare" and events[0]["prepare_seconds"] >= 0
    assert events[0]["cache_total_bytes"] == 240 and events[0]["pin_memory"] is False
    assert all(e["synchronized_restore_seconds"] >= 0 for e in events[1:])
    assert all(e["full_checkpoint_bytes"] == 48 for e in events[1:])


@pytest.mark.parametrize("failure", ["ram", "shape", "dtype", "index"])
def test_preparation_rejects_incomplete_or_incompatible_cache(checkpoints, tmp_path, cpu_worker, monkeypatch, failure):
    model, _ = cpu_worker
    path = Path(checkpoints["B"])
    if failure == "ram":
        monkeypatch.setattr(memory, "_available_cpu_bytes", lambda: 0)
    elif failure in {"shape", "dtype"}:
        tensor = torch.ones(
            (3 if failure == "shape" else 4, 4), dtype=torch.float16 if failure == "dtype" else torch.bfloat16
        )
        save_file({"weight": tensor, "scale": torch.ones(4)}, str(path / "model.safetensors"))
    else:
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    with pytest.raises(ValueError):
        memory.memory_teacher_loader(model, marker("prepare", teachers=checkpoints, event_log=str(tmp_path / "log")))
    assert getattr(model, memory._STATE).poisoned
    with pytest.raises(ValueError, match="poisoned"):
        memory.memory_teacher_loader(model, marker("load"))


def test_prepare_is_once_only_and_failed_restore_permanently_poisoned(checkpoints, tmp_path, cpu_worker, monkeypatch):
    model, _ = cpu_worker
    prepare = marker("prepare", teachers=checkpoints, event_log=str(tmp_path / "log"))
    memory.memory_teacher_loader(model, prepare)
    with pytest.raises(ValueError, match="only be prepared once"):
        memory.memory_teacher_loader(model, prepare)
    assert getattr(model, memory._STATE).poisoned

    model = SimpleNamespace()
    memory.memory_teacher_loader(model, prepare)

    def failed_apply(*args):
        raise RuntimeError("partial GPU update")

    monkeypatch.setattr(memory, "_apply_full_weights", failed_apply)
    with pytest.raises(RuntimeError, match="partial GPU"):
        memory.memory_teacher_loader(model, marker("load", "B"))
    with pytest.raises(ValueError, match="poisoned"):
        memory.memory_teacher_loader(model, marker("load", "A"))


def test_pin_failure_does_not_fall_back_to_pageable(checkpoints, tmp_path, cpu_worker, monkeypatch):
    model, _ = cpu_worker

    def fail_pin(tensor):
        raise RuntimeError("insufficient locked RAM")

    monkeypatch.setattr(torch.Tensor, "pin_memory", fail_pin)
    with pytest.raises(RuntimeError, match="locked RAM"):
        memory.memory_teacher_loader(
            model, marker("prepare", teachers=checkpoints, event_log=str(tmp_path / "log"), pin_memory=True)
        )
    assert getattr(model, memory._STATE).poisoned


def test_native_full_weight_path_calls_postprocess_and_only_stages_one_tensor(monkeypatch):
    events = []

    class Tensor:
        def __init__(self, name):
            self.name = name

        def to(self, *, device, non_blocking):
            events.append(("copy", self.name, device, non_blocking))
            return self.name

    class NativeLoader:
        @staticmethod
        def load_weights_and_postprocess(model, tensors, device):
            assert not isinstance(tensors, list)
            for name, value in tensors:
                events.append(("load", name, value))
            events.append(("postprocess", device))

    monkeypatch.setitem(
        sys.modules, "sglang.srt.model_loader.loader", SimpleNamespace(DefaultModelLoader=NativeLoader)
    )
    memory._apply_full_weights(object(), [(name, Tensor(name)) for name in ("x", "y")], "cuda-test")
    assert events == [
        ("copy", "x", "cuda-test", False),
        ("load", "x", "x"),
        ("copy", "y", "cuda-test", False),
        ("load", "y", "y"),
        ("postprocess", "cuda-test"),
    ]


def test_factory_keeps_disk_default_and_memory_is_explicit(checkpoints, tmp_path):
    value = {
        "version": 1,
        "tp_size": 1,
        "teachers": checkpoints,
        "initial_teacher": "A",
        "event_log": str(tmp_path / "log"),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    manifest = disk.read_manifest(path)
    assert type(disk.create_backend("http://engine", manifest)) is disk.SGLangDiskBackend
    value.update(backend="sglang-memory-v1", pin_memory=True)
    path.write_text(json.dumps(value))
    backend = disk.create_backend("http://engine", disk.read_manifest(path))
    assert isinstance(backend, memory.SGLangMemoryBackend) and backend.pin_memory
    with pytest.raises(ValueError, match="boolean"):
        disk.create_backend("http://engine", {**value, "pin_memory": "true"})


def test_real_http_protocol_uses_cache_after_checkpoint_removal_and_onload(
    checkpoints, tmp_path, cpu_worker, monkeypatch
):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    model, _ = cpu_worker
    monkeypatch.setattr(memory, "serialize_command", lambda command: [json.dumps(command)])
    paths = []

    async def scenario():
        async def handler(request):
            paths.append(request.path)
            if request.path == "/server_info":
                return web.json_response(
                    {
                        "tp_size": 1,
                        "model_path": checkpoints["A"],
                        "disable_cuda_graph": True,
                        "custom_weight_loader": [memory.LOADER],
                    }
                )
            payload = await request.json()
            if request.path == "/update_weights_from_tensor":
                assert payload["load_format"] == memory.LOADER
                command = json.loads(payload["serialized_named_tensors"][0])
                memory.memory_teacher_loader(
                    model,
                    marker(**{"operation": command.pop("operation"), "teacher": command.pop("teacher_id"), **command}),
                )
            if request.path == "/flush_cache":
                return web.Response(text="Cache flushed")
            if request.path == "/generate":
                return web.json_response({"value": model.live["weight"][0, 0].item()})
            return web.json_response({"success": True})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        async with TestServer(app) as server:
            backend = memory.SGLangMemoryBackend(
                str(server.make_url("")), 1, checkpoints, event_log=tmp_path / "events"
            )
            try:
                await backend.start("A")
                for directory in checkpoints.values():
                    (Path(directory) / "model.safetensors").unlink()
                for teacher in "BCDEA":
                    await backend.switch(teacher)
                    status, result = await backend.generate({"input_ids": [1]})
                    assert status == 200 and result["value"] == ord(teacher) - ord("A")
                before = paths.count("/update_weights_from_tensor")
                await backend.switch("A")
                assert paths.count("/update_weights_from_tensor") == before
                backend.mark_nonresident()
                await backend.restore_after_onload()
                assert paths.count("/update_weights_from_tensor") == before + 1
            finally:
                await backend.close()
        assert "/update_weights_from_disk" not in paths
        events = [json.loads(line) for line in (tmp_path / "events").read_text().splitlines()]
        assert events[0]["event"] == "memory_cache_prepare"
        assert all(e["event"] == "memory_full_checkpoint_reload" for e in events[1:])
        assert all(not e["checkpoint_read_allowed"] for e in events[1:])
        assert events[-1]["phase"] == "onload_restore"

    asyncio.run(scenario())


def test_memory_update_failure_poison_prevents_resume(checkpoints, tmp_path, monkeypatch):
    async def scenario():
        backend = memory.SGLangMemoryBackend("http://engine", 1, checkpoints, event_log=tmp_path / "log")
        backend.session = object()
        backend.current_teacher = "A"
        backend.resident = True
        calls = []
        monkeypatch.setattr(memory, "serialize_command", lambda command: [json.dumps(command)])

        async def request(path, payload=None, **kwargs):
            calls.append(path)
            return {"success": path != "/update_weights_from_tensor"}

        backend._request = request
        with pytest.raises(ValueError, match="acknowledge"):
            await backend.switch("B")
        assert backend.failed and not backend.resident and backend.current_teacher is None
        assert "/continue_generation" not in calls and "/update_weights_from_disk" not in calls

    asyncio.run(scenario())
