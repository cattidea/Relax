# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU/HTTP tests of the full-checkpoint baseline, without GPU or codec
imports."""

import asyncio
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from relax.utils.opd import disk_teacher_backend as module


@pytest.fixture
def checkpoints(tmp_path):
    teachers = {}
    for name in ("A", "B", "C", "D", "E"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps({"model_type": "llama", "hidden_size": 16, "_name_or_path": name})
        )
        (directory / "tokenizer.json").write_text('{"model":{"vocab":{"token":0}}}')
        (directory / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')
        # These tests inspect completeness/metadata, never parse weight tensors.
        (directory / "model.safetensors").write_bytes(b"test checkpoint placeholder")
        teachers[name] = str(directory)
    return teachers


def backend(checkpoints, tmp_path):
    return module.SGLangDiskBackend("http://engine.test:1234", 1, checkpoints, event_log=tmp_path / "events.jsonl")


def ready_backend(checkpoints, tmp_path):
    value = backend(checkpoints, tmp_path)
    value.session = object()
    value.current_teacher = "A"
    value.resident = True
    value._event_stream = io.StringIO()
    return value


def test_five_teacher_preflight_and_relative_manifest(checkpoints, tmp_path):
    value = backend(checkpoints, tmp_path)
    fingerprint = value.preflight()
    assert fingerprint["config"] == {"model_type": "llama", "hidden_size": 16}
    assert "tokenizer.json" in fingerprint["tokenizer"]
    manifest = tmp_path / "teachers.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "sglang-disk-v1",
                "tp_size": 1,
                "initial_teacher": "A",
                "teachers": {key: key for key in checkpoints},
                "event_log": "measurements/events.jsonl",
            }
        )
    )
    loaded = module.read_manifest(manifest)
    assert loaded["teachers"] == checkpoints
    assert loaded["event_log"] == str(tmp_path / "measurements/events.jsonl")


@pytest.mark.parametrize(
    "change", ["config", "tokenizer", "tokenizer_missing", "shard_missing", "unsafe_shard", "unindexed_shards"]
)
def test_preflight_rejects_incompatible_or_incomplete_checkpoint(checkpoints, tmp_path, change):
    path = Path(checkpoints["C"])
    if change == "config":
        (path / "config.json").write_text('{"model_type":"llama","hidden_size":32}')
    elif change == "tokenizer":
        (path / "tokenizer.json").write_text('{"model":{"vocab":{"other":0}}}')
    elif change == "tokenizer_missing":
        (path / "tokenizer.json").unlink()
    elif change in {"shard_missing", "unsafe_shard"}:
        shard = "missing.safetensors" if change == "shard_missing" else "../A/model.safetensors"
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": shard}}))
    else:
        (path / "extra.safetensors").write_bytes(b"other shard")
    with pytest.raises(ValueError):
        backend(checkpoints, tmp_path).preflight()


def test_quantized_checkpoints_with_identical_config_are_not_rejected(checkpoints, tmp_path):
    for path in checkpoints.values():
        (Path(path) / "config.json").write_text(
            json.dumps({"model_type": "llama", "hidden_size": 16, "quantization_config": {"quant_method": "fp8"}})
        )
    assert backend(checkpoints, tmp_path).preflight()["config"]["quantization_config"] == {"quant_method": "fp8"}


@pytest.mark.parametrize("filename", sorted(module._PROCESSOR_FILES))
def test_processor_assets_must_match_including_presence(checkpoints, tmp_path, filename):
    for directory in checkpoints.values():
        (Path(directory) / filename).write_text('{"size":224}')
    assert filename in backend(checkpoints, tmp_path).preflight()["processor"]
    path = Path(checkpoints["C"]) / filename
    path.write_text('{"size":448}')
    with pytest.raises(ValueError, match="processor assets"):
        backend(checkpoints, tmp_path).preflight()
    path.unlink()
    with pytest.raises(ValueError, match="processor assets"):
        backend(checkpoints, tmp_path).preflight()


def test_full_reload_protocol_timing_and_noop_same_teacher(checkpoints, tmp_path, monkeypatch):
    async def scenario():
        value = ready_backend(checkpoints, tmp_path)
        value._recapture_cuda_graph = True
        calls = []

        async def request(path, payload=None, **kwargs):
            calls.append((path, payload, kwargs))
            return {"success": True}

        value._request = request
        ticks = iter(range(100))
        monkeypatch.setattr(module.time, "perf_counter", lambda: float(next(ticks)))
        await value.switch("E")
        assert [item[0] for item in calls] == [
            "/pause_generation",
            "/flush_cache",
            "/update_weights_from_disk",
            "/flush_cache",
            "/update_weight_version",
            "/continue_generation",
        ]
        assert calls[0][1] == {"mode": "in_place"}
        update = calls[2][1]
        assert update == {
            "model_path": checkpoints["E"],
            "load_format": "safetensors",
            "abort_all_requests": False,
            "is_async": False,
            "keep_pause": True,
            "flush_cache": False,
            "torch_empty_cache": False,
            "recapture_cuda_graph": True,
        }
        assert calls[4][1] == {"new_version": "disk:E", "abort_all_requests": False}
        assert value.current_teacher == "E" and value.resident
        await value.switch("E")
        assert len(calls) == 6
        event = json.loads(value._event_stream.getvalue())
        assert event["phase"] == "teacher_switch" and event["success"] is True
        assert (event["source_teacher"], event["target_teacher"]) == ("A", "E")
        assert event["source_checkpoint"] == checkpoints["A"] and event["target_checkpoint"] == checkpoints["E"]
        assert event["wall_seconds"] == 13.0
        assert event["stage_seconds"] == {
            stage: 1.0 for stage in ("pause", "flush_before", "disk_update", "flush_after", "weight_version", "resume")
        }
        assert event["os_page_cache_cleared"] is False
        assert "OS page cache" in event["page_cache_policy"]

    asyncio.run(scenario())


def test_memory_saver_restore_forces_disk_reload_of_same_teacher(checkpoints, tmp_path):
    async def scenario():
        value = ready_backend(checkpoints, tmp_path)
        calls = []

        async def request(path, payload=None, **kwargs):
            calls.append((path, payload))
            return {"success": True}

        value._request = request
        value.mark_nonresident()
        assert value.current_teacher == "A" and not value.resident
        with pytest.raises(RuntimeError, match="nonresident"):
            await value.switch("A")
        with pytest.raises(RuntimeError, match="nonresident"):
            await value.generate({"input_ids": [1]})
        await value.restore_after_onload()
        assert value.resident and value.current_teacher == "A"
        assert calls[2] == (
            "/update_weights_from_disk",
            {
                "model_path": checkpoints["A"],
                "load_format": "safetensors",
                "abort_all_requests": False,
                "is_async": False,
                "keep_pause": True,
                "flush_cache": False,
                "torch_empty_cache": False,
                "recapture_cuda_graph": False,
            },
        )
        first = json.loads(value._event_stream.getvalue())
        assert first["phase"] == "onload_restore"
        assert first["source_teacher"] == first["target_teacher"] == "A"
        value.mark_nonresident()
        await value.restore_after_onload()
        assert len([path for path, _ in calls if path == "/update_weights_from_disk"]) == 2
        assert len(value._event_stream.getvalue().splitlines()) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("failed_call", range(6))
def test_any_transaction_failure_poison_prevents_resume_and_serving(checkpoints, tmp_path, failed_call):
    async def scenario():
        value = ready_backend(checkpoints, tmp_path)
        calls = []

        async def request(path, payload=None, **kwargs):
            calls.append(path)
            if len(calls) - 1 == failed_call:
                raise RuntimeError("injected failure")
            return {"success": True}

        value._request = request
        with pytest.raises(RuntimeError, match="injected failure"):
            await value.switch("B")
        assert value.failed and value.current_teacher is None and not value.resident
        if failed_call < 5:
            assert "/continue_generation" not in calls
        else:
            assert calls[-1] == "/pause_generation"
        event = json.loads(value._event_stream.getvalue())
        assert event["success"] is False and event["error_type"] == "RuntimeError"
        assert event["wall_seconds"] >= 0
        before = len(calls)
        with pytest.raises(RuntimeError, match="unhealthy"):
            await value.switch("A")
        with pytest.raises(RuntimeError, match="unhealthy"):
            await value.generate({"input_ids": [1]})
        with pytest.raises(RuntimeError, match="unhealthy"):
            await value.restore_after_onload()
        assert len(calls) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("reply", [{}, {"success": False}, {"success": "true"}, True])
def test_disk_update_requires_explicit_success(checkpoints, tmp_path, reply):
    async def scenario():
        value = ready_backend(checkpoints, tmp_path)
        calls = []

        async def request(path, payload=None, **kwargs):
            calls.append(path)
            return reply if path == "/update_weights_from_disk" else {"success": True}

        value._request = request
        with pytest.raises(ValueError, match="explicitly acknowledge"):
            await value.switch("B")
        assert value.failed and "/continue_generation" not in calls

    asyncio.run(scenario())


def test_backend_refuses_offload_or_switch_while_generate_is_active(checkpoints, tmp_path):
    async def scenario():
        value = ready_backend(checkpoints, tmp_path)
        value._in_flight = 1
        with pytest.raises(RuntimeError, match="in flight"):
            value.mark_nonresident()
        with pytest.raises(RuntimeError, match="drained"):
            await value.switch("B")
        assert value.current_teacher == "A" and value.resident and not value.failed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "info_override",
    [
        {"tp_size": 2},
        {"pp_size": 2},
        {"dp_size": 2},
        {"ep_size": 2},
        {"enable_weights_cpu_backup": True},
        {"cpu_offload_gb": 1},
        {"speculative_algorithm": "EAGLE"},
        {"model_path": "/wrong/checkpoint"},
    ],
)
def test_start_rejects_unsafe_engine_configuration(checkpoints, tmp_path, info_override):
    async def scenario():
        value = backend(checkpoints, tmp_path)

        async def request(path, *args, **kwargs):
            assert path == "/server_info"
            return {"tp_size": 1, "model_path": checkpoints["A"], "disable_cuda_graph": True, **info_override}

        value._request = request
        with pytest.raises(ValueError):
            await value.start("A")
        assert value.session is None and not value.resident

    asyncio.run(scenario())


def test_http_disk_reload_and_generate_with_memory_saver_enabled(checkpoints, tmp_path):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def scenario():
        events = []
        state = {"paused": False, "checkpoint": checkpoints["A"]}

        async def handler(request):
            events.append(request.path)
            if request.path == "/server_info":
                return web.json_response(
                    {
                        "tp_size": 1,
                        "model_path": checkpoints["A"],
                        "tokenizer_path": checkpoints["A"],
                        "enable_memory_saver": True,
                        "disable_cuda_graph": True,
                        "quantization": "fp8",
                    }
                )
            payload = await request.json()
            if request.path == "/pause_generation":
                state["paused"] = True
            elif request.path == "/flush_cache":
                assert state["paused"]
                return web.Response(text="Cache flushed")
            elif request.path == "/update_weights_from_disk":
                assert state["paused"] and payload["keep_pause"] is True
                assert payload["model_path"] in checkpoints.values()
                state["checkpoint"] = payload["model_path"]
            elif request.path == "/continue_generation":
                state["paused"] = False
            elif request.path == "/generate":
                assert not state["paused"]
                return web.json_response({"checkpoint": state["checkpoint"], "echo": payload})
            return web.json_response({"success": True})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        async with TestServer(app) as server:
            value = module.SGLangDiskBackend(
                str(server.make_url("")), 1, checkpoints, event_log=tmp_path / "events.jsonl"
            )
            try:
                await value.start("A")
                for teacher in checkpoints:
                    await value.switch(teacher)
                    status, response = await value.generate({"input_ids": [1, 2], "return_logprob": True})
                    assert status == 200 and response["checkpoint"] == checkpoints[teacher]
                value.mark_nonresident()
                await value.restore_after_onload()
                assert value.current_teacher == "E"
                assert events.count("/update_weights_from_disk") == 5
            finally:
                await value.close()
        logged = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert len(logged) == 5 and logged[-1]["phase"] == "onload_restore"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["disk", "memory"])
def test_full_weight_factory_does_not_import_adp_torch_or_sglang(tmp_path, checkpoints, kind):
    """The optional ADP branch must not leak into either baseline process."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "backend": f"sglang-{kind}-v1",
                "tp_size": 1,
                "teachers": checkpoints,
                "initial_teacher": "A",
                "event_log": "events",
            }
        )
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(module.__file__).resolve().parents[3])
    env["PYTHONSAFEPATH"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from relax.utils.opd.disk_teacher_backend import create_backend, read_manifest
backend = create_backend('http://engine', read_manifest(sys.argv[1]))
assert backend._weight_source == sys.argv[2]
for name in sys.modules:
    assert not name.startswith(('torch', 'sglang', 'adaptive_delta', 'batched_delta'))
    assert 'adp_teacher' not in name
""",
            str(manifest),
            kind,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("tp_size", [0, 2, 8, True])
def test_tp_more_than_one_is_rejected_until_full_rank_acknowledgement_exists(checkpoints, tmp_path, tp_size):
    with pytest.raises(ValueError, match="TP1"):
        module.SGLangDiskBackend("http://engine.test", tp_size, checkpoints, event_log=tmp_path / "events.jsonl")
