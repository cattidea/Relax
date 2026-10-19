# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

import pytest

from relax.utils.opd import adp_teacher_backend as module


def test_switch_quiesces_flushes_commits_and_resumes(monkeypatch):
    monkeypatch.setattr(module, "serialize_command", lambda op, teacher, tp: [f"{op}:{teacher}"] * tp)

    async def scenario():
        backend = module.SGLangADPBackend("http://engine.test:1234", 2)
        calls = []

        async def request(path, payload=None, **kwargs):
            calls.append((path, payload, kwargs))
            return {"success": True}

        backend._request = request
        backend.current_teacher = "A"
        await backend.switch("B")
        assert [call[0] for call in calls] == [
            "/pause_generation",
            "/flush_cache",
            "/update_weights_from_tensor",
            "/flush_cache",
            "/update_weight_version",
            "/continue_generation",
        ]
        assert calls[0][1] == {"mode": "in_place"}
        assert calls[2][1]["serialized_named_tensors"] == ["switch:B", "switch:B"]
        assert calls[2][1]["disable_draft_model"] is True
        assert calls[2][1]["abort_all_requests"] is False
        assert backend.current_teacher == "B"
        await backend.switch("B")
        assert len(calls) == 6

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_path", ["/flush_cache", "/update_weights_from_tensor", "/update_weight_version", "/continue_generation"]
)
def test_partial_switch_failure_never_resumes_or_serves(monkeypatch, failure_path):
    monkeypatch.setattr(module, "serialize_command", lambda *args: ["command"])

    async def scenario():
        backend = module.SGLangADPBackend("http://engine.test:1234", 1)
        calls = []

        async def request(path, *args, **kwargs):
            calls.append(path)
            if path == failure_path:
                raise RuntimeError("injected failure")
            return {"success": True}

        backend._request = request
        backend.current_teacher = "A"
        with pytest.raises(RuntimeError, match="injected failure"):
            await backend.switch("B")
        assert backend.failed and backend.current_teacher is None
        if failure_path != "/continue_generation":
            assert "/continue_generation" not in calls
        count = len(calls)
        with pytest.raises(RuntimeError, match="unhealthy"):
            await backend.switch("A")
        with pytest.raises(RuntimeError, match="not ready"):
            await backend.generate({"input_ids": [1]})
        assert len(calls) == count

    asyncio.run(scenario())


def test_http_protocol_against_sglang_shaped_server(monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    monkeypatch.setattr(module, "serialize_command", lambda op, teacher, tp: [f"{op}:{teacher}"] * tp)

    async def scenario():
        events = []
        state = {"paused": False, "teacher": "A"}

        async def handler(request):
            path = request.path
            events.append(path)
            if path == "/server_info":
                return web.json_response(
                    {"tp_size": 1, "disable_cuda_graph": True, "custom_weight_loader": [module.LOADER]}
                )
            payload = await request.json()
            if path == "/pause_generation":
                state["paused"] = True
            elif path == "/continue_generation":
                state["paused"] = False
            elif path == "/flush_cache":
                assert state["paused"]
                return web.Response(text="Cache flushed")
            elif path == "/update_weights_from_tensor":
                assert state["paused"]
                state["teacher"] = payload["serialized_named_tensors"][0].split(":")[-1]
            elif path == "/generate":
                assert not state["paused"]
                return web.json_response({"teacher": state["teacher"], "echo": payload})
            return web.json_response({"success": True})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handler)
        async with TestServer(app) as server:
            backend = module.SGLangADPBackend(str(server.make_url("")), 1)
            try:
                await backend.start("A")
                await backend.switch("B")
                status, response = await backend.generate({"input_ids": [1, 2], "return_logprob": True})
                assert status == 200 and response["teacher"] == "B"
                assert response["echo"]["input_ids"] == [1, 2]
                assert events.count("/update_weights_from_tensor") == 2
            finally:
                await backend.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("origin", ["file:///tmp/engine", "http://engine.test/generate", "engine.test:1234"])
def test_reject_non_origin_upstream(origin):
    with pytest.raises(ValueError):
        module.SGLangADPBackend(origin, 1)
