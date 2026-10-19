# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from tests.backends.sglang.test_router_registration import sglang_engine_module as sglang_engine_module


@pytest.fixture(autouse=True)
def record_lifecycle_logs(monkeypatch, sglang_engine_module, caplog):
    # Other collected modules can lazily reconfigure the process root logger.
    # Record the actual Logger.exception call directly, without relying on its
    # global handlers or replacing exception formatting with a mock.
    recorder = logging.Logger("test.shared_teacher.lifecycle")
    recorder.addHandler(caplog.handler)
    recorder.propagate = False
    monkeypatch.setattr(sglang_engine_module, "logger", recorder)


@pytest.mark.parametrize("method", ["release_memory_occupation", "resume_memory_occupation"])
def test_teacher_native_lifecycle_stalled_http_is_bounded(sglang_engine_module, method, caplog):
    entered, release = threading.Event(), threading.Event()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            seen.append(self.path)
            entered.set()
            release.wait(3)
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(b"{}")
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    engine.server_host, engine.server_port = server.server_address
    engine.node_rank = 0
    try:
        with caplog.at_level(logging.ERROR, logger=sglang_engine_module.logger.name):
            with pytest.raises(requests.exceptions.ReadTimeout) as raised:
                getattr(engine, method)(timeout_seconds=0.05)
        assert entered.is_set()
        assert seen[-1] == "/" + method
        if method == "release_memory_occupation":
            assert seen == ["/flush_cache", "/release_memory_occupation"]
        records = [record for record in caplog.records if record.name == sglang_engine_module.logger.name]
        assert len(records) == 1
        assert records[0].getMessage() == f"Shared Teacher native lifecycle failed: SGLangEngine.{method}"
        assert records[0].exc_info[1] is raised.value
        # Exercise the actual native requests/urllib3 cause chain. The emitted
        # record must retain it before Ray's rank broadcast bounds the message.
        rendered = logging.Formatter().format(records[0])
        assert "requests.exceptions.ReadTimeout:" in rendered
        assert "urllib3.exceptions.ReadTimeoutError:" in rendered
        assert f"port={engine.server_port}" in rendered
        assert "read timeout=" in rendered
        assert f"in {method}" in rendered
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_teacher_does_not_release_if_idle_flush_fails(monkeypatch, sglang_engine_module, caplog):
    error = requests.HTTPError("still has requests")
    response = SimpleNamespace(raise_for_status=Mock(side_effect=error))
    monkeypatch.setattr(sglang_engine_module.requests, "get", lambda *a, **k: response)
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    engine.server_host, engine.server_port, engine.node_rank = "127.0.0.1", 1, 0
    engine._make_request = Mock()
    with pytest.raises(requests.HTTPError) as raised:
        engine.release_memory_occupation(timeout_seconds=20)
    engine._make_request.assert_not_called()
    assert raised.value is error
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info[1] is error
    assert "SGLangEngine.release_memory_occupation" in caplog.records[0].getMessage()


@pytest.mark.parametrize("method", ["release_memory_occupation", "resume_memory_occupation"])
@pytest.mark.parametrize("bounded", [False, True])
def test_teacher_native_lifecycle_success_has_no_error_log(monkeypatch, sglang_engine_module, caplog, method, bounded):
    response = SimpleNamespace(raise_for_status=Mock(), json=lambda: {"success": True})
    get = Mock(return_value=response)
    post = Mock(return_value=response)
    monkeypatch.setattr(sglang_engine_module.requests, "get", get)
    monkeypatch.setattr(sglang_engine_module.requests, "post", post)
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    engine.server_host, engine.server_port, engine.node_rank = "127.0.0.1", 12345, 0
    engine.flush_cache = Mock()
    kwargs = {"timeout_seconds": 20} if bounded else {}
    if method == "resume_memory_occupation":
        kwargs["tags"] = ["weights", "private-payload-sentinel"]
    assert getattr(engine, method)(**kwargs) == {"success": True}
    assert not caplog.records
    assert post.call_count == 1
    assert post.call_args.args[0] == f"http://127.0.0.1:12345/{method}"
    timeout = post.call_args.kwargs["timeout"]
    assert (timeout is not None and 0 < timeout <= 20) if bounded else timeout is None
    if method == "release_memory_occupation":
        assert get.call_count == int(bounded)
        assert engine.flush_cache.call_count == int(not bounded)


@pytest.mark.parametrize("method", ["release_memory_occupation", "resume_memory_occupation"])
def test_legacy_native_lifecycle_failure_does_not_add_shared_log(sglang_engine_module, caplog, method):
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    error = requests.ReadTimeout("legacy failure")
    engine.flush_cache = Mock()
    engine._make_request = Mock(side_effect=error)
    with pytest.raises(requests.ReadTimeout) as raised:
        getattr(engine, method)()
    assert raised.value is error
    assert not caplog.records


def test_teacher_native_resume_log_does_not_format_payload(sglang_engine_module, caplog):
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    error = requests.ReadTimeout("native failure")
    engine._make_request = Mock(side_effect=error)
    with pytest.raises(requests.ReadTimeout) as raised:
        engine.resume_memory_occupation(tags=["private-payload-sentinel"], timeout_seconds=20)
    assert raised.value is error
    assert len(caplog.records) == 1
    assert "private-payload-sentinel" not in logging.Formatter().format(caplog.records[0])
