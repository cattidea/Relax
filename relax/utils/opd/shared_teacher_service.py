# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""An HTTP scoring gateway with a separate event loop from its Ray owner."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Coroutine

from relax.utils.opd.shared_teacher_gateway import SharedTeacherGateway, create_app


class SharedTeacherService:
    """Keep request admission closed throughout the owner's GPU offload window.

    The owner calls ``suspend`` before releasing GPU memory, and calls
    ``restore_after_onload`` only after allocating it again. A control timeout
    is fatal; it must never turn into an unverified retry or a blind resume.
    """

    def __init__(
        self,
        backend: Any,
        teacher_ids: list[str],
        initial_teacher: str,
        host: str,
        *,
        port: int = 0,
        timeout_seconds: float = 1800,
        max_batch_size: int = 32,
        max_queue_size: int = 1024,
        batch_wait_seconds: float = 0.005,
    ) -> None:
        self.backend = backend
        self.initial_teacher = initial_teacher
        self.host, self.port = host, port
        self.timeout_seconds = timeout_seconds
        self.gateway = SharedTeacherGateway(
            backend,
            teacher_ids,
            max_batch_size=max_batch_size,
            max_queue_size=max_queue_size,
            batch_wait_seconds=batch_wait_seconds,
        )
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, name="shared-teacher-http", daemon=True)
        self.runner: Any = None
        self.origin: str | None = None
        self.failed = False
        self.closed = False
        self._closing = False
        self._control_lock = threading.Lock()
        self._controls: set[Future] = set()
        self._close_future: Future | None = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _require_available(self) -> None:
        if self.failed or self.closed or self._closing:
            raise RuntimeError("Shared teacher service requires a restart")

    def _mark_failed(self) -> None:
        # The caller-thread latch closes HTTP admission before the loop callback
        # can run. Never cancel a control whose upstream completion is unknown.
        self.failed = True
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(
                self.gateway._fail_closed, RuntimeError("Shared teacher service requires a restart")
            )

    def _discard_control(self, future: Future) -> None:
        with self._control_lock:
            self._controls.discard(future)

    def _call(self, coroutine: Coroutine, *, timeout_seconds: float | None = None) -> Any:
        with self._control_lock:
            try:
                self._require_available()
            except RuntimeError:
                coroutine.close()
                raise
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            self._controls.add(future)
        future.add_done_callback(self._discard_control)
        try:
            return future.result(timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds)
        except FutureTimeoutError as exc:
            self._mark_failed()
            raise RuntimeError("Shared teacher control timed out; completion is unknown") from exc
        except Exception:
            self._mark_failed()
            raise

    async def _start(self) -> str:
        from aiohttp import web

        await self.backend.start(self.initial_teacher)
        self._require_available()

        @web.middleware
        async def admission(request: Any, handler: Any) -> Any:
            if self.failed or self.closed or self._closing:
                if request.path == "/health":
                    health = self.gateway.health()
                    health.update(
                        healthy=False,
                        accepting_requests=False,
                        state="failed" if self.failed else "closing",
                    )
                    return web.json_response(health, status=503)
                if request.method == "POST" and request.path.endswith("/generate"):
                    return web.json_response({"error": "shared teacher service requires a restart"}, status=503)
            return await handler(request)

        app = create_app(self.gateway)
        app.middlewares.append(admission)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self._require_available()
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(128)
            listener.setblocking(False)
            self.port = listener.getsockname()[1]
            await web.SockSite(self.runner, listener).start()
            self._require_available()
        except BaseException:
            listener.close()
            raise
        host = f"[{self.host}]" if family == socket.AF_INET6 else self.host
        return f"http://{host}:{self.port}"

    def start(self) -> str:
        self.thread.start()
        try:
            self.origin = self._call(self._start())
            return self.origin
        except BaseException:
            self.close()
            raise

    async def _suspend(self) -> None:
        await self.gateway.suspend()
        self.backend.mark_nonresident()

    def suspend(self, *, timeout_seconds: float | None = None) -> None:
        self._call(self._suspend(), timeout_seconds=timeout_seconds)

    async def _restore_after_onload(self) -> None:
        await self.backend.restore_after_onload()
        self._require_available()
        await self.gateway.resume()

    def restore_after_onload(self, *, timeout_seconds: float | None = None) -> None:
        self._call(self._restore_after_onload(), timeout_seconds=timeout_seconds)

    async def _snapshot(self) -> dict[str, Any]:
        return {"health": self.gateway.health(), "metrics": self.gateway.metrics()}

    def snapshot(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        return self._call(self._snapshot(), timeout_seconds=timeout_seconds)

    async def _close(self) -> None:
        await self.gateway.close()
        with self._control_lock:
            pending = list(self._controls)
        if pending:
            await asyncio.gather(*(asyncio.wrap_future(future) for future in pending), return_exceptions=True)
        if self.runner is not None:
            await self.runner.cleanup()
        await self.backend.close()

    def begin_close(self) -> None:
        """Close admission immediately, including while a control is in
        flight."""
        self._closing = True

    def close(self, *, timeout_seconds: float | None = None) -> None:
        if self.closed:
            return
        deadline = time.monotonic() + (self.timeout_seconds if timeout_seconds is None else timeout_seconds)
        with self._control_lock:
            self.begin_close()
            if self.thread.is_alive() and self._close_future is None:
                self._close_future = asyncio.run_coroutine_threadsafe(self._close(), self.loop)
        if self.thread.is_alive():
            future = self._close_future
            assert future is not None
            try:
                future.result(timeout=max(0, deadline - time.monotonic()))
            except FutureTimeoutError as exc:
                self._mark_failed()
                raise RuntimeError("Cannot close shared teacher while upstream completion is unknown") from exc
            finally:
                # A timeout leaves the loop alive to finish draining safely.
                if future.done():
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    self.thread.join(timeout=max(0, min(10, deadline - time.monotonic())))
        if not self.thread.is_alive():
            self.loop.close()
            self.closed = True
        else:
            raise RuntimeError("Shared Teacher HTTP thread termination is unconfirmed")
