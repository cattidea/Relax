# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional HTTP scheduler for a single runtime shared by static teachers.

The backend must finish ``switch`` before accepting generation, and
``generate`` must return only after that upstream request has finished. An
exception means completion is uncertain, so the gateway stops accepting work
until restarted. Only this scheduler may switch the runtime: external weight
writers are unsafe. The gateway does not own or close the injected backend.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol


class TeacherBackend(Protocol):
    async def switch(self, teacher_id: str) -> None: ...

    async def generate(self, payload: dict[str, Any]) -> tuple[int, Any]: ...


@dataclass
class _Request:
    payload: dict[str, Any]
    result: asyncio.Future


class SharedTeacherGateway:
    """Bounded per-teacher queues with round-robin, non-preemptible batches.

    ``max_queue_size`` bounds waiting requests across all teachers; at most
    ``max_batch_size`` more requests can be executing. Canceling a caller only
    abandons its response: queued or executing backend work is still drained.
    ``close`` rejects queued work and waits for the current batch without a
    timeout; a stuck upstream requires operator recovery, never a blind switch.
    """

    def __init__(
        self,
        backend: TeacherBackend,
        teacher_ids: list[str],
        *,
        max_queue_size: int = 256,
        max_batch_size: int = 8,
        batch_wait_seconds: float = 0.005,
    ) -> None:
        if not teacher_ids or len(set(teacher_ids)) != len(teacher_ids):
            raise ValueError("teacher_ids must be nonempty and unique")
        if any(not isinstance(t, str) or not t or "/" in t for t in teacher_ids):
            raise ValueError("teacher_ids must be nonempty URL path segments")
        if not isinstance(max_queue_size, int) or max_queue_size <= 0:
            raise ValueError("max_queue_size must be a positive integer")
        if not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if not math.isfinite(batch_wait_seconds) or batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be finite and nonnegative")
        self.backend = backend
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.batch_wait_seconds = batch_wait_seconds
        self._queues: dict[str, deque[_Request]] = {teacher: deque() for teacher in teacher_ids}
        self._ready: deque[str] = deque()
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task | None = None
        self._closing = False
        self._suspended = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._fatal_error: str | None = None
        self._active_teacher: str | None = None
        self._resident_teacher: str | None = None
        self._queued = 0
        self._in_flight = 0
        self._metrics = {
            teacher: dict(
                requests=0,
                completed=0,
                http_success=0,
                http_error=0,
                caller_cancelled=0,
                switches=0,
                switch_seconds=0.0,
                generate_seconds=0.0,
                batches=0,
                batch_seconds=0.0,
                last_batch_seconds=0.0,
                max_batch_seconds=0.0,
            )
            for teacher in teacher_ids
        }

    async def start(self) -> None:
        if self._closing or self._fatal_error is not None:
            raise RuntimeError("gateway cannot be restarted after shutdown or failure")
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="shared-teacher-gateway")

    async def generate(self, teacher_id: str, payload: dict[str, Any]) -> tuple[int, Any]:
        if teacher_id not in self._queues:
            return self._error(404, "unknown teacher")
        if not isinstance(payload, dict) or payload.get("stream"):
            return self._error(400, "a JSON object with streaming disabled is required")
        async with self._condition:
            if self._closing or self._suspended or self._fatal_error is not None:
                return self._error(503, "teacher gateway is unavailable")
            if self._worker is None:
                await self.start()
            if self._queued >= self.max_queue_size:
                return self._error(429, "teacher request queue is full")
            future = asyncio.get_running_loop().create_future()
            queue = self._queues[teacher_id]
            if not queue and teacher_id != self._active_teacher:
                self._ready.append(teacher_id)
            queue.append(_Request(dict(payload), future))
            self._queued += 1
            self._idle.clear()
            self._metrics[teacher_id]["requests"] += 1
            self._condition.notify()
        try:
            # HTTP cancellation cannot propagate to the worker or its upstreams.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self._metrics[teacher_id]["caller_cancelled"] += 1
            raise

    @staticmethod
    def _error(status: int, message: str) -> tuple[int, dict[str, str]]:
        return status, {"error": message}

    def _complete(self, teacher: str, request: _Request, response: tuple[int, Any]) -> None:
        if not request.result.done():
            request.result.set_result(response)
            self._metrics[teacher]["completed"] += 1
            # Native SGLang /generate returns 200 on success. This records the
            # response status only; OPD still validates the log-probability body.
            counter = "http_success" if response[0] == 200 else "http_error"
            self._metrics[teacher][counter] += 1

    def _record_batch(self, teacher: str, started: float) -> None:
        elapsed = time.monotonic() - started
        metrics = self._metrics[teacher]
        metrics["batches"] += 1
        metrics["batch_seconds"] += elapsed
        metrics["last_batch_seconds"] = elapsed
        metrics["max_batch_seconds"] = max(metrics["max_batch_seconds"], elapsed)

    def _reject_waiting(self, message: str) -> None:
        for teacher, queue in self._queues.items():
            while queue:
                self._complete(teacher, queue.popleft(), self._error(503, message))
        self._ready.clear()
        self._queued = 0

    def _fail_closed(self, exc: BaseException) -> None:
        if self._fatal_error is None:
            self._fatal_error = f"{type(exc).__name__}: {exc}"
        self._active_teacher = None
        self._resident_teacher = None
        self._reject_waiting("teacher state is uncertain; gateway stopped")

    async def _next_batch(self) -> tuple[str, list[_Request]] | None:
        async with self._condition:
            while not self._ready and not self._closing and self._fatal_error is None:
                await self._condition.wait()
            if self._closing or self._fatal_error is not None:
                return None
            teacher = self._ready.popleft()
            self._active_teacher = teacher
            collect = len(self._queues[teacher]) < self.max_batch_size and not self._ready
        if collect and self.batch_wait_seconds:
            await asyncio.sleep(self.batch_wait_seconds)
        async with self._condition:
            if self._closing or self._fatal_error is not None:
                return None
            queue = self._queues[teacher]
            if not queue:
                return None
            batch = [queue.popleft() for _ in range(min(len(queue), self.max_batch_size))]
            self._queued -= len(batch)
            self._in_flight = len(batch)
            return teacher, batch

    async def _generate_one(self, teacher: str, request: _Request) -> tuple[int, Any]:
        # A service control timeout may poison the gateway between dispatching
        # siblings. Do not start an upstream that has not yet been issued.
        if self._fatal_error is not None:
            return self._error(503, "teacher state is uncertain; gateway stopped")
        started = time.monotonic()
        try:
            response = await self.backend.generate(request.payload)
            if not isinstance(response, tuple) or len(response) != 2:
                raise TypeError("backend.generate must return (status, JSON)")
            if not isinstance(response[0], int) or not 200 <= response[0] <= 599:
                raise ValueError("backend.generate returned an invalid final HTTP status")
            return response
        finally:
            self._metrics[teacher]["generate_seconds"] += time.monotonic() - started

    async def _run(self) -> None:
        batch: list[_Request] = []
        teacher: str | None = None
        batch_started: float | None = None
        try:
            while (selected := await self._next_batch()) is not None:
                teacher, batch = selected
                batch_started = time.monotonic()
                if teacher != self._resident_teacher:
                    started = time.monotonic()
                    self._metrics[teacher]["switches"] += 1
                    try:
                        await self.backend.switch(teacher)
                    finally:
                        self._metrics[teacher]["switch_seconds"] += time.monotonic() - started
                    if self._fatal_error is None:
                        self._resident_teacher = teacher
                if self._fatal_error is not None:
                    # The switch itself was already issued and must finish,
                    # but a late acknowledgement cannot authorize new scoring.
                    raise RuntimeError("gateway failed while a switch was pending")
                # Wait for *all* upstreams, even if one fails early. Never use a
                # fail-fast gather that would release the lease with siblings live.
                responses = await asyncio.gather(
                    *(self._generate_one(teacher, request) for request in batch), return_exceptions=True
                )
                failure = next((result for result in responses if isinstance(result, BaseException)), None)
                if failure is not None:
                    self._fail_closed(failure)
                for request, response in zip(batch, responses):
                    if isinstance(response, BaseException):
                        response = self._error(502, "upstream completion is uncertain; gateway stopped")
                    self._complete(teacher, request, response)
                self._record_batch(teacher, batch_started)
                batch_started = None
                batch = []
                self._in_flight = 0
                if failure is not None or self._fatal_error is not None:
                    return
                async with self._condition:
                    # New teachers queued during this batch go first, preventing
                    # a busy resident teacher from starving another teacher.
                    if self._queues[teacher]:
                        self._ready.append(teacher)
                    self._active_teacher = None
                    if self._queued == 0:
                        self._idle.set()
        except (Exception, asyncio.CancelledError) as exc:
            self._fail_closed(exc)
            if teacher is not None:
                for request in batch:
                    self._complete(teacher, request, self._error(503, "teacher switch failed; gateway stopped"))
        finally:
            if teacher is not None and batch_started is not None:
                self._record_batch(teacher, batch_started)
            self._active_teacher = None
            self._in_flight = 0
            self._idle.set()

    async def suspend(self) -> None:
        """Close admission and finish every admitted request before offload."""
        async with self._condition:
            if self._closing or self._fatal_error is not None:
                raise RuntimeError("unhealthy gateway cannot be suspended")
            self._suspended = True
        await self._idle.wait()
        if self._fatal_error is not None:
            raise RuntimeError("teacher state became uncertain while draining")
        self._resident_teacher = None

    async def resume(self) -> None:
        """Reopen admission only after the owner restored valid GPU weights."""
        async with self._condition:
            if self._closing or self._fatal_error is not None or self._worker is None:
                raise RuntimeError("unhealthy gateway cannot be resumed")
            self._suspended = False

    async def close(self) -> None:
        async with self._condition:
            self._closing = True
            self._reject_waiting("teacher gateway is shutting down")
            self._condition.notify_all()
        if self._worker is not None:
            # Do not cancel a running switch or upstream request on HTTP shutdown.
            await asyncio.shield(self._worker)

    def health(self) -> dict[str, Any]:
        healthy = not self._closing and self._fatal_error is None and self._worker is not None
        return {
            "healthy": healthy,
            "state": "failed"
            if self._fatal_error
            else "closing"
            if self._closing
            else "suspended"
            if self._suspended
            else "ready"
            if healthy
            else "stopped",
            "active_teacher": self._active_teacher,
            "resident_teacher": self._resident_teacher,
            "queued_requests": self._queued,
            "in_flight": self._in_flight,
            "fatal_error": self._fatal_error,
            "accepting_requests": healthy and not self._suspended,
        }

    def metrics(self) -> dict[str, dict[str, int | float]]:
        """Admitted requests only; HTTP 200 is success, not proof of valid
        scores.

        Batch wall includes switching and concurrent generation, excluding
        queue and collection waits. ``generate_seconds`` sums individual
        requests. Switch and batch counts include failed attempts.
        """
        return {teacher: dict(values) for teacher, values in self._metrics.items()}


def create_app(gateway: SharedTeacherGateway, *, client_max_size: int = 16 * 1024**2) -> Any:
    """Create an aiohttp app without importing Ray, PyTorch, or SGLang."""
    from aiohttp import web

    async def generate(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"error": "invalid JSON"}, status=400)
        status, result = await gateway.generate(request.match_info["teacher_id"], payload)
        return web.json_response(result, status=status)

    async def health(request: web.Request) -> web.Response:
        result = gateway.health()
        return web.json_response(result, status=200 if result["healthy"] else 503)

    async def metrics(request: web.Request) -> web.Response:
        return web.json_response(gateway.metrics())

    async def lifecycle(app: web.Application) -> Any:
        await gateway.start()
        try:
            yield
        finally:
            await gateway.close()

    app = web.Application(client_max_size=client_max_size)
    app.router.add_post("/teachers/{teacher_id}/generate", generate)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.cleanup_ctx.append(lifecycle)
    return app
