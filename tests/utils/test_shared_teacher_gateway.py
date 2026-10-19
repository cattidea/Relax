# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-only scheduler and HTTP tests; no model, Ray, SGLang, or GPU
required."""

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from relax.utils.opd.shared_teacher_gateway import SharedTeacherGateway, create_app


class RecordingBackend:
    def __init__(self) -> None:
        self.teacher: str | None = None
        self.active = 0
        self.max_active = 0
        self.switches: list[str] = []
        self.started: list[tuple[str, str]] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.fail_switch: str | None = None

    async def switch(self, teacher_id: str) -> None:
        if self.active:
            raise AssertionError("weight switch overlapped an upstream request")
        self.switches.append(teacher_id)
        if teacher_id == self.fail_switch:
            raise RuntimeError("injected switch failure")
        self.teacher = teacher_id

    async def generate(self, payload: dict[str, Any]) -> tuple[int, Any]:
        teacher = self.teacher
        identifier = payload["id"]
        self.started.append((teacher, identifier))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if identifier in self.gates:
                await self.gates[identifier].wait()
            if payload.get("fail"):
                raise ConnectionError("upstream completion unknown")
            if teacher != self.teacher:
                raise AssertionError("teacher changed during generation")
            return payload.get("status", 200), {"teacher": teacher, "id": identifier}
        finally:
            self.active -= 1


async def wait_until(predicate: Any) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout=3)


class TestSharedTeacherGateway(unittest.IsolatedAsyncioTestCase):
    async def test_external_poison_during_collection_never_switches_rejected_requests(self) -> None:
        backend = RecordingBackend()
        collecting, release = asyncio.Event(), asyncio.Event()
        switch_called = asyncio.Event()
        original_switch = backend.switch

        async def switch(teacher: str) -> None:
            switch_called.set()
            await original_switch(teacher)

        async def collect(seconds: float) -> None:
            collecting.set()
            await release.wait()

        backend.switch = switch
        gateway = SharedTeacherGateway(backend, ["a", "b"], max_batch_size=4, batch_wait_seconds=1)
        with patch("asyncio.sleep", side_effect=collect):
            abandoned = asyncio.create_task(gateway.generate("a", {"id": "abandoned"}))
            rejected = asyncio.create_task(gateway.generate("a", {"id": "rejected"}))
            await collecting.wait()
            abandoned.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await abandoned
            gateway._fail_closed(TimeoutError("control timed out"))
            self.assertEqual((await rejected)[0], 503)
            self.assertEqual(gateway.health()["queued_requests"], 0)
            unexpected_switch = asyncio.create_task(switch_called.wait())
            try:
                release.set()
                done, _ = await asyncio.wait(
                    [gateway._worker, unexpected_switch], timeout=3, return_when=asyncio.FIRST_COMPLETED
                )
                self.assertIn(gateway._worker, done, "poisoned collection attempted another weight switch")
                self.assertEqual(backend.switches, [])
                self.assertEqual(backend.started, [])
                self.assertEqual(gateway.metrics()["a"]["caller_cancelled"], 1)
                self.assertEqual(gateway.metrics()["a"]["completed"], 2)
                self.assertEqual(gateway.metrics()["a"]["http_error"], 2)
            finally:
                release.set()
                unexpected_switch.cancel()
                await asyncio.gather(unexpected_switch, return_exceptions=True)
                await gateway.close()

    async def test_external_poison_during_switch_rejects_unsent_scoring_after_late_success(self) -> None:
        backend = RecordingBackend()
        entered, release = asyncio.Event(), asyncio.Event()
        original_switch = backend.switch

        async def switch(teacher: str) -> None:
            entered.set()
            await release.wait()
            await original_switch(teacher)

        backend.switch = switch
        gateway = SharedTeacherGateway(backend, ["a", "b"], batch_wait_seconds=0)
        request = asyncio.create_task(gateway.generate("a", {"id": "never-started"}))
        try:
            await entered.wait()
            gateway._fail_closed(TimeoutError("control timed out"))
            self.assertEqual(gateway.health()["in_flight"], 1)
            release.set()
            self.assertEqual((await request)[0], 503)
            self.assertEqual(backend.switches, ["a"])
            self.assertEqual(backend.started, [])
            self.assertIsNone(gateway.health()["resident_teacher"])
            self.assertIsNone(gateway.health()["active_teacher"])
            self.assertEqual(gateway.health()["fatal_error"], "TimeoutError: control timed out")
            self.assertEqual(gateway.health()["in_flight"], 0)
            self.assertEqual(gateway.metrics()["a"]["batches"], 1)
            self.assertEqual(gateway.metrics()["a"]["http_error"], 1)
            with self.assertRaises(RuntimeError):
                await gateway.resume()
        finally:
            release.set()
            await gateway.close()

    async def test_external_poison_between_sibling_dispatches_drains_only_already_started_upstream(self) -> None:
        backend = RecordingBackend()
        entered, release = asyncio.Event(), asyncio.Event()
        backend.gates["started"] = release
        original_generate = backend.generate
        gateway = SharedTeacherGateway(backend, ["a"], max_batch_size=2, batch_wait_seconds=0)

        async def generate(payload: dict) -> tuple:
            if payload["id"] == "started":
                gateway._fail_closed(TimeoutError("control timed out"))
                entered.set()
            return await original_generate(payload)

        backend.generate = generate
        first = asyncio.create_task(gateway.generate("a", {"id": "started"}))
        unsent = asyncio.create_task(gateway.generate("a", {"id": "unsent"}))
        try:
            await entered.wait()
            self.assertFalse(first.done())
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(first, unsent), 3)
            self.assertEqual([response[0] for response in responses], [200, 503])
            self.assertEqual(backend.started, [("a", "started")])
            self.assertEqual(backend.active, 0)
            self.assertIsNone(gateway.health()["resident_teacher"])
            self.assertEqual(gateway.metrics()["a"]["completed"], 2)
            self.assertEqual(gateway.metrics()["a"]["http_success"], 1)
            self.assertEqual(gateway.metrics()["a"]["http_error"], 1)
        finally:
            release.set()
            await gateway.close()

    async def test_poisoned_generation_survives_close_waiter_cancellation_until_drained(self) -> None:
        backend = RecordingBackend()
        entered, release, closing_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_generate = backend.generate
        backend.gates["started"] = release

        async def generate(payload: dict) -> tuple:
            entered.set()
            return await original_generate(payload)

        backend.generate = generate
        gateway = SharedTeacherGateway(backend, ["a", "b"], batch_wait_seconds=0)
        request = asyncio.create_task(gateway.generate("a", {"id": "started"}))

        async def close() -> None:
            closing_started.set()
            await gateway.close()

        try:
            await entered.wait()
            gateway._fail_closed(TimeoutError("control timed out"))
            closing = asyncio.create_task(close())
            await closing_started.wait()
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            self.assertEqual(backend.active, 1)
            self.assertFalse(gateway._worker.done())
            self.assertEqual((await gateway.generate("b", {"id": "late"}))[0], 503)
            release.set()
            self.assertEqual((await request)[0], 200)
            await asyncio.wait_for(gateway.close(), 3)
            self.assertEqual(backend.active, 0)
            self.assertEqual(backend.switches, ["a"])
            self.assertIsNone(gateway.health()["resident_teacher"])
        finally:
            release.set()
            await gateway.close()

    async def test_suspend_drains_admitted_teachers_and_blocks_new_requests_until_resume(self) -> None:
        backend = RecordingBackend()
        gate = asyncio.Event()
        backend.gates["slow"] = gate
        gateway = SharedTeacherGateway(backend, ["a", "b"], batch_wait_seconds=0)
        slow = asyncio.create_task(gateway.generate("a", {"id": "slow"}))
        try:
            await wait_until(lambda: backend.active == 1)
            queued = asyncio.create_task(gateway.generate("b", {"id": "queued"}))
            await wait_until(lambda: gateway.health()["queued_requests"] == 1)
            suspending = asyncio.create_task(gateway.suspend())
            await wait_until(lambda: gateway.health()["state"] == "suspended")
            self.assertEqual((await gateway.generate("a", {"id": "late"}))[0], 503)
            self.assertFalse(suspending.done())
            gate.set()
            self.assertEqual((await slow)[0], 200)
            self.assertEqual((await queued)[1]["teacher"], "b")
            await asyncio.wait_for(suspending, 3)
            self.assertEqual(backend.active, 0)
            self.assertFalse(gateway.health()["accepting_requests"])
            await gateway.resume()
            self.assertEqual((await gateway.generate("a", {"id": "next-round"}))[1]["teacher"], "a")
        finally:
            gate.set()
            await gateway.close()

    async def test_suspend_surfaces_switch_failure_instead_of_allowing_offload(self) -> None:
        backend = RecordingBackend()
        backend.fail_switch = "a"
        gateway = SharedTeacherGateway(backend, ["a", "b"], batch_wait_seconds=0.02)
        request = asyncio.create_task(gateway.generate("a", {"id": "bad"}))
        try:
            await wait_until(lambda: gateway.health()["queued_requests"] == 1)
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(gateway.suspend(), 3)
            self.assertEqual((await request)[0], 503)
            with self.assertRaises(RuntimeError):
                await gateway.resume()
        finally:
            await gateway.close()

    async def test_batches_are_concurrent_within_teacher_and_fair_between_teachers(self) -> None:
        backend = RecordingBackend()
        gate = asyncio.Event()
        backend.gates = {"a1": gate, "a2": gate}
        gateway = SharedTeacherGateway(backend, ["a", "b", "c"], max_batch_size=2, batch_wait_seconds=0.01)
        tasks = [
            asyncio.create_task(gateway.generate(teacher, {"id": identifier}))
            for teacher, identifier in [("a", "a1"), ("a", "a2"), ("a", "a3"), ("b", "b1"), ("c", "c1")]
        ]
        try:
            await wait_until(lambda: backend.active == 2)
            self.assertEqual(backend.started, [("a", "a1"), ("a", "a2")])
            gate.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 3)
            self.assertEqual(backend.switches, ["a", "b", "c", "a"])
            self.assertEqual(backend.max_active, 2)
            self.assertTrue(all(status == 200 for status, _ in results))
            self.assertEqual(gateway.metrics()["a"]["requests"], 3)
            self.assertEqual(gateway.metrics()["a"]["completed"], 3)
            self.assertEqual(gateway.metrics()["a"]["http_success"], 3)
            self.assertEqual(gateway.metrics()["a"]["http_error"], 0)
            self.assertEqual(gateway.metrics()["a"]["switches"], 2)
            self.assertEqual(gateway.metrics()["a"]["batches"], 2)
            self.assertGreater(gateway.metrics()["a"]["batch_seconds"], 0)
            self.assertGreater(gateway.metrics()["a"]["generate_seconds"], 0)
        finally:
            gate.set()
            await gateway.close()

    async def test_cancelled_caller_does_not_release_teacher_lease(self) -> None:
        backend = RecordingBackend()
        gate = asyncio.Event()
        backend.gates["slow"] = gate
        gateway = SharedTeacherGateway(backend, ["a", "b"], batch_wait_seconds=0)
        abandoned = asyncio.create_task(gateway.generate("a", {"id": "slow"}))
        try:
            await wait_until(lambda: backend.active == 1)
            abandoned.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await abandoned
            next_teacher = asyncio.create_task(gateway.generate("b", {"id": "next"}))
            await wait_until(lambda: gateway.health()["queued_requests"] == 1)
            self.assertEqual(backend.switches, ["a"])
            self.assertFalse(next_teacher.done())
            gate.set()
            self.assertEqual((await next_teacher)[1]["teacher"], "b")
            self.assertEqual(gateway.metrics()["a"]["caller_cancelled"], 1)
        finally:
            gate.set()
            await gateway.close()

    async def test_switch_failure_fails_closed_for_batch_queued_and_future_requests(self) -> None:
        backend = RecordingBackend()
        backend.fail_switch = "a"
        gateway = SharedTeacherGateway(backend, ["a", "b"], max_batch_size=2)
        tasks = [asyncio.create_task(gateway.generate(t, {"id": str(i)})) for i, t in enumerate(["a", "a", "b"])]
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), 3)
            self.assertEqual([r[0] for r in results], [503, 503, 503])
            self.assertEqual(backend.started, [])
            self.assertEqual(backend.switches, ["a"])
            self.assertEqual((await gateway.generate("b", {"id": "later"}))[0], 503)
            self.assertEqual(gateway.health()["state"], "failed")
            self.assertFalse(gateway.health()["healthy"])
            self.assertEqual(gateway.health()["queued_requests"], 0)
            self.assertEqual(gateway.metrics()["a"]["http_success"], 0)
            self.assertEqual(gateway.metrics()["a"]["http_error"], 2)
            self.assertEqual(gateway.metrics()["a"]["batches"], 1)
            self.assertEqual(gateway.metrics()["b"]["http_error"], 1)
            self.assertEqual(gateway.metrics()["b"]["batches"], 0)
        finally:
            await gateway.close()

    async def test_generation_exception_drains_siblings_before_stopping(self) -> None:
        backend = RecordingBackend()
        gate = asyncio.Event()
        backend.gates["slow"] = gate
        gateway = SharedTeacherGateway(backend, ["a", "b"], max_batch_size=2)
        bad = asyncio.create_task(gateway.generate("a", {"id": "bad", "fail": True}))
        slow = asyncio.create_task(gateway.generate("a", {"id": "slow"}))
        next_teacher = asyncio.create_task(gateway.generate("b", {"id": "next"}))
        try:
            await wait_until(lambda: ("a", "slow") in backend.started)
            self.assertEqual(backend.switches, ["a"])
            self.assertFalse(bad.done())
            self.assertFalse(next_teacher.done())
            gate.set()
            results = await asyncio.wait_for(asyncio.gather(bad, slow, next_teacher), 3)
            self.assertEqual([r[0] for r in results], [502, 200, 503])
            self.assertEqual(backend.active, 0)
            self.assertEqual(backend.switches, ["a"])
            self.assertFalse(gateway.health()["healthy"])
            self.assertEqual(gateway.metrics()["a"]["http_success"], 1)
            self.assertEqual(gateway.metrics()["a"]["http_error"], 1)
            self.assertEqual(gateway.metrics()["b"]["http_success"], 0)
            self.assertEqual(gateway.metrics()["b"]["http_error"], 1)
        finally:
            gate.set()
            await gateway.close()

    async def test_queue_bound_and_close_wait_for_running_upstream(self) -> None:
        backend = RecordingBackend()
        gate = asyncio.Event()
        backend.gates["slow"] = gate
        gateway = SharedTeacherGateway(backend, ["a", "b"], max_queue_size=1, max_batch_size=1)
        slow = asyncio.create_task(gateway.generate("a", {"id": "slow"}))
        try:
            await wait_until(lambda: backend.active == 1)
            queued = asyncio.create_task(gateway.generate("b", {"id": "queued"}))
            await wait_until(lambda: gateway.health()["queued_requests"] == 1)
            self.assertEqual((await gateway.generate("a", {"id": "overflow"}))[0], 429)
            closing = asyncio.create_task(gateway.close())
            self.assertEqual((await queued)[0], 503)
            self.assertFalse(closing.done())
            self.assertEqual((await gateway.generate("b", {"id": "late"}))[0], 503)
            gate.set()
            self.assertEqual((await slow)[0], 200)
            await asyncio.wait_for(closing, 3)
            self.assertEqual(backend.switches, ["a"])
        finally:
            gate.set()
            await gateway.close()

    async def test_http_disconnect_validation_status_and_metrics(self) -> None:
        from aiohttp.test_utils import TestClient, TestServer

        backend = RecordingBackend()
        gate = asyncio.Event()
        backend.gates["disconnected"] = gate
        gateway = SharedTeacherGateway(backend, ["a", "b"], batch_wait_seconds=0)
        client = TestClient(TestServer(create_app(gateway), handler_cancellation=True))
        await client.start_server()
        try:
            self.assertEqual((await client.get("/health")).status, 200)
            abandoned = asyncio.create_task(client.post("/teachers/a/generate", json={"id": "disconnected"}))
            await wait_until(lambda: backend.active == 1)
            abandoned.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await abandoned
            next_teacher = asyncio.create_task(client.post("/teachers/b/generate", json={"id": "next", "status": 422}))
            await wait_until(lambda: gateway.health()["queued_requests"] == 1)
            self.assertEqual(backend.switches, ["a"])
            gate.set()
            response = await next_teacher
            self.assertEqual(response.status, 422)
            self.assertEqual((await response.json())["teacher"], "b")
            for path, payload, expected in [
                ("/teachers/a/generate", {"stream": True}, 400),
                ("/teachers/a/generate", [], 400),
                ("/teachers/unknown/generate", {}, 404),
            ]:
                self.assertEqual((await client.post(path, json=payload)).status, expected)
            self.assertEqual((await client.post("/teachers/a/generate", data="not JSON")).status, 400)
            metrics = await (await client.get("/metrics")).json()
            self.assertEqual(metrics["a"]["completed"], 1)
            self.assertEqual(metrics["b"]["completed"], 1)
            self.assertEqual(metrics["a"]["http_success"], 1)
            self.assertEqual(metrics["a"]["http_error"], 0)
            self.assertEqual(metrics["b"]["http_success"], 0)
            self.assertEqual(metrics["b"]["http_error"], 1)
            self.assertTrue(gateway.health()["healthy"])
        finally:
            gate.set()
            await client.close()


if __name__ == "__main__":
    unittest.main()
