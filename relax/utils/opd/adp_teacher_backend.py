# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exclusive, resident SGLang teacher control for the experimental ADP gateway.

The gateway is the only client allowed to access this engine. In particular,
Ray memory-saver/offload and student weight synchronization must not own it.
"""

from typing import Any
from urllib.parse import urlparse


LOADER = "relax.backends.sglang.adp_teacher_loader.adp_teacher_loader"


def serialize_command(operation: str, teacher_id: str, tp_size: int) -> list[str]:
    """Only a name and an empty tensor cross HTTP, never a compressed
    package."""
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    if operation not in {"capture", "prepare", "switch"}:
        raise ValueError(f"Unsupported ADP operation: {operation}")
    if not teacher_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in teacher_id
    ):
        raise ValueError("Teacher ID must contain only letters, digits, underscore, or hyphen")
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    marker = [(f"relax_adp:{operation}:{teacher_id}", torch.empty(0))]
    serialized = MultiprocessingSerializer.serialize(marker, output_str=True)
    return [serialized] * tp_size


class SGLangADPBackend:
    """A switch transaction; callers must drain all generation requests first.

    A failed transaction permanently poisons this object and leaves the engine
    paused. Restart the engine and gateway to recover a known base, rather than
    guessing whether any rank changed its weights.
    """

    def __init__(self, upstream: str, tp_size: int, *, timeout_seconds: float = 300.0) -> None:
        parsed = urlparse(upstream)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("upstream must be an HTTP(S) engine origin without a path")
        if tp_size < 1 or timeout_seconds <= 0:
            raise ValueError("tp_size and timeout must be positive")
        self.upstream = upstream.rstrip("/")
        self.tp_size = tp_size
        self.timeout_seconds = timeout_seconds
        self.session: Any = None
        self.current_teacher: str | None = None
        self.failed = False

    async def _request(self, path: str, payload: dict | None = None, *, json_response: bool = True) -> Any:
        if self.session is None:
            raise RuntimeError("Backend has not been started")
        async with self.session.request(
            "GET" if payload is None else "POST", self.upstream + path, json=payload
        ) as response:
            if response.status != 200:
                await response.read()
                raise RuntimeError(f"SGLang {path} returned HTTP {response.status}")
            result = await response.json() if json_response else await response.text()
            if isinstance(result, dict) and result.get("success") is False:
                raise RuntimeError(f"SGLang {path} rejected the operation")
            return result

    async def start(self, initial_teacher: str, *, prepare: bool = True) -> None:
        import aiohttp

        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds))
        try:
            info = await self._request("/server_info")
            if info.get("tp_size") != self.tp_size:
                raise ValueError("SGLang TP size differs from the configured package TP size")
            if any(info.get(key, 1) != 1 for key in ("pp_size", "dp_size", "ep_size")):
                raise ValueError("The pilot supports TP only; PP, DP and EP must be 1")
            if info.get("quantization") or info.get("speculative_algorithm") or info.get("enable_memory_saver"):
                raise ValueError("Quantization, speculative decoding and memory-saver are unsupported by the pilot")
            if not info.get("disable_cuda_graph"):
                raise ValueError("The pilot requires --disable-cuda-graph")
            if LOADER not in (info.get("custom_weight_loader") or []):
                raise ValueError(f"Start SGLang with --custom-weight-loader {LOADER}")
            if prepare:
                await self.control("prepare", initial_teacher)
        except BaseException:
            await self.close()
            raise

    async def control(self, operation: str, teacher_id: str) -> None:
        if self.failed:
            raise RuntimeError("ADP backend is unhealthy; restart from the initial checkpoint")
        serialized = serialize_command(operation, teacher_id, self.tp_size)
        try:
            # The central gateway has already drained its generation batch.
            await self._request("/pause_generation", {"mode": "in_place"})
            await self._request("/flush_cache", {}, json_response=False)
            await self._request(
                "/update_weights_from_tensor",
                {
                    "serialized_named_tensors": serialized,
                    "load_format": LOADER,
                    "flush_cache": False,
                    "abort_all_requests": False,
                    "disable_draft_model": True,
                },
            )
            await self._request("/flush_cache", {}, json_response=False)
            if operation in {"prepare", "switch"}:
                await self._request(
                    "/update_weight_version", {"new_version": f"adp:{teacher_id}", "abort_all_requests": False}
                )
            await self._request("/continue_generation", {})
        except BaseException:
            self.failed = True
            self.current_teacher = None
            # No unconditional continue: partial TP updates must never serve.
            raise
        if operation in {"prepare", "switch"}:
            self.current_teacher = teacher_id

    async def switch(self, teacher_id: str) -> None:
        if self.failed:
            raise RuntimeError("ADP backend is unhealthy")
        if teacher_id != self.current_teacher:
            await self.control("switch", teacher_id)

    async def generate(self, payload: dict) -> tuple[int, Any]:
        if self.failed or self.session is None or self.current_teacher is None:
            raise RuntimeError("ADP backend is not ready")
        if payload.get("stream"):
            raise ValueError("The shared teacher pilot supports non-streaming OPD scoring only")
        async with self.session.post(self.upstream + "/generate", json=payload) as response:
            return response.status, await response.json()

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None


def main() -> None:
    """Run an external shared teacher endpoint for Relax's existing URL
    routes."""
    import argparse

    from aiohttp import web

    from relax.utils.opd.shared_teacher_gateway import SharedTeacherGateway, create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=["serve", "capture"], default="serve")
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--initial-teacher", required=True)
    parser.add_argument("--teachers", nargs="+")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-queue-size", type=int, default=1024)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    args = parser.parse_args()
    backend = SGLangADPBackend(args.upstream, args.tp_size)
    if args.operation == "capture":
        import asyncio

        async def capture() -> None:
            try:
                await backend.start(args.initial_teacher, prepare=False)
                await backend.control("capture", args.initial_teacher)
            finally:
                await backend.close()

        asyncio.run(capture())
        return
    if not args.teachers or args.host is None or args.port is None:
        parser.error("serve requires --teachers, --host and --port")
    if args.initial_teacher not in args.teachers:
        parser.error("--initial-teacher must be in --teachers")
    gateway = SharedTeacherGateway(
        backend,
        args.teachers,
        max_batch_size=args.max_batch_size,
        max_queue_size=args.max_queue_size,
        batch_wait_seconds=args.batch_wait_ms / 1000,
    )
    app = create_app(gateway)

    async def prepare_backend(_app: Any) -> None:
        await backend.start(args.initial_teacher)

    async def close_backend(_app: Any) -> None:
        await backend.close()

    app.on_startup.insert(0, prepare_backend)
    app.on_cleanup.append(close_backend)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
