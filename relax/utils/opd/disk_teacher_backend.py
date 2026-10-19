# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Full-checkpoint disk reload baseline for one shared SGLang teacher engine.

This module does not use delta packages or retain CPU weight snapshots. The
gateway must be the engine's exclusive client and drain its generation batch
before switching. Checkpoints and tokenizer assets must be mounted locally.
Filesystem reads may hit the OS page cache: this is NOT a cold physical-disk
benchmark, and no system page cache is cleared.

The installed SGLang disk endpoint broadcasts to TP workers but acknowledges
only the leader's result. Until that endpoint gains all-rank error consensus,
this backend requires TP=PP=DP=EP=1. Standard dense/quantized checkpoints use
SGLang's native load_weights_and_postprocess path; a quantization mode still
needs native reload support in the installed SGLang version.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_BACKEND = "sglang-disk-v1"
_BACKENDS = {_BACKEND, "sglang-memory-v1", "sglang-adp-v1"}
_TEACHER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CONFIG_IDENTITY_KEYS = {"_name_or_path", "_commit_hash", "transformers_version"}
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "chat_template.jinja",
    "chat_template.json",
}
_PROCESSOR_FILES = {
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_fingerprint(directory: Path) -> dict[str, str]:
    _require(directory.is_dir(), f"tokenizer directory does not exist: {directory}")
    files = {name: directory / name for name in _TOKENIZER_FILES if (directory / name).is_file()}
    for path in (directory / "chat_templates").glob("*.jinja"):
        files[path.relative_to(directory).as_posix()] = path
    has_vocabulary = any(name in files for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt"))
    has_vocabulary = has_vocabulary or {"vocab.json", "merges.txt"}.issubset(files)
    _require(has_vocabulary, f"checkpoint lacks local tokenizer vocabulary: {directory}")
    return {name: _sha256_file(path) for name, path in sorted(files.items())}


def _checkpoint_fingerprint(directory: Path) -> dict[str, Any]:
    _require(directory.is_dir(), f"checkpoint directory does not exist: {directory}")
    config = json.loads((directory / "config.json").read_text())
    _require(isinstance(config, dict), "checkpoint config.json must contain an object")
    config = {key: value for key, value in config.items() if key not in _CONFIG_IDENTITY_KEYS}
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map")
        _require(isinstance(weight_map, dict) and bool(weight_map), f"invalid safetensors weight_map: {directory}")
        shards = set()
        for name, filename in weight_map.items():
            _require(isinstance(name, str) and bool(name), "invalid weight name in safetensors index")
            _require(isinstance(filename, str) and Path(filename).name == filename, "invalid safetensors shard path")
            _require(filename.endswith(".safetensors"), "disk baseline requires safetensors shards")
            shards.add(directory / filename)
    else:
        shards = set(directory.glob("*.safetensors"))
        _require(len(shards) == 1, f"checkpoint needs one safetensors file or a complete shard index: {directory}")
    _require(
        all(path.is_file() and path.stat().st_size > 0 for path in shards),
        f"missing/empty checkpoint shard: {directory}",
    )
    processor = {
        name: _sha256_file(directory / name) for name in sorted(_PROCESSOR_FILES) if (directory / name).is_file()
    }
    return {"config": config, "tokenizer": _tokenizer_fingerprint(directory), "processor": processor}


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a full-weight baseline manifest; resolve relative checkpoint paths.

    Schema: version=1, backend="sglang-disk-v1" (default) or "sglang-memory-v1", tp_size=1, initial_teacher,
    teachers={teacher_id: checkpoint_directory}, event_log=jsonl_path.
    The manifest may contain five or any other positive number of teachers.
    """
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    _require(
        isinstance(value, dict) and value.get("version") == 1 and value.get("backend", _BACKEND) in _BACKENDS,
        "invalid full-weight teacher manifest",
    )
    _require(
        type(value.get("tp_size")) is int and value["tp_size"] == 1,
        "disk baseline requires tp_size=1 for complete rank acknowledgement",
    )
    _require(
        isinstance(value.get("teachers"), dict) and bool(value["teachers"]),
        "manifest teachers must be a nonempty mapping",
    )
    _require(value.get("initial_teacher") in value["teachers"], "initial_teacher is absent from teachers")

    def resolve(raw: Any) -> str:
        _require(isinstance(raw, str) and bool(raw), "manifest paths must be nonempty strings")
        return str((path.parent / raw).resolve())

    result = dict(value)
    result["backend"] = value.get("backend", _BACKEND)
    result["teachers"] = {teacher: resolve(directory) for teacher, directory in value["teachers"].items()}
    result["event_log"] = resolve(value.get("event_log"))
    if result["backend"] == "sglang-adp-v1":
        result["package_dir"] = resolve(value.get("package_dir"))
        dependencies = value.get("dependency_pythonpath", [])
        _require(isinstance(dependencies, list), "dependency_pythonpath must be a list")
        result["dependency_pythonpath"] = [resolve(raw) for raw in dependencies]
        _require(
            all(Path(raw).is_dir() for raw in result["dependency_pythonpath"]), "ADP dependency directory missing"
        )
    return result


def create_backend(upstream: str, manifest: dict[str, Any], *, timeout_seconds: float = 1800) -> Any:
    """Select the explicit full-weight or colocated delta backend."""
    kind = manifest.get("backend", _BACKEND)
    _require(kind in _BACKENDS, "unsupported full-weight teacher backend")
    backend = SGLangDiskBackend
    options = {}
    if kind == "sglang-memory-v1":
        from relax.utils.opd.memory_teacher_backend import SGLangMemoryBackend

        backend = SGLangMemoryBackend
        options["pin_memory"] = manifest.get("pin_memory", False)
    elif kind == "sglang-adp-v1":
        from relax.utils.opd.colocated_adp_teacher_backend import SGLangColocatedADPBackend

        backend = SGLangColocatedADPBackend
        options["package_dir"] = manifest["package_dir"]
    return backend(
        upstream,
        manifest["tp_size"],
        manifest["teachers"],
        event_log=manifest["event_log"],
        timeout_seconds=timeout_seconds,
        **options,
    )


class SGLangDiskBackend:
    """A full reload transaction with permanent poison after uncertain failure.

    ``mark_nonresident`` follows gateway suspension/drain and precedes memory
    release. After the owner resumes BOTH weights and KV memory, it must await
    ``restore_after_onload`` before reopening gateway admission. That method
    reloads the logical teacher even when its ID has not changed.
    """

    _weight_source = "disk"

    def __init__(
        self,
        upstream: str,
        tp_size: int,
        teachers: dict[str, str],
        *,
        event_log: str | Path,
        timeout_seconds: float = 1800.0,
    ) -> None:
        parsed = urlparse(upstream)
        _require(
            parsed.scheme in {"http", "https"} and bool(parsed.netloc) and parsed.path in {"", "/"},
            "upstream must be an HTTP(S) engine origin without a path",
        )
        _require(
            not parsed.query and not parsed.fragment and parsed.username is None,
            "upstream must not contain query, fragment, or credentials",
        )
        _require(
            type(tp_size) is int and tp_size == 1,
            "disk baseline requires TP1: installed SGLang does not aggregate TP disk-update failures",
        )
        _require(math.isfinite(timeout_seconds) and timeout_seconds > 0, "timeout_seconds must be finite and positive")
        _require(isinstance(teachers, dict) and bool(teachers), "teachers must be a nonempty path mapping")
        _require(all(isinstance(name, str) and _TEACHER_ID.fullmatch(name) for name in teachers), "invalid teacher ID")
        _require(
            all(isinstance(path, str) and bool(path) for path in teachers.values()),
            "teacher paths must be nonempty strings",
        )
        self.upstream = upstream.rstrip("/")
        self.tp_size = tp_size
        self.teachers = {teacher: str(Path(path).resolve()) for teacher, path in teachers.items()}
        self.event_log = Path(event_log).resolve()
        self.timeout_seconds = timeout_seconds
        self.session: Any = None
        self.current_teacher: str | None = None
        self.resident = False
        self.failed = False
        self._closed = False
        self._updating = False
        self._in_flight = 0
        self._event_stream: Any = None
        self._recapture_cuda_graph = False
        self._fingerprint: dict[str, Any] | None = None

    def preflight(self) -> dict[str, Any]:
        """Read metadata/tokenizer bytes only; never load checkpoint
        tensors."""
        first = None
        for teacher, directory in self.teachers.items():
            fingerprint = _checkpoint_fingerprint(Path(directory))
            if first is None:
                first = fingerprint
            _require(fingerprint["config"] == first["config"], f"checkpoint config.json differs for teacher {teacher}")
            _require(fingerprint["tokenizer"] == first["tokenizer"], f"tokenizer assets differ for teacher {teacher}")
            _require(fingerprint["processor"] == first["processor"], f"processor assets differ for teacher {teacher}")
        return first

    async def _request(self, path: str, payload: dict | None = None, *, json_response: bool = True) -> Any:
        if self.session is None:
            raise RuntimeError("Disk backend has not been started")
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

    async def start(self, initial_teacher: str) -> None:
        import aiohttp

        if self.session is not None or self._closed or self.failed:
            raise RuntimeError("Disk backend cannot be started twice or after close/failure")
        _require(initial_teacher in self.teachers, "unknown initial teacher")
        self._fingerprint = await asyncio.to_thread(self.preflight)
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds))
        try:
            info = await self._request("/server_info")
            _require(info.get("tp_size") == 1, "engine must use TP1 for complete disk-update acknowledgement")
            _require(
                all(info.get(key, 1) == 1 for key in ("pp_size", "dp_size", "ep_size", "attn_cp_size")),
                "disk baseline supports a single PP/DP/EP/CP group",
            )
            _require(not info.get("speculative_algorithm"), "draft/speculative model reload is unsupported")
            _require(
                not info.get("enable_weights_cpu_backup") and not info.get("enable_draft_weights_cpu_backup"),
                "disk baseline forbids full CPU weight backups",
            )
            _require(not info.get("cpu_offload_gb", 0), "disk baseline forbids CPU weight offloading")
            actual_path = info.get("model_path")
            _require(
                isinstance(actual_path, str) and Path(actual_path).resolve() == Path(self.teachers[initial_teacher]),
                "engine model_path does not match the initial teacher checkpoint",
            )
            tokenizer_path = info.get("tokenizer_path") or actual_path
            tokenizer = await asyncio.to_thread(_tokenizer_fingerprint, Path(tokenizer_path).resolve())
            _require(tokenizer == self._fingerprint["tokenizer"], "engine tokenizer differs from teacher checkpoints")
            self._recapture_cuda_graph = not info.get("disable_cuda_graph", False)
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            self._event_stream = self.event_log.open("a", encoding="utf-8", buffering=1)
            self.current_teacher = initial_teacher
            self.resident = True
        except BaseException:
            await self.close()
            raise

    def _ensure_ready(self, *, require_resident: bool = True) -> None:
        if self.failed:
            raise RuntimeError("Disk backend is unhealthy; restart the engine from a known checkpoint")
        if self.session is None or self._closed or self.current_teacher is None:
            raise RuntimeError("Disk backend is not ready")
        if self._updating:
            raise RuntimeError("Disk backend is switching weights")
        if require_resident and not self.resident:
            raise RuntimeError("Disk backend is nonresident; restore_after_onload is required")

    def mark_nonresident(self) -> None:
        """Invalidate physical weights after admission has been
        suspended/drained."""
        self._ensure_ready(require_resident=False)
        if self._in_flight:
            raise RuntimeError("Cannot release teacher memory while generation is in flight")
        self.resident = False

    async def restore_after_onload(self) -> None:
        """Owner must resume weights and KV memory first; always reload from
        disk."""
        self._ensure_ready(require_resident=False)
        if self.resident:
            raise RuntimeError("Mark the teacher nonresident before restoring onload")
        await self._reload(self.current_teacher, phase="onload_restore")

    def _record(self, event: dict[str, Any]) -> None:
        if self._event_stream is None:
            raise RuntimeError("Disk reload event log is not open")
        self._event_stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        self._event_stream.flush()

    def _reload_event_metadata(self) -> dict[str, Any]:
        return {}

    def _reload_request(self, teacher_id: str) -> tuple[str, str, dict[str, Any]]:
        return (
            "disk_update",
            "/update_weights_from_disk",
            {
                "model_path": self.teachers[teacher_id],
                "load_format": "safetensors",
                "abort_all_requests": False,
                "is_async": False,
                "keep_pause": True,
                "flush_cache": False,
                "torch_empty_cache": False,
                "recapture_cuda_graph": self._recapture_cuda_graph,
            },
        )

    async def _reload(self, teacher_id: str, *, phase: str) -> None:
        self._ensure_ready(require_resident=phase != "onload_restore")
        _require(teacher_id in self.teachers, "unknown target teacher")
        if self._in_flight:
            raise RuntimeError("Teacher generation must be drained before a disk reload")
        self._updating = True
        source = self.current_teacher
        event = {
            "version": 1,
            "event": "disk_full_checkpoint_reload",
            "phase": phase,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_teacher": source,
            "target_teacher": teacher_id,
            "source_checkpoint": self.teachers[source],
            "target_checkpoint": self.teachers[teacher_id],
            "load_format": "safetensors",
            "tp_size": self.tp_size,
            "rank_acknowledgement": "single_tp_rank",
            "os_page_cache_cleared": False,
            "page_cache_policy": "uncontrolled; filesystem reads may be served from OS page cache",
            "recapture_cuda_graph": self._recapture_cuda_graph,
            "stage_seconds": {},
        }
        event.update(self._reload_event_metadata())
        stage = "pause"
        resumed = False
        started = time.perf_counter()

        async def request(label: str, path: str, payload: dict, **kwargs: Any) -> Any:
            nonlocal stage
            stage = label
            tick = time.perf_counter()
            try:
                return await self._request(path, payload, **kwargs)
            finally:
                event["stage_seconds"][label] = time.perf_counter() - tick

        try:
            await request("pause", "/pause_generation", {"mode": "in_place"})
            await request("flush_before", "/flush_cache", {}, json_response=False)
            label, endpoint, payload = self._reload_request(teacher_id)
            result = await request(label, endpoint, payload)
            _require(
                isinstance(result, dict) and result.get("success") is True,
                "weight update did not explicitly acknowledge success",
            )
            await request("flush_after", "/flush_cache", {}, json_response=False)
            await request(
                "weight_version",
                "/update_weight_version",
                {"new_version": f"{self._weight_source}:{teacher_id}", "abort_all_requests": False},
            )
            await request("resume", "/continue_generation", {})
            resumed = True
            event.update(success=True, wall_seconds=time.perf_counter() - started)
            stage = "event_log"
            self._record(event)
            self.current_teacher = teacher_id
            self.resident = True
        except BaseException as exc:
            self.failed = True
            self.current_teacher = None
            self.resident = False
            event.update(
                success=False,
                wall_seconds=time.perf_counter() - started,
                failed_stage=stage,
                error_type=type(exc).__name__,
            )
            if resumed or stage == "resume":
                # A log failure after continue must not reopen gateway admission.
                with suppress(BaseException):
                    await self._request("/pause_generation", {"mode": "in_place"})
            with suppress(Exception):
                self._record(event)
            raise
        finally:
            self._updating = False

    async def switch(self, teacher_id: str) -> None:
        self._ensure_ready()
        _require(teacher_id in self.teachers, "unknown target teacher")
        if teacher_id != self.current_teacher:
            await self._reload(teacher_id, phase="teacher_switch")

    async def generate(self, payload: dict) -> tuple[int, Any]:
        self._ensure_ready()
        if not isinstance(payload, dict) or payload.get("stream"):
            raise ValueError("Shared teacher scoring requires a JSON object with streaming disabled")
        self._in_flight += 1
        try:
            async with self.session.post(self.upstream + "/generate", json=payload) as response:
                return response.status, await response.json()
        except BaseException:
            # Completion is uncertain; never permit a subsequent weight switch.
            self.failed = True
            self.current_teacher = None
            self.resident = False
            raise
        finally:
            self._in_flight -= 1

    async def close(self) -> None:
        self._closed = True
        self.resident = False
        if self.session is not None:
            await self.session.close()
            self.session = None
        if self._event_stream is not None:
            self._event_stream.close()
            self._event_stream = None


def main() -> None:
    """Run a dedicated disk-baseline gateway without changing existing
    routes."""
    from aiohttp import web

    from relax.utils.opd.shared_teacher_gateway import SharedTeacherGateway, create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-queue-size", type=int, default=1024)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    backend = create_backend(args.upstream, manifest, timeout_seconds=args.timeout_seconds)
    gateway = SharedTeacherGateway(
        backend,
        list(manifest["teachers"]),
        max_batch_size=args.max_batch_size,
        max_queue_size=args.max_queue_size,
        batch_wait_seconds=args.batch_wait_ms / 1000,
    )
    app = create_app(gateway)

    async def start_backend(_app: Any) -> None:
        await backend.start(manifest["initial_teacher"])

    async def close_backend(_app: Any) -> None:
        await backend.close()

    app.on_startup.insert(0, start_backend)
    app.on_cleanup.append(close_backend)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
