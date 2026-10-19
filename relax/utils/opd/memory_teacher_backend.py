# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Complete CPU RAM checkpoint baseline for one SGLang TP1 worker.

The custom loader clones every checkpoint into ordinary CPU-owned tensors once.
Subsequent updates transfer every cached tensor to CUDA through the native HF
weight loader and postprocessing path. No deltas, mmap-backed retained weights,
GPU teacher backups, or disk-reload fallback are used. HTTP carries only an
empty tensor with a JSON control name. CPU caches live until the engine exits.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from relax.utils.opd.disk_teacher_backend import SGLangDiskBackend, _checkpoint_fingerprint, _require


LOADER = "relax.utils.opd.memory_teacher_backend.memory_teacher_loader"
_PREFIX = "relax_full_memory:"
_STATE = "_relax_full_memory_teacher_state"


def serialize_command(command: dict[str, Any]) -> list[str]:
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    name = _PREFIX + json.dumps(command, ensure_ascii=True, separators=(",", ":"))
    _require(len(name) <= 131072, "memory teacher control metadata is too large")
    return [MultiprocessingSerializer.serialize([(name, torch.empty(0, dtype=torch.uint8))], output_str=True)]


class SGLangMemoryBackend(SGLangDiskBackend):
    """Reuse the exclusive transaction/lifecycle, with tensor controls only."""

    _weight_source = "memory"

    def __init__(self, *args: Any, pin_memory: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _require(type(pin_memory) is bool, "pin_memory must be a boolean")
        self.pin_memory = pin_memory
        self._preparing = False
        self.worker_event_log = str(self.event_log) + ".worker.jsonl"

    def preflight(self) -> dict[str, Any]:
        fingerprint = super().preflight()
        config = fingerprint["config"]
        _require(
            not config.get("quantization_config") and not config.get("text_config", {}).get("quantization_config"),
            "memory baseline requires native BF16/F32 checkpoints without quantization",
        )
        return fingerprint

    async def start(self, initial_teacher: str) -> None:
        await super().start(initial_teacher)
        try:
            info = await self._request("/server_info")
            _require(not info.get("quantization"), "memory baseline does not support quantized engines")
            _require(info.get("disable_cuda_graph") is True, "memory baseline requires disabled CUDA graphs")
            _require(LOADER in (info.get("custom_weight_loader") or []), f"engine must allow custom loader {LOADER}")
            self._preparing = True
            await self._reload(initial_teacher, phase="cache_prepare")
            self._preparing = False
            # Explicitly exercise RAM loading before opening gateway admission.
            await self._reload(initial_teacher, phase="initial_restore")
        except BaseException:
            self.failed = True
            self.current_teacher = None
            await self.close()
            raise
        finally:
            self._preparing = False

    def _reload_event_metadata(self) -> dict[str, Any]:
        return {
            "event": "memory_cache_prepare" if self._preparing else "memory_full_checkpoint_reload",
            "weight_source": "cpu_ram",
            "pin_memory": self.pin_memory,
            "load_format": LOADER,
            "checkpoint_read_allowed": self._preparing,
            "page_cache_policy": "checkpoint reads only during preparation; RAM reloads never reopen weight files",
            "worker_event_log": self.worker_event_log,
            "cuda_completion": "worker synchronizes before and after native full-weight loading",
        }

    def _reload_request(self, teacher_id: str) -> tuple[str, str, dict[str, Any]]:
        command: dict[str, Any] = {"operation": "prepare" if self._preparing else "load", "teacher_id": teacher_id}
        if self._preparing:
            command.update(teachers=self.teachers, event_log=self.worker_event_log, pin_memory=self.pin_memory)
        return (
            "cache_prepare" if self._preparing else "memory_update",
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": serialize_command(command),
                "load_format": LOADER,
                "flush_cache": False,
                "abort_all_requests": False,
                "disable_draft_model": True,
                "torch_empty_cache": False,
            },
        )


@dataclass
class _Cache:
    weights: dict[str, list[tuple[str, Any]]] = field(default_factory=dict)
    current: str | None = None
    poisoned: bool = False
    prepared: bool = False
    event_log: Path | None = None
    bytes_by_teacher: dict[str, int] = field(default_factory=dict)
    pin_memory: bool = False


def _available_cpu_bytes() -> int:
    """Use Linux available RAM and, when set, the container's memory
    headroom."""
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        values[key] = int(value.split()[0]) * 1024
    available = values["MemAvailable"]
    limit_path = Path("/sys/fs/cgroup/memory.max")
    current_path = Path("/sys/fs/cgroup/memory.current")
    if limit_path.is_file() and current_path.is_file():
        raw = limit_path.read_text().strip()
        if raw != "max":
            available = min(available, max(0, int(raw) - int(current_path.read_text())))
    return available


def _checkpoint_files(directory: Path) -> tuple[list[Path], dict[str, str] | None]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        files = [directory / name for name in sorted(set(mapping.values()))]
    else:
        mapping = None
        files = sorted(directory.glob("*.safetensors"))
    _require(bool(files), "checkpoint has no safetensors weights")
    return files, mapping


def _load_cpu_checkpoint(
    files: list[Path], expected: dict[str, str] | None, *, pin_memory: bool = False
) -> list[tuple[str, Any]]:
    import torch
    from safetensors import safe_open

    result = []
    seen: dict[str, str] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                _require(name not in seen, f"duplicate checkpoint tensor: {name}")
                tensor = reader.get_tensor(name)
                _require(tensor.dtype in (torch.bfloat16, torch.float32), f"non-native BF16/F32 tensor: {name}")
                # A clone owns ordinary RAM independently of safetensors' mmap.
                owned = tensor.detach().clone(memory_format=torch.contiguous_format)
                if pin_memory:
                    owned = owned.pin_memory()
                    _require(owned.is_pinned(), "pin_memory allocation failed; no pageable fallback")
                result.append((name, owned))
                seen[name] = path.name
    _require(bool(result), "checkpoint has no tensors")
    _require(expected is None or seen == expected, "safetensors index does not match the actual complete tensor set")
    return result


def _prepare_cache(state: _Cache, teachers: dict[str, str]) -> None:
    _require(isinstance(teachers, dict) and bool(teachers), "prepare requires complete teacher directories")
    first = None
    plans = {}
    for teacher, directory in teachers.items():
        _require(isinstance(teacher, str) and isinstance(directory, str), "invalid teacher cache mapping")
        path = Path(directory)
        fingerprint = _checkpoint_fingerprint(path)
        config = fingerprint["config"]
        _require(
            not config.get("quantization_config") and not config.get("text_config", {}).get("quantization_config"),
            "memory cache requires nonquantized checkpoints",
        )
        if first is None:
            first = fingerprint
        _require(fingerprint == first, f"teacher {teacher} config/tokenizer differs")
        plans[teacher] = _checkpoint_files(path)
    sizes = [file.stat().st_size for files, _ in plans.values() for file in files]
    required = sum(sizes) + max(sizes) + 512 * 1024**2
    available = _available_cpu_bytes()
    _require(
        available >= required,
        f"insufficient CPU RAM for complete teacher cache: need {required} bytes, available {available}; no disk fallback",
    )
    layout = None
    for teacher, (files, expected) in plans.items():
        weights = _load_cpu_checkpoint(files, expected, pin_memory=state.pin_memory)
        actual = {name: (tuple(tensor.shape), str(tensor.dtype)) for name, tensor in weights}
        if layout is None:
            layout = actual
        _require(actual == layout, f"teacher {teacher} complete tensor names/shapes/dtypes differ")
        state.weights[teacher] = weights
        state.bytes_by_teacher[teacher] = sum(t.numel() * t.element_size() for _, t in weights)
    state.prepared = True


def _runtime_device(model: Any) -> Any:
    import torch
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group().cpu_group
    _require(
        group is not None and dist.is_initialized() and dist.get_world_size(group=group) == 1,
        "memory loader needs TP1",
    )
    params = list(model.parameters())
    _require(bool(params), "model has no parameters")
    device = params[0].device
    _require(device.type == "cuda", "memory loader requires a CUDA model")
    _require(
        all(p.device == device and p.dtype in (torch.bfloat16, torch.float32) for p in params),
        "memory loader requires one CUDA device and native BF16/F32 parameters",
    )
    for _, module in model.named_modules():
        _require(getattr(module, "quant_config", None) is None, "quantized modules are unsupported")
        method = getattr(module, "quant_method", None)
        _require(method is None or type(method).__name__.startswith("Unquantized"), "quantized weight method rejected")
    return device


def _synchronize(device: Any) -> None:
    import torch

    torch.cuda.synchronize(device)


def _apply_full_weights(model: Any, weights: list[tuple[str, Any]], device: Any) -> None:
    import torch
    from sglang.srt.model_loader.loader import DefaultModelLoader

    # A bounded one-tensor CUDA staging copy also prevents native loaders from
    # mutating the retained CPU snapshot. There is no second full GPU model.
    with torch.inference_mode():
        staging = ((name, tensor.to(device=device, non_blocking=False)) for name, tensor in weights)
        DefaultModelLoader.load_weights_and_postprocess(model, staging, device)


def _record_worker(state: _Cache, event: dict[str, Any]) -> None:
    _require(state.event_log is not None, "memory worker event log is not configured")
    with state.event_log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def memory_teacher_loader(model: Any, named_tensors: Any) -> None:
    """SGLang custom weight callback.

    An acknowledgement follows CUDA completion.
    """
    state = getattr(model, _STATE, None)
    if state is None:
        state = _Cache()
        setattr(model, _STATE, state)
    _require(not state.poisoned, "memory teacher worker is poisoned; restart required")
    event: dict[str, Any] = {"version": 1, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    try:
        _require(isinstance(named_tensors, list) and len(named_tensors) == 1, "expected one memory control marker")
        name, marker = named_tensors[0]
        _require(isinstance(name, str) and name.startswith(_PREFIX) and len(name) <= 131072, "invalid memory command")
        _require(marker.numel() == 0, "memory commands cannot contain weight tensors")
        command = json.loads(name[len(_PREFIX) :])
        operation, teacher = command["operation"], command["teacher_id"]
        _require(operation in {"prepare", "load"}, "unsupported memory operation")
        device = _runtime_device(model)
        event.update(
            event="memory_cache_prepare" if operation == "prepare" else "memory_full_restore", target_teacher=teacher
        )
        if operation == "prepare":
            _require(not state.prepared and not state.weights, "memory cache can only be prepared once")
            _require(teacher in command["teachers"], "initial teacher is missing")
            state.event_log = Path(command["event_log"])
            state.event_log.parent.mkdir(parents=True, exist_ok=True)
            state.pin_memory = command.get("pin_memory", False)
            _require(type(state.pin_memory) is bool, "pin_memory must be a boolean")
            tick = time.perf_counter()
            _prepare_cache(state, command["teachers"])
            event.update(
                prepare_seconds=time.perf_counter() - tick,
                cache_bytes_by_teacher=state.bytes_by_teacher,
                cache_total_bytes=sum(state.bytes_by_teacher.values()),
            )
        else:
            _require(state.prepared and teacher in state.weights, "teacher is absent from the prepared CPU cache")
            event.update(source_teacher=state.current, full_checkpoint_bytes=state.bytes_by_teacher[teacher])
            _synchronize(device)
            tick = time.perf_counter()
            _apply_full_weights(model, state.weights[teacher], device)
            _synchronize(device)
            event.update(synchronized_restore_seconds=time.perf_counter() - tick)
            state.current = teacher
        event.update(success=True, wall_seconds=time.perf_counter() - started, pin_memory=state.pin_memory)
        _record_worker(state, event)
    except BaseException as exc:
        state.poisoned = True
        state.current = None
        event.update(success=False, error_type=type(exc).__name__, wall_seconds=time.perf_counter() - started)
        try:
            if state.event_log is not None:
                _record_worker(state, event)
        except Exception:
            pass
        raise
