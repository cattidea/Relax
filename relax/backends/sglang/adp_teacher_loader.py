# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional, in-place ADP1 teacher switching inside SGLang TP workers.

Register ``relax.backends.sglang.adp_teacher_loader.adp_teacher_loader`` as a
custom weight loader. Send exactly one empty tensor named
``relax_adp:{capture|prepare|prepare_colocated|switch|restore|verify}:{teacher_id}``.
The scheduler must quiesce
requests and flush KV caches around the call; this callback does not manage
request admission. Quantization, graphs, draft models and additional parallel
groups must be disabled. Ordinary prepare requires resident weights; offload
is supported only by the explicit prepare_colocated/restore lifecycle below.

RELAX_ADP_TEACHER_DIR/manifest.json has version=1, runtime="sglang-dense-v1",
tp_size, initial_teacher, group_bytes=67108864, teachers={id: {captures: [one
relative .pt path per TP rank]}}, and edges=[{source, target, packages: [one
relative ADP1 path per TP rank], reverse: false}]. All directed teacher pairs
are required. An edge with reverse=true may reference the opposite edge's
package only when every packet is XOR/PLANES (SKIP blocks are implicit).
Logical directions share one prepared plan per resolved package path. The
loader verifies directional provenance/hashes during preparation, then relies
on tracked Teacher identity and unchanged storage to re-arm that XOR plan.
Capture files contain params, auxiliary and metadata; they are preparation
inputs only. Prepared workers retain compressed plans, hashes and live storage
bindings. Ordinary prepare retains no full CPU weight snapshots; the opt-in
prepare_colocated also retains one pinned CPU initial-teacher recovery anchor.
Restore requires unchanged runtime bindings after memory-saver onload, copies
the anchor, then applies its directed package to the requested teacher.
RELAX_ADP_EVENT_LOG optionally records worker timing and retained CPU bytes.
Verify is an explicit offline-validation operation: it hashes current runtime
bytes against preparation's retained hashes, without reading capture files.
It must not be included in online switch latency or training requests.
Packages from another runtime are
not interchangeable, even when the checkpoint names and shapes match.
Strict releases provide package_integrity={relative_path: {sha256, bytes}} and
algorithm_sources={module_name: {sha256}} for adaptive_delta, batched_delta and
xor_codec. The whole package is authenticated by the codec while reading it.
Legacy complete build_progress.json digests are accepted for migration; a
manifest with neither integrity source retains the research-only v1 behavior.
Strict source checks reject modules imported before this loader attested them.

Tensor views fully contained in one parameter's storage region are recorded as
parameter views, not independent auxiliary values. Their shape/stride/offset
and live bindings must stay fixed; updating the parameter updates the view.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


_RUNTIME = "sglang-dense-v1"
_GROUP_BYTES = 64 * 1024**2
_REVERSIBLE_MODES = frozenset({"XOR_ANS", "XOR_GROUP2_ANS", "PLANES", "PLANES_ANS"})
_STATE_ATTRIBUTE = "_relax_adp_teacher_state"
_TEACHER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ALGORITHM_MODULES = ("xor_codec", "adaptive_delta", "batched_delta")
_ALGORITHM_PACKAGE = "relax.backends.sglang.adp"
_ATTESTED_ALGORITHM_MODULES: dict[str, tuple[Any, str, str]] = {}


@dataclass
class _State:
    current: str | None = None
    poisoned: bool = False
    prepared: bool = False
    manifest_digest: str | None = None
    bindings: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    plans: dict[tuple[str, str], Any] = field(default_factory=dict)
    teachers: frozenset[str] = frozenset()
    config_json: str | None = None
    base_teacher: str | None = None
    recovery_params: dict[str, Any] = field(default_factory=dict)
    recovery_auxiliary: dict[str, Any] = field(default_factory=dict)
    recovery_parameter_bytes: int = 0
    recovery_auxiliary_bytes: int = 0
    recovery_cpu_bytes: int = 0
    package_bank_bytes: int = 0
    teacher_hashes: dict[str, dict[str, str]] = field(default_factory=dict)
    auxiliary_hashes: dict[str, str] = field(default_factory=dict)
    algorithm_sources: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class _RuntimeView:
    params: dict[str, Any]
    auxiliary: dict[str, Any]
    metadata: dict[str, Any]
    bindings: dict[str, tuple[Any, ...]]
    config_json: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _layout(tensor: Any) -> dict[str, Any]:
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "stride": list(tensor.stride())}


def _binding(tensor: Any) -> tuple[Any, ...]:
    return (
        id(tensor),
        _class_name(tensor),
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
        tensor.storage_offset(),
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.device),
        str(tensor.layout),
    )


def _check_parameter(name: str, tensor: Any) -> None:
    import torch

    _require(tensor.device.type == "cuda", f"parameter {name} must be CUDA resident")
    _require(tensor.layout == torch.strided and tensor.is_contiguous(), f"noncontiguous parameter: {name}")
    _require(tensor.dtype in (torch.bfloat16, torch.float16, torch.float32), f"unsupported parameter dtype: {name}")
    _require(tensor.numel() > 0, f"empty parameter: {name}")


def _is_tensor_free_metadata(value: Any, seen: set[int]) -> bool:
    """Prove a skipped branch contains only known metadata, including its
    keys."""
    scalar_types = (type(None), bool, int, float, complex, str, bytes)
    if type(value) in scalar_types:
        return True
    if isinstance(value, Enum):
        return type(value.value) in scalar_types
    if type(value) not in (dict, list, tuple) or id(value) in seen:
        return False
    seen = seen | {id(value)}
    if isinstance(value, dict):
        return all(
            _is_tensor_free_metadata(key, seen) and _is_tensor_free_metadata(child, seen)
            for key, child in value.items()
        )
    return all(_is_tensor_free_metadata(child, seen) for child in value)


def _tensor_attributes(value: Any, path: str, seen: set[int]) -> Iterable[tuple[str, Any]]:
    """Inspect tensor attributes and ordinary containers without following
    modules."""
    import torch

    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (dict, list, tuple)):
        _require(id(value) not in seen, f"cyclic auxiliary container: {path}")
        seen = seen | {id(value)}
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in items:
            if not isinstance(key, (str, int)):
                # Qwen3-VL/Qwen3.5 has {Modality.IMAGE: True, ...} configuration
                # maps. Skip only proven metadata; the ordinary tensor walker
                # cannot prove that sets or unknown objects contain no tensors.
                _require(not isinstance(key, torch.Tensor), f"tensor auxiliary container key: {path}")
                _require(
                    _is_tensor_free_metadata(key, seen) and _is_tensor_free_metadata(child, seen),
                    f"unsupported auxiliary container key containing tensors or unknown metadata: {path}",
                )
                continue
            suffix = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
            yield from _tensor_attributes(child, f"{path}[{suffix}]", seen)


def _parameter_view(
    name: str, tensor: Any, storage_parameters: dict[tuple[str, int], list[tuple[str, Any]]]
) -> dict[str, Any] | None:
    """Describe a nonempty view covered by one contiguous parameter allocation.

    Storage identity alone is insufficient: parameters can occupy disjoint
    slices of a larger allocation. The complete strided access span must be
    inside one parameter's actual byte range. Unknown overlapping views fail
    closed instead of being silently treated as constant auxiliary state.
    """
    storage = tensor.untyped_storage()
    parameters = storage_parameters.get((str(tensor.device), storage.data_ptr()))
    if not parameters or tensor.numel() == 0:
        return None
    _require(not tensor.is_conj() and not tensor.is_neg(), f"lazy conjugate/negative parameter view: {name}")
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    _require(len(shape) == len(strides) and all(stride >= 0 for stride in strides), f"invalid view strides: {name}")
    element_bytes = tensor.element_size()
    start = tensor.storage_offset() * element_bytes
    span = (1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides))) * element_bytes
    end = start + span
    _require(0 <= start < end <= storage.nbytes(), f"parameter view outside its storage: {name}")
    _require(tensor.data_ptr() == storage.data_ptr() + start, f"inconsistent parameter view pointer: {name}")
    for parameter_name, parameter in parameters:
        parameter_storage = parameter.untyped_storage()
        parameter_start = parameter.storage_offset() * parameter.element_size()
        parameter_end = parameter_start + parameter.numel() * parameter.element_size()
        if parameter_start <= start and end <= parameter_end:
            _require(parameter.is_contiguous(), f"parameter view owner is not contiguous: {name}")
            _require(parameter.dtype == tensor.dtype, f"parameter view dtype reinterpretation: {name}")
            _require(
                parameter_storage.nbytes() == storage.nbytes()
                and parameter.data_ptr() == storage.data_ptr() + parameter_start,
                f"inconsistent shared parameter storage: {name}",
            )
            return {
                "parameter": parameter_name,
                "offset_bytes": start - parameter_start,
                "span_bytes": span,
                **_layout(tensor),
            }
    raise ValueError(f"parameter view is not fully covered by one parameter: {name}")


def _runtime_view(model: Any, *, prepared: _State | None = None) -> _RuntimeView:
    import torch

    params = dict(model.named_parameters())
    _require(bool(params), "model has no parameters")
    canonical = {id(tensor): name for name, tensor in params.items()}
    aliases = {
        name: canonical[id(tensor)]
        for name, tensor in model.named_parameters(remove_duplicate=False)
        if name != canonical[id(tensor)]
    }
    ranges: list[tuple[int, int, str]] = []
    storage_parameters: dict[tuple[str, int], list[tuple[str, Any]]] = {}
    for name, tensor in params.items():
        _check_parameter(name, tensor)
        ranges.append((tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size(), name))
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        storage_parameters.setdefault(key, []).append((name, tensor))
    _require(len({str(tensor.device) for tensor in params.values()}) == 1, "parameters span multiple devices")
    ranges.sort()
    _require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), "distinct parameters overlap in storage")

    auxiliary = {
        f"buffer:{name}": tensor
        for name, tensor in model.named_buffers(remove_duplicate=False)
        if id(tensor) not in canonical
    }
    for module_name, module in model.named_modules():
        for key in ("quant_config", "quantization_config"):
            _require(getattr(module, key, None) is None, f"quantized model is unsupported: {module_name}.{key}")
        method = getattr(module, "quant_method", None)
        _require(
            method is None or type(method).__name__.startswith("Unquantized"),
            f"quantized/unknown weight method is unsupported: {module_name}",
        )
        for attr, value in vars(module).items():
            if attr in ("_parameters", "_buffers", "_modules", _STATE_ATTRIBUTE):
                continue
            for path, tensor in _tensor_attributes(value, f"attr:{module_name}:{attr}", set()):
                if id(tensor) not in canonical:
                    auxiliary[path] = tensor
    parameter_views, view_tensors = {}, {}
    for name, tensor in list(auxiliary.items()):
        _require(tensor.layout == torch.strided, f"unsupported auxiliary tensor layout: {name}")
        _require(tensor.device.type in ("cpu", "cuda"), f"unsupported auxiliary tensor device: {name}")
        view = _parameter_view(name, tensor, storage_parameters)
        if view is not None:
            parameter_views[name], view_tensors[name] = view, tensor
            del auxiliary[name]

    config = getattr(model, "config", None)
    config_json = None
    if config is not None:
        config = config.to_dict() if hasattr(config, "to_dict") else config
        _require(isinstance(config, dict), "model config must provide to_dict() or be a mapping")
        config = {
            k: v for k, v in config.items() if k not in ("_name_or_path", "_commit_hash", "transformers_version")
        }
        _require(config.get("quantization_config") is None, "quantized model config is unsupported")
        config_json = json.dumps(config, sort_keys=True, default=str)
    if prepared is not None:
        _require(config_json == prepared.config_json, "runtime parameter/auxiliary layout or storage changed: config")
        config = prepared.metadata["config_sha256"]
    else:
        config = hashlib.sha256(config_json.encode()).hexdigest() if config_json is not None else None
    metadata = {
        "model_class": _class_name(model),
        "config_sha256": config,
        "layout": {name: _layout(tensor) for name, tensor in params.items()},
        "aliases": aliases,
        "auxlayout": {name: _layout(tensor) for name, tensor in auxiliary.items()},
    }
    # Preserve capture compatibility for models without parameter-backed views.
    # Models with such views require new captures with this explicit contract.
    if parameter_views:
        metadata["parameter_views"] = parameter_views
    bindings = {f"param:{name}": _binding(tensor) for name, tensor in params.items()}
    bindings.update({f"aux:{name}": _binding(tensor) for name, tensor in auxiliary.items()})
    bindings.update({f"view:{name}": _binding(tensor) for name, tensor in view_tensors.items()})
    return _RuntimeView(params, auxiliary, metadata, bindings, config_json)


def _hash_tensor(tensor: Any) -> str:
    """Slow preparation-only byte hash, with bounded GPU-to-CPU transfers."""
    import torch

    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), _GROUP_BYTES):
        chunk = raw[offset : offset + _GROUP_BYTES].to(device="cpu")
        digest.update(memoryview(chunk.numpy()))
    return digest.hexdigest()


def _tp_context() -> tuple[Any, Any, int, int]:
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group().cpu_group
    _require(group is not None and dist.is_initialized(), "SGLang TP CPU process group is not initialized")
    return dist, group, dist.get_rank(group=group), dist.get_world_size(group=group)


def _poison(state: _State) -> None:
    state.current = None
    state.poisoned = True


def _consensus(
    context: tuple[Any, Any, int, int],
    state: _State,
    route: tuple[Any, ...],
    error: str | None,
    digest: str | None = None,
) -> None:
    dist, group, _, size = context
    records: list[Any] = [None] * size
    local = {"route": route, "error": error, "digest": digest}
    try:
        dist.all_gather_object(records, local, group=group)
    except Exception:
        _poison(state)
        raise
    errors = [f"TP rank {rank}: {item['error']}" for rank, item in enumerate(records) if item["error"]]
    if any(item["route"] != route for item in records):
        errors.append("TP workers disagree on command or current teacher state")
    if not errors and any(item["digest"] != digest for item in records):
        errors.append("TP workers read different manifests")
    if errors:
        if any(item["route"][1] in ("switch", "restore", "verify") for item in records):
            _poison(state)
        raise RuntimeError("ADP TP consensus failed: " + "; ".join(errors))


def _command(named_tensors: Any) -> tuple[str, str]:
    import torch

    values = list(named_tensors)
    _require(len(values) == 1, "ADP control requires exactly one named empty tensor")
    name, tensor = values[0]
    _require(isinstance(name, str), "ADP command name must be a string")
    parts = name.split(":")
    _require(len(parts) == 3 and parts[0] == "relax_adp", "invalid ADP command name")
    operation, teacher = parts[1:]
    _require(
        operation in ("capture", "prepare", "prepare_colocated", "switch", "restore", "verify"),
        "unknown ADP operation",
    )
    _require(bool(_TEACHER_ID.fullmatch(teacher)), "invalid teacher ID")
    _require(isinstance(tensor, torch.Tensor) and tensor.numel() == 0, "ADP command tensor must be empty")
    return operation, teacher


def _root_directory() -> Path:
    directory = os.environ.get("RELAX_ADP_TEACHER_DIR")
    _require(bool(directory), "RELAX_ADP_TEACHER_DIR is required")
    return Path(directory).resolve()


def _relative_file(root: Path, value: Any) -> Path:
    _require(
        isinstance(value, str) and bool(value) and not Path(value).is_absolute(), "expected relative package path"
    )
    path = (root / value).resolve()
    _require(root in path.parents, "package path escapes RELAX_ADP_TEACHER_DIR")
    return path


def _capture(view: _RuntimeView, root: Path, teacher: str, rank: int, size: int) -> None:
    import torch

    snapshot = {
        "version": 1,
        "runtime": _RUNTIME,
        "teacher_id": teacher,
        "tp_rank": rank,
        "tp_size": size,
        "metadata": view.metadata,
        "params": {
            name: tensor.detach().to(device="cpu", copy=True).contiguous() for name, tensor in view.params.items()
        },
        "auxiliary": {
            name: tensor.detach().to(device="cpu", copy=True).contiguous() for name, tensor in view.auxiliary.items()
        },
    }
    path = _relative_file(root, f"captures/{teacher}/rank_{rank}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
    try:
        torch.save(snapshot, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_hashes(
    path: Path, view: _RuntimeView, teacher: str, rank: int, size: int
) -> tuple[dict[str, str], dict[str, str]]:
    import torch

    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    _require(snapshot["version"] == 1 and snapshot["runtime"] == _RUNTIME, "unsupported capture runtime/version")
    _require(
        (snapshot["teacher_id"], snapshot["tp_rank"], snapshot["tp_size"]) == (teacher, rank, size),
        "capture teacher/TP identity mismatch",
    )
    _require(snapshot["metadata"] == view.metadata, f"capture runtime layout/config mismatch: {teacher}")
    result = []
    for key, expected in (("params", view.params), ("auxiliary", view.auxiliary)):
        values = snapshot[key]
        _require(set(values) == set(expected), f"capture {key} coverage mismatch: {teacher}")
        hashes = {}
        for name, tensor in values.items():
            live = expected[name]
            _require(isinstance(tensor, torch.Tensor), f"invalid capture tensor: {name}")
            _require(
                tensor.device.type == "cpu" and tensor.dtype == live.dtype and tensor.shape == live.shape,
                f"capture tensor layout mismatch: {name}",
            )
            hashes[name] = _hash_tensor(tensor)
        result.append(hashes)
    return result[0], result[1]


def _package_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        _require(stream.read(8) == b"ADP1\0\0\0\0", "invalid ADP1 magic")
        length = stream.read(8)
        _require(len(length) == 8, "truncated ADP1 header")
        size = struct.unpack("<Q", length)[0]
        _require(0 < size < _GROUP_BYTES, "invalid ADP1 metadata length")
        header = stream.read(size)
        _require(len(header) == size, "truncated ADP1 metadata")
    return json.loads(header)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _package_integrity(root: Path, manifest: dict[str, Any]) -> dict[Path, str]:
    """Validate a complete physical-file contract before allocating any
    plans."""
    references = {name for edge in manifest["edges"] for name in edge["packages"]}
    integrity = manifest.get("package_integrity")
    if "package_integrity" not in manifest:
        progress_path = root / "build_progress.json"
        if not progress_path.exists():
            return {}
        progress = json.loads(progress_path.read_text())
        _require(
            progress.get("status") == "complete"
            and type(progress.get("completed_edges")) is int
            and progress["completed_edges"] == len(manifest["edges"]),
            "package integrity migration requires a complete build",
        )
        recorded = progress["packages"]
        _require(
            isinstance(recorded, dict) and set(recorded) == references, "build package integrity coverage mismatch"
        )
        integrity = {
            name: {"sha256": record["file_sha256"], "bytes": record["file_bytes"]} for name, record in recorded.items()
        }
    _require(isinstance(integrity, dict) and set(integrity) == references, "package integrity coverage mismatch")
    result = {}
    for relative, record in integrity.items():
        path = _relative_file(root, relative)
        _require(isinstance(record, dict) and _sha256(record.get("sha256")), "invalid package integrity SHA256")
        _require(type(record.get("bytes")) is int and record["bytes"] >= 16, "invalid package integrity byte size")
        _require(path.stat().st_size == record["bytes"], "package integrity byte size mismatch")
        _require(path not in result or result[path] == record["sha256"], "conflicting package path integrity")
        result[path] = record["sha256"]
    return result


def _source_identity(origin: Any, expected: str) -> tuple[str, str]:
    _require(isinstance(origin, str) and bool(origin), "algorithm module has no source origin")
    path = Path(origin).resolve()
    _require(path.suffix == ".py" and path.is_file(), "algorithm module must have a Python source file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(digest == expected, f"algorithm source SHA256 mismatch: {path.name}")
    return str(path), digest


def _attest_algorithm_sources(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    if "algorithm_sources" not in manifest:
        return {}
    expected = manifest["algorithm_sources"]
    _require(
        isinstance(expected, dict) and set(expected) == set(_ALGORITHM_MODULES),
        "algorithm_sources must cover exactly the three ADP modules",
    )
    resolved = {}
    # Check the entire set before executing an import. Unknown preloaded bytecode
    # cannot be authenticated by hashing a potentially newer file on disk.
    for name in _ALGORITHM_MODULES:
        qualified = f"{_ALGORITHM_PACKAGE}.{name}"
        entry = expected[name]
        _require(isinstance(entry, dict) and _sha256(entry.get("sha256")), "invalid algorithm source SHA256")
        module = sys.modules.get(qualified)
        if module is not None:
            known = _ATTESTED_ALGORITHM_MODULES.get(name)
            _require(known is not None and known[0] is module, f"unattested preloaded algorithm module: {name}")
            identity = _source_identity(getattr(module, "__file__", None), entry["sha256"])
            _require(identity == known[1:], f"attested algorithm module origin changed: {name}")
        else:
            spec = importlib.util.find_spec(qualified)
            _require(spec is not None, f"algorithm module is unavailable: {name}")
            identity = _source_identity(spec.origin, entry["sha256"])
        resolved[name] = identity
    for name in _ALGORITHM_MODULES:
        module = importlib.import_module(f"{_ALGORITHM_PACKAGE}.{name}")
        identity = _source_identity(getattr(module, "__file__", None), expected[name]["sha256"])
        _require(identity == resolved[name], f"algorithm module origin changed during import: {name}")
        _ATTESTED_ALGORITHM_MODULES[name] = (module, *identity)
    adaptive, batched, codec = (
        sys.modules[f"{_ALGORITHM_PACKAGE}.{name}"] for name in ("adaptive_delta", "batched_delta", "xor_codec")
    )
    _require(
        adaptive.AdaptiveDeltaPlan.__module__ == f"{_ALGORITHM_PACKAGE}.adaptive_delta"
        and batched.BatchedDeltaPlan.__module__ == f"{_ALGORITHM_PACKAGE}.batched_delta"
        and codec.GPUCodec.__module__ == f"{_ALGORITHM_PACKAGE}.xor_codec"
        and adaptive.GPUCodec is codec.GPUCodec,
        "algorithm classes do not originate from the attested modules",
    )
    return {name: {"origin": origin, "sha256": digest} for name, (origin, digest) in resolved.items()}


def _load_plan(path: Path, destinations: dict[str, Any], *, expected_sha256: str | None = None) -> Any:
    from relax.backends.sglang.adp.adaptive_delta import AdaptiveDeltaPlan
    from relax.backends.sglang.adp.batched_delta import BatchedDeltaPlan

    integrity = {"expected_sha256": expected_sha256} if expected_sha256 is not None else {}
    plan = AdaptiveDeltaPlan.load(path, destinations, a_bank=None, verify_base=False, **integrity)
    return BatchedDeltaPlan(plan, group_bytes=_GROUP_BYTES, compact_sources=True)


def _require_pinned(tensor: Any) -> None:
    _require(tensor.device.type == "cpu" and (not tensor.numel() or tensor.is_pinned()), "CPU bank is not pinned")


def _package_bank_bytes(plans: dict[tuple[str, str], Any]) -> int:
    """Count actual retained pinned allocations, including packet alignment
    padding."""
    allocations = {}
    for plan in plans.values():
        bank = plan.bank
        _require_pinned(bank)
        storage = bank.untyped_storage()
        allocations[storage.data_ptr()] = storage.nbytes()
    return sum(allocations.values())


def _synchronize(view: _RuntimeView) -> None:
    import torch

    devices = {tensor.device for tensor in (*view.params.values(), *view.auxiliary.values()) if tensor.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)


def _allocate_pinned(tensor: Any) -> Any:
    import torch

    return torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu", pin_memory=True)


def _prepare_recovery(view: _RuntimeView, candidate: _State, teacher: str) -> None:
    """Retain one anchor only, with one phase sync for all asynchronous D2H
    copies."""
    import torch

    snapshots: dict[int, Any] = {}
    banks, sizes = [], []
    try:
        with torch.inference_mode():
            for values in (view.params, view.auxiliary):
                bank, count = {}, 0
                for name, tensor in values.items():
                    # Exact-object auxiliary aliases need only one retained allocation.
                    if id(tensor) not in snapshots:
                        snapshot = _allocate_pinned(tensor)
                        _require_pinned(snapshot)
                        snapshots[id(tensor)] = snapshot
                        snapshot.copy_(tensor.detach(), non_blocking=True)
                        count += snapshot.untyped_storage().nbytes()
                    bank[name] = snapshots[id(tensor)]
                banks.append(bank)
                sizes.append(count)
    finally:
        # Drain copies even on allocation/copy failure before releasing CPU banks.
        _synchronize(view)
    candidate.recovery_params, candidate.recovery_auxiliary = banks
    candidate.recovery_parameter_bytes, candidate.recovery_auxiliary_bytes = sizes
    candidate.recovery_cpu_bytes = sum(sizes)
    candidate.base_teacher = teacher


def _validate_recovery(view: _RuntimeView, state: _State, teacher: str) -> None:
    _require(state.prepared and state.base_teacher is not None, "prepare_colocated is required before restore")
    _require(teacher in state.teachers, "unknown target teacher")
    _require(
        teacher == state.base_teacher or (state.base_teacher, teacher) in state.plans,
        "directed base teacher package is unavailable",
    )
    for live, bank in ((view.params, state.recovery_params), (view.auxiliary, state.recovery_auxiliary)):
        _require(set(live) == set(bank), "recovery snapshot coverage changed")
        for name, tensor in live.items():
            snapshot = bank[name]
            _require_pinned(snapshot)
            _require(
                snapshot.shape == tensor.shape and snapshot.dtype == tensor.dtype,
                f"recovery snapshot layout changed: {name}",
            )


def _restore(view: _RuntimeView, state: _State, teacher: str) -> None:
    import torch

    try:
        with torch.inference_mode():
            for live, bank in ((view.params, state.recovery_params), (view.auxiliary, state.recovery_auxiliary)):
                for name, tensor in live.items():
                    tensor.copy_(bank[name], non_blocking=True)
            if teacher != state.base_teacher:
                plan = state.plans[state.base_teacher, teacher]
                plan.mark_base_restored(verify=False)
                # The batched executor waits on the current stream before H2D/scatter.
                plan.apply(overlap=True)
    finally:
        # Includes the base==target path and failure after a partial asynchronous copy.
        _synchronize(view)


def _verify(view: _RuntimeView, state: _State, teacher: str) -> None:
    """Offline validation only; compare original bytes, including NaN
    payloads."""
    _require(
        {name: _hash_tensor(tensor) for name, tensor in view.params.items()} == state.teacher_hashes[teacher],
        "runtime parameter bytes do not match the tracked teacher",
    )
    _require(
        {name: _hash_tensor(tensor) for name, tensor in view.auxiliary.items()} == state.auxiliary_hashes,
        "runtime auxiliary bytes do not match preparation",
    )


def _event(
    operation: str | None,
    teacher: str | None,
    source: str | None,
    state: _State,
    rank: int,
    started: float,
    phase_seconds: float | None,
    error: str | None,
) -> None:
    destination = os.environ.get("RELAX_ADP_EVENT_LOG")
    if not destination or operation == "capture":
        return
    record = {
        "operation": operation,
        "source": source,
        "target": teacher,
        "base_teacher": state.base_teacher,
        "tp_rank": rank,
        "pid": os.getpid(),
        "timestamp": time.time(),
        "status": "failed" if error else "local_complete",
        "error": error,
        "total_seconds": time.perf_counter() - started,
        "prepare_seconds": phase_seconds if operation in ("prepare", "prepare_colocated") else None,
        "synchronized_seconds": phase_seconds if operation in ("switch", "restore") else None,
        "verify_seconds": phase_seconds if operation == "verify" else None,
        "recovery_parameter_bytes": state.recovery_parameter_bytes,
        "recovery_auxiliary_bytes": state.recovery_auxiliary_bytes,
        "recovery_cpu_bytes": state.recovery_cpu_bytes,
        "package_bank_bytes": state.package_bank_bytes,
        "logical_edge_count": len(state.plans),
        "physical_package_count": len({id(plan) for plan in state.plans.values()}),
        "algorithm_sources": state.algorithm_sources,
    }
    data = (json.dumps(record, sort_keys=True) + "\n").encode()
    # A single append avoids interleaving rank records; fail closed on short writes.
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        _require(os.write(descriptor, data) == len(data), "short ADP event write")
    finally:
        os.close(descriptor)


def _prepare(view: _RuntimeView, root: Path, teacher: str, rank: int, size: int) -> _State:
    manifest = json.loads((root / "manifest.json").read_text())
    _require(manifest["version"] == 1 and manifest["runtime"] == _RUNTIME, "unsupported manifest runtime/version")
    _require(type(manifest["tp_size"]) is int and manifest["tp_size"] == size, "manifest TP size mismatch")
    _require(manifest["initial_teacher"] == teacher, "prepare must name the manifest initial_teacher")
    _require(manifest.get("group_bytes", _GROUP_BYTES) == _GROUP_BYTES, "ADP requires 64 MiB bounded groups")
    teachers = manifest["teachers"]
    _require(isinstance(teachers, dict) and teacher in teachers, "invalid teacher manifest")
    _require(all(_TEACHER_ID.fullmatch(name) for name in teachers), "invalid teacher ID in manifest")
    _require(len(teachers) >= 2, "at least two teachers are required")
    package_integrity = _package_integrity(root, manifest)
    hashes: dict[str, dict[str, str]] = {}
    initial_aux = None
    for name, info in teachers.items():
        captures = info["captures"]
        _require(isinstance(captures, list) and len(captures) == size, "capture paths must cover every TP rank")
        hashes[name], aux = _snapshot_hashes(_relative_file(root, captures[rank]), view, name, rank, size)
        if initial_aux is None:
            initial_aux = aux
        _require(aux == initial_aux, "auxiliary tensor values differ across teachers; derived state is unsupported")
    _require(
        {name: _hash_tensor(tensor) for name, tensor in view.params.items()} == hashes[teacher],
        "initial teacher bytes mismatch",
    )
    _require(
        {name: _hash_tensor(tensor) for name, tensor in view.auxiliary.items()} == initial_aux,
        "resident auxiliary tensor bytes mismatch",
    )
    expected_edges = {(source, target) for source in teachers for target in teachers if source != target}
    edges = manifest["edges"]
    _require(isinstance(edges, list), "manifest edges must be a list")
    identities = [(edge["source"], edge["target"]) for edge in edges]
    _require(
        len(identities) == len(set(identities)) and set(identities) == expected_edges,
        "all directed teacher pairs are required exactly once",
    )
    paths, headers, reverse_paths = {}, {}, set()
    for edge in edges:
        source, target = edge["source"], edge["target"]
        reverse = edge.get("reverse", False)
        _require(type(reverse) is bool, "edge reverse must be a boolean")
        package_source, package_target = (target, source) if reverse else (source, target)
        packages = edge["packages"]
        _require(isinstance(packages, list) and len(packages) == size, "package paths must cover every TP rank")
        path = _relative_file(root, packages[rank])
        if path not in headers:
            headers[path] = _package_header(path)
        header = headers[path]
        _require(
            header["format"] == "ADP1" and header["version"] == 1 and header["policy"] == "size",
            "requires ADP1 size policy",
        )
        provenance = header["provenance"]
        for key, value in {
            "runtime": _RUNTIME,
            "source": package_source,
            "target": package_target,
            "tp_rank": rank,
            "tp_size": size,
        }.items():
            _require(provenance.get(key) == value, f"package provenance mismatch: {key}")
        _require(
            set(header["tensors"]) == set(view.params), "package must cover all parameters, including SKIP tensors"
        )
        for name, tensor in view.params.items():
            entry = header["tensors"][name]
            _require(
                entry["shape"] == list(tensor.shape) and entry["dtype"] == str(tensor.dtype),
                f"package layout mismatch: {name}",
            )
            _require(
                entry["a_sha256"] == hashes[package_source][name]
                and entry["b_sha256"] == hashes[package_target][name],
                f"package teacher hash mismatch: {name}",
            )
        if reverse:
            packets = header.get("packets")
            _require(
                isinstance(packets, list)
                and all(isinstance(packet, dict) and packet.get("mode") in _REVERSIBLE_MODES for packet in packets),
                "reverse package requires only XOR/PLANES packets; RAW/ADD or unknown modes are not reversible",
            )
            reverse_paths.add(path)
        paths[source, target] = path
    # No full CPU capture tensors are retained by _snapshot_hashes or by load().
    # Both directions retain the exact same wrapper, pinned bank and GPU scratch.
    # Load only after every logical edge passed provenance and hash validation.
    algorithm_sources = _attest_algorithm_sources(manifest)
    physical_plans = {}
    for path in dict.fromkeys(paths.values()):
        integrity = {"expected_sha256": package_integrity[path]} if package_integrity else {}
        plan = _load_plan(path, view.params, **integrity)
        if path in reverse_paths:
            _require(
                all(packet["mode"] in _REVERSIBLE_MODES for packet in plan.plan.packets),
                "loaded reverse package contains nonreversible packets",
            )
        physical_plans[path] = plan
    plans = {edge: physical_plans[path] for edge, path in paths.items()}
    return _State(
        current=teacher,
        prepared=True,
        manifest_digest=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        bindings=view.bindings,
        metadata=view.metadata,
        plans=plans,
        teachers=frozenset(teachers),
        config_json=view.config_json,
        package_bank_bytes=_package_bank_bytes(plans),
        teacher_hashes=hashes,
        auxiliary_hashes=initial_aux,
        algorithm_sources=algorithm_sources,
    )


def adp_teacher_loader(model: Any, named_tensors: Any) -> None:
    """SGLang custom_weight_loader callback; every TP rank must enter together.

    Exceptions are agreed on the existing TP CPU group before being raised. Any
    switch/restore/verify failure poisons every reachable worker:
    restart/reload the engine to recover. A later prepare cannot bless
    potentially half-switched weights.
    """
    started = time.perf_counter()
    context = _tp_context()
    _, _, rank, size = context
    state = getattr(model, _STATE_ATTRIBUTE, None)
    if state is None:
        state = _State()
        setattr(model, _STATE_ATTRIBUTE, state)
    source = state.current
    operation = teacher = None
    error = None
    view = None
    try:
        operation, teacher = _command(named_tensors)
        _require(not state.poisoned, "ADP worker state is unknown; engine restart/reload is required")
        view = _runtime_view(model, prepared=state if state.prepared else None)
        if state.prepared:
            _require(
                view.metadata == state.metadata and view.bindings == state.bindings,
                "runtime parameter/auxiliary layout or storage changed",
            )
        if operation in ("prepare", "prepare_colocated"):
            _require(not state.prepared, "ADP is already prepared")
        elif operation == "switch":
            _require(state.prepared and state.current is not None, "ADP prepare is required before switch")
            _require(teacher in state.teachers, "unknown target teacher")
            _require(
                teacher == state.current or (state.current, teacher) in state.plans,
                "directed teacher package is unavailable",
            )
        elif operation == "restore":
            _validate_recovery(view, state, teacher)
        elif operation == "verify":
            _require(state.prepared and teacher == state.current, "verify must name the prepared tracked teacher")
        elif operation == "capture":
            _require(not state.prepared, "prepared capture is forbidden; use verify to preserve source captures")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    route = (
        "preflight",
        operation,
        teacher,
        state.current,
        state.prepared,
        state.poisoned,
        state.base_teacher,
        bool(os.environ.get("RELAX_ADP_EVENT_LOG")),
    )
    try:
        _consensus(context, state, route, error, state.manifest_digest)
    except Exception as exc:
        try:
            _event(operation, teacher, source, state, rank, started, None, str(exc))
        except Exception:
            pass  # Preserve the original failure; logging cannot reopen a poisoned worker.
        raise

    candidate = None
    error = None
    phase_seconds = None
    phase_started = None
    try:
        if operation in ("switch", "restore"):
            _synchronize(view)
        phase_started = time.perf_counter()
        if operation == "capture":
            _capture(view, _root_directory(), teacher, rank, size)
        elif operation in ("prepare", "prepare_colocated"):
            candidate = _prepare(view, _root_directory(), teacher, rank, size)
            if operation == "prepare_colocated":
                _prepare_recovery(view, candidate, teacher)
        elif operation == "restore":
            _restore(view, state, teacher)
        elif operation == "verify":
            _verify(view, state, teacher)
        elif operation == "switch" and teacher != state.current:
            plan = state.plans[state.current, teacher]
            # The tracked teacher and full runtime bindings authorize this reset.
            # Never perform slow content verification on the switching path.
            plan.mark_base_restored(verify=False)
            try:
                plan.apply(overlap=True)
            finally:
                _synchronize(view)
        phase_seconds = time.perf_counter() - phase_started
        if operation != "capture":
            expected = candidate or state
            after = _runtime_view(model, prepared=expected)
            _require(
                after.metadata == expected.metadata and after.bindings == expected.bindings,
                "ADP operation changed runtime layout or storage",
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if phase_started is not None:
            phase_seconds = time.perf_counter() - phase_started
    route = ("commit", operation, teacher, state.current, state.prepared, state.poisoned, state.base_teacher)
    try:
        _consensus(context, state, route, error, candidate.manifest_digest if candidate else state.manifest_digest)
        if os.environ.get("RELAX_ADP_EVENT_LOG") and operation != "capture":
            event_error = None
            try:
                _event(operation, teacher, source, candidate or state, rank, started, phase_seconds, None)
            except Exception as exc:
                event_error = f"ADP event write failed: {type(exc).__name__}: {exc}"
                _poison(state)
            # Logging is optional, but an enabled failed write cannot be acknowledged.
            # Every rank participates even if only a peer's event destination failed.
            try:
                _consensus(context, state, ("event", operation, teacher), event_error)
            except Exception:
                _poison(state)
                raise
    except Exception as exc:
        try:
            _event(operation, teacher, source, candidate or state, rank, started, phase_seconds, str(exc))
        except Exception:
            pass
        raise
    if candidate is not None:
        setattr(model, _STATE_ATTRIBUTE, candidate)
    elif operation in ("switch", "restore"):
        state.current = teacher
