# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build directed ADP packages from captures of the actual SGLang TP layout.

Stop the capture engine first: the builder temporarily holds one runtime shard
on a GPU. This is offline preparation, never the online teacher switching path.
ADP1 modules must be supplied on PYTHONPATH; this script installs nothing. Use
--max-new-edges 1 for the first real pair, then --resume for remaining edges.
Incomplete builds never publish manifest.json. Resume rehashes captures and
packages, and rejects unrecorded pre-existing package files rather than
trusting them. A crash between package publication and journal commit can leave
such an orphan; inspect/remove that one file before resuming, not the completed
journal. With --reuse-symmetric-pairs, logical reverse edges reuse a physical
package only when every TP rank uses exclusively XOR/PLANES packet modes. RAW
and ADD_ANS still require separate directed packages. The published graph
always contains all logical directions, with reverse=true on reused entries.
Existing partial directed builds can opt in with --resume without replacing any
capture/package.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable


_RUNTIME = "sglang-dense-v1"
_BLOCK_BYTES = 16 * 1024**2
_READ_BYTES = 8 * 1024**2
_MODES = {"RAW", "PLANES", "PLANES_ANS", "XOR_ANS", "XOR_GROUP2_ANS", "ADD_ANS"}
_SYMMETRIC_MODES = {"PLANES", "PLANES_ANS", "XOR_ANS", "XOR_GROUP2_ANS"}
_DIRECTED_STRATEGY = "directed-v1"
_SYMMETRIC_STRATEGY = "symmetric-pairs-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_READ_BYTES):
            digest.update(chunk)
    after = path.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"File changed while hashing: {path}",
    )
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _tensor_info(tensor: Any) -> dict[str, Any]:
    import torch

    _require(tensor.device.type == "cpu", "Capture tensors must reside on CPU")
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, raw.numel(), _READ_BYTES):
        digest.update(memoryview(raw[offset : offset + _READ_BYTES].numpy()))
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "bytes": raw.numel(),
        "sha256": digest.hexdigest(),
    }


def _capture_info(path: Path, teacher: str, rank: int, tp_size: int) -> dict[str, Any]:
    import torch

    signature = _file_signature(path)
    fingerprint = _file_fingerprint(path)
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    provenance = {key: snapshot[key] for key in ("version", "runtime", "teacher_id", "tp_rank", "tp_size")}
    _require(
        provenance == dict(version=1, runtime=_RUNTIME, teacher_id=teacher, tp_rank=rank, tp_size=tp_size),
        "Capture provenance differs from the requested runtime layout",
    )
    result = {
        "file": fingerprint,
        "provenance": provenance,
        "metadata": snapshot["metadata"],
        "params": {name: _tensor_info(tensor) for name, tensor in snapshot["params"].items()},
        "auxiliary": {name: _tensor_info(tensor) for name, tensor in snapshot.get("auxiliary", {}).items()},
    }
    _require(_file_signature(path) == signature, "Capture changed while validating its tensors")
    return result


def _pair_compatible(a: dict, b: dict) -> None:
    _require(
        a["metadata"] == b["metadata"] and a["params"].keys() == b["params"].keys(), "Teachers have different layouts"
    )
    for name, info in a["params"].items():
        _require(
            all(info[key] == b["params"][name][key] for key in ("shape", "dtype", "bytes")),
            f"Parameter layout differs: {name}",
        )
    _require(a["auxiliary"] == b["auxiliary"], "Changing derived tensors/buffers are unsupported by the dense pilot")


def _package_info(path: Path, provenance: dict, a: dict, b: dict) -> dict[str, Any]:
    """Validate header identity and stream both packet and complete-file
    hashes."""
    signature = _file_signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        prefix = stream.read(16)
        _require(len(prefix) == 16 and prefix[:8] == b"ADP1\0\0\0\0", "Invalid ADP1 prefix")
        size = struct.unpack("<Q", prefix[8:])[0]
        _require(0 < size < 64 * 1024**2, "Invalid ADP1 metadata size")
        raw_header = stream.read(size)
        _require(len(raw_header) == size, "Truncated ADP1 metadata")
        digest.update(prefix + raw_header)
        header = json.loads(raw_header)
        _require(
            (header["format"], header["version"], header["policy"], header["block_bytes"])
            == ("ADP1", 1, "size", _BLOCK_BYTES),
            "ADP1 encoder settings differ",
        )
        _require(header["provenance"] == provenance, "ADP1 capture provenance differs")
        _require(set(header["tensors"]) == set(a["params"]), "ADP1 parameter coverage differs")
        for name, info in header["tensors"].items():
            expected_a, expected_b = a["params"][name], b["params"][name]
            _require(
                all(info[key] == expected_a[key] for key in ("shape", "dtype", "bytes"))
                and info["a_sha256"] == expected_a["sha256"]
                and info["b_sha256"] == expected_b["sha256"],
                f"ADP1 parameter hash/layout differs: {name}",
            )
        payload_bytes, restored_bytes = 0, 0
        mode_bytes, mode_target_bytes, mode_packets, intervals = {}, {}, {}, {}
        for packet in header["packets"]:
            mode, name = packet["mode"], packet["name"]
            count, offset, n = packet["payload_bytes"], packet["offset"], packet["n"]
            _require(mode in _MODES and name in a["params"], "Invalid packet identity")
            _require(
                all(type(value) is int for value in (count, offset, n, packet["payload_offset"]))
                and 0 <= count <= _BLOCK_BYTES + 8
                and 0 < n <= _BLOCK_BYTES
                and 0 <= offset <= a["params"][name]["bytes"] - n
                and packet["payload_offset"] == payload_bytes,
                "Invalid packet boundaries",
            )
            _require(mode != "RAW" or count == n, "Invalid RAW packet length")
            intervals.setdefault(name, []).append((offset, offset + n))
            packet_hash = hashlib.sha256()
            remaining = count
            while remaining:
                chunk = stream.read(min(remaining, _READ_BYTES))
                _require(bool(chunk), "Truncated ADP1 payload")
                digest.update(chunk)
                packet_hash.update(chunk)
                remaining -= len(chunk)
            _require(packet_hash.hexdigest() == packet["sha256"], "ADP1 packet checksum differs")
            payload_bytes += count
            restored_bytes += n
            mode_bytes[mode] = mode_bytes.get(mode, 0) + count
            mode_target_bytes[mode] = mode_target_bytes.get(mode, 0) + n
            mode_packets[mode] = mode_packets.get(mode, 0) + 1
        for spans in intervals.values():
            spans.sort()
            _require(all(first[1] <= second[0] for first, second in zip(spans, spans[1:])), "Overlapping ADP1 packets")
        _require(
            all(
                name in intervals or info["sha256"] == b["params"][name]["sha256"]
                for name, info in a["params"].items()
            ),
            "Changed parameter has no ADP1 packets",
        )
        _require(not stream.read(1), "Extra ADP1 payload bytes")
    _require(_file_signature(path) == signature, "Package changed while validating its bytes")
    raw_bytes = sum(info["bytes"] for info in a["params"].values())
    return {
        "file_bytes": 16 + size + payload_bytes,
        "file_sha256": digest.hexdigest(),
        "header_bytes": 16 + size,
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "payload_bytes": payload_bytes,
        "mode_payload_bytes": mode_bytes,
        "mode_target_bytes": mode_target_bytes,
        "mode_packet_counts": mode_packets,
        "runtime_parameter_bytes": raw_bytes,
        "skipped_bytes": raw_bytes - restored_bytes,
        "source_capture": a["file"],
        "target_capture": b["file"],
        "provenance": provenance,
        "encoder_stats": header.get("stats", {}),
    }


def _pin_parameters(parameters: dict) -> dict:
    pinned = {}
    for name in list(parameters):
        pinned[name] = parameters.pop(name).pin_memory()
    return pinned


def _progress_emitter(logger: Any, source: str, target: str, rank: int) -> Callable[..., None]:
    """Log CPU progress fields only; never synchronize or inspect live
    tensors."""
    last_reported = time.perf_counter()

    def emit(event: str, **fields: Any) -> None:
        nonlocal last_reported
        if event != "adaptive_prepare":
            return
        now = time.perf_counter()
        final = fields["tensors"] == fields["total_tensors"]
        if now - last_reported < 20 and not final:
            return
        logger.info(
            "Encoding %s -> %s rank=%s tensors=%s/%s payload=%s seconds=%.3f",
            source,
            target,
            rank,
            fields["tensors"],
            fields["total_tensors"],
            fields["payload_bytes"],
            fields["seconds"],
        )
        last_reported = now

    return emit


def _encode_package(
    directory: Path, source: str, target: str, rank: int, tp_size: int, device: str, path: Path
) -> dict:
    """The existing encoder and policy are unchanged; called only for missing
    edges."""
    import inspect

    import torch

    from relax.backends.sglang.adp.adaptive_delta import AdaptiveDeltaPlan
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    with torch.cuda.device(device):
        try:
            a = torch.load(directory / f"captures/{source}/rank_{rank}.pt", map_location="cpu", weights_only=True)
            b = torch.load(directory / f"captures/{target}/rank_{rank}.pt", map_location="cpu", weights_only=True)
            destinations = {name: tensor.to(device) for name, tensor in a["params"].items()}
            a_bank, b_bank = _pin_parameters(a["params"]), _pin_parameters(b["params"])
            del a, b
            pairs = [(name, a_bank[name], b_bank[name], destinations[name]) for name in a_bank]
            plan = AdaptiveDeltaPlan(
                pairs, policy="size", block_bytes=_BLOCK_BYTES, emit=_progress_emitter(logger, source, target, rank)
            )
            # Per-tensor events do not include counts; report the real final
            # encoder counts once its statistics have been finalized.
            logger.info("Encoded %s -> %s rank=%s counts=%s", source, target, rank, plan.stats["counts"])
            plan.save(
                path, provenance=dict(runtime=_RUNTIME, source=source, target=target, tp_rank=rank, tp_size=tp_size)
            )
            torch.cuda.synchronize()
            return {
                "torch_version": str(torch.__version__),
                "encoder_source_sha256": _file_fingerprint(Path(inspect.getfile(AdaptiveDeltaPlan)))["sha256"],
            }
        finally:
            # Frame-local captures/plans are released on return/exception; run a
            # final cache cleanup in the caller after this frame has gone away.
            torch.cuda.synchronize()


def _release_encoder_memory(device: str) -> None:
    import torch

    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def _manifest(teachers: list[str], tp_size: int) -> dict:
    _require(
        type(tp_size) is int and tp_size > 0 and len(teachers) >= 2 and len(set(teachers)) == len(teachers),
        "Specify a positive TP size and at least two distinct teachers",
    )
    for teacher in teachers:
        _require(
            bool(teacher) and all(c.isascii() and (c.isalnum() or c in "_-") for c in teacher),
            "Teacher IDs must contain only letters, digits, underscore, or hyphen",
        )
    return {
        "version": 1,
        "runtime": _RUNTIME,
        "tp_size": tp_size,
        "initial_teacher": teachers[0],
        "group_bytes": 64 * 1024**2,
        "teachers": {
            teacher: {"captures": [f"captures/{teacher}/rank_{rank}.pt" for rank in range(tp_size)]}
            for teacher in teachers
        },
        "edges": [
            {
                "source": source,
                "target": target,
                "packages": [f"edges/{source}/{target}/rank_{rank}.adp" for rank in range(tp_size)],
            }
            for source in teachers
            for target in teachers
            if source != target
        ],
    }


def _capture_key(teacher: str, rank: int) -> str:
    return f"captures/{teacher}/rank_{rank}.pt"


def _edge_key(edge: dict) -> str:
    return f"{edge['source']}->{edge['target']}"


def _complete_edges(manifest: dict, packages: dict, reused: dict | None = None) -> int:
    return sum(
        _edge_key(edge) in (reused or {}) or all(path in packages for path in edge["packages"])
        for edge in manifest["edges"]
    )


def _validate_reused_edges(manifest: dict, packages: dict, reused: dict) -> None:
    edges = {_edge_key(edge): edge for edge in manifest["edges"]}
    _require(isinstance(reused, dict) and set(reused) <= set(edges), "Progress contains unknown reverse edges")
    for key, entry in reused.items():
        edge = edges[key]
        inverse = edges[f"{edge['target']}->{edge['source']}"]
        _require(
            entry == {**edge, "packages": inverse["packages"], "reverse": True} and type(entry.get("reverse")) is bool,
            "Reverse edge identity or physical package paths differ",
        )
        _require(
            not any(path in packages for path in edge["packages"]), "Reverse reuse would discard existing packages"
        )
        _require(
            all(path in packages for path in inverse["packages"]), "Reverse edge has incomplete physical packages"
        )
        _require(
            all(set(packages[path]["mode_packet_counts"]) <= _SYMMETRIC_MODES for path in inverse["packages"]),
            "Reverse edge references asymmetric packet modes",
        )


def _register_symmetric_reuse(manifest: dict, packages: dict, reused: dict) -> int:
    """Only reuse complete physical edges; preserve existing/partly built
    reverse files."""
    edges = {_edge_key(edge): edge for edge in manifest["edges"]}
    added = 0
    for key, edge in edges.items():
        if key in reused or any(path in packages for path in edge["packages"]):
            continue
        inverse = edges[f"{edge['target']}->{edge['source']}"]
        if all(path in packages for path in inverse["packages"]) and all(
            set(packages[path]["mode_packet_counts"]) <= _SYMMETRIC_MODES for path in inverse["packages"]
        ):
            reused[key] = {**edge, "packages": list(inverse["packages"]), "reverse": True}
            added += 1
    return added


def _runtime_manifest(manifest: dict, reused: dict, packages: dict | None = None) -> dict:
    result = {**manifest, "edges": [reused.get(_edge_key(edge), edge) for edge in manifest["edges"]]}
    if packages is not None:
        source_root = Path(__file__).resolve().parents[3] / "relax/backends/sglang/adp"
        result["algorithm_sources"] = {
            name: {"sha256": hashlib.sha256((source_root / f"{name}.py").read_bytes()).hexdigest()}
            for name in ("xor_codec", "adaptive_delta", "batched_delta")
        }
        result["package_integrity"] = {
            name: {"sha256": record["file_sha256"], "bytes": record["file_bytes"]} for name, record in packages.items()
        }
    return result


def _build_locked(
    directory: Path, manifest: dict, device: str, max_new_edges: int | None, resume: bool, reuse_symmetric_pairs: bool
) -> dict:
    from relax.utils.logging_utils import get_logger

    logger = get_logger(__name__)
    progress_path, manifest_path = directory / "build_progress.json", directory / "manifest.json"
    requested_strategy = _SYMMETRIC_STRATEGY if reuse_symmetric_pairs else _DIRECTED_STRATEGY
    migration = False
    if resume:
        progress = json.loads(progress_path.read_text())
        _require(
            progress.get("version") == 1 and progress.get("manifest") == manifest,
            "Build progress configuration differs",
        )
        _require(
            progress.get("encoder") == {"policy": "size", "block_bytes": _BLOCK_BYTES}, "Build encoder settings differ"
        )
        previous_strategy = progress.get("build_strategy", _DIRECTED_STRATEGY)
        _require(previous_strategy in (_DIRECTED_STRATEGY, _SYMMETRIC_STRATEGY), "Unknown recorded build strategy")
        if previous_strategy != requested_strategy:
            _require(
                previous_strategy == _DIRECTED_STRATEGY
                and requested_strategy == _SYMMETRIC_STRATEGY
                and not manifest_path.exists(),
                "Build strategy differs; opt in only on an unpublished directed build, and retain the flag on resume",
            )
            migration = True
        _require(
            previous_strategy == _SYMMETRIC_STRATEGY or not progress.get("reused_edges"),
            "Directed progress unexpectedly contains reverse edges",
        )
    else:
        if progress_path.exists() or manifest_path.exists():
            raise FileExistsError("Build progress/manifest already exists; use --resume or a fresh teacher directory")
        progress = {
            "version": 1,
            "manifest": manifest,
            "encoder": {"policy": "size", "block_bytes": _BLOCK_BYTES},
            "captures": {},
            "packages": {},
            "status": "incomplete",
            "created_at": time.time(),
            "build_strategy": requested_strategy,
            "reused_edges": {},
        }
        _atomic_json(progress_path, progress)
    expected = {path for edge in manifest["edges"] for path in edge["packages"]}
    _require(set(progress["packages"]) <= expected, "Progress contains unknown packages")
    captures = progress["captures"]
    signatures = {}
    allowed_captures = {path for info in manifest["teachers"].values() for path in info["captures"]}
    _require(set(captures) <= allowed_captures, "Progress contains unknown captures")
    for key, info in captures.items():
        teacher, rank_text = key.split("/")[1:]
        rank = int(rank_text.removeprefix("rank_").removesuffix(".pt"))
        _require(
            info["provenance"]
            == dict(version=1, runtime=_RUNTIME, teacher_id=teacher, tp_rank=rank, tp_size=manifest["tp_size"]),
            "Recorded capture provenance differs",
        )
        _require(_file_fingerprint(directory / key) == info["file"], f"Capture changed since preparation: {key}")
        signatures[key] = _file_signature(directory / key)
    package_signatures = {}
    for edge in manifest["edges"]:
        for rank, path in enumerate(edge["packages"]):
            if path in progress["packages"]:
                a, b = captures[_capture_key(edge["source"], rank)], captures[_capture_key(edge["target"], rank)]
                _pair_compatible(a, b)
                provenance = dict(
                    runtime=_RUNTIME,
                    source=edge["source"],
                    target=edge["target"],
                    tp_rank=rank,
                    tp_size=manifest["tp_size"],
                )
                actual = _package_info(directory / path, provenance, a, b)
                recorded = progress["packages"][path]
                _require(
                    all(recorded.get(key) == value for key, value in actual.items()),
                    f"Completed package changed: {path}",
                )
                package_signatures[path] = _file_signature(directory / path)
            elif (directory / path).exists():
                raise FileExistsError(f"Unrecorded package exists; refusing reuse/overwrite: {path}")
    reused = progress.setdefault("reused_edges", {})
    _validate_reused_edges(manifest, progress["packages"], reused)
    if manifest_path.exists():
        _require(
            all(_file_signature(directory / path) == signature for path, signature in package_signatures.items()),
            "Physical package changed during resume validation",
        )
        _require(
            _complete_edges(manifest, progress["packages"], reused) == len(manifest["edges"])
            and json.loads(manifest_path.read_text())
            in (_runtime_manifest(manifest, reused, progress["packages"]), _runtime_manifest(manifest, reused)),
            "Existing manifest is not a complete validated build",
        )
        return progress
    new_reused = _register_symmetric_reuse(manifest, progress["packages"], reused) if reuse_symmetric_pairs else 0
    progress["build_strategy"] = requested_strategy
    if migration:
        progress.setdefault("strategy_migrations", []).append(
            {
                "from": _DIRECTED_STRATEGY,
                "to": requested_strategy,
                "timestamp": time.time(),
                "packages_preserved": len(progress["packages"]),
            }
        )
    if migration or new_reused:
        progress.update(
            completed_edges=_complete_edges(manifest, progress["packages"], reused), updated_at=time.time()
        )
        _atomic_json(progress_path, progress)
    new_edges = 0
    for edge in manifest["edges"]:
        if _edge_key(edge) in reused or all(path in progress["packages"] for path in edge["packages"]):
            continue
        if max_new_edges is not None and new_edges >= max_new_edges:
            break
        for rank, relative in enumerate(edge["packages"]):
            if relative in progress["packages"]:
                continue
            started = time.perf_counter()
            keys = [_capture_key(edge[which], rank) for which in ("source", "target")]
            for key, teacher in zip(keys, (edge["source"], edge["target"])):
                if key not in captures:
                    captures[key] = _capture_info(directory / key, teacher, rank, manifest["tp_size"])
                    signatures[key] = _file_signature(directory / key)
            a, b = (captures[key] for key in keys)
            _pair_compatible(a, b)
            capture_seconds = time.perf_counter() - started
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".adp.building")
            encode_started = time.perf_counter()
            _require(
                all(_file_signature(directory / key) == signatures[key] for key in keys),
                "Capture changed before encoding; captures must remain immutable",
            )
            try:
                encoder = _encode_package(
                    directory, edge["source"], edge["target"], rank, manifest["tp_size"], device, temporary
                )
            finally:
                _release_encoder_memory(device)
            encode_seconds = time.perf_counter() - encode_started
            _require(
                all(_file_signature(directory / key) == signatures[key] for key in keys),
                "Capture changed during encoding; captures must remain immutable",
            )
            provenance = dict(
                runtime=_RUNTIME,
                source=edge["source"],
                target=edge["target"],
                tp_rank=rank,
                tp_size=manifest["tp_size"],
            )
            info = _package_info(temporary, provenance, a, b)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(path)
            _sync_directory(path.parent)
            package_signatures[relative] = _file_signature(path)
            info.update(
                preparation_seconds=time.perf_counter() - started,
                capture_validation_seconds=capture_seconds,
                encode_seconds=encode_seconds,
                device=device,
                encoder=encoder,
                completed_at=time.time(),
            )
            progress["packages"][relative] = info
            if reuse_symmetric_pairs:
                new_reused += _register_symmetric_reuse(manifest, progress["packages"], reused)
            progress["completed_edges"] = _complete_edges(manifest, progress["packages"], reused)
            progress["updated_at"] = time.time()
            _atomic_json(progress_path, progress)
            logger.info(
                "Prepared %s -> %s rank=%s bytes=%s payload=%s seconds=%.3f",
                edge["source"],
                edge["target"],
                rank,
                info["file_bytes"],
                info["payload_bytes"],
                info["preparation_seconds"],
            )
        new_edges += 1
    _validate_reused_edges(manifest, progress["packages"], reused)
    _require(
        all(_file_signature(directory / path) == signature for path, signature in package_signatures.items()),
        "Physical package changed before manifest publication",
    )
    complete = _complete_edges(manifest, progress["packages"], reused) == len(manifest["edges"])
    progress.update(
        status="complete" if complete else "incomplete",
        completed_edges=_complete_edges(manifest, progress["packages"], reused),
        completed_physical_edges=_complete_edges(manifest, progress["packages"]),
        physical_package_files=len(progress["packages"]),
        updated_at=time.time(),
        last_run_new_edges=new_edges,
        last_run_reused_edges=new_reused,
    )
    _atomic_json(progress_path, progress)
    if complete:
        _atomic_json(manifest_path, _runtime_manifest(manifest, reused, progress["packages"]))
    return progress


def build_packages(
    directory: Path,
    teachers: list[str],
    tp_size: int,
    device: str,
    *,
    max_new_edges: int | None = None,
    resume: bool = False,
    reuse_symmetric_pairs: bool = False,
) -> dict:
    _require(
        max_new_edges is None or (type(max_new_edges) is int and max_new_edges > 0), "max_new_edges must be positive"
    )
    _require(type(reuse_symmetric_pairs) is bool, "reuse_symmetric_pairs must be a boolean")
    manifest = _manifest(teachers, tp_size)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".build_packages.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another package builder owns this directory") from exc
        try:
            return _build_locked(directory, manifest, device, max_new_edges, resume, reuse_symmetric_pairs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument(
        "--teachers",
        nargs="+",
        required=True,
        help="The first teacher must be the serving engine's startup checkpoint",
    )
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--device", required=True, help="Explicit offline builder GPU, e.g. cuda:0")
    parser.add_argument(
        "--max-new-edges",
        type=int,
        help="Stop after N newly encoded directed edges; free reverse aliases do not count; incomplete builds publish no manifest",
    )
    parser.add_argument("--resume", action="store_true", help="Rehash and reuse only journaled completed packages")
    parser.add_argument(
        "--reuse-symmetric-pairs",
        action="store_true",
        help="Reuse XOR/PLANES-only packages for reverse logical edges; RAW/ADD remain directed",
    )
    args = parser.parse_args()
    build_packages(
        args.teacher_dir,
        args.teachers,
        args.tp_size,
        args.device,
        max_new_edges=args.max_new_edges,
        resume=args.resume,
        reuse_symmetric_pairs=args.reuse_symmetric_pairs,
    )


if __name__ == "__main__":
    main()
