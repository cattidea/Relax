# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare local Qwen3.5-9B inputs for the five-Teacher full-reload baseline.

No downloads, GPU work, weight conversion or delta construction occur here.
Only checkpoint metadata, tokenizer assets and safetensors headers are read.
The default 20 authored math prompts exercise routing and training, not model
quality. Supply --prompt-data for an existing T1..T5-routed JSONL or Parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from relax.utils.opd.disk_teacher_backend import _checkpoint_fingerprint


TEACHER_IDS = tuple(f"T{index}" for index in range(1, 6))
DTYPE_BYTES = {"BF16": 2, "F32": 4}


def checkpoint_layout(directory: Path) -> dict[str, dict[str, Any]]:
    """Validate headers/indexes without intentionally reading tensor
    payloads."""
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else None
    names = sorted(set(index["weight_map"].values())) if index else [p.name for p in directory.glob("*.safetensors")]
    if not names or (index is None and len(names) != 1):
        raise ValueError(f"Expected an indexed checkpoint or one safetensors file: {directory}")
    layout, actual_map = {}, {}
    total_bytes = 0
    for filename in names:
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid shard filename: {filename!r}")
        path = directory / filename
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors header: {path}")
            size = struct.unpack("<Q", prefix)[0]
            if not 0 < size <= 16 * 1024**2:
                raise ValueError(f"Invalid safetensors header size: {path}")
            header = json.loads(stream.read(size))
        ranges = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in layout:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            shape, dtype = entry["shape"], entry["dtype"]
            if (
                dtype not in DTYPE_BYTES
                or not isinstance(shape, list)
                or any(type(d) is not int or d < 0 for d in shape)
            ):
                raise ValueError(f"Expected BF16/F32 dense tensors, got {name}: {dtype}/{shape}")
            start, end = entry["data_offsets"]
            if type(start) is not int or type(end) is not int or start < 0 or end < start:
                raise ValueError(f"Invalid tensor offsets: {name}")
            if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
                raise ValueError(f"Tensor shape and byte length disagree: {name}")
            ranges.append((start, end))
            layout[name] = {"shape": shape, "dtype": dtype}
            actual_map[name] = filename
            total_bytes += end - start
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError(f"Overlapping or missing tensor payload range: {path}")
            cursor = end
        if cursor != path.stat().st_size - size - 8:
            raise ValueError(f"Checkpoint payload is truncated or has trailing bytes: {path}")
    if index is not None:
        if index.get("weight_map") != actual_map:
            raise ValueError(f"Index disagrees with actual tensor names/shards: {directory}")
        if index.get("metadata", {}).get("total_size", total_bytes) != total_bytes:
            raise ValueError(f"Index total_size disagrees with tensor headers: {directory}")
    return layout


def require_qwen35_9b(config: dict, directory: Path) -> None:
    text = config.get("text_config", {})
    expected = {"hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 32, "vocab_size": 248320}
    if config.get("model_type") != "qwen3_5" or not config.get("vision_config"):
        raise ValueError(f"Expected native Qwen3.5-9B VL architecture: {directory}")
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint is not the requested Qwen3.5-9B architecture: {directory}")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError(f"Quantized checkpoints are outside this BF16 baseline: {directory}")


def validate_models(student: Path, teachers: dict[str, str]) -> dict:
    if set(teachers) != set(TEACHER_IDS):
        raise ValueError("Provide exactly --teacher T1=PATH through --teacher T5=PATH")
    paths = {name: Path(path).resolve() for name, path in teachers.items()}
    if len(set(paths.values())) != 5:
        raise ValueError("The five Teachers must use five distinct checkpoint directories")
    reference, reference_layout, raw_teacher_config = None, None, None
    rows = []
    for name, directory in [("student", student.resolve()), *paths.items()]:
        fingerprint = _checkpoint_fingerprint(directory)
        require_qwen35_9b(fingerprint["config"], directory)
        layout = checkpoint_layout(directory)
        if reference is None:
            reference, reference_layout = fingerprint, layout
        if fingerprint["tokenizer"] != reference["tokenizer"]:
            raise ValueError(f"Tokenizer assets differ for {name}; no automatic rewriting is performed")
        if fingerprint.get("processor", {}) != reference.get("processor", {}):
            raise ValueError(f"Image/video processor assets differ for {name}")
        if layout != reference_layout:
            raise ValueError(f"Tensor names/shapes/dtypes differ for {name}")
        if fingerprint["config"] != reference["config"]:
            raise ValueError(f"Effective config differs for {name}; select compatible checkpoints first")
        raw_config = (directory / "config.json").read_bytes()
        if name != "student":
            if raw_teacher_config is None:
                raw_teacher_config = raw_config
            elif raw_config != raw_teacher_config:
                raise ValueError(f"Original Teacher config.json is not byte-identical for {name}")
        rows.append(
            {
                "name": name,
                "path": str(directory),
                "config_sha256": hashlib.sha256(raw_config).hexdigest(),
                "tensor_count": len(layout),
                "weight_bytes": sum(
                    math.prod(entry["shape"]) * DTYPE_BYTES[entry["dtype"]] for entry in layout.values()
                ),
            }
        )
    return {
        "models": rows,
        "tokenizer_asset_sha256": reference["tokenizer"],
        "processor_asset_sha256": reference.get("processor", {}),
        "weight_payload_checksums_verified": False,
    }


def make_rows(rows_per_teacher: int) -> list[dict]:
    if rows_per_teacher <= 0:
        raise ValueError("--rows-per-teacher must be positive")
    return [
        {
            "prompt": f"A shop has {index + 3} boxes with 7 pencils each. It sells 7 pencils. How many remain? End with 'Answer: <number>'.",
            "label": str((index + 2) * 7),
            "data_source": teacher,
            "extra_info": {"rm_type": "mopd", "data_source": teacher, "engineering_pilot": True},
        }
        for index in range(rows_per_teacher)
        for teacher in TEACHER_IDS
    ]


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.suffix != ".jsonl":
        raise ValueError("--prompt-data must be a JSONL or Parquet file")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_rows(rows: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for index, row in enumerate(rows):
        metadata = row.get("extra_info")
        teacher = metadata.get("data_source") if isinstance(metadata, dict) else None
        if teacher not in TEACHER_IDS or metadata.get("rm_type") != "mopd":
            raise ValueError(f"Row {index} requires extra_info.rm_type=mopd and extra_info.data_source=T1..T5")
        if not row.get("prompt") or "label" not in row:
            raise ValueError(f"Row {index} requires a nonempty prompt and a label")
        if row.get("data_source", teacher) != teacher:
            raise ValueError(f"Row {index} has conflicting data_source values")
        counts[teacher] += 1
    if set(counts) != set(TEACHER_IDS):
        raise ValueError("Dataset must contain at least one routed sample for every Teacher T1..T5")
    return dict(sorted(counts.items()))


def prepare(args: argparse.Namespace) -> dict:
    pin_memory = bool(getattr(args, "pin_memory", False))
    if pin_memory and args.weight_source != "memory":
        raise ValueError("--pin-memory applies only to --weight-source memory")
    teachers = {}
    for item in args.teacher:
        name, separator, path = item.partition("=")
        if not separator or not path or name in teachers:
            raise ValueError("Each --teacher must be a unique T1=PATH through T5=PATH")
        teachers[name] = str(Path(path).resolve())
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("--output-dir must be fresh; existing run files are preserved")
    report = validate_models(args.student, teachers)
    rows = read_rows(args.prompt_data) if args.prompt_data else make_rows(args.rows_per_teacher)
    report.update(
        source_counts=validate_rows(rows),
        purpose="five-Teacher MOPD engineering baseline, not a model quality benchmark",
        weight_source=args.weight_source,
        teacher_cache_pin_memory=pin_memory if args.weight_source == "memory" else None,
        gpu_layout={
            "actor": 8,
            "actor_tp": 4,
            "actor_dp": 2,
            "rollout": 7,
            "rollout_tp": 1,
            "teacher": 1,
            "teacher_tp": 1,
        },
    )
    manifest = {
        "version": 1,
        "backend": f"sglang-{args.weight_source}-v1",
        "tp_size": 1,
        "initial_teacher": "T1",
        "teachers": {name: teachers[name] for name in TEACHER_IDS},
        "event_log": str(output / "full_reload_events.jsonl"),
    }
    if args.weight_source == "memory":
        manifest["pin_memory"] = pin_memory
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "teacher_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", action="append", required=True, metavar="T1=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-source", choices=("disk", "memory"), default="disk")
    parser.add_argument("--pin-memory", action="store_true", help="Pin the full CPU Teacher cache (memory mode only)")
    parser.add_argument("--prompt-data", type=Path)
    parser.add_argument("--rows-per-teacher", type=int, default=4)
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Baseline preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
