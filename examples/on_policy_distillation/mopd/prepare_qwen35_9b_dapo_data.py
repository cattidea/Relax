# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare local DAPO text data without changing prompts, labels, or source
files.

The output order requires the eager Dataset, no rollout shuffle, and no runtime
length filtering. StreamingDataset shuffles even without --rollout-shuffle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TEACHERS = ("T1", "T2", "T3", "T4", "T5")
BATCH_SIZE = 20
SOURCE = "zhuzilin/dapo-math-17k"
REWARD_SOURCE = "dapo-math-17k"
SOURCE_REVISION = "2e65612930298bde4c5d58fd97b3f23a483aaff9"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def prompt_hash(prompt: list[dict]) -> str:
    return hashlib.sha256(canonical(prompt)).hexdigest()


def read_source(path: Path) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank source row at line {line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Source row {line_number} must be an object")
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Source row {line_number} must retain a nonempty prompt message list")
        for message in prompt:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant", "system"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ValueError(f"Source row {line_number} must contain text-only chat messages")
        if any(row.get(key) for key in ("image", "images", "video", "videos", "audio", "audios", "tools")):
            raise ValueError(f"Source row {line_number} contains unsupported media or tools")
        if not isinstance(row.get("label"), str) or not row["label"].strip():
            raise ValueError(f"Source row {line_number} must have a nonempty string label")
        metadata = row.get("extra_info", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Source row {line_number} extra_info must be an object")
        reserved = {"rm_type", "data_source", "reward_source", "dapo_provenance"}
        if reserved & metadata.keys():
            raise ValueError(f"Source row {line_number} already contains reserved routing metadata")
        canonical(row)
        rows.append(row)
    return rows, {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def split_indices(rows: list[dict], *, seed: int, eval_ratio: float) -> tuple[list[int], list[int], dict]:
    if not math.isfinite(eval_ratio) or not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be strictly between 0 and 1")
    groups: dict[str, list[int]] = {}
    labels = {}
    for index, row in enumerate(rows):
        key = prompt_hash(row["prompt"])
        if key in groups:
            if row["prompt"] != rows[groups[key][0]]["prompt"]:
                raise ValueError("Prompt hash collision")
            if labels[key] != row["label"]:
                raise ValueError(f"Identical prompt has conflicting labels at source row {index + 1}")
        else:
            groups[key] = []
            labels[key] = row["label"]
        groups[key].append(index)
    ordered = sorted(groups, key=lambda key: (hashlib.sha256(f"{seed}:{key}".encode()).hexdigest(), key))
    target_train = math.floor(len(rows) * (1 - eval_ratio))
    count, split_group, train_count = 0, None, 0
    for group_index, key in enumerate(ordered):
        count += len(groups[key])
        if count > target_train:
            break
        if count % BATCH_SIZE == 0 and len(rows) - count >= len(TEACHERS):
            split_group, train_count = group_index + 1, count
    if split_group is None or not train_count:
        raise ValueError("Cannot split whole prompt groups into nonempty 20-row train batches and >=5 eval rows")
    train = [index for key in ordered[:split_group] for index in groups[key]]
    evaluation = [index for key in ordered[split_group:] for index in groups[key]]
    report = {
        "seed": seed,
        "requested_eval_ratio": eval_ratio,
        "actual_eval_ratio": len(evaluation) / len(rows),
        "unique_prompts": len(groups),
        "duplicate_prompt_groups": sum(len(indices) > 1 for indices in groups.values()),
        "extra_duplicate_rows": len(rows) - len(groups),
        "duplicate_policy": "Keep all source rows; identical prompts stay in one split; conflicting labels fail.",
        "prompt_identity": "SHA256 of canonical JSON message list; no whitespace or semantic normalization.",
        "train_eval_prompt_overlap": len(
            {prompt_hash(rows[i]["prompt"]) for i in train} & {prompt_hash(rows[i]["prompt"]) for i in evaluation}
        ),
        "dropped_rows": 0,
        "replicated_rows": 0,
    }
    return train, evaluation, report


def route_rows(rows: list[dict], indices: list[int], split: str, source_sha256: str) -> list[dict]:
    routed = []
    for position, source_index in enumerate(indices):
        row = copy.deepcopy(rows[source_index])
        teacher = TEACHERS[position % len(TEACHERS)]
        provenance = {
            "source": SOURCE,
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_sha256,
            "source_line_1based": source_index + 1,
            "prompt_sha256": prompt_hash(row["prompt"]),
            "split": split,
        }
        if "data_source" in row:
            provenance["original_data_source"] = row["data_source"]
        row["data_source"] = teacher
        metadata = row.setdefault("extra_info", {})
        metadata.update(rm_type="mopd", data_source=teacher, reward_source=REWARD_SOURCE, dapo_provenance=provenance)
        routed.append(row)
    return routed


def route_summary(rows: list[dict], *, complete_batches: bool) -> dict:
    counts = Counter(row["extra_info"]["data_source"] for row in rows)
    if set(counts) != set(TEACHERS):
        raise ValueError("Every split must contain all five Teachers")
    if complete_batches and len(rows) % BATCH_SIZE:
        raise ValueError("Training row count must be divisible by 20")
    full_batches = len(rows) // BATCH_SIZE
    for start in range(0, full_batches * BATCH_SIZE, BATCH_SIZE):
        batch_counts = Counter(row["extra_info"]["data_source"] for row in rows[start : start + BATCH_SIZE])
        if batch_counts != Counter(dict.fromkeys(TEACHERS, 4)):
            raise ValueError(f"Unbalanced 20-prompt batch starting at row {start}")
    return {
        "rows": len(rows),
        "teacher_counts": dict(sorted(counts.items())),
        "complete_batches_of_20": full_batches,
        "every_complete_batch_has_four_per_teacher": True,
        "tail_rows": len(rows) % BATCH_SIZE,
    }


def token_length_report(
    splits: dict[str, list[dict]], tokenizer: Any, *, max_context_len: int, max_response_len: int
) -> dict:
    budget = max_context_len - max_response_len
    if max_response_len <= 0 or budget <= 0:
        raise ValueError("Context length must leave a positive prompt and response budget")
    reports = {}
    for split, rows in splits.items():
        lengths = []
        over_budget = []
        for position, row in enumerate(rows):
            rendered = tokenizer.apply_chat_template(
                row["prompt"], tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            length = len(tokenizer.encode(rendered, add_special_tokens=False))
            lengths.append(length)
            if length > budget:
                over_budget.append({"output_row_0based": position, "prompt_tokens": length})
        ordered = sorted(lengths)
        reports[split] = {
            "rows": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "p50": ordered[(len(ordered) - 1) // 2],
            "p99": ordered[math.ceil(len(ordered) * 0.99) - 1],
            "first_60_max": max(lengths[:60]),
            "first_60_over_budget": sum(length > budget for length in lengths[:60]),
            "over_budget_count": len(over_budget),
            "over_budget_rows": over_budget,
            "ordered_lengths_sha256": hashlib.sha256(canonical(lengths)).hexdigest(),
        }
    return {
        "checked": True,
        "template_kwargs": {"enable_thinking": False, "add_generation_prompt": True},
        "encode_add_special_tokens": False,
        "max_context_len": max_context_len,
        "max_response_len": max_response_len,
        "prompt_budget": budget,
        "fits_context_budget": all(report["over_budget_count"] == 0 for report in reports.values()),
        "splits": reports,
    }


def prepare(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise ValueError("Output directory must be fresh; source and earlier preparations are never overwritten")
    rows, source = read_source(args.input)
    train_indices, eval_indices, split_report = split_indices(rows, seed=args.seed, eval_ratio=args.eval_ratio)
    splits = {
        "train": route_rows(rows, train_indices, "train", source["sha256"]),
        "eval": route_rows(rows, eval_indices, "eval", source["sha256"]),
    }
    report = {
        "version": 1,
        "scope": "Deterministic local text DAPO preparation; no model, GPU, training, or quality evaluation run.",
        "multimodal_training_verified": False,
        "source": {**source, "repo_id": SOURCE, "revision": SOURCE_REVISION, "rows": len(rows)},
        "split": split_report,
        "routing": {name: route_summary(part, complete_batches=name == "train") for name, part in splits.items()},
        "runtime_requirements": {
            "use_streaming_dataset": False,
            "rollout_shuffle": False,
            "rollout_batch_size": BATCH_SIZE,
            "rollout_max_prompt_len": None,
            "note": "Filtering, shuffling, or different batching can invalidate ordered batch balance.",
        },
        "reward_source": REWARD_SOURCE,
        "token_lengths": {"checked": False},
        "files": {},
    }
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
        report["token_lengths"] = {
            "tokenizer_path": str(args.tokenizer.resolve()),
            **token_length_report(
                splits, tokenizer, max_context_len=args.max_context_len, max_response_len=args.max_response_len
            ),
        }
    payloads = {name: b"".join(canonical(row) + b"\n" for row in part) for name, part in splits.items()}
    for name, payload in payloads.items():
        report["files"][f"{name}.jsonl"] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (args.output_dir / f"{name}.jsonl").write_bytes(payload)
    (args.output_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.tokenizer is not None and not report["token_lengths"]["fits_context_budget"]:
        raise ValueError("Prompt length budget exceeded; inspect validation.json before using these files")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing raw DAPO JSONL; never modified")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory for train/eval and validation")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--tokenizer", type=Path, help="Optional local tokenizer for complete CPU length preflight")
    parser.add_argument("--max-context-len", type=int, default=16384)
    parser.add_argument("--max-response-len", type=int, default=1024)
    args = parser.parse_args()
    report = prepare(args)
    sys.stdout.write(json.dumps({"output_dir": str(args.output_dir), "routing": report["routing"]}) + "\n")


if __name__ == "__main__":
    main()
