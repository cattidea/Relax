# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create explicit, non-destructive shared-runtime views for text MOPD.

All six views use the official Student's config/tokenizer/processor bytes. Each
view links its own original weight shards. This is a new scoring contract, not
a claim that community-native tokenization/templates/configs were identical.
Only local metadata and tensor headers are read. No downloads or GPU work
occur.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.on_policy_distillation.mopd.prepare_qwen35_9b_baseline import (
    TEACHER_IDS,
    checkpoint_layout,
    require_qwen35_9b,
    validate_models,
)
from relax.utils.opd.disk_teacher_backend import _PROCESSOR_FILES, _TOKENIZER_FILES


# These are explicit replacements in the shared scoring contract. Forward
# architecture fields (including text/vision dimensions, RoPE and tying) are not
# ignored. AutoConfig resolves aliases/defaults before this comparison.
_METADATA_PATHS = {
    "/_name_or_path",
    "/transformers_version",
    "/model_name",
    "/unsloth_version",
    "/unsloth_fixed",
    "/unsloth_fixed_mtp",
}
_INPUT_PATHS = {"/eos_token_id", "/pad_token_id", "/use_cache"}
_DTYPE_PATHS = {"/dtype", "/vision_config/dtype"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def differences(left: Any, right: Any, prefix: str = "") -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys(), key=str):
            path = f"{prefix}/{key}"
            if key not in left:
                result.append({"path": path, "official_missing": True, "original": right[key]})
            elif key not in right:
                result.append({"path": path, "official": left[key], "original_missing": True})
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    return [] if left == right else [{"path": prefix, "official": left, "original": right}]


def architecture_changes(reference: dict, candidate: dict) -> list[dict]:
    changes = differences(reference, candidate)
    for row in changes:
        path = row["path"]
        if path in _METADATA_PATHS:
            row["policy"] = "serialization_metadata_replaced_by_official"
        elif path in _INPUT_PATHS:
            row["policy"] = "explicit_shared_scoring_input_contract"
        elif path in _DTYPE_PATHS:
            values = (row.get("official"), row.get("original"))
            require(all(value in (None, "bfloat16") for value in values), f"Unsupported dtype difference: {row}")
            row["policy"] = "explicit_bfloat16_runtime_contract"
        elif path == "/mtp_num_hidden_layers":
            expected = reference.get("text_config", {}).get("mtp_num_hidden_layers")
            require(row.get("original") == expected, "Top-level MTP disagrees with official text architecture")
            row["policy"] = "duplicate_of_identical_text_mtp_setting"
        else:
            raise ValueError(f"Unapproved forward/config difference: {row}")
    return changes


def metadata_files(path: Path) -> dict[str, Path]:
    names = {"config.json", "generation_config.json", *_TOKENIZER_FILES, *_PROCESSOR_FILES}
    result = {name: path / name for name in sorted(names) if (path / name).is_file()}
    for item in sorted((path / "chat_templates").glob("*.jinja")):
        result[item.relative_to(path).as_posix()] = item
    return result


def hashes(files: dict[str, Path]) -> dict[str, str]:
    return {name: file_hash(path) for name, path in files.items()}


def runtime_metadata(path: Path) -> dict:
    """Load only local config/tokenizer; never instantiate a model or
    processor."""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast, "This contract requires an inspectable fast tokenizer backend")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    model = backend["model"]
    require(model["type"] == "BPE", "Expected the Qwen BPE tokenizer")
    merges = [item.split(" ") if isinstance(item, str) else item for item in model["merges"]]
    return {
        "config": json.loads(canonical(config.to_dict())),
        "token_to_id": tokenizer.get_vocab(),
        "merges": merges,
        "tokenizer_class": type(tokenizer).__name__,
        "pipeline": {key: value for key, value in backend.items() if key != "model"},
        "special_token_ids": {
            key: getattr(tokenizer, key) for key in ("bos_token_id", "eos_token_id", "pad_token_id")
        },
        "padding_side": tokenizer.padding_side,
        "chat_template_sha256": digest(tokenizer.chat_template),
    }


def validate_audit(path: Path, sources: dict[str, Path]) -> dict:
    audit = json.loads(path.read_text())
    require(audit.get("original_files_unchanged") is True, "Audit did not establish unmodified source files")
    require(audit.get("cuda_initialized") is False, "Expected an offline CPU-only audit")
    for name, directory in sources.items():
        row = audit["models"][name]
        for filename, entry in row["files"].items():
            relative = Path(filename)
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe audited metadata path")
            require(file_hash(directory / relative) == entry["sha256"], f"Stale audit metadata: {name}/{filename}")
    return audit


def build_views(student: Path, teachers: dict[str, str], output: Path, audit_report: Path) -> dict:
    require(set(teachers) == set(TEACHER_IDS), "Exactly T1..T5 are required")
    sources = {"student": student.resolve(), **{name: Path(teachers[name]).resolve() for name in TEACHER_IDS}}
    require(len(set(sources.values())) == 6, "Student and Teachers require six distinct source directories")
    output = output.resolve()
    require(not output.exists(), "Output must be a new directory; existing views are preserved")
    for source in sources.values():
        require(source.is_dir(), f"Missing checkpoint: {source}")
        require(source not in output.parents and output not in source.parents, "Output must be outside model sources")
    audit = validate_audit(audit_report, sources)
    assets = {name: metadata_files(path) for name, path in sources.items()}
    original_hashes = {name: hashes(files) for name, files in assets.items()}
    reference = None
    reference_layout = None
    rows = {}
    for name, path in sources.items():
        raw_config = json.loads((path / "config.json").read_text())
        require_qwen35_9b(raw_config, path)
        layout = checkpoint_layout(path)
        runtime = runtime_metadata(path)
        if reference is None:
            reference, reference_layout = runtime, layout
        require(layout == reference_layout, f"Complete tensor names/shapes/dtypes differ: {name}")
        require(runtime["token_to_id"] == reference["token_to_id"], f"Effective token-to-ID mapping differs: {name}")
        require(runtime["merges"] == reference["merges"], f"Ordered BPE merges differ: {name}")
        require(
            all(0 <= value < raw_config["text_config"]["vocab_size"] for value in runtime["token_to_id"].values()),
            f"Token IDs fall outside model vocabulary: {name}",
        )
        changes = architecture_changes(reference["config"], runtime["config"])
        rows[name] = {
            "original_checkpoint": str(path),
            "original_asset_sha256": original_hashes[name],
            "original_raw_config_differences": differences(
                json.loads((student / "config.json").read_text()), raw_config
            ),
            "effective_config_changes_and_policy": changes,
            "effective_token_to_id_equal": True,
            "effective_vocab_count": len(runtime["token_to_id"]),
            "effective_vocab_sha256": digest(runtime["token_to_id"]),
            "ordered_merges_equal": True,
            "ordered_merges_sha256": digest(runtime["merges"]),
            "layout_equal": True,
            "layout_sha256": digest(layout),
            "tensor_count": len(layout),
            "original_tokenizer_class": runtime["tokenizer_class"],
            "original_special_token_ids": runtime["special_token_ids"],
            "original_padding_side": runtime["padding_side"],
            "original_chat_template_sha256": runtime["chat_template_sha256"],
            "runtime_tokenizer_pipeline_differences": differences(reference["pipeline"], runtime["pipeline"]),
        }
    contract = {
        "id": "qwen35-9b-official-student-text-score-v1",
        "source": str(sources["student"]),
        "scope": "DAPO text prompts and sampled response input_ids; image/audio/video behavior is not validated",
        "tokenization": "Official Student assets and the installed Transformers tokenizer implementation for all roles",
        "teacher_scoring": "OPD forwards sampled input_ids with max_new_tokens=0; no Teacher-native chat rerendering",
        "native_behavior_change": "Community-native templates, padding, tokenizer implementation and listed config fields are replaced only in derived views",
        "vision_boundary": "Vision tensors/config and official processor assets are retained; pixel preprocessing and multimodal training require separate validation",
        "dtype": "bfloat16",
        "asset_sha256": original_hashes["student"],
        "effective_tokenizer_class": reference["tokenizer_class"],
        "effective_special_token_ids": reference["special_token_ids"],
        "effective_padding_side": reference["padding_side"],
        "effective_vocab_sha256": digest(reference["token_to_id"]),
        "code_evidence": {
            "relax/utils/opd/opd_main_worker.py": "build_prefill_payload_base: input_ids, max_new_tokens=0",
            "relax/utils/opd/opd_opsd_worker.py": "teacher_input_ids: teacher_rollout_tokens or rollout_tokens or sample.tokens",
            "relax/utils/opd/opd_multimodal.py": "image inputs require placeholder/expanded-ID checks; audio/video rejected",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    view_paths = {}
    for name, path in sources.items():
        view = output / ("student" if name == "student" else f"teachers/{name}")
        view.mkdir(parents=True)
        view_paths[name] = str(view)
        for filename, source in assets["student"].items():
            target = view / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o444)
        originals = output / "original_metadata" / name
        originals.mkdir(parents=True)
        for filename, source in assets[name].items():
            # Keep original complete tokenizer bytes too, making provenance
            # independent of later edits to the source metadata directory.
            target = originals / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        index = path / "model.safetensors.index.json"
        if index.is_file():
            shutil.copyfile(index, view / index.name)
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = [item.name for item in path.glob("*.safetensors")]
        links = {}
        for filename in files:
            target = path / filename
            (view / filename).symlink_to(target)
            links[filename] = {"target": str(target), "bytes": target.stat().st_size}
        rows[name]["weight_symlinks"] = links
        rows[name]["view"] = str(view)
    strict_validation = validate_models(Path(view_paths["student"]), {name: view_paths[name] for name in TEACHER_IDS})
    unchanged = all(hashes(assets[name]) == original_hashes[name] for name in sources)
    require(unchanged, "Source metadata changed during view creation")
    shutil.copyfile(audit_report, output / "original_tokenizer_audit.json")
    report = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "contract": contract,
        "models": rows,
        "reference_tensor_layout": reference_layout,
        "original_audit_sha256": file_hash(audit_report),
        "original_audit_created_at": audit.get("created_at"),
        "strict_existing_validation": strict_validation,
        "original_metadata_unchanged": unchanged,
        "weights_copied_or_modified": False,
        "payload_sha256_rechecked": False,
        "transformers_version": importlib.metadata.version("transformers"),
    }
    (output / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", action="append", required=True, metavar="T1=PATH")
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    teachers = {}
    for item in args.teacher:
        name, separator, path = item.partition("=")
        if not separator or not path or name in teachers:
            parser.error("Provide unique --teacher T1=PATH through T5=PATH")
        teachers[name] = path
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        build_views(args.student, teachers, args.output_dir, args.audit_report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Shared-runtime view preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
