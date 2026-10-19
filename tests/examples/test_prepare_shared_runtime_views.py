# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU fixtures for the explicit shared-runtime contract, without model
loading."""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import pytest

from examples.on_policy_distillation.mopd import prepare_shared_runtime_views as module


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    sources = {}
    config = {
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": {"rope_theta": 10000000},
        },
        "vision_config": {"hidden_size": 1152},
    }
    for name in ("student", *module.TEACHER_IDS):
        directory = tmp_path / name
        directory.mkdir()
        sources[name] = directory
        cfg = copy.deepcopy(config)
        if name != "student":
            cfg.update(unsloth_version=name, pad_token_id=248055)
        (directory / "config.json").write_text(json.dumps(cfg))
        (directory / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"padding_side": "right" if name == "student" else "left"})
        )
        (directory / "preprocessor_config.json").write_text(json.dumps({"source": name}))
        (directory / "chat_template.jinja").write_text(name + " original template")
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    audit = tmp_path / "audit.json"

    def refresh_audit():
        audit.write_text(
            json.dumps(
                {
                    "original_files_unchanged": True,
                    "cuda_initialized": False,
                    "models": {
                        name: {
                            "files": {
                                filename: {"sha256": module.file_hash(path)}
                                for filename, path in module.metadata_files(directory).items()
                            }
                        }
                        for name, directory in sources.items()
                    },
                }
            )
        )

    refresh_audit()

    def runtime(path):
        cfg = json.loads((path / "config.json").read_text())
        vocab = json.loads((path / "tokenizer.json").read_text())["model"]["vocab"]
        return {
            "config": cfg,
            "token_to_id": vocab,
            "merges": [["a", "b"]],
            "tokenizer_class": "FixtureTokenizer",
            "pipeline": {"normalizer": path.name},
            "special_token_ids": {"pad_token_id": cfg.get("pad_token_id", 248044)},
            "padding_side": "right" if path.name == "student" else "left",
            "chat_template_sha256": module.file_hash(path / "chat_template.jinja"),
        }

    monkeypatch.setattr(module, "runtime_metadata", runtime)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test-version")
    return sources, audit, refresh_audit


def build(inputs, output):
    sources, audit, _ = inputs
    return module.build_views(
        sources["student"], {name: str(sources[name]) for name in module.TEACHER_IDS}, output, audit
    )


def test_derived_views_preserve_originals_and_pass_existing_strict_validation(inputs, tmp_path):
    sources, audit, _ = inputs
    before = {
        name: {path.name: path.read_bytes() for path in directory.iterdir()} for name, directory in sources.items()
    }
    output = tmp_path / "views"
    report = build(inputs, output)
    assert report["status"] == "complete" and report["original_metadata_unchanged"]
    assert report["weights_copied_or_modified"] is False
    assert len(report["strict_existing_validation"]["models"]) == 6
    for name, directory in sources.items():
        view = Path(report["models"][name]["view"])
        assert (view / "model.safetensors").is_symlink()
        assert (view / "model.safetensors").resolve() == directory / "model.safetensors"
        assert (view / "config.json").read_bytes() == before["student"]["config.json"]
        assert not (view / "config.json").is_symlink()
        assert (view / "tokenizer_config.json").read_bytes() == before["student"]["tokenizer_config.json"]
        assert (output / "original_metadata" / name / "config.json").read_bytes() == before[name]["config.json"]
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before[name]
    assert (output / "original_tokenizer_audit.json").read_bytes() == audit.read_bytes()
    assert report["models"]["T5"]["original_raw_config_differences"]
    assert report["models"]["T5"]["runtime_tokenizer_pipeline_differences"]
    assert "not validated" in report["contract"]["scope"]
    with pytest.raises(ValueError, match="new directory"):
        build(inputs, output)


@pytest.mark.parametrize("change", ["rope", "tie", "shape", "token_id", "stale_audit"])
def test_rejects_unapproved_changes_before_creating_views(inputs, tmp_path, change):
    sources, _, refresh = inputs
    source = sources["T5"]
    if change in {"rope", "tie"}:
        path = source / "config.json"
        cfg = json.loads(path.read_text())
        if change == "rope":
            cfg["text_config"]["rope_parameters"]["rope_theta"] = 1000
        else:
            cfg["tie_word_embeddings"] = True
        path.write_text(json.dumps(cfg))
    elif change == "shape":
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}).encode()
        (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 6)
    else:
        (source / "tokenizer.json").write_text('{"model":{"vocab":{"a":1,"b":0}}}')
    if change != "stale_audit":
        refresh()
    output = tmp_path / "views"
    with pytest.raises(ValueError):
        build(inputs, output)
    assert not output.exists()


def test_rejects_different_merge_order(inputs, tmp_path, monkeypatch):
    original = module.runtime_metadata

    def runtime(path):
        value = original(path)
        if path.name == "T5":
            value["merges"] = [["b", "a"]]
        return value

    monkeypatch.setattr(module, "runtime_metadata", runtime)
    with pytest.raises(ValueError, match="BPE merges"):
        build(inputs, tmp_path / "views")


def test_rejects_output_inside_original_directory(inputs):
    with pytest.raises(ValueError, match="outside model sources"):
        build(inputs, inputs[0]["student"] / "views")


def test_mtp_alias_must_match_and_dtype_cannot_change_forward_precision():
    reference = {"text_config": {"mtp_num_hidden_layers": 1}, "dtype": None}
    assert module.architecture_changes(reference, {**reference, "mtp_num_hidden_layers": 1})
    with pytest.raises(ValueError, match="MTP"):
        module.architecture_changes(reference, {**reference, "mtp_num_hidden_layers": 2})
    with pytest.raises(ValueError, match="dtype"):
        module.architecture_changes(reference, {**reference, "dtype": "float32"})
