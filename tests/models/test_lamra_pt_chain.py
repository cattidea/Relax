# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import torch

from relax.backends.fsdp import checkpoint
from relax.backends.fsdp.lora import lora_metadata_dict
from relax.models.lamra.encoding import collate_pairs
from relax.models.lamra.loss import native_infonce
from relax.models.lamra.model import load_retrieval_model
from relax.utils.replay.bundle import sha256_file
from tests.models.test_lamra_checkpoint import tiny_config


def test_native_pt_adapter_merges_and_initializes_fresh_ft(tmp_path, monkeypatch):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from examples.lamra.export_merged import export

    base = tmp_path / "base"
    Qwen2VLForConditionalGeneration(tiny_config()).save_pretrained(base)
    args = SimpleNamespace(hf_checkpoint=str(base), lora_rank=2, lora_alpha=4, lora_dropout=0.0)
    model, metadata = load_retrieval_model(args)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    pairs = [
        {side: {"input_ids": torch.tensor([1, value + offset, 9])} for side, offset in (("query", 0), ("positive", 1))}
        for value in (3, 5)
    ]
    features = model(collate_pairs(pairs, pad_token_id=0, emb_token_id=9, image_token_id=31))
    native_infonce(*features.chunk(2)).backward()
    optimizer.step()
    save = tmp_path / "pt"
    checkpoint.save_checkpoint(
        str(save),
        "lamra",
        0,
        model,
        optimizer,
        trainer_state={"next_rollout_id": 1},
        weight_sync_manifest={},
        adapter_contract={
            "family": "lamra",
            "save_mode": "adapter",
            "dataset": "nli",
            "base_model_sha256": {p.name: sha256_file(p) for p in base.glob("*.safetensors")},
            "lora": lora_metadata_dict(
                rank=2, alpha=4, dropout=0.0, target_modules=metadata["target_modules"], task_type="CAUSAL_LM"
            ),
        },
        adapter_only=True,
    )
    adapter = save / "adapter"
    checkpoint.export_peft_adapter(str(save), "lamra", 0, str(adapter))
    (save / "training_verification.json").write_text(
        json.dumps({"status": "passed", "completed_steps": 1, "nonzero_lora_b": metadata["language_lora_layers"]})
    )
    monkeypatch.setattr(
        AutoProcessor,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(save_pretrained=lambda path: None),
    )
    merged = tmp_path / "merged"
    export(SimpleNamespace(base=str(base), adapter=str(adapter), output=str(merged), expected_steps=1))
    assert json.loads((merged / "merge_verification.json").read_text())["hf_model_reload_exact"]
    # PT rank2 -> merged base -> a new rank4 FT adapter. No PT optimizer or
    # nonzero adapter B is carried across the phase boundary.
    ft, info = load_retrieval_model(
        SimpleNamespace(hf_checkpoint=str(merged), lora_rank=4, lora_alpha=8, lora_dropout=0.0)
    )
    assert info["trainable_parameters"] == 2 * metadata["trainable_parameters"]
    assert all(not torch.count_nonzero(p) for n, p in ft.named_parameters() if ".lora_B." in n)
