# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy

import torch

from relax.backends.fsdp import checkpoint
from relax.backends.fsdp.lora import inject_lora_adapter, lora_metadata_dict, peft_adapter_state_dict
from relax.models.lamra.encoding import LegacyQwen2VLEncoder, collate_pairs
from relax.models.lamra.loss import native_infonce
from relax.models.lamra.model import LANGUAGE_TARGETS, load_retrieval_model


def tiny_config():
    from transformers import Qwen2VLConfig

    return Qwen2VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={"depth": 1, "embed_dim": 16, "hidden_size": 16, "num_heads": 2},
        image_token_id=31,
        video_token_id=30,
    )


def test_qwen_adapter_checkpoint_export_and_reload_preserve_every_tensor(tmp_path):
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import Qwen2VLForConditionalGeneration

    config = tiny_config()
    hf_model = Qwen2VLForConditionalGeneration(config)
    untouched = copy.deepcopy(hf_model)
    model = LegacyQwen2VLEncoder(hf_model.model, image_token_id=31, subbatch_size=2)
    model.requires_grad_(False)
    targets = [
        name
        for name, _ in model.named_modules()
        if ".language_model.layers." in name and name.rsplit(".", 1)[-1] in LANGUAGE_TARGETS
    ]
    inject_lora_adapter(model, rank=2, alpha=4, target_modules=targets, task_type="CAUSAL_LM")
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad])
    pairs = [
        {
            "query": {"input_ids": torch.tensor([1, value, 9, 2])},
            "positive": {"input_ids": torch.tensor([1, value + 1, 9, 2])},
        }
        for value in (3, 5)
    ]
    features = model(collate_pairs(pairs, pad_token_id=0, emb_token_id=9, image_token_id=31))
    loss = native_infonce(*features.chunk(2))
    loss.backward()
    optimizer.step()
    expected = {
        name: value.detach().clone() for name, value in peft_adapter_state_dict(list(model.named_parameters())).items()
    }
    assert any(torch.count_nonzero(value) for name, value in expected.items() if ".lora_B." in name)
    checkpoint.save_checkpoint(
        str(tmp_path),
        "lamra",
        0,
        model,
        optimizer,
        trainer_state={"next_rollout_id": 1},
        weight_sync_manifest={},
        adapter_contract={
            "family": "lamra",
            "save_mode": "adapter",
            "lora": lora_metadata_dict(
                rank=2,
                alpha=4,
                dropout=0.0,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        },
        adapter_only=True,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(10)
    state = checkpoint.load_checkpoint(str(tmp_path), "lamra", 0, model, optimizer, adapter_only=True)
    assert state["next_rollout_id"] == 1
    restored = peft_adapter_state_dict(list(model.named_parameters()))
    assert all(torch.equal(restored[name], value) for name, value in expected.items())
    exported = checkpoint.export_peft_adapter(str(tmp_path), "lamra", 0, str(tmp_path / "adapter"))
    reloaded = get_peft_model_state_dict(PeftModel.from_pretrained(untouched, exported))
    assert reloaded.keys() == expected.keys()
    assert all(torch.equal(reloaded[name], value) for name, value in expected.items())


def test_legacy_checkpoint_maps_entire_language_stack_and_reloads_fp32_initial_adapter(tmp_path):
    from types import SimpleNamespace

    from safetensors.torch import save_file
    from transformers import Qwen2VLForConditionalGeneration

    original = Qwen2VLForConditionalGeneration(tiny_config()).to(torch.bfloat16)
    original.config.save_pretrained(tmp_path)
    legacy = {
        name.replace("model.language_model.", "model.").replace("model.visual.", "visual."): value.clone()
        for name, value in original.state_dict().items()
        if name != "lm_head.weight"
    }
    save_file(legacy, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    args = SimpleNamespace(hf_checkpoint=str(tmp_path), lora_rank=2, lora_alpha=4, lora_dropout=0.05)
    model, _ = load_retrieval_model(args)
    assert torch.equal(
        model.model.language_model.embed_tokens.weight, original.model.language_model.embed_tokens.weight
    )
    for expected_layer, layer in zip(
        original.model.language_model.layers, model.model.language_model.layers, strict=True
    ):
        assert torch.equal(layer.self_attn.q_proj.base_layer.weight, expected_layer.self_attn.q_proj.weight)
    initial = peft_adapter_state_dict(list(model.named_parameters()))
    assert all(value.dtype == torch.float32 for value in initial.values())
    save_file(initial, str(tmp_path / "initial.safetensors"))
    args.lamra_initial_adapter = str(tmp_path / "initial.safetensors")
    reloaded, metadata = load_retrieval_model(args)
    actual = peft_adapter_state_dict(list(reloaded.named_parameters()))
    assert metadata["initial_adapter_exact"]
    assert actual.keys() == initial.keys()
    assert all(torch.equal(actual[name], tensor) for name, tensor in initial.items())
