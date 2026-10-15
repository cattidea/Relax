# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen2-VL retrieval model using Relax's existing PEFT injection helpers."""

from typing import Any

import torch

from relax.backends.fsdp.lora import inject_lora_adapter, peft_adapter_state_dict
from relax.models.lamra.encoding import LegacyQwen2VLEncoder


LANGUAGE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_retrieval_model(args: Any) -> tuple[LegacyQwen2VLEncoder, dict[str, Any]]:
    from transformers import Qwen2VLForConditionalGeneration

    # The full HF class owns the official legacy-checkpoint key conversions.
    # Loading its bare submodel bypasses those mappings in recent Transformers
    # and can silently leave the complete language stack randomly initialized.
    conditional, loading = Qwen2VLForConditionalGeneration.from_pretrained(
        args.hf_checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu",
        output_loading_info=True,
    )
    # Retrieval never registers or calls lm_head. Some native merged PT files
    # intentionally omit it. Every parameter actually used by the encoder must
    # load; no language/vision projection may fall back to initialization.
    missing_encoder = set(loading.get("missing_keys", ())) - {"lm_head.weight"}
    if missing_encoder or any(loading.get(name) for name in ("unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"LamRA pretrained checkpoint did not load completely: {loading}")
    backbone = conditional.model
    model = LegacyQwen2VLEncoder(
        backbone,
        image_token_id=backbone.config.image_token_id,
        subbatch_size=int(getattr(args, "lamra_forward_subbatch_size", 32)),
    )
    model.requires_grad_(False)
    targets = [
        name
        for name, _ in model.named_modules()
        if ".language_model.layers." in name and name.rsplit(".", 1)[-1] in LANGUAGE_TARGETS
    ]
    expected_layers = len(backbone.language_model.layers) * len(LANGUAGE_TARGETS)
    if len(targets) != expected_layers:
        raise ValueError("Qwen2-VL language LoRA target coverage is incomplete")
    injected = inject_lora_adapter(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=targets,
        dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
    )
    if injected != expected_layers:
        raise ValueError("PEFT did not inject every requested language projection")
    from relax.backends.fsdp.runtime import upcast_trainable_params

    upcast_trainable_params(model, torch.float32)
    initial = getattr(args, "lamra_initial_adapter", None)
    if initial:
        from safetensors.torch import load_file

        values = load_file(initial)
        parameters = dict(model.named_parameters())
        portable = peft_adapter_state_dict(list(parameters.items()))
        if values.keys() != portable.keys():
            raise ValueError("Initial LamRA adapter keys do not match the complete language adapter")
        with torch.no_grad():
            for key, value in values.items():
                name = key.removeprefix("base_model.model.")
                name = name.replace(".lora_A.weight", ".lora_A.default.weight")
                name = name.replace(".lora_B.weight", ".lora_B.default.weight")
                parameter = parameters[name]
                if parameter.shape != value.shape or parameter.dtype != value.dtype:
                    raise ValueError(f"Initial adapter shape/dtype mismatch for {key}")
                parameter.copy_(value)
                if not torch.equal(parameter, value):
                    raise ValueError(f"Initial adapter did not reload exactly: {key}")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != expected_layers * 2 or any(".lora_" not in name for name, _ in trainable):
        raise ValueError("Only the complete language LoRA adapter may be trainable")
    return model, {
        "language_lora_layers": injected,
        "trainable_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "initial_adapter_exact": bool(initial),
        "pretrained_encoder_complete": True,
        "discarded_missing_lm_head": "lm_head.weight" in loading.get("missing_keys", ()),
        "target_modules": targets,
    }
