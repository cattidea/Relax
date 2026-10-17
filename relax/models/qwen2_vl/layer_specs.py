# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen2-VL's HF normalization with the existing Megatron TE projections."""

from copy import deepcopy
from typing import Any


class HFEarlyCastRMSNorm:
    """Adapt the official HF norm to Megatron's norm builder interface."""

    def __new__(cls, config: Any, hidden_size: int, eps: float = 1e-6) -> Any:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRMSNorm

        if config.normalization != "RMSNorm" or config.layernorm_zero_centered_gamma or config.sequence_parallel:
            raise ValueError("Qwen2-VL requires ordinary RMSNorm without sequence parallelism")
        # Keep HF's FP32 normalization -> input-dtype cast -> gamma multiply.
        # A dtype-only conversion also preserves an outer meta/device context.
        return Qwen2VLRMSNorm(hidden_size, eps=eps).to(dtype=config.params_dtype)


def qwen2_vl_layer_spec(config: Any, vp_stage: int | None = None) -> Any:
    """Unfuse only the norms; keep the TE attention, GEMMs and MLP
    implementation."""
    from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
    from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules

    sizes = (config.tensor_model_parallel_size, config.pipeline_model_parallel_size, config.context_parallel_size)
    if sizes != (1, 1, 1) or config.sequence_parallel or vp_stage not in (None, 0):
        raise ValueError("The initial Qwen2-VL layer spec requires TP=PP=CP=1 and no sequence parallelism")
    if config.attention_backend == AttnBackend.local:
        raise ValueError("The initial Qwen2-VL layer spec requires Transformer Engine attention")
    if config.use_transformer_engine_full_layer_spec or config.use_transformer_engine_op_fuser:
        raise ValueError("Qwen2-VL HF normalization requires the unfused Transformer Engine layer spec")

    from relax.models.qwen2_vl.peft import PairLayoutTransformerLayer

    layer = deepcopy(transformer_engine_layer_spec(config))
    layer.module = PairLayoutTransformerLayer
    submodules = layer.submodules
    submodules.input_layernorm = HFEarlyCastRMSNorm
    submodules.pre_mlp_layernorm = HFEarlyCastRMSNorm
    submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
    get_submodules(submodules.mlp).linear_fc1 = TEColumnParallelLinear
    fused_norm_prefixes = {"self_attention.linear_qkv.layer_norm_", "mlp.linear_fc1.layer_norm_"}
    submodules.sharded_state_dict_keys_map = {
        key: value for key, value in submodules.sharded_state_dict_keys_map.items() if key not in fused_norm_prefixes
    }
    return ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(
            layer_specs=[deepcopy(layer) for _ in range(config.num_layers)],
            layer_norm=HFEarlyCastRMSNorm,
        ),
    )
