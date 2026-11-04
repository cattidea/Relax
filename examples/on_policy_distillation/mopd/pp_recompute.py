# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""PP5 MOPD recompute: PP1 block10, PP2/PP3 selective, PP0/PP4 full."""

from argparse import Namespace
from typing import Any


def configure(args: Namespace) -> None:
    from megatron.core import mpu

    from relax.utils.logging_utils import get_logger

    assert mpu.get_pipeline_model_parallel_world_size() == 5, "MOPD PP5 requires PP5"
    assert mpu.get_virtual_pipeline_model_parallel_world_size() is None, "MOPD PP5 does not use VPP"
    assert (args.recompute_granularity, args.recompute_method, args.recompute_num_layers) == ("full", "uniform", 1)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    if pp_rank == 1:
        # Checkpoint local layers 1-10 (global 7-16); retain activations for 17-20.
        # Native full/block gives these last four layers no selective protection.
        args.recompute_method = "block"
        args.recompute_num_layers = 10
    elif pp_rank in (2, 3):
        args.recompute_granularity = "selective"
        args.recompute_method = None
        args.recompute_num_layers = None
        args.recompute_modules = ["gdn", "core_attn", "moe_act", "layernorm"]
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        get_logger(__name__).info(
            "MOPD PP5 recompute configured: pp=%d granularity=%s method=%s num_layers=%s modules=%s "
            "response_only_lm_head=%s",
            pp_rank,
            args.recompute_granularity,
            args.recompute_method,
            args.recompute_num_layers,
            args.recompute_modules,
            getattr(args, "response_only_lm_head", False),
        )


def before_train_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Check constructed module flags once; never change a live model's
    strategy."""
    if (rollout_id, step_id) != (0, 0):
        return

    from megatron.core import mpu
    from megatron.core.utils import unwrap_model

    from relax.utils.logging_utils import get_logger

    chunks = unwrap_model(model)
    assert len(chunks) == 1, "MOPD PP5 expects one model chunk"
    chunk = chunks[0]
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    selective = pp_rank in (2, 3)
    decoder = chunk.language_model.decoder
    config = decoder.config
    expected = ("selective", None, None) if selective else ("full", "uniform", 1)
    if pp_rank == 1:
        expected = ("full", "block", 10)
    assert (config.recompute_granularity, config.recompute_method, config.recompute_num_layers) == expected
    if selective:
        assert set(config.recompute_modules) == {"gdn", "core_attn", "moe_act", "layernorm"}
    layer_counts = (6, 14, 14, 14, 12)
    offset = sum(layer_counts[:pp_rank])
    assert [layer.layer_number for layer in decoder.layers] == list(
        range(offset + 1, offset + layer_counts[pp_rank] + 1)
    ), f"MOPD PP5 unexpected layer layout on PP{pp_rank}"

    gdn_layers = 0
    for layer in decoder.layers:
        attention = layer.self_attention
        if hasattr(attention, "recompute_gdn"):
            assert attention.recompute_gdn == selective, f"layer {layer.layer_number}: GDN recompute"
            gdn_layers += 1
        else:
            assert attention.checkpoint_core_attention == selective, f"layer {layer.layer_number}: attention recompute"
        assert layer.mlp.experts.activation_recompute == selective, f"layer {layer.layer_number}: MoE activation"
        assert not layer.mlp.moe_layer_recompute, f"layer {layer.layer_number}: unexpected whole-MoE recompute"
        assert layer.recompute_pre_mlp_layernorm == selective, f"layer {layer.layer_number}: pre-MoE layernorm"
        # The attention input norm is fused into TE; the outer module is IdentityOp.
        assert not layer.recompute_input_layernorm, f"layer {layer.layer_number}: unexpected input layernorm recompute"
    assert (gdn_layers, len(decoder.layers) - gdn_layers) == ((5, 1), (10, 4), (11, 3), (10, 4), (9, 3))[pp_rank], (
        f"MOPD PP5 unexpected GDN/full-attention mix on PP{pp_rank}"
    )
    if pp_rank == 0:
        vision = chunk.vision_model.config
        assert (vision.recompute_granularity, vision.recompute_method, vision.recompute_num_layers) == (
            "full",
            "uniform",
            1,
        )

    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        get_logger(__name__).info(
            "MOPD PP5 recompute verified: pp=%d layers=%d gdn=%d full_attention=%d selective=%s response_only_lm_head=%s",
            pp_rank,
            len(decoder.layers),
            gdn_layers,
            len(decoder.layers) - gdn_layers,
            selective,
            getattr(args, "response_only_lm_head", False),
        )
