# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""HF norm semantics and the model-specific Megatron spec boundary."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRMSNorm


SOURCE = Path(__file__).resolve().parents[3] / "relax/models/qwen2_vl/layer_specs.py"
spec = importlib.util.spec_from_file_location("qwen2_vl_layer_specs_under_test", SOURCE)
layer_specs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layer_specs)


def norm_config(**overrides):
    values = dict(
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=False,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
    )
    return SimpleNamespace(**(values | overrides))


def test_hf_norm_preserves_bf16_output_and_input_gradient():
    torch.manual_seed(42)
    actual = layer_specs.HFEarlyCastRMSNorm(norm_config(), hidden_size=48, eps=1e-6)
    reference = Qwen2VLRMSNorm(48, eps=1e-6).to(dtype=torch.bfloat16)
    with torch.no_grad():
        actual.weight.copy_(torch.randn_like(actual.weight))
        reference.weight.copy_(actual.weight)
    x = torch.randn(16, 48, dtype=torch.bfloat16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    cotangent = torch.randn_like(x)
    output = actual(x)
    expected = reference(other)
    (dx,) = torch.autograd.grad(output, x, cotangent)
    (expected_dx,) = torch.autograd.grad(expected, other, cotangent)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
    assert output.dtype == torch.bfloat16
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    late_cast = (actual.weight.float() * normalized).to(x.dtype)
    assert not torch.equal(output, late_cast)


def test_norm_builder_respects_meta_initialization():
    with torch.device("meta"):
        norm = layer_specs.HFEarlyCastRMSNorm(norm_config(), hidden_size=48)
    assert norm.weight.device.type == "meta"
    assert norm.weight.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "overrides", [{"normalization": "LayerNorm"}, {"layernorm_zero_centered_gamma": True}, {"sequence_parallel": True}]
)
def test_norm_rejects_unsupported_semantics(overrides):
    with pytest.raises(ValueError, match="ordinary RMSNorm"):
        layer_specs.HFEarlyCastRMSNorm(norm_config(**overrides), hidden_size=48)


def test_spec_keeps_te_projections_and_covers_final_norm():
    provider_module = pytest.importorskip(
        "megatron.bridge.models.gpt_provider", reason="Requires the Megatron/TE runtime used by the remote GPU control"
    )
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_block import TransformerBlock

    config = provider_module.GPTModelProvider(
        num_layers=2, hidden_size=24, num_attention_heads=3, num_query_groups=1, ffn_hidden_size=48
    )
    original = provider_module.transformer_engine_layer_spec(config)
    block = layer_specs.qwen2_vl_layer_spec(config)
    assert block.module is TransformerBlock
    assert block.submodules.layer_norm is layer_specs.HFEarlyCastRMSNorm
    layers = block.submodules.layer_specs
    assert len(layers) == 2 and layers[0] is not layers[1]
    for layer in layers:
        submodules = layer.submodules
        assert submodules.input_layernorm is layer_specs.HFEarlyCastRMSNorm
        assert submodules.pre_mlp_layernorm is layer_specs.HFEarlyCastRMSNorm
        assert submodules.self_attention.submodules.linear_qkv is TEColumnParallelLinear
        assert get_submodules(submodules.mlp).linear_fc1 is TEColumnParallelLinear
        assert (
            submodules.self_attention.submodules.core_attention
            is original.submodules.self_attention.submodules.core_attention
        )
        assert get_submodules(submodules.mlp).linear_fc2 is get_submodules(original.submodules.mlp).linear_fc2
    assert original.submodules.self_attention.submodules.linear_qkv is not TEColumnParallelLinear
