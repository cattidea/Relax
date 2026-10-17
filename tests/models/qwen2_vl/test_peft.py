# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Layout/graph state regression tests; genuine SDK tests run on GPU
separately."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[3] / "relax/models/qwen2_vl/peft.py"
spec = importlib.util.spec_from_file_location("qwen2_vl_pair_peft_under_test", SOURCE)
peft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = peft
spec.loader.exec_module(peft)


def metadata(lengths, dummy=0):
    cuts = [0]
    for length in lengths:
        cuts.append(cuts[-1] + length)
    if dummy:
        cuts.append(cuts[-1] + dummy)
    return SimpleNamespace(qkv_format="thd", retrieval_pair_sequence_lengths=tuple(lengths), cu_seqlens_q_cpu=cuts)


@pytest.mark.parametrize("lengths,dummy", [([2, 3], 0), ([2, 3, 4, 1], 5), ([1] * 6, 1)])
def test_native_rows_are_grouped_queries_then_positives_with_zero_padding(lengths, dummy):
    packed = metadata(lengths, dummy)
    count = packed.cu_seqlens_q_cpu[-1]
    layout = peft.pair_projection_layout(packed, torch.device("cpu"), count)
    x = torch.arange(count * 3, dtype=torch.float64).reshape(count, 1, 3).requires_grad_()
    dense = layout.unpack(x)
    pairs = len(lengths) // 2
    for i, length in enumerate(lengths):
        row = i // 2 + (pairs if i % 2 else 0)
        start = packed.cu_seqlens_q_cpu[i]
        torch.testing.assert_close(dense[row, :length], x[start : start + length, 0], rtol=0, atol=0)
        assert torch.count_nonzero(dense[row, length:]) == 0
    result = layout.pack(dense)
    real = sum(lengths)
    torch.testing.assert_close(result[:real], x[:real], rtol=0, atol=0)
    assert torch.count_nonzero(result[real:]) == 0
    result.sum().backward()
    torch.testing.assert_close(x.grad[:real], torch.ones_like(x.grad[:real]), rtol=0, atol=0)
    assert torch.count_nonzero(x.grad[real:]) == 0


def test_nested_forward_layouts_restore_their_own_scope_after_error():
    a = peft.pair_projection_layout(metadata([2, 3]), torch.device("cpu"), 5)
    b = peft.pair_projection_layout(metadata([4, 1, 2, 1]), torch.device("cpu"), 8)
    assert peft._PROJECTION_LAYOUT.get() is None
    with peft.projection_layout(a):
        with pytest.raises(RuntimeError), peft.projection_layout(b):
            assert peft._PROJECTION_LAYOUT.get() is b
            raise RuntimeError("second batch exits")
        assert peft._PROJECTION_LAYOUT.get() is a
    assert peft._PROJECTION_LAYOUT.get() is None
    with pytest.raises(FrozenInstanceError):
        a.width = b.width


@pytest.mark.parametrize(
    "packed,total",
    [(metadata([2]), 2), (metadata([0, 2]), 2), (metadata([2, 3]), 4), (metadata([2, 3]), 6)],
)
def test_rejects_missing_pairs_or_inconsistent_physical_storage(packed, total):
    with pytest.raises(ValueError, match="Qwen2-VL"):
        peft.pair_projection_layout(packed, torch.device("cpu"), total)


def test_adapter_binding_is_idempotent_and_preserves_parameters_and_state_keys():
    module = torch.nn.Linear(3, 2, bias=False)
    identities = {name: id(value) for name, value in module.named_parameters()}
    original = module.forward
    peft._bind(module, peft._adapter_forward)
    saved = module._qwen2_vl_original_forward
    peft._bind(module, peft._adapter_forward)
    assert saved is module._qwen2_vl_original_forward
    assert identities == {name: id(value) for name, value in module.named_parameters()}
    assert tuple(module.state_dict()) == ("weight",)
    x = torch.randn(3, 1, 3)
    torch.testing.assert_close(module(x), original(x), rtol=0, atol=0)


def test_shared_adapter_keeps_train_eval_dropout_and_dense_shape():
    module = torch.nn.Dropout(0.5)
    original = module.forward
    peft._bind(module, peft._adapter_forward)
    layout = peft.pair_projection_layout(metadata([2, 3, 1, 2], dummy=2), torch.device("cpu"), 10)
    x = torch.ones(10, 1, 16)
    torch.manual_seed(71)
    expected = layout.pack(original(layout.unpack(x)))
    torch.manual_seed(71)
    with peft.projection_layout(layout):
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    module.eval()
    with peft.projection_layout(layout):
        torch.testing.assert_close(module(x), layout.pack(layout.unpack(x)), rtol=0, atol=0)


def test_fc1_accepts_transformer_engine_empty_bias_sentinel():
    base = torch.nn.Linear(3, 8, bias=False).double().requires_grad_(False)
    base.bias = torch.nn.Parameter(torch.empty(0, dtype=torch.float64), requires_grad=False)
    base.use_bias = False
    module = SimpleNamespace(
        to_wrap=base, _adapter_enabled=True, adapter=SimpleNamespace(adapter_gate=None, adapter_up=None)
    )
    layout = peft.pair_projection_layout(metadata([2, 3]), torch.device("cpu"), 5)
    x = torch.randn(5, 1, 3, dtype=torch.float64, requires_grad=True)
    with peft.projection_layout(layout):
        actual, bias = peft._fc1_forward(module, x)
    expected = torch.nn.functional.linear(x, base.weight)
    assert bias is None and peft._linear_bias(base) is None
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    g = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, g)[0], torch.autograd.grad(expected, x, g)[0], rtol=1e-15, atol=1e-15
    )
    base.bias = torch.nn.Parameter(torch.ones(8, dtype=torch.float64), requires_grad=False)
    with pytest.raises(ValueError, match="nonempty bias"):
        peft._linear_bias(base)
