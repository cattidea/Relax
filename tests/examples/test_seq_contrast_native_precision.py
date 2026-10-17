# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packing/mask and profile-resume regressions for the opt-in NLI recipe."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[2] / "examples/seq_contrast_sft/native_precision.py"


def load_profile():
    spec = importlib.util.spec_from_file_location("qwen2_vl_native_precision_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dummy", [False, True])
def test_sdpa_packing_matches_separate_unpadded_causal_sequences(dummy):
    pytest.importorskip("transformers.integrations.sdpa_attention")
    module = load_profile()
    generator = torch.Generator().manual_seed(17)
    lengths = [1 + i % 5 for i in range(128)] + ([3] if dummy else [])
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    q = torch.randn(cu[-1], 12, 128, generator=generator, requires_grad=True)
    k = torch.randn(cu[-1], 2, 128, generator=generator, requires_grad=True)
    v = torch.randn(cu[-1], 2, 128, generator=generator, requires_grad=True)
    packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q_cpu=cu, retrieval_pair_sequence_lengths=lengths[:128])
    actual = module._sdpa_forward(None, q, k, v, None, "causal", packed_seq_params=packed)
    expected = []
    for a, b in zip(cu, cu[1:]):
        query = q[a:b].transpose(0, 1).unsqueeze(0)
        key = k[a:b].transpose(0, 1).repeat_interleave(6, dim=0).unsqueeze(0)
        value = v[a:b].transpose(0, 1).repeat_interleave(6, dim=0).unsqueeze(0)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=128**-0.5)
        expected.append(out[0].transpose(0, 1).reshape(b - a, 1536))
    expected = torch.cat(expected)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for observed, reference in zip(
        torch.autograd.grad(actual.sum(), (q, k, v)), torch.autograd.grad(expected.sum(), (q, k, v)), strict=True
    ):
        torch.testing.assert_close(observed, reference, rtol=2e-4, atol=5e-6)


def test_existing_checkpoint_profile_drift_is_rejected_before_overwrite(tmp_path, monkeypatch):
    module = load_profile()
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    args = SimpleNamespace(load=str(tmp_path / "old"), save=str(tmp_path / "new"))
    fresh = SimpleNamespace(load=str(tmp_path / "absent"), save=str(tmp_path / "fresh"))
    module._record_profile(fresh, object())
    assert (
        json.loads((tmp_path / "fresh/qwen2_vl_native_precision.json").read_text())["adopted_unstamped_load"] is False
    )
    (tmp_path / "old").mkdir()
    (tmp_path / "old/latest_checkpointed_iteration.txt").write_text("5")
    module._record_profile(args, object())
    path = tmp_path / "new/qwen2_vl_native_precision.json"
    original = path.read_bytes()
    assert json.loads(original)["adopted_unstamped_load"] is True
    (tmp_path / "old/qwen2_vl_native_precision.json").write_text(json.dumps({"profile": {"version": -1}}))
    with pytest.raises(ValueError, match="profile differs"):
        module._record_profile(args, object())
    assert path.read_bytes() == original


def test_sdk_first_batch_existing_handle_remains_a_noop():
    module = load_profile()
    # The first batch's existing-handle path must remain a true SDK no-op.
    calls = []
    bucket = SimpleNamespace(grad_data=object())
    group = SimpleNamespace(
        is_first_batch=True,
        grad_reduce_handle=object(),
        buckets=[bucket],
        _native_precision_original_start=lambda **kwargs: calls.append(kwargs),
    )
    original = bucket.grad_data
    module._stage_grad_sync(group)
    assert calls == [{"force_all_reduce": False}]
    assert bucket.grad_data is original
