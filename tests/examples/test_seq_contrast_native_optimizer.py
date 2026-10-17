# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optimizer profile drift and fresh/restored state preservation
regressions."""

import json
from types import SimpleNamespace

import pytest
import torch

from examples.seq_contrast_sft import native_optimizer, native_precision


def group(parameter, *, step=0, lr=0.0):
    return dict(
        params=[parameter], step=step, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, bias_correction=True
    )


def test_fresh_state_initialization_preserves_restored_moments():
    parameter = torch.nn.Parameter(torch.ones(5))
    basic = SimpleNamespace(state={parameter: {}})
    config = group(parameter)
    native_optimizer._validate_group(config)
    native_optimizer._initialize_state(basic, config)
    state = basic.state[parameter]
    assert torch.count_nonzero(state["exp_avg"]) == torch.count_nonzero(state["exp_avg_sq"]) == 0
    state["exp_avg"].fill_(0.125)
    state["exp_avg_sq"].fill_(0.25)
    before = {key: (value, value.clone()) for key, value in state.items()}
    native_optimizer._initialize_state(basic, group(parameter, step=8, lr=2e-5))
    for key, (original, snapshot) in before.items():
        assert state[key] is original
        assert torch.equal(state[key], snapshot)


def test_missing_or_partial_restored_state_is_rejected_without_reset():
    parameter = torch.nn.Parameter(torch.ones(5))
    basic = SimpleNamespace(state={parameter: {}})
    with pytest.raises(ValueError, match="missing restored"):
        native_optimizer._initialize_state(basic, group(parameter, step=8))
    assert basic.state[parameter] == {}
    existing = torch.full_like(parameter, 0.125)
    basic.state[parameter]["exp_avg"] = existing
    with pytest.raises(ValueError, match="state schema"):
        native_optimizer._initialize_state(basic, group(parameter, step=8))
    assert basic.state[parameter] == {"exp_avg": existing}


@pytest.mark.parametrize(
    "override",
    [
        {"weight_decay": 0.01},
        {"step": -1},
        {"step": 2**24},
        {"lr": float("nan")},
        {"lr": -1.0},
        {"amsgrad": True},
        {"maximize": True},
        {"bias_correction": False},
    ],
)
def test_unsupported_optimizer_group_is_rejected(override):
    config = group(torch.nn.Parameter(torch.ones(1)))
    config.update(override)
    with pytest.raises(ValueError):
        native_optimizer._validate_group(config)


def test_default_recipe_rejects_optimizer_resume_and_explicit_mode_preserves_stamp(tmp_path, monkeypatch):
    monkeypatch.setattr(native_precision.dist, "get_rank", lambda group: 0)
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    (old / "latest_checkpointed_iteration.txt").write_text("8")
    (old / "qwen2_vl_native_precision.json").write_text(json.dumps({"profile": native_precision.PROFILE}))
    path = old / native_precision.OPTIMIZER_PROFILE_FILE
    path.write_text(json.dumps({"effective_optimizer_profile": native_optimizer.PROFILE}))
    args = SimpleNamespace(load=str(old), save=str(new))
    with pytest.raises(ValueError, match="optimizer profile differs"):
        native_precision._record_profile(args, object())
    assert not new.exists()
    native_precision._record_profile(args, object(), native_optimizer.PROFILE)
    native_optimizer._record_profile(args, object())
    assert (
        json.loads((new / "qwen2_vl_native_precision.json").read_text())["effective_optimizer_profile"]
        == native_optimizer.PROFILE
    )
    original = (new / native_precision.OPTIMIZER_PROFILE_FILE).read_bytes()
    path.write_text(json.dumps({"effective_optimizer_profile": {"version": -1}}))
    with pytest.raises(ValueError, match="profile differs"):
        native_optimizer._record_profile(args, object())
    assert (new / native_precision.OPTIMIZER_PROFILE_FILE).read_bytes() == original


def test_optimizer_adoption_is_explicit_and_requires_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(native_optimizer.dist, "get_rank", lambda group: 0)
    args = SimpleNamespace(load=str(tmp_path / "absent"), save=str(tmp_path / "fresh"))
    native_optimizer._record_profile(args, object())
    name = native_precision.OPTIMIZER_PROFILE_FILE
    assert json.loads((tmp_path / "fresh" / name).read_text())["adopted_checkpoint_without_optimizer_profile"] is False
    old = tmp_path / "old"
    old.mkdir()
    (old / "latest_checkpointed_iteration.txt").write_text("7")
    args = SimpleNamespace(load=str(old), save=str(tmp_path / "adopted"))
    native_optimizer._record_profile(args, object())
    assert (
        json.loads((tmp_path / "adopted" / name).read_text())["adopted_checkpoint_without_optimizer_profile"] is True
    )
    assert not (old / name).exists()


def test_missing_optimizer_sidecar_rejects_complete_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(native_precision.dist, "get_rank", lambda group: 0)
    old = tmp_path / "old"
    old.mkdir()
    (old / "latest_checkpointed_iteration.txt").write_text("8")
    (old / "qwen2_vl_native_precision.json").write_text(
        json.dumps(dict(profile=native_precision.PROFILE, effective_optimizer_profile=native_optimizer.PROFILE))
    )
    with pytest.raises(ValueError, match="optimizer profile missing"):
        native_precision._record_profile(
            SimpleNamespace(load=str(old), save=str(tmp_path / "new")), object(), native_optimizer.PROFILE
        )
