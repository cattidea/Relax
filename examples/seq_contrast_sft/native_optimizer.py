# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Experimental optimizer opt-in for the measured Qwen2-VL NLI profile.

Uses the owned ChainedOptimizer's norm and base Adam instances. The SDK still
clips gradients, copies FP32 masters to model weights, steps the scheduler and
saves/loads DCP. This is not a universal DeepSpeed norm or historical PT
alignment claim. Unsupported optimizer modes fail before an update.
"""

import json
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

from examples.seq_contrast_sft import native_precision
from examples.seq_contrast_sft.optimizer_math import adam_step, global_norm_fp64


PROFILE = {
    "version": 1,
    "scope": "experimental_DP8_TP1_PP1_CP1_single_DistributedOptimizer",
    "norm": "global_FP64_L2_then_FP32_sqrt_not_universal_DS",
    "adam": "Torch210_FP64_beta_FMA_first_double_bias",
    "weight_decay": 0,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Qwen2-VL experimental optimizer: {message}")


def _validate_group(group: dict[str, Any]) -> None:
    _require(tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8, "unsupported Adam hyperparameters")
    _require(group["weight_decay"] == 0 and group["bias_correction"], "WD0 with bias correction required")
    _require(not group.get("amsgrad", False) and not group.get("maximize", False), "unsupported Adam mode")
    step, lr = group.get("step", 0), group["lr"]
    _require(type(step) is int and 0 <= step < 2**24, "unsupported group step")
    _require(type(lr) in (int, float) and math.isfinite(lr) and lr >= 0, "ordinary finite group LR required")


def _initialize_state(basic: Any, group: dict[str, Any]) -> None:
    for parameter in group["params"]:
        state = basic.state[parameter]
        if not state:
            _require(group.get("step", 0) == 0, "missing restored optimizer state")
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        _require(set(state) == {"exp_avg", "exp_avg_sq"}, "unexpected Adam state schema")
        _require(
            all(
                tensor.dtype == torch.float32
                and tensor.shape == parameter.shape
                and tensor.is_contiguous()
                and tensor.device == parameter.device
                for tensor in state.values()
            ),
            "FP32 moments must match their master shard",
        )


def _record_profile(args: Any, process_group: Any) -> None:
    name = native_precision.OPTIMIZER_PROFILE_FILE
    for root in {args.load, args.save} - {None, ""}:
        path = Path(root) / name
        if path.exists():
            _require(
                json.loads(path.read_text())["effective_optimizer_profile"] == PROFILE, "checkpoint profile differs"
            )
    if dist.get_rank(process_group) == 0:
        root = Path(args.save)
        root.mkdir(parents=True, exist_ok=True)
        target = root / name
        payload = {
            "effective_optimizer_profile": PROFILE,
            "base_model_profile": native_precision.PROFILE,
            "effective_profile_requires_both_records": True,
            "adopted_checkpoint_without_optimizer_profile": bool(
                args.load
                and (Path(args.load) / "latest_checkpointed_iteration.txt").is_file()
                and not (Path(args.load) / name).exists()
            ),
        }
        temporary = target.with_name(name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(target)


def _install(args: Any, models: list[Any], optimizer: Any) -> None:
    from megatron.core.optimizer.optimizer import _get_param_grad_norm_group, _is_separate_grad_norm_group

    from relax.utils.megatron_peft_utils import is_lora_adapter_param

    owner = tuple(id(model) for model in models)
    installed = getattr(optimizer, "_qwen2_vl_experimental_optimizer", None)
    if installed is not None:
        _require(installed == (owner, PROFILE), "optimizer owner or profile changed")
        return
    _require(type(optimizer).__name__ == "ChainedOptimizer", "owned SDK ChainedOptimizer required")
    children = optimizer.chained_optimizers
    _require(
        len(children) == 1 and type(children[0]).__name__ == "DistributedOptimizer", "one distributed child required"
    )
    child, basic = children[0], children[0].optimizer
    _require(
        type(basic).__module__ == "transformer_engine.pytorch.optimizers.fused_adam"
        and type(basic).__name__ == "FusedAdam",
        "TE FusedAdam state required",
    )
    _require(
        not basic.capturable and not basic.master_weights and not getattr(basic, "use_decoupled_grad", False),
        "ordinary uncaptured FP32 master state required",
    )
    _require(getattr(child, "_state_offloader", None) is None, "offload unsupported")
    _require(
        not child.config.use_precision_aware_optimizer
        and not child.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        and not getattr(args, "fp8", None),
        "precision-aware/FP8 optimizer unsupported",
    )
    group = child.get_grad_stats_parallel_group()
    if group is None:
        group = dist.group.WORLD
    _require(
        dist.get_world_size(group=group) == 8 and dist.get_process_group_ranks(group) == list(range(8)),
        "explicit single DP8 group required",
    )
    named = {
        name: parameter
        for model in models
        for name, parameter in model.named_parameters()
        if is_lora_adapter_param(name)
    }
    _require(
        len(named) == 392 and all(p.dtype == torch.bfloat16 for p in named.values()), "196 BF16 adapters required"
    )
    owned = {p.main_param for p in named.values() if p in child.model_param_gbuf_map}
    parameters = [p for g in basic.param_groups for p in g["params"]]
    _require(
        bool(parameters) and set(parameters) == owned and len(parameters) == len(owned), "foreign optimizer parameter"
    )
    _require(
        all(not _is_separate_grad_norm_group(_get_param_grad_norm_group(p)) for p in parameters),
        "separate norm groups unsupported",
    )
    _require(
        all(p.is_cuda and p.dtype == torch.float32 and p.is_contiguous() for p in parameters),
        "CUDA FP32 masters required",
    )
    for param_group in basic.param_groups:
        _validate_group(param_group)
        _initialize_state(basic, param_group)

    def norm(self: Any) -> torch.Tensor:
        gradients = [p.grad for p in parameters]
        _require(all(gradient is not None for gradient in gradients), "every owned shard must have a gradient")
        covered = child.get_grads_for_grad_norm()
        _require(
            sorted((g.data_ptr(), g.numel()) for g in gradients) == sorted((g.data_ptr(), g.numel()) for g in covered),
            "norm coordinate coverage changed",
        )
        return global_norm_fp64(gradients, process_group=group)

    def step(self: Any, closure: Any = None, grad_scaler: Any = None) -> None:
        _require(closure is None and grad_scaler is None, "closure/scaler unsupported")
        with torch.no_grad():
            for param_group in self.param_groups:
                _validate_group(param_group)
                next_step = param_group.get("step", 0) + 1
                for parameter in param_group["params"]:
                    state = self.state[parameter]
                    _require(parameter.grad is not None, "missing live optimizer gradient")
                    outputs = adam_step(
                        parameter,
                        parameter.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        lr=param_group["lr"],
                        step=next_step,
                        betas=param_group["betas"],
                        eps=param_group["eps"],
                    )
                    parameter.copy_(outputs[0])
                    state["exp_avg"].copy_(outputs[1])
                    state["exp_avg_sq"].copy_(outputs[2])
                param_group["step"] = next_step

    # ChainedOptimizer.step calls its norm, then the SDK clip function directly.
    # Keep that clip path and DistributedOptimizer.step_with_ready_grads intact.
    optimizer.get_grad_norm = MethodType(norm, optimizer)
    basic.step = MethodType(step, basic)
    optimizer._qwen2_vl_experimental_optimizer = (owner, PROFILE)
    _record_profile(args, group)


def before_train(args: Any, rollout_id: int, step_id: int, models: list[Any], optimizer: Any, scheduler: Any) -> None:
    """Install the model profile and experimental optimizer after checkpoint
    load."""
    native_precision.before_train(args, rollout_id, step_id, models, optimizer, scheduler, optimizer_profile=PROFILE)
    _install(args, models, optimizer)
