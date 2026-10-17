# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in Qwen2-VL NLI numerical profile; keeps the shared optimizer and DCP.

This profile reproduces four measured forward/backward factors. It does not
replace the norm or Adam kernels, establish historical PT equivalence, or
support model parallelism. Install after checkpoint loading through the
existing before-train hook. No captured tensors or reference gradients enter
computation.
"""

import json
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


PROFILE = {
    "version": 1,
    "rope": "BF16_values_FP32_storage",
    "attention": "HF_SDPA_interleaved_pad8_bool_causal_keymask",
    "objective": "BF16_original_InfoNCE",
    "gradient_communication": "BF16_SUM_AR_original_scaling_FP32_storage",
    "optimizer": "unchanged",
}
_OWNER: tuple[int, ...] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Qwen2-VL native precision: {message}")


def _sdpa_forward(
    self: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    attn_mask_type: Any,
    attention_bias: Any = None,
    packed_seq_params: Any = None,
    num_splits: Any = None,
) -> torch.Tensor:
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    _require(attention_mask is None and attention_bias is None, "unsupported attention bias/mask")
    _require(packed_seq_params is not None and packed_seq_params.qkv_format == "thd", "THD metadata required")
    _require(str(attn_mask_type).endswith("causal") and num_splits is None, "only ordinary causal attention")
    _require(query.shape[1:] == (12, 128) and key.shape[1:] == value.shape[1:] == (2, 128), "2B heads required")
    cu = packed_seq_params.cu_seqlens_q_cpu
    lengths = getattr(packed_seq_params, "retrieval_pair_sequence_lengths", None)
    _require(lengths is not None and len(lengths) == 128, "one complete 64-pair local batch required")
    _require(len(cu) in (129, 130) and cu[0] == 0 and cu[-1] == query.shape[0], "unsupported packed boundaries")
    lengths = [b - a for a, b in zip(cu, cu[1:])]
    _require(min(lengths) > 0, "empty attention sequence")
    outputs = []
    for start, end in [(0, 128)] + ([(128, 129)] if len(lengths) == 129 else []):
        sizes = lengths[start:end]
        width = math.ceil(max(sizes) / 8) * 8

        def padded(tensor: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [F.pad(tensor[cu[i] : cu[i + 1]], (0, 0, 0, 0, 0, width - lengths[i])) for i in range(start, end)]
            ).transpose(1, 2)

        q, k, v = (padded(tensor) for tensor in (query, key, value))
        positions = torch.arange(width, device=query.device)
        allowed = (positions[None, :] <= positions[:, None])[None, None, :, :]
        allowed = allowed & (positions[None, :] < torch.tensor(sizes, device=query.device)[:, None])[:, None, None, :]
        helper = SimpleNamespace(num_key_value_groups=6, is_causal=True)
        out, _ = sdpa_attention_forward(helper, q, k, v, allowed, dropout=0.0, scaling=128**-0.5)
        outputs.extend(out[i, :length] for i, length in enumerate(sizes))
    return torch.cat(outputs, dim=0).reshape(query.shape[0], 1536)


def _stage_grad_sync(self: Any, force_all_reduce: bool = False) -> Any:
    original = self._native_precision_original_start
    if self.is_first_batch and self.grad_reduce_handle is not None:
        return original(force_all_reduce=force_all_reduce)
    _require(not force_all_reduce and self.grad_reduce_handle is None, "unexpected collective dispatch")
    _require(self.previous_grad_reduce_bucket_group is None, "predecessor communication unsupported")
    buffers = []
    try:
        for bucket in self.buckets:
            _require(not bucket.params_with_extra_main_grads, "extra main gradients unsupported")
            _require(bucket.grad_data.is_cuda and bucket.grad_data.dtype == torch.float32, "FP32 CUDA bucket required")
            _require(bucket.gradient_scaling_factor == 0.125, "SDK must apply 1/8 exactly once")
            for parameter in bucket.params_list:
                _require(id(parameter) in self._native_precision_owned, "foreign parameter in bucket")
                _require(getattr(parameter, "_cudagraph_wgrad_ready_event", None) is None, "CUDA graphs unsupported")
                _require(parameter.main_grad.dtype == torch.float32, "FP32 main_grad required")
                _require(
                    parameter.main_grad.untyped_storage().data_ptr() == bucket.grad_data.untyped_storage().data_ptr(),
                    "main_grad must alias its original bucket",
                )
            value = bucket.grad_data
            staging = value.bfloat16()
            torch._assert_async(value.eq(staging.float()).all(), "Local unfused gradients must be BF16-exact")
            buffers.append((bucket, value, staging))
            bucket.grad_data = staging
        # The SDK performs its original scaling, synchronous SUM and bookkeeping.
        result = original(force_all_reduce=True)
        with torch.no_grad():
            for bucket, value, staging in buffers:
                _require(bucket.grad_data is staging, "SDK changed staging buffer ownership")
                value.copy_(staging)
        return result
    finally:
        for bucket, value, _ in buffers:
            bucket.grad_data = value


def _install_bucket(group: Any, owned: set[int], process_group: Any) -> None:
    config = group.ddp_config
    _require(dist.get_world_size(process_group) == 8, "DP=8 required")
    _require(
        dist.get_process_group_ranks(group.intra_distributed_optimizer_instance_group)
        == dist.get_process_group_ranks(process_group),
        "unexpected communication group",
    )
    _require(config.use_distributed_optimizer and config.grad_reduce_in_fp32, "FP32 distributed optimizer required")
    _require(not config.overlap_grad_reduce and not config.average_in_collective, "only synchronous SUM reduction")
    _require(not config.reduce_scatter_with_fp32_accumulation, "FP32 accumulation RS unsupported")
    _require(config.num_distributed_optimizer_instances == 1, "one distributed optimizer instance required")
    _require(group.previous_grad_reduce_bucket_group is None, "linked communication buckets unsupported")
    group._native_precision_owned = owned
    group._native_precision_original_start = group.start_grad_sync
    group.start_grad_sync = MethodType(_stage_grad_sync, group)


def _install_objective() -> None:
    import relax.utils.training.retrieval_utils as retrieval

    original = retrieval.retrieval_infonce
    _require(not getattr(original, "_native_precision", False), "objective already installed by another owner")

    def native_objective(
        query: torch.Tensor, positive: torch.Tensor, *, temperature: float = 0.05, process_group: Any = None
    ) -> torch.Tensor:
        _require(query.shape == positive.shape == (64, 1536), "64 local pairs required")
        _require(process_group is not None and dist.get_world_size(process_group) == 8, "explicit DP8 group required")
        _require(temperature == 0.05 and not torch.is_autocast_enabled("cuda"), "unsupported objective context")
        return original(query.bfloat16(), positive.bfloat16(), temperature=temperature, process_group=process_group)

    native_objective._native_precision = True
    retrieval.retrieval_infonce = native_objective


def _record_profile(args: Any, process_group: Any) -> None:
    """Keep an explicit profile alongside DCP; never silently change a stamped
    run."""
    if not args.save:
        raise ValueError("Qwen2-VL native precision requires persistent --save")
    name = "qwen2_vl_native_precision.json"
    for root in {args.load, args.save} - {None, ""}:
        path = Path(root) / name
        if path.exists():
            _require(json.loads(path.read_text())["profile"] == PROFILE, "checkpoint precision profile differs")
    if dist.get_rank(process_group) == 0:
        root = Path(args.save)
        root.mkdir(parents=True, exist_ok=True)
        target = root / name
        payload = {
            "profile": PROFILE,
            "adopted_unstamped_load": bool(
                args.load
                and (Path(args.load) / "latest_checkpointed_iteration.txt").is_file()
                and not (Path(args.load) / name).exists()
            ),
        }
        temporary = target.with_name(name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(target)


def before_train(args: Any, rollout_id: int, step_id: int, models: list[Any], optimizer: Any, scheduler: Any) -> None:
    """Install once after DCP load; leave optimizer, scheduler and state schema
    intact."""
    global _OWNER
    from megatron.core import mpu
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.models.common.embeddings.rotary_pos_embedding import MultimodalRotaryEmbedding

    from relax.models.qwen2_vl.model import Qwen2VLModel
    from relax.utils.megatron_peft_utils import is_lora_adapter_param

    owner = tuple(id(model) for model in models)
    if _OWNER == owner:
        return
    _require(_OWNER is None, "only one owned model group per process")
    _require(args.loss_type == "sft" and args.task_type == "seq_contrast", "sequence contrastive SFT required")
    _require(args.bf16 and args.global_batch_size == 512, "BF16 512-pair profile required")
    _require(not args.gradient_accumulation_fusion, "disable gradient accumulation fusion")
    _require(args.lora_dropout == 0.05, "LoRA dropout must be the declared .05")
    _require(
        all(
            getattr(args, name) == 1
            for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
        ),
        "TP=PP=CP=1 required",
    )
    _require(not args.sequence_parallel, "sequence parallelism unsupported")
    _require(len(models) == 1 and isinstance(models[0], DistributedDataParallel), "one SDK DDP model required")
    group = mpu.get_data_parallel_group(with_context_parallel=False)
    roots = [module for module in models[0].modules() if isinstance(module, Qwen2VLModel)]
    _require(len(roots) == 1, "one Qwen2-VL model required")
    root = roots[0]
    _require(root._qwen2_vl_native_pair_lora, "public pair-aware CanonicalLoRA required")
    _require(not getattr(root.language_model, "rotary_pos_emb_cache", {}), "inference RoPE cache unsupported")
    owned = {id(parameter) for name, parameter in models[0].named_parameters() if is_lora_adapter_param(name)}
    _require(len(owned) == 392, "196 language adapters required")
    ropes = [module for module in root.modules() if isinstance(module, MultimodalRotaryEmbedding)]
    attention = [module for module in root.modules() if isinstance(module, TEDotProductAttention)]
    _require(len(ropes) == 1 and len(attention) == 28, "2B RoPE and attention geometry required")
    rope = ropes[0]
    _require(
        rope.inv_freq.dtype == torch.float32 and "inv_freq" not in dict(rope.named_buffers()),
        "FP32 runtime RoPE required",
    )
    _require(
        all(module.config.attention_dropout == 0 and module.config.window_size is None for module in attention),
        "attention dropout/window unsupported",
    )
    _require(not models[0].expert_parallel_bucket_groups and models[0].bucket_groups, "ordinary DDP buckets required")
    _record_profile(args, group)
    rope.inv_freq = rope.inv_freq.bfloat16().float()
    for module in attention:
        module.forward = MethodType(_sdpa_forward, module)
    for bucket in models[0].bucket_groups:
        _install_bucket(bucket, owned, group)
    _install_objective()
    _OWNER = owner
