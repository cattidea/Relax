# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail closed for the initial offline FSDP retrieval implementation."""

import math
from argparse import Namespace


def validate_offline_fsdp_config(args: Namespace) -> None:
    errors: list[str] = []
    if getattr(args, "loss_type", None) != "sft":
        errors.append("Offline FSDP currently supports LamRA retrieval through loss_type=sft only")
    if getattr(args, "model_adapter_path", None) != "relax.models.lamra.model.load_retrieval_model":
        errors.append("Offline FSDP requires the LamRA retrieval model adapter")
    dataset = getattr(args, "custom_dataset_class_path", None)
    nli = dataset == "relax.models.lamra.dataset.NLIDataset"
    if dataset not in ("relax.models.lamra.dataset.MBEIRDataset", "relax.models.lamra.dataset.NLIDataset"):
        errors.append("LamRA requires its atomic-pair M-BEIR or NLI dataset")
    fields = [
        "hf_checkpoint",
        "prompt_data",
        "save",
        "lamra_processor_path",
        "lamra_audit_path",
    ]
    if not nli:
        fields.extend(("lamra_candidate_path", "lamra_instructions_path", "lamra_media_roots"))
    for field in fields:
        if not getattr(args, field, None):
            errors.append(f"LamRA requires {field}")
    if int(getattr(args, "lamra_expected_queries", 0)) <= 0:
        errors.append("LamRA requires an explicit positive expected full query count")
    for field in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        if getattr(args, field, 1) != 1:
            errors.append(f"LamRA offline FSDP requires {field}=1")
    for field in (
        "fully_async",
        "hybrid",
        "offload_train",
        "fsdp_cpu_offload",
        "dynamic_context_parallel",
        "use_routing_replay",
        "enable_mtp_training",
        "sft_image_preprocess_on_rank",
    ):
        if getattr(args, field, False):
            errors.append(f"LamRA offline FSDP does not implement {field}")
    for field in ("sft_predict_interval", "eval_prompt_data", "eval_size", "critic_train_num_nodes"):
        if getattr(args, field, None):
            errors.append(f"LamRA offline FSDP does not implement {field}")
    if int(getattr(args, "sft_tq_num_shards", 1) or 1) != 1:
        errors.append("LamRA requires one atomic global-pair TQ partition per step")
    world = int(getattr(args, "actor_num_nodes", 1)) * int(getattr(args, "actor_num_gpus_per_node", 1))
    batch = int(getattr(args, "global_batch_size", 0) or 0)
    if world <= 0 or batch <= 0 or batch % world:
        errors.append("Global pair count must be positive and divisible by actor GPU count")
    if int(getattr(args, "n_samples_per_prompt", 1)) != 1:
        errors.append("Each LamRA TQ row is exactly one pair; n_samples_per_prompt must be 1")
    if getattr(args, "fsdp_trainable_mode", None) != "lora" or int(getattr(args, "lora_rank", 0)) <= 0:
        errors.append("LamRA requires fsdp_trainable_mode=lora and a positive lora_rank")
    if getattr(args, "fsdp_param_dtype", "bf16") != "bf16":
        errors.append("LamRA requires BF16 forward parameters")
    if getattr(args, "fsdp_reduce_dtype", "fp32") != "fp32":
        errors.append("LamRA requires FP32 gradient reduction")
    if getattr(args, "fsdp_master_dtype", None) not in (None, "fp32"):
        errors.append("LamRA requires FP32 optimizer masters")
    if not getattr(args, "use_dynamic_batch_size", False):
        errors.append("Reuse the SFT dynamic transport; the retrieval actor still forms one full-pair loss")
    if not isinstance(getattr(args, "prompt_data", None), str):
        errors.append("LamRA v1 requires one complete query JSONL path")
    if int(getattr(args, "num_rollout", 0) or 0) <= 0 or int(getattr(args, "save_interval", 0) or 0) <= 0:
        errors.append("LamRA requires explicit positive optimizer step count and checkpoint interval")
    temperature = float(getattr(args, "lamra_temperature", 0.05))
    if not math.isfinite(temperature) or temperature <= 0:
        errors.append("LamRA temperature must be finite and positive")
    if errors:
        raise ValueError("Invalid offline FSDP retrieval configuration:\n- " + "\n- ".join(errors))
