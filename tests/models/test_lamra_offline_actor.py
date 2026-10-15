# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.backends.fsdp.offline_arguments import validate_offline_fsdp_config


def arguments():
    return SimpleNamespace(
        loss_type="sft",
        model_adapter_path="relax.models.lamra.model.load_retrieval_model",
        custom_dataset_class_path="relax.models.lamra.dataset.MBEIRDataset",
        hf_checkpoint="model",
        prompt_data="queries.jsonl",
        save="checkpoints",
        lamra_candidate_path="candidates.jsonl",
        lamra_instructions_path="instructions.tsv",
        lamra_media_roots=["media"],
        lamra_processor_path="processor",
        lamra_audit_path="audit.json",
        lamra_expected_queries=1332790,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        global_batch_size=512,
        fsdp_trainable_mode="lora",
        lora_rank=128,
        use_dynamic_batch_size=True,
        num_rollout=2604,
        save_interval=200,
    )


def test_offline_validator_accepts_complete_pair_recipe():
    validate_offline_fsdp_config(arguments())


def test_offline_fsdp_dispatch_selects_retrieval_actor_and_validator():
    from relax.backends.fsdp.arguments import validate_generative_config
    from relax.backends.fsdp.offline_actor import FSDPOfflineTrainRayActor
    from relax.distributed.ray.actor_group import _resolve_train_actor_class

    args = arguments()
    args.train_backend = "fsdp"
    assert _resolve_train_actor_class(args) is FSDPOfflineTrainRayActor
    validate_generative_config(args)
    args.global_batch_size = 511
    with pytest.raises(ValueError, match="Global pair count"):
        validate_generative_config(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("loss_type", "rm"),
        ("model_adapter_path", "unsupported"),
        ("offload_train", True),
        ("global_batch_size", 511),
        ("sft_tq_num_shards", 2),
        ("save_interval", 0),
    ],
)
def test_offline_validator_rejects_unimplemented_paths(field, value):
    args = arguments()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_offline_fsdp_config(args)


def test_rank_fetch_uses_pair_atomic_partition_and_explicit_cache_key():
    from relax.backends.fsdp.offline_actor import FSDPOfflineTrainRayActor

    pair = {"query": {}, "positive": {}}

    class Client:
        def get_meta(self, **kwargs):
            assert kwargs["batch_size"] == 64
            assert kwargs["partition_id"] == "sft_7"
            assert kwargs["sampling_config"] == {
                "partition_id": "sft_7",
                "task_name": "sft_train",
                "dp_rank": 3,
                "batch_index": 0,
            }
            return SimpleNamespace(size=64)

        def get_data(self, meta):
            return {"multimodal_train_inputs": [SimpleNamespace(data=pair) for _ in range(meta.size)]}

    actor = object.__new__(FSDPOfflineTrainRayActor)
    actor.args = arguments()
    actor.args.rollout_request_timeout = 1
    actor._rank, actor._world_size, actor.client = 3, 8, Client()
    assert actor._read_pairs(7) == [pair] * 64
