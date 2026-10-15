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
        distributed_timeout_minutes=30,
    )


def test_offline_validator_accepts_complete_pair_recipe():
    validate_offline_fsdp_config(arguments())


def test_nli_validator_uses_same_actor_without_requiring_multimodal_candidates():
    args = arguments()
    args.custom_dataset_class_path = "relax.models.lamra.dataset.NLIDataset"
    args.lamra_expected_queries = 275601
    args.lora_rank = 64
    args.num_rollout = 1078
    for name in ("lamra_candidate_path", "lamra_instructions_path", "lamra_media_roots"):
        delattr(args, name)
    validate_offline_fsdp_config(args)


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
    actor._rank, actor._world_size, actor.client = 3, 8, Client()
    assert actor._read_pairs(7) == [pair] * 64


def test_final_export_can_recover_after_interrupted_artifact_write(tmp_path, monkeypatch):
    import json

    import torch
    from safetensors.torch import save_file

    from relax.backends.fsdp import checkpoint as ckpt
    from relax.backends.fsdp.offline_actor import FSDPOfflineTrainRayActor

    exports = []

    def export(save, task, iteration, output):
        from pathlib import Path

        exports.append((save, task, iteration))
        path = Path(output)
        path.mkdir(exist_ok=True)
        save_file(
            {"layer.lora_A.weight": torch.ones(2, 2), "layer.lora_B.weight": torch.ones(2, 2)},
            str(path / "adapter_model.safetensors"),
        )

    monkeypatch.setattr(ckpt, "export_peft_adapter", export)
    actor = object.__new__(FSDPOfflineTrainRayActor)
    actor.args = arguments()
    actor.args.save = str(tmp_path)
    actor.next_step = 2604
    actor.contract = {"query_count": 1332790}
    actor.model_metadata = {"trainable_tensors": 2, "language_lora_layers": 1}
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"partial")
    actor._export_final_adapter(2603)
    assert exports == [(str(tmp_path), "lamra", 2603)]
    result = json.loads((tmp_path / "training_verification.json").read_text())
    assert result["status"] == "passed"
    assert result["completed_steps"] == 2604
    assert result["nonzero_lora_b"] == 1
