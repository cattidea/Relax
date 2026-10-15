# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline retrieval on the existing Actor/SFT/TransferQueue control plane."""

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from relax.distributed.ray.train_actor import TrainRayActor
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class FSDPOfflineTrainRayActor(TrainRayActor):
    """One optimizer step per global batch of atomic retrieval pairs."""

    def _agree(self, error: str | None) -> None:
        errors: list[str | None] = [None] * self._world_size
        dist.all_gather_object(errors, error, group=self.control_group)
        if any(errors):
            raise RuntimeError(
                "Offline FSDP rank failure: "
                + "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message)
            )

    def _rank0(self, function: Any) -> Any:
        result = [None, None]
        if self._rank == 0:
            try:
                result[0] = function()
            except Exception as error:
                result[1] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(result, src=0, group=self.control_group)
        if result[1]:
            raise RuntimeError(result[1])
        return result[0]

    def init(self, args: Any, role: str, with_ref: bool = False, with_opd_teacher: bool = False) -> int:
        import numpy as np
        import transfer_queue as tq
        from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

        from relax.backends.fsdp import checkpoint as ckpt
        from relax.backends.fsdp.lora import lora_metadata_dict
        from relax.backends.fsdp.runtime import fsdp2_wrap
        from relax.utils.distributed_utils import get_gloo_group
        from relax.utils.utils import load_function

        super().init(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        self.control_group = get_gloo_group()
        self.device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        self._pending_metrics: list[tuple[int, torch.Tensor, torch.cuda.Event]] = []
        error = None
        try:
            self.model, self.model_metadata = load_function(args.model_adapter_path)(args)
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
            self.pad_token_id = tokenizer.pad_token_id
            self.emb_token_id = tokenizer.convert_tokens_to_ids("<emb>")
            if self.emb_token_id is None or self.emb_token_id == tokenizer.unk_token_id:
                raise ValueError("The pretrained LamRA tokenizer must already contain <emb>")
            self.image_token_id = self.model.model.config.image_token_id
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        self._agree(error)
        self.contract = self._rank0(self._input_contract)
        self.contract["lora"] = lora_metadata_dict(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=self.model_metadata["target_modules"],
            task_type="CAUSAL_LM",
        )
        self.model = fsdp2_wrap(
            self.model.to(self.device),
            param_dtype=args.fsdp_param_dtype,
            reduce_dtype=args.fsdp_reduce_dtype,
            reshard_after_forward=args.fsdp_reshard_after_forward,
            activation_checkpointing=args.fsdp_activation_checkpointing,
            master_dtype="fp32",
        )
        self.parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=args.lr_warmup_iters, num_training_steps=args.num_rollout
        )
        next_step = 0
        if args.load:
            latest = ckpt.find_latest_committed(args.load, "lamra")
            if latest is not None:
                recorded = ckpt.read_adapter_contract(args.load, "lamra", latest)
                if recorded != self.contract:
                    raise ValueError("Checkpoint model/data/optimizer contract differs from this run")
                state = ckpt.load_checkpoint(args.load, "lamra", latest, self.model, self.optimizer, adapter_only=True)
                self.scheduler.load_state_dict(state["scheduler"])
                # LambdaLR state does not restore optimizer param-group LR.
                for group, lr in zip(self.optimizer.param_groups, self.scheduler.get_last_lr(), strict=True):
                    group["lr"] = lr
                next_step = state["next_rollout_id"]
                if next_step != latest + 1:
                    raise ValueError("Checkpoint iteration and next dataset step disagree")
        self.next_step = next_step
        self._get_parallel_config()
        tq.init(args.tq_config)
        self.client = tq.get_client()
        self.model.train()
        if self._rank == 0:
            from relax.utils.tracking_utils import init_tracking

            init_tracking(args, primary=False)
        path = Path(args.save) / "lamra"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"runtime_rank{self._rank}.json").write_text(
            json.dumps(
                {
                    **self.model_metadata,
                    "rank": self._rank,
                    "world_size": self._world_size,
                    "next_step": next_step,
                    "parameter_compute_dtype": args.fsdp_param_dtype,
                    "optimizer_master_dtype": "fp32",
                    "global_pairs": args.global_batch_size,
                    "forward_subbatch_sequences": getattr(args, "lamra_forward_subbatch_size", 32),
                    "gather": "detached_remote_local_grad",
                    "world_size_loss_multiplier": False,
                },
                indent=2,
            )
            + "\n"
        )
        logger.info(f"LamRA rank {self._rank} ready at step {next_step}/{args.num_rollout}")
        return next_step

    def _input_contract(self) -> dict[str, Any]:
        from relax.utils.replay.bundle import sha256_file

        weights = sorted(Path(self.args.hf_checkpoint).glob("*.safetensors"))
        if not weights:
            raise ValueError("LamRA requires a local safetensors pretrained checkpoint")
        inputs = {
            "query": self.args.prompt_data,
            "candidates": self.args.lamra_candidate_path,
            "instructions": self.args.lamra_instructions_path,
        }
        return {
            "family": "lamra",
            "save_mode": "adapter",
            "base_model_sha256": {path.name: sha256_file(path) for path in weights},
            "input_sha256": {key: sha256_file(Path(value)) for key, value in inputs.items()},
            "seed": self.args.seed,
            "global_batch_size": self.args.global_batch_size,
            "num_rollout": self.args.num_rollout,
            "world_size": self._world_size,
            "learning_rate": self.args.lr,
            "warmup_steps": self.args.lr_warmup_iters,
            "weight_decay": self.args.weight_decay,
            "adam_betas": [self.args.adam_beta1, self.args.adam_beta2],
            "adam_eps": self.args.adam_eps,
            "clip_grad": self.args.clip_grad,
            "temperature": getattr(self.args, "lamra_temperature", 0.05),
            "sampling": "relax_index_manager_access_seed_current_epoch_head_padding",
        }

    def _read_pairs(self, rollout_id: int) -> list[dict[str, Any]]:
        from relax.engine.sft.runtime import sft_partition_id, sft_task_name

        partition = sft_partition_id(self.args, rollout_id)
        task = sft_task_name(self.args, component="backend")
        local_pairs = self.args.global_batch_size // self._world_size
        deadline = time.monotonic() + self.args.rollout_request_timeout
        while time.monotonic() < deadline:
            meta = self.client.get_meta(
                data_fields=["multimodal_train_inputs", "total_lengths"],
                batch_size=local_pairs,
                partition_id=partition,
                task_name=task,
                sampling_config={
                    "partition_id": partition,
                    "task_name": task,
                    "dp_rank": self._rank,
                    "batch_index": 0,
                },
            )
            if meta.size:
                if meta.size != local_pairs:
                    raise ValueError("TransferQueue returned a partial retrieval batch")
                rows = self.client.get_data(meta)["multimodal_train_inputs"]
                return [getattr(row, "data", row) for row in rows]
            time.sleep(0.1)
        raise TimeoutError(f"Retrieval partition {partition!r} did not provide a full rank batch")

    def train(self, rollout_id: int, rollout_data_ref: Any = None) -> None:
        from relax.models.lamra.encoding import collate_pairs
        from relax.models.lamra.loss import native_infonce

        if rollout_id != self.next_step:
            raise ValueError(f"Expected retrieval step {self.next_step}, got {rollout_id}")
        self._drain_metrics()
        error = None
        try:
            batch = collate_pairs(
                self._read_pairs(rollout_id),
                pad_token_id=self.pad_token_id,
                emb_token_id=self.emb_token_id,
                image_token_id=self.image_token_id,
            ).to(self.device)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        self._agree(error)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = self.model(batch)
            queries, positives = features.chunk(2)
            loss = native_infonce(
                queries,
                positives,
                temperature=getattr(self.args, "lamra_temperature", 0.05),
                process_group=dist.group.WORLD,
            )
        torch._assert_async(torch.isfinite(loss), "Nonfinite LamRA loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.args.clip_grad)
        if hasattr(norm, "full_tensor"):
            norm = norm.full_tensor()
        torch._assert_async(torch.isfinite(norm), "Nonfinite LamRA gradient")
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.next_step = rollout_id + 1
        if self._rank == 0:
            values = torch.stack([loss.detach().float(), norm.detach().float(), loss.new_tensor(lr).float()])
            host = torch.empty(3, dtype=torch.float32, pin_memory=True)
            host.copy_(values, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self._pending_metrics.append((self.next_step, host, event))
        final = self.next_step == self.args.num_rollout
        if final or self.next_step % self.args.save_interval == 0:
            self.save_model(rollout_id)
        self._drain_metrics()

    def _drain_metrics(self) -> None:
        from relax.utils.tracking_utils import log

        while self._pending_metrics and self._pending_metrics[0][2].query():
            step, host, _ = self._pending_metrics.pop(0)
            loss, norm, lr = host.tolist()  # Pinned CPU tensor; event.query never blocks the training stream.
            log(self.args, {"train/loss": loss, "train/grad_norm": norm, "train/lr": lr, "step": step}, "step")
            logger.info(f"LamRA optimizer step {step}/{self.args.num_rollout}: loss={loss:.6f}, grad_norm={norm:.6f}")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        from relax.backends.fsdp import checkpoint as ckpt
        from relax.utils.rotate_ckpt import rotate_ckpt

        ckpt.save_checkpoint(
            self.args.save,
            "lamra",
            rollout_id,
            self.model,
            self.optimizer,
            trainer_state={
                "rollout_id": rollout_id,
                "next_rollout_id": self.next_step,
                "scheduler": self.scheduler.state_dict(),
            },
            weight_sync_manifest={},
            adapter_contract=self.contract,
            adapter_only=True,
            is_rank0=self._rank == 0,
        )
        self._rank0(
            lambda: rotate_ckpt(
                self.args, global_step=rollout_id, save_dir=os.path.join(self.args.save, "lamra"), committed_only=True
            )
        )
        if self.next_step == self.args.num_rollout:
            self._rank0(
                lambda: ckpt.export_peft_adapter(
                    self.args.save, "lamra", rollout_id, os.path.join(self.args.save, "adapter")
                )
            )

    def _get_parallel_config(self) -> dict[str, Any]:
        self.train_parallel_config = {"train_backend": "fsdp", "dp_size": self._world_size, "tp_size": 1, "pp_size": 1}
        return self.train_parallel_config

    def sleep(self, tags: Any = None) -> None:
        """Offline-only actors retain their GPU placement throughout
        training."""

    def wake_up(self, tags: Any = None) -> None:
        """No inference engine shares the offline actor's placement."""

    def update_weights(self) -> None:
        """Offline retrieval has no rollout-engine weight consumer."""
