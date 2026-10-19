# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ADP controls with the same admission/offload transaction as full reloads.

The worker's opt-in ``prepare_colocated`` operation retains one pinned initial
runtime snapshot. After memory-saver remaps model memory, ``restore`` first
recovers that known base, then applies its compressed edge to the logical
Teacher. This full-base CPU allocation and restoration must be reported; it is
not part of an ordinary resident Teacher-to-Teacher delta switch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from relax.utils.opd.adp_teacher_backend import LOADER
from relax.utils.opd.disk_teacher_backend import SGLangDiskBackend, _require


def serialize_control(operation: str, teacher: str) -> list[str]:
    """Only an operation/Teacher marker crosses HTTP, never package bytes."""
    import torch
    from sglang.srt.utils import MultiprocessingSerializer

    _require(operation in {"prepare_colocated", "switch", "restore", "verify"}, "unsupported colocated ADP operation")
    _require(
        bool(teacher) and all(c.isalnum() or c in "_-" for c in teacher) and teacher.isascii(),
        "invalid ADP Teacher ID",
    )
    marker = [(f"relax_adp:{operation}:{teacher}", torch.empty(0, dtype=torch.uint8))]
    return [MultiprocessingSerializer.serialize(marker, output_str=True)]


class SGLangColocatedADPBackend(SGLangDiskBackend):
    """TP1 compressed packages plus one explicit full CPU recovery base.

    This backend requires the extended ADP worker loader. It is separate from
    the resident-only pilot so the latter cannot silently acquire CPU backups
    or claim support for offload. Failures inherit the full-reload backend's
    permanent poison and paused-engine behavior.
    """

    _weight_source = "adp"

    def __init__(self, *args: Any, package_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.package_dir = Path(package_dir).resolve()
        self._preparing = False
        self._restoring = False
        self._verifying = False

    def preflight(self) -> dict[str, Any]:
        fingerprint = super().preflight()
        config = fingerprint["config"]
        _require(
            not config.get("quantization_config") and not config.get("text_config", {}).get("quantization_config"),
            "ADP requires native unquantized checkpoints",
        )
        manifest = json.loads((self.package_dir / "manifest.json").read_text())
        _require(
            manifest.get("version") == 1 and manifest.get("runtime") == "sglang-dense-v1",
            "unsupported ADP package runtime/version",
        )
        _require(manifest.get("tp_size") == 1, "colocated ADP requires TP1 packages")
        _require(set(manifest.get("teachers", {})) == set(self.teachers), "ADP package Teacher set differs")
        return fingerprint

    async def start(self, initial_teacher: str) -> None:
        await super().start(initial_teacher)
        try:
            info = await self._request("/server_info")
            _require(not info.get("quantization"), "ADP does not support quantized engines")
            _require(info.get("disable_cuda_graph") is True, "ADP requires disabled CUDA graphs")
            _require(info.get("enable_memory_saver") is True, "colocated ADP requires memory-saver")
            _require(LOADER in (info.get("custom_weight_loader") or []), f"engine must allow custom loader {LOADER}")
            manifest = json.loads((self.package_dir / "manifest.json").read_text())
            _require(manifest.get("initial_teacher") == initial_teacher, "ADP initial Teacher differs")
            self._preparing = True
            await self._reload(initial_teacher, phase="adp_prepare")
        except BaseException:
            self.failed = True
            self.current_teacher = None
            self.resident = False
            await self.close()
            raise
        finally:
            self._preparing = False

    async def restore_after_onload(self) -> None:
        self._restoring = True
        try:
            await super().restore_after_onload()
        finally:
            self._restoring = False

    def _operation(self) -> str:
        if self._preparing:
            return "prepare_colocated"
        if self._restoring:
            return "restore"
        return "verify" if self._verifying else "switch"

    async def verify_bytes(self) -> None:
        """Offline acceptance only; caller must own the drained engine
        exclusively.

        Full runtime D2H/hash work is deliberately logged as a separate phase
        and is never performed by the MOPD switching path.
        """
        self._verifying = True
        try:
            await self._reload(self.current_teacher, phase="validation_verify")
        finally:
            self._verifying = False

    def _reload_event_metadata(self) -> dict[str, Any]:
        return {
            "event": "adp_prepare"
            if self._preparing
            else "adp_restore_base_and_delta"
            if self._restoring
            else "adp_validation_verify"
            if self._verifying
            else "adp_delta_switch",
            "weight_source": "pinned_cpu_delta",
            "load_format": LOADER,
            "worker_operation": self._operation(),
            "package_dir": str(self.package_dir),
            "checkpoint_read_allowed": self._preparing,
            "page_cache_policy": "preparation only; online controls use retained CPU snapshots and compressed plans",
            "full_cpu_recovery_base_retained": True,
            "full_cpu_recovery_base_used_this_transaction": self._restoring,
        }

    def _reload_request(self, teacher_id: str) -> tuple[str, str, dict[str, Any]]:
        operation = self._operation()
        return (
            f"adp_{operation}",
            "/update_weights_from_tensor",
            {
                "serialized_named_tensors": serialize_control(operation, teacher_id),
                "load_format": LOADER,
                "flush_cache": False,
                "abort_all_requests": False,
                "disable_draft_model": True,
                "torch_empty_cache": False,
            },
        )
