# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One managed SGLang engine serving multiple checkpoint teachers."""

from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ray

from relax.core.node_group_affinity import with_control_plane_affinity
from relax.utils.logging_utils import get_logger
from relax.utils.opd.shared_teacher_lifecycle import (
    CHILD_SHUTDOWN_WAIT_S,
    ENGINE_LIFECYCLE_WAIT_S,
    SERVICE_CLOSE_WAIT_S,
    SERVICE_CONTROL_WAIT_S,
    SERVICE_DRAIN_WAIT_S,
)
from relax.utils.opd.shared_teacher_service import SharedTeacherService


logger = get_logger(__name__)


@ray.remote(concurrency_groups={"shutdown": 1})
class SharedTeacherManager:
    """Compose the existing engine manager with an exclusive scoring gateway.

    Full-weight baselines and the opt-in ADP backend use one TP1 engine. Five
    logical routes do not create five engine replicas or five GPU checkpoint
    copies.
    """

    def __init__(self, args: Any, manifest_path: str, *, pg: tuple, runtime_env: dict | None = None) -> None:
        from relax.distributed.ray.teacher_manager import TeacherManager
        from relax.utils.opd.disk_teacher_backend import create_backend, read_manifest

        manifest = read_manifest(manifest_path)
        if manifest["tp_size"] != 1:
            raise ValueError("The shared teacher baseline requires TP1")
        self.args = args
        self.manifest = manifest
        self.service: SharedTeacherService | None = None
        self.engine_manager: Any = None
        self._onloaded = True
        self._failed = False
        self._cleanup_errors: list[str] = []
        self._control_timeout = float(getattr(args, "opd_teacher_timeout_s", 1800))
        if not math.isfinite(self._control_timeout) or self._control_timeout <= 0:
            raise ValueError("Shared Teacher control timeout must be finite and positive")
        engine_args = copy.copy(args)
        engine_args.teacher_hf_checkpoint = manifest["teachers"][manifest["initial_teacher"]]
        engine_args.teacher_sglang_enable_weights_cpu_backup = False
        engine_args.teacher_sglang_enable_draft_weights_cpu_backup = False
        engine_args.teacher_sglang_disable_cuda_graph = True
        engine_args._shared_teacher_strict_lifecycle = True
        if manifest["backend"] in {"sglang-memory-v1", "sglang-adp-v1"}:
            if manifest["backend"] == "sglang-memory-v1":
                from relax.utils.opd.memory_teacher_backend import LOADER
            else:
                from relax.utils.opd.adp_teacher_backend import LOADER

                runtime_env = copy.deepcopy(runtime_env or {})
                runtime_env.setdefault("env_vars", {}).update(
                    RELAX_ADP_TEACHER_DIR=manifest["package_dir"],
                    RELAX_ADP_EVENT_LOG=str(self.manifest["event_log"]) + ".worker.jsonl",
                )
                inherited_path = runtime_env["env_vars"].get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
                runtime_env["env_vars"]["PYTHONPATH"] = os.pathsep.join(
                    path
                    for path in [*inherited_path.split(os.pathsep), *manifest.get("dependency_pythonpath", [])]
                    if path
                )

            configured = getattr(engine_args, "teacher_sglang_custom_weight_loader", None) or []
            loaders = [configured] if isinstance(configured, str) else list(configured)
            if LOADER not in loaders:
                loaders.append(LOADER)
            engine_args.teacher_sglang_custom_weight_loader = loaders
        try:
            self.engine_manager = TeacherManager.options(
                **with_control_plane_affinity(args, {"num_cpus": 1, "num_gpus": 0, "runtime_env": runtime_env})
            ).remote(engine_args, 1, 1, pg=pg, shared_pg=True, bundle_offset=0)
            urls = self._get_control(self.engine_manager.get_urls.remote())
            if len(urls) != 1:
                raise RuntimeError("Shared teacher requires exactly one physical engine")
            upstream = urls[0].removesuffix("/generate").rstrip("/")
            backend = create_backend(upstream, manifest, timeout_seconds=self._control_timeout)
            self.service = SharedTeacherService(
                backend,
                list(manifest["teachers"]),
                manifest["initial_teacher"],
                ray.util.get_node_ip_address(),
                timeout_seconds=getattr(args, "opd_teacher_timeout_s", 1800),
                max_batch_size=getattr(args, "opd_shared_teacher_max_batch_size", 32),
                max_queue_size=getattr(args, "opd_shared_teacher_max_queue_size", 1024),
                batch_wait_seconds=getattr(args, "opd_shared_teacher_batch_wait_ms", 5) / 1000,
            )
            self.service.start()
            logger.info("Shared Teacher ready: one TP1 engine, logical teachers=%s", list(manifest["teachers"]))
        except BaseException:
            try:
                self.shutdown()
            except Exception:
                logger.exception("Shared Teacher startup cleanup failed")
            raise

    def get_routes(self) -> dict[str, list[str]]:
        return {
            teacher: [f"{self.service.origin}/teachers/{teacher}/generate"] for teacher in self.manifest["teachers"]
        }

    def get_urls(self) -> list[str]:
        return [urls[0] for urls in self.get_routes().values()]

    def _get_control(self, reference: Any, *, timeout_seconds: float | None = None) -> Any:
        try:
            result = ray.get(reference, timeout=self._control_timeout if timeout_seconds is None else timeout_seconds)
            self._ensure_healthy()
            return result
        except Exception:
            self._failed = True
            if self.service is not None:
                # A timed-out remote operation may still finish. Keep admission
                # poisoned even if a caller attempts another lifecycle action.
                self.service.failed = True
            raise

    def _ensure_healthy(self) -> None:
        if self._failed:
            raise RuntimeError("Shared Teacher control failed; restart is required")

    def offload(self) -> None:
        self._ensure_healthy()
        if not self._onloaded:
            return
        try:
            self.service.suspend(timeout_seconds=SERVICE_CONTROL_WAIT_S)
            self._persist_gateway_metrics()
            self._get_control(self.engine_manager.offload.remote(), timeout_seconds=ENGINE_LIFECYCLE_WAIT_S + 2)
            self._onloaded = False
        except BaseException:
            self._failed = True
            self.service.failed = True
            raise

    def _persist_gateway_metrics(self) -> None:
        """Persist drained scoring evidence before releasing the teacher
        GPU."""
        try:
            snapshot = self.service.snapshot(timeout_seconds=SERVICE_CLOSE_WAIT_S)
            path = Path(self.manifest["event_log"]).with_name("gateway_metrics.jsonl")
            record = {
                "schema_version": 1,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "before_offload",
                "counter_scope": "cumulative admitted requests; http_success means HTTP 200, not validated scores",
                **snapshot,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            # Admission is already suspended. Missing scoring evidence must not
            # silently become a successful training round or permit a retry.
            self._failed = True
            self.service.failed = True
            raise

    def onload(self, tags: list[str] | None = None) -> None:
        self._ensure_healthy()
        if tags is not None:
            raise ValueError("Shared teacher onload must restore all weights and caches")
        if self._onloaded:
            return
        try:
            self._get_control(self.engine_manager.onload.remote(), timeout_seconds=ENGINE_LIFECYCLE_WAIT_S + 2)
            self.service.restore_after_onload(timeout_seconds=SERVICE_CONTROL_WAIT_S)
            self._ensure_healthy()
            self._onloaded = True
        except BaseException:
            self._failed = True
            self.service.failed = True
            raise

    def health_check(self) -> bool:
        if self._failed:
            return False
        snapshot = self.service.snapshot(timeout_seconds=SERVICE_CLOSE_WAIT_S)
        if not snapshot["health"]["healthy"]:
            return False
        return not self._onloaded or self._get_control(self.engine_manager.health_check.remote())

    def get_metrics(self) -> dict[str, Any]:
        return self.service.snapshot(timeout_seconds=SERVICE_CLOSE_WAIT_S)

    @ray.method(concurrency_group="shutdown")
    def shutdown(self) -> None:
        # This method must run even if the ordinary actor thread is waiting on
        # a remote lifecycle call. Poison first; a late completion cannot rearm.
        self._failed = True
        errors = list(getattr(self, "_cleanup_errors", []))
        service = self.service
        if service is not None:
            service.begin_close()
            try:
                service.close(timeout_seconds=SERVICE_DRAIN_WAIT_S)
            except Exception:
                # Do not wait for an uncertain GPU operation before stopping
                # its owned engine. Retry local draining after engine shutdown.
                logger.warning("Shared Teacher still draining; stopping its owned engine")
        manager = self.engine_manager
        if manager is not None:
            try:
                reference = manager.shutdown.options(concurrency_group="shutdown").remote()
                ray.get(reference, timeout=CHILD_SHUTDOWN_WAIT_S)
            except Exception as error:
                errors.append(f"owned engine cleanup unconfirmed: {type(error).__name__}: {error}")
            finally:
                try:
                    ray.kill(manager)
                except Exception as error:
                    errors.append(f"manager termination unconfirmed: {error}")
                self.engine_manager = None
        try:
            if service is not None:
                service.close(timeout_seconds=SERVICE_CLOSE_WAIT_S)
        except Exception as error:
            errors.append(f"gateway cleanup unconfirmed: {type(error).__name__}: {error}")
        if errors:
            self._cleanup_errors = errors
            raise RuntimeError("; ".join(errors))
        self._onloaded = False
