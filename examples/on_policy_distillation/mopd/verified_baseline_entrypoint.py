# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Enter the intended shared checkout before importing any Relax module."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path


SOURCES = (
    "examples/on_policy_distillation/mopd/verified_baseline_entrypoint.py",
    "relax/entrypoints/train.py",
    "relax/core/controller.py",
    "relax/components/actor.py",
    "relax/backends/megatron/actor.py",
    "relax/utils/opd/opd_utils.py",
    "relax/distributed/ray/shared_teacher_manager.py",
    "relax/distributed/ray/teacher_manager.py",
    "relax/backends/sglang/sglang_engine.py",
    "relax/utils/opd/disk_teacher_backend.py",
    "relax/utils/opd/memory_teacher_backend.py",
    "relax/utils/opd/adp_teacher_backend.py",
    "relax/utils/opd/colocated_adp_teacher_backend.py",
    "relax/backends/sglang/adp_teacher_loader.py",
    "relax/backends/sglang/adp/__init__.py",
    "relax/backends/sglang/adp/xor_codec.py",
    "relax/backends/sglang/adp/adaptive_delta.py",
    "relax/backends/sglang/adp/batched_delta.py",
    "relax/utils/opd/shared_teacher_gateway.py",
    "relax/utils/opd/shared_teacher_service.py",
    "relax/utils/opd/shared_teacher_lifecycle.py",
    "relax/engine/rollout/on_policy_distillation.py",
    "relax/engine/rewards/mopd.py",
)


def verify_sources(root: Path, expected: dict[str, str]) -> dict:
    if set(expected) != set(SOURCES):
        raise ValueError("Missing or unexpected baseline source fingerprints")
    os.chdir(root)
    sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("relax")
    if spec is None or Path(spec.origin).resolve() != root / "relax/__init__.py":
        raise RuntimeError("A different Relax package shadows the selected checkout")
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    if actual != expected:
        raise RuntimeError("Baseline source files changed after submission; refusing to allocate GPUs")
    return {"checkout": str(root), "relax_package": spec.origin, "source_sha256": actual}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    expected = json.loads(os.environ["RELAX_BASELINE_SOURCE_SHA256"])
    evidence = verify_sources(root, expected)
    if sys.argv[1:] == ["--check-only"]:
        sys.stdout.write(json.dumps(evidence) + "\n")
        return
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info("Verified baseline source: %s", json.dumps(evidence, sort_keys=True))
    sys.argv[0] = "relax.entrypoints.train"
    runpy.run_module("relax.entrypoints.train", run_name="__main__", alter_sys=True)


def worker_setup() -> None:
    """Apply the same checkout and fingerprint gate to every Ray worker."""
    root = Path(__file__).resolve().parents[3]
    evidence = verify_sources(root, json.loads(os.environ["RELAX_BASELINE_SOURCE_SHA256"]))
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).info("Verified baseline worker source: %s", evidence["relax_package"])
    previous = os.environ.get("RELAX_BASELINE_PRIOR_WORKER_SETUP_HOOK", "")
    if previous:
        module_name, function_name = previous.rsplit(".", 1)
        getattr(importlib.import_module(module_name), function_name)()


if __name__ == "__main__":
    main()
