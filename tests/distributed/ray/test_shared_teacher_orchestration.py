# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import Namespace
from types import ModuleType

import pytest

from relax.utils.opd import opd_utils


@pytest.mark.parametrize("backend", ["sglang-disk-v1", "sglang-memory-v1"])
def test_five_logical_teachers_launch_one_manager_on_one_gpu_and_offload_before_actor(monkeypatch, backend):
    import ray

    from relax.utils.opd import disk_teacher_backend

    paths = {f"T{i}": f"/checkpoints/T{i}" for i in range(1, 6)}
    routes = {key: [f"http://gateway/teachers/{key}/generate"] for key in paths}
    manifest = {"backend": backend, "tp_size": 1, "initial_teacher": "T1", "teachers": paths}
    monkeypatch.setattr(disk_teacher_backend, "read_manifest", lambda path: manifest)
    pg = ("placement", list(range(8)), list(range(8)))
    # This is a control-flow unit test; importing the full Ray Serve service
    # would pull unrelated platform/dependency initialization into CPU checks.
    service = ModuleType("relax.core.service")
    service.create_placement_group = lambda **kwargs: pg
    monkeypatch.setitem(sys.modules, service.__name__, service)
    calls = []

    class Method:
        def __init__(self, name):
            self.name = name

        def remote(self):
            calls.append(self.name)
            return routes if self.name == "get_routes" else None

    handle = Namespace(get_routes=Method("get_routes"), offload=Method("offload"))

    class Manager:
        @classmethod
        def options(cls, **kwargs):
            assert kwargs["num_gpus"] == 0
            return cls

        @classmethod
        def remote(cls, args, manifest_path, **kwargs):
            calls.append("spawn")
            assert kwargs["pg"] == pg and manifest_path == "manifest.json"
            return handle

    module = ModuleType("relax.distributed.ray.shared_teacher_manager")
    module.SharedTeacherManager = Manager
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(ray, "get", lambda result, **kwargs: result)
    args = Namespace(
        use_opd=True,
        opd_type="sglang",
        colocate=True,
        hybrid=False,
        teacher_hf_checkpoint=paths["T1"],
        resource={"actor": [1, 8], "rollout": [1, 7], "teacher": [1, 1]},
        rollout_num_gpus=7,
        teacher_num_gpus_per_engine=1,
        offload_rollout=True,
    )
    actual_pg, managers = opd_utils._start_managed_shared_teacher(args, "manifest.json")
    assert actual_pg == pg and managers == [handle]
    assert calls == ["spawn", "get_routes", "offload"]
    assert args.opd_teacher_routes_map == routes and args.opd_require_all_teacher_scores


@pytest.mark.parametrize("teacher_gpus,rollout_gpus", [(5, 3), (2, 6), (1, 6)])
def test_shared_teacher_rejects_wrong_physical_gpu_budget_before_start(monkeypatch, teacher_gpus, rollout_gpus):
    from relax.utils.opd import disk_teacher_backend

    manifest = {"backend": "sglang-disk-v1", "tp_size": 1, "initial_teacher": "T1", "teachers": {"T1": "/T1"}}
    monkeypatch.setattr(disk_teacher_backend, "read_manifest", lambda path: manifest)
    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    args = Namespace(
        teacher_hf_checkpoint="/T1",
        resource={"actor": [1, 8], "teacher": [1, teacher_gpus]},
        rollout_num_gpus=rollout_gpus,
        offload_rollout=True,
        teacher_num_gpus_per_engine=1,
    )
    with pytest.raises(ValueError):
        opd_utils._start_managed_shared_teacher(args, "manifest.json")


def test_initial_offload_failure_cleans_manager_and_placement_before_return(monkeypatch):
    import ray

    from relax.utils.opd import disk_teacher_backend

    manifest = {"backend": "sglang-adp-v1", "tp_size": 1, "initial_teacher": "T1", "teachers": {"T1": "/T1"}}
    monkeypatch.setattr(disk_teacher_backend, "read_manifest", lambda path: manifest)
    monkeypatch.setattr(opd_utils, "is_managed_opd_teacher_colocate", lambda args: True)
    service = ModuleType("relax.core.service")
    pg = (object(), list(range(8)), list(range(8)))
    service.create_placement_group = lambda **kwargs: pg
    monkeypatch.setitem(sys.modules, service.__name__, service)
    events = []

    class Method:
        def __init__(self, name):
            self.name = name

        def remote(self):
            events.append(self.name)
            return self.name

    handle = Namespace(get_routes=Method("get_routes"), offload=Method("offload"), shutdown=Method("shutdown"))
    module = ModuleType("relax.distributed.ray.shared_teacher_manager")
    module.SharedTeacherManager = Namespace(options=lambda **kwargs: Namespace(remote=lambda *a, **k: handle))
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def get(ref, *, timeout):
        assert timeout > 0
        if ref == "offload":
            raise RuntimeError("injected offload failure")
        return {"T1": ["http://gateway/teachers/T1/generate"]} if ref == "get_routes" else None

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "kill", lambda actor: events.append(("kill", actor is handle)))
    monkeypatch.setattr(ray.util, "remove_placement_group", lambda group: events.append(("remove_pg", group is pg[0])))
    args = Namespace(
        teacher_hf_checkpoint="/T1",
        resource={"actor": [1, 8], "teacher": [1, 1]},
        rollout_num_gpus=7,
        offload_rollout=True,
        teacher_num_gpus_per_engine=1,
    )
    with pytest.raises(RuntimeError, match="injected offload failure"):
        opd_utils._start_managed_shared_teacher(args, "manifest.json")
    assert events == ["get_routes", "offload", "shutdown", ("kill", True), ("remove_pg", True)]
    assert not hasattr(args, "opd_teacher_routes_map")


def test_shutdown_failure_propagates_after_attempting_all_owned_managers(monkeypatch):
    import ray

    managers = [
        Namespace(shutdown=Namespace(remote=lambda: "bad")),
        Namespace(shutdown=Namespace(remote=lambda: "ok")),
    ]
    visited, killed = [], []

    def get(ref, *, timeout):
        visited.append(ref)
        assert timeout == 50
        if ref == "bad":
            raise TimeoutError("cleanup still running")

    monkeypatch.setattr(ray, "get", get)
    monkeypatch.setattr(ray, "kill", killed.append)
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        opd_utils.shutdown_managed_opd_teacher(managers)
    assert visited == ["bad", "ok"] and killed == managers
