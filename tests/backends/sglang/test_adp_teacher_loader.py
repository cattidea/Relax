# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contract tests; codec execution is stubbed, without importing SGLang."""

import copy
import gc
import hashlib
import json
import os
import struct
import subprocess
import sys
import weakref
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest

from relax.backends.sglang import adp
from relax.backends.sglang import adp_teacher_loader as loader


_ADP_PACKAGE = "relax.backends.sglang.adp"


torch = pytest.importorskip("torch")


class Modality(Enum):
    IMAGE = "image"
    VIDEO = "video"


@pytest.mark.parametrize("key", [Modality.IMAGE, None, ("tuple", "key")])
def test_tensor_free_unusual_container_keys_are_metadata_but_tensors_still_fail(key):
    assert (
        list(loader._tensor_attributes({key: {"enabled": True, "nested": [False, None]}}, "attr:metadata", set()))
        == []
    )
    with pytest.raises(ValueError, match="key containing tensors"):
        list(loader._tensor_attributes({key: {"hidden": [torch.ones(1)]}}, "attr:tensors", set()))


@pytest.mark.parametrize("kind", ["direct", "tuple", "nested_dict"])
def test_unusual_metadata_does_not_hide_tensor_keys(kind):
    tensor = torch.ones(1)
    value = {
        "direct": {tensor: True},
        "tuple": {("nested", tensor): True},
        "nested_dict": {Modality.IMAGE: [{tensor: False}]},
    }[kind]
    with pytest.raises(ValueError, match="container key"):
        list(loader._tensor_attributes(value, "attr:metadata", set()))


@pytest.mark.parametrize("kind", ["tensor_set", "empty_set", "frozenset", "object", "nested_object"])
def test_unusual_metadata_rejects_unproven_containers_and_objects(kind):
    value = {
        "tensor_set": {torch.ones(1)},
        "empty_set": set(),
        "frozenset": frozenset({1}),
        "object": object(),
        "nested_object": {"nested": [SimpleNamespace(tensor=torch.ones(1))]},
    }[kind]
    with pytest.raises(ValueError, match="unknown metadata"):
        list(loader._tensor_attributes({Modality.IMAGE: value}, "attr:metadata", set()))


@pytest.mark.parametrize("location", ["key", "value", "nested_key"])
def test_enum_with_nonscalar_value_cannot_prove_tensor_free_metadata(location):
    class TensorEnum(Enum):
        ITEM = torch.ones(1)

    value = {
        "key": {TensorEnum.ITEM: False},
        "value": {Modality.IMAGE: TensorEnum.ITEM},
        "nested_key": {Modality.IMAGE: {TensorEnum.ITEM: False}},
    }[location]
    with pytest.raises(ValueError, match="unknown metadata"):
        list(loader._tensor_attributes(value, "attr:metadata", set()))


def test_unusual_metadata_rejects_cycles_without_recursion_error():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match="unknown metadata"):
        list(loader._tensor_attributes({Modality.IMAGE: value}, "attr:metadata", set()))


def test_qwen_auto_enum_boolean_map_preserves_runtime_metadata(fixture):
    class QwenModality(Enum):
        IMAGE = auto()
        VIDEO = auto()

    before = loader._runtime_view(fixture.model)
    fixture.model.use_deepstack = {QwenModality.IMAGE: True, QwenModality.VIDEO: True}
    assert list(loader._tensor_attributes(fixture.model.use_deepstack, "attr::use_deepstack", set())) == []
    after = loader._runtime_view(fixture.model)
    assert after.metadata == before.metadata
    assert after.bindings == before.bindings
    assert set(after.auxiliary) == set(before.auxiliary)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=False)
        self.tied = self.weight
        self.register_buffer("cos_sin_cache", torch.arange(8, dtype=torch.float32).reshape(2, 4))
        self.extra = {"frequencies": torch.tensor([1.0, 2.0])}
        self.weight_alias = [self.weight]
        self.config = {"hidden_size": 3, "_name_or_path": "model-location"}


class FakeDist:
    def __init__(self, size: int = 1, rank: int = 0) -> None:
        self.size = size
        self.rank = rank
        self.group = object()
        self.records = []
        self.peer_override = None

    def all_gather_object(self, output, value, *, group) -> None:
        assert group is self.group
        self.records.append(copy.deepcopy(value))
        output[:] = [copy.deepcopy(value) for _ in range(self.size)]
        if self.peer_override is not None:
            self.peer_override(output, value, self.rank)


class FakePlan:
    def __init__(self, destinations, target) -> None:
        self.destinations = destinations
        self.target = target
        self.resets = []
        self.applies = []
        self.fail = False
        self.bank = torch.empty(0, dtype=torch.uint8)

    def mark_base_restored(self, *, verify: bool) -> None:
        self.resets.append(verify)

    def apply(self, *, overlap: bool) -> None:
        self.applies.append(overlap)
        with torch.no_grad():
            for name, tensor in self.destinations.items():
                tensor.copy_(self.target[name])
                if self.fail:
                    raise RuntimeError("injected partial write failure")


class FakeXorPlan:
    """CPU byte-XOR executor; packet decoding and GPU allocation remain
    stubbed."""

    def __init__(self, destinations, a, b, header):
        self.destinations = destinations
        self.delta = {name: _raw(a[name]) ^ _raw(b[name]) for name in destinations}
        self.bank = torch.empty(32, dtype=torch.uint8)
        self.buffers = [torch.empty(8, dtype=torch.uint8) for _ in range(2)]
        self.decoded = torch.empty(8, dtype=torch.uint8)
        self.plan = SimpleNamespace(packets=copy.deepcopy(header["packets"]))
        self.resets, self.applies = [], []
        self.armed = False

    def mark_base_restored(self, *, verify=True):
        assert verify is False
        self.resets.append(verify)
        self.armed = True

    def apply(self, *, overlap):
        assert self.armed
        self.armed = False
        self.applies.append(overlap)
        with torch.no_grad():
            for name, tensor in self.destinations.items():
                _raw(tensor).bitwise_xor_(self.delta[name])


def command(operation: str, teacher: str) -> list:
    return [(f"relax_adp:{operation}:{teacher}", torch.empty(0))]


def write_header(path: Path, header: dict) -> None:
    data = json.dumps(header).encode()
    path.write_bytes(b"ADP1\0\0\0\0" + struct.pack("<Q", len(data)) + data)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Only the production CUDA-device gate is bypassed: all storage, metadata,
    # capture, hash, graph and consensus checks below use the real implementation.
    monkeypatch.setattr(loader, "_check_parameter", lambda name, tensor: None)
    monkeypatch.setenv("RELAX_ADP_TEACHER_DIR", str(tmp_path))
    dist = FakeDist()
    monkeypatch.setattr(loader, "_tp_context", lambda: (dist, dist.group, dist.rank, dist.size))
    model = TinyModel()
    banks = {}
    for teacher in ("A", "B"):
        with torch.no_grad():
            model.weight.add_(1 if teacher == "B" else 0)
        banks[teacher] = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}
        loader.adp_teacher_loader(model, command("capture", teacher))
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            tensor.copy_(banks["A"][name])
    manifest = {
        "version": 1,
        "runtime": "sglang-dense-v1",
        "tp_size": 1,
        "initial_teacher": "A",
        "group_bytes": 67108864,
        "teachers": {teacher: {"captures": [f"captures/{teacher}/rank_0.pt"]} for teacher in banks},
        "edges": [],
    }
    paths = {}
    for source, target in (("A", "B"), ("B", "A")):
        path = tmp_path / f"{source}_{target}.adp"
        header = {
            "format": "ADP1",
            "version": 1,
            "policy": "size",
            "provenance": {
                "runtime": "sglang-dense-v1",
                "source": source,
                "target": target,
                "tp_rank": 0,
                "tp_size": 1,
            },
            "tensors": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "a_sha256": loader._hash_tensor(banks[source][name]),
                    "b_sha256": loader._hash_tensor(banks[target][name]),
                }
                for name, tensor in banks[source].items()
            },
        }
        write_header(path, header)
        paths[source, target] = path
        manifest["edges"].append({"source": source, "target": target, "packages": [path.name]})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    plans = {}

    def load_plan(path, destinations):
        source, target = path.stem.split("_")
        plan = FakePlan(destinations, banks[target])
        plans[source, target] = plan
        return plan

    monkeypatch.setattr(loader, "_load_plan", load_plan)
    return SimpleNamespace(
        root=tmp_path, model=model, banks=banks, plans=plans, paths=paths, dist=dist, manifest=manifest
    )


def prepare(fixture) -> None:
    loader.adp_teacher_loader(fixture.model, command("prepare", "A"))


def _shared_xor_package(fixture, monkeypatch, modes=("XOR_GROUP2_ANS",)):
    path = fixture.paths["A", "B"]
    header = loader._package_header(path)
    header["packets"] = [{"name": "weight", "mode": mode} for mode in modes]
    write_header(path, header)
    for edge in fixture.manifest["edges"]:
        if (edge["source"], edge["target"]) == ("B", "A"):
            edge.update(packages=[path.name], reverse=True)
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    calls = []

    def load_plan(path, destinations):
        loaded = loader._package_header(path)
        source, target = loaded["provenance"]["source"], loaded["provenance"]["target"]
        plan = FakeXorPlan(destinations, fixture.banks[source], fixture.banks[target], loaded)
        fixture.plans[source, target] = plan
        calls.append((path, plan))
        return plan

    monkeypatch.setattr(loader, "_load_plan", load_plan)
    monkeypatch.setattr(loader, "_require_pinned", lambda tensor: None)
    return calls


@pytest.mark.parametrize("mode", ["XOR_ANS", "XOR_GROUP2_ANS", "PLANES", "PLANES_ANS"])
def test_shared_xor_roundtrip_loads_one_bank_and_scratch_without_online_io(fixture, monkeypatch, mode):
    calls = _shared_xor_package(fixture, monkeypatch, (mode,))
    event_path = fixture.root / "events.jsonl"
    monkeypatch.setenv("RELAX_ADP_EVENT_LOG", str(event_path))
    prepare(fixture)
    state = fixture.model._relax_adp_teacher_state
    assert len(calls) == 1
    plan = calls[0][1]
    assert state.plans == {("A", "B"): plan, ("B", "A"): plan}
    assert state.package_bank_bytes == plan.bank.untyped_storage().nbytes() == 32
    bindings = state.bindings
    allocations = [tensor.data_ptr() for tensor in (plan.bank, *plan.buffers, plan.decoded)]

    def forbidden(*args, **kwargs):
        raise AssertionError("online XOR switching performed preparation work")

    for name in ("_package_header", "_load_plan", "_snapshot_hashes", "_hash_tensor", "_root_directory"):
        monkeypatch.setattr(loader, name, forbidden)
    for teacher in ("B", "A", "A", "B", "A"):
        loader.adp_teacher_loader(fixture.model, command("switch", teacher))
        assert state.current == teacher
        assert loader._runtime_view(fixture.model).bindings == bindings
        assert [tensor.data_ptr() for tensor in (plan.bank, *plan.buffers, plan.decoded)] == allocations
        for name, tensor in fixture.model.named_parameters():
            assert torch.equal(_raw(tensor), _raw(fixture.banks[teacher][name]))
    assert plan.resets == [False] * 4 and plan.applies == [True] * 4
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert all(row["logical_edge_count"] == 2 and row["physical_package_count"] == 1 for row in events)
    assert all(row["package_bank_bytes"] == 32 for row in events)


def test_shared_xor_supports_reverse_first_manifest_initial_teacher_and_resolved_path_alias(fixture, monkeypatch):
    calls = _shared_xor_package(fixture, monkeypatch)
    fixture.manifest["initial_teacher"] = "B"
    fixture.manifest["edges"].reverse()
    fixture.manifest["edges"][0]["packages"] = ["./A_B.adp"]
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    with torch.no_grad():
        for name, tensor in fixture.model.named_parameters():
            tensor.copy_(fixture.banks["B"][name])
    loader.adp_teacher_loader(fixture.model, command("prepare", "B"))
    assert len(calls) == 1
    for teacher in ("A", "B", "A"):
        loader.adp_teacher_loader(fixture.model, command("switch", teacher))
        for name, tensor in fixture.model.named_parameters():
            assert torch.equal(_raw(tensor), _raw(fixture.banks[teacher][name]))


@pytest.mark.parametrize("mode", ["RAW", "ADD_ANS", "SKIP", "UNKNOWN"])
def test_reverse_rejects_nonreversible_or_unknown_packet_before_loading(fixture, monkeypatch, mode):
    calls = _shared_xor_package(fixture, monkeypatch, ("XOR_GROUP2_ANS", mode))
    with pytest.raises(RuntimeError, match="requires only XOR/PLANES"):
        prepare(fixture)
    assert calls == []
    assert not fixture.model._relax_adp_teacher_state.prepared


def test_loaded_reverse_plan_cannot_silently_change_to_raw(fixture, monkeypatch):
    calls = _shared_xor_package(fixture, monkeypatch)
    original_load = loader._load_plan

    def changed_load(*args, **kwargs):
        plan = original_load(*args, **kwargs)
        plan.plan.packets[0]["mode"] = "RAW"
        return plan

    monkeypatch.setattr(loader, "_load_plan", changed_load)
    with pytest.raises(RuntimeError, match="loaded reverse package contains nonreversible"):
        prepare(fixture)
    assert len(calls) == 1
    assert not fixture.model._relax_adp_teacher_state.prepared


@pytest.mark.parametrize("value", [1, 0, "true", None, []])
def test_reverse_requires_explicit_boolean_type(fixture, monkeypatch, value):
    calls = _shared_xor_package(fixture, monkeypatch)
    fixture.manifest["edges"][1]["reverse"] = value
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    with pytest.raises(RuntimeError, match="reverse must be a boolean"):
        prepare(fixture)
    assert calls == []


@pytest.mark.parametrize(
    "corruption",
    ["source", "target", "runtime", "tp_rank", "tp_size", "a_hash", "b_hash", "forward_flag", "missing_flag"],
)
def test_shared_reverse_requires_original_provenance_hashes_and_logical_direction(fixture, monkeypatch, corruption):
    calls = _shared_xor_package(fixture, monkeypatch)
    path = fixture.paths["A", "B"]
    header = loader._package_header(path)
    if corruption in ("source", "target", "runtime"):
        header["provenance"][corruption] = "wrong"
    elif corruption in ("tp_rank", "tp_size"):
        header["provenance"][corruption] = 3
    elif corruption in ("a_hash", "b_hash"):
        header["tensors"]["weight"][corruption.replace("_hash", "_sha256")] = "0" * 64
    elif corruption == "forward_flag":
        fixture.manifest["edges"][0]["reverse"] = True
    else:
        del fixture.manifest["edges"][1]["reverse"]
    write_header(path, header)
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    with pytest.raises(RuntimeError, match="package (provenance|teacher hash) mismatch"):
        prepare(fixture)
    assert calls == []


def test_legacy_explicit_false_keeps_independent_directional_plans(fixture):
    for edge, mode in zip(fixture.manifest["edges"], ("RAW", "ADD_ANS")):
        edge["reverse"] = False
        path = fixture.root / edge["packages"][0]
        header = loader._package_header(path)
        header["packets"] = [{"name": "weight", "mode": mode}]
        write_header(path, header)
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    prepare(fixture)
    plans = fixture.model._relax_adp_teacher_state.plans
    assert len(plans) == 2 and plans["A", "B"] is not plans["B", "A"]
    for teacher in ("B", "A"):
        loader.adp_teacher_loader(fixture.model, command("switch", teacher))
        for name, tensor in fixture.model.named_parameters():
            assert torch.equal(_raw(tensor), _raw(fixture.banks[teacher][name]))


def test_shared_all_skip_package_has_no_explicit_skip_packet(fixture, monkeypatch):
    fixture.banks["B"] = {name: tensor.clone() for name, tensor in fixture.banks["A"].items()}
    capture = fixture.root / "captures/B/rank_0.pt"
    snapshot = torch.load(capture, weights_only=True)
    snapshot["params"] = fixture.banks["B"]
    torch.save(snapshot, capture)
    path = fixture.paths["A", "B"]
    header = loader._package_header(path)
    for entry in header["tensors"].values():
        entry["b_sha256"] = entry["a_sha256"]
    write_header(path, header)
    calls = _shared_xor_package(fixture, monkeypatch, ())
    prepare(fixture)
    assert len(calls) == 1 and calls[0][1].plan.packets == []
    for teacher in ("B", "A"):
        loader.adp_teacher_loader(fixture.model, command("switch", teacher))
        for name, tensor in fixture.model.named_parameters():
            assert torch.equal(_raw(tensor), _raw(fixture.banks[teacher][name]))


@pytest.mark.parametrize("failure", ["binding", "unknown_teacher", "peer_direction"])
def test_shared_reverse_preflight_failure_never_rearms_or_applies(fixture, monkeypatch, failure):
    calls = _shared_xor_package(fixture, monkeypatch)
    prepare(fixture)
    loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    plan = calls[0][1]
    teacher = "A"
    if failure == "binding":
        fixture.model.weight = torch.nn.Parameter(fixture.model.weight.detach().clone(), requires_grad=False)
    elif failure == "unknown_teacher":
        teacher = "C"
    else:
        fixture.dist.size = 2

        def other_source(records, local, rank):
            if local["route"][0] == "preflight":
                route = list(local["route"])
                route[3] = "A"
                records[1]["route"] = tuple(route)

        fixture.dist.peer_override = other_source
    with pytest.raises(RuntimeError):
        loader.adp_teacher_loader(fixture.model, command("switch", teacher))
    assert plan.resets == [False] and plan.applies == [True]
    assert fixture.model._relax_adp_teacher_state.poisoned
    assert fixture.model._relax_adp_teacher_state.current is None


def test_shared_reverse_partial_failure_poisons_both_logical_directions(fixture, monkeypatch):
    calls = _shared_xor_package(fixture, monkeypatch)
    prepare(fixture)
    loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    plan = calls[0][1]
    apply = plan.apply

    def fail_after_write(*, overlap):
        apply(overlap=overlap)
        raise RuntimeError("partial shared reverse write")

    monkeypatch.setattr(plan, "apply", fail_after_write)
    with pytest.raises(RuntimeError, match="partial shared reverse write"):
        loader.adp_teacher_loader(fixture.model, command("switch", "A"))
    state = fixture.model._relax_adp_teacher_state
    assert state.poisoned and state.current is None
    for teacher in ("A", "B"):
        with pytest.raises(RuntimeError, match="state is unknown"):
            loader.adp_teacher_loader(fixture.model, command("switch", teacher))
    assert plan.resets == [False, False] and plan.applies == [True, True]


def test_shared_reverse_restores_from_opposite_direction_anchor(colocated, monkeypatch):
    fixture = colocated
    calls = _shared_xor_package(fixture, monkeypatch)
    fixture.manifest["initial_teacher"] = "B"
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    with torch.no_grad():
        for name, tensor in fixture.model.named_parameters():
            tensor.copy_(fixture.banks["B"][name])
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "B"))
    state = fixture.model._relax_adp_teacher_state
    assert state.base_teacher == "B" and len(calls) == 1
    for teacher in ("A", "B", "A"):
        _wipe(fixture.model)
        loader.adp_teacher_loader(fixture.model, command("restore", teacher))
        for name, tensor in fixture.model.named_parameters():
            assert torch.equal(_raw(tensor), _raw(fixture.banks[teacher][name]))
    assert calls[0][1].resets == [False, False]
    assert state.package_bank_bytes == 32 and state.recovery_cpu_bytes == 60


def test_five_teachers_keep_twenty_routes_but_only_ten_physical_plans(fixture, monkeypatch):
    calls = _shared_xor_package(fixture, monkeypatch)
    for index, teacher in enumerate(("C", "D", "E"), start=2):
        with torch.no_grad():
            for name, tensor in fixture.model.named_parameters():
                tensor.copy_(fixture.banks["A"][name])
            fixture.model.weight.add_(index)
        fixture.banks[teacher] = {name: tensor.detach().clone() for name, tensor in fixture.model.named_parameters()}
        loader.adp_teacher_loader(fixture.model, command("capture", teacher))
    with torch.no_grad():
        for name, tensor in fixture.model.named_parameters():
            tensor.copy_(fixture.banks["A"][name])
    names = sorted(fixture.banks)
    fixture.manifest["teachers"] = {name: {"captures": [f"captures/{name}/rank_0.pt"]} for name in names}
    fixture.manifest["edges"] = []
    prototype = loader._package_header(fixture.paths["A", "B"])
    for index, source in enumerate(names):
        for target in names[index + 1 :]:
            path = fixture.root / f"{source}_{target}.adp"
            header = copy.deepcopy(prototype)
            header["provenance"].update(source=source, target=target)
            for name, entry in header["tensors"].items():
                entry["a_sha256"] = loader._hash_tensor(fixture.banks[source][name])
                entry["b_sha256"] = loader._hash_tensor(fixture.banks[target][name])
            write_header(path, header)
            fixture.manifest["edges"].extend(
                [
                    {"source": source, "target": target, "packages": [path.name]},
                    {"source": target, "target": source, "packages": [path.name], "reverse": True},
                ]
            )
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    prepare(fixture)
    state = fixture.model._relax_adp_teacher_state
    assert len(state.plans) == 20 and len(calls) == 10
    assert len({id(plan) for plan in state.plans.values()}) == 10
    assert state.package_bank_bytes == 10 * 32
    for source, target in state.plans:
        assert state.plans[source, target] is state.plans[target, source]
        for teacher in (source, target):
            loader.adp_teacher_loader(fixture.model, command("switch", teacher))
            for name, tensor in fixture.model.named_parameters():
                assert torch.equal(_raw(tensor), _raw(fixture.banks[teacher][name]))


def test_capture_contains_aliases_auxiliary_and_cpu_snapshots(fixture) -> None:
    snapshot = torch.load(fixture.root / "captures/A/rank_0.pt", weights_only=True)
    assert snapshot["runtime"] == "sglang-dense-v1"
    assert (snapshot["teacher_id"], snapshot["tp_rank"], snapshot["tp_size"]) == ("A", 0, 1)
    assert snapshot["metadata"]["aliases"] == {"tied": "weight"}
    assert set(snapshot["auxiliary"]) == {"buffer:cos_sin_cache", 'attr::extra["frequencies"]'}
    assert all(tensor.device.type == "cpu" for tensor in snapshot["params"].values())
    assert snapshot["params"]["weight"].data_ptr() != fixture.model.weight.data_ptr()


def test_repeated_directed_switches_preserve_all_storage_without_io_or_hashing(fixture, monkeypatch) -> None:
    prepare(fixture)
    bindings = loader._runtime_view(fixture.model).bindings

    def forbidden(*args, **kwargs):
        raise AssertionError("switch performed offline work")

    monkeypatch.setattr(loader, "_hash_tensor", forbidden)
    monkeypatch.setattr(loader, "_root_directory", forbidden)
    monkeypatch.setattr(loader, "_package_header", forbidden)
    monkeypatch.setattr(loader, "_load_plan", forbidden)
    for teacher in ("B", "A", "A", "B"):
        loader.adp_teacher_loader(fixture.model, command("switch", teacher))
        assert fixture.model._relax_adp_teacher_state.current == teacher
        assert loader._runtime_view(fixture.model).bindings == bindings
        for name, tensor in fixture.model.named_parameters():
            assert torch.equal(tensor, fixture.banks[teacher][name])
    assert fixture.plans["A", "B"].resets == [False, False]
    assert fixture.plans["B", "A"].resets == [False]
    assert fixture.plans["A", "B"].applies == [True, True]


def test_prepare_does_not_retain_full_capture_tensors(fixture, monkeypatch) -> None:
    references = []
    original = torch.load

    def track(*args, **kwargs):
        snapshot = original(*args, **kwargs)
        references.extend(
            weakref.ref(tensor) for section in ("params", "auxiliary") for tensor in snapshot[section].values()
        )
        return snapshot

    monkeypatch.setattr(torch, "load", track)
    prepare(fixture)
    gc.collect()
    assert references and all(reference() is None for reference in references)


@pytest.mark.parametrize("which", ["parameter", "auxiliary", "config", "alias"])
def test_storage_or_layout_drift_fails_before_writes_and_poison_is_permanent(fixture, which: str) -> None:
    prepare(fixture)
    if which == "parameter":
        fixture.model.scale = torch.nn.Parameter(fixture.model.scale.detach().clone(), requires_grad=False)
    elif which == "auxiliary":
        fixture.model.cos_sin_cache = fixture.model.cos_sin_cache.clone()
    elif which == "config":
        fixture.model.config["hidden_size"] += 1
    else:
        fixture.model.tied = fixture.model.scale
    with pytest.raises(RuntimeError, match="layout or storage changed"):
        loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    state = fixture.model._relax_adp_teacher_state
    assert state.current is None and state.poisoned
    assert fixture.plans["A", "B"].applies == []
    with pytest.raises(RuntimeError, match="unknown; engine restart"):
        loader.adp_teacher_loader(fixture.model, command("prepare", "A"))


def test_partial_write_failure_poison_prevents_any_later_operation(fixture) -> None:
    prepare(fixture)
    fixture.plans["A", "B"].fail = True
    with pytest.raises(RuntimeError, match="partial write failure"):
        loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    state = fixture.model._relax_adp_teacher_state
    assert state.current is None and state.poisoned
    for operation in ("switch", "prepare", "capture"):
        with pytest.raises(RuntimeError, match="engine restart/reload"):
            loader.adp_teacher_loader(fixture.model, command(operation, "A"))


@pytest.mark.parametrize("phase", ["preflight", "commit"])
@pytest.mark.parametrize("rank", [0, 1])
def test_peer_failure_poison_is_agreed_with_explicit_tp_cpu_group(fixture, phase: str, rank: int) -> None:
    prepare(fixture)
    fixture.dist.size, fixture.dist.rank = 2, rank

    def peer_failure(records, value, local_rank):
        if value["route"][0] == phase:
            records[1 - local_rank]["error"] = "injected peer failure"

    fixture.dist.peer_override = peer_failure
    with pytest.raises(RuntimeError, match="injected peer failure"):
        loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    state = fixture.model._relax_adp_teacher_state
    assert state.current is None and state.poisoned
    assert len(fixture.plans["A", "B"].applies) == (phase == "commit")


def test_tp_command_disagreement_never_starts_switch(fixture) -> None:
    prepare(fixture)
    fixture.dist.size = 2

    def other_teacher(records, value, rank):
        peer = list(value["route"])
        peer[2] = "A"
        records[1]["route"] = tuple(peer)

    fixture.dist.peer_override = other_teacher
    with pytest.raises(RuntimeError, match="disagree on command"):
        loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    assert fixture.plans["A", "B"].applies == []
    assert fixture.model._relax_adp_teacher_state.poisoned


@pytest.mark.parametrize(
    "corruption", ["skip_coverage", "skip_base", "target_hash", "reference_runtime", "missing_edge", "auxiliary"]
)
def test_prepare_rejects_incomplete_or_incompatible_packages_before_loading(fixture, corruption: str) -> None:
    path = fixture.paths["A", "B"]
    header = loader._package_header(path)
    if corruption == "skip_coverage":
        del header["tensors"]["scale"]
    elif corruption == "skip_base":
        with torch.no_grad():
            fixture.model.scale.add_(1)
    elif corruption == "target_hash":
        header["tensors"]["weight"]["b_sha256"] = "0" * 64
    elif corruption == "reference_runtime":
        header["provenance"]["runtime"] = "official-deepseek-reference"
    elif corruption == "missing_edge":
        fixture.manifest["edges"].pop()
        (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    else:
        capture = fixture.root / "captures/B/rank_0.pt"
        snapshot = torch.load(capture, weights_only=True)
        snapshot["auxiliary"]["buffer:cos_sin_cache"].add_(1)
        torch.save(snapshot, capture)
    write_header(path, header)
    with pytest.raises(RuntimeError):
        prepare(fixture)
    assert not fixture.model._relax_adp_teacher_state.prepared
    assert fixture.plans == {}


@pytest.mark.parametrize(
    "control", [[], [("relax_adp:switch:B", torch.ones(1))], [("relax_adp:switch:../B", torch.empty(0))]]
)
def test_command_protocol_rejects_nonempty_or_unsafe_controls(fixture, control) -> None:
    with pytest.raises(RuntimeError):
        loader.adp_teacher_loader(fixture.model, control)


def test_runtime_rejects_quantization_and_overlapping_parameters(fixture) -> None:
    fixture.model.quant_method = SimpleNamespace()
    with pytest.raises(ValueError, match="quantized/unknown"):
        loader._runtime_view(fixture.model)
    del fixture.model.quant_method
    fixture.model.overlap = torch.nn.Parameter(fixture.model.weight.detach(), requires_grad=False)
    with pytest.raises(ValueError, match="overlap"):
        loader._runtime_view(fixture.model)


def test_production_device_gate_rejects_cpu_parameters() -> None:
    with pytest.raises(ValueError, match="CUDA resident"):
        loader._check_parameter("cpu", torch.zeros(1))


def test_load_plan_uses_lazy_codec_imports_without_retaining_cpu_base(monkeypatch, tmp_path) -> None:
    calls = []
    destinations = {"weight": torch.ones(1)}
    inner = object()

    def load(path, dst, *, a_bank, verify_base):
        calls.append((path, dst, a_bank, verify_base))
        return inner

    def batch(plan, *, group_bytes, compact_sources):
        assert plan is inner
        assert group_bytes == 64 * 1024**2 and compact_sources is True
        return "wrapped"

    monkeypatch.setitem(
        sys.modules, f"{_ADP_PACKAGE}.adaptive_delta", SimpleNamespace(AdaptiveDeltaPlan=SimpleNamespace(load=load))
    )
    monkeypatch.setitem(sys.modules, f"{_ADP_PACKAGE}.batched_delta", SimpleNamespace(BatchedDeltaPlan=batch))
    path = tmp_path / "directed.adp"
    assert loader._load_plan(path, destinations) == "wrapped"
    assert calls == [(path, destinations, None, False)]


def _add_integrity(fixture):
    fixture.manifest["package_integrity"] = {
        path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        for path in set(fixture.paths.values())
    }
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))


def test_strict_package_integrity_is_passed_to_every_unique_plan(fixture, monkeypatch):
    _add_integrity(fixture)
    original = loader._load_plan
    calls = []

    def checked(path, destinations, *, expected_sha256):
        assert expected_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        calls.append(path)
        return original(path, destinations)

    monkeypatch.setattr(loader, "_load_plan", checked)
    prepare(fixture)
    assert set(calls) == set(fixture.paths.values())


def test_integrity_reverse_alias_keeps_one_physical_load(fixture, monkeypatch):
    calls = _shared_xor_package(fixture, monkeypatch)
    path = fixture.paths["A", "B"]
    fixture.manifest["package_integrity"] = {
        path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    }
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    original = loader._load_plan

    def checked(path, destinations, *, expected_sha256):
        assert expected_sha256 == fixture.manifest["package_integrity"][path.name]["sha256"]
        return original(path, destinations)

    monkeypatch.setattr(loader, "_load_plan", checked)
    prepare(fixture)
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["missing", "extra", "bad_hash", "bad_size", "bool_size", "null"])
def test_invalid_integrity_rejects_before_loading_plans(fixture, monkeypatch, damage):
    _add_integrity(fixture)
    integrity = fixture.manifest["package_integrity"]
    record = integrity["A_B.adp"]
    if damage == "missing":
        del integrity["A_B.adp"]
    elif damage == "extra":
        integrity["extra.adp"] = record
    elif damage == "bad_hash":
        record["sha256"] = "x" * 64
    elif damage == "bad_size":
        record["bytes"] += 1
    elif damage == "bool_size":
        record["bytes"] = True
    else:
        fixture.manifest["package_integrity"] = None
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    monkeypatch.setattr(loader, "_load_plan", lambda *args, **kwargs: pytest.fail("allocated invalid package"))
    with pytest.raises(RuntimeError, match="integrity"):
        prepare(fixture)
    assert not getattr(fixture.model, loader._STATE_ATTRIBUTE).prepared


@pytest.mark.parametrize("damage", [None, "incomplete", "count", "missing", "extra", "hash"])
def test_complete_progress_integrity_migration_only(fixture, monkeypatch, damage):
    _add_integrity(fixture)
    integrity = fixture.manifest.pop("package_integrity")
    (fixture.root / "manifest.json").write_text(json.dumps(fixture.manifest))
    progress = {
        "status": "complete",
        "completed_edges": 2,
        "packages": {
            name: {"file_sha256": entry["sha256"], "file_bytes": entry["bytes"]} for name, entry in integrity.items()
        },
    }
    if damage == "incomplete":
        progress["status"] = "building"
    elif damage == "count":
        progress["completed_edges"] = 1
    elif damage == "missing":
        del progress["packages"]["B_A.adp"]
    elif damage == "extra":
        progress["packages"]["extra.adp"] = progress["packages"]["A_B.adp"]
    elif damage == "hash":
        progress["packages"]["A_B.adp"]["file_sha256"] = "invalid"
    (fixture.root / "build_progress.json").write_text(json.dumps(progress))
    original = loader._load_plan
    calls = []

    def checked(path, destinations, *, expected_sha256):
        calls.append(path)
        assert expected_sha256 == integrity[path.name]["sha256"]
        return original(path, destinations)

    monkeypatch.setattr(loader, "_load_plan", checked)
    if damage is None:
        prepare(fixture)
        assert len(calls) == 2
    else:
        with pytest.raises(RuntimeError, match="integrity|complete build"):
            prepare(fixture)
        assert calls == []


def test_package_digest_failure_never_commits_a_prepared_state(fixture, monkeypatch):
    _add_integrity(fixture)

    def corrupted(*args, **kwargs):
        raise ValueError("ADP1 complete-file SHA256 mismatch")

    monkeypatch.setattr(loader, "_load_plan", corrupted)
    with pytest.raises(RuntimeError, match="complete-file SHA256"):
        prepare(fixture)
    state = getattr(fixture.model, loader._STATE_ATTRIBUTE)
    assert state.current is None and not state.prepared and not state.plans
    with pytest.raises(RuntimeError, match="prepare is required"):
        loader.adp_teacher_loader(fixture.model, command("switch", "B"))


def test_load_plan_forwards_complete_file_digest(monkeypatch, tmp_path):
    expected = "a" * 64
    calls = []

    def load(path, destinations, **kwargs):
        calls.append(kwargs)
        return "inner"

    monkeypatch.setitem(
        sys.modules, f"{_ADP_PACKAGE}.adaptive_delta", SimpleNamespace(AdaptiveDeltaPlan=SimpleNamespace(load=load))
    )
    monkeypatch.setitem(
        sys.modules, f"{_ADP_PACKAGE}.batched_delta", SimpleNamespace(BatchedDeltaPlan=lambda *args, **kwargs: "outer")
    )
    assert loader._load_plan(tmp_path / "x.adp", {}, expected_sha256=expected) == "outer"
    assert calls == [{"a_bank": None, "verify_base": False, "expected_sha256": expected}]


@pytest.fixture
def source_modules(tmp_path, monkeypatch):
    bodies = {
        "xor_codec": "class GPUCodec: pass\n",
        "adaptive_delta": "from .xor_codec import GPUCodec\nclass AdaptiveDeltaPlan: pass\n",
        "batched_delta": "class BatchedDeltaPlan: pass\n",
    }
    manifest = {"algorithm_sources": {}}
    for name, body in bodies.items():
        monkeypatch.delitem(sys.modules, f"{_ADP_PACKAGE}.{name}", raising=False)
        path = tmp_path / f"{name}.py"
        path.write_text(body)
        manifest["algorithm_sources"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(adp, "__path__", [str(tmp_path)])
    monkeypatch.setattr(loader, "_ATTESTED_ALGORITHM_MODULES", {})
    yield manifest, tmp_path
    # import_module inserts outside monkeypatch; remove before restoring callers.
    for name in bodies:
        sys.modules.pop(f"{_ADP_PACKAGE}.{name}", None)


def test_strict_source_first_import_and_known_object_retry(source_modules):
    manifest, root = source_modules
    first = loader._attest_algorithm_sources(manifest)
    assert set(first) == set(loader._ALGORITHM_MODULES)
    for name, actual in first.items():
        assert actual == {"origin": str(root / f"{name}.py"), **manifest["algorithm_sources"][name]}
    assert loader._attest_algorithm_sources(manifest) == first


@pytest.mark.parametrize("damage", ["wrong_hash", "missing", "extra", "preloaded", "shadow"])
def test_strict_source_rejects_untrusted_import_before_execution(source_modules, monkeypatch, damage):
    manifest, root = source_modules
    if damage == "wrong_hash":
        manifest["algorithm_sources"]["adaptive_delta"]["sha256"] = "0" * 64
    elif damage == "missing":
        del manifest["algorithm_sources"]["adaptive_delta"]
    elif damage == "extra":
        manifest["algorithm_sources"]["other"] = {"sha256": "0" * 64}
    elif damage == "preloaded":
        # Even a module pointing to the expected current source is not proof of
        # which bytecode was imported before an in-place source update.
        monkeypatch.setitem(
            sys.modules, f"{_ADP_PACKAGE}.adaptive_delta", SimpleNamespace(__file__=str(root / "adaptive_delta.py"))
        )
    else:
        shadow = root / "shadow"
        shadow.mkdir()
        (shadow / "adaptive_delta.py").write_text("raise AssertionError('must never execute')\n")
        monkeypatch.setattr(adp, "__path__", [str(shadow), str(root)])
    with pytest.raises(ValueError, match="SHA256|exactly|unattested"):
        loader._attest_algorithm_sources(manifest)
    assert f"{_ADP_PACKAGE}.xor_codec" not in sys.modules


@pytest.mark.parametrize("damage", ["object", "origin", "content", "expected_version"])
def test_known_source_cache_binds_object_origin_and_expected_hash(source_modules, monkeypatch, damage):
    manifest, root = source_modules
    loader._attest_algorithm_sources(manifest)
    module = sys.modules[f"{_ADP_PACKAGE}.adaptive_delta"]
    if damage == "object":
        monkeypatch.setitem(sys.modules, f"{_ADP_PACKAGE}.adaptive_delta", SimpleNamespace(__file__=module.__file__))
    elif damage == "origin":
        other = root / "copy.py"
        other.write_bytes((root / "adaptive_delta.py").read_bytes())
        monkeypatch.setattr(module, "__file__", str(other))
    elif damage == "content":
        (root / "adaptive_delta.py").write_text("class AdaptiveDeltaPlan: pass\n")
    else:
        manifest["algorithm_sources"]["adaptive_delta"]["sha256"] = "1" * 64
    with pytest.raises(ValueError, match="unattested|origin changed|SHA256"):
        loader._attest_algorithm_sources(manifest)


def test_strict_prepare_records_actual_sources_and_switch_does_no_source_io(fixture, source_modules, monkeypatch):
    sources, _ = source_modules
    fixture.manifest.update(sources)
    _add_integrity(fixture)
    original = loader._load_plan

    def checked(path, destinations, *, expected_sha256):
        assert expected_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        return original(path, destinations)

    monkeypatch.setattr(loader, "_load_plan", checked)
    event_path = fixture.root / "events.jsonl"
    monkeypatch.setenv("RELAX_ADP_EVENT_LOG", str(event_path))
    prepare(fixture)
    state = getattr(fixture.model, loader._STATE_ATTRIBUTE)
    event = json.loads(event_path.read_text().splitlines()[0])
    assert set(state.algorithm_sources) == set(loader._ALGORITHM_MODULES)
    assert event["algorithm_sources"] == state.algorithm_sources
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("online source/package read"))
    monkeypatch.setattr(loader, "_attest_algorithm_sources", lambda *args: pytest.fail("online source import"))
    loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    assert state.current == "B"


def test_unexpected_bound_codec_class_rejected(source_modules, monkeypatch):
    manifest, _ = source_modules
    loader._attest_algorithm_sources(manifest)
    monkeypatch.setattr(sys.modules[f"{_ADP_PACKAGE}.adaptive_delta"], "GPUCodec", object)
    with pytest.raises(ValueError, match="classes do not originate"):
        loader._attest_algorithm_sources(manifest)


def test_import_that_changes_actual_origin_is_not_attested(source_modules, monkeypatch):
    manifest, root = source_modules
    original = loader.importlib.import_module
    moved = root / "moved.py"
    moved.write_bytes((root / "xor_codec.py").read_bytes())

    def changed(name):
        module = original(name)
        if name == f"{_ADP_PACKAGE}.xor_codec":
            module.__file__ = str(moved)
        return module

    monkeypatch.setattr(loader.importlib, "import_module", changed)
    with pytest.raises(ValueError, match="origin changed during import"):
        loader._attest_algorithm_sources(manifest)
    assert not loader._ATTESTED_ALGORITHM_MODULES


def _capture_with_parameter_views(fixture) -> None:
    model = fixture.model
    # Qwen3.5 keeps conv1d.weight.view(...) in RadixLinearAttention.
    model.conv_weights = model.weight.view(3, 2)
    model.nested_views = {"transpose": model.weight.transpose(0, 1)}
    model.register_buffer("weight_slice", model.weight[:, 1:])
    for teacher in ("A", "B"):
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                tensor.copy_(fixture.banks[teacher][name])
        loader.adp_teacher_loader(model, command("capture", teacher))
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            tensor.copy_(fixture.banks["A"][name])


def test_parameter_views_are_captured_as_layouts_and_follow_repeated_switches(fixture, monkeypatch) -> None:
    _capture_with_parameter_views(fixture)
    snapshot = torch.load(fixture.root / "captures/A/rank_0.pt", weights_only=True)
    views = snapshot["metadata"]["parameter_views"]
    assert set(views) == {"attr::conv_weights", 'attr::nested_views["transpose"]', "buffer:weight_slice"}
    assert views["attr::conv_weights"] == {
        "parameter": "weight",
        "offset_bytes": 0,
        "span_bytes": 12,
        "dtype": "torch.bfloat16",
        "shape": [3, 2],
        "stride": [2, 1],
    }
    assert views['attr::nested_views["transpose"]']["stride"] == [1, 3]
    assert views["buffer:weight_slice"]["offset_bytes"] == 2
    assert views["buffer:weight_slice"]["span_bytes"] == 10
    assert not set(views) & set(snapshot["auxiliary"])
    prepare(fixture)
    bindings = loader._runtime_view(fixture.model).bindings

    def forbidden(*args, **kwargs):
        raise AssertionError("parameter-view switch must not read captures or hash tensor contents")

    monkeypatch.setattr(loader, "_hash_tensor", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    for teacher in ("B", "A", "B"):
        loader.adp_teacher_loader(fixture.model, command("switch", teacher))
        expected = fixture.banks[teacher]["weight"]
        assert torch.equal(fixture.model.conv_weights, expected.view(3, 2))
        assert torch.equal(fixture.model.nested_views["transpose"], expected.transpose(0, 1))
        assert torch.equal(fixture.model.weight_slice, expected[:, 1:])
        assert loader._runtime_view(fixture.model).bindings == bindings


def test_equal_clone_is_still_auxiliary_and_varying_independent_values_are_rejected(fixture) -> None:
    _capture_with_parameter_views(fixture)
    fixture.model.independent = fixture.model.weight.detach().clone()
    runtime = loader._runtime_view(fixture.model)
    assert "attr::independent" in runtime.auxiliary
    assert "attr::independent" not in runtime.metadata["parameter_views"]
    del fixture.model.independent
    capture = fixture.root / "captures/B/rank_0.pt"
    snapshot = torch.load(capture, weights_only=True)
    snapshot["auxiliary"]["buffer:cos_sin_cache"].add_(1)
    torch.save(snapshot, capture)
    with pytest.raises(RuntimeError, match="auxiliary tensor values differ"):
        prepare(fixture)
    assert fixture.plans == {}


@pytest.mark.parametrize("replacement", ["clone", "offset", "transpose", "new_identity"])
def test_parameter_view_binding_drift_rejects_switch_before_any_write(fixture, replacement: str) -> None:
    _capture_with_parameter_views(fixture)
    prepare(fixture)
    model = fixture.model
    if replacement == "clone":
        model.conv_weights = model.conv_weights.clone()
    elif replacement == "offset":
        model.conv_weights = model.weight.reshape(-1)[1:]
    elif replacement == "transpose":
        model.conv_weights = model.conv_weights.transpose(0, 1)
    else:
        model.conv_weights = model.conv_weights.view_as(model.conv_weights)
    with pytest.raises(RuntimeError, match="layout or storage changed"):
        loader.adp_teacher_loader(model, command("switch", "B"))
    assert fixture.plans["A", "B"].applies == []
    assert model._relax_adp_teacher_state.poisoned


def test_alias_offset_is_relative_to_parameter_not_entire_storage(fixture) -> None:
    model = torch.nn.Module()
    backing = torch.arange(20, dtype=torch.float32)
    model.weight = torch.nn.Parameter(backing[4:12].reshape(2, 4), requires_grad=False)
    model.view = model.weight[:, 1::2]
    model.scalar = model.weight[0, 2]
    model.broadcast = model.weight[:1, :1].expand(3, 5)
    model.empty = model.weight[:0]
    view = loader._runtime_view(model)
    assert view.metadata["parameter_views"]["attr::view"] == {
        "parameter": "weight",
        "offset_bytes": 4,
        "span_bytes": 28,
        "dtype": "torch.float32",
        "shape": [2, 2],
        "stride": [4, 2],
    }
    assert view.metadata["parameter_views"]["attr::scalar"]["offset_bytes"] == 8
    assert view.metadata["parameter_views"]["attr::broadcast"]["span_bytes"] == 4
    # Empty tensors have no parameter bytes to restore and retain the ordinary
    # auxiliary metadata/binding checks; they are never proof of coverage.
    assert "attr::empty" in view.auxiliary


@pytest.mark.parametrize("kind", ["before", "after", "cross_parameter", "dtype", "lazy_negative"])
def test_unsafe_shared_storage_is_not_silently_excluded_from_auxiliary(fixture, kind: str) -> None:
    model = torch.nn.Module()
    backing = torch.arange(20, dtype=torch.float32)
    model.weight = torch.nn.Parameter(backing[4:12], requires_grad=False)
    if kind == "before":
        model.view = backing[3:6]
    elif kind == "after":
        model.view = backing[10:14]
    elif kind == "cross_parameter":
        model.other = torch.nn.Parameter(backing[12:16], requires_grad=False)
        model.view = backing[10:14]
    elif kind == "dtype":
        model.view = model.weight.view(torch.uint8)
    else:
        model.view = torch._neg_view(model.weight)
    with pytest.raises(ValueError, match="parameter view"):
        loader._runtime_view(model)


def test_models_without_parameter_views_keep_original_capture_metadata(fixture) -> None:
    assert "parameter_views" not in loader._runtime_view(fixture.model).metadata


@pytest.fixture
def colocated(fixture, monkeypatch):
    # Exercise real copies and byte/layout checks on CPU. Only allocation's
    # pin gate and phase synchronization are stubbed; no CUDA context is made.
    monkeypatch.setattr(
        loader, "_allocate_pinned", lambda tensor: torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu")
    )
    monkeypatch.setattr(loader, "_require_pinned", lambda tensor: None)
    fixture.syncs = []
    monkeypatch.setattr(loader, "_synchronize", lambda view: fixture.syncs.append(True))
    return fixture


def _raw(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)


def _wipe(model):
    view = loader._runtime_view(model)
    with torch.no_grad():
        for tensor in (*view.params.values(), *view.auxiliary.values()):
            tensor.zero_()


def test_colocated_restores_base_target_and_cpu_auxiliary_without_reads_hashes_or_rebinding(colocated, monkeypatch):
    fixture = colocated
    _capture_with_parameter_views(fixture)
    original = loader._runtime_view(fixture.model)
    auxiliary = {name: _raw(tensor).clone() for name, tensor in original.auxiliary.items()}
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    state = fixture.model._relax_adp_teacher_state
    assert state.base_teacher == "A"
    assert state.recovery_parameter_bytes == 6 * 2 + 2 * 4
    assert state.recovery_auxiliary_bytes == 8 * 4 + 2 * 4
    assert state.recovery_cpu_bytes == 60
    assert state.package_bank_bytes == 0
    assert set(state.recovery_auxiliary) == set(original.auxiliary)
    for name, snapshot in state.recovery_params.items():
        assert snapshot.device.type == "cpu"
        assert snapshot.data_ptr() != original.params[name].data_ptr()

    def forbidden(*args, **kwargs):
        raise AssertionError("online restore performed filesystem reads or hashing")

    for name in ("_hash_tensor", "_root_directory", "_package_header", "_load_plan", "_prepare"):
        monkeypatch.setattr(loader, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(loader.hashlib, "sha256", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    # Same-target restore is required: logical teacher identity survived the wipe.
    for target in ("B", "B", "A", "A", "B"):
        with torch.no_grad():
            for tensor in (*original.params.values(), *original.auxiliary.values()):
                tensor.zero_()
        loader.adp_teacher_loader(fixture.model, command("restore", target))
        after = loader._runtime_view(fixture.model, prepared=state)
        assert after.bindings == original.bindings and state.current == target
        for name, tensor in after.params.items():
            assert torch.equal(_raw(tensor), _raw(fixture.banks[target][name]))
        for name, tensor in after.auxiliary.items():
            assert torch.equal(_raw(tensor), auxiliary[name])
        assert torch.equal(fixture.model.conv_weights, fixture.banks[target]["weight"].view(3, 2))
        assert torch.equal(fixture.model.weight_slice, fixture.banks[target]["weight"][:, 1:])
    assert fixture.plans["A", "B"].resets == [False] * 3
    assert fixture.plans["B", "A"].applies == []
    assert len(fixture.syncs) == 1 + 2 * 5  # One D2H phase, two gates per restore, never per tensor.


def test_resident_prepare_keeps_no_recovery_anchor_and_restore_is_rejected(fixture):
    prepare(fixture)
    state = fixture.model._relax_adp_teacher_state
    assert state.base_teacher is None
    assert not state.recovery_params and not state.recovery_auxiliary
    assert state.recovery_cpu_bytes == 0
    with pytest.raises(RuntimeError, match="prepare_colocated is required"):
        loader.adp_teacher_loader(fixture.model, command("restore", "B"))
    assert state.poisoned and state.current is None


def test_colocated_prepare_failure_to_pin_has_no_pageable_fallback(fixture, monkeypatch):
    monkeypatch.setattr(loader, "_allocate_pinned", lambda tensor: torch.empty_like(tensor))
    with pytest.raises(RuntimeError, match="not pinned"):
        loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    state = fixture.model._relax_adp_teacher_state
    assert not state.prepared and not state.recovery_params


@pytest.mark.parametrize(
    "dtype,integer,codes",
    [
        (torch.bfloat16, torch.int16, [0, -32768, 32705, -61, 16256, -16512]),
        (torch.float32, torch.int32, [0, -2147483648, 2143289345, -4194243, 1065353216, -1082130432]),
    ],
)
def test_restore_preserves_negative_zero_nan_payloads_and_noncontiguous_auxiliary(colocated, dtype, integer, codes):
    fixture = colocated
    model = fixture.model
    values = torch.tensor(codes, dtype=integer).view(dtype).reshape(2, 3)
    model.payload = values.transpose(0, 1).clone(memory_format=torch.preserve_format)
    # Recapture the genuine preparation inputs with the unusual independent auxiliary.
    for teacher in ("A", "B"):
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                tensor.copy_(fixture.banks[teacher][name])
        loader.adp_teacher_loader(model, command("capture", teacher))
    with torch.no_grad():
        for name, tensor in model.named_parameters():
            tensor.copy_(fixture.banks["A"][name])
    expected = _raw(model.payload).clone()
    binding = loader._binding(model.payload)
    loader.adp_teacher_loader(model, command("prepare_colocated", "A"))
    _wipe(model)
    loader.adp_teacher_loader(model, command("restore", "B"))
    assert loader._binding(model.payload) == binding
    assert torch.equal(_raw(model.payload), expected)
    loader.adp_teacher_loader(model, command("verify", "B"))
    # A NaN payload bit flip still looks like NaN, but must fail byte verification.
    with torch.no_grad():
        model.payload.view(integer)[2, 0] ^= 1
    with pytest.raises(RuntimeError, match="auxiliary bytes"):
        loader.adp_teacher_loader(model, command("verify", "B"))
    assert model._relax_adp_teacher_state.poisoned


@pytest.mark.parametrize("drift", ["parameter", "auxiliary", "view", "config", "snapshot", "unknown"])
def test_restore_rejects_drift_before_writing_and_poison_is_permanent(colocated, drift):
    fixture = colocated
    _capture_with_parameter_views(fixture)
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    _wipe(fixture.model)
    state = fixture.model._relax_adp_teacher_state
    target = "B"
    if drift == "parameter":
        fixture.model.scale = torch.nn.Parameter(fixture.model.scale.detach().clone(), requires_grad=False)
    elif drift == "auxiliary":
        fixture.model.cos_sin_cache = fixture.model.cos_sin_cache.clone()
    elif drift == "view":
        fixture.model.conv_weights = fixture.model.conv_weights.view_as(fixture.model.conv_weights)
    elif drift == "config":
        fixture.model.config["hidden_size"] = 99
    elif drift == "snapshot":
        state.recovery_params.pop("scale")
    else:
        target = "unknown"
    with pytest.raises(RuntimeError):
        loader.adp_teacher_loader(fixture.model, command("restore", target))
    assert not fixture.model.weight.count_nonzero()
    assert not fixture.plans["A", "B"].applies
    assert state.poisoned and state.current is None
    with pytest.raises(RuntimeError, match="restart/reload"):
        loader.adp_teacher_loader(fixture.model, command("restore", "A"))


def test_partial_restore_copy_failure_drains_phase_and_poison(colocated, monkeypatch):
    fixture = colocated
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    _wipe(fixture.model)
    original = torch.Tensor.copy_

    def broken(tensor, source, **kwargs):
        if tensor is fixture.model.scale:
            raise RuntimeError("injected recovery copy failure")
        return original(tensor, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", broken)
    with pytest.raises(RuntimeError, match="recovery copy failure"):
        loader.adp_teacher_loader(fixture.model, command("restore", "B"))
    assert torch.equal(fixture.model.weight, fixture.banks["A"]["weight"])
    assert fixture.model._relax_adp_teacher_state.poisoned
    assert len(fixture.syncs) == 3
    assert not fixture.plans["A", "B"].applies


def test_partial_delta_failure_during_restore_poison(colocated):
    fixture = colocated
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    fixture.plans["A", "B"].fail = True
    with pytest.raises(RuntimeError, match="partial write failure"):
        loader.adp_teacher_loader(fixture.model, command("restore", "B"))
    assert fixture.model._relax_adp_teacher_state.poisoned


@pytest.mark.parametrize("phase", ["preflight", "commit"])
@pytest.mark.parametrize("disagreement", [False, True])
def test_restore_peer_failure_or_route_disagreement_poison(colocated, phase, disagreement):
    fixture = colocated
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    _wipe(fixture.model)
    fixture.dist.size = 2

    def peer(records, value, rank):
        if value["route"][0] == phase:
            if disagreement:
                route = list(value["route"])
                route[2] = "A"
                records[1]["route"] = tuple(route)
            else:
                records[1]["error"] = "peer restore failed"

    fixture.dist.peer_override = peer
    with pytest.raises(RuntimeError):
        loader.adp_teacher_loader(fixture.model, command("restore", "B"))
    assert fixture.model._relax_adp_teacher_state.poisoned
    assert bool(fixture.plans["A", "B"].applies) == (phase == "commit")


def test_package_bank_bytes_counts_unique_pinned_storage(colocated):
    backing = torch.empty(32, dtype=torch.uint8)
    plans = {("A", "B"): SimpleNamespace(bank=backing[:16]), ("B", "A"): SimpleNamespace(bank=backing[16:])}
    assert loader._package_bank_bytes(plans) == 32


def test_events_separate_anchor_package_bytes_prepare_and_completed_gpu_phase(colocated, monkeypatch):
    fixture = colocated
    path = fixture.root / "events.jsonl"
    monkeypatch.setenv("RELAX_ADP_EVENT_LOG", str(path))
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    loader.adp_teacher_loader(fixture.model, command("switch", "B"))
    _wipe(fixture.model)
    loader.adp_teacher_loader(fixture.model, command("restore", "A"))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["operation"] for event in events] == ["prepare_colocated", "switch", "restore"]
    assert all(event["status"] == "local_complete" for event in events)
    assert all(event["recovery_cpu_bytes"] == 60 and event["base_teacher"] == "A" for event in events)
    assert all(event["package_bank_bytes"] == 0 for event in events)
    assert events[0]["prepare_seconds"] >= 0 and events[0]["synchronized_seconds"] is None
    assert events[1]["synchronized_seconds"] >= 0 and events[1]["prepare_seconds"] is None
    assert events[2]["source"] == "B" and events[2]["target"] == "A"


@pytest.mark.parametrize("operation", ["prepare_colocated", "switch", "restore"])
def test_event_write_failure_poison_prevents_success_ack(colocated, monkeypatch, operation):
    fixture = colocated
    if operation != "prepare_colocated":
        loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    monkeypatch.setenv("RELAX_ADP_EVENT_LOG", str(fixture.root / "missing" / "event.jsonl"))
    with pytest.raises(RuntimeError, match="event write failed"):
        loader.adp_teacher_loader(fixture.model, command(operation, "A" if operation == "prepare_colocated" else "B"))
    assert fixture.model._relax_adp_teacher_state.poisoned


def test_failed_restore_records_error_without_suppressing_original_failure(colocated, monkeypatch):
    fixture = colocated
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    path = fixture.root / "events.jsonl"
    monkeypatch.setenv("RELAX_ADP_EVENT_LOG", str(path))
    fixture.plans["A", "B"].fail = True
    with pytest.raises(RuntimeError, match="partial write failure"):
        loader.adp_teacher_loader(fixture.model, command("restore", "B"))
    event = json.loads(path.read_text())
    assert event["status"] == "failed" and "partial write failure" in event["error"]
    assert event["recovery_cpu_bytes"] == 60
    assert event["synchronized_seconds"] >= 0


def test_verify_is_offline_hash_only_without_capture_reads_or_live_mutation(colocated, monkeypatch):
    fixture = colocated
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    state = fixture.model._relax_adp_teacher_state
    assert set(state.teacher_hashes) == {"A", "B"}
    bindings = loader._runtime_view(fixture.model).bindings
    seen = []
    original_hash = loader._hash_tensor

    def record_hash(tensor):
        seen.append(id(tensor))
        return original_hash(tensor)

    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not read captures or mutate live tensors")

    monkeypatch.setattr(loader, "_hash_tensor", record_hash)
    monkeypatch.setattr(loader, "_root_directory", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.Tensor, "copy_", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    loader.adp_teacher_loader(fixture.model, command("verify", "A"))
    assert len(seen) == 4
    assert state.current == "A" and not state.poisoned
    assert loader._runtime_view(fixture.model).bindings == bindings


def test_prepared_capture_cannot_overwrite_source_snapshots(fixture):
    prepare(fixture)
    path = fixture.root / "captures/A/rank_0.pt"
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="prepared capture is forbidden"):
        loader.adp_teacher_loader(fixture.model, command("capture", "A"))
    assert path.read_bytes() == before
    assert not fixture.model._relax_adp_teacher_state.poisoned


@pytest.mark.parametrize("kind", ["parameter", "auxiliary", "teacher"])
def test_verify_failure_permanently_poison_even_without_binding_drift(fixture, kind):
    prepare(fixture)
    target = "A"
    with torch.no_grad():
        if kind == "parameter":
            fixture.model.weight.view(torch.int16)[0, 0] ^= 1
        elif kind == "auxiliary":
            fixture.model.extra["frequencies"][0] += 1
        else:
            target = "B"
    with pytest.raises(RuntimeError):
        loader.adp_teacher_loader(fixture.model, command("verify", target))
    assert fixture.model._relax_adp_teacher_state.poisoned


@pytest.mark.parametrize(
    "dtype,integer,codes",
    [
        (torch.bfloat16, torch.int16, [0, -32768, 32705, -61, 16256, -16512]),
        (torch.float32, torch.int32, [0, -2147483648, 2143289345, -4194243, 1065353216, -1082130432]),
    ],
)
def test_base_parameter_recovery_is_byte_exact_for_nan_payloads_and_signed_zero(colocated, dtype, integer, codes):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor(codes, dtype=integer).view(dtype), requires_grad=False)
    view = loader._runtime_view(model)
    state = loader._State(prepared=True, current="A", teachers=frozenset({"A"}))
    state.teacher_hashes = {"A": {"weight": loader._hash_tensor(model.weight)}}
    expected = _raw(model.weight).clone()
    binding = loader._binding(model.weight)
    loader._prepare_recovery(view, state, "A")
    with torch.no_grad():
        model.weight.zero_()
    loader._validate_recovery(view, state, "A")
    loader._restore(view, state, "A")
    loader._verify(view, state, "A")
    assert loader._binding(model.weight) == binding
    assert torch.equal(_raw(model.weight), expected)


def test_verify_event_has_separate_duration_excluded_from_switch(colocated, monkeypatch):
    fixture = colocated
    prepare(fixture)
    path = fixture.root / "events.jsonl"
    monkeypatch.setenv("RELAX_ADP_EVENT_LOG", str(path))
    loader.adp_teacher_loader(fixture.model, command("verify", "A"))
    event = json.loads(path.read_text())
    assert event["operation"] == "verify" and event["verify_seconds"] >= 0
    assert event["synchronized_seconds"] is None and event["prepare_seconds"] is None


def test_tp_disagreement_on_event_opt_in_rejects_before_restore(colocated, monkeypatch):
    fixture = colocated
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    fixture.dist.size = 2

    def peer(records, value, rank):
        route = list(value["route"])
        route[-1] = not route[-1]
        records[1]["route"] = tuple(route)

    fixture.dist.peer_override = peer
    with pytest.raises(RuntimeError, match="disagree"):
        loader.adp_teacher_loader(fixture.model, command("restore", "B"))
    assert not fixture.plans["A", "B"].applies
    assert fixture.model._relax_adp_teacher_state.poisoned


def test_restore_detects_post_apply_binding_change(colocated, monkeypatch):
    fixture = colocated
    loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    plan = fixture.plans["A", "B"]
    apply = plan.apply

    def replace_after_apply(*, overlap):
        apply(overlap=overlap)
        fixture.model.scale = torch.nn.Parameter(fixture.model.scale.detach().clone(), requires_grad=False)

    monkeypatch.setattr(plan, "apply", replace_after_apply)
    with pytest.raises(RuntimeError, match="changed runtime layout or storage"):
        loader.adp_teacher_loader(fixture.model, command("restore", "B"))
    assert fixture.model._relax_adp_teacher_state.poisoned


@pytest.mark.parametrize("phase", ["preflight", "commit"])
def test_verify_peer_failure_poison(fixture, phase):
    prepare(fixture)
    fixture.dist.size = 2

    def peer(records, value, rank):
        if value["route"][0] == phase:
            records[1]["error"] = "peer verification failure"

    fixture.dist.peer_override = peer
    with pytest.raises(RuntimeError, match="peer verification failure"):
        loader.adp_teacher_loader(fixture.model, command("verify", "A"))
    assert fixture.model._relax_adp_teacher_state.poisoned


def test_peer_event_write_failure_poison_prepared_candidate(colocated, monkeypatch):
    fixture = colocated
    monkeypatch.setenv("RELAX_ADP_EVENT_LOG", str(fixture.root / "events.jsonl"))
    original = fixture.dist.all_gather_object

    def peer(output, value, *, group):
        original(output, value, group=group)
        # Preserve the fixture's TP1 preparation files while simulating a peer
        # reporting failure in the final optional event-write consensus.
        if value["route"][0] == "event":
            output.append({"route": value["route"], "error": "peer event write failed", "digest": None})

    monkeypatch.setattr(fixture.dist, "all_gather_object", peer)
    with pytest.raises(RuntimeError, match="peer event write failed"):
        loader.adp_teacher_loader(fixture.model, command("prepare_colocated", "A"))
    state = fixture.model._relax_adp_teacher_state
    assert state.poisoned and not state.prepared


def test_native_codec_imports_ignore_old_top_level_modules_without_initializing_cuda(tmp_path):
    repo = Path(loader.__file__).resolve().parents[3]
    for name in ("xor_codec", "adaptive_delta", "batched_delta"):
        (tmp_path / f"{name}.py").write_text("raise AssertionError('old algorithm imported')\n")
    code = """
import hashlib,json,sys
from pathlib import Path
import torch
from relax.backends.sglang import adp_teacher_loader as loader
repo=Path(sys.argv[1])
manifest={'algorithm_sources':{name:{'sha256':hashlib.sha256((repo/'relax/backends/sglang/adp'/f'{name}.py').read_bytes()).hexdigest()} for name in ('xor_codec','adaptive_delta','batched_delta')}}
actual=loader._attest_algorithm_sources(manifest)
for name in actual:
    module=sys.modules['relax.backends.sglang.adp.'+name]
    assert Path(module.__file__).resolve()==repo/'relax/backends/sglang/adp'/f'{name}.py'
    assert name not in sys.modules
assert not torch.cuda.is_initialized()
print(json.dumps({'cuda_initialized':False,'sources':actual}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(repo)],
        cwd=tmp_path,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            PYTHONSAFEPATH="1",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=os.pathsep.join((str(tmp_path), str(repo))),
        ),
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["cuda_initialized"] is False
