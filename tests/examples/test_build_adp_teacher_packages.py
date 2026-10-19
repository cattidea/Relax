# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Offline builder protocol tests; no codecs or GPU calls are executed."""

from __future__ import annotations

import fcntl
import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.on_policy_distillation.mopd import build_adp_teacher_packages as builder


torch = pytest.importorskip("torch")


def test_progress_emitter_throttles_at_twenty_seconds_and_always_reports_final(monkeypatch):
    now, messages = [100.0], []
    monkeypatch.setattr(builder.time, "perf_counter", lambda: now[0])
    emit = builder._progress_emitter(SimpleNamespace(info=lambda *args: messages.append(args)), "T1", "T2", 3)
    emit("unrelated")
    emit("adaptive_prepare", tensors=1, total_tensors=5, payload_bytes=10, seconds=0.0)
    now[0] = 119.99
    emit("adaptive_prepare", tensors=2, total_tensors=5, payload_bytes=20, seconds=19.99)
    assert not messages
    now[0] = 120.0
    emit("adaptive_prepare", tensors=3, total_tensors=5, payload_bytes=30, seconds=20.0)
    now[0] = 139.99
    emit("adaptive_prepare", tensors=4, total_tensors=5, payload_bytes=40, seconds=39.99)
    emit("adaptive_prepare", tensors=5, total_tensors=5, payload_bytes=50, seconds=39.99)
    assert len(messages) == 2
    assert messages[0][1:] == ("T1", "T2", 3, 3, 5, 30, 20.0)
    assert messages[1][1:] == ("T1", "T2", 3, 5, 5, 50, 39.99)


def write_package(path: Path, header: dict, payload: bytes) -> None:
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ADP1\0\0\0\0" + struct.pack("<Q", len(raw)) + raw + payload)


def read_package(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[8:16])[0]
    return json.loads(raw[16 : 16 + size]), raw[16 + size :]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    teachers, calls = ["T1", "T2", "T3"], []
    for teacher_index, teacher in enumerate(teachers):
        for rank in range(2):
            path = tmp_path / builder._capture_key(teacher, rank)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 1,
                    "runtime": "sglang-dense-v1",
                    "teacher_id": teacher,
                    "tp_rank": rank,
                    "tp_size": 1,
                    "metadata": {"layout": "same-runtime", "rank": rank},
                    "params": {
                        "weight": torch.arange(6, dtype=torch.bfloat16) + teacher_index,
                        "unchanged": torch.ones(2, dtype=torch.float32),
                    },
                    "auxiliary": {"cos": torch.arange(4, dtype=torch.float32)},
                },
                path,
            )

    def encode(directory, source, target, rank, tp_size, device, path):
        calls.append((source, target, rank))
        a = torch.load(directory / builder._capture_key(source, rank), weights_only=True)
        b = torch.load(directory / builder._capture_key(target, rank), weights_only=True)
        payload = b["params"]["weight"].view(torch.uint8).numpy().tobytes()
        header = {
            "format": "ADP1",
            "version": 1,
            "policy": "size",
            "block_bytes": 16 * 1024**2,
            "provenance": dict(runtime="sglang-dense-v1", source=source, target=target, tp_rank=rank, tp_size=tp_size),
            "tensors": {
                name: {
                    **{k: v for k, v in builder._tensor_info(tensor).items() if k != "sha256"},
                    "a_sha256": builder._tensor_info(tensor)["sha256"],
                    "b_sha256": builder._tensor_info(b["params"][name])["sha256"],
                }
                for name, tensor in a["params"].items()
            },
            "packets": [
                {
                    "mode": "RAW",
                    "name": "weight",
                    "offset": 0,
                    "n": len(payload),
                    "payload_offset": 0,
                    "payload_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
            "stats": {"fixture": True},
        }
        write_package(path, header, payload)
        return {"fixture": True}

    monkeypatch.setattr(builder, "_encode_package", encode)
    monkeypatch.setattr(builder, "_release_encoder_memory", lambda device: None)
    return SimpleNamespace(root=tmp_path, teachers=teachers, calls=calls, encode=encode)


def build(fixture, **kwargs):
    return builder.build_packages(fixture.root, fixture.teachers, 1, "cuda:fixture", **kwargs)


def test_first_pair_stops_without_runtime_manifest_and_records_exact_bytes(fixture):
    report = build(fixture, max_new_edges=1)
    assert fixture.calls == [("T1", "T2", 0)]
    assert report["status"] == "incomplete" and report["completed_edges"] == 1
    assert not (fixture.root / "manifest.json").exists()
    relative = "edges/T1/T2/rank_0.adp"
    record = report["packages"][relative]
    assert record["file_bytes"] == (fixture.root / relative).stat().st_size
    assert record["file_sha256"] == hashlib.sha256((fixture.root / relative).read_bytes()).hexdigest()
    assert record["payload_bytes"] == 12 and record["runtime_parameter_bytes"] == 20
    assert record["mode_payload_bytes"] == {"RAW": 12}
    assert record["mode_target_bytes"] == {"RAW": 12}
    assert record["mode_packet_counts"] == {"RAW": 1} and record["skipped_bytes"] == 8
    assert record["preparation_seconds"] >= record["encode_seconds"] >= 0
    assert record["source_capture"] == builder._file_fingerprint(fixture.root / "captures/T1/rank_0.pt")
    assert json.loads((fixture.root / "build_progress.json").read_text()) == report


def test_resume_reuses_only_valid_completed_package_and_publishes_complete_graph(fixture, monkeypatch):
    first = build(fixture, max_new_edges=1)
    first_file = fixture.root / "edges/T1/T2/rank_0.adp"
    before = (first_file.read_bytes(), first_file.stat().st_mtime_ns)
    result = build(fixture, resume=True)
    assert len(fixture.calls) == 6 and len(set(fixture.calls)) == 6
    assert (first_file.read_bytes(), first_file.stat().st_mtime_ns) == before
    assert result["packages"]["edges/T1/T2/rank_0.adp"] == first["packages"]["edges/T1/T2/rank_0.adp"]
    assert result["status"] == "complete" and result["last_run_new_edges"] == 5
    runtime = json.loads((fixture.root / "manifest.json").read_text())
    assert {k: runtime[k] for k in result["manifest"]} == result["manifest"]
    assert runtime["package_integrity"] == {
        name: {"sha256": record["file_sha256"], "bytes": record["file_bytes"]}
        for name, record in result["packages"].items()
    }
    root = Path(builder.__file__).resolve().parents[3] / "relax/backends/sglang/adp"
    assert runtime["algorithm_sources"] == {
        name: {"sha256": hashlib.sha256((root / f"{name}.py").read_bytes()).hexdigest()}
        for name in ("xor_codec", "adaptive_delta", "batched_delta")
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Complete resume must not invoke the encoder or GPU cleanup")

    monkeypatch.setattr(builder, "_encode_package", forbidden)
    monkeypatch.setattr(builder, "_release_encoder_memory", forbidden)
    assert build(fixture, resume=True)["status"] == "complete"


def test_limited_resume_counts_new_edges_not_already_completed_edges(fixture):
    build(fixture, max_new_edges=1)
    report = build(fixture, resume=True, max_new_edges=1)
    assert fixture.calls == [("T1", "T2", 0), ("T1", "T3", 0)]
    assert report["completed_edges"] == 2 and report["last_run_new_edges"] == 1
    assert not (fixture.root / "manifest.json").exists()


def test_default_cli_behavior_still_builds_full_directed_graph(fixture):
    report = build(fixture)
    assert report["status"] == "complete" and len(report["packages"]) == 6
    with pytest.raises(FileExistsError, match="resume"):
        build(fixture)


@pytest.mark.parametrize("corruption", ["capture", "payload", "header", "progress_sha", "recorded_provenance"])
def test_resume_rejects_corruption_before_encoding_more_packages(fixture, corruption):
    build(fixture, max_new_edges=1)
    path = fixture.root / "edges/T1/T2/rank_0.adp"
    if corruption == "capture":
        path = fixture.root / "captures/T1/rank_0.pt"
        path.write_bytes(path.read_bytes() + b"changed")
    elif corruption == "payload":
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    elif corruption == "header":
        header, payload = read_package(path)
        header["provenance"]["target"] = "T3"
        write_package(path, header, payload)
    else:
        progress_path = fixture.root / "build_progress.json"
        report = json.loads(progress_path.read_text())
        if corruption == "progress_sha":
            report["packages"]["edges/T1/T2/rank_0.adp"]["file_sha256"] = "0" * 64
        else:
            report["captures"]["captures/T1/rank_0.pt"]["provenance"]["teacher_id"] = "T2"
        progress_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        build(fixture, resume=True)
    assert len(fixture.calls) == 1
    assert not (fixture.root / "manifest.json").exists()


def test_unrecorded_files_are_not_trusted_or_overwritten(fixture):
    build(fixture, max_new_edges=1)
    path = fixture.root / "edges/T1/T3/rank_0.adp"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"untrusted package")
    with pytest.raises(FileExistsError, match="Unrecorded"):
        build(fixture, resume=True)
    assert path.read_bytes() == b"untrusted package" and len(fixture.calls) == 1


def test_interrupted_next_edge_preserves_durable_completed_edges(fixture, monkeypatch):
    def interrupted(*args):
        if len(fixture.calls) == 1:
            raise KeyboardInterrupt("injected interruption")
        return fixture.encode(*args)

    monkeypatch.setattr(builder, "_encode_package", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build(fixture)
    progress = json.loads((fixture.root / "build_progress.json").read_text())
    assert len(progress["packages"]) == 1
    assert not (fixture.root / "manifest.json").exists()
    monkeypatch.setattr(builder, "_encode_package", fixture.encode)
    assert build(fixture, resume=True)["status"] == "complete"
    assert fixture.calls.count(("T1", "T2", 0)) == 1


def test_limit_counts_complete_tp_edges_and_partial_rank_resume(fixture, monkeypatch):
    for teacher in fixture.teachers:
        for rank in range(2):
            path = fixture.root / builder._capture_key(teacher, rank)
            snapshot = torch.load(path, weights_only=True)
            snapshot["tp_size"] = 2
            torch.save(snapshot, path)
    calls = []

    def interrupted(*args):
        calls.append((args[1], args[2], args[3]))
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return fixture.encode(*args)

    monkeypatch.setattr(builder, "_encode_package", interrupted)
    with pytest.raises(KeyboardInterrupt):
        builder.build_packages(fixture.root, fixture.teachers, 2, "cuda:fixture", max_new_edges=1)
    report = json.loads((fixture.root / "build_progress.json").read_text())
    assert report["completed_edges"] == 0 and len(report["packages"]) == 1
    monkeypatch.setattr(builder, "_encode_package", fixture.encode)
    report = builder.build_packages(fixture.root, fixture.teachers, 2, "cuda:fixture", max_new_edges=1, resume=True)
    assert report["completed_edges"] == 1 and len(report["packages"]) == 2
    assert fixture.calls == [("T1", "T2", 0), ("T1", "T2", 1)]
    assert not (fixture.root / "manifest.json").exists()


@pytest.mark.parametrize("corruption", ["auxiliary", "provenance", "layout"])
def test_capture_validation_rejects_unsupported_sources_before_encoder(fixture, corruption):
    path = fixture.root / "captures/T2/rank_0.pt"
    snapshot = torch.load(path, weights_only=True)
    if corruption == "auxiliary":
        snapshot["auxiliary"]["cos"].add_(1)
    elif corruption == "provenance":
        snapshot["teacher_id"] = "T3"
    else:
        snapshot["metadata"]["layout"] = "different"
    torch.save(snapshot, path)
    with pytest.raises(ValueError):
        build(fixture, max_new_edges=1)
    assert fixture.calls == [] and not (fixture.root / "manifest.json").exists()


def test_capture_mutation_during_encode_is_rejected_without_publication(fixture, monkeypatch):
    def mutate(*args):
        result = fixture.encode(*args)
        path = fixture.root / "captures/T1/rank_0.pt"
        path.write_bytes(path.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(builder, "_encode_package", mutate)
    with pytest.raises(ValueError, match="Capture changed during"):
        build(fixture, max_new_edges=1)
    assert not (fixture.root / "edges/T1/T2/rank_0.adp").exists()
    assert not json.loads((fixture.root / "build_progress.json").read_text())["packages"]


def test_advisory_lock_prevents_two_builders_mutating_same_journal(fixture):
    with (fixture.root / ".build_packages.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another package builder"):
            build(fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_edge_limit_fails_before_progress_or_encoder(fixture, limit):
    with pytest.raises(ValueError, match="positive"):
        build(fixture, max_new_edges=limit)
    assert not (fixture.root / "build_progress.json").exists()


def test_missing_resume_journal_cannot_adopt_existing_outputs(fixture):
    with pytest.raises(FileNotFoundError):
        build(fixture, resume=True)
    assert fixture.calls == []


def test_changed_build_identity_cannot_resume(fixture):
    build(fixture, max_new_edges=1)
    with pytest.raises(ValueError, match="configuration differs"):
        builder.build_packages(fixture.root, list(reversed(fixture.teachers)), 1, "cuda:fixture", resume=True)
    assert len(fixture.calls) == 1


def test_complete_resume_revalidates_packages_even_with_existing_manifest(fixture):
    build(fixture)
    path = fixture.root / "edges/T3/T2/rank_0.adp"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        build(fixture, resume=True)
    assert len(fixture.calls) == 6


def test_five_teachers_profile_one_then_resume_remaining_nineteen(fixture):
    template = torch.load(fixture.root / "captures/T3/rank_0.pt", weights_only=True)
    for number in (4, 5):
        teacher = f"T{number}"
        snapshot = dict(template, teacher_id=teacher)
        snapshot["params"] = {
            name: tensor + number if name == "weight" else tensor for name, tensor in template["params"].items()
        }
        path = fixture.root / builder._capture_key(teacher, 0)
        path.parent.mkdir(parents=True)
        torch.save(snapshot, path)
        fixture.teachers.append(teacher)
    first = build(fixture, max_new_edges=1)
    assert first["completed_edges"] == 1 and not (fixture.root / "manifest.json").exists()
    final = build(fixture, resume=True)
    assert final["completed_edges"] == 20 and final["last_run_new_edges"] == 19
    assert len(fixture.calls) == 20 and len(set(fixture.calls)) == 20
    assert final["status"] == "complete" and (fixture.root / "manifest.json").exists()


def test_cli_forwards_optional_first_pair_and_resume_flags(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(builder, "build_packages", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--teacher-dir",
            str(tmp_path),
            "--teachers",
            "A",
            "B",
            "--tp-size",
            "1",
            "--device",
            "cuda:2",
            "--max-new-edges",
            "1",
            "--resume",
        ],
    )
    builder.main()
    assert calls == [
        ((tmp_path, ["A", "B"], 1, "cuda:2"), {"max_new_edges": 1, "resume": True, "reuse_symmetric_pairs": False})
    ]


def test_crash_between_package_publish_and_journal_commit_requires_orphan_review(fixture, monkeypatch):
    atomic = builder._atomic_json
    count = 0

    def interrupted(path, value):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt("journal interrupted")
        atomic(path, value)

    monkeypatch.setattr(builder, "_atomic_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build(fixture, max_new_edges=1)
    assert (fixture.root / "edges/T1/T2/rank_0.adp").exists()
    assert not json.loads((fixture.root / "build_progress.json").read_text())["packages"]
    monkeypatch.setattr(builder, "_atomic_json", atomic)
    with pytest.raises(FileExistsError, match="Unrecorded"):
        build(fixture, resume=True)
    assert len(fixture.calls) == 1


def use_modes(fixture, monkeypatch, choose=lambda source, target, rank: "PLANES"):
    """CPU encoder protocol stub; PLANES payloads contain real symmetric XOR
    bits."""

    def encode(directory, source, target, rank, tp_size, device, path):
        result = fixture.encode(directory, source, target, rank, tp_size, device, path)
        header, payload = read_package(path)
        mode = choose(source, target, rank)
        packet = header["packets"][0]
        if mode == "PLANES":
            a = torch.load(directory / builder._capture_key(source, rank), weights_only=True)
            raw = a["params"]["weight"].view(torch.uint8).numpy().tobytes()
            delta = [x ^ y for x, y in zip(raw, payload)]
            stride = (len(delta) + 7) // 8
            packed = bytearray(8 * stride)
            for bit in range(8):
                for index, value in enumerate(delta):
                    packed[bit * stride + index // 8] |= ((value >> bit) & 1) << (index % 8)
            payload = bytes(packed)
            packet.update(mask=255, ones=0, stride=stride)
        elif mode.endswith("ANS"):
            # This suite validates package graph/provenance, not nvCOMP execution.
            packet["decoded_n"] = packet["n"]
        packet.update(mode=mode, payload_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        write_package(path, header, payload)
        return result

    monkeypatch.setattr(builder, "_encode_package", encode)
    return encode


def add_five_teachers(fixture):
    template = torch.load(fixture.root / "captures/T3/rank_0.pt", weights_only=True)
    for number in (4, 5):
        teacher = f"T{number}"
        snapshot = dict(template, teacher_id=teacher)
        snapshot["params"] = {
            name: tensor + number if name == "weight" else tensor for name, tensor in template["params"].items()
        }
        path = fixture.root / builder._capture_key(teacher, 0)
        path.parent.mkdir(parents=True)
        torch.save(snapshot, path)
        fixture.teachers.append(teacher)


def test_symmetric_five_teacher_graph_uses_ten_files_and_twenty_logical_edges(fixture, monkeypatch):
    add_five_teachers(fixture)
    use_modes(fixture, monkeypatch)
    first = build(fixture, max_new_edges=1, reuse_symmetric_pairs=True)
    assert first["completed_edges"] == 2 and first["completed_physical_edges"] == 1
    assert first["last_run_new_edges"] == 1 and first["last_run_reused_edges"] == 1
    assert not (fixture.root / "manifest.json").exists()
    report = build(fixture, resume=True, reuse_symmetric_pairs=True)
    manifest = json.loads((fixture.root / "manifest.json").read_text())
    assert len(fixture.calls) == 10 and len(set(fixture.calls)) == 10
    assert report["completed_edges"] == 20 and report["physical_package_files"] == 10
    assert len(report["reused_edges"]) == 10 and len(manifest["edges"]) == 20
    assert len({path for edge in manifest["edges"] for path in edge["packages"]}) == 10
    for edge in manifest["edges"]:
        if edge.get("reverse"):
            physical = report["packages"][edge["packages"][0]]["provenance"]
            assert (physical["source"], physical["target"]) == (edge["target"], edge["source"])
    before = dict(report["packages"])
    assert build(fixture, resume=True, reuse_symmetric_pairs=True)["packages"] == before
    assert len(fixture.calls) == 10


def test_legacy_first_edge_can_migrate_without_rewriting_packages_or_captures(fixture, monkeypatch):
    use_modes(fixture, monkeypatch, lambda source, target, rank: "XOR_GROUP2_ANS")
    build(fixture, max_new_edges=1)
    progress_path = fixture.root / "build_progress.json"
    legacy = json.loads(progress_path.read_text())
    legacy.pop("build_strategy")
    legacy.pop("reused_edges")
    progress_path.write_text(json.dumps(legacy))
    first_path = fixture.root / "edges/T1/T2/rank_0.adp"
    before = (first_path.read_bytes(), first_path.stat().st_mtime_ns)
    captures = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (fixture.root / "captures").rglob("*.pt")}
    migrated = build(fixture, resume=True, max_new_edges=1, reuse_symmetric_pairs=True)
    assert migrated["build_strategy"] == "symmetric-pairs-v1"
    assert migrated["strategy_migrations"][0]["packages_preserved"] == 1
    assert migrated["completed_edges"] == 4 and len(migrated["packages"]) == 2
    assert migrated["last_run_new_edges"] == 1 and len(fixture.calls) == 2
    assert (first_path.read_bytes(), first_path.stat().st_mtime_ns) == before
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in captures.items())
    assert not (fixture.root / "manifest.json").exists()
    final = build(fixture, resume=True, reuse_symmetric_pairs=True)
    assert final["completed_edges"] == 6 and len(final["packages"]) == 3


def test_mixed_raw_add_and_symmetric_packages_build_required_directed_files(fixture, monkeypatch):
    modes = {("T1", "T3"): "RAW", ("T2", "T3"): "ADD_ANS"}
    use_modes(fixture, monkeypatch, lambda source, target, rank: modes.get((source, target), "PLANES"))
    result = build(fixture, reuse_symmetric_pairs=True)
    assert result["completed_edges"] == 6 and len(result["packages"]) == 5
    assert set(result["reused_edges"]) == {"T2->T1"}
    assert ("T3", "T1", 0) in fixture.calls and ("T3", "T2", 0) in fixture.calls


@pytest.mark.parametrize("mode", ["XOR_ANS", "XOR_GROUP2_ANS", "PLANES", "PLANES_ANS"])
def test_actual_symmetric_mode_names_are_eligible(fixture, monkeypatch, mode):
    use_modes(fixture, monkeypatch, lambda source, target, rank: mode)
    result = build(fixture, max_new_edges=1, reuse_symmetric_pairs=True)
    assert result["reused_edges"]["T2->T1"]["reverse"] is True
    assert result["packages"]["edges/T1/T2/rank_0.adp"]["mode_packet_counts"] == {mode: 1}


@pytest.mark.parametrize("rank_one_mode,expected_files", [("PLANES", 2), ("RAW", 4)])
def test_tp_reverse_reuse_requires_every_rank_symmetric(fixture, monkeypatch, rank_one_mode, expected_files):
    fixture.teachers = fixture.teachers[:2]
    for teacher in fixture.teachers:
        for rank in range(2):
            path = fixture.root / builder._capture_key(teacher, rank)
            value = torch.load(path, weights_only=True)
            value["tp_size"] = 2
            torch.save(value, path)
    use_modes(fixture, monkeypatch, lambda source, target, rank: "PLANES" if rank == 0 else rank_one_mode)
    result = builder.build_packages(fixture.root, fixture.teachers, 2, "cuda:fixture", reuse_symmetric_pairs=True)
    assert result["completed_edges"] == 2 and len(result["packages"]) == expected_files
    assert len(result["reused_edges"]) == (rank_one_mode == "PLANES")


def test_existing_both_direction_packages_are_preserved_on_migration(fixture, monkeypatch):
    use_modes(fixture, monkeypatch)
    first = build(fixture, max_new_edges=3)
    paths = {relative: (fixture.root / relative).read_bytes() for relative in first["packages"]}
    result = build(fixture, resume=True, reuse_symmetric_pairs=True)
    assert "T2->T1" not in result["reused_edges"] and len(result["packages"]) == 4
    assert all((fixture.root / path).read_bytes() == value for path, value in paths.items())


@pytest.mark.parametrize("corruption", ["physical_bytes", "reverse_flag", "wrong_pair", "asymmetric"])
def test_resume_rejects_corrupt_reverse_metadata_or_physical_package(fixture, monkeypatch, corruption):
    use_modes(fixture, monkeypatch)
    result = build(fixture, max_new_edges=1, reuse_symmetric_pairs=True)
    path = fixture.root / "edges/T1/T2/rank_0.adp"
    if corruption == "physical_bytes":
        path.write_bytes(path.read_bytes()[:-1])
    elif corruption == "reverse_flag":
        result["reused_edges"]["T2->T1"]["reverse"] = 1
    elif corruption == "wrong_pair":
        result["reused_edges"]["T2->T1"]["packages"] = ["edges/T1/T3/rank_0.adp"]
    else:
        header, payload = read_package(path)
        header["packets"][0].update(mode="ADD_ANS", decoded_n=header["packets"][0]["n"])
        write_package(path, header, payload)
        record = result["packages"]["edges/T1/T2/rank_0.adp"]
        record.update(
            builder._package_info(
                path,
                record["provenance"],
                result["captures"]["captures/T1/rank_0.pt"],
                result["captures"]["captures/T2/rank_0.pt"],
            )
        )
    (fixture.root / "build_progress.json").write_text(json.dumps(result))
    with pytest.raises(ValueError):
        build(fixture, resume=True, reuse_symmetric_pairs=True)
    assert len(fixture.calls) == 1 and not (fixture.root / "manifest.json").exists()


def test_published_reverse_manifest_cannot_be_tampered_or_silently_downgraded(fixture, monkeypatch):
    use_modes(fixture, monkeypatch)
    build(fixture, reuse_symmetric_pairs=True)
    with pytest.raises(ValueError, match="strategy differs"):
        build(fixture, resume=True)
    path = fixture.root / "manifest.json"
    manifest = json.loads(path.read_text())
    next(edge for edge in manifest["edges"] if edge.get("reverse"))["reverse"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Existing manifest"):
        build(fixture, resume=True, reuse_symmetric_pairs=True)
    assert len(fixture.calls) == 3


def test_published_legacy_manifest_cannot_change_strategy(fixture, monkeypatch):
    use_modes(fixture, monkeypatch)
    build(fixture)
    with pytest.raises(ValueError, match="strategy differs"):
        build(fixture, resume=True, reuse_symmetric_pairs=True)
    assert len(fixture.calls) == 6


def test_all_skip_empty_packet_package_is_symmetric(fixture, monkeypatch):
    fixture.teachers = fixture.teachers[:2]
    a = torch.load(fixture.root / "captures/T1/rank_0.pt", weights_only=True)
    torch.save(dict(a, teacher_id="T2"), fixture.root / "captures/T2/rank_0.pt")

    def encode(*args):
        result = fixture.encode(*args)
        path = args[-1]
        header, _ = read_package(path)
        header["packets"] = []
        write_package(path, header, b"")
        return result

    monkeypatch.setattr(builder, "_encode_package", encode)
    result = build(fixture, reuse_symmetric_pairs=True)
    assert len(result["packages"]) == 1 and result["completed_edges"] == 2
    assert result["reused_edges"]["T2->T1"]["reverse"] is True
    assert result["packages"]["edges/T1/T2/rank_0.adp"]["mode_packet_counts"] == {}


def test_modifying_earlier_physical_file_during_build_prevents_manifest(fixture, monkeypatch):
    encode = use_modes(fixture, monkeypatch)

    def corrupt_after_last(*args):
        result = encode(*args)
        if len(fixture.calls) == 3:
            path = fixture.root / "edges/T1/T2/rank_0.adp"
            path.write_bytes(path.read_bytes()[:-1])
        return result

    monkeypatch.setattr(builder, "_encode_package", corrupt_after_last)
    with pytest.raises(ValueError, match="Physical package changed before manifest"):
        build(fixture, reuse_symmetric_pairs=True)
    assert not (fixture.root / "manifest.json").exists()
