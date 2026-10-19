# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU Gloo participation checks; no Ray cluster or CUDA contexts."""

from datetime import timedelta
from multiprocessing import get_context

import pytest


def gloo_worker(rank, rendezvous, fail, results):
    import torch.distributed as dist

    from relax.utils.opd.shared_teacher_lifecycle import run_rank_zero_lifecycle

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=10))
    called = []

    def operation():
        called.append(rank)
        if fail:
            raise TimeoutError("injected Teacher onload timeout")

    error = None
    try:
        run_rank_zero_lifecycle(operation, process_group=dist.group.WORLD, label="onload")
    except RuntimeError as exc:
        error = str(exc)
    # A second real collective proves both ranks got past the same protocol,
    # including the nonleader when only rank zero's operation failed.
    dist.barrier(group=dist.group.WORLD)
    results.put((rank, called, error))
    dist.destroy_process_group()


@pytest.mark.parametrize("fail", [False, True])
def test_gloo_all_ranks_observe_leader_lifecycle_result(tmp_path, monkeypatch, fail):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ctx = get_context("spawn")
    results = ctx.Queue()
    rendezvous = (tmp_path / "gloo").as_uri()
    processes = [ctx.Process(target=gloo_worker, args=(rank, rendezvous, fail, results)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        rows = sorted(results.get(timeout=30) for _ in processes)
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        assert [row[1] for row in rows] == [[0], []]
        if fail:
            assert rows[0][2] == rows[1][2]
            assert "injected Teacher onload timeout" in rows[0][2]
        else:
            assert [row[2] for row in rows] == [None, None]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()  # Only this test's CPU children.
            if process.pid is not None:
                process.join(timeout=5)
        results.close()
