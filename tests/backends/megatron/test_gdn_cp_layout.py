# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for GDN CP partitions, layout routes, and boundary caches.

Checks token ownership, agreement with Relax's sharding, route equivalence, and
cache reuse/invalidation. Real NCCL layout communication is covered in
``test_gdn_cp_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch


cpl = pytest.importorskip("megatron.core.context_parallel_layout", reason="requires the patched Megatron-LM")

from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402

from relax.backends.megatron.cp_utils import gdn_cp_slice, slice_with_cp  # noqa: E402


DIRECTIONS = [("zigzag", "contiguous"), ("contiguous", "zigzag")]
# GDN route-reuse training requires docker/patch/megatron/gdn-thd-route-reuse.patch
# applied to Megatron on every training node; Relax does not apply it automatically.
# Without the route API, these regressions are skipped, not validated.
requires_legacy_routes = pytest.mark.skipif(
    not hasattr(cpl, "build_thd_cp_partition_route"),
    reason="requires the legacy direction-specific route and cache API",
)


@pytest.fixture(autouse=True)
def _require_legacy_index_api(request):
    if not request.node.originalname.startswith(("test_native_", "test_boundary_")) and not hasattr(
        cpl, "get_thd_context_parallel_rank_indices"
    ):
        pytest.skip("legacy index helpers were replaced by native CP layout routes")


# Packed boundary shapes worth covering: single sequence, uneven multi-sequence,
# and a duplicated boundary (an empty padding slot), which the compaction step
# has to drop before the segments line up.
LENGTH_CASES = [
    [1],
    [3, 1, 2],
    [2, 0, 1, 3],
]


def _cu(lengths: list[int], unit: int = 1) -> torch.Tensor:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n * unit)
    return torch.tensor(cu, dtype=torch.int64)


def _tagged_tokens(total: int, width: int = 3) -> torch.Tensor:
    """[total, width] where row t is (t, t+1e6, t+2e6): token identity is
    unambiguous."""
    base = torch.arange(total, dtype=torch.float64).unsqueeze(1)
    return base + torch.arange(width, dtype=torch.float64).unsqueeze(0) * 1e6


# ---------------------------------------------------------------------------
# Partition definitions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
@pytest.mark.parametrize("lengths_factor", [[1], [1, 2, 3], [3, 1, 1, 2]])
def test_thd_rank_indices_partition_all_tokens_exactly_once(cp_size, layout, lengths_factor):
    lengths = [2 * cp_size * f for f in lengths_factor]
    cu = _cu(lengths)
    owned = torch.cat([cpl.get_thd_context_parallel_rank_indices(cu, cp_size, r, layout) for r in range(cp_size)])
    assert owned.numel() == int(cu[-1])
    assert torch.equal(torch.sort(owned).values, torch.arange(int(cu[-1])))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_zigzag_rank_indices_match_relax_data_sharding(cp_size):
    """MCore's zigzag partition must be token-for-token what Relax's data path
    produces.

    If these ever disagree, chunkwise CP would silently permute tokens relative
    to the all-gather fallback and the attention layers.
    """
    lengths = [2 * cp_size * f for f in (1, 3, 2)]
    cu = _cu(lengths)
    full = _tagged_tokens(int(cu[-1])).reshape(-1, 1, 3)  # [s, b=1, C]

    for rank in range(cp_size):
        mcore_idx = cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, "zigzag")
        mcore_shard = full[mcore_idx]

        # Relax data.py: per-sample slice_with_cp then concat.
        relax_shard = torch.cat(
            [
                slice_with_cp(
                    full[cu[i] : cu[i + 1]],
                    pad_value=0.0,
                    qkv_format="thd",
                    dynamic_cp_size=cp_size,
                    dynamic_cp_rank=rank,
                )
                for i in range(len(lengths))
            ],
            dim=0,
        )
        assert torch.equal(mcore_shard, relax_shard)

        # Relax model.py (all-gather fallback) re-slices with gdn_cp_slice.
        assert torch.equal(mcore_shard, gdn_cp_slice(full, cu, cp_size, rank))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("lengths_factor", [[1], [1, 2, 3], [3, 1, 1, 2]])
def test_both_layouts_are_permutations_of_each_other(cp_size, lengths_factor):
    """The two partitions must describe the same token set with the same per-
    rank size.

    That is the precondition for the all-to-all between them to be a pure
    permutation -- no token invented, dropped, or duplicated. The real collective
    round trip is asserted in ``test_gdn_cp_gpu.py``.
    """
    lengths = [2 * cp_size * f for f in lengths_factor]
    cu = _cu(lengths)
    total = int(cu[-1])
    zig_by_rank = []
    con_by_rank = []
    for rank in range(cp_size):
        zig = cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, "zigzag")
        con = cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, "contiguous")
        zig_by_rank.append(zig)
        con_by_rank.append(con)
        assert zig.numel() == con.numel() == total // cp_size
        # contiguous is exactly this rank's span of the flattened buffer
        assert torch.equal(con, torch.arange(rank * (total // cp_size), (rank + 1) * (total // cp_size)))

    # Across the whole CP group, both layouts are permutations of exactly the
    # same global token rows.
    assert torch.equal(
        torch.cat(zig_by_rank).sort().values,
        torch.cat(con_by_rank).sort().values,
    )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_rank_indices_reject_lengths_not_divisible_by_two_cp(cp_size):
    bad = _cu([2 * cp_size, 2 * cp_size + 1])
    with pytest.raises(ValueError, match="divisible by"):
        cpl.get_thd_context_parallel_rank_indices(bad, cp_size, 0, "zigzag")


def test_gdn_rejects_packed_lengths_not_divisible_by_cp():
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    cu = _cu([8, 6])
    with pytest.raises(ValueError, match="divisible by cp_size=4"):
        GatedDeltaNet._resolve_cu_seqlens(None, None, cu, int(cu[-1]), "cu_seqlens_q", cp_size=4)


def test_rank_indices_reject_unknown_layout():
    with pytest.raises(ValueError, match="Unsupported context-parallel layout"):
        cpl.get_thd_context_parallel_rank_indices(_cu([16, 16]), 2, 0, "contiguous_ish")


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_rank_indices_ignore_duplicate_boundaries(layout):
    compact = torch.tensor([0, 16, 40], dtype=torch.int64)
    padded = torch.tensor([0, 16, 40, 40, 40], dtype=torch.int64)
    for rank in range(2):
        assert torch.equal(
            cpl.get_thd_context_parallel_rank_indices(compact, 2, rank, layout),
            cpl.get_thd_context_parallel_rank_indices(padded, 2, rank, layout),
        )


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_rank_indices_reject_decreasing_boundaries(layout):
    with pytest.raises(ValueError, match="nondecreasing"):
        cpl.get_thd_context_parallel_rank_indices(torch.tensor([0, 16, 8]), 2, 0, layout)


def _packed_seq_params(cu: torch.Tensor) -> PackedSeqParams:
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=int(cu[-1]),
        max_seqlen_kv=int(cu[-1]),
    )


def _apply_route_across_ranks(
    x: torch.Tensor, cu: torch.Tensor, cp_size: int, source: str, target: str
) -> list[torch.Tensor]:
    """Run the route-driven swap for every rank, emulating the all-to-all
    locally."""
    source_by_rank = [cpl.get_thd_context_parallel_rank_indices(cu, cp_size, r, source) for r in range(cp_size)]
    routes = [cpl.build_thd_cp_partition_route(cu, cp_size, r, source, target) for r in range(cp_size)]

    send_bufs = []
    for rank, route in enumerate(routes):
        local = x[source_by_rank[rank]]
        assert local.size(0) == route.local_source_length
        send_bufs.append(local if route.send_rows is None else local.index_select(0, route.send_rows))

    outputs = []
    for dst, route in enumerate(routes):
        parts = []
        for src in range(cp_size):
            offset = sum(routes[src].input_split_sizes[:dst])
            length = routes[src].input_split_sizes[dst]
            assert length == route.output_split_sizes[src], "split sizes disagree between peers"
            parts.append(send_bufs[src][offset : offset + length])
        recv = torch.cat(parts, dim=0)
        if route.recv_rows is None:
            outputs.append(recv)
        else:
            out = recv.new_empty((route.local_target_length,) + tuple(x.shape[1:]))
            out.index_copy_(0, route.recv_rows, recv)
            outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# The route is the same permutation phase 1 computed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("source,target", DIRECTIONS)
@pytest.mark.parametrize("lengths", LENGTH_CASES)
@requires_legacy_routes
def test_route_reproduces_index_based_partition(cp_size, source, target, lengths):
    cu = _cu(lengths, unit=2 * cp_size)
    total = int(cu[-1])
    x = torch.arange(total * 3, dtype=torch.float64).reshape(total, 3)

    got = _apply_route_across_ranks(x, cu, cp_size, source, target)
    for rank in range(cp_size):
        want = x[cpl.get_thd_context_parallel_rank_indices(cu, cp_size, rank, target)]
        assert torch.equal(got[rank], want), f"cp_size={cp_size} rank={rank} {source}->{target}"


# ---------------------------------------------------------------------------
# Fail-fast parity with the index-based builder
# ---------------------------------------------------------------------------
@requires_legacy_routes
def test_route_rejects_lengths_not_divisible_by_two_cp():
    cu = torch.tensor([0, 12], dtype=torch.int64)  # 12 % (2 * 4) != 0
    with pytest.raises(ValueError, match="divisible by"):
        cpl.get_thd_context_parallel_rank_indices(cu, 4, 0, "zigzag")
    with pytest.raises(ValueError, match="divisible by"):
        cpl.build_thd_cp_partition_route(cu, 4, 0, "zigzag", "contiguous")


@requires_legacy_routes
def test_route_rejects_malformed_cu_seqlens():
    with pytest.raises(ValueError, match="must start at 0"):
        cpl.build_thd_cp_partition_route(torch.tensor([8, 16], dtype=torch.int64), 2, 0, "zigzag", "contiguous")
    with pytest.raises(ValueError, match="nondecreasing"):
        cpl.build_thd_cp_partition_route(torch.tensor([0, 16, 8], dtype=torch.int64), 2, 0, "zigzag", "contiguous")


@requires_legacy_routes
def test_route_rejects_unknown_layout():
    cu = _cu([1], unit=4)
    with pytest.raises(ValueError, match="Unsupported CP layout conversion"):
        cpl.build_thd_cp_partition_route(cu, 2, 0, "zigzag", "interleaved")


# ---------------------------------------------------------------------------
# Caching: reuse only within the micro-batch it was built for
# ---------------------------------------------------------------------------
@requires_legacy_routes
def test_route_is_cached_per_packed_seq_params():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)

    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    second = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    assert second is first, "a second layer of the same micro-batch must reuse the route"
    assert psp.cp_partition_route_zigzag_to_contiguous is first


@requires_legacy_routes
def test_both_directions_are_cached_separately():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)

    to_contiguous = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    to_zigzag = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "contiguous", "zigzag")
    assert to_contiguous is not to_zigzag
    assert psp.cp_partition_route_zigzag_to_contiguous is to_contiguous
    assert psp.cp_partition_route_contiguous_to_zigzag is to_zigzag


@requires_legacy_routes
def test_route_is_rebuilt_for_new_packed_boundaries():
    """Packed boundaries move every micro-batch; a stale route would corrupt
    tokens."""
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    next_cu = _cu([2, 2], unit=8)
    psp.cu_seqlens_q = next_cu
    rebuilt = cpl.get_thd_cp_partition_route(psp, next_cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt is not first
    assert rebuilt.cu_seqlens is next_cu

    want = cpl.build_thd_cp_partition_route(next_cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt.input_split_sizes == want.input_split_sizes
    assert rebuilt.output_split_sizes == want.output_split_sizes
    for field in ("send_rows", "recv_rows"):
        got_rows, want_rows = getattr(rebuilt, field), getattr(want, field)
        assert (got_rows is None) == (want_rows is None)
        if want_rows is not None:
            assert torch.equal(got_rows, want_rows)


@requires_legacy_routes
def test_route_is_rebuilt_when_cu_seqlens_is_mutated_in_place():
    """Identity alone would miss a caller refilling a preallocated boundary
    buffer.

    Megatron already has that pattern elsewhere (persistent ``_cu_seqlens_buffer``
    written with ``buf[0] = 0``), so the cache also fingerprints autograd's version
    counter, which every in-place write bumps.
    """
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    # Same tensor object, refilled with different boundaries.
    cu.copy_(_cu([2, 2], unit=8))
    rebuilt = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt is not first, "an in-place refill must invalidate the cached route"

    want = cpl.build_thd_cp_partition_route(cu, 4, 1, "zigzag", "contiguous")
    assert rebuilt.input_split_sizes == want.input_split_sizes
    assert rebuilt.output_split_sizes == want.output_split_sizes


@requires_legacy_routes
def test_route_is_rebuilt_when_a_view_of_cu_seqlens_is_mutated():
    """Views share the version counter with their base, so writes through one
    count."""
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    cu[1:] = _cu([2, 2], unit=8)[1:]
    assert cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous") is not first


@requires_legacy_routes
def test_route_is_rebuilt_when_the_dynamic_cp_geometry_changes():
    """Dynamic CP varies cp_size/cp_rank across micro-batches on one module."""
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    cp4 = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")

    cp2 = cpl.get_thd_cp_partition_route(psp, cu, 2, 1, "zigzag", "contiguous")
    assert cp2 is not cp4
    assert (cp2.cp_size, cp2.cp_rank) == (2, 1)

    other_rank = cpl.get_thd_cp_partition_route(psp, cu, 2, 0, "zigzag", "contiguous")
    assert other_rank is not cp2
    assert other_rank.cp_rank == 0


@requires_legacy_routes
def test_prebuild_populates_both_directions():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)

    class _FakeGroup:
        def size(self):
            return 4

        def rank(self):
            return 2

    psp.cp_group = _FakeGroup()
    psp.local_cp_size = 4
    cpl.prebuild_thd_cp_partition_routes(psp)

    for attr in ("cp_partition_route_zigzag_to_contiguous", "cp_partition_route_contiguous_to_zigzag"):
        route = getattr(psp, attr)
        assert route is not None
        assert (route.cp_size, route.cp_rank) == (4, 2)


@requires_legacy_routes
def test_prebuild_is_a_noop_without_context_parallelism():
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    cpl.prebuild_thd_cp_partition_routes(psp)
    assert getattr(psp, "cp_partition_route_zigzag_to_contiguous", None) is None

    non_thd = PackedSeqParams(qkv_format="sbhd")
    cpl.prebuild_thd_cp_partition_routes(non_thd)
    assert getattr(non_thd, "cp_partition_route_zigzag_to_contiguous", None) is None


@requires_legacy_routes
def test_route_does_not_cache_unversioned_inference_boundaries():
    with torch.inference_mode():
        cu = _cu([3, 1], unit=8)
        psp = _packed_seq_params(cu)
        first = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
        cu.copy_(_cu([2, 2], unit=8))
        second = cpl.get_thd_cp_partition_route(psp, cu, 4, 1, "zigzag", "contiguous")
        assert second is not first
        expected = cpl.build_thd_cp_partition_route(cu, 4, 1, "zigzag", "contiguous")
        assert second.input_split_sizes == expected.input_split_sizes


def _boundary_resolver():
    from types import SimpleNamespace

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    if not hasattr(GatedDeltaNet, "_resolve_thd_cu_seqlens"):
        pytest.skip("installed MCore lacks the cached THD boundary resolver from the current Relax patch")

    calls = []
    module = SimpleNamespace(cp_size=8)

    def resolve(*args, **kwargs):
        calls.append(kwargs["cp_size"])
        return GatedDeltaNet._resolve_cu_seqlens(module, *args, **kwargs)

    module._resolve_cu_seqlens = resolve
    return module, calls, GatedDeltaNet._resolve_thd_cu_seqlens


def test_boundary_validation_is_shared_across_layers_and_recompute():
    module, calls, resolve = _boundary_resolver()
    # Static max CP=8 would reject a length of 12; runtime CP=2 is legal.
    cu = _cu([3, 1], unit=4)
    psp = _packed_seq_params(cu)
    first = resolve(module, psp, 16, 2)
    assert calls == [2, 2]
    assert resolve(module, psp, 16, 2) is first
    assert calls == [2, 2]

    other_module, other_calls, _ = _boundary_resolver()
    assert resolve(other_module, psp, 16, 2) is first
    assert other_calls == []


@pytest.mark.parametrize("change", ["replace", "inplace", "view", "new_pack", "runtime_cp", "global_length", "padded"])
def test_boundary_cache_is_invalidated_when_inputs_change(change):
    module, calls, resolve = _boundary_resolver()
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    resolve(module, psp, 32, 2)
    cp_size, total = 2, 32
    if change == "replace":
        psp.cu_seqlens_q = cu.clone()
        psp.cu_seqlens_kv = psp.cu_seqlens_q
    elif change == "inplace":
        cu.copy_(_cu([2, 2], unit=8))
    elif change == "view":
        cu[1:2].fill_(16)
    elif change == "new_pack":
        psp = _packed_seq_params(cu)
    elif change == "runtime_cp":
        cp_size = 4
    elif change == "global_length":
        total = 64
    elif change == "padded":
        psp.cu_seqlens_q_padded = _cu([2, 2], unit=8)
        psp.cu_seqlens_kv_padded = psp.cu_seqlens_q_padded
    if change == "global_length":
        with pytest.raises(ValueError, match="total_sequence_length"):
            resolve(module, psp, total, cp_size)
        assert len(calls) == 3
    else:
        resolve(module, psp, total, cp_size)
        assert len(calls) == 4


def test_boundary_validation_rechecks_inference_tensors():
    module, calls, resolve = _boundary_resolver()
    with torch.inference_mode():
        cu = _cu([3, 1], unit=8)
        psp = _packed_seq_params(cu)
        resolve(module, psp, 32, 2)
        cu[1] = 16
        resolve(module, psp, 32, 2)
        assert len(calls) == 4


def test_boundary_cache_does_not_hide_q_kv_mismatch():
    module, calls, resolve = _boundary_resolver()
    cu = _cu([3, 1], unit=8)
    psp = _packed_seq_params(cu)
    resolve(module, psp, 32, 2)
    psp.cu_seqlens_kv = _cu([2, 2], unit=8)
    with pytest.raises(AssertionError, match="cu_seqlens_q equals"):
        resolve(module, psp, 32, 2)


@pytest.fixture
def native_routes():
    return pytest.importorskip("megatron.core.context_parallel_layout.routes", reason="requires native CP routes")


def _native_rank_indices(cu, cp_size, rank, layout):
    if layout == "contiguous":
        local = int(cu[-1]) // cp_size
        return torch.arange(rank * local, (rank + 1) * local)
    parts = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        chunk = (end - start) // (2 * cp_size)
        parts.extend(
            torch.arange(start + owner * chunk, start + (owner + 1) * chunk)
            for owner in (rank, 2 * cp_size - rank - 1)
        )
    return torch.cat(parts)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("source,target", DIRECTIONS)
@pytest.mark.parametrize("lengths", LENGTH_CASES)
def test_native_route_preserves_token_ownership(native_routes, cp_size, source, target, lengths):
    cu = _cu(lengths, unit=2 * cp_size)
    full = _tagged_tokens(int(cu[-1]))
    routes = [native_routes.build_thd_cp_partition_route(cu, cp_size, rank) for rank in range(cp_size)]
    send = []
    for rank, route in enumerate(routes):
        local = full[_native_rank_indices(cu, cp_size, rank, source)]
        index = getattr(route, f"{source}_index")
        send.append(local if index is None else local.index_select(0, index))
    for rank, route in enumerate(routes):
        parts = []
        for peer, peer_route in enumerate(routes):
            splits = getattr(peer_route, f"{source}_split_sizes")
            assert splits[rank] == getattr(route, f"{target}_split_sizes")[peer]
            parts.append(send[peer].narrow(0, sum(splits[:rank]), splits[rank]))
        received = torch.cat(parts)
        index = getattr(route, f"{target}_index")
        actual = received
        if index is not None:
            actual = torch.empty_like(received)
            actual.index_copy_(0, index, received)
        assert torch.equal(actual, full[_native_rank_indices(cu, cp_size, rank, target)])


def test_native_prebuild_reuses_both_directions_and_refreshes_for_next_batch(native_routes):
    from types import SimpleNamespace

    packed = _packed_seq_params(_cu([3, 1], unit=8))
    packed.cp_group = SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    native_routes.prebuild_thd_cp_partition_routes(packed)
    first = native_routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous")
    assert native_routes.get_thd_cp_partition_route(packed, "contiguous", "zigzag") is first

    packed.cu_seqlens_q = packed.cu_seqlens_kv = _cu([2, 2], unit=8)
    packed.cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    native_routes.prebuild_thd_cp_partition_routes(packed)
    refreshed = native_routes.get_thd_cp_partition_route(packed, "zigzag", "contiguous")
    assert refreshed is not first
    expected = native_routes.build_thd_cp_partition_route(packed.cu_seqlens_q, 2, 0)
    assert refreshed.zigzag_split_sizes == expected.zigzag_split_sizes
    assert refreshed.contiguous_split_sizes == expected.contiguous_split_sizes
    for name in ("zigzag_index", "contiguous_index"):
        actual, wanted = getattr(refreshed, name), getattr(expected, name)
        assert (actual is None) == (wanted is None)
        if wanted is not None:
            assert torch.equal(actual, wanted)


def test_native_routes_reject_invalid_packed_boundaries(native_routes):
    for cu, match in (
        (torch.tensor([8, 16]), "must start at 0"),
        (torch.tensor([0, 16, 8]), "nondecreasing"),
        (torch.tensor([0, 12]), "divisible by"),
    ):
        with pytest.raises(ValueError, match=match):
            native_routes.build_thd_cp_partition_route(cu, 4, 0)


@requires_legacy_routes
def test_route_cache_survives_interleaved_batches_and_refreshes_a_copied_batch():
    from copy import copy

    first_cu = _cu([3, 1], unit=8)
    second_cu = _cu([2, 2], unit=8)
    first, second = _packed_seq_params(first_cu), _packed_seq_params(second_cu)
    route = cpl.get_thd_cp_partition_route(first, first_cu, 4, 1, "zigzag", "contiguous")
    other = cpl.get_thd_cp_partition_route(second, second_cu, 4, 1, "zigzag", "contiguous")
    assert other is not route
    assert cpl.get_thd_cp_partition_route(first, first_cu, 4, 1, "zigzag", "contiguous") is route

    copied = copy(first)
    copied.cu_seqlens_q = copied.cu_seqlens_kv = second_cu
    refreshed = cpl.get_thd_cp_partition_route(copied, second_cu, 4, 1, "zigzag", "contiguous")
    assert refreshed is not route
    assert refreshed.input_split_sizes == other.input_split_sizes
    assert refreshed.output_split_sizes == other.output_split_sizes
    assert cpl.get_thd_cp_partition_route(first, first_cu, 4, 1, "zigzag", "contiguous") is route


@requires_legacy_routes
def test_route_cache_rebuilds_indices_for_a_different_device_without_cuda():
    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    source, target = "contiguous", "zigzag"
    original = cpl.get_thd_cp_partition_route(packed, cu, 4, 1, source, target)
    # Meta validates device-dependent indices without allocating CUDA memory.
    meta = cpl.get_thd_cp_partition_route(packed, cu, 4, 1, source, target, device=torch.device("meta"))
    assert meta is not original
    indices = meta.send_rows if meta.send_rows is not None else meta.recv_rows
    assert indices is not None and indices.device.type == "meta"
    restored = cpl.get_thd_cp_partition_route(packed, cu, 4, 1, source, target)
    assert restored is not meta
    for name in ("send_rows", "recv_rows"):
        actual, expected = getattr(restored, name), getattr(original, name)
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert torch.equal(actual, expected)


def _checkpoint_route_gloo_worker(rank, init_method, module_path):
    import gc
    import importlib.util
    import weakref
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist
    from torch.distributed.nn.functional import all_to_all_single
    from torch.utils.checkpoint import checkpoint

    # Load the same dependency revision as the parent, including an isolated review copy.
    spec = importlib.util.spec_from_file_location("gdn_layout_under_test", module_path)
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    group = dist.group.WORLD

    def cpu_all_to_all(cp_group, value, output_split_sizes, input_split_sizes):
        # MCore's collective allocates on CUDA; use PyTorch's real CPU collective/autograd boundary.
        output = value.new_empty((sum(output_split_sizes),) + tuple(value.shape[1:]))
        return all_to_all_single(output, value, output_split_sizes, input_split_sizes, group=cp_group)

    try:
        with patch.object(layout, "all_to_all", cpu_all_to_all):
            for use_reentrant, seq_dim in ((True, 0), (False, 1)):
                cu = _cu([3, 0, 1], unit=4)
                packed = _packed_seq_params(cu)
                full = torch.arange(int(cu[-1]) * 3, dtype=torch.float64).reshape(-1, 3) / 100
                zigzag = _native_rank_indices(cu, 2, rank, "zigzag")
                contiguous = _native_rank_indices(cu, 2, rank, "contiguous")
                initial = full[zigzag].movedim(0, seq_dim).clone().requires_grad_(True)
                activation_ref = weakref.ref(initial)

                def forward(value, packed_params=packed):
                    for layer in range(2):
                        to_contiguous = layout.get_thd_cp_partition_route(
                            packed_params, cu, 2, rank, "zigzag", "contiguous"
                        )
                        to_zigzag = layout.get_thd_cp_partition_route(
                            packed_params, cu, 2, rank, "contiguous", "zigzag"
                        )
                        value = layout.zigzag_to_contiguous_chunks(
                            value, group, seq_dim, cu, thd_cp_partition_route=to_contiguous
                        )
                        assert torch.equal(value, full.pow(2**layer)[contiguous].movedim(0, seq_dim))
                        value = layout.contiguous_to_zigzag_chunks(
                            value.square(), group, seq_dim, cu, thd_cp_partition_route=to_zigzag
                        )
                    return value

                with patch.object(
                    layout, "build_thd_cp_partition_route", wraps=layout.build_thd_cp_partition_route
                ) as build:
                    result = checkpoint(forward, initial, use_reentrant=use_reentrant)
                    assert torch.equal(result, initial.square().square())
                    result.sum().backward()
                    torch.testing.assert_close(initial.grad, 4 * initial.detach().pow(3), rtol=1e-12, atol=1e-12)
                    assert build.call_count == 2, "both layers and checkpoint replay must share two directional routes"

                # Keeping metadata alive must not retain consumed activations.
                del result, initial
                gc.collect()
                assert activation_ref() is None
                route_refs = [
                    weakref.ref(rows)
                    for route in (
                        packed.cp_partition_route_zigzag_to_contiguous,
                        packed.cp_partition_route_contiguous_to_zigzag,
                    )
                    for rows in (route.send_rows, route.recv_rows)
                    if rows is not None
                ]
                del forward, packed
                gc.collect()
                assert all(ref() is None for ref in route_refs), "route indices must die with the microbatch"
    finally:
        dist.destroy_process_group()


@requires_legacy_routes
@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_routes_preserve_gradients_and_cache_lifetime_through_real_checkpoint(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _checkpoint_route_gloo_worker,
        args=((tmp_path / "gloo-rendezvous").as_uri(), cpl.__file__),
        nprocs=2,
        join=True,
    )


@requires_legacy_routes
def test_route_cache_remains_usable_after_batch_device_transfer():
    from relax.backends.megatron.data import move_tensors_to_device

    cu = _cu([3, 1], unit=8)
    packed = _packed_seq_params(cu)
    original = cpl.get_thd_cp_partition_route(packed, cu, 4, 1, "contiguous", "zigzag")
    moved = move_tensors_to_device(packed, torch.device("cpu"))
    route = cpl.get_thd_cp_partition_route(moved, moved.cu_seqlens_q, 4, 1, "contiguous", "zigzag")
    assert route.input_split_sizes == original.input_split_sizes
    assert route.output_split_sizes == original.output_split_sizes
    assert torch.equal(route.send_rows, original.send_rows)


@requires_legacy_routes
@pytest.mark.parametrize("source,target", DIRECTIONS)
def test_route_built_during_inference_can_later_participate_in_autograd(source, target):
    cu = _cu([3, 1], unit=8)  # Versioned tensor, unlike the inference-only boundary case above.
    packed = _packed_seq_params(cu)
    with torch.inference_mode():
        cpl.get_thd_cp_partition_route(packed, cu, 4, 1, source, target)
    route = cpl.get_thd_cp_partition_route(packed, cu, 4, 1, source, target)
    value = torch.arange(route.local_source_length, dtype=torch.float64).requires_grad_(True)
    # Exercise the real gather/scatter autograd save, which rejects inference indices.
    rows = route.send_rows if route.send_rows is not None else route.recv_rows
    assert rows is not None
    if route.send_rows is not None:
        output = value.index_select(0, rows)
    else:
        output = torch.empty_like(value).index_copy_(0, rows, value)
    output.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))
