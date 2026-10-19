# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Batch existing ADP1 ANS frames, then restore every packet in one GPU kernel.

This executor does not re-encode or change the original size-plan's codec
choices. Preparation packs existing frames and parses their headers once to
make an nvCOMP batch DecompressConfig. Online each bounded group has at most
one H2D, one nvCOMP *API* decode call, and one descriptor-driven restore
kernel. nvCOMP may launch multiple internal kernels; the API-call count is not
a kernel count.
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl


_BLOCK = 4096
_COLUMNS = 9
_KINDS = {"RAW": 0, "XOR_ANS": 1, "ADD_ANS": 2, "XOR_GROUP2_ANS": 3, "PLANES": 4, "PLANES_ANS": 4}


def batch_layout(sizes, group_bytes=64 * 1024**2, alignment=256):
    """CPU-only layout for (encoded bytes, decoded bytes) per packet.

    decoded bytes is zero for RAW/direct PLANES. Both the encoded input arena
    and decoded output arena must independently fit group_bytes. Zero-payload
    constant-plane packets still get a descriptor. None requests one full
    group.
    """
    sizes = list(sizes)
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    if group_bytes is not None and (type(group_bytes) is not int or group_bytes <= 0):
        raise ValueError("group_bytes must be positive or None")
    for payload, decoded in sizes:
        if type(payload) is not int or type(decoded) is not int or min(payload, decoded) < 0:
            raise ValueError("packet sizes must be nonnegative integers")
        if decoded and not payload:
            raise ValueError("an ANS frame cannot have an empty payload")
        if group_bytes is not None and max(payload, decoded) > group_bytes:
            raise ValueError("group_bytes cannot fit the largest independent frame")
    groups, current = [], []
    used_input = used_output = bank_offset = 0
    align = lambda n: (n + alignment - 1) // alignment * alignment
    for index, (payload, decoded) in enumerate(sizes):
        input_offset = align(used_input) if payload else used_input
        output_offset = align(used_output) if decoded else used_output
        if current and group_bytes is not None and max(input_offset + payload, output_offset + decoded) > group_bytes:
            groups.append(
                dict(bank_offset=bank_offset, input_bytes=used_input, decoded_bytes=used_output, packets=current)
            )
            bank_offset += used_input
            used_input = used_output = input_offset = output_offset = 0
            current = []
        current.append(
            dict(
                index=index,
                input_offset=input_offset,
                payload_bytes=payload,
                output_offset=output_offset,
                decoded_bytes=decoded,
            )
        )
        used_input, used_output = input_offset + payload, output_offset + decoded
    if current:
        groups.append(
            dict(bank_offset=bank_offset, input_bytes=used_input, decoded_bytes=used_output, packets=current)
        )
    return groups


def _descriptor_rows(plan, group):
    rows = []
    for item in group["packets"]:
        packet = plan.packets[item["index"]]
        ans = packet["mode"].endswith("ANS")
        source_offset = item["output_offset"] if ans else item["input_offset"]
        for tile in range(0, packet["n"], _BLOCK):
            # kind, arena, destination pointer, output bytes, output tile offset,
            # source offset, plane stride/mask/constant-one mask.
            rows.append(
                (
                    _KINDS[packet["mode"]],
                    int(ans),
                    packet["dst"].data_ptr(),
                    packet["n"],
                    tile,
                    source_offset,
                    packet.get("stride", 0),
                    packet.get("mask", 0),
                    packet.get("ones", 0),
                )
            )
    return rows


@triton.jit
def _batch_restore(packed, decoded, descriptors, BLOCK: tl.constexpr):
    row = tl.program_id(0) * 9
    kind = tl.load(descriptors + row)
    arena = tl.load(descriptors + row + 1)
    dst = tl.load(descriptors + row + 2).to(tl.pointer_type(tl.uint8))
    n = tl.load(descriptors + row + 3)
    offset = tl.load(descriptors + row + 4)
    source_offset = tl.load(descriptors + row + 5)
    stride = tl.load(descriptors + row + 6)
    mask = tl.load(descriptors + row + 7)
    ones = tl.load(descriptors + row + 8).to(tl.uint8)
    if arena == 1:
        source = decoded + source_offset
    else:
        source = packed + source_offset
    i = offset + tl.arange(0, BLOCK)
    valid = i < n
    if kind == 0:
        value = tl.load(source + i, valid, other=0)
    elif kind == 4:
        delta = tl.full((BLOCK,), 0, tl.uint8) + ones
        plane = 0
        for bit in tl.static_range(8):
            if mask & (1 << bit):
                bits = tl.load(source + plane * stride + i // 8, valid, other=0)
                delta = delta | (((bits >> (i % 8)) & 1).to(tl.uint8) << bit)
                plane += 1
        value = tl.load(dst + i, valid, other=0) ^ delta
    else:
        if kind == 3:
            j = i // 2 + (i % 2) * (n // 2)
        else:
            j = i
        residual = tl.load(source + j, valid, other=0)
        old = tl.load(dst + i, valid, other=0)
        if kind == 2:
            value = (old.to(tl.uint16) + residual.to(tl.uint16)).to(tl.uint8)
        else:
            value = old ^ residual
    tl.store(dst + i, value, valid)


class BatchedDeltaPlan:
    """Bounded batched execution of an AdaptiveDeltaPlan with policy='size'.

    Input frames are copied byte-for-byte, including after loading a .adp file.
    At most two input buffers plus one decoded-output arena are allocated. An
    original plan and its wrappers share the A->B guard and cannot run
    concurrently. With compact_sources=True they also share one pinned CPU
    payload bank.
    """

    def __init__(self, plan, group_bytes=64 * 1024**2, *, compact_sources=True, alignment=256, emit=None):
        if plan.policy != "size":
            raise ValueError("BatchedDeltaPlan currently supports only policy='size'")
        if any(packet["mode"] not in _KINDS for packet in plan.packets):
            raise ValueError("unsupported ADP1 packet mode")
        if torch.cuda.current_stream().device != plan.stream.device:
            raise ValueError("construct on the original plan's CUDA device")
        started = time.perf_counter()
        self.plan, self.stream = plan, plan.stream
        sizes = [
            (packet["payload"].numel(), packet["decoded_n"] if packet["mode"].endswith("ANS") else 0)
            for packet in plan.packets
        ]
        self.groups = batch_layout(sizes, group_bytes, alignment)
        bank_bytes = sum(group["input_bytes"] for group in self.groups)
        self.bank = torch.empty(bank_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        self.bank.zero_()
        self.input_capacity = max(1, max((group["input_bytes"] for group in self.groups), default=0))
        self.decoded_capacity = max(1, max((group["decoded_bytes"] for group in self.groups), default=0))
        self.buffer_count = min(2, max(1, len(self.groups)))
        self.buffers = [
            torch.empty(self.input_capacity, dtype=torch.uint8, device=self.stream.device)
            for _ in range(self.buffer_count)
        ]
        self.decoded = torch.empty(self.decoded_capacity, dtype=torch.uint8, device=self.stream.device)
        self.copy_stream = torch.cuda.Stream(device=self.stream.device)
        self.ready = [torch.cuda.Event() for _ in self.buffers]
        self.done = [torch.cuda.Event() for _ in self.buffers]
        metadata_bytes = 0
        with torch.cuda.stream(self.stream):
            for group in self.groups:
                begin = group["bank_offset"]
                group["srcs"] = [[] for _ in self.buffers]
                group["outs"] = []
                for item in group["packets"]:
                    packet = plan.packets[item["index"]]
                    off, size = item["input_offset"], item["payload_bytes"]
                    self.bank[begin + off : begin + off + size].copy_(packet["payload"])
                    if item["decoded_bytes"]:
                        for slot, buffer in enumerate(self.buffers):
                            group["srcs"][slot].append(plan.codec.nvcomp.as_array(buffer[off : off + size]))
                        out_off, out_size = item["output_offset"], item["decoded_bytes"]
                        group["outs"].append(plan.codec.nvcomp.as_array(self.decoded[out_off : out_off + out_size]))
                rows = _descriptor_rows(plan, group)
                group["tiles"] = len(rows)
                group["descriptors"] = torch.tensor(rows, dtype=torch.int64, device=self.stream.device)
                metadata_bytes += len(rows) * _COLUMNS * 8
                if group["outs"]:
                    # Parse the original independent frame headers once, offline.
                    # Do not assume scalar compression configs can be concatenated.
                    size = group["input_bytes"]
                    self.buffers[0][:size].copy_(self.bank[begin : begin + size], non_blocking=True)
                    group["config"] = plan.codec.codec.decompression_config(group["srcs"][0])
                else:
                    group["config"] = None
        self.stream.synchronize()
        if compact_sources:
            for group in self.groups:
                begin = group["bank_offset"]
                for item in group["packets"]:
                    off, size = item["input_offset"], item["payload_bytes"]
                    plan.packets[item["index"]]["payload"] = self.bank[begin + off : begin + off + size]
        storages = {}
        for tensor in [self.bank, *(packet["payload"] for packet in plan.packets)]:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        self.stats = dict(
            executor="batched_ADP1",
            policy="size",
            groups=len(self.groups),
            requested_group_bytes=group_bytes,
            alignment=alignment,
            original_packets=len(plan.packets),
            original_ans_packets=sum(packet["mode"].endswith("ANS") for packet in plan.packets),
            online_h2d_calls=sum(group["input_bytes"] > 0 for group in self.groups),
            online_batch_decode_api_calls=sum(bool(group["outs"]) for group in self.groups),
            online_restore_kernel_launches=sum(group["tiles"] > 0 for group in self.groups),
            original_payload_bytes=sum(payload for payload, decoded in sizes),
            transferred_bytes=bank_bytes,
            largest_input_group_bytes=self.input_capacity,
            largest_decoded_group_bytes=self.decoded_capacity,
            transfer_buffers=self.buffer_count,
            additional_gpu_buffer_bytes=self.input_capacity * self.buffer_count + self.decoded_capacity,
            additional_gpu_metadata_bytes=metadata_bytes,
            allocated_pinned_bank_bytes=bank_bytes,
            steady_payload_storage_bytes=sum(storages.values()),
            source_payloads_compacted=compact_sources,
            reencoded=False,
            call_count_scope="decode API calls; nvCOMP internal GPU kernel count is not measured",
            memory_scope="Wrapper buffers/descriptors and live package storage; excludes original plan scratch, "
            "nvCOMP internal workspace, allocator caching, A/B snapshots, other retained wrappers",
            prepare_seconds=time.perf_counter() - started,
        )
        if emit:
            emit("batched_prepared", **self.stats)

    def mark_base_restored(self, *, verify=True):
        return self.plan.mark_base_restored(verify=verify)

    def _restore_group(self, group, slot):
        if group["outs"]:
            self.plan.codec.codec.decode(group["srcs"][slot], out=group["outs"], decompression_config=group["config"])
        if group["tiles"]:
            _batch_restore[(group["tiles"],)](self.buffers[slot], self.decoded, group["descriptors"], _BLOCK)

    @torch.no_grad()
    def apply(self, overlap=True):
        plan = self.plan
        assert plan._state == "A", "restore and mark A before applying the directed batch"
        assert all(
            t.data_ptr() == ptr and tuple(t.shape) == shape and tuple(t.stride()) == stride and t.dtype == dtype
            for t, ptr, shape, stride, dtype in plan.bindings
        ), "runtime storage changed"
        current = torch.cuda.current_stream()
        assert current.device == self.stream.device, "wrong current CUDA device"
        self.stream.wait_stream(current)
        plan._state = "updating"
        with torch.cuda.stream(self.stream):
            if not overlap:
                for group in self.groups:
                    begin, size = group["bank_offset"], group["input_bytes"]
                    if size:
                        self.buffers[0][:size].copy_(self.bank[begin : begin + size], non_blocking=True)
                    self._restore_group(group, 0)
            else:
                self.copy_stream.wait_stream(self.stream)
                used = [False] * self.buffer_count
                for index, group in enumerate(self.groups):
                    slot = index % self.buffer_count
                    begin, size = group["bank_offset"], group["input_bytes"]
                    with torch.cuda.stream(self.copy_stream):
                        if used[slot]:
                            self.copy_stream.wait_event(self.done[slot])
                        if size:
                            self.buffers[slot][:size].copy_(self.bank[begin : begin + size], non_blocking=True)
                        self.ready[slot].record(self.copy_stream)
                    self.stream.wait_event(self.ready[slot])
                    self._restore_group(group, slot)
                    self.done[slot].record(self.stream)
                    used[slot] = True
                self.stream.wait_stream(self.copy_stream)
            self.stream.synchronize()
        plan._state = "B"
