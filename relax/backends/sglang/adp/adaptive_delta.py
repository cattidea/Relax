# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""ADP1: exact byte residuals, adaptive planes/ANS, and GPU in-place restore.

Offline selection measures transfer + reconstruction, not just entropy. The
implementation builds on the project's ZipNN-inspired XOR/ANS baseline. All
online payloads are pinned CPU memory. RAW packets contain B (directional).
"""

import hashlib
import json
import os
import statistics
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

from .xor_codec import GPUCodec, apply_xor


@triton.jit
def _planes_apply(
    dst, packed, N: tl.constexpr, STRIDE: tl.constexpr, MASK: tl.constexpr, ONES: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.full((BLOCK,), ONES, tl.uint8)
    p = 0
    for bit in tl.static_range(8):
        if MASK & (1 << bit):
            v = tl.load(packed + p * STRIDE + i // 8, i < N, other=0)
            d = d | (((v >> (i % 8)) & 1).to(tl.uint8) << bit)
            p += 1
    a = tl.load(dst + i, i < N, other=0)
    tl.store(dst + i, a ^ d, i < N)


@triton.jit
def _count_mismatch(a, b, count, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    different = (tl.load(a + i, i < N, other=0) != tl.load(b + i, i < N, other=0)) & (i < N)
    n = tl.sum(different.to(tl.int32), 0)
    tl.atomic_add(count, n.to(tl.int64), sem="relaxed")


@triton.jit
def _apply_add(dst, residual, N: tl.constexpr, BLOCK: tl.constexpr):
    """Restore raw bytes as (A + residual) mod 256; never interpret floats."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(dst + i, i < N, other=0).to(tl.uint16)
    d = tl.load(residual + i, i < N, other=0).to(tl.uint16)
    tl.store(dst + i, (a + d).to(tl.uint8), i < N)


def raw_view(t):
    return t.detach().reshape(-1).view(torch.uint8)


def pack_planes(x):
    """Drop all-zero/all-one XOR planes; bit-pack variable planes LSB first."""
    masks = 1 << np.arange(8, dtype=np.uint8)
    used = int(np.bitwise_or.reduce(x, initial=np.uint8(0)))
    ones = int(np.bitwise_and.reduce(x, initial=np.uint8(255)))
    variable = used & ~ones
    pieces = [np.packbits((x >> b) & 1, bitorder="little") for b in range(8) if variable & int(masks[b])]
    return (np.concatenate(pieces) if pieces else np.empty(0, np.uint8)), variable, ones


class AdaptiveDeltaPlan:
    VERSION = 1
    MAGIC = b"ADP1\x00\x00\x00\x00"
    MAX_BLOCK_BYTES = 256 * 1024**2

    def __init__(self, pairs, block_bytes=16 * 1024**2, policy="latency", emit=None, calibrate_repeats=3):
        assert policy in ("latency", "size")
        assert 16384 <= block_bytes <= self.MAX_BLOCK_BYTES and block_bytes % 8 == 0
        assert isinstance(calibrate_repeats, int) and calibrate_repeats > 0
        assert len({name for name, *_ in pairs}) == len(pairs), "duplicate runtime tensor name"
        self._a_bank = {name: a for name, a, b, dst in pairs}
        self._destinations = {name: dst for name, a, b, dst in pairs}
        self._state = "unknown"
        # This is offline preparation, outside switch timing. A stale/wrong
        # resident base must not silently survive in SKIP blocks.
        verify_bytes(self._destinations, self._a_bank)
        self.policy, self.block_bytes = policy, block_bytes
        self.stream = torch.cuda.current_stream()
        self.copy_stream = torch.cuda.Stream()
        self.codec = GPUCodec("ANS", self.stream)
        # Bit-plane packing can pad at most seven bytes.
        self.staging = [torch.empty(block_bytes + 8, dtype=torch.uint8, device="cuda") for _ in range(2)]
        self.decoded = torch.empty(block_bytes + 8, dtype=torch.uint8, device="cuda")
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.done = [torch.cuda.Event() for _ in range(2)]
        self.packets, self.tensors, self.choices, self.bindings = [], {}, [], []
        start = time.perf_counter()
        counts, size_by_mode = Counter(), Counter()
        logical_bytes, skipped_bytes = 0, 0
        for tensor_i, (name, a, b, dst) in enumerate(pairs):
            assert a.shape == b.shape == dst.shape and a.dtype == b.dtype == dst.dtype, name
            assert dst.is_cuda and (a.numel() == 0 or (a.is_pinned() and b.is_pinned())), name
            assert a.is_contiguous() and b.is_contiguous() and dst.is_contiguous(), name
            av, bv, dv = raw_view(a), raw_view(b), raw_view(dst)
            self.bindings.append((dst, dst.data_ptr(), tuple(dst.shape), tuple(dst.stride()), dst.dtype))
            logical_bytes += av.numel()
            self.tensors[name] = dict(
                shape=list(a.shape),
                dtype=str(a.dtype),
                bytes=av.numel(),
                a_sha256=hashlib.sha256(av.numpy()).hexdigest(),
                b_sha256=hashlib.sha256(bv.numpy()).hexdigest(),
                ptr=dst.data_ptr(),
            )
            for off in range(0, av.numel(), block_bytes):
                aa, bb, dd = av[off : off + block_bytes], bv[off : off + block_bytes], dv[off : off + block_bytes]
                x = np.bitwise_xor(aa.numpy(), bb.numpy())
                n = x.size
                if not np.any(x):
                    counts["SKIP"] += 1
                    skipped_bytes += n
                    continue
                base = dict(name=name, offset=off, n=n, dst=dd)
                candidates = [dict(base, mode="RAW", payload=bb)]
                if n >= 16384:
                    candidates.append(dict(base, mode="XOR_ANS", payload=self._compress(x), decoded_n=n))
                    # Modulo subtraction is defined on each stored byte. It is
                    # reversible for FP4/FP8/NaN bit patterns without float math.
                    delta = np.subtract(bb.numpy(), aa.numpy(), dtype=np.uint8)
                    candidates.append(dict(base, mode="ADD_ANS", payload=self._compress(delta), decoded_n=n))
                    if n % 2 == 0:
                        grouped = x.reshape(-1, 2).T.copy().reshape(-1)
                        candidates.append(
                            dict(base, mode="XOR_GROUP2_ANS", payload=self._compress(grouped), decoded_n=n)
                        )
                    packed, mask, ones = pack_planes(x)
                    planes = dict(base, mask=mask, ones=ones, stride=(n + 7) // 8)
                    candidates.append(dict(planes, mode="PLANES", payload=torch.from_numpy(packed).pin_memory()))
                    if packed.size:
                        candidates.append(
                            dict(planes, mode="PLANES_ANS", payload=self._compress(packed), decoded_n=packed.size)
                        )
                # Unprofitable compressed packets have no advantage over raw B.
                candidates = [p for p in candidates if p["mode"] == "RAW" or p["payload"].numel() < n]
                for p in candidates:
                    self._bind_packet(p)
                    # JIT, allocations and restoring A are outside candidate timings.
                    dd.copy_(aa, non_blocking=True)
                    self._serial_packet(p)
                    self.stream.synchronize()
                    assert torch.equal(dd.cpu(), bb), (name, off, p["mode"], "candidate decode mismatch")
                    measurements = []
                    for _ in range(calibrate_repeats):
                        dd.copy_(aa, non_blocking=True)
                        self.stream.synchronize()
                        before = time.perf_counter()
                        self._serial_packet(p)
                        self.stream.synchronize()
                        measurements.append(time.perf_counter() - before)
                    p["calibration_seconds"] = statistics.median(measurements)
                if policy == "size":
                    best = min(candidates, key=lambda p: (p["payload"].numel(), p["calibration_seconds"]))
                else:
                    fastest = min(p["calibration_seconds"] for p in candidates)
                    # Prefer a smaller packet when timings differ by less than 3%.
                    best = min(
                        (p for p in candidates if p["calibration_seconds"] <= fastest * 1.03),
                        key=lambda p: p["payload"].numel(),
                    )
                self.choices.append(
                    dict(
                        name=name,
                        offset=off,
                        bytes=n,
                        selected=best["mode"],
                        candidates=[
                            dict(mode=p["mode"], payload_bytes=p["payload"].numel(), seconds=p["calibration_seconds"])
                            for p in candidates
                        ],
                    )
                )
                self.packets.append(best)
                counts[best["mode"]] += 1
                size_by_mode[best["mode"]] += best["payload"].numel()
                dd.copy_(aa, non_blocking=True)
            if emit:
                emit(
                    "adaptive_prepare",
                    tensors=tensor_i + 1,
                    total_tensors=len(pairs),
                    payload_bytes=sum(size_by_mode.values()),
                    seconds=time.perf_counter() - start,
                )
        self.stream.synchronize()
        self._state = "A"
        torch.cuda.empty_cache()
        self.stats = dict(
            format="ADP1",
            version=1,
            policy=policy,
            block_bytes=block_bytes,
            input_tensor_bytes=logical_bytes,
            skipped_bytes=skipped_bytes,
            payload_bytes=sum(size_by_mode.values()),
            counts=dict(counts),
            payload_bytes_by_mode=dict(size_by_mode),
            scratch_bytes=3 * (block_bytes + 8),
            prepare_seconds=time.perf_counter() - start,
            selection="per-block measured serial pinned-H2D plus restore; final pipeline benchmark is authoritative",
        )

    def _compress(self, array):
        src = torch.from_numpy(array).to("cuda")
        encoded = self.codec.encode(src)
        out = torch.empty(encoded.numel(), dtype=torch.uint8, device="cpu", pin_memory=True)
        out.copy_(encoded, non_blocking=True)
        self.stream.synchronize()
        return out

    def _bind_packet(self, p):
        if p["mode"].endswith("ANS"):
            n = p["decoded_n"]
            p["out"] = self.codec.nvcomp.as_array(self.decoded[:n])
            p["src"] = [self.codec.nvcomp.as_array(s[: p["payload"].numel()]) for s in self.staging]
            p["config"] = self.codec.config(n)[1]

    def _restore(self, p, slot):
        mode = p["mode"]
        if mode.endswith("ANS"):
            self.codec.codec.decode(p["src"][slot], out=p["out"], decompression_config=p["config"])
            residual = self.decoded[: p["decoded_n"]]
        else:
            residual = self.staging[slot]
        if mode.startswith("PLANES"):
            _planes_apply[(triton.cdiv(p["n"], 2048),)](
                p["dst"], residual, p["n"], p["stride"], p["mask"], p["ones"], 2048
            )
        elif mode == "XOR_GROUP2_ANS":
            apply_xor(p["dst"], residual, grouped=True)
        elif mode == "XOR_ANS":
            # Existing kernel accepts arbitrary byte count except odd assertions.
            from .xor_codec import _apply_xor

            _apply_xor[(triton.cdiv(p["n"], 4096),)](p["dst"], residual, p["n"], False, 4096)
        elif mode == "ADD_ANS":
            _apply_add[(triton.cdiv(p["n"], 4096),)](p["dst"], residual, p["n"], 4096)
        else:
            raise ValueError(mode)

    def _serial_packet(self, p):
        if p["mode"] == "RAW":
            p["dst"].copy_(p["payload"], non_blocking=True)
        else:
            if p["payload"].numel():
                self.staging[0][: p["payload"].numel()].copy_(p["payload"], non_blocking=True)
            self._restore(p, 0)

    def mark_base_restored(self, *, verify=True):
        """Re-arm A->B after the caller restores A outside the timed interval.

        verify=False is only for a caller that has just completed and
        synchronized its own trusted A snapshot copy. A directional RAW/XOR/ADD
        package cannot be reused on B or inverted to restore A. After
        release_base_snapshots(), verification hashes resident GPU bytes with
        bounded temporary CPU memory. Both verification paths cover package
        tensors (including their SKIP blocks); they do not validate model
        tensors absent from this package.
        """
        # A failed verification must also invalidate any previous A-ready state.
        self._state = "unknown"
        if verify:
            if self._a_bank is None:
                verify_tensor_hashes(self._destinations, self.tensors)
            else:
                verify_bytes(self._destinations, self._a_bank)
        self._state = "A"

    def release_base_snapshots(self):
        """Drop this plan's CPU A references; online apply only needs resident
        A.

        This reports references dropped, not process memory freed: callers and
        PyTorch's host allocator may retain storage. Fresh RAW packet views may
        still retain a larger B tensor; loading an ADP1 package or compacting
        its payloads separates those allocations. This method addresses A only.
        """
        bank, self._a_bank = self._a_bank, None
        return dict(
            released_tensor_references=len(bank) if bank is not None else 0,
            referenced_tensor_bytes=sum(t.numel() * t.element_size() for t in bank.values())
            if bank is not None
            else 0,
            remaining_base_snapshot_bytes=0,
        )

    @torch.no_grad()
    def apply(self, overlap=True):
        assert self._state == "A", "A->B plan requires A; restore A then mark_base_restored()"
        assert all(
            t.data_ptr() == ptr and tuple(t.shape) == shape and tuple(t.stride()) == stride and t.dtype == dtype
            for t, ptr, shape, stride, dtype in self.bindings
        ), "runtime storage changed"
        caller_stream = torch.cuda.current_stream()
        assert caller_stream.device == self.stream.device, "wrong current CUDA device"
        self.stream.wait_stream(caller_stream)
        # nvCOMP is bound to self.stream. Triton and direct copies must execute
        # on that same stream even when apply() is called from another stream.
        self._state = "updating"
        with torch.cuda.stream(self.stream):
            self._apply(overlap)
        self._state = "B"

    def _apply(self, overlap):
        if not overlap:
            for p in self.packets:
                self._serial_packet(p)
        else:
            self.copy_stream.wait_stream(self.stream)
            used = [False, False]
            index = 0
            for p in self.packets:
                if p["mode"] == "RAW":
                    with torch.cuda.stream(self.copy_stream):
                        p["dst"].copy_(p["payload"], non_blocking=True)
                    continue
                slot = index % 2
                with torch.cuda.stream(self.copy_stream):
                    if used[slot]:
                        self.copy_stream.wait_event(self.done[slot])
                    if p["payload"].numel():
                        self.staging[slot][: p["payload"].numel()].copy_(p["payload"], non_blocking=True)
                    self.ready[slot].record(self.copy_stream)
                self.stream.wait_event(self.ready[slot])
                self._restore(p, slot)
                self.done[slot].record(self.stream)
                used[slot] = True
                index += 1
            self.stream.wait_stream(self.copy_stream)
        self.stream.synchronize()

    def save(self, path, provenance=None):
        """Persist selected packets; loading/preparation is never on the timed
        path."""
        manifest = dict(
            format="ADP1",
            version=self.VERSION,
            policy=self.policy,
            block_bytes=self.block_bytes,
            tensors={name: {k: v for k, v in meta.items() if k != "ptr"} for name, meta in self.tensors.items()},
            packets=[],
            stats=self.stats,
            provenance=provenance or {},
        )
        offset = 0
        for p in self.packets:
            meta = {k: p[k] for k in ("name", "offset", "n", "mode", "decoded_n", "mask", "ones", "stride") if k in p}
            meta.update(
                payload_offset=offset,
                payload_bytes=p["payload"].numel(),
                sha256=hashlib.sha256(p["payload"].numpy()).hexdigest(),
            )
            manifest["packets"].append(meta)
            offset += p["payload"].numel()
        header = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            for p in self.packets:
                f.write(memoryview(p["payload"].numpy()))
        tmp.replace(path)
        return dict(file=str(path), bytes=path.stat().st_size, metadata_bytes=16 + len(header))

    @classmethod
    def load(cls, path, destinations, a_bank=None, *, verify_base=True, expected_sha256=None):
        """Load a directed package into pinned RAM and bind existing GPU
        storage.

        An optional CPU a_bank must describe resident A. Without it, validation
        hashes package tensors' resident GPU bytes using bounded temporary CPU
        memory, and no A snapshot is retained. The caller quiesces inference
        before loading. With verify_base=False, apply remains disabled until
        mark_base_restored(). expected_sha256 authenticates metadata and
        payload from this same open stream before the prepared plan is
        returned. Legacy callers may omit it.
        """

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        def integer(value):
            return type(value) is int

        require(
            expected_sha256 is None
            or (
                isinstance(expected_sha256, str)
                and len(expected_sha256) == 64
                and all(c in "0123456789abcdef" for c in expected_sha256)
            ),
            "invalid expected package SHA256",
        )
        file_digest = hashlib.sha256() if expected_sha256 is not None else None
        path = Path(path)
        with path.open("rb") as f:
            magic = f.read(8)
            require(magic == cls.MAGIC, "invalid ADP1 magic")
            raw_length = f.read(8)
            require(len(raw_length) == 8, "truncated ADP1 size")
            length = struct.unpack("<Q", raw_length)[0]
            require(0 < length < 64 * 1024**2, "invalid metadata size")
            raw_header = f.read(length)
            require(len(raw_header) == length, "truncated ADP1 metadata")
            if file_digest is not None:
                file_digest.update(magic)
                file_digest.update(raw_length)
                file_digest.update(raw_header)
            meta = json.loads(raw_header)
            require(meta["format"] == "ADP1" and meta["version"] == cls.VERSION, "invalid ADP1 version")
            block_bytes = meta["block_bytes"]
            require(
                integer(block_bytes) and 16384 <= block_bytes <= cls.MAX_BLOCK_BYTES and block_bytes % 8 == 0,
                "invalid block_bytes",
            )
            require(meta["policy"] in ("size", "latency"), "invalid policy")
            require(isinstance(meta["tensors"], dict) and isinstance(meta["packets"], list), "invalid manifest")
            # Validate all dimensions and offsets before allocating or binding
            # GPU buffers. No decoded shape is trusted merely because it is JSON.
            for name, info in meta["tensors"].items():
                require(name in destinations, f"missing tensor: {name}")
                dst = destinations[name]
                require(list(dst.shape) == info["shape"] and str(dst.dtype) == info["dtype"], f"layout: {name}")
                require(dst.is_cuda and dst.is_contiguous(), f"storage: {name}")
                require(dst.device == torch.device("cuda", torch.cuda.current_device()), f"device: {name}")
                require(integer(info["bytes"]) and info["bytes"] == dst.numel() * dst.element_size(), f"bytes: {name}")
                for key in ("a_sha256", "b_sha256"):
                    digest = info[key]
                    require(
                        isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                        f"invalid {key}: {name}",
                    )
                if a_bank is not None:
                    require(name in a_bank, f"missing base tensor: {name}")
                    base = a_bank[name]
                    require(
                        base.shape == dst.shape
                        and base.dtype == dst.dtype
                        and base.device.type == "cpu"
                        and base.is_contiguous(),
                        f"base layout: {name}",
                    )
                    assert hashlib.sha256(raw_view(base).numpy()).hexdigest() == info["a_sha256"], name
            expected_offset, intervals = 0, {}
            for p in meta["packets"]:
                mode = p["mode"]
                require(
                    mode in {"RAW", "PLANES", "PLANES_ANS", "XOR_ANS", "XOR_GROUP2_ANS", "ADD_ANS"}, "invalid mode"
                )
                require(p["name"] in meta["tensors"], "unknown packet tensor")
                require(
                    all(integer(p[key]) for key in ("payload_offset", "payload_bytes", "offset", "n")),
                    "packet dimensions must be integers",
                )
                n, size, offset = p["n"], p["payload_bytes"], p["offset"]
                require(p["payload_offset"] == expected_offset, "noncontiguous payload offsets")
                require(0 < n <= block_bytes and 0 <= size <= block_bytes + 8, "invalid packet size")
                require(0 <= offset <= meta["tensors"][p["name"]]["bytes"] - n, "packet outside tensor")
                intervals.setdefault(p["name"], []).append((offset, offset + n))
                if mode == "RAW":
                    require(size == n, "RAW payload length must equal destination length")
                elif mode.startswith("PLANES"):
                    mask, ones, stride = p["mask"], p["ones"], p["stride"]
                    require(
                        all(integer(v) for v in (mask, ones, stride)) and 0 <= mask < 256 and 0 <= ones < 256,
                        "invalid plane masks",
                    )
                    require(mask & ones == 0 and stride == (n + 7) // 8, "invalid plane layout")
                    packed_n = mask.bit_count() * stride
                    require(packed_n <= block_bytes + 8, "plane output outside buffer")
                    if mode == "PLANES":
                        require(size == packed_n, "plane payload size mismatch")
                    else:
                        require(
                            integer(p["decoded_n"]) and p["decoded_n"] == packed_n and packed_n > 0 and size > 0,
                            "invalid compressed plane size",
                        )
                else:
                    require(
                        integer(p["decoded_n"]) and p["decoded_n"] == n and size > 0,
                        "invalid decoded byte-residual size",
                    )
                    require(mode != "XOR_GROUP2_ANS" or n % 2 == 0, "odd grouped XOR size")
                expected_offset += size
            for name, spans in intervals.items():
                spans.sort()
                require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), f"overlapping packets: {name}")
            payload_start = f.tell()
            require(os.fstat(f.fileno()).st_size == payload_start + expected_offset, "ADP1 payload size mismatch")
            self = cls.__new__(cls)
            self.policy, self.block_bytes = meta["policy"], meta["block_bytes"]
            self.stream = torch.cuda.current_stream()
            self.copy_stream = torch.cuda.Stream()
            self.codec = GPUCodec("ANS", self.stream)
            self.staging = [torch.empty(self.block_bytes + 8, dtype=torch.uint8, device="cuda") for _ in range(2)]
            self.decoded = torch.empty(self.block_bytes + 8, dtype=torch.uint8, device="cuda")
            self.ready = [torch.cuda.Event() for _ in range(2)]
            self.done = [torch.cuda.Event() for _ in range(2)]
            self.tensors, self.stats, self.choices, self.packets = meta["tensors"], meta["stats"], [], []
            self._a_bank = {name: a_bank[name] for name in self.tensors} if a_bank is not None else None
            self._destinations = {name: destinations[name] for name in self.tensors}
            self._state = "unknown"
            self.bindings = []
            for name, info in self.tensors.items():
                dst = destinations[name]
                info["ptr"] = dst.data_ptr()
                self.bindings.append((dst, dst.data_ptr(), tuple(dst.shape), tuple(dst.stride()), dst.dtype))
            for p in meta["packets"]:
                dst = raw_view(destinations[p["name"]])
                payload = torch.empty(p["payload_bytes"], dtype=torch.uint8, pin_memory=True, device="cpu")
                require(f.readinto(memoryview(payload.numpy())) == p["payload_bytes"], "truncated ADP1")
                require(hashlib.sha256(payload.numpy()).hexdigest() == p["sha256"], "corrupt packet")
                if file_digest is not None:
                    file_digest.update(memoryview(payload.numpy()))
                p.update(payload=payload, dst=dst[p["offset"] : p["offset"] + p["n"]])
                self._bind_packet(p)
                self.packets.append(p)
            require(f.tell() == payload_start + expected_offset and not f.read(1), "extra ADP1 data")
            require(
                file_digest is None or file_digest.hexdigest() == expected_sha256, "ADP1 complete-file SHA256 mismatch"
            )
        if verify_base:
            self.mark_base_restored(verify=True)
        return self


@torch.no_grad()
def verify_tensor_hashes(destinations, tensors, block_bytes=64 * 1024**2):
    """Offline validation of package tensors without retaining CPU A snapshots.

    One pageable CPU buffer is reused for synchronous D2H chunks. No full
    tensor CPU copy is built, and no GPU scratch or persistent CPU tensor is
    retained.
    """
    assert type(block_bytes) is int and block_bytes > 0, "invalid hash chunk size"
    maximum = max((meta["bytes"] for meta in tensors.values()), default=0)
    scratch = torch.empty(min(block_bytes, maximum), dtype=torch.uint8, device="cpu")
    total = 0
    for name, meta in tensors.items():
        dst = destinations[name]
        assert dst.device == torch.device("cuda", torch.cuda.current_device()), name
        assert dst.is_contiguous() and list(dst.shape) == meta["shape"] and str(dst.dtype) == meta["dtype"], name
        view = raw_view(dst)
        assert view.numel() == meta["bytes"], name
        digest = hashlib.sha256()
        for offset in range(0, view.numel(), block_bytes):
            count = min(block_bytes, view.numel() - offset)
            chunk = scratch[:count]
            chunk.copy_(view[offset : offset + count], non_blocking=False)
            digest.update(memoryview(chunk.numpy()))
        assert digest.hexdigest() == meta["a_sha256"], f"resident base SHA-256 mismatch: {name}"
        total += view.numel()
    return dict(bytes=total, passed=True, temporary_cpu_bytes=scratch.numel(), retained_base_snapshot_bytes=0)


@torch.no_grad()
def verify_bytes(destinations, bank, block_bytes=64 * 1024**2):
    """Compare every byte, accumulating on GPU to avoid one CPU sync per
    tensor."""
    scratch = torch.empty(block_bytes, dtype=torch.uint8, device="cuda")
    count = torch.zeros((), dtype=torch.int64, device="cuda")
    total = 0
    for name, dst in destinations.items():
        assert dst.device == torch.device("cuda", torch.cuda.current_device()), name
        assert bank[name].device.type == "cpu" and bank[name].shape == dst.shape and bank[name].dtype == dst.dtype, (
            name
        )
        assert dst.is_contiguous() and bank[name].is_contiguous(), name
        a, b = raw_view(dst), raw_view(bank[name])
        assert a.numel() == b.numel(), name
        for off in range(0, a.numel(), block_bytes):
            n = min(block_bytes, a.numel() - off)
            scratch[:n].copy_(b[off : off + n], non_blocking=True)
            _count_mismatch[(triton.cdiv(n, 4096),)](a[off : off + n], scratch, count, n, 4096)
        total += a.numel()
    errors = count.item()
    assert errors == 0, f"{errors} weight bytes differ"
    return dict(bytes=total, mismatched_bytes=errors, passed=True)
