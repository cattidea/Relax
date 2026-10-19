# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU ANS codec and lossless byte XOR used by the ADP1 runtime.

The transform acts on bytes, never subtracts floating-point weights. Checkpoint
preparation and allocations are outside the timed switch. This is an
engineering baseline borrowing the XOR/grouping idea from ZipNN, not a novelty
claim.
"""

import pathlib
import sys
from types import ModuleType
from typing import Any

import torch
import triton
import triton.language as tl


def load_nvcomp() -> ModuleType:
    # A regular global `nvidia` package can hide the venv namespace packages.
    # Extend only this process's import path; do not change installed packages.
    import nvidia

    for p in sys.path:
        directory = pathlib.Path(p) / "nvidia"
        if directory.is_dir() and str(directory) not in nvidia.__path__:
            nvidia.__path__.append(str(directory))
    from nvidia import nvcomp

    return nvcomp


@triton.jit
def _apply_xor(dst, residual, N: tl.constexpr, GROUPED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    j = i // 2 + (i % 2) * (N // 2) if GROUPED else i
    a = tl.load(dst + i, i < N, other=0)
    d = tl.load(residual + j, i < N, other=0)
    tl.store(dst + i, a ^ d, i < N)


def apply_xor(dst: torch.Tensor, residual: torch.Tensor, grouped: bool = True) -> None:
    assert dst.dtype == residual.dtype == torch.uint8
    assert dst.is_contiguous() and dst.numel() % 2 == 0
    _apply_xor[(triton.cdiv(dst.numel(), 4096),)](dst, residual, dst.numel(), grouped, 4096)


class GPUCodec:
    def __init__(self, algorithm: str = "ANS", stream: torch.cuda.Stream | None = None) -> None:
        self.nvcomp = load_nvcomp()
        self.stream = stream or torch.cuda.current_stream()
        self.codec = self.nvcomp.Codec(
            algorithm=algorithm, cuda_stream=self.stream.cuda_stream, uncomp_chunk_size=65536
        )
        self.algorithm = algorithm
        self.configs = {}

    def config(self, n: int) -> tuple[Any, Any]:
        if n not in self.configs:
            comp = self.codec.compression_config(n)
            self.configs[n] = (comp, self.codec.decompression_config(comp))
        return self.configs[n]

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        comp, _ = self.config(data.numel())
        result = self.codec.encode(self.nvcomp.as_array(data), compression_config=comp)
        # DLPack exposes allocation capacity; buffer_size is the actual payload.
        tensor = torch.from_dlpack(result).view(torch.uint8)[: result.buffer_size]
        # nvCOMP's DLPack view does not keep its allocation owner alive.
        tensor._nvcomp_owner = result
        return tensor

    def decode(self, data: torch.Tensor, out: torch.Tensor) -> None:
        _, decomp = self.config(out.numel())
        self.codec.decode(self.nvcomp.as_array(data), out=self.nvcomp.as_array(out), decompression_config=decomp)
