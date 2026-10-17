# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen2-VL pair-layout and per-projection Canonical LoRA computation graph.

Keep the SDK modules, Parameters and checkpoint keys. Only this model's forward
methods are adapted after standard PEFT injection; all adapter math/RNG remains
in the original Bridge implementation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairProjectionLayout:
    source: torch.Tensor
    destination: torch.Tensor
    packed_rows: int
    sequences: int
    width: int

    def unpack(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[:2] != (self.packed_rows, 1):
            raise ValueError("Qwen2-VL pair projections require packed [T, 1, hidden] inputs")
        return (
            value.new_zeros((self.sequences * self.width, value.shape[-1]))
            .index_copy(0, self.destination, value[:, 0].index_select(0, self.source))
            .reshape(self.sequences, self.width, value.shape[-1])
        )

    def pack(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[:2] != (self.sequences, self.width):
            raise ValueError("Qwen2-VL projection changed the pair layout")
        return (
            value.new_zeros((self.packed_rows, value.shape[-1]))
            .index_copy(
                0, self.source, value.reshape(self.sequences * self.width, -1).index_select(0, self.destination)
            )
            .unsqueeze(1)
        )


def pair_projection_layout(packed: Any, device: torch.device, total_tokens: int) -> PairProjectionLayout:
    """Use explicit logical lengths; an alignment dummy is never a pair row."""
    lengths = getattr(packed, "retrieval_pair_sequence_lengths", None)
    starts = getattr(packed, "cu_seqlens_q_cpu", None)
    if packed.qkv_format != "thd" or not lengths or len(lengths) % 2 or starts is None:
        raise ValueError("Qwen2-VL retrieval LoRA requires explicit complete-pair packed metadata")
    lengths = tuple(lengths)
    if len(starts) not in (len(lengths) + 1, len(lengths) + 2) or starts[0] != 0 or starts[-1] != total_tokens:
        raise ValueError("Qwen2-VL pair lengths do not match packed boundaries")
    if starts[len(lengths)] > total_tokens:
        raise ValueError("Qwen2-VL logical sequences exceed physical packed storage")
    if any(length <= 0 or length != starts[i + 1] - starts[i] for i, length in enumerate(lengths)):
        raise ValueError("Qwen2-VL retrieval does not support per-sequence padding or context parallelism")
    pairs = len(lengths) // 2
    width = (max(lengths) + 7) // 8 * 8
    source, destination = [], []
    for index, length in enumerate(lengths):
        row = index // 2 + (pairs if index % 2 else 0)
        source.extend(range(starts[index], starts[index] + length))
        destination.extend(range(row * width, row * width + length))
    return PairProjectionLayout(
        torch.tensor(source, dtype=torch.long, device=device),
        torch.tensor(destination, dtype=torch.long, device=device),
        total_tokens,
        len(lengths),
        width,
    )


_PROJECTION_LAYOUT: ContextVar[PairProjectionLayout | None] = ContextVar("qwen2_vl_projection_layout", default=None)


@contextmanager
def projection_layout(layout: PairProjectionLayout | None) -> Iterator[None]:
    token = _PROJECTION_LAYOUT.set(layout)
    try:
        yield
    finally:
        _PROJECTION_LAYOUT.reset(token)


def _adapter_forward(self: Any, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    layout = _PROJECTION_LAYOUT.get()
    original = self._qwen2_vl_original_forward
    if layout is None:
        return original(value, *args, **kwargs)
    return layout.pack(original(layout.unpack(value), *args, **kwargs))


def _native_adapter(module: Any, value: torch.Tensor) -> torch.Tensor:
    # The input is already dense. Do not apply the packed-layout wrapper twice.
    return module._qwen2_vl_original_forward(value)


def _linear_bias(module: Any) -> torch.Tensor | None:
    # Transformer Engine registers an empty tensor when use_bias=False.
    bias = getattr(module, "bias", None)
    if bias is None or bias.numel() == 0:
        return None
    if not getattr(module, "use_bias", True):
        raise ValueError("Qwen2-VL projection has a nonempty bias while use_bias=False")
    return bias


def _qkv_forward(self: Any, value: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    layout = _PROJECTION_LAYOUT.get()
    if layout is None or not self._adapter_enabled:
        return self._qwen2_vl_original_forward(value, *args, **kwargs)
    if args or kwargs:
        raise ValueError("Qwen2-VL native pair projections do not support additional linear arguments")
    from megatron.bridge.models.conversion.param_mapping import split_qkv_biases, split_qkv_weights

    weights = split_qkv_weights(self.to_wrap.config, self.to_wrap.weight)
    bias = _linear_bias(self.to_wrap)
    biases = split_qkv_biases(self.to_wrap.config, bias) if bias is not None else (None, None, None)
    dense = layout.unpack(value)
    outputs = []
    for name, weight, bias in zip(("q", "k", "v"), weights, biases, strict=True):
        base = F.linear(dense, weight, bias)
        adapter = getattr(self.adapter, "adapter_" + name, None)
        outputs.append(base if adapter is None else base + _native_adapter(adapter, dense))
    return layout.pack(self._interleave_qkv(*outputs)), None


def _fc1_forward(self: Any, value: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    layout = _PROJECTION_LAYOUT.get()
    if layout is None or not self._adapter_enabled:
        return self._qwen2_vl_original_forward(value, *args, **kwargs)
    if args or kwargs:
        raise ValueError("Qwen2-VL native pair projections do not support additional linear arguments")
    if _linear_bias(self.to_wrap) is not None:
        raise ValueError("Qwen2-VL gated MLP projections must be bias-free")
    gate_weight, up_weight = self.to_wrap.weight.chunk(2, dim=0)
    dense = layout.unpack(value)
    gate = F.linear(dense, gate_weight)
    gate_adapter = getattr(self.adapter, "adapter_gate", None)
    if gate_adapter is not None:
        gate = gate + _native_adapter(gate_adapter, dense)
    up = F.linear(dense, up_weight)
    up_adapter = getattr(self.adapter, "adapter_up", None)
    if up_adapter is not None:
        up = up + _native_adapter(up_adapter, dense)
    return layout.pack(torch.cat((gate, up), dim=-1)), None


def _bind(module: Any, forward: Any) -> None:
    if "_qwen2_vl_original_forward" in vars(module):
        return
    module._qwen2_vl_original_forward = module.forward
    module.forward = MethodType(forward, module)


def install_native_pair_lora(model: Any) -> int:
    """Model-local method dispatch; preserve every existing module and
    Parameter."""
    from megatron.bridge.peft.canonical_lora import LoRALinearSplitFC1UpGate, LoRALinearSplitQKV
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    from relax.models.qwen2_vl.model import Qwen2VLModel

    if not isinstance(model, Qwen2VLModel):
        return 0
    sizes = (
        model.config.tensor_model_parallel_size,
        model.config.pipeline_model_parallel_size,
        model.config.context_parallel_size,
    )
    if sizes != (1, 1, 1) or model.config.sequence_parallel:
        raise ValueError("Qwen2-VL pair LoRA requires TP=PP=CP=1 without sequence parallelism")
    for module in model.language_model.modules():
        if isinstance(module, (LoRALinearSplitQKV, LoRALinearSplitFC1UpGate)):
            if any(parameter.requires_grad for parameter in module.to_wrap.parameters()):
                raise ValueError("Qwen2-VL native pair LoRA requires frozen base projections")
    installed = 0
    for module in model.language_model.modules():
        if isinstance(module, ParallelLinearAdapter):
            _bind(module, _adapter_forward)
            installed += 1
        elif isinstance(module, LoRALinearSplitQKV):
            _bind(module, _qkv_forward)
        elif isinstance(module, LoRALinearSplitFC1UpGate):
            _bind(module, _fc1_forward)
    model._qwen2_vl_native_pair_lora = bool(installed)
    return installed


class PairLayoutTransformerLayer:
    """Lazy layer builder so optional Megatron imports stay in the GPU path."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        return _pair_layout_layer()(*args, **kwargs)


@lru_cache(maxsize=1)
def _pair_layout_layer() -> Any:
    from megatron.core.transformer.transformer_layer import TransformerLayer

    class _PairLayoutTransformerLayer(TransformerLayer):
        def forward(self, *args: Any, packed_seq_params: Any = None, **kwargs: Any) -> Any:
            # Full activation recomputation re-enters this same layer with the
            # immutable metadata captured by its own original forward call.
            layout = getattr(packed_seq_params, "qwen2_vl_projection_layout", None)
            with projection_layout(layout):
                return super().forward(*args, packed_seq_params=packed_seq_params, **kwargs)

    return _PairLayoutTransformerLayer
