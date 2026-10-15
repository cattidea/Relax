# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""The native LamRA one-way InfoNCE and its detached distributed gather.

Every rank computes the complete global loss, with gradients attached only to
its own query and positive features. The backend then averages parameter
gradients. This intentionally preserves the native recipe's 1/world-size
gradient scale; neither an autograd gather nor a world-size multiplier is used.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


def gather_with_local_grad(features: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
    """Gather detached features, retaining this rank's autograd connection."""
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return features
    gathered = [torch.empty_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features.contiguous(), group=process_group)
    gathered[dist.get_rank(process_group)] = features
    return torch.cat(gathered, dim=0)


def native_infonce(
    query: torch.Tensor,
    positive: torch.Tensor,
    *,
    temperature: float = 0.05,
    process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Mean query-to-positive CE over the complete contrastive batch.

    ``None`` is permitted only without distributed initialization. No dtype
    conversion is applied: the caller's AMP context owns the numeric protocol.
    """
    if query.ndim != 2 or query.shape != positive.shape or query.shape[0] == 0:
        raise ValueError("LamRA requires equally sized, nonempty [pairs, hidden] query and positive tensors")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("LamRA temperature must be finite and positive")
    if process_group is not None:
        query = gather_with_local_grad(query, process_group)
        positive = gather_with_local_grad(positive, process_group)
    elif dist.is_initialized():
        raise ValueError("Distributed LamRA requires an explicit process_group")
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    scores = F.cosine_similarity(query.unsqueeze(1), positive.unsqueeze(0), dim=-1) / temperature
    labels = torch.arange(query.shape[0], device=scores.device)
    return F.cross_entropy(scores, labels)
