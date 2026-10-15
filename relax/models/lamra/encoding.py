# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Right-padded retrieval batches and legacy Qwen2-VL features."""

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RetrievalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    feature_positions: list[int]
    image_counts: list[int]
    pixel_offsets: list[int]
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None

    def to(self, device: torch.device) -> "RetrievalBatch":
        """Move tensors while retaining host routing metadata."""
        return replace(
            self,
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            pixel_values=None if self.pixel_values is None else self.pixel_values.to(device, non_blocking=True),
            image_grid_thw=None if self.image_grid_thw is None else self.image_grid_thw.to(device, non_blocking=True),
        )


def collate_pairs(
    pairs: list[dict[str, dict[str, Any]]],
    *,
    pad_token_id: int,
    emb_token_id: int,
    image_token_id: int,
) -> RetrievalBatch:
    """Keep pairs atomic in transport, then order all queries before positives.

    Per-sequence inputs are unpadded CPU tensors from the original processor.
    Routing and sentinel validation happen here, before device transfer.
    """
    if not pairs:
        raise ValueError("Cannot collate an empty retrieval batch")
    sequences = [pair[side] for side in ("query", "positive") for pair in pairs]
    ids = [sequence["input_ids"].reshape(-1) for sequence in sequences]
    if any(tokens.device.type != "cpu" for tokens in ids):
        raise ValueError("Retrieval collation and routing metadata must be computed on CPU")
    length = max(tokens.numel() for tokens in ids)
    positions: list[int] = []
    counts: list[int] = []
    pixels: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    offsets = [0]
    for sequence, tokens in zip(sequences, ids, strict=True):
        matches = torch.where(tokens == emb_token_id)[0].tolist()
        if not matches or matches[0] == 0:
            raise ValueError("Each retrieval sequence must have an <emb> token with a preceding token")
        positions.append(matches[0] - 1)
        has_image = bool((tokens == image_token_id).any())
        counts.append(int(has_image))
        values, grid = sequence.get("pixel_values"), sequence.get("image_grid_thw")
        if has_image:
            if values is None or grid is None or grid.shape != (1, 3):
                raise ValueError("Native M-BEIR requires exactly one image grid per image-bearing sequence")
            if values.device.type != "cpu" or grid.device.type != "cpu":
                raise ValueError("Image routing metadata must be computed before device transfer")
            patch_count = int(grid.prod())
            if len(values) != patch_count:
                raise ValueError("Pixel patch count disagrees with image_grid_thw")
            pixels.append(values)
            grids.append(grid)
            offsets.append(offsets[-1] + patch_count)
        elif values is not None or grid is not None:
            raise ValueError("Text-only retrieval sequence unexpectedly carries image tensors")
    return RetrievalBatch(
        input_ids=torch.stack([F.pad(tokens, (0, length - tokens.numel()), value=pad_token_id) for tokens in ids]),
        attention_mask=torch.stack([F.pad(torch.ones_like(tokens), (0, length - tokens.numel())) for tokens in ids]),
        feature_positions=positions,
        image_counts=counts,
        pixel_offsets=offsets,
        pixel_values=torch.cat(pixels) if pixels else None,
        image_grid_thw=torch.cat(grids) if grids else None,
    )


class LegacyQwen2VLEncoder(torch.nn.Module):
    """Extract the hidden state before <emb>, using sequential positions.

    ``backbone`` is the HF Qwen2-VL model without the unused language head.
    Only language decoder blocks are nested FSDP units. The root owns frozen
    vision parameters, so ranks with text-only chunks never enter conditional
    vision-block collectives while another rank does.
    """

    def __init__(self, backbone: torch.nn.Module, *, image_token_id: int, subbatch_size: int = 32) -> None:
        super().__init__()
        if subbatch_size <= 0:
            raise ValueError("Forward subbatch size must be positive")
        self.backbone = backbone
        self.image_token_id = image_token_id
        self.subbatch_size = subbatch_size
        self._no_split_modules = sorted({type(layer).__name__ for layer in backbone.language_model.layers})

    def forward(self, batch: RetrievalBatch) -> torch.Tensor:
        ids = batch.input_ids
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, len(ids), -1)
        features: list[torch.Tensor] = []
        image_cursor = 0
        for start in range(0, len(ids), self.subbatch_size):
            end = min(start + self.subbatch_size, len(ids))
            chunk_ids = ids[start:end]
            embeds = self.backbone.language_model.embed_tokens(chunk_ids)
            image_count = sum(batch.image_counts[start:end])
            if image_count:
                if batch.pixel_values is None or batch.image_grid_thw is None:
                    raise ValueError("Image tensors are missing")
                next_cursor = image_cursor + image_count
                pixels = batch.pixel_values[batch.pixel_offsets[image_cursor] : batch.pixel_offsets[next_cursor]]
                grid = batch.image_grid_thw[image_cursor:next_cursor]
                visual = self.backbone.visual
                image_embeds = visual(pixels.to(visual.dtype), grid_thw=grid)
                if hasattr(image_embeds, "pooler_output"):
                    image_embeds = image_embeds.pooler_output
                embeds = embeds.clone() if self.training else embeds
                embeds[chunk_ids == self.image_token_id] = image_embeds.to(embeds.device)
                image_cursor = next_cursor
            outputs = self.backbone.language_model(
                input_ids=None,
                inputs_embeds=embeds,
                position_ids=positions[:, start:end],
                attention_mask=batch.attention_mask[start:end],
                use_cache=False,
                return_dict=True,
            )
            rows = torch.arange(end - start, device=ids.device)
            columns = torch.tensor(batch.feature_positions[start:end], device=ids.device)
            features.append(outputs.last_hidden_state[rows, columns])
        expected = 0 if batch.image_grid_thw is None else len(batch.image_grid_thw)
        if image_cursor != expected:
            raise ValueError("Image tensors do not match the sequence routing metadata")
        return torch.cat(features)
