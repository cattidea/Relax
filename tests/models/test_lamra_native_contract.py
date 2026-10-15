# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Numeric contracts that must survive the LamRA backend migration."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relax.models.lamra.encoding import LegacyQwen2VLEncoder, collate_pairs
from relax.models.lamra.loss import native_infonce


def sequence(*tokens: int) -> dict:
    return {"input_ids": torch.tensor(tokens)}


class Language(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(30, 3)
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.seen_positions: list[torch.Tensor] = []

    def forward(self, *, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, **kwargs) -> SimpleNamespace:
        self.seen_positions.append(position_ids.clone())
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class Vision(torch.nn.Module):
    dtype = torch.float32

    def forward(self, pixels: torch.Tensor, *, grid_thw: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(pooler_output=pixels.reshape(len(grid_thw), 4, 3).mean(dim=1))


def test_right_padding_and_embedding_position_precede_sentinel() -> None:
    pairs = [{"query": sequence(4, 8, 9), "positive": sequence(6, 7, 8, 9)}]
    batch = collate_pairs(pairs, pad_token_id=0, emb_token_id=9, image_token_id=29)
    assert batch.input_ids.tolist() == [[4, 8, 9, 0], [6, 7, 8, 9]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    backbone = torch.nn.Module()
    backbone.language_model = Language()
    encoder = LegacyQwen2VLEncoder(backbone, image_token_id=29)
    actual = encoder(batch)
    expected = backbone.language_model.embed_tokens(torch.tensor([8, 8]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert backbone.language_model.seen_positions[0].tolist() == [[[0, 1, 2, 3]] * 2] * 3


def test_queries_precede_positives_and_forward_chunks_do_not_split_loss_batch() -> None:
    pairs = [{"query": sequence(2, 9), "positive": sequence(3, 4, 9)} for _ in range(34)]
    batch = collate_pairs(pairs, pad_token_id=0, emb_token_id=9, image_token_id=29)
    backbone = torch.nn.Module()
    backbone.language_model = Language()
    features = LegacyQwen2VLEncoder(backbone, image_token_id=29)(batch)
    assert [x.shape[1] for x in backbone.language_model.seen_positions] == [32, 32, 4]
    torch.testing.assert_close(features[:34], backbone.language_model.embed_tokens(torch.full((34,), 2)))
    torch.testing.assert_close(features[34:], backbone.language_model.embed_tokens(torch.full((34,), 4)))
    loss = native_infonce(*features.chunk(2))
    torch.testing.assert_close(loss, torch.tensor(34.0).log())


def test_image_routing_crosses_a_forward_chunk_boundary() -> None:
    image = sequence(1, 29, 9)
    image.update(pixel_values=torch.full((4, 3), 7.0), image_grid_thw=torch.tensor([[1, 2, 2]]))
    pairs = [{"query": sequence(2, 9), "positive": image} for _ in range(2)]
    batch = collate_pairs(pairs, pad_token_id=0, emb_token_id=9, image_token_id=29)
    assert batch.image_counts == [0, 0, 1, 1]
    assert batch.pixel_offsets == [0, 4, 8]
    backbone = torch.nn.Module()
    backbone.language_model = Language()
    backbone.visual = Vision()
    encoder = LegacyQwen2VLEncoder(backbone, image_token_id=29, subbatch_size=3)
    actual = encoder(batch)
    torch.testing.assert_close(actual[2:], torch.full((2, 3), 7.0))
    assert "Vision" not in encoder._no_split_modules


@pytest.mark.parametrize("tokens", [(2, 3), (9, 2)])
def test_bad_embedding_sentinel_fails_before_device_transfer(tokens: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="preceding token"):
        collate_pairs(
            [{"query": sequence(*tokens), "positive": sequence(2, 9)}],
            pad_token_id=0,
            emb_token_id=9,
            image_token_id=29,
        )


def test_detached_gather_preserves_native_averaged_parameter_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare against full-batch CE, mocking only the network collectives."""
    from relax.models.lamra import loss as module

    generator = torch.Generator().manual_seed(9)
    query_inputs = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    positive_inputs = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    initial = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    weight = initial.clone().requires_grad_()
    reference = native_infonce(query_inputs @ weight, positive_inputs @ weight)
    reference_grad = torch.autograd.grad(reference, weight)[0]
    gradients = []
    for rank in range(2):
        weight = initial.clone().requires_grad_()
        all_features = [query_inputs @ weight, positive_inputs @ weight]
        pending = iter(all_features)

        def gather(output: list[torch.Tensor], value: torch.Tensor, *, group: object) -> None:
            source = next(pending).detach()
            for target, shard in zip(output, source.chunk(2), strict=True):
                target.copy_(shard)

        monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)
        monkeypatch.setattr(module.dist, "get_rank", lambda group, rank=rank: rank)
        monkeypatch.setattr(module.dist, "all_gather", gather)
        local_query, local_positive = [feature.chunk(2)[rank] for feature in all_features]
        actual = native_infonce(local_query, local_positive, process_group=object())
        torch.testing.assert_close(actual, reference)
        gradients.append(torch.autograd.grad(actual, weight)[0])
    averaged = torch.stack(gradients).mean(dim=0)
    torch.testing.assert_close(averaged, reference_grad / 2, rtol=1e-12, atol=1e-12)


def test_native_loss_is_one_way_mean_without_dtype_conversion() -> None:
    query = torch.tensor([[1.0, 2.0], [2.0, -1.0], [1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    positive = torch.tensor([[1.5, 0.5], [-0.5, -1.0], [1.0, 2.0]], dtype=torch.float64, requires_grad=True)
    q, p = F.normalize(query, dim=-1), F.normalize(positive, dim=-1)
    expected = F.cross_entropy(F.cosine_similarity(q[:, None], p[None, :], dim=-1) / 0.05, torch.arange(3))
    actual = native_infonce(query, positive)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
