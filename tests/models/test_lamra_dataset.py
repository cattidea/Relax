# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from relax.models.lamra import dataset as module


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text(
        "".join(
            json.dumps({"qid": f"0:{i}", "query_txt": str(i), "query_modality": "text", "pos_cand_list": ["0:0"]})
            + "\n"
            for i in range(10)
        )
    )
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps({"did": "0:0", "txt": "positive", "modality": "text"}) + "\n")
    instructions = tmp_path / "instructions.tsv"
    instructions.write_text("query\tcandidate\tname\tid\tprompt\ntext\ttext\tfixture\t0\tfind\n")

    def encode(pair, processor_path):
        assert processor_path == "unused"
        return {side: {"input_ids": torch.tensor([len(pair[side]["text"]), 99])} for side in ("query", "positive")}

    monkeypatch.setattr(module, "process_pair_in_worker", encode)
    with ThreadPoolExecutor(2) as executor:
        pool = type("Pool", (), {"executor": executor})()
        yield module.MBEIRDataset(
            str(query_path),
            candidate_path=str(candidates),
            instructions_path=str(instructions),
            media_roots=[str(tmp_path)],
            processor_path="unused",
            processor_pool=pool,
        )


def test_tail_repeats_current_epoch_head_without_dropping_pairs(dataset):
    dataset.shuffle(0)
    order = list(dataset.index_manager.indices)
    first, _ = asyncio.run(dataset.get_batch_async(8))
    tail, _ = asyncio.run(dataset.get_batch_async(8))
    assert [item.source_idx for item in first] == order[:8]
    assert [item.source_idx for item in tail] == order[8:] + order[:6]
    assert dataset.index_manager.current_epoch == 0
    assert dataset.index_manager.position == 10
    for sample in tail:
        assert set(sample.multimodal_train_inputs) == {"query", "positive"}
        assert sample.total_length == 4


def test_resume_uses_same_lazy_choices_and_pair_order(dataset):
    dataset.shuffle(0)
    asyncio.run(dataset.get_batch_async(8))
    expected, _ = asyncio.run(dataset.get_batch_async(8))
    dataset.shuffle(0, position=8)
    actual, _ = asyncio.run(dataset.get_batch_async(8))
    assert [item.source_idx for item in actual] == [item.source_idx for item in expected]
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left.tokens, right.tokens)
    assert dataset._pair(0, 0) == dataset._pair(0, 0)


def test_missing_media_raises_and_recovered_root_wins(dataset, tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset._image_path("missing.jpg")
    with pytest.raises(ValueError, match="relative"):
        dataset._image_path("../outside.jpg")
    recovered = tmp_path / "recovered"
    recovered.mkdir()
    (recovered / "present.jpg").touch()
    (tmp_path / "present.jpg").touch()
    dataset.media_roots.insert(0, recovered)
    assert dataset._image_path("present.jpg") == str(recovered / "present.jpg")


def test_native_text_normalization_and_message_contract():
    assert module.format_text(' "hello\r world" ') == "Hello world."
    assert module.format_text("ready?") == "Ready?"
    message = module.retrieval_messages("", "image.jpg")
    assert message[0]["content"][1]["text"] == "\nSummarize above image in one word: "
    assert message[1]["content"][0]["text"] == "<emb>."


def test_existing_tensordict_transport_preserves_complete_pairs(dataset):
    from relax.engine.sft.dataset.streaming import pack_samples_for_tq
    from relax.utils.utils import dict_to_tensordict

    samples, _ = asyncio.run(dataset.get_batch_async(8))
    payload = pack_samples_for_tq(samples, force_multimodal_field=True)
    transported = dict_to_tensordict(payload, batch_size=len(samples))
    for row, sample in zip(transported["multimodal_train_inputs"], samples, strict=True):
        pair = getattr(row, "data", row)
        assert set(pair) == {"query", "positive"}
        for side in pair:
            assert torch.equal(pair[side]["input_ids"], sample.multimodal_train_inputs[side]["input_ids"])


def test_nli_complete_epochs_and_resume_include_tail_padding_without_skips(tmp_path, monkeypatch):
    path = tmp_path / "nli.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": role, "content": f"raw {index}" if role == "user" else "<emb>."}
                        for role in ("user", "assistant", "user", "assistant")
                    ]
                }
            )
            + "\n"
            for index in range(10)
        )
    )

    def encode(pair, processor_path):
        assert "raw " in pair["query"]["messages"][0]["content"]
        assert all(pair[side]["image"] is None for side in pair)
        return {side: {"input_ids": torch.tensor([1, 99])} for side in pair}

    monkeypatch.setattr(module, "process_pair_in_worker", encode)
    with ThreadPoolExecutor(2) as executor:
        data = module.NLIDataset(
            str(path), processor_path="unused", processor_pool=type("Pool", (), {"executor": executor})(), batch_size=8
        )
        data.shuffle(0)
        epoch0 = list(data.index_manager.indices)
        batches = [asyncio.run(data.get_batch_async(8))[0] for _ in range(4)]
        epoch1 = list(data.index_manager.indices)
        assert [p.source_idx for p in batches[0]] == epoch0[:8]
        assert [p.source_idx for p in batches[1]] == epoch0[8:] + epoch0[:6]
        assert [p.source_idx for p in batches[2]] == epoch1[:8]
        assert [p.source_idx for p in batches[3]] == epoch1[8:] + epoch1[:6]
        # Producer encodes resumed step 2 as consumed=16 -> epoch=1,pos=6.
        data.shuffle(1, position=6)
        resumed = asyncio.run(data.get_batch_async(8))[0]
        assert [p.source_idx for p in resumed] == epoch1[:8]
        data.shuffle(2, position=4)  # step3=24 -> epoch=2,pos=4
        resumed_tail = asyncio.run(data.get_batch_async(8))[0]
        assert [p.source_idx for p in resumed_tail] == epoch1[8:] + epoch1[:6]


def test_nli_recipe_keeps_local_coordinator_for_padded_epoch_resume(monkeypatch):
    from types import SimpleNamespace

    from relax.components import sft

    # Custom datasets must not enter the remote producer helper, which directly
    # rewrites IndexManager using unpadded offsets instead of dataset.shuffle.
    monkeypatch.setattr(sft, "sft_tq_num_shards", lambda config: 2)
    monkeypatch.setattr(sft.ray, "is_initialized", lambda: True)
    cls = getattr(sft.SFT, "func_or_class", sft.SFT)
    service = SimpleNamespace(_logger=SimpleNamespace(info=lambda message: None))
    service.config = SimpleNamespace(
        custom_dataset_class_path="relax.models.lamra.dataset.NLIDataset",
        global_batch_size=512,
        sft_async_prepack=True,
    )
    assert not cls._should_use_remote_batch_producer(service)
    service.config.custom_dataset_class_path = None
    assert cls._should_use_remote_batch_producer(service)
