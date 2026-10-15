# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Lazy M-BEIR pairs carried by the existing SFT/TransferQueue pipeline."""

import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch

from relax.engine.sft.dataset.streaming import ProcessedSample
from relax.utils.data.streaming_dataset import IndexManager, StreamingReader


def format_text(text: str | None) -> str:
    text = (text or "").replace("\r", "").strip().strip('"')
    if text:
        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            text += "."
    return text


def retrieval_messages(text: str, image: str | None) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = []
    if image:
        content.append({"type": "image", "image": image})
    subject = "image and sentence" if image and text else "image" if image else "sentence"
    content.append({"type": "text", "text": f"{text}\nSummarize above {subject} in one word: "})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
    ]


_retrieval_processor: Any = None
_retrieval_processor_path: str | None = None


def resolve_media_path(relative: str | None, roots: list[Path]) -> str | None:
    if not relative:
        return None
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("M-BEIR media references must be relative to an explicit media root")
    for root in roots:
        path = root / relative
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"M-BEIR image unavailable in configured media roots: {relative}")


def process_pair_in_worker(pair: dict[str, Any], processor_path: str) -> dict[str, dict[str, torch.Tensor]]:
    """Use the SFT process pool, with an explicit slow processor contract.

    Only this recipe's processor is cached; the shared pool's processor and
    global preprocessing defaults are untouched. Its tokenizer is reused so the
    pretrained adapter's added <emb> token has exactly the same ID.
    """
    from relax.utils.data.processing_utils import load_processor
    from relax.utils.data.processor_pool import get_worker_processor
    from relax.utils.multimodal.config import MultimodalConfig
    from relax.utils.multimodal.image_utils import fetch_image, load_image

    global _retrieval_processor, _retrieval_processor_path
    if _retrieval_processor is None or _retrieval_processor_path != processor_path:
        _retrieval_processor = load_processor(processor_path, use_fast=False)
        if _retrieval_processor is None:
            raise ValueError("LamRA requires a Qwen2-VL processor")
        _retrieval_processor.tokenizer = get_worker_processor().tokenizer
        _retrieval_processor.tokenizer.padding_side = "right"
        _retrieval_processor_path = processor_path
    processor = _retrieval_processor
    output: dict[str, dict[str, torch.Tensor]] = {}
    config = MultimodalConfig(image_min_token_num=4, image_max_token_num=300)
    for side in ("query", "positive"):
        item = pair[side]
        messages = item.get("messages")
        if messages is None:
            tokens = processor.tokenizer(
                item["text"], truncation=True, max_length=480, padding=False, add_special_tokens=False
            )["input_ids"]
            text = processor.tokenizer.decode(tokens)
            messages = retrieval_messages(text, item["image"])
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        images = None
        if item["image"]:
            image = load_image(item["image"])
            if max(image.size) / min(image.size) > 200:
                raise ValueError("Native Qwen2-VL rejects image aspect ratios greater than 200")
            images = [fetch_image({"image": image}, image_patch_size=14, config=config)]
        encoded = processor(text=[rendered], images=images, padding=False, return_tensors="pt")
        if "messages" in item and encoded["input_ids"].shape[-1] > 1024:
            raise ValueError("Native NLI sequence exceeds 1024 tokens; refusing to truncate or filter it")
        output[side] = {"input_ids": encoded["input_ids"][0].contiguous()}
        for name in ("pixel_values", "image_grid_thw"):
            if name in encoded:
                value = encoded[name].contiguous()
                # Native visual forward casts these pixels to BF16 as well.
                if name == "pixel_values":
                    value = value.to(torch.bfloat16)
                output[side][name] = value.share_memory_()
    return output


class MBEIRDataset:
    """One queue row is one query and its lazily selected positive.

    Reuses Relax's readers, index manager and SFT processor pool. The final
    global batch repeats the *current* epoch's head, as Accelerate's native
    even-batch policy does. Access-position seeds make lazy choices resumable
    without depending on worker scheduling; they do not reproduce HF's RNG
    trajectory or Relax's permutation as HF's permutation.
    """

    def __init__(
        self,
        path: str,
        *,
        candidate_path: str,
        instructions_path: str,
        media_roots: list[str],
        processor_path: str,
        processor_pool: Any,
        seed: int = 42,
    ) -> None:
        self.path = path
        self.reader = StreamingReader(path)
        self.index_manager = IndexManager(len(self.reader), seed=seed)
        self.processor_pool = processor_pool
        self.processor_path = processor_path
        self.media_roots = [Path(root) for root in media_roots]
        self.seed = seed
        self.candidates: dict[str, dict[str, Any]] = {}
        with open(candidate_path, encoding="utf-8") as stream:
            for line in stream:
                candidate = json.loads(line)
                did = candidate["did"]
                if did in self.candidates:
                    raise ValueError(f"Duplicate candidate ID: {did}")
                self.candidates[did] = candidate
        self.instructions: dict[str, list[str]] = {}
        with open(instructions_path, encoding="utf-8") as stream:
            rows = csv.reader(stream, delimiter="\t")
            next(rows)
            for row in rows:
                key = f"{row[3]}, {row[0]}, {row[1]}"
                self.instructions[key] = [value for value in row[4:] if value]
        if not self.candidates or not self.media_roots or processor_pool is None:
            raise ValueError("LamRA requires candidates, media roots and the existing SFT processor pool")

    @classmethod
    def from_args(cls, config: Any, *, tokenizer: Any, processor_pool: Any, pad_token_ids: Any) -> "MBEIRDataset":
        del tokenizer, pad_token_ids
        return cls(
            config.prompt_data,
            candidate_path=config.lamra_candidate_path,
            instructions_path=config.lamra_instructions_path,
            media_roots=config.lamra_media_roots,
            processor_path=config.lamra_processor_path,
            processor_pool=processor_pool,
            seed=config.seed,
        )

    def __len__(self) -> int:
        return len(self.reader)

    def shuffle(self, epoch_id: int, position: int = 0) -> None:
        self.index_manager.shuffle(epoch_id)
        if not 0 <= position <= len(self):
            raise ValueError("Resume position lies outside the M-BEIR epoch")
        self.index_manager.position = position

    def _image_path(self, relative: str | None) -> str | None:
        return resolve_media_path(relative, self.media_roots)

    def _pair(self, index: int, access_position: int) -> dict[str, Any]:
        row = self.reader[index]
        rng = random.Random(f"{self.seed}:{self.index_manager.current_epoch}:{access_position}")
        positive = self.candidates[rng.choice(row["pos_cand_list"])]
        key = f"{row['qid'].split(':')[0]}, {row['query_modality']}, {positive['modality']}"
        prompts = self.instructions.get(key)
        if not prompts:
            raise ValueError(f"No M-BEIR instructions for {key}")
        prompt = format_text(rng.choice(prompts))
        if not prompt:
            raise ValueError(f"Empty M-BEIR instruction for {key}")
        return {
            "query": {
                "text": format_text(f"{prompt} {row.get('query_txt') or ''}"),
                "image": self._image_path(row.get("query_img_path")),
            },
            "positive": {
                "text": format_text(positive.get("txt")),
                "image": self._image_path(positive.get("img_path")),
            },
        }

    async def get_batch_async(self, n: int) -> tuple[list[ProcessedSample], bool]:
        if n <= 0:
            raise ValueError("Pair batch size must be positive")
        manager = self.index_manager
        if manager.indices is None:
            manager.shuffle(0)
        if manager.position == len(self):
            manager.shuffle(manager.current_epoch + 1)
        start = manager.position
        take = min(n, len(self) - start)
        indices = manager.indices[start : start + take]
        if take < n:
            indices += [manager.indices[index % len(self)] for index in range(n - take)]
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(
                self.processor_pool.executor,
                process_pair_in_worker,
                self._pair(index, start + offset),
                self.processor_path,
            )
            for offset, index in enumerate(indices)
        ]
        pairs = await asyncio.gather(*futures)
        manager.position += take
        samples = []
        for index, pair in zip(indices, pairs, strict=True):
            tokens = torch.cat([pair[side]["input_ids"] for side in ("query", "positive")])
            samples.append(ProcessedSample(tokens, torch.ones_like(tokens), len(tokens), pair, index))
        return samples, False

    def stop(self) -> None:
        """The SFT component owns and shuts down its shared process pool."""


class NLIDataset(MBEIRDataset):
    """Native sent0/sent1 messages using the same atomic pair transport.

    Prepared JSONL preserves pandas NA-to-string behavior and the original
    prompt without M-BEIR normalization/truncation. The native NLI dataset
    ignores hard_neg, so each row still has exactly two sequences.
    """

    def __init__(
        self, path: str, *, processor_path: str, processor_pool: Any, batch_size: int, seed: int = 42
    ) -> None:
        self.path = path
        self.reader = StreamingReader(path)
        self.index_manager = IndexManager(len(self.reader), seed=seed)
        self.processor_pool = processor_pool
        self.processor_path = processor_path
        self.batch_size = batch_size
        if not len(self) or batch_size <= 0 or processor_pool is None:
            raise ValueError("NLI requires a nonempty complete dataset, positive batch and SFT processor pool")

    @classmethod
    def from_args(cls, config: Any, *, tokenizer: Any, processor_pool: Any, pad_token_ids: Any) -> "NLIDataset":
        del tokenizer, pad_token_ids
        return cls(
            config.prompt_data,
            processor_path=config.lamra_processor_path,
            processor_pool=processor_pool,
            batch_size=config.global_batch_size,
            seed=config.seed,
        )

    def shuffle(self, epoch_id: int, position: int = 0) -> None:
        # SFT encodes resume as consumed // real_length, consumed % real_length.
        # Native even batches consume a padded epoch; undo that encoding here.
        consumed = epoch_id * len(self) + position
        padded = ((len(self) + self.batch_size - 1) // self.batch_size) * self.batch_size
        epoch, offset = divmod(consumed, padded)
        if offset % self.batch_size or offset >= len(self):
            raise ValueError("NLI resume must point to a complete global-batch boundary")
        self.index_manager.shuffle(epoch)
        self.index_manager.position = offset

    def _pair(self, index: int, access_position: int) -> dict[str, Any]:
        del access_position
        messages = self.reader[index]["messages"]
        if len(messages) != 4 or [m["role"] for m in messages] != ["user", "assistant"] * 2:
            raise ValueError("Native NLI requires exactly two user/assistant conversations")
        return {
            side: {"messages": messages[start : start + 2], "image": None}
            for side, start in (("query", 0), ("positive", 2))
        }
