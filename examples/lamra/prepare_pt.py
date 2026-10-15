#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prepare an independent Qwen base and the complete native NLI pair
dataset."""

import argparse
import json
from pathlib import Path


def prepare(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from relax.utils.replay.bundle import sha256_file

    model_dir, data_dir = Path(args.output_model), Path(args.output_data)
    if model_dir.exists() or data_dir.exists():
        raise FileExistsError("Preparation outputs must be new directories; reuse their passed audit when resuming")
    processor = AutoProcessor.from_pretrained(args.base_model, use_fast=False, local_files_only=True)
    tokenizer = processor.tokenizer
    if tokenizer.add_tokens(["<emb>"]) != 1:
        raise ValueError("Native PT must start from the original Qwen tokenizer without <emb>")
    emb = tokenizer.convert_tokens_to_ids("<emb>")
    model, loading = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cpu", output_loading_info=True
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Original Qwen base must load completely: {loading}")
    # Same operation as the native LamRA special-token patch. Shrinking a
    # padded vocabulary preserves its existing rows; do not invent a new row.
    model.resize_token_embeddings(len(tokenizer))
    model.config.emb_token_ids = [emb]
    model.save_pretrained(model_dir, safe_serialization=True)
    processor.save_pretrained(model_dir)
    del model

    frame = pd.read_csv(args.csv)
    if len(frame) != args.expected_rows:
        raise ValueError("NLI CSV differs from the requested complete dataset")
    data_dir.mkdir(parents=True)
    output = data_dir / "nli_cn.jsonl"
    texts: list[str] = []
    maximum = 0
    with output.open("w", encoding="utf-8") as stream:
        for index, (left, right) in enumerate(frame[["sent0", "sent1"]].itertuples(index=False, name=None)):
            messages = []
            for value in (left, right):
                pair = [
                    {"role": "user", "content": f"{value}\nSummarize above sentence in one word: "},
                    {"role": "assistant", "content": "<emb>."},
                ]
                messages.extend(pair)
                texts.append(processor.apply_chat_template(pair, tokenize=False, add_generation_prompt=False))
            stream.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            if len(texts) >= 1024 or index + 1 == len(frame):
                ids = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
                maximum = max(maximum, max(map(len, ids)))
                if maximum > 1024 or any(sequence.count(emb) != 1 for sequence in ids):
                    raise ValueError(
                        "Full native NLI audit found overlength input or ambiguous <emb>; no rows skipped"
                    )
                texts.clear()
    report = {
        "status": "passed",
        "dataset": "nli",
        "input_sha256": {"query": sha256_file(output)},
        "source_csv_sha256": sha256_file(Path(args.csv)),
        "query_count": len(frame),
        "candidate_count": len(frame),
        "validated_sequences": 2 * len(frame),
        "max_sequence_tokens": maximum,
        "hard_negative_policy": "ignored_native_pair_only",
        "na_policy": "pandas_read_csv_fstring_nan",
        "data_filtered_or_truncated": False,
        "emb_token_id": emb,
        "vocab_size": len(tokenizer),
        "base_source_sha256": {p.name: sha256_file(p) for p in sorted(Path(args.base_model).glob("*.safetensors"))},
    }
    (data_dir / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--expected-rows", type=int, default=275601)
    prepare(parser.parse_args())
