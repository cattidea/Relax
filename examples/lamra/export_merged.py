#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Merge a verified Relax PT adapter into a new HF checkpoint for FT."""

import argparse
import json
from pathlib import Path


def export(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    from relax.utils.replay.bundle import sha256_file

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Merged output must be a new directory")
    verification = json.loads((Path(args.adapter).parent / "training_verification.json").read_text())
    if verification.get("status") != "passed" or verification["completed_steps"] != args.expected_steps:
        raise ValueError("Only a complete, verified PT adapter may initialize FT")
    final = Path(args.adapter).parent / "lamra" / f"iter_{args.expected_steps - 1:07d}"
    contract = json.loads((final / "adapter_contract.json").read_text())
    hashes = {p.name: sha256_file(p) for p in sorted(Path(args.base).glob("*.safetensors"))}
    if not (final / "COMMITTED").exists() or contract.get("base_model_sha256") != hashes:
        raise ValueError("PT base does not match its committed training checkpoint")
    if contract.get("dataset", "mbeir") != getattr(args, "dataset", "nli"):
        raise ValueError("The merged export dataset differs from the committed training phase")
    base = Qwen2VLForConditionalGeneration.from_pretrained(args.base, dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    expected = {}
    for name, layer in model.named_modules():
        if hasattr(layer, "lora_A") and "default" in layer.lora_A:
            delta = layer.lora_B["default"].weight @ layer.lora_A["default"].weight
            expected[name.removeprefix("base_model.model.") + ".weight"] = (
                (layer.base_layer.weight + delta * layer.scaling["default"]).detach().half().clone()
            )
    if len(expected) != verification["nonzero_lora_b"]:
        raise ValueError("PT merge target coverage is incomplete")
    merged = model.merge_and_unload(safe_merge=True).half()
    state = merged.state_dict()
    if any(not torch.equal(state[name], value) for name, value in expected.items()):
        raise ValueError("PT merge differs from the explicit LoRA equation")
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="10GB", save_original_format=False)
    AutoProcessor.from_pretrained(args.base, use_fast=False).save_pretrained(output)
    serialized = load_file(str(output / "model.safetensors"))
    # HF omits one copy of tied weights; every serialized tensor must be exact.
    if any(not torch.equal(value, state[name]) for name, value in serialized.items()):
        raise ValueError("Serialized merged PT model did not reload exactly")
    reloaded, loading = Qwen2VLForConditionalGeneration.from_pretrained(
        output, dtype=torch.float16, device_map="cpu", output_loading_info=True
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Merged PT model did not load completely: {loading}")
    if any(not torch.equal(value, state[name]) for name, value in reloaded.state_dict().items()):
        raise ValueError("Merged PT model class reload differs from the verified merge")
    report = {
        "status": "passed",
        "completed_pt_steps": verification["completed_steps"],
        "merged_layers": len(expected),
        "serialized_tensors": len(serialized),
        "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
        "merged_sha256": sha256_file(output / "model.safetensors"),
        "merge_equation_exact": True,
        "serialized_reload_exact": True,
        "hf_model_reload_exact": True,
        "base_model_sha256": hashes,
    }
    (output / "merge_verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-steps", type=int, default=1078)
    parser.add_argument("--dataset", choices=("nli", "mbeir"), default="nli")
    export(parser.parse_args())
