# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU tests for real-data preservation, split isolation, and Teacher
routing."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_helper(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples/on_policy_distillation/mopd" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load_helper("prepare_qwen35_9b_dapo_data")
baseline = load_helper("prepare_qwen35_9b_baseline")


def fixture_rows(count: int = 105) -> list[dict]:
    return [
        {
            "prompt": [
                {"role": "user", "content": f"Question {index}: 请计算 {index}+2.\nKeep original whitespace. "}
            ],
            "label": str(index + 2),
            "source": "original-source-field",
            "data_source": "original-domain",
            "extra_info": {"original_field": index},
        }
        for index in range(count)
    ]


class DapoDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.jsonl"
        self.rows = fixture_rows()
        self.write_source(self.rows)

    def tearDown(self):
        self.temporary.cleanup()

    def write_source(self, rows):
        self.source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def args(self, name="prepared", seed=20261015):
        return Namespace(
            input=self.source,
            output_dir=self.root / name,
            seed=seed,
            eval_ratio=0.05,
            tokenizer=None,
            max_context_len=16384,
            max_response_len=1024,
        )

    def outputs(self, args):
        return {name: baseline.read_rows(args.output_dir / f"{name}.jsonl") for name in ("train", "eval")}

    def test_preserves_all_original_prompts_labels_and_provenance_without_source_mutation(self):
        original_bytes = self.source.read_bytes()
        args = self.args()
        report = data.prepare(args)
        routed = self.outputs(args)
        all_rows = routed["train"] + routed["eval"]
        self.assertEqual(len(all_rows), len(self.rows))
        self.assertEqual(self.source.read_bytes(), original_bytes)
        for row in all_rows:
            provenance = row["extra_info"]["dapo_provenance"]
            original = self.rows[provenance["source_line_1based"] - 1]
            self.assertEqual(row["prompt"], original["prompt"])
            self.assertEqual(row["label"], original["label"])
            self.assertEqual(row["source"], original["source"])
            self.assertEqual(row["extra_info"]["original_field"], original["extra_info"]["original_field"])
            self.assertEqual(provenance["original_data_source"], original["data_source"])
            self.assertEqual(row["extra_info"]["reward_source"], "dapo-math-17k")
            self.assertEqual(row["data_source"], row["extra_info"]["data_source"])
        self.assertEqual(report["split"]["dropped_rows"], 0)
        self.assertEqual(report["split"]["replicated_rows"], 0)
        self.assertFalse(report["multimodal_training_verified"])

    def test_balanced_batches_include_epoch_boundary_and_pass_actual_baseline_validator(self):
        args = self.args()
        data.prepare(args)
        routed = self.outputs(args)
        for rows in routed.values():
            self.assertEqual(set(baseline.validate_rows(rows)), set(data.TEACHERS))
        train = routed["train"]
        repeated = train * 2
        for start in range(0, len(repeated), 20):
            counts = Counter(row["data_source"] for row in repeated[start : start + 20])
            self.assertEqual(counts, Counter(dict.fromkeys(data.TEACHERS, 4)))
        self.assertFalse(
            json.loads((args.output_dir / "validation.json").read_text())["runtime_requirements"]["rollout_shuffle"]
        )

    def test_deterministic_bytes_and_seed_changes_split(self):
        a, b, c = self.args("a"), self.args("b"), self.args("c", seed=99)
        for args in (a, b, c):
            data.prepare(args)
        for filename in ("train.jsonl", "eval.jsonl", "validation.json"):
            self.assertEqual((a.output_dir / filename).read_bytes(), (b.output_dir / filename).read_bytes())
        self.assertNotEqual((a.output_dir / "eval.jsonl").read_bytes(), (c.output_dir / "eval.jsonl").read_bytes())

    def test_duplicate_prompts_stay_together_without_dropping_rows(self):
        self.rows += [copy.deepcopy(self.rows[3]), copy.deepcopy(self.rows[9])]
        self.write_source(self.rows)
        args = self.args()
        report = data.prepare(args)
        routed = self.outputs(args)
        hashes = {name: {data.prompt_hash(row["prompt"]) for row in rows} for name, rows in routed.items()}
        self.assertFalse(hashes["train"] & hashes["eval"])
        self.assertEqual(sum(len(rows) for rows in routed.values()), len(self.rows))
        self.assertEqual(report["split"]["extra_duplicate_rows"], 2)
        self.assertEqual(report["split"]["train_eval_prompt_overlap"], 0)

    def test_rejects_conflicting_labels_instead_of_leaking_or_dropping(self):
        duplicate = copy.deepcopy(self.rows[0])
        duplicate["label"] = "conflict"
        self.write_source(self.rows + [duplicate])
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            data.prepare(self.args())
        self.assertFalse(self.args().output_dir.exists())

    def test_rejects_nontext_or_malformed_or_already_routed_inputs(self):
        for replacement in (
            {"prompt": "flattened"},
            {"prompt": [{"role": "user", "content": [{"type": "image", "image": "x"}]}]},
            {"images": ["x"]},
            {"label": 0},
            {"extra_info": {"data_source": "T1"}},
        ):
            rows = fixture_rows()
            rows[0].update(replacement)
            self.write_source(rows)
            with self.assertRaises(ValueError):
                data.prepare(self.args())
        self.assertFalse(self.args().output_dir.exists())

    def test_rejects_existing_outputs_and_impossible_small_split(self):
        args = self.args()
        data.prepare(args)
        before = (args.output_dir / "train.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "must be fresh"):
            data.prepare(args)
        self.assertEqual(before, (args.output_dir / "train.jsonl").read_bytes())
        self.write_source(fixture_rows(24))
        with self.assertRaisesRegex(ValueError, "Cannot split"):
            data.prepare(self.args("too_small"))
        for ratio in (0, 1, float("nan")):
            with self.assertRaisesRegex(ValueError, "eval_ratio"):
                data.split_indices(self.rows, seed=42, eval_ratio=ratio)

    def test_known_dataset_size_keeps_all_rows_and_balances_training(self):
        rows = fixture_rows(17398)
        train, evaluation, report = data.split_indices(rows, seed=20261015, eval_ratio=0.05)
        self.assertEqual((len(train), len(evaluation)), (16520, 878))
        self.assertEqual(report["unique_prompts"], 17398)
        self.assertEqual(set(train) | set(evaluation), set(range(17398)))

    def test_length_preflight_matches_runtime_render_and_encode_and_reports_overflow(self):
        class Tokenizer:
            def apply_chat_template(inner_self, prompt, **kwargs):
                self.assertEqual(
                    kwargs, dict(tools=None, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                )
                return prompt[0]["content"]

            def encode(inner_self, text, *, add_special_tokens):
                self.assertFalse(add_special_tokens)
                return list(range(len(text)))

        splits = {"train": [{"prompt": [{"role": "user", "content": "x" * length}]} for length in (10, 20, 31)]}
        report = data.token_length_report(splits, Tokenizer(), max_context_len=40, max_response_len=10)
        self.assertFalse(report["fits_context_budget"])
        self.assertEqual(
            report["splits"]["train"]["over_budget_rows"], [{"output_row_0based": 2, "prompt_tokens": 31}]
        )
        self.assertEqual(report["splits"]["train"]["first_60_max"], 31)
        with self.assertRaisesRegex(ValueError, "positive"):
            data.token_length_report(splits, Tokenizer(), max_context_len=10, max_response_len=10)


if __name__ == "__main__":
    unittest.main()
