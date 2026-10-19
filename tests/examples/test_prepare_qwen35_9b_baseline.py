# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU fixtures verify preparation failures; they do not emulate 9B
training."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "examples/on_policy_distillation/mopd/prepare_qwen35_9b_baseline.py"
SPEC = importlib.util.spec_from_file_location("prepare_qwen35_9b_baseline", SCRIPT)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def fixture(path: Path, shape: list[int] | None = None) -> None:
    path.mkdir()
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "vocab_size": 248320,
        },
        "vision_config": {"hidden_size": 1152},
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text('{"fixture": true}')
    shape = shape or [2]
    count = shape[0] * 2
    header = json.dumps({"fixture.weight": {"dtype": "BF16", "shape": shape, "data_offsets": [0, count]}}).encode()
    header += b" " * (-len(header) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * count)


class PrepareBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.student = self.root / "student"
        fixture(self.student)
        self.teachers = {}
        for teacher in baseline.TEACHER_IDS:
            directory = self.root / teacher
            fixture(directory)
            self.teachers[teacher] = str(directory)

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, source="disk"):
        return Namespace(
            student=self.student,
            teacher=[f"{name}={path}" for name, path in self.teachers.items()],
            output_dir=self.root / source,
            weight_source=source,
            prompt_data=None,
            rows_per_teacher=4,
        )

    def test_prepares_twenty_balanced_prompts_and_both_full_reload_manifests(self):
        for source in ("disk", "memory"):
            args = self.args(source)
            report = baseline.prepare(args)
            manifest = json.loads((args.output_dir / "teacher_manifest.json").read_text())
            self.assertEqual(manifest["backend"], f"sglang-{source}-v1")
            self.assertEqual(manifest["tp_size"], 1)
            self.assertEqual(set(manifest["teachers"]), set(baseline.TEACHER_IDS))
            self.assertEqual(report["source_counts"], dict.fromkeys(baseline.TEACHER_IDS, 4))
            self.assertEqual(len((args.output_dir / "train.jsonl").read_text().splitlines()), 20)
            self.assertFalse(report["weight_payload_checksums_verified"])

    def test_rejects_small_model_before_emitting_a_manifest(self):
        path = self.student / "config.json"
        config = json.loads(path.read_text())
        config["text_config"]["hidden_size"] = 896
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "not the requested Qwen3.5-9B"):
            baseline.prepare(self.args())
        self.assertFalse(self.args().output_dir.exists())

    def test_rejects_teacher_with_different_token_ids(self):
        (Path(self.teachers["T5"]) / "tokenizer.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "Tokenizer assets differ for T5"):
            baseline.prepare(self.args())

    def test_rejects_image_preprocessing_difference_before_shared_engine_launch(self):
        for directory in [self.student, *(Path(path) for path in self.teachers.values())]:
            (directory / "preprocessor_config.json").write_text('{"do_normalize":true}')
        (Path(self.teachers["T5"]) / "preprocessor_config.json").write_text('{"do_normalize":false}')
        with self.assertRaisesRegex(ValueError, "processor assets differ for T5"):
            baseline.prepare(self.args())

    def test_rejects_tensor_shape_mismatch(self):
        path = Path(self.teachers["T5"])
        (path / "model.safetensors").unlink()
        header = json.dumps({"fixture.weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}).encode()
        (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 6)
        with self.assertRaisesRegex(ValueError, "Tensor names/shapes/dtypes differ for T5"):
            baseline.prepare(self.args())

    def test_rejects_truncated_payload_and_bad_index(self):
        path = Path(self.teachers["T5"])
        weight = path / "model.safetensors"
        original = weight.read_bytes()
        weight.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, "truncated"):
            baseline.checkpoint_layout(path)
        weight.write_bytes(original)
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 4}, "weight_map": {"wrong.name": "model.safetensors"}})
        )
        with self.assertRaisesRegex(ValueError, "Index disagrees"):
            baseline.checkpoint_layout(path)

    def test_rejects_duplicate_checkpoints_and_missing_teachers(self):
        self.teachers["T5"] = self.teachers["T4"]
        with self.assertRaisesRegex(ValueError, "distinct checkpoint directories"):
            baseline.prepare(self.args())
        del self.teachers["T5"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            baseline.prepare(self.args())

    def test_rejects_top_level_route_without_extra_info_or_missing_sources(self):
        rows = baseline.make_rows(1)
        rows[0].pop("extra_info")
        with self.assertRaisesRegex(ValueError, "extra_info.data_source"):
            baseline.validate_rows(rows)
        with self.assertRaisesRegex(ValueError, "every Teacher"):
            baseline.validate_rows(baseline.make_rows(1)[:-1])

    def test_preserves_existing_output_and_original_configs(self):
        args = self.args()
        configs = {name: (Path(path) / "config.json").read_bytes() for name, path in self.teachers.items()}
        baseline.prepare(args)
        with self.assertRaisesRegex(ValueError, "must be fresh"):
            baseline.prepare(args)
        for name, path in self.teachers.items():
            self.assertEqual((Path(path) / "config.json").read_bytes(), configs[name])

    def test_pin_memory_is_explicit_and_cannot_be_mislabeled_as_disk(self):
        args = self.args("memory")
        args.pin_memory = True
        report = baseline.prepare(args)
        manifest = json.loads((args.output_dir / "teacher_manifest.json").read_text())
        self.assertTrue(manifest["pin_memory"])
        self.assertTrue(report["teacher_cache_pin_memory"])
        args = self.args("disk")
        args.pin_memory = True
        with self.assertRaisesRegex(ValueError, "only to --weight-source memory"):
            baseline.prepare(args)


if __name__ == "__main__":
    unittest.main()
