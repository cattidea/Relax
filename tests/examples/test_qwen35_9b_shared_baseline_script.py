# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Inspect the Qwen3.5-9B launcher without Ray, model weights, or GPU
imports."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/on_policy_distillation/mopd/scripts/run-mopd-qwen35-9b-5teacher-baseline.sh"
ADP_SCRIPT = SCRIPT.with_name("run-mopd-qwen35-9b-5teacher-adp.sh")


class Qwen35SharedBaselineScriptTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.manifest_path = self.base / "prepared with spaces" / "teacher_manifest.json"
        self.manifest_path.parent.mkdir()
        self.manifest = {
            "version": 1,
            "backend": "sglang-disk-v1",
            "tp_size": 1,
            "initial_teacher": "T3",
            "teachers": {f"T{i}": f"checkpoints/T{i}" for i in range(1, 6)},
            "event_log": "original_events.jsonl",
            "provenance": {"preserve": "original validation metadata"},
        }
        self._write_manifest()
        self.marker = self.base / "unexpected_ray_call"
        ray = self.base / "ray"
        ray.write_text(f"#!/bin/sh\ntouch '{self.marker}'\nexit 91\n")
        ray.chmod(0o755)
        self.env = {
            **{key: os.environ[key] for key in ("HOME", "TMPDIR") if key in os.environ},
            "PATH": str(self.base) + os.pathsep + os.environ["PATH"],
            "MODEL_DIR": str(self.base / "models with spaces"),
            "DATA_DIR": str(self.base / "data with spaces"),
            "SAVE_DIR": str(self.base / "persistent checkpoints"),
            "SHARED_TEACHER_MANIFEST": str(self.manifest_path),
        }

    def _write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def _run(self, overrides=None, *, expect_success=True, extra_args=("--dry-run",), script=SCRIPT):
        env = dict(self.env)
        env.update(overrides or {})
        result = subprocess.run(
            ["bash", str(script), *extra_args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertFalse(self.marker.exists(), "Dry-run invoked Ray or entrypoint cluster initialization")
        self.assertFalse(Path(env["SAVE_DIR"]).exists(), "Dry-run created launch/checkpoint artifacts")
        self.assertEqual(json.loads(self.manifest_path.read_text()), self.manifest, "Source manifest was modified")
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    @staticmethod
    def _flag(output, name):
        argv = output["train_argv"]
        return argv[argv.index(name) + 1]

    def test_shell_syntax(self):
        for script in (SCRIPT, ADP_SCRIPT):
            subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)

    def test_dry_run_needs_neither_ray_nor_model_weights(self):
        output = self._run()
        self.assertTrue(output["dry_run"])
        self.assertEqual(output["train_argv"][:3], ["python3", "-m", "relax.entrypoints.train"])
        self.assertEqual(output["ray_argv"][:3], ["ray", "job", "submit"])
        self.assertNotIn("--no-wait", output["ray_argv"])
        self.assertEqual(self._flag(output, "--hf-checkpoint"), str(Path(self.env["MODEL_DIR"]) / "Qwen3.5-9B"))
        self.assertEqual(
            self._flag(output, "--teacher-hf-checkpoint"), str(self.manifest_path.parent / "checkpoints/T3")
        )
        self.assertNotIn("--use-clearml", output["train_argv"])
        self.assertNotIn("--use-metrics-service", output["train_argv"])

    def test_adp_entry_reuses_the_same_training_recipe(self):
        baseline = self._run({"EXP_NAME": "comparison"})
        self.manifest.update(backend="sglang-adp-v1", package_dir="packages")
        self._write_manifest()
        adp = self._run({"EXP_NAME": "comparison", "WEIGHT_SOURCE": "disk"}, script=ADP_SCRIPT)
        self.assertEqual(adp["weight_source"], "adp")
        self.assertEqual(adp["runtime_manifest"]["backend"], "sglang-adp-v1")
        self.assertEqual(adp["runtime_manifest"]["package_dir"], str(self.manifest_path.parent / "packages"))
        first, second = baseline["train_argv"][:], adp["train_argv"][:]
        for argv in (first, second):
            argv[argv.index("--custom-config-path") + 1] = "<launch-config>"
            argv[argv.index("--tb-experiment-name") + 1] = "<launch-label>"
        self.assertEqual(first, second)
        self.assertIn("--teacher-sglang-disable-overlap-schedule", second)
        self.assertIn("--teacher-sglang-disable-cuda-graph", second)

    def test_baseline_rejects_accidental_adp_environment(self):
        self.assertIn("use the ADP script", self._run({"WEIGHT_SOURCE": "adp"}, expect_success=False))

    def test_adp_entry_rejects_increased_teacher_concurrency(self):
        self.manifest.update(backend="sglang-adp-v1", package_dir="packages")
        self._write_manifest()
        error = self._run({"TEACHER_MAX_RUNNING_REQUESTS": "2"}, script=ADP_SCRIPT, expect_success=False)
        self.assertIn("requires TEACHER_MAX_RUNNING_REQUESTS=1", error)

    def test_fixed_9b_architecture_and_single_teacher_resources(self):
        output = self._run({"MODEL_CONFIG_NAME": "qwen3-1.7B", "MODEL_CONFIG_DIR": "/does-not-exist"})
        self.assertEqual(
            json.loads(self._flag(output, "--resource")), {"actor": [1, 2], "rollout": [1, 1], "teacher": [1, 1]}
        )
        for flag, expected in {
            "--num-layers": "32",
            "--hidden-size": "4096",
            "--vocab-size": "248320",
            "--tensor-model-parallel-size": "2",
            "--pipeline-model-parallel-size": "1",
            "--context-parallel-size": "1",
            "--rollout-num-gpus": "1",
            "--rollout-num-gpus-per-engine": "1",
            "--teacher-num-gpus-per-engine": "1",
            "--teacher-sglang-max-running-requests": "1",
            "--opd-teacher-key": "data_source",
            "--metadata-key": "extra_info",
        }.items():
            self.assertEqual(self._flag(output, flag), expected)
        for flag in ("--colocate", "--offload", "--bf16", "--sequence-parallel", "--opd-only-reward"):
            self.assertIn(flag, output["train_argv"])
        self.assertNotIn("--opd-teacher-routes", output["train_argv"])
        self.assertIn("--attention-output-gate", output["train_argv"])
        self.assertNotIn("--use-gated-attention", output["train_argv"])
        self.assertNotIn("--finetune", output["train_argv"])
        self.assertNotIn("--no-load-optim", output["train_argv"])
        self.assertNotIn("--no-save-optim", output["train_argv"])
        self.assertIn("--use-checkpoint-opt-param-scheduler", output["train_argv"])
        self.assertIn("--teacher-sglang-disable-overlap-schedule", output["train_argv"])

    def test_eight_gpu_profile_keeps_the_previous_training_topology(self):
        output = self._run({"NUM_GPUS": "8"})
        self.assertEqual(
            json.loads(self._flag(output, "--resource")), {"actor": [1, 8], "rollout": [1, 7], "teacher": [1, 1]}
        )
        self.assertEqual(self._flag(output, "--tensor-model-parallel-size"), "4")
        self.assertEqual(self._flag(output, "--rollout-num-gpus"), "7")
        self.assertEqual(self._flag(output, "--teacher-num-gpus-per-engine"), "1")
        self.assertTrue(self._flag(output, "--save").endswith("disk-8xgpu-colocate"))
        for name in ("--global-batch-size", "--rollout-batch-size", "--rollout-max-response-len"):
            self.assertEqual(self._flag(output, name), self._flag(self._run(), name))

    def test_dp1_accepts_an_odd_batch_but_dp2_requires_even_partitioning(self):
        overrides = {"ROLLOUT_BATCH_SIZE": "5", "N_SAMPLES_PER_PROMPT": "3", "GLOBAL_BATCH_SIZE": "15"}
        self.assertEqual(self._flag(self._run(overrides), "--global-batch-size"), "15")
        self.assertIn("divisible by actor DP2", self._run({**overrides, "NUM_GPUS": "8"}, expect_success=False))

    def test_adp_honors_the_eight_gpu_profile(self):
        self.manifest.update(backend="sglang-adp-v1", package_dir="packages")
        self._write_manifest()
        output = self._run({"NUM_GPUS": "8"}, script=ADP_SCRIPT)
        self.assertEqual(json.loads(self._flag(output, "--resource"))["actor"], [1, 8])
        self.assertEqual(self._flag(output, "--tensor-model-parallel-size"), "4")
        self.assertTrue(self._flag(output, "--save").endswith("adp-8xgpu-colocate"))

    def test_reject_unsupported_gpu_counts_before_ray(self):
        for count in ("1", "3", "4", "0", "02", "two"):
            with self.subTest(count=count):
                self.assertIn("NUM_GPUS must be 2 or 8", self._run({"NUM_GPUS": count}, expect_success=False))

    def test_each_run_has_isolated_event_log_but_stable_checkpoint(self):
        first, second = self._run(), self._run()
        expected = str(Path(self.env["SAVE_DIR"]) / "qwen35-9b-5teacher-disk-2xgpu-colocate")
        self.assertEqual(self._flag(first, "--load"), expected)
        self.assertEqual(self._flag(first, "--save"), expected)
        self.assertEqual(self._flag(second, "--save"), expected)
        self.assertEqual(self._flag(first, "--save-interval"), "5")
        self.assertEqual(self._flag(first, "--max-actor-ckpt-to-keep"), "2")
        self.assertNotEqual(first["runtime_manifest_path"], second["runtime_manifest_path"])
        self.assertNotEqual(first["runtime_manifest"]["event_log"], second["runtime_manifest"]["event_log"])
        for output in (first, second):
            self.assertTrue(output["runtime_manifest_path"].startswith(expected + "/runs/"))
            self.assertEqual(Path(output["runtime_manifest"]["event_log"]).name, "full_reload_events.jsonl")
            self.assertEqual(output["runtime_manifest"]["provenance"], self.manifest["provenance"])
            self.assertEqual(output["custom_config"]["opd_shared_teacher_manifest"], output["runtime_manifest_path"])
            self.assertEqual(self._flag(output, "--custom-config-path"), output["custom_config_path"])
            self.assertTrue(output["custom_config"]["opd_require_all_teacher_scores"])

    def test_runtime_env_preserves_entrypoint_settings_and_propagates_new_values(self):
        output = self._run(
            {
                "RUNTIME_ENV_JSON": json.dumps(
                    {
                        "worker_process_setup_hook": "existing.hook",
                        "env_vars": {"EXISTING": "keep"},
                        "pip": ["fixture"],
                    }
                ),
                "RELAX_PROPAGATE_ENV_VARS": "EXTRA_SETTING",
                "EXTRA_SETTING": "preserve-me",
                "SGLANG_LOGITS_PROCESSER_CHUNK_SIZE": "2048",
            }
        )
        runtime = output["runtime_env"]
        self.assertEqual(
            runtime["worker_process_setup_hook"],
            "examples.on_policy_distillation.mopd.verified_baseline_entrypoint.worker_setup",
        )
        self.assertEqual(runtime["env_vars"]["RELAX_BASELINE_PRIOR_WORKER_SETUP_HOOK"], "existing.hook")
        self.assertEqual(runtime["env_vars"]["PYTHONSAFEPATH"], "1")
        self.assertEqual(runtime["env_vars"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(runtime["pip"], ["fixture"])
        worker = runtime["env_vars"]
        self.assertEqual(worker["EXISTING"], "keep")
        self.assertEqual(worker["EXTRA_SETTING"], "preserve-me")
        self.assertEqual(worker["HF_HUB_OFFLINE"], "1")
        self.assertEqual(worker["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(worker["SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"], "2048")
        self.assertEqual(worker["RELAX_SKIP_GLOBAL_CLEANUP"], "1")
        self.assertEqual(worker["SHARED_TEACHER_MANIFEST"], output["runtime_manifest_path"])
        self.assertIn("HF_HUB_OFFLINE", worker["RELAX_PROPAGATE_ENV_VARS"].split(","))
        encoded = next(arg.split("=", 1)[1] for arg in output["ray_argv"] if arg.startswith("--runtime-env-json="))
        self.assertEqual(json.loads(encoded), runtime)

    def test_memory_backend_and_parquet_input(self):
        self.manifest["backend"] = "sglang-memory-v1"
        self._write_manifest()
        prompt = str(Path(self.env["DATA_DIR"]) / "existing prompts.parquet")
        output = self._run({"WEIGHT_SOURCE": "memory", "PROMPT_SET": prompt})
        self.assertEqual(output["runtime_manifest"]["backend"], "sglang-memory-v1")
        self.assertEqual(self._flag(output, "--prompt-data"), prompt)
        self.assertIn("memory", self._flag(output, "--save"))

    def test_explicit_paths_monitoring_and_wait_overrides(self):
        for no_wait in ("", "0", "1"):
            with self.subTest(no_wait=no_wait):
                output = self._run(
                    {
                        "RAY_NO_WAIT": no_wait,
                        "USE_CLEARML": "0",
                        "USE_METRICS_SERVICE": "0",
                        "EXP_NAME": "explicit-experiment",
                        "STUDENT_MODEL": str(Path(self.env["MODEL_DIR"]) / "custom-local-9b"),
                        "WORKING_DIR": "checkout with spaces",
                        "RAY_SUBMISSION_ID": "chosen-submission",
                        "RAY_DASHBOARD_ADDRESS": "http://existing-head:8265",
                    }
                )
                self.assertEqual("--no-wait" in output["ray_argv"], bool(no_wait))
                self.assertNotIn("--use-clearml", output["train_argv"])
                self.assertNotIn("--use-metrics-service", output["train_argv"])
                self.assertEqual(self._flag(output, "--load"), str(Path(self.env["SAVE_DIR"]) / "explicit-experiment"))
                self.assertTrue(self._flag(output, "--hf-checkpoint").endswith("custom-local-9b"))
                self.assertIn("checkout with spaces", output["ray_argv"])
                self.assertIn("chosen-submission", output["ray_argv"])

    def test_real_launch_fails_local_validation_before_entrypoint_or_ray(self):
        prompt = self.base / "missing-checkpoints.jsonl"
        prompt.write_text("{}\n")
        error = self._run(
            {"PROMPT_SET": str(prompt), "RAY_ADDRESS": "unused:6379", "RAY_DASHBOARD_ADDRESS": "http://unused:8265"},
            expect_success=False,
            extra_args=(),
        )
        self.assertIn("checkpoint directory does not exist", error)

    def test_manifest_backend_must_match_selected_source(self):
        self.assertIn("backend does not match", self._run({"WEIGHT_SOURCE": "memory"}, expect_success=False))

    def test_reject_invalid_manifest_topology_or_routes(self):
        for field, value, message in (
            ("version", 2, "version=1"),
            ("tp_size", 2, "tp_size=1"),
            ("tp_size", True, "tp_size=1"),
            ("initial_teacher", "T6", "initial_teacher"),
            ("teachers", {"T1": "one"}, "exactly Teachers T1..T5"),
            ("teachers", {f"T{i}": "same" for i in range(1, 6)}, "distinct checkpoint"),
        ):
            with self.subTest(field=field, value=value):
                original = self.manifest[field]
                self.manifest[field] = value
                self._write_manifest()
                self.assertIn(message, self._run(expect_success=False))
                self.manifest[field] = original
                self._write_manifest()

    def test_reject_invalid_batch_and_path_overrides(self):
        for overrides, message in (
            ({"WEIGHT_SOURCE": "delta"}, "disk or memory"),
            ({"ROLLOUT_BATCH_SIZE": "16"}, "divisible by five"),
            ({"GLOBAL_BATCH_SIZE": "80"}, "GLOBAL_BATCH_SIZE must equal"),
            ({"N_SAMPLES_PER_PROMPT": "1", "GLOBAL_BATCH_SIZE": "20"}, "at least two samples"),
            ({"MAX_RESPONSE_LEN": "20000"}, "smaller than MAX_CONTEXT_LEN"),
            ({"PROMPT_SET": "/data/prompts.csv"}, "JSONL or Parquet"),
            ({"EXP_NAME": "../escape"}, "stable directory name"),
            ({"USE_CLEARML": "yes"}, "Monitoring overrides"),
        ):
            with self.subTest(overrides=overrides):
                self.assertIn(message, self._run(overrides, expect_success=False))

    def test_env_dry_run_is_equivalent_and_architecture_flags_are_not_overridable(self):
        self.assertTrue(self._run({"DRY_RUN": "1"}, extra_args=())["dry_run"])
        error = self._run(extra_args=("--dry-run", "--hidden-size", "896"), expect_success=False)
        self.assertIn("Unknown argument", error)


if __name__ == "__main__":
    unittest.main()
