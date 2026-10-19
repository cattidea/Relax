# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Refuse stale source and avoid a foreign checkout's current directory."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/on_policy_distillation/mopd/verified_baseline_entrypoint.py"
spec = importlib.util.spec_from_file_location("baseline_guard_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourceGuardTests(unittest.TestCase):
    def run_guard(self, *, stale=False):
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in module.SOURCES}
        if stale:
            hashes["relax/utils/opd/opd_utils.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            foreign = Path(temporary) / "relax"
            foreign.mkdir()
            (foreign / "__init__.py").write_text("raise RuntimeError('wrong checkout imported')\n")
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--check-only"],
                cwd=temporary,
                env={**os.environ, "RELAX_BASELINE_SOURCE_SHA256": json.dumps(hashes)},
                text=True,
                capture_output=True,
            )

    def test_foreign_current_directory_cannot_select_old_relax(self):
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["relax_package"], str(ROOT / "relax/__init__.py"))

    def test_changed_source_fails_before_training(self):
        result = self.run_guard(stale=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source files changed", result.stderr)


if __name__ == "__main__":
    unittest.main()
