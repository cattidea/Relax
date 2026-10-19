# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU propagation through real Actor/runner/Controller/entrypoint functions.

Only orchestration methods are loaded from their source AST; Ray/Serve handles
are doubles. The actual background thread and async completion are exercised.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
ERROR = "Shared Teacher onload failed on rank 0: ReadTimeout: native resume exceeded 20 seconds"


def functions(path, names, namespace, *, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    nodes = [
        node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


def runner_class():
    name = "shared_teacher_cpu_background_loop"
    spec = importlib.util.spec_from_file_location(name, ROOT / "relax/components/background_loop.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.BackgroundLoopRunner


def actor_case(*, shared=True, health=True, error=None, report_error=None):
    namespace = functions("relax/components/actor.py", {"_background_run"}, {}, class_name="Actor")
    actor = SimpleNamespace(
        config=SimpleNamespace(
            num_rollout=2,
            fully_async=False,
            colocate=True,
            debug_train_only=False,
            use_health_check=health,
            opd_shared_teacher_manifest="teachers.json" if shared else None,
        ),
        step=1,
        _stop_event=threading.Event(),
        _lock=threading.RLock(),
        _logger=Mock(),
        healthy=Mock(),
        _wait_for_rollout_data=Mock(return_value=True),
        _execute_training=Mock(side_effect=error or RuntimeError(ERROR)),
        data_system_client=Mock(),
    )
    if report_error:
        actor.healthy.report_error.remote.side_effect = report_error
    runner = runner_class()()

    async def operation():
        await runner.run(lambda: namespace["_background_run"](actor))

    return actor, operation, runner


def controller_class(operation, *, shared=True, health_callback=None, events=None):
    events = [] if events is None else events
    namespace = {
        "asyncio": asyncio,
        "Awaitable": object,
        "Any": object,
        "logger": Mock(),
        "run": asyncio.run,
        "ROLES": SimpleNamespace(actor="actor"),
    }
    # Real type annotations must compile without importing the heavy module.
    from typing import Any, Awaitable

    namespace.update(Any=Any, Awaitable=Awaitable)
    names = {"training_loop", "_supervise_shared_teacher_services", "restart_serve", "_on_service_unhealthy"}
    functions("relax/core/controller.py", names, namespace, class_name="Controller")

    class ControllerCase:
        def __init__(self, config, runtime_env=None):
            self.config = SimpleNamespace(
                debug_train_only=True,
                debug_rollout_only=False,
                opd_shared_teacher_manifest="teachers.json" if shared else None,
            )
            self._pending_task_refs_lock = threading.Lock()
            self._pending_task_refs = []
            self._shared_teacher_failure = None
            self._restarting = False
            self._cleanup_s3_model_weights_after_init = Mock()
            self._report_error_to_metrics_service = Mock()
            self.dcs = object()
            self._health_manager = Mock()
            self._global_restart = Mock(side_effect=AssertionError("must not rebuild shared Teacher"))
            self._restart_consumed_event = Mock(side_effect=AssertionError("must not start restart cycle"))
            self.service_entered = threading.Event()

            async def service_operation():
                self.service_entered.set()
                if health_callback:
                    health_callback(self)
                await operation()

            self.serve_dict = {"actor": SimpleNamespace(role="actor", run=service_operation)}

        def shutdown(self):
            events.append("controller_cleanup")

    for name in names:
        if name in namespace:
            setattr(ControllerCase, name, namespace[name])
    return ControllerCase


def entrypoint_case(controller, events, *, real_exit=False):
    namespace = {
        "os": os,
        "sys": sys,
        "signal": Mock(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, Signals=signal.Signals),
        "threading": threading,
        "atexit": Mock(),
        "yaml": Mock(safe_load=Mock(return_value={})),
        "cur_file_dir": ROOT,
        "logger": Mock(),
        "ray": Mock(),
        "serve": Mock(),
        "Controller": controller,
        "post_process_env": lambda args, env: env,
        "init_tracking": Mock(),
        "maybe_pin_baseline_to_stable": Mock(),
        "_ctrl": None,
        "_shutdown_done": False,
        "_kernel_cache_heartbeat": None,
        "_SHUTDOWN_TIMEOUT_SEC": 120,
        "_NORMAL_SHUTDOWN_TIMEOUT_SEC": 1800,
    }
    functions(
        "relax/entrypoints/train.py",
        {"main", "_graceful_shutdown", "_hard_exit", "_shutdown_serve_for_job"},
        namespace,
    )

    def record(value):
        events.append(value)
        if real_exit:
            sys.stdout.write(json.dumps({"stage": value}) + "\n")
            sys.stdout.flush()

    namespace["serve"].delete.side_effect = lambda name: record("serve_delete:" + name)
    namespace["ray"].shutdown.side_effect = lambda: record("ray_disconnect")
    if not real_exit:

        def hard_exit(code):
            record(("exit", code))
            raise SystemExit(code)

        namespace["_hard_exit"] = hard_exit
    return namespace


class SharedFailureTests(unittest.TestCase):
    def test_shared_error_reaches_all_waiters_and_preserves_original_when_health_report_fails(self):
        for report_error in (None, RuntimeError("health transport down")):
            with self.subTest(report_error=report_error):
                error = RuntimeError(ERROR)
                actor, operation, runner = actor_case(error=error, report_error=report_error)

                async def execute():
                    return await asyncio.wait_for(asyncio.gather(operation(), operation(), return_exceptions=True), 2)

                results = asyncio.run(execute())
                self.assertTrue(all(result is error for result in results))
                actor.healthy.report_error.remote.assert_called()
                actor.healthy.update_heartbeat.remote.assert_not_called()
                self.assertEqual(actor.step, 1)
                self.assertFalse(runner._state.thread.is_alive())

    def test_nonshared_health_policy_is_unchanged(self):
        for health in (True, False):
            with self.subTest(health=health):
                error = RuntimeError(ERROR)
                actor, operation, _ = actor_case(shared=False, health=health, error=error)
                if health:
                    self.assertIsNone(asyncio.run(operation()))
                else:
                    with self.assertRaises(RuntimeError) as result:
                        asyncio.run(operation())
                    self.assertIs(result.exception, error)
                actor.healthy.report_error.remote.assert_called_once()

    def test_actor_failure_runs_actual_controller_and_entrypoint_cleanup_before_exit_one(self):
        _, operation, _ = actor_case()
        events = []
        controller = controller_class(operation, events=events)
        namespace = entrypoint_case(controller, events)
        with patch.dict(os.environ, {"RELAX_SKIP_GLOBAL_CLEANUP": "1"}), self.assertRaises(SystemExit) as result:
            namespace["main"](SimpleNamespace())
        self.assertEqual(result.exception.code, 1)
        self.assertEqual(
            events,
            [
                "controller_cleanup",
                "serve_delete:actor",
                "serve_delete:dcs_coordinator",
                "ray_disconnect",
                ("exit", 1),
            ],
        )
        namespace["serve"].shutdown.assert_not_called()
        namespace["logger"].exception.assert_called_once()
        namespace["_ctrl"]._global_restart.assert_not_called()

    def test_health_callback_without_actor_exception_fails_through_scoped_cleanup(self):
        async def pending():
            await asyncio.Event().wait()

        events = []
        controller = controller_class(
            pending, health_callback=lambda ctrl: ctrl._on_service_unhealthy("actor"), events=events
        )
        namespace = entrypoint_case(controller, events)
        with patch.dict(os.environ, {"RELAX_SKIP_GLOBAL_CLEANUP": "1"}), self.assertRaises(SystemExit) as result:
            namespace["main"](SimpleNamespace())
        self.assertEqual(result.exception.code, 1)
        self.assertEqual(events[-1], ("exit", 1))
        self.assertIn("controller_cleanup", events)
        namespace["_ctrl"]._global_restart.assert_not_called()
        self.assertFalse(namespace["_ctrl"]._restarting)

    def test_health_callback_from_actor_thread_races_error_without_restarting(self):
        actor, operation, _ = actor_case()
        events = []
        namespace = entrypoint_case(controller_class(operation, events=events), events)
        callback_threads = []

        def health_report(*args):
            callback_threads.append(threading.get_ident())
            namespace["_ctrl"]._on_service_unhealthy("actor")

        actor.healthy.report_error.remote.side_effect = health_report
        with patch.dict(os.environ, {"RELAX_SKIP_GLOBAL_CLEANUP": "1"}), self.assertRaises(SystemExit) as result:
            namespace["main"](SimpleNamespace())
        self.assertEqual(result.exception.code, 1)
        self.assertEqual(len(callback_threads), 1)
        self.assertNotEqual(callback_threads[0], threading.get_ident())
        self.assertIn("controller_cleanup", events)
        namespace["_ctrl"]._global_restart.assert_not_called()

    def test_shared_success_still_exits_zero_with_scoped_cleanup(self):
        async def successful():
            return None

        events = []
        namespace = entrypoint_case(controller_class(successful, events=events), events)
        with patch.dict(os.environ, {"RELAX_SKIP_GLOBAL_CLEANUP": "1"}), self.assertRaises(SystemExit) as result:
            namespace["main"](SimpleNamespace())
        self.assertEqual(result.exception.code, 0)
        self.assertIn("controller_cleanup", events)
        namespace["logger"].exception.assert_not_called()

    def test_latched_failure_before_start_or_after_completion_cannot_return_success(self):
        async def successful():
            return None

        controller = controller_class(successful)
        for before in (True, False):
            with self.subTest(before=before):
                obj = controller(None)
                if before:
                    obj._on_service_unhealthy("actor")

                async def operation():
                    if not before:
                        obj._on_service_unhealthy("rollout")

                with self.assertRaisesRegex(RuntimeError, "Shared Teacher"):
                    asyncio.run(obj._supervise_shared_teacher_services(operation()))
                obj._global_restart.assert_not_called()

    def test_two_health_callbacks_latch_first_error_without_rebuild(self):
        async def no_op():
            pass

        obj = controller_class(no_op)(None)
        obj._on_service_unhealthy("actor")
        first = obj._shared_teacher_failure
        obj._on_service_unhealthy("rollout")
        self.assertIs(obj._shared_teacher_failure, first)
        obj._global_restart.assert_not_called()
        obj._health_manager.increment_restart_count.assert_not_called()

    def test_real_subprocess_exit_code_is_nonzero_not_only_mocked(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe"],
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "RELAX_SKIP_GLOBAL_CLEANUP": "1"},
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        rows = [json.loads(line)["stage"] for line in result.stdout.splitlines()]
        self.assertEqual(rows, ["serve_delete:actor", "serve_delete:dcs_coordinator", "ray_disconnect"])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--probe"]:
        if len(sys.argv) > 2:
            ROOT = Path(sys.argv[2])
        _, operation, _ = actor_case()
        events = []
        namespace = entrypoint_case(controller_class(operation, events=events), events, real_exit=True)
        namespace["main"](SimpleNamespace())
    else:
        unittest.main()
