# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Instrument only the container's frozen ea3f0cee test copies, fail closed.

Usage: python /duration-harness/instrument_duration.py /workspace/Relax
No test is imported/executed. All files are verified and compiled before any
write. Removing marker statements/contexts must reproduce the original AST.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

from duration_trace import TARGET_NODES


FROZEN_SOURCE = "ea3f0cee94b9400cfe1591beb4f2dbfe712d362c"
EXPECTED = {
    "tests/backends/fsdp/test_weight_transaction.py": "3fdc8cd86858df6a1d758531bb482575ceaa93515c6f9324d70d7be0ad9fd1bd",
    "tests/backends/megatron/test_qsa_mixed.py": "dfca1b1e8bbd7a9144c716016c26e85665ee8ce4a73881fcfd022f8c709912a8",
    "tests/backends/megatron/test_gdn_cp_gpu.py": "2482b095802a0b0d4d24341c817f81ef1af29dc296e6d8ed0b5ec9f58b711aa6",
}


def replace_once(source: str, anchor: str, replacement: str) -> str:
    if source.count(anchor) != 1:
        raise ValueError(f"Expected one source anchor, found {source.count(anchor)}: {anchor!r}")
    return source.replace(anchor, replacement, 1)


def wrap(source: str, anchor: str, phase: str, *, collection: bool = False) -> str:
    first = anchor.splitlines()[0]
    indent = first[: len(first) - len(first.lstrip())]
    option = ", collection=True" if collection else ""
    replacement = f'{indent}with _ci_trace.span("{phase}"{option}):\n' + textwrap.indent(anchor, "    ")
    return replace_once(source, anchor, replacement)


def wrap_until(source: str, start: str, stop: str, phase: str, *, collection: bool = False) -> str:
    if source.count(start) != 1 or source.count(stop) != 1:
        raise ValueError(f"Nonunique interval anchors: {start!r}, {stop!r}")
    left, right = source.index(start), source.index(stop)
    if left >= right:
        raise ValueError(f"Reversed interval: {phase}")
    return wrap(source, source[left:right], phase, collection=collection)


def function(source: str, name: str, transform: Callable[[str], str]) -> str:
    definitions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(definitions) != 1:
        raise ValueError(f"Expected one function: {name}")
    node = definitions[0]
    lines = source.splitlines(keepends=True)
    original = "".join(lines[node.lineno - 1 : node.end_lineno])
    return replace_once(source, original, transform(original))


def module(source: str, name: str) -> str:
    source = replace_once(
        source,
        "from __future__ import annotations\n",
        "from __future__ import annotations\n\nimport duration_trace as _ci_trace\n"
        f'_ci_trace.mark("{name}.module_enter", collection=True)\n',
    )
    return wrap(source, "import torch\n", f"{name}.import_torch", collection=True)


def fsdp(source: str) -> str:
    source = module(source, "fsdp")
    source = wrap(source, "import torch.distributed as dist\n", "fsdp.import_torch_distributed", collection=True)
    source = wrap_until(
        source,
        "from relax.backends.fsdp.actor import (\n",
        "\n\nclass _Ref:",
        "fsdp.import_dependencies",
        collection=True,
    )

    def entry(body: str) -> str:
        body = replace_once(
            body,
            "    # Spawn reimports this module",
            '    _ci_trace.set_rank(rank)\n    _ci_trace.mark("fsdp.worker_enter")\n    # Spawn reimports this module',
        )
        body = wrap(
            body,
            "    if not start.wait(timeout=startup_timeout):\n"
            '        raise TimeoutError(f"rank {rank} did not receive the test start signal")\n',
            "fsdp.worker_wait_start",
        )
        return wrap(body, "        worker(rank, port, *worker_args, queue)\n", "fsdp.worker_body")

    source = function(source, "_failure_worker_entry", entry)

    def parent(body: str) -> str:
        body = wrap(body, "        for process in processes:\n            process.start()\n", "fsdp.process_start")
        body = wrap_until(
            body,
            "        deadline = time.monotonic() + startup_timeout\n",
            "        # Keep the strict execution deadline",
            "fsdp.wait_all_ready",
        )
        body = wrap(body, "        start.set()\n", "fsdp.release_start")
        return wrap(
            body,
            "        for process in processes:\n            process.join(timeout=max(0, deadline - time.monotonic()))\n",
            "fsdp.join_workers",
        )

    source = function(source, "_run_failure_workers", parent)

    def worker(body: str) -> str:
        body = wrap(
            body,
            '    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=10))\n',
            "fsdp.gloo_init",
        )
        body = wrap(body, "    from relax.utils import distributed_utils\n", "fsdp.worker_import")
        body = wrap(
            body,
            '    distributed_utils.GLOO_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))\n',
            "fsdp.gloo_new_group",
        )
        body = wrap_until(
            body,
            '        local_error = "RuntimeError:',
            "    except WeightSyncError as exc:\n",
            "fsdp.error_collectives",
        )
        return wrap(body, "        dist.destroy_process_group()\n", "fsdp.destroy_group")

    source = function(source, "_serialization_failure_worker", worker)
    return source + '\n_ci_trace.mark("fsdp.module_ready", collection=True)\n'


def qsa(source: str) -> str:
    source = module(source, "qsa")
    source = replace_once(
        source,
        'torch.cuda.is_available(), reason="Mixed QSA precision requires CUDA BF16"',
        '_ci_trace.call("qsa.cuda_available_precision", torch.cuda.is_available, collection=True), '
        'reason="Mixed QSA precision requires CUDA BF16"',
    )
    source = replace_once(
        source,
        'torch.cuda.is_available(), reason="Mixed QSA adapter validation requires CUDA"',
        '_ci_trace.call("qsa.cuda_available_binding", torch.cuda.is_available, collection=True), '
        'reason="Mixed QSA adapter validation requires CUDA"',
    )

    def routes(body: str) -> str:
        body = wrap(
            body, "    from megatron.core.transformer.experimental_attention_variant import qsa\n", "qsa.routes_import"
        )
        return wrap(
            body,
            '    positions = torch.cat([torch.arange(length, device="cuda") for length in lengths]).int().unsqueeze(0)\n',
            "qsa.first_cuda_allocation",
        )

    source = function(source, "_routes", routes)

    def run(body: str) -> str:
        body = wrap_until(
            body,
            "    from relax.models.qwen3_8 import qsa_community as patch\n",
            "    torch.manual_seed(20261013)\n",
            "qsa.import_dependencies",
        )
        body = wrap(body, "    torch.manual_seed(20261013)\n", "qsa.seed")
        body = wrap(body, "    selection = _routes(lengths, budget)\n", "qsa.routes")
        body = wrap_until(
            body, "    length = sum(lengths)\n", "    actual = patch._community_id_sparse_forward(\n", "qsa.inputs"
        )
        body = wrap_until(
            body,
            "    actual = patch._community_id_sparse_forward(\n",
            "    gradients = torch.autograd.grad(actual, actual_inputs, upstream)\n",
            "qsa.forward",
        )
        body = wrap(body, "    gradients = torch.autograd.grad(actual, actual_inputs, upstream)\n", "qsa.backward")
        body = wrap_until(
            body,
            "    indices = torch.nn.functional.pad(token_routes(selection, 1, length)[0], (0, 1), value=-1)\n",
            "    expected = _reference(values, indices, upstream)\n",
            "qsa.reference_routes",
        )
        body = wrap(body, "    expected = _reference(values, indices, upstream)\n", "qsa.reference_forward_backward")
        return wrap(body, body[body.index('    report = {"lengths":') :], "qsa.assertions_and_original_report")

    source = function(source, "_run_case", run)
    source = wrap(
        source,
        "    subprocess.run(\n"
        "        [sys.executable, __file__, json.dumps(lengths), str(budget)], env=environment, check=True, timeout=600\n"
        "    )\n",
        "qsa.subprocess",
    )
    return replace_once(
        source,
        'if __name__ == "__main__":\n',
        '_ci_trace.mark("qsa.module_ready", collection=True)\n\n'
        'if __name__ == "__main__":\n    _ci_trace.mark("qsa.child_main")\n',
    )


def gdn(source: str) -> str:
    source = module(source, "gdn")
    source = wrap(source, "import torch.multiprocessing as mp\n", "gdn.import_multiprocessing", collection=True)
    source = replace_once(
        source,
        "not torch.cuda.is_available() or torch.cuda.device_count() < 2,\n",
        'not _ci_trace.call("gdn.cuda_available", torch.cuda.is_available, collection=True)\n'
        '    or _ci_trace.call("gdn.cuda_device_count", torch.cuda.device_count, collection=True) < 2,\n',
    )

    def build(body: str) -> str:
        body = wrap_until(
            body, "    from megatron.core import parallel_state\n", "    torch.manual_seed(123)\n", "gdn.model_imports"
        )
        body = wrap_until(body, "    torch.manual_seed(123)\n", "    return (\n", "gdn.model_seed_and_config")
        return wrap(body, body[body.index("    return (\n") :], "gdn.model_construct_and_move")

    source = function(source, "_build_gdn", build)

    def compute(body: str) -> str:
        body = wrap(body, "    from torch.utils.checkpoint import checkpoint\n", "gdn.checkpoint_import")
        body = wrap_until(body, "    gdn.zero_grad(set_to_none=True)\n", "    def forward(x):\n", "gdn.fb_prepare")
        body = wrap(
            body,
            "    out = checkpoint(forward, hidden, use_reentrant=False) if recompute else forward(hidden)\n",
            "gdn.forward",
        )
        body = wrap(body, "    (out.float() * grad_out).sum().backward()\n", "gdn.backward")
        return wrap(
            body,
            "    grads = {name: param.grad.detach().float().clone() for name, param in gdn.named_parameters()}\n",
            "gdn.clone_gradients",
        )

    source = function(source, "_forward_backward", compute)

    def worker(body: str) -> str:
        body = replace_once(
            body,
            "    torch.cuda.set_device(rank)\n",
            '    _ci_trace.set_rank(rank)\n    _ci_trace.mark("gdn.worker_enter")\n    torch.cuda.set_device(rank)\n',
        )
        body = wrap(body, "    torch.cuda.set_device(rank)\n", "gdn.set_device")
        body = wrap(
            body,
            "    _use_one_fla_config()  # Install before Megatron/FLA imports create kernels.\n",
            "gdn.fla_config",
        )
        body = wrap_until(
            body,
            "    import torch.distributed as dist\n",
            '    dist.init_process_group("nccl",',
            "gdn.import_dependencies",
        )
        body = wrap(
            body,
            '    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=2)\n',
            "gdn.nccl_init",
        )
        body = wrap(
            body, "    parallel_state.initialize_model_parallel(context_parallel_size=2)\n", "gdn.model_parallel_init"
        )
        body = wrap_until(
            body,
            '        set_args(SimpleNamespace(recompute_granularity="full"))\n',
            "        _check_layout_round_trip(cp_group)\n",
            "gdn.patch_and_group",
        )
        body = wrap(body, "        _check_layout_round_trip(cp_group)\n", "gdn.layout_checks")
        body = wrap(body, "        solo = [dist.new_group([r]) for r in range(2)][rank]\n", "gdn.solo_groups")
        body = wrap(body, "        models = {mode: _build_gdn(mode) for mode in modes}\n", "gdn.build_models")
        body = wrap_until(
            body,
            '        for param in models["chunkwise"].parameters():\n',
            "        # Unequal packed samples;",
            "gdn.broadcast_and_load_state",
        )
        body = wrap_until(
            body,
            "        lengths = [768, 256]\n",
            "        ref_out, ref_dx, ref_grads = _forward_backward(",
            "gdn.inputs_and_packed_metadata",
        )
        body = wrap(
            body,
            '        ref_out, ref_dx, ref_grads = _forward_backward(models["chunkwise"], hidden, packed(solo, 1), grad_out)\n',
            "gdn.cp1_forward_backward",
        )
        body = wrap(
            body,
            "            out, dx, grads = _forward_backward(\n"
            '                gdn, shard, packed(cp_group, 2), grad_shard, recompute=mode == "all_gather" or len(models) == 1\n'
            "            )\n",
            "gdn.cp2_forward_backward",
        )
        body = wrap_until(
            body,
            '            for name, got, expected in (("output", out, ref_out),',
            "            for name, grad in grads.items():\n",
            "gdn.output_assertions",
        )
        body = wrap_until(
            body,
            "            for name, grad in grads.items():\n",
            "    finally:\n",
            "gdn.gradient_collectives_and_assertions",
        )
        body = wrap(body, "        parallel_state.destroy_model_parallel()\n", "gdn.destroy_model_parallel")
        return wrap(body, "        dist.destroy_process_group()\n", "gdn.destroy_nccl")

    source = function(source, "_worker_gdn_cp", worker)

    def parent(body: str) -> str:
        body = wrap(
            body,
            "    from megatron.core.ssm.gated_delta_net import GatedDeltaNet\n",
            "gdn.parent_import_model",
        )
        return wrap(
            body,
            "    mp.spawn(_worker_gdn_cp, args=(f\"file://{tmp_path / 'gdn_cp_init'}\", mode), nprocs=2, join=True)\n",
            "gdn.spawn_and_join",
        )

    source = function(source, "test_gdn_cp_layout_and_gradients", parent)
    return source + '\n_ci_trace.mark("gdn.module_ready", collection=True)\n'


def _trace_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_ci_trace"
        and node.func.attr == name
    )


class RemoveInstrumentation(ast.NodeTransformer):
    """Prove the original statements/expressions remain in their old order."""

    def visit_Import(self, node: ast.Import) -> ast.Import | None:
        if len(node.names) == 1 and node.names[0].name == "duration_trace" and node.names[0].asname == "_ci_trace":
            return None
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if _trace_call(node.value, "mark") or _trace_call(node.value, "set_rank"):
            return None
        return self.generic_visit(node)

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        if len(node.items) == 1 and _trace_call(node.items[0].context_expr, "span"):
            self.generic_visit(node)
            return node.body
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if _trace_call(node, "call"):
            if len(node.args) != 2 or any(keyword.arg != "collection" for keyword in node.keywords):
                raise ValueError("Unexpected traced availability-call signature")
            return ast.Call(func=self.visit(node.args[1]), args=[], keywords=[])
        return self.generic_visit(node)


def instrument(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    transformed: dict[Path, str] = {}
    evidence = []
    for (relative, expected), transform in zip(EXPECTED.items(), (fsdp, qsa, gdn), strict=True):
        path = root / relative
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Source must be a regular file inside the checkout: {relative}")
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError(f"Frozen-source SHA256 mismatch: {relative}")
        source = original.decode("utf-8")
        result = transform(source)
        compiled = ast.parse(result, filename=relative)
        restored = RemoveInstrumentation().visit(ast.parse(result))
        if ast.dump(restored, include_attributes=False) != ast.dump(ast.parse(source), include_attributes=False):
            raise ValueError(f"Instrumentation altered the original AST: {relative}")
        compile(compiled, relative, "exec")
        transformed[path] = result
        evidence.append(
            {
                "path": relative,
                "original_sha256": expected,
                "instrumented_sha256": hashlib.sha256(result.encode()).hexdigest(),
            }
        )
    manifest = {
        "source_ref": FROZEN_SOURCE,
        "mode": "original_calls_no_extra_sync",
        "time_basis": "host_wall_no_added_cuda_synchronization",
        "nodes": TARGET_NODES,
        "files": evidence,
    }
    destination = os.environ.get("CI_DURATION_TRACE_DIR")
    if destination:
        Path(destination).mkdir(parents=True, exist_ok=True)
    # No file is changed until every hash, anchor, AST and compile check passes.
    # On any write error the command fails and the workflow must not run pytest.
    for path, result in transformed.items():
        path.write_text(result, encoding="utf-8")
    if destination:
        (Path(destination) / "instrumentation.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: instrument_duration.py CHECKOUT_ROOT")
    instrument(Path(sys.argv[1]))
