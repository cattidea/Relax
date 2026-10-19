# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Nested deadlines for the opt-in, single-engine shared Teacher lifecycle.

HTTP completion is uncertain after a timeout. These limits bound waiting; they
never authorize reopening admission or declaring GPU memory released.
"""

from typing import Any, Callable


NATIVE_LIFECYCLE_TIMEOUT_S = 20.0
SERVICE_CONTROL_WAIT_S = 60.0
ENGINE_LIFECYCLE_WAIT_S = 25.0
ENGINE_SHUTDOWN_WAIT_S = 30.0
CHILD_SHUTDOWN_WAIT_S = 35.0
SERVICE_DRAIN_WAIT_S = 2.0
SERVICE_CLOSE_WAIT_S = 5.0
MANAGER_SHUTDOWN_WAIT_S = 50.0


def run_rank_zero_lifecycle(operation: Callable[[], None], *, process_group: Any, label: str) -> None:
    """Every training rank calls this before proceeding to GPU collectives.

    Ordinary control/Ray/HTTP exceptions become the same bounded error on all
    ranks. Process termination and KeyboardInterrupt are not swallowed. The
    caller supplies the existing world Gloo group whose global leader is 0.
    """
    import torch.distributed as dist

    from relax.utils.logging_utils import get_logger

    error = [None]
    rank = dist.get_rank(group=process_group)
    if rank == 0:
        try:
            operation()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"[:4096]
    dist.broadcast_object_list(error, src=0, group=process_group)
    if error[0] is not None:
        get_logger(__name__).error(
            "[shared_teacher_lifecycle_failed] rank=%s operation=%s error=%s", rank, label, error[0]
        )
        raise RuntimeError(f"Shared Teacher {label} failed on rank 0: {error[0]}")
