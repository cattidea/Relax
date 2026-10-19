# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace
from typing import Any

import pytest

from relax.engine.rollout import on_policy_distillation as opd
from relax.utils.types import Sample


class _Response:
    def __init__(self, body: dict, status: int = 200) -> None:
        self.body = body
        self.status = status

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def json(self) -> dict:
        return self.body

    async def text(self) -> str:
        return "teacher gateway is unavailable"

    def raise_for_status(self) -> None:
        raise RuntimeError(f"HTTP {self.status}")


class _Session(_Response):
    def __init__(self, failed_response: _Response) -> None:
        super().__init__({})
        self.failed_response = failed_response

    def post(self, url: str, **kwargs: Any) -> _Response:
        payload = json.loads(kwargs["data"])
        if payload["input_ids"][0] == 2:
            return self.failed_response
        return _Response({"meta_info": {"input_token_logprobs": [[None, 5], [-0.2, 6], [-0.3, 7]]}})


def _manager(strict: bool | None) -> opd.OpdManager:
    args = SimpleNamespace(
        opd_token_selection="student_sampled",
        opd_log_prob_top_k=0,
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
        opd_teacher_url="http://gateway/teachers/A/generate",
    )
    if strict is not None:
        args.opd_require_all_teacher_scores = strict
    return opd.OpdManager(args)


def _samples() -> list[Sample]:
    return [Sample(index=i, tokens=[i + 1, 5, 6, 7], response_length=2) for i in range(2)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_response",
    [
        _Response({}, status=503),
        _Response({"meta_info": {}}),
        _Response({"meta_info": {"input_token_logprobs": [[None, 5], [-0.2, 6]]}}),
    ],
    ids=["http-failure", "missing-logprobs", "wrong-logprob-length"],
)
async def test_strict_teacher_prefill_rejects_partial_success_before_assembly(
    monkeypatch: pytest.MonkeyPatch, failed_response: _Response
) -> None:
    monkeypatch.setattr(opd, "_create_teacher_client_session", lambda args: _Session(failed_response))
    manager = _manager(True)
    assembled = []
    monkeypatch.setattr(manager, "_assemble_transfer", lambda samples: assembled.append(samples))
    samples = _samples()

    with pytest.raises(RuntimeError, match="Strict OPD teacher scoring failed for 1 non-empty samples"):
        await manager.prefill(samples)

    assert samples[0].teacher_log_probs == pytest.approx([-0.2, -0.3])
    assert samples[1].teacher_log_probs is None
    assert not assembled


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [None, False], ids=["default", "explicitly-disabled"])
async def test_default_teacher_prefill_preserves_partial_success_behavior(
    monkeypatch: pytest.MonkeyPatch, strict: bool | None
) -> None:
    monkeypatch.setattr(opd, "_create_teacher_client_session", lambda args: _Session(_Response({}, status=503)))
    manager = _manager(strict)
    assembled = []
    monkeypatch.setattr(manager, "_assemble_transfer", lambda samples: assembled.append(samples))
    samples = _samples()

    await manager.prefill(samples)

    assert samples[0].teacher_log_probs == pytest.approx([-0.2, -0.3])
    assert samples[1].teacher_log_probs is None
    assert assembled == [samples]


@pytest.mark.asyncio
async def test_strict_teacher_prefill_accepts_success_and_skips_empty_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(opd, "_create_teacher_client_session", lambda args: _Session(_Response({}, status=503)))
    manager = _manager(True)
    samples = _samples()
    samples[1].response_length = 0
    assembled = []
    monkeypatch.setattr(manager, "_assemble_transfer", lambda samples: assembled.append(samples))

    await manager.prefill(samples)

    assert samples[0].teacher_log_probs == pytest.approx([-0.2, -0.3])
    assert samples[1].teacher_log_probs is None
    assert assembled == [samples]


def test_strict_teacher_guard_ignores_failed_empty_samples() -> None:
    empty = Sample(index=0, response_length=0)
    scored = Sample(index=1, response_length=2)
    opd.OpdManager._raise_if_any_failed([empty, scored], [False, True])


@pytest.mark.asyncio
async def test_default_teacher_prefill_still_rejects_total_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opd, "_create_teacher_client_session", lambda args: _Session(_Response({}, status=503)))
    manager = _manager(None)
    failed = _samples()[1]

    with pytest.raises(RuntimeError, match="All OPD teacher fetches failed"):
        await manager.prefill(failed)
