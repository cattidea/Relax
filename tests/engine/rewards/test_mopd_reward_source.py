# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep dataset reward routing independent of logical Teacher IDs."""

from relax.engine.rewards import mopd


def test_teacher_route_keeps_explicit_dapo_scorer(monkeypatch):
    calls = []

    def dapo(response, label):
        calls.append((response, label))
        return {"score": 0.75}

    monkeypatch.setattr(mopd, "compute_score_dapo", dapo)
    monkeypatch.setattr(mopd, "_compute_gsm8k_score", lambda *_: -99)
    assert mopd.get_mopd_reward("answer", "label", {"data_source": "T3", "reward_source": "dapo-math-17k"}) == 0.75
    assert calls == [("answer", "label")]


def test_existing_domain_routes_keep_their_scorer(monkeypatch):
    monkeypatch.setattr(mopd, "compute_score_dapo", lambda *_: {"score": 0.5})
    monkeypatch.setattr(mopd, "_compute_gsm8k_score", lambda *_: -0.5)
    assert mopd.get_mopd_reward("answer", "label", {"data_source": "dapo-math-17k"}) == 0.5
    assert mopd.get_mopd_reward("answer", "label", {"data_source": "T1"}) == -0.5
