"""Sliding-window mode-vote bookkeeping."""

from __future__ import annotations

import pytest

from playercount.aggregation import TrackState


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        TrackState(window=0)


def test_majority_returns_none_for_unknown_id():
    s = TrackState()
    assert s.majority(42) is None


def test_simple_majority():
    s = TrackState(window=5)
    for v in [0, 0, 1, 0, 1]:
        s.vote(7, v)
    assert s.majority(7) == 0


def test_window_bounds_history():
    s = TrackState(window=3)
    # First 3 votes: all 0 → majority 0.
    s.vote_many([(1, 0), (1, 0), (1, 0)])
    assert s.majority(1) == 0
    # Push three 1s — the original 0s slide out.
    s.vote_many([(1, 1), (1, 1), (1, 1)])
    assert s.majority(1) == 1


def test_none_votes_are_excluded_from_mode_but_still_consume_window():
    s = TrackState(window=4)
    # Two real votes for team 1, then two None votes that push the team-1
    # votes to the edge of the window. The mode is still 1 because Nones
    # do not contribute to counts.
    s.vote_many([(2, 1), (2, 1), (2, None), (2, None)])
    assert s.majority(2) == 1
    # One more None push: now the only non-None votes are gone.
    s.vote(2, None)
    s.vote(2, None)
    assert s.majority(2) is None


def test_tie_break_by_latest_vote():
    s = TrackState(window=4)
    # Tied 2-2 — the most recent vote (1) wins.
    for v in [0, 0, 1, 1]:
        s.vote(3, v)
    assert s.majority(3) == 1


def test_forget_clears_state():
    s = TrackState()
    s.vote(9, 0)
    s.vote(9, 1)
    assert 9 in s
    s.forget(9)
    assert 9 not in s
    assert s.majority(9) is None


def test_forget_many_is_idempotent():
    s = TrackState()
    s.vote(1, 0)
    s.forget_many([1, 99, 1])  # 99 never existed, second 1 is no-op
    assert 1 not in s


def test_active_ids_returns_known_set():
    s = TrackState()
    s.vote_many([(1, 0), (2, 1), (3, None)])
    assert s.active_ids() == {1, 2, 3}


def test_len_and_contains():
    s = TrackState()
    assert len(s) == 0
    s.vote(1, 0)
    assert len(s) == 1
    assert 1 in s
    assert 2 not in s
