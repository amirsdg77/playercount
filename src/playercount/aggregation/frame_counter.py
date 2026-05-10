"""Pure function that turns a per-frame track list into a :class:`FrameResult`.

This is the load-bearing counting algorithm:

1. For every track in this frame, record its (possibly ``None``) per-frame team
   guess in :class:`TrackState`.
2. Stabilize each track's team via a sliding-window mode vote.
3. Group by class id (player vs goalkeeper vs referee) and stabilized team,
   then count distinct track ids per group.

By counting *distinct active track ids* (not detections), we are robust to
duplicate detections of the same player in one frame. By stabilizing the team
label per track, we are robust to single-frame mis-classifications.

The function is deliberately pure so it is trivially unit-testable without any
ML stack.
"""

from __future__ import annotations

from playercount.aggregation.track_state import TrackState
from playercount.schemas import FrameResult, Track

# Class-id constants — duplicated from schemas.CLASS_NAMES to avoid the dict
# lookup in the hot path. Kept in sync by tests/unit/test_frame_counter.py.
_CLS_PLAYER = 0
_CLS_GOALKEEPER = 1
_CLS_REFEREE = 2
_CLS_BALL = 3

_TEAM_A = 0
_TEAM_B = 1


def build_frame_result(
    frame_idx: int,
    timestamp_s: float,
    tracks: list[Track],
    state: TrackState,
) -> FrameResult:
    """Aggregate one frame's tracks into a :class:`FrameResult`.

    ``state`` is mutated: each track's ``team_id`` (which the team classifier
    set, or ``None`` if it could not assign) is recorded as one vote, and the
    stabilized team label is what feeds the counts.

    Counts are over **distinct track ids**, so a single track that appears
    twice in the input list (defensive: should not happen, but cheap to
    guard) is still counted once.

    Referee count is over distinct referee track ids in this frame regardless
    of stabilization. Goalkeepers are counted *per stabilized team*.
    """
    seen_player_ids_per_team: dict[int, set[int]] = {_TEAM_A: set(), _TEAM_B: set()}
    seen_gk_ids_per_team: dict[int, set[int]] = {_TEAM_A: set(), _TEAM_B: set()}
    seen_referee_ids: set[int] = set()
    # Track which votable ids the tracker reported this frame, so we can prune
    # state for ids ByteTrack has dropped (otherwise TrackState leaks one deque
    # per dead track for the whole video).
    votable_this_frame: set[int] = set()

    for track in tracks:
        cls_id = track.detection.class_id
        tid = track.track_id

        if cls_id == _CLS_BALL:
            continue  # the ball does not contribute to player counts

        # Refs are filtered before they ever vote; their stabilized team would
        # always be None anyway, but skipping the vote keeps the state cleaner.
        if cls_id == _CLS_REFEREE:
            seen_referee_ids.add(tid)
            continue

        # Players and goalkeepers feed the team-vote stabilizer.
        state.vote(tid, track.team_id)
        votable_this_frame.add(tid)
        team = state.majority(tid)
        if team is None:
            # Not yet enough evidence — wait for more frames before counting.
            continue
        if team not in (_TEAM_A, _TEAM_B):
            # Defensive: an unexpected team id should never reach this code.
            continue

        if cls_id == _CLS_PLAYER:
            seen_player_ids_per_team[team].add(tid)
        elif cls_id == _CLS_GOALKEEPER:
            seen_gk_ids_per_team[team].add(tid)

    # Memory-leak fix: drop votes for ids ByteTrack stopped emitting. We
    # forget aggressively (after one missing frame) because the *tracker*
    # already keeps a lost-track buffer (default 30 frames) before truly
    # ending an id; if the tracker re-emits it later we'll just start the
    # vote window over, which the stabiliser handles correctly.
    stale = state.active_ids() - votable_this_frame
    if stale:
        state.forget_many(stale)

    return FrameResult(
        frame_index=frame_idx,
        timestamp_s=timestamp_s,
        team_a_count=len(seen_player_ids_per_team[_TEAM_A]),
        team_b_count=len(seen_player_ids_per_team[_TEAM_B]),
        referee_count=len(seen_referee_ids),
        goalkeeper_a_count=len(seen_gk_ids_per_team[_TEAM_A]),
        goalkeeper_b_count=len(seen_gk_ids_per_team[_TEAM_B]),
    )


__all__ = ["build_frame_result"]
