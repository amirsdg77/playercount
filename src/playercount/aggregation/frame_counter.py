"""Per-frame counting: tracks → :class:`FrameResult`."""

from __future__ import annotations

from playercount.aggregation.track_state import TrackState
from playercount.constants import (
    CLS_BALL,
    CLS_GOALKEEPER,
    CLS_PLAYER,
    CLS_REFEREE,
    TEAM_A,
    TEAM_B,
)
from playercount.schemas import FrameResult, Track


def build_frame_result(
    frame_idx: int,
    timestamp_s: float,
    tracks: list[Track],
    state: TrackState,
) -> FrameResult:
    """Aggregate one frame's tracks into a :class:`FrameResult`.

    For each player or goalkeeper track, record the per-frame team guess in
    ``state``, then read back the stabilised (sliding-window mode) team label.
    Counts are over distinct stabilised ``track_id`` values, so duplicate
    detections of the same player are counted once. Referees are tallied
    separately and never get a team. Track ids that the tracker stopped
    reporting are pruned from ``state`` to bound memory.
    """
    seen_player_ids_per_team: dict[int, set[int]] = {TEAM_A: set(), TEAM_B: set()}
    seen_gk_ids_per_team: dict[int, set[int]] = {TEAM_A: set(), TEAM_B: set()}
    seen_referee_ids: set[int] = set()
    votable_this_frame: set[int] = set()

    for track in tracks:
        cls_id = track.detection.class_id
        tid = track.track_id

        if cls_id == CLS_BALL:
            continue
        if cls_id == CLS_REFEREE:
            seen_referee_ids.add(tid)
            continue

        state.vote(tid, track.team_id)
        votable_this_frame.add(tid)
        team = state.majority(tid)
        if team is None or team not in (TEAM_A, TEAM_B):
            continue

        if cls_id == CLS_PLAYER:
            seen_player_ids_per_team[team].add(tid)
        elif cls_id == CLS_GOALKEEPER:
            seen_gk_ids_per_team[team].add(tid)

    # Prune dead tracks from state so memory stays bounded over long videos.
    stale = state.active_ids() - votable_this_frame
    if stale:
        state.forget_many(stale)

    return FrameResult(
        frame_index=frame_idx,
        timestamp_s=timestamp_s,
        team_a_count=len(seen_player_ids_per_team[TEAM_A]),
        team_b_count=len(seen_player_ids_per_team[TEAM_B]),
        referee_count=len(seen_referee_ids),
        goalkeeper_a_count=len(seen_gk_ids_per_team[TEAM_A]),
        goalkeeper_b_count=len(seen_gk_ids_per_team[TEAM_B]),
    )


__all__ = ["build_frame_result"]
