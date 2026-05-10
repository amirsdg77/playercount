"""Frame counting end-to-end (pure logic, no ML)."""

from __future__ import annotations

from playercount.aggregation import TrackState, build_frame_result
from playercount.schemas import BBox, Detection, Track


def _track(track_id: int, *, cls: int, team_id: int | None) -> Track:
    names = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    return Track(
        track_id=track_id,
        detection=Detection(
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            score=0.9,
            class_id=cls,  # type: ignore[arg-type]
            class_name=names[cls],
        ),
        team_id=team_id,
    )


def test_counts_distinct_tracks_per_team():
    state = TrackState(window=3)
    tracks = [
        _track(1, cls=0, team_id=0),
        _track(2, cls=0, team_id=0),
        _track(3, cls=0, team_id=1),
        _track(4, cls=1, team_id=1),  # goalkeeper team B
        _track(5, cls=2, team_id=None),  # referee
        _track(6, cls=3, team_id=None),  # ball — ignored
    ]
    # Vote three times to settle the majority for each track.
    fr = build_frame_result(0, 0.0, tracks, state)
    fr = build_frame_result(1, 0.1, tracks, state)
    fr = build_frame_result(2, 0.2, tracks, state)

    assert fr.team_a_count == 2  # tracks 1, 2
    assert fr.team_b_count == 1  # track 3
    assert fr.goalkeeper_a_count == 0
    assert fr.goalkeeper_b_count == 1  # track 4
    assert fr.referee_count == 1  # track 5
    # Ball never appears in any count.


def test_duplicate_track_ids_in_one_frame_count_once():
    """Defensive: if the tracker emits the same id twice in one frame, count once."""
    state = TrackState(window=2)
    tracks = [
        _track(1, cls=0, team_id=0),
        _track(1, cls=0, team_id=0),
    ]
    fr = build_frame_result(0, 0.0, tracks, state)
    fr = build_frame_result(1, 0.1, tracks, state)
    assert fr.team_a_count == 1


def test_no_count_until_majority_settles():
    """Tracks with empty / all-None history are not counted."""
    state = TrackState(window=4)
    tracks = [_track(1, cls=0, team_id=None)]  # never assigned yet
    fr = build_frame_result(0, 0.0, tracks, state)
    assert fr.team_a_count == 0
    assert fr.team_b_count == 0


def test_referee_never_assigned_to_a_team():
    state = TrackState(window=2)
    # Even if (somehow) team_id were set on a referee, the counter must skip
    # it and only put it in referee_count.
    tracks = [_track(7, cls=2, team_id=0)]
    fr = build_frame_result(0, 0.0, tracks, state)
    fr = build_frame_result(1, 0.1, tracks, state)
    assert fr.referee_count == 1
    assert fr.team_a_count == 0
    assert fr.team_b_count == 0


def test_timestamp_and_frame_index_passed_through():
    state = TrackState()
    fr = build_frame_result(123, 12.3, [], state)
    assert fr.frame_index == 123
    assert fr.timestamp_s == 12.3


def test_stabilization_kills_single_frame_flicker():
    """A track that has been team A for ten frames should not flip on a single
    frame where the classifier mis-fires team B."""
    state = TrackState(window=11)
    a = _track(1, cls=0, team_id=0)
    b = _track(1, cls=0, team_id=1)
    # 10 frames of team A
    for f in range(10):
        build_frame_result(f, float(f), [a], state)
    # 1 frame of team B (the flicker)
    fr = build_frame_result(10, 10.0, [b], state)
    # Stabilizer keeps team A on top.
    assert fr.team_a_count == 1
    assert fr.team_b_count == 0
