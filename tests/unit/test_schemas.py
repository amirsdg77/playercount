"""Validate Pydantic v2 schemas: validators, frozen-ness, naming invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from playercount.schemas import (
    CLASS_NAMES,
    AnalyzeRequest,
    BBox,
    Detection,
    FrameResult,
    Track,
)

# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------


def test_bbox_happy_path():
    bb = BBox(x1=1, y1=2, x2=10, y2=20)
    assert bb.width == 9
    assert bb.height == 18
    assert bb.area == 9 * 18
    assert bb.as_xyxy() == (1, 2, 10, 20)


@pytest.mark.parametrize("x1,x2", [(10, 10), (10, 5)])
def test_bbox_x2_must_exceed_x1(x1, x2):
    with pytest.raises(ValidationError):
        BBox(x1=x1, y1=0, x2=x2, y2=10)


@pytest.mark.parametrize("y1,y2", [(10, 10), (10, 5)])
def test_bbox_y2_must_exceed_y1(y1, y2):
    with pytest.raises(ValidationError):
        BBox(x1=0, y1=y1, x2=10, y2=y2)


def test_bbox_is_frozen():
    bb = BBox(x1=0, y1=0, x2=1, y2=1)
    with pytest.raises(ValidationError):
        bb.x1 = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detection_class_name_must_match_id():
    with pytest.raises(ValidationError):
        Detection(
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            score=0.5,
            class_id=0,
            class_name="goalkeeper",  # mismatch
        )


@pytest.mark.parametrize("cid", [0, 1, 2, 3])
def test_detection_known_class_ids(cid):
    det = Detection(
        bbox=BBox(x1=0, y1=0, x2=1, y2=1),
        score=0.7,
        class_id=cid,
        class_name=CLASS_NAMES[cid],
    )
    assert det.class_id == cid


def test_detection_score_bounds():
    with pytest.raises(ValidationError):
        Detection(
            bbox=BBox(x1=0, y1=0, x2=1, y2=1),
            score=1.5,
            class_id=0,
            class_name="player",
        )


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------


def test_track_team_id_optional():
    det = Detection(
        bbox=BBox(x1=0, y1=0, x2=1, y2=1),
        score=0.9,
        class_id=0,
        class_name="player",
    )
    t = Track(track_id=7, detection=det)
    assert t.team_id is None
    t.team_id = 1  # mutable on purpose — pipeline fills this in
    assert t.team_id == 1


def test_track_team_id_must_be_zero_or_one():
    det = Detection(
        bbox=BBox(x1=0, y1=0, x2=1, y2=1),
        score=0.9,
        class_id=0,
        class_name="player",
    )
    with pytest.raises(ValidationError):
        Track(track_id=1, detection=det, team_id=2)


# ---------------------------------------------------------------------------
# FrameResult
# ---------------------------------------------------------------------------


def test_frame_result_is_frozen():
    fr = FrameResult(
        frame_index=0,
        timestamp_s=0.0,
        team_a_count=3,
        team_b_count=4,
        referee_count=1,
        goalkeeper_a_count=1,
        goalkeeper_b_count=1,
    )
    with pytest.raises(ValidationError):
        fr.team_a_count = 99  # type: ignore[misc]


def test_frame_result_rejects_negative_counts():
    with pytest.raises(ValidationError):
        FrameResult(
            frame_index=0,
            timestamp_s=0.0,
            team_a_count=-1,
            team_b_count=0,
            referee_count=0,
            goalkeeper_a_count=0,
            goalkeeper_b_count=0,
        )


# ---------------------------------------------------------------------------
# AnalyzeRequest
# ---------------------------------------------------------------------------


def test_analyze_request_default_summary():
    r = AnalyzeRequest()
    assert r.return_mode == "summary"
    assert r.sampling_stride == 1


def test_analyze_request_rejects_bad_scheme():
    with pytest.raises(ValidationError):
        AnalyzeRequest(video_uri="ftp://example.com/x.mp4")


def test_analyze_request_accepts_supported_schemes():
    # gs:// was advertised but never implemented; dropped from the schema.
    for uri in [
        "https://example.com/x.mp4",
        "http://example.com/x.mp4",
        "file:///tmp/x.mp4",
    ]:
        r = AnalyzeRequest(video_uri=uri)
        assert r.video_uri is not None


def test_analyze_request_rejects_gs_scheme_until_implemented():
    """We dropped gs:// support; readd when the GCS fetcher lands in v3."""
    with pytest.raises(ValidationError):
        AnalyzeRequest(video_uri="gs://bucket/x.mp4")


def test_analyze_request_stride_must_be_positive():
    with pytest.raises(ValidationError):
        AnalyzeRequest(sampling_stride=0)


def test_analyze_request_forbids_extra_keys():
    with pytest.raises(ValidationError):
        AnalyzeRequest.model_validate({"unknown_field": 1})
