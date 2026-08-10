from __future__ import annotations

from pathlib import Path

import numpy as np

from refinerank.decode import (
    ScoredCandidate,
    build_stg_prediction_rows,
    select_with_iter97_dp,
    summarize_scores,
)
from refinerank.train_data import CachedTrainingGroup
from refinerank.types import CandidateGroup, FrozenCandidate


def _candidate(
    candidate_id: str,
    timestamp: float,
    box: tuple[float, float, float, float],
    score: float,
    rank: int,
) -> ScoredCandidate:
    return ScoredCandidate(candidate_id, timestamp, box, score, rank)


def test_iter97_shape_dp_golden_sequence() -> None:
    pools = {
        1.0: (
            _candidate("a", 1.0, (0.0, 0.0, 100.0, 100.0), 0.80, 1),
            _candidate("b", 1.0, (800.0, 0.0, 900.0, 100.0), 0.85, 2),
        ),
        2.0: (
            _candidate("c", 2.0, (0.0, 0.0, 100.0, 100.0), 0.80, 1),
            _candidate("d", 2.0, (800.0, 0.0, 900.0, 100.0), 0.70, 2),
        ),
        3.0: (
            _candidate("e", 3.0, (0.0, 0.0, 100.0, 100.0), 0.80, 1),
            _candidate("f", 3.0, (800.0, 0.0, 900.0, 100.0), 0.85, 2),
        ),
    }

    selected = select_with_iter97_dp(
        dataset="CholecTrack20",
        timestamp_candidates=pools,
    )

    assert [item.candidate_id for item in selected.values()] == ["a", "c", "e"]
    assert len(selected) == len(pools)


def test_non_cholec_keeps_per_timestamp_top1() -> None:
    pools = {
        1.0: (
            _candidate("low", 1.0, (0.0, 0.0, 1.0, 1.0), 0.1, 1),
            _candidate("high", 1.0, (0.0, 0.0, 1.0, 1.0), 0.9, 2),
        )
    }

    selected = select_with_iter97_dp(dataset="CoPESD", timestamp_candidates=pools)

    assert selected[1.0].candidate_id == "high"


def test_score_summary_scopes_reused_candidate_id_by_timestamp() -> None:
    def group(timestamp: float, ious: tuple[float, float]) -> CachedTrainingGroup:
        count = len(ious)
        return CachedTrainingGroup(
            group_key=f"sequence@{timestamp:.3f}",
            sample_id="sample",
            row_index=0,
            dataset="CholecTrack20",
            original_id="sequence",
            video_id="video-1",
            requested_timestamp=timestamp,
            candidate_ids=("shared-proposal", f"other-{timestamp:.3f}"),
            boxes_xyxy=((0.0, 0.0, 10.0, 10.0), (50.0, 50.0, 60.0, 60.0)),
            ranks=(0, 1),
            q_last=np.zeros(3584, dtype=np.float16),
            roi_final=np.zeros((count, 3584), dtype=np.float16),
            roi_l31=np.zeros((count, 1280), dtype=np.float16),
            roi_l23=np.zeros((count, 1280), dtype=np.float16),
            structure=np.zeros((count, 12), dtype=np.float32),
            missing=np.zeros((count, 12), dtype=np.float32),
            ious=np.asarray(ious, dtype=np.float32),
            native_scores=np.asarray((0.9, 0.1), dtype=np.float32),
        )

    groups = (group(1.0, (0.8, 0.0)), group(2.0, (0.6, 0.0)))
    scores = {item.group_key: (0.9, 0.1) for item in groups}

    metrics = summarize_scores(groups, scores)

    assert np.isclose(metrics["per_dataset"]["CholecTrack20"], 0.7)
    assert metrics["group_count"] == 2


def test_score_summary_weights_source_rows_with_duplicate_original_ids() -> None:
    def group(
        row_index: int, timestamp: float, selected_iou: float
    ) -> CachedTrainingGroup:
        return CachedTrainingGroup(
            group_key=f"row-{row_index}@{timestamp:.3f}",
            sample_id=f"sample-{row_index}",
            row_index=row_index,
            dataset="CoPESD",
            original_id="reused-original-id",
            video_id="video-1",
            requested_timestamp=timestamp,
            candidate_ids=("candidate",),
            boxes_xyxy=((0.0, 0.0, 10.0, 10.0),),
            ranks=(0,),
            q_last=np.zeros(1, dtype=np.float16),
            roi_final=np.zeros((1, 1), dtype=np.float16),
            roi_l31=None,
            roi_l23=None,
            structure=np.zeros((1, 12), dtype=np.float32),
            missing=np.zeros((1, 12), dtype=np.float32),
            ious=np.asarray((selected_iou,), dtype=np.float32),
            native_scores=np.asarray((1.0,), dtype=np.float32),
        )

    groups = (
        group(0, 1.0, 1.0),
        group(1, 1.0, 0.0),
        group(1, 2.0, 0.0),
        group(1, 3.0, 0.0),
    )
    scores = {item.group_key: (1.0,) for item in groups}

    metrics = summarize_scores(groups, scores)

    assert metrics["per_dataset"]["CoPESD"] == 0.5
    assert metrics["row_weighted_mean"] == 0.5
    assert metrics["group_count"] == 4


def test_prediction_rows_preserve_original_coordinates_and_schema() -> None:
    candidates = (
        FrozenCandidate(
            candidate_id="candidate-1",
            candidate_time=1.0,
            bbox_xyxy=(1.234, 2.0, 30.0, 40.567),
            native_score=0.5,
            rank=1,
            source="cached_dino",
            structure=(0.0,) * 12,
            missing=(0.0,) * 12,
        ),
    )
    group = CandidateGroup(
        group_key="group-1",
        sample_id="sample-1",
        original_id="original-1",
        video_id="video-1",
        row_index=7,
        dataset="CoPESD",
        requested_timestamp=1.0,
        clean_question="Where is the tool?",
        frame_path=Path("unused.jpg"),
        frame_size=(100, 100),
        candidates=candidates,
    )

    rows, audits = build_stg_prediction_rows(
        (group,),
        {"group-1": (0.9,)},
        local_replay_ids=True,
    )

    assert rows == [
        {
            "id": "original-1__local_replay_000007",
            "original_id": "original-1",
            "prediction": "1.0 seconds: [1.23, 2, 30, 40.57]",
            "qa_type": "stg",
            "sample_id": "original-1__local_replay_000007",
        }
    ]
    assert audits[0]["requested_timestamp_count"] == 1


def test_local_replay_cholec_output_uses_annotation_canvas() -> None:
    candidate = FrozenCandidate(
        candidate_id="candidate-1",
        candidate_time=1.0,
        bbox_xyxy=(463.14, 20.25, 681.22, 186.75),
        native_score=0.5,
        rank=1,
        source="cached_dino",
        structure=(0.0,) * 12,
        missing=(0.0,) * 12,
    )
    group = CandidateGroup(
        group_key="group-1",
        sample_id="sample-1",
        original_id="original-1",
        video_id="video-1",
        row_index=7,
        dataset="CholecTrack20",
        requested_timestamp=1.0,
        clean_question="Where is the tool?",
        frame_path=Path("unused.jpg"),
        frame_size=(1920, 1080),
        candidates=(candidate,),
    )

    rows, audits = build_stg_prediction_rows(
        (group,),
        {"group-1": (0.9,)},
        local_replay_ids=True,
    )

    assert rows[0]["prediction"] == "1.0 seconds: [206, 9, 303, 83]"
    assert audits[0]["output_coordinate_space"] == "dataset_annotation_canvas"
    assert audits[0]["annotation_frame_sizes"] == [(854, 480)]
