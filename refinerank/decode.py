"""Calibrated-score adapter for the unchanged iter97 STG decoder."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Any

import numpy as np

from refinerank.coordinates import (
    annotation_frame_size,
    image_box_to_annotation,
)
from refinerank.geometry import denormalize_box_xyxy, normalize_box_xyxy
from refinerank.types import CandidateGroup

if TYPE_CHECKING:
    from refinerank.train_data import CachedTrainingGroup


@dataclass(frozen=True)
class ScoredCandidate:
    """Original-coordinate candidate plus a calibrated [0, 1] ranker score."""

    candidate_id: str
    timestamp: float
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("decoder candidate score must be finite")


def select_with_iter97_dp(
    *,
    dataset: str,
    timestamp_candidates: Mapping[float, Sequence[ScoredCandidate]],
    dp_lambda: float = 0.2,
    cholec_top_k: int = 20,
    require_calibrated: bool = True,
) -> dict[float, ScoredCandidate]:
    """Select top-1 normally and the unchanged top-20 shape DP for Cholec."""

    if dp_lambda != 0.2:
        raise ValueError("v1 decoder keeps the frozen iter97 dp_lambda=0.2")
    if not timestamp_candidates:
        raise ValueError("decoder requires at least one requested timestamp")
    ordered = sorted(timestamp_candidates.items())
    for timestamp, candidates in ordered:
        if not candidates:
            raise ValueError(f"timestamp {timestamp} has no candidates")
        if require_calibrated and any(
            candidate.score < 0.0 or candidate.score > 1.0 for candidate in candidates
        ):
            raise ValueError("decoder candidate score must be calibrated to [0, 1]")
    if dataset != "CholecTrack20":
        return {
            timestamp: _sorted_options(candidates)[0]
            for timestamp, candidates in ordered
        }
    options = [
        (timestamp, _sorted_options(candidates)[:cholec_top_k])
        for timestamp, candidates in ordered
    ]
    states: list[list[tuple[float, int | None]]] = [
        [(candidate.score, None) for candidate in options[0][1]]
    ]
    for step in range(1, len(options)):
        previous = options[step - 1][1]
        current = options[step][1]
        current_states: list[tuple[float, int | None]] = []
        for candidate in current:
            best_score = -1.0e30
            best_previous: int | None = None
            for previous_index, previous_candidate in enumerate(previous):
                score = (
                    states[step - 1][previous_index][0]
                    + candidate.score
                    - dp_lambda * _shape_transition(previous_candidate, candidate)
                )
                if score > best_score:
                    best_score = score
                    best_previous = previous_index
            current_states.append((best_score, best_previous))
        states.append(current_states)
    last_index = max(
        range(len(states[-1])),
        key=lambda index: (states[-1][index][0], -index),
    )
    selected: dict[float, ScoredCandidate] = {}
    for step in range(len(options) - 1, -1, -1):
        timestamp, candidates = options[step]
        selected[timestamp] = candidates[last_index]
        previous = states[step][last_index][1]
        if previous is None:
            break
        last_index = previous
    return dict(sorted(selected.items()))


def build_stg_prediction_rows(
    groups: Sequence[CandidateGroup],
    scores: Mapping[str, Sequence[float]],
    *,
    local_replay_ids: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Format one original-coordinate box per requested timestamp."""

    grouped: dict[tuple[int, str, str, str], list[CandidateGroup]] = {}
    for group in groups:
        key = (group.row_index, group.sample_id, group.original_id, group.dataset)
        grouped.setdefault(key, []).append(group)
    prediction_rows: list[dict[str, str]] = []
    audits: list[dict[str, Any]] = []
    for (row_index, sample_id, original_id, dataset), sequence in sorted(
        grouped.items()
    ):
        timestamp_candidates: dict[float, tuple[ScoredCandidate, ...]] = {}
        image_sizes: dict[float, tuple[int, int]] = {}
        for group in sequence:
            group_scores = tuple(float(value) for value in scores[group.group_key])
            if len(group_scores) != len(group.candidates):
                raise ValueError(f"score shape mismatch for {group.group_key}")
            if group.requested_timestamp in timestamp_candidates:
                raise ValueError("duplicate requested timestamp in one STG row")
            image_sizes[group.requested_timestamp] = group.frame_size
            timestamp_candidates[group.requested_timestamp] = tuple(
                ScoredCandidate(
                    candidate_id=candidate.candidate_id,
                    timestamp=group.requested_timestamp,
                    bbox_xyxy=candidate.bbox_xyxy,
                    score=score,
                    rank=candidate.rank,
                )
                for candidate, score in zip(
                    group.candidates,
                    group_scores,
                    strict=True,
                )
            )
        _validate_cholec_dp_frame_sizes(dataset, image_sizes)
        selected = select_with_iter97_dp(
            dataset=dataset,
            timestamp_candidates=timestamp_candidates,
        )
        if set(selected) != set(timestamp_candidates):
            raise RuntimeError("decoder did not return exactly one box per timestamp")
        prediction = format_stg_prediction(
            (
                timestamp,
                image_box_to_annotation(
                    candidate.bbox_xyxy,
                    dataset=dataset,
                    image_size=image_sizes[timestamp],
                ),
            )
            for timestamp, candidate in selected.items()
        )
        replay_id = (
            f"{original_id}__local_replay_{row_index:06d}"
            if local_replay_ids
            else sample_id
        )
        prediction_rows.append(
            {
                "id": replay_id,
                "original_id": original_id,
                "prediction": prediction,
                "qa_type": "stg",
                "sample_id": replay_id,
            }
        )
        audits.append(
            {
                "row_index": row_index,
                "original_id": original_id,
                "dataset": dataset,
                "requested_timestamp_count": len(timestamp_candidates),
                "selected_candidate_ids": {
                    _format_timestamp(timestamp): candidate.candidate_id
                    for timestamp, candidate in selected.items()
                },
                "selected_score_mean": fmean(
                    candidate.score for candidate in selected.values()
                ),
                "output_coordinate_space": "dataset_annotation_canvas",
                "annotation_frame_sizes": sorted(
                    {
                        annotation_frame_size(dataset, image_size)
                        for image_size in image_sizes.values()
                    }
                ),
            }
        )
    return prediction_rows, audits


def build_source_indexed_stg_prediction_rows(
    groups: Sequence["CachedTrainingGroup"],
    scores: Mapping[str, Sequence[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Format public ranker groups in decoded-image coordinates."""

    grouped: dict[tuple[int, str, str, str], list["CachedTrainingGroup"]] = {}
    for group in groups:
        key = (group.row_index, group.sample_id, group.original_id, group.dataset)
        grouped.setdefault(key, []).append(group)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for (source_index, sample_id, original_id, dataset), sequence in sorted(
        grouped.items()
    ):
        timestamp_candidates: dict[float, tuple[ScoredCandidate, ...]] = {}
        image_sizes: dict[float, tuple[int, int]] = {}
        for group in sequence:
            group_scores = tuple(float(value) for value in scores[group.group_key])
            if len(group_scores) != len(group.candidate_ids):
                raise ValueError(f"score shape mismatch for {group.group_key}")
            if group.requested_timestamp in timestamp_candidates:
                raise ValueError("duplicate requested timestamp in one STG row")
            image_sizes[group.requested_timestamp] = group.frame_size
            timestamp_candidates[group.requested_timestamp] = tuple(
                ScoredCandidate(
                    candidate_id=candidate_id,
                    timestamp=group.requested_timestamp,
                    bbox_xyxy=box,
                    score=score,
                    rank=rank,
                )
                for candidate_id, box, score, rank in zip(
                    group.candidate_ids,
                    group.boxes_xyxy,
                    group_scores,
                    group.ranks,
                    strict=True,
                )
            )
        _validate_cholec_dp_frame_sizes(dataset, image_sizes)
        selected = select_with_iter97_dp(
            dataset=dataset,
            timestamp_candidates=timestamp_candidates,
        )
        prediction = format_stg_prediction(
            (
                timestamp,
                denormalize_box_xyxy(
                    normalize_box_xyxy(
                        candidate.bbox_xyxy,
                        width=image_sizes[timestamp][0],
                        height=image_sizes[timestamp][1],
                    ),
                    width=image_sizes[timestamp][0],
                    height=image_sizes[timestamp][1],
                ),
            )
            for timestamp, candidate in selected.items()
        )
        rows.append(
            {
                "source_index": source_index,
                "id": sample_id,
                "qa_type": "stg",
                "prediction": prediction,
            }
        )
        audits.append(
            {
                "source_index": source_index,
                "original_id": original_id,
                "dataset": dataset,
                "requested_timestamp_count": len(timestamp_candidates),
                "selected_candidate_ids": {
                    _format_timestamp(timestamp): candidate.candidate_id
                    for timestamp, candidate in selected.items()
                },
                "output_coordinate_space": "decoded_image_canvas",
                "image_frame_sizes": sorted(set(image_sizes.values())),
                "image_frame_sizes_by_timestamp": {
                    _format_timestamp(timestamp): list(image_sizes[timestamp])
                    for timestamp in sorted(image_sizes)
                },
            }
        )
    if len({int(row["source_index"]) for row in rows}) != len(rows):
        raise RuntimeError("public STG rows have duplicate source_index values")
    return rows, audits


def format_stg_prediction(
    assignments: Iterable[tuple[float, tuple[float, float, float, float]]],
) -> str:
    """Preserve the existing MedVidBench STG prediction text schema."""

    return " ".join(
        f"{_format_timestamp(timestamp)} seconds: "
        f"[{', '.join(_format_coordinate(value) for value in box)}]"
        for timestamp, box in assignments
    )


def _sorted_options(candidates: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
    return sorted(
        candidates,
        key=lambda item: (item.score, -item.rank, item.candidate_id),
        reverse=True,
    )


def _shape_transition(left: ScoredCandidate, right: ScoredCandidate) -> float:
    lx1, ly1, lx2, ly2 = left.bbox_xyxy
    rx1, ry1, rx2, ry2 = right.bbox_xyxy
    left_center = ((lx1 + lx2) * 0.5, (ly1 + ly2) * 0.5)
    right_center = ((rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5)
    center_jump = math.dist(left_center, right_center)
    left_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    right_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    area_jump = abs(math.log(left_area / right_area))
    # Exact iter97 `transition_penalty_shape`: its base penalty contributes
    # 0.05 * area_jump and the shape extension contributes another 0.05.
    return (
        center_jump / 1500.0
        + 0.10 * area_jump
        + 0.002 * abs(float(left.rank) - float(right.rank))
    )


def _validate_cholec_dp_frame_sizes(
    dataset: str,
    image_sizes: Mapping[float, tuple[int, int]],
) -> None:
    """Fail fast when frozen pixel-space Cholec DP sees incompatible canvases."""

    if dataset.casefold() != "cholectrack20".casefold():
        return
    for image_size in image_sizes.values():
        annotation_frame_size(dataset, image_size)
    unique_sizes = set(image_sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(
            "CholecTrack20 iter97 pixel-space DP requires one decoded frame size "
            f"per source row, got {sorted(unique_sizes)}"
        )


def _format_timestamp(value: float) -> str:
    return str(float(value))


def _format_coordinate(value: float) -> str:
    return f"{round(float(value), 2):g}"


def summarize_scores(
    groups: Sequence[CachedTrainingGroup],
    scores: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Summarize selected IoU overall and by source dataset."""

    selected_by_row: dict[tuple[str, int], list[float]] = defaultdict(list)
    effective_counts: list[int] = []
    selected_indices: dict[str, int] = {}
    cholec_sequences: dict[
        tuple[str, int], list[CachedTrainingGroup]
    ] = defaultdict(list)
    for group in groups:
        group_scores = np.asarray(scores[group.group_key], dtype=np.float64)
        if group_scores.shape != group.ious.shape:
            raise ValueError(f"score shape mismatch for {group.group_key}")
        if group.dataset == "CholecTrack20":
            cholec_sequences[(group.dataset, group.row_index)].append(group)
        else:
            selected_indices[group.group_key] = max(
                range(len(group_scores)),
                key=lambda index: (
                    group_scores[index],
                    group.candidate_ids[index],
                ),
            )
        effective_counts.append(len(group_scores))
    for sequence in cholec_sequences.values():
        timestamp_candidates = {}
        by_timestamp_id: dict[
            tuple[float, str], tuple[CachedTrainingGroup, int]
        ] = {}
        for group in sequence:
            candidates = []
            for index, (candidate_id, box, rank, score) in enumerate(
                zip(
                    group.candidate_ids,
                    group.boxes_xyxy,
                    group.ranks,
                    scores[group.group_key],
                    strict=True,
                )
            ):
                candidates.append(
                    ScoredCandidate(
                        candidate_id=candidate_id,
                        timestamp=group.requested_timestamp,
                        bbox_xyxy=box,
                        score=float(score),
                        rank=rank,
                    )
                )
                by_timestamp_id[(group.requested_timestamp, candidate_id)] = (
                    group,
                    index,
                )
            timestamp_candidates[group.requested_timestamp] = tuple(candidates)
        selected = select_with_iter97_dp(
            dataset="CholecTrack20",
            timestamp_candidates=timestamp_candidates,
            require_calibrated=False,
        )
        for timestamp, candidate in selected.items():
            group, index = by_timestamp_id[(timestamp, candidate.candidate_id)]
            selected_indices[group.group_key] = index
    if len(selected_indices) != len(groups):
        raise RuntimeError("decoder did not select exactly one candidate per group")
    for group in groups:
        selected_by_row[(group.dataset, group.row_index)].append(
            float(group.ious[selected_indices[group.group_key]])
        )
    selected_by_dataset: dict[str, list[float]] = defaultdict(list)
    for (dataset, _row_index), values in selected_by_row.items():
        selected_by_dataset[dataset].append(fmean(values))
    per_dataset = {
        dataset: fmean(values)
        for dataset, values in sorted(selected_by_dataset.items())
    }
    return {
        "overall_dataset_mean": fmean(per_dataset.values()),
        "row_weighted_mean": fmean(
            value for values in selected_by_dataset.values() for value in values
        ),
        "per_dataset": per_dataset,
        "group_count": len(groups),
        "effective_candidate_count_mean": fmean(effective_counts),
        "effective_candidate_count_min": min(effective_counts),
        "effective_candidate_count_max": max(effective_counts),
    }
