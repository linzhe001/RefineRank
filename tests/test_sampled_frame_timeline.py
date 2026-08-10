from __future__ import annotations

from pathlib import Path

import pytest

from refinerank.timeline import SampledFrameTimeline


def _paths(
    tmp_path: Path, frame_ids: list[int], suffix: str = ".png"
) -> tuple[Path, ...]:
    return tuple(tmp_path / f"{frame_id:06d}{suffix}" for frame_id in frame_ids)


def _timeline(
    tmp_path: Path,
    frame_ids: list[int],
    *,
    sampling_fps: float,
    start_frame: int | None = None,
    suffix: str = ".png",
) -> SampledFrameTimeline:
    return SampledFrameTimeline(
        frame_paths=_paths(tmp_path, frame_ids, suffix),
        sampled_video_frames=tuple(frame_ids),
        sampling_fps=sampling_fps,
        start_frame=frame_ids[0] if start_frame is None else start_frame,
        require_paths_exist=False,
    )


def test_cholec_missing_frames_resolve_by_source_identity(tmp_path: Path) -> None:
    timeline = _timeline(
        tmp_path,
        [34726, 34751, 34776, 37076, 37101, 37126, 37151, 37201],
        sampling_fps=1.0,
    )

    at_95 = timeline.resolve(95.0)
    at_99 = timeline.resolve(99.0)

    assert timeline.source_frame_cadence == 25
    assert at_95.source_frame_id == 37101
    assert at_95.frame_path.name == "037101.png"
    assert at_99.source_frame_id == 37201
    assert at_99.frame_path.name == "037201.png"
    assert at_95.exact and at_99.exact


@pytest.mark.parametrize(
    ("dataset", "sampling_fps", "frame_ids", "timestamp", "expected"),
    [
        ("CoPESD-0.5", 0.5, [1, 3, 5], 4.0, 5),
        ("CoPESD-1.0", 1.0, [101, 102, 103], 2.0, 103),
        ("EgoSurgery-0.5", 0.5, [501, 503, 505], 2.0, 503),
    ],
)
def test_dataset_sampling_rates_resolve_exactly(
    tmp_path: Path,
    dataset: str,
    sampling_fps: float,
    frame_ids: list[int],
    timestamp: float,
    expected: int,
) -> None:
    del dataset
    timeline = _timeline(
        tmp_path,
        frame_ids,
        sampling_fps=sampling_fps,
        suffix=".jpg",
    )

    assert timeline.resolve(timestamp).source_frame_id == expected
    assert timeline.resolve(timestamp).exact


def test_half_period_nearest_is_allowed_and_tie_is_earlier(tmp_path: Path) -> None:
    timeline = _timeline(tmp_path, [0, 2], sampling_fps=0.5)

    resolved = timeline.resolve(1.0)

    assert resolved.source_frame_id == 0
    assert resolved.alignment_error_seconds == 1.0
    assert not resolved.exact


def test_more_than_half_period_is_rejected(tmp_path: Path) -> None:
    timeline = _timeline(tmp_path, [0, 2], sampling_fps=0.5)

    with pytest.raises(ValueError, match="exceeding half-period"):
        timeline.resolve(3.1)


def test_single_non_nominal_delta_cannot_define_source_cadence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cadence is underdetermined"):
        _timeline(tmp_path, [0, 50], sampling_fps=1.0)


def test_source_cadence_uses_base_interval_across_missing_frames(
    tmp_path: Path,
) -> None:
    timeline = _timeline(tmp_path, [0, 25, 75, 125], sampling_fps=1.0)

    assert timeline.source_frame_cadence == 25
    assert timeline.resolve(3.0).source_frame_id == 75


def test_source_cadence_preserves_irregular_copesd_gap(tmp_path: Path) -> None:
    timeline = _timeline(tmp_path, [1295, 1297, 1299, 1302], sampling_fps=0.5)

    assert timeline.source_frame_cadence == 2
    resolved = timeline.resolve(7.0)
    assert resolved.source_frame_id == 1302
    assert resolved.exact


def test_low_frequency_endpoint_gap_does_not_reduce_copesd_cadence(
    tmp_path: Path,
) -> None:
    frame_ids = list(range(757, 779, 2)) + [778]
    timeline = _timeline(tmp_path, frame_ids, sampling_fps=0.5)

    assert timeline.source_frame_cadence == 2
    assert timeline.resolve(20.0).source_frame_id == 777


def test_non_monotonic_frame_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _timeline(tmp_path, [1, 3, 2], sampling_fps=0.5)


def test_path_and_frame_lengths_must_match(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lengths do not match"):
        SampledFrameTimeline(
            frame_paths=_paths(tmp_path, [1]),
            sampled_video_frames=(1, 2),
            sampling_fps=1.0,
            start_frame=1,
            require_paths_exist=False,
        )


def test_path_identity_must_match_sampled_frame_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity mismatch"):
        SampledFrameTimeline(
            frame_paths=_paths(tmp_path, [2]),
            sampled_video_frames=(1,),
            sampling_fps=1.0,
            start_frame=1,
            require_paths_exist=False,
        )


def test_egosurgery_prefixed_filename_uses_numeric_suffix(tmp_path: Path) -> None:
    timeline = SampledFrameTimeline(
        frame_paths=(tmp_path / "13_1_0337.jpg", tmp_path / "13_1_0339.jpg"),
        sampled_video_frames=(337, 339),
        sampling_fps=0.5,
        start_frame=337,
        require_paths_exist=False,
    )

    assert timeline.resolve(2.0).source_frame_id == 339


def test_start_frame_must_match_first_sampled_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_frame must equal"):
        _timeline(tmp_path, [2, 4], sampling_fps=0.5, start_frame=1)
