from __future__ import annotations

import pytest
import torch

from refinerank.coordinates import (
    annotation_box_to_image,
    annotation_frame_size,
    image_box_to_annotation,
    normalize_annotation_box_xyxy,
)
from refinerank.geometry import (
    area_weighted_roi_pool,
    denormalize_box_xyxy,
    inverse_window_group_order,
    normalize_box_xyxy,
    patch_tokens_to_grid,
)


def test_window_inverse_restores_synthetic_patch_grid() -> None:
    expected_grid = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
    grouped = (
        expected_grid.reshape(1, 2, 2, 2, 2, 1).permute(0, 1, 3, 2, 4, 5).reshape(16, 1)
    )
    window_index = torch.tensor([2, 0, 3, 1])
    windowed = grouped.reshape(4, 4, 1)[window_index].reshape(16, 1)

    restored = inverse_window_group_order(
        windowed,
        window_index,
        merge_size=2,
    )
    actual_grid = patch_tokens_to_grid(restored, (1, 4, 4), merge_size=2)

    torch.testing.assert_close(actual_grid, expected_grid)


def test_area_weighted_roi_pool_uses_fractional_cell_overlap() -> None:
    grid = torch.tensor([[[[1.0], [3.0]], [[5.0], [7.0]]]])
    boxes = torch.tensor([[0.25, 0.0, 0.75, 0.5], [0.0, 0.0, 1.0, 1.0]])

    pooled = area_weighted_roi_pool(grid, boxes)

    torch.testing.assert_close(pooled[:, 0], torch.tensor([2.0, 4.0]))


def test_coordinate_round_trip_and_invalid_box_fail_fast() -> None:
    source = (12.5, 4.0, 80.0, 42.5)
    normalized = normalize_box_xyxy(source, width=100, height=50)

    assert denormalize_box_xyxy(normalized, width=100, height=50) == pytest.approx(
        source
    )
    with pytest.raises(ValueError, match="non-positive area"):
        normalize_box_xyxy((10, 10, 10, 20), width=100, height=50)
    with pytest.raises(ValueError, match="exceeds source image"):
        normalize_box_xyxy((10, 10, 103, 20), width=100, height=50)


def test_subpixel_detector_boundary_drift_snaps_only_for_pooling() -> None:
    source = (-0.25, 1.0, 100.31, 49.0)

    normalized = normalize_box_xyxy(source, width=100, height=50)

    assert normalized == pytest.approx((0.0, 0.02, 1.0, 0.98))
    with pytest.raises(ValueError, match="exceeds source image"):
        normalize_box_xyxy((0.0, 0.0, 102.51, 10.0), width=100, height=50)


def test_cholec_annotation_image_round_trip_is_resolution_invariant() -> None:
    annotation_box = (206.0, 9.0, 303.0, 83.0)

    image_box = annotation_box_to_image(
        annotation_box,
        dataset="CholecTrack20",
        image_size=(1920, 1080),
    )
    restored = image_box_to_annotation(
        image_box,
        dataset="CholecTrack20",
        image_size=(1920, 1080),
    )

    assert image_box == pytest.approx((463.14, 20.25, 681.22, 186.75), abs=0.01)
    assert restored == pytest.approx(annotation_box)
    assert normalize_annotation_box_xyxy(
        annotation_box,
        dataset="CholecTrack20",
        image_size=(1920, 1080),
    ) == pytest.approx(
        normalize_annotation_box_xyxy(
            annotation_box,
            dataset="CholecTrack20",
            image_size=(854, 480),
        )
    )


def test_non_cholec_annotation_coordinates_are_identity_mapped() -> None:
    for dataset, image_size in (
        ("CoPESD", (1306, 1009)),
        ("EgoSurgery", (1920, 1080)),
    ):
        box = (10.0, 20.0, image_size[0] - 10.0, image_size[1] - 20.0)
        assert annotation_frame_size(dataset, image_size) == image_size
        assert annotation_box_to_image(
            box,
            dataset=dataset,
            image_size=image_size,
        ) == pytest.approx(box)
        assert image_box_to_annotation(
            box,
            dataset=dataset,
            image_size=image_size,
        ) == pytest.approx(box)
        boundary_box = (-0.25, 1.0, image_size[0] + 0.25, image_size[1] - 1.0)
        assert image_box_to_annotation(
            boundary_box,
            dataset=dataset,
            image_size=image_size,
        ) == pytest.approx(boundary_box)


def test_cholec_coordinate_protocol_rejects_incompatible_aspect_ratio() -> None:
    with pytest.raises(ValueError, match="aspect ratio"):
        annotation_frame_size("CholecTrack20", (1000, 1000))
