"""Dataset-aware STG annotation and image coordinate conversion."""

from __future__ import annotations

import math
from collections.abc import Sequence

from refinerank.geometry import (
    denormalize_box_xyxy,
    normalize_box_xyxy,
    validate_box_xyxy,
)

CHOLECTRACK20_DATASET = "CholecTrack20"
CHOLECTRACK20_ANNOTATION_FRAME_SIZE = (854, 480)
CHOLECTRACK20_ANNOTATION_BOUNDARY_TOLERANCE_PIXELS = 3.0
_CHOLECTRACK20_ASPECT_RATIO_RELATIVE_TOLERANCE = 0.01


def annotation_frame_size(
    dataset: str,
    image_size: Sequence[int],
) -> tuple[int, int]:
    """Return the dataset annotation canvas for one decoded image."""

    width, height = _validate_frame_size(image_size, name="image")
    if dataset.casefold() != CHOLECTRACK20_DATASET.casefold():
        return width, height
    annotation_width, annotation_height = CHOLECTRACK20_ANNOTATION_FRAME_SIZE
    image_ratio = width / height
    annotation_ratio = annotation_width / annotation_height
    relative_error = abs(image_ratio - annotation_ratio) / annotation_ratio
    if relative_error > _CHOLECTRACK20_ASPECT_RATIO_RELATIVE_TOLERANCE:
        raise ValueError(
            "CholecTrack20 image aspect ratio is incompatible with its "
            f"{annotation_width}x{annotation_height} annotation canvas: "
            f"{width}x{height}"
        )
    return CHOLECTRACK20_ANNOTATION_FRAME_SIZE


def normalize_annotation_box_xyxy(
    box: Sequence[float],
    *,
    dataset: str,
    image_size: Sequence[int],
) -> tuple[float, float, float, float]:
    """Normalize an annotation-space box without assuming image resolution."""

    annotation_width, annotation_height = annotation_frame_size(dataset, image_size)
    if dataset.casefold() != CHOLECTRACK20_DATASET.casefold():
        return normalize_box_xyxy(
            box,
            width=annotation_width,
            height=annotation_height,
        )
    x1, y1, x2, y2 = validate_box_xyxy(
        box,
        width=annotation_width,
        height=annotation_height,
        tolerance=CHOLECTRACK20_ANNOTATION_BOUNDARY_TOLERANCE_PIXELS,
    )
    x1 = min(max(x1, 0.0), float(annotation_width))
    y1 = min(max(y1, 0.0), float(annotation_height))
    x2 = min(max(x2, 0.0), float(annotation_width))
    y2 = min(max(y2, 0.0), float(annotation_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("annotation xyxy box has no area after boundary snap")
    return (
        x1 / annotation_width,
        y1 / annotation_height,
        x2 / annotation_width,
        y2 / annotation_height,
    )


def annotation_box_to_image(
    box: Sequence[float],
    *,
    dataset: str,
    image_size: Sequence[int],
) -> tuple[float, float, float, float]:
    """Convert an annotation-space box to the decoded image canvas."""

    width, height = _validate_frame_size(image_size, name="image")
    if dataset.casefold() != CHOLECTRACK20_DATASET.casefold():
        return validate_box_xyxy(box, width=width, height=height)
    normalized = normalize_annotation_box_xyxy(
        box,
        dataset=dataset,
        image_size=(width, height),
    )
    return denormalize_box_xyxy(normalized, width=width, height=height)


def image_box_to_annotation(
    box: Sequence[float],
    *,
    dataset: str,
    image_size: Sequence[int],
) -> tuple[float, float, float, float]:
    """Convert a decoded-image box to the dataset annotation canvas."""

    width, height = _validate_frame_size(image_size, name="image")
    if dataset.casefold() != CHOLECTRACK20_DATASET.casefold():
        return validate_box_xyxy(box, width=width, height=height)
    annotation_width, annotation_height = annotation_frame_size(
        dataset,
        (width, height),
    )
    normalized = normalize_box_xyxy(box, width=width, height=height)
    return denormalize_box_xyxy(
        normalized,
        width=annotation_width,
        height=annotation_height,
    )


def _validate_frame_size(
    frame_size: Sequence[int],
    *,
    name: str,
) -> tuple[int, int]:
    if len(frame_size) != 2:
        raise ValueError(f"{name} frame size must contain width and height")
    if not all(math.isfinite(float(value)) for value in frame_size):
        raise ValueError(f"{name} frame size must be finite")
    width, height = (int(value) for value in frame_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} frame size must be positive")
    return width, height
