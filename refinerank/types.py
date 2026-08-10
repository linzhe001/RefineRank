"""Typed boundaries for frozen STG candidate ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from refinerank.geometry import BOUNDARY_SNAP_TOLERANCE_PIXELS
from refinerank.timeline import source_frame_id_from_path

STRUCTURAL_FEATURE_NAMES = (
    "dino_box_score",
    "temporal_selector_score",
    "coverage_ratio",
    "smoothness",
    "time_delta_norm",
    "is_exact_timestamp",
    "x1_norm",
    "y1_norm",
    "x2_norm",
    "y2_norm",
    "log_area",
    "log_aspect",
)


@dataclass(frozen=True)
class FrozenCandidate:
    """One GT-free proposal at a requested timestamp."""

    candidate_id: str
    candidate_time: float
    bbox_xyxy: tuple[float, float, float, float]
    native_score: float
    rank: int
    source: str
    structure: tuple[float, ...]
    missing: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = len(STRUCTURAL_FEATURE_NAMES)
        if len(self.structure) != expected or len(self.missing) != expected:
            raise ValueError(
                f"candidate structural tensors must have {expected} values and masks"
            )
        if not all(math.isfinite(value) for value in self.structure):
            raise ValueError("candidate structural features must be finite")
        if any(value not in {0.0, 1.0} for value in self.missing):
            raise ValueError("candidate missing mask must contain only 0/1 values")
        if not math.isfinite(self.native_score):
            raise ValueError("candidate native score must be finite")


@dataclass(frozen=True)
class CandidateGroup:
    """GT-free candidate set for one sample and requested timestamp."""

    group_key: str
    sample_id: str
    original_id: str
    row_index: int
    dataset: str
    requested_timestamp: float
    clean_question: str
    frame_path: Path
    frame_size: tuple[int, int]
    candidates: tuple[FrozenCandidate, ...]
    video_id: str = ""
    source_frame_id: int | None = None
    resolved_frame_timestamp: float | None = None
    alignment_error_seconds: float | None = None
    frame_alignment_exact: bool | None = None
    timeline_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.group_key or not self.original_id or not self.dataset:
            raise ValueError("candidate group identifiers must be non-empty")
        if not self.candidates:
            raise ValueError(f"candidate group {self.group_key} is empty")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"candidate group {self.group_key} has duplicate ids")
        width, height = self.frame_size
        if width <= 0 or height <= 0:
            raise ValueError("candidate group frame size must be positive")
        requested_timestamp = _finite_float(
            self.requested_timestamp,
            "candidate group requested timestamp",
        )
        provenance = (
            self.source_frame_id,
            self.resolved_frame_timestamp,
            self.alignment_error_seconds,
            self.frame_alignment_exact,
            self.timeline_digest,
        )
        if any(value is not None for value in provenance):
            if any(value is None for value in provenance):
                raise ValueError("candidate group frame provenance must be complete")
            source_frame_id = self.source_frame_id
            if isinstance(source_frame_id, bool) or not isinstance(
                source_frame_id, int
            ):
                raise TypeError("candidate group source frame ID must be an integer")
            resolved_timestamp = _finite_float(
                self.resolved_frame_timestamp,
                "candidate group resolved frame timestamp",
            )
            alignment_error = _finite_float(
                self.alignment_error_seconds,
                "candidate group alignment error",
            )
            if alignment_error < 0.0:
                raise ValueError("candidate group alignment error must be non-negative")
            if not isinstance(self.frame_alignment_exact, bool):
                raise TypeError(
                    "candidate group exact-alignment flag must be a boolean"
                )
            if source_frame_id_from_path(self.frame_path) != source_frame_id:
                raise ValueError("candidate group frame path does not match source ID")
            expected_error = abs(resolved_timestamp - requested_timestamp)
            if abs(expected_error - alignment_error) > 1.0e-9:
                raise ValueError("candidate group alignment error is inconsistent")
            if self.frame_alignment_exact != (expected_error <= 1.0e-9):
                raise ValueError("candidate group exact-alignment flag is inconsistent")
            if not self.timeline_digest:
                raise ValueError("candidate group timeline digest must be non-empty")
        for candidate in self.candidates:
            x1, y1, x2, y2 = candidate.bbox_xyxy
            if not all(math.isfinite(value) for value in candidate.bbox_xyxy):
                raise ValueError("candidate box must be finite")
            if (
                x1 < -BOUNDARY_SNAP_TOLERANCE_PIXELS
                or y1 < -BOUNDARY_SNAP_TOLERANCE_PIXELS
                or x2 <= x1
                or y2 <= y1
            ):
                raise ValueError("candidate box must be valid source-space xyxy")
            if (
                x2 > width + BOUNDARY_SNAP_TOLERANCE_PIXELS
                or y2 > height + BOUNDARY_SNAP_TOLERANCE_PIXELS
            ):
                raise ValueError("candidate box exceeds the source frame")


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


@dataclass(frozen=True)
class GroupFeatures:
    """Frozen features for a single variable-length candidate group."""

    group_key: str
    candidate_ids: tuple[str, ...]
    q_last: np.ndarray
    roi_l23: np.ndarray | None
    roi_l31: np.ndarray | None
    roi_final: np.ndarray
    structure: np.ndarray
    missing: np.ndarray
    frame_size: tuple[int, int]
    grid_thw: tuple[int, int, int]
    q_mean: np.ndarray | None = None
    roi_l7: np.ndarray | None = None
    roi_l15: np.ndarray | None = None

    def validate(self) -> None:
        candidate_count = len(self.candidate_ids)
        if self.q_last.shape != (3584,):
            raise ValueError(f"q_last must have shape (3584,), got {self.q_last.shape}")
        if self.q_mean is not None and self.q_mean.shape != (3584,):
            raise ValueError(
                f"q_mean must have shape (3584,), got {self.q_mean.shape}"
            )
        if self.roi_final.shape != (candidate_count, 3584):
            raise ValueError("roi_final shape does not match candidate count")
        for name, value in (
            ("roi_l7", self.roi_l7),
            ("roi_l15", self.roi_l15),
            ("roi_l23", self.roi_l23),
            ("roi_l31", self.roi_l31),
        ):
            if value is not None and value.shape != (candidate_count, 1280):
                raise ValueError(f"{name} shape does not match candidate count")
        expected_struct = (candidate_count, len(STRUCTURAL_FEATURE_NAMES))
        if (
            self.structure.shape != expected_struct
            or self.missing.shape != expected_struct
        ):
            raise ValueError("structural feature or missing-mask shape mismatch")
        arrays = [self.q_last, self.roi_final, self.structure, self.missing]
        if self.q_mean is not None:
            arrays.append(self.q_mean)
        arrays.extend(
            value
            for value in (
                self.roi_l7,
                self.roi_l15,
                self.roi_l23,
                self.roi_l31,
            )
            if value is not None
        )
        if any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("feature tensors must contain only finite values")
        if not np.isin(self.missing, (0.0, 1.0)).all():
            raise ValueError("feature missing mask must contain only 0/1 values")


@dataclass(frozen=True)
class SpatialGridFeatures:
    """Raw frozen visual grids used for exact-coordinate proposal repooling."""

    group_key: str
    grid_l31: np.ndarray
    grid_final: np.ndarray
    frame_size: tuple[int, int]
    grid_thw: tuple[int, int, int]
    grid_l7: np.ndarray | None = None

    def validate(self) -> None:
        if not self.group_key:
            raise ValueError("spatial-grid group key must be non-empty")
        width, height = self.frame_size
        if width <= 0 or height <= 0:
            raise ValueError("spatial-grid frame size must be positive")
        grid_t, grid_h, grid_w = self.grid_thw
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ValueError("spatial-grid dimensions must be positive")
        if grid_h % 2 or grid_w % 2:
            raise ValueError("spatial-grid height and width must be divisible by two")
        if self.grid_l31.shape != (grid_t, grid_h, grid_w, 1280):
            raise ValueError(
                "grid_l31 must match grid_thw with 1280 visual channels"
            )
        if self.grid_l7 is not None and self.grid_l7.shape != (
            grid_t,
            grid_h,
            grid_w,
            1280,
        ):
            raise ValueError("grid_l7 must match grid_thw with 1280 visual channels")
        if self.grid_final.shape != (grid_t, grid_h // 2, grid_w // 2, 3584):
            raise ValueError(
                "grid_final must match the merged grid with 3584 visual channels"
            )
        arrays = [self.grid_l31, self.grid_final]
        if self.grid_l7 is not None:
            arrays.append(self.grid_l7)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("spatial-grid tensors must contain only finite values")
