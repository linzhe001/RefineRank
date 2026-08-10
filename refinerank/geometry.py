"""Coordinate-safe grid restoration and proposal ROI pooling."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

BOUNDARY_SNAP_TOLERANCE_PIXELS = 2.5


def validate_box_xyxy(
    box: Sequence[float],
    *,
    width: int,
    height: int,
    tolerance: float = BOUNDARY_SNAP_TOLERANCE_PIXELS,
) -> tuple[float, float, float, float]:
    """Validate an absolute source-pixel xyxy proposal."""

    if len(box) != 4:
        raise ValueError(f"xyxy box must have four values, got {len(box)}")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"xyxy box contains non-finite values: {values}")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"xyxy box has non-positive area: {values}")
    if x1 < -tolerance or y1 < -tolerance:
        raise ValueError(f"xyxy box starts outside source image: {values}")
    if x2 > width + tolerance or y2 > height + tolerance:
        raise ValueError(f"xyxy box exceeds source image {width}x{height}: {values}")
    return values


def normalize_box_xyxy(
    box: Sequence[float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert source-pixel xyxy coordinates to [0, 1] coordinates."""

    x1, y1, x2, y2 = validate_box_xyxy(box, width=width, height=height)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("xyxy box has no area after boundary snap")
    return x1 / width, y1 / height, x2 / width, y2 / height


def denormalize_box_xyxy(
    box: Sequence[float], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Restore normalized xyxy coordinates to source pixels."""

    if len(box) != 4:
        raise ValueError("normalized xyxy box must have four values")
    x1, y1, x2, y2 = (float(value) for value in box)
    if min(x1, y1) < 0.0 or max(x2, y2) > 1.0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid normalized xyxy box: {tuple(box)}")
    return x1 * width, y1 * height, x2 * width, y2 * height


def inverse_window_group_order(
    tokens: Tensor,
    window_index: Tensor,
    *,
    merge_size: int,
) -> Tensor:
    """Undo Qwen2.5-VL window ordering while preserving merge-unit order."""

    if tokens.ndim != 2:
        raise ValueError(f"vision tokens must be 2D, got {tokens.shape}")
    merge_unit = merge_size * merge_size
    if tokens.shape[0] % merge_unit:
        raise ValueError("vision token count is not divisible by merge unit")
    group_count = tokens.shape[0] // merge_unit
    if window_index.ndim != 1 or window_index.numel() != group_count:
        raise ValueError("window_index length does not match vision merge groups")
    grouped = tokens.reshape(group_count, merge_unit, tokens.shape[-1])
    reverse_indices = torch.argsort(window_index.to(device=tokens.device))
    return grouped[reverse_indices].reshape_as(tokens)


def patch_tokens_to_grid(
    tokens: Tensor,
    grid_thw: Sequence[int],
    *,
    merge_size: int,
) -> Tensor:
    """Restore merge-group-major patch tokens to T x H x W x C order."""

    if len(grid_thw) != 3:
        raise ValueError("grid_thw must contain temporal, height, and width")
    grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("vision grid is not divisible by merge_size")
    expected = grid_t * grid_h * grid_w
    if tokens.ndim != 2 or tokens.shape[0] != expected:
        raise ValueError(
            f"patch tokens must have shape ({expected}, C), got {tokens.shape}"
        )
    channels = tokens.shape[-1]
    grouped = tokens.reshape(
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channels,
    )
    return grouped.permute(0, 1, 3, 2, 4, 5).reshape(grid_t, grid_h, grid_w, channels)


def merged_tokens_to_grid(
    tokens: Tensor,
    grid_thw: Sequence[int],
    *,
    merge_size: int,
) -> Tensor:
    """Restore final visual tokens to the merged T x H/2 x W/2 grid."""

    grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
    merged_h = grid_h // merge_size
    merged_w = grid_w // merge_size
    expected = grid_t * merged_h * merged_w
    if tokens.ndim != 2 or tokens.shape[0] != expected:
        raise ValueError(
            f"merged tokens must have shape ({expected}, C), got {tokens.shape}"
        )
    return tokens.reshape(grid_t, merged_h, merged_w, tokens.shape[-1])


def area_weighted_roi_pool(
    grid: Tensor,
    normalized_boxes: Tensor,
    *,
    temporal_index: int = 0,
) -> Tensor:
    """Pool grid features by exact overlap area with normalized xyxy boxes."""

    if grid.ndim != 4:
        raise ValueError(f"grid must have shape (T,H,W,C), got {grid.shape}")
    if normalized_boxes.ndim != 2 or normalized_boxes.shape[-1] != 4:
        raise ValueError("normalized_boxes must have shape (K,4)")
    if temporal_index < 0 or temporal_index >= grid.shape[0]:
        raise ValueError("temporal_index is outside the visual grid")
    if normalized_boxes.numel() == 0:
        return grid.new_empty((0, grid.shape[-1]))
    if not torch.isfinite(normalized_boxes).all():
        raise ValueError("normalized_boxes contain non-finite values")
    x1, y1, x2, y2 = normalized_boxes.unbind(dim=-1)
    invalid = (x1 < 0) | (y1 < 0) | (x2 > 1) | (y2 > 1) | (x2 <= x1) | (y2 <= y1)
    if invalid.any():
        raise ValueError("normalized_boxes contain invalid xyxy coordinates")

    height, width = grid.shape[1:3]
    cell_x1 = torch.arange(width, device=grid.device, dtype=grid.dtype) / width
    cell_x2 = cell_x1 + 1.0 / width
    cell_y1 = torch.arange(height, device=grid.device, dtype=grid.dtype) / height
    cell_y2 = cell_y1 + 1.0 / height

    overlap_x = (
        torch.minimum(x2[:, None], cell_x2[None, :])
        - torch.maximum(x1[:, None], cell_x1[None, :])
    ).clamp_min(0)
    overlap_y = (
        torch.minimum(y2[:, None], cell_y2[None, :])
        - torch.maximum(y1[:, None], cell_y1[None, :])
    ).clamp_min(0)
    weights = overlap_y[:, :, None] * overlap_x[:, None, :]
    weight_sum = weights.sum(dim=(1, 2))
    if (weight_sum <= 0).any():
        raise ValueError("at least one ROI has zero overlap with the visual grid")
    features = grid[temporal_index]
    return torch.einsum("khw,hwc->kc", weights, features) / weight_sum[:, None]
