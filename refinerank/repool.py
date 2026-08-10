"""Exact ROI repooling of original/refined proposals on raw spatial grids.

Extracted from the research repo's ``ranking/nested_proposal.py``. Only the
three repooling entry points (``pool_spatial_rois``,
``pool_spatial_rois_with_l7``, ``repool_proposals``) plus their direct
helpers are kept; the nested-OOF / holdout / screening-gate machinery in the
original module is comparison-path code and is intentionally excluded.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from refinerank.geometry import area_weighted_roi_pool
from refinerank.proposal import ProposalCandidate, ProposalTarget
from refinerank.train_data import CachedTrainingGroup
from refinerank.types import SpatialGridFeatures


@dataclass(frozen=True)
class RepooledGroups:
    """Proposal-aligned groups plus coordinate/feature reconstruction audits."""

    groups: tuple[CachedTrainingGroup, ...]
    audit: Mapping[str, Any]



def pool_spatial_rois(
    spatial: SpatialGridFeatures,
    normalized_boxes: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool and L2-normalize final/l31 grids at exact normalized boxes."""

    spatial.validate()
    boxes = torch.as_tensor(np.asarray(normalized_boxes), dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] == 0:
        raise ValueError("repooling boxes must have non-empty shape (K,4)")
    final = area_weighted_roi_pool(
        torch.as_tensor(spatial.grid_final, dtype=torch.float32), boxes
    )
    l31 = area_weighted_roi_pool(
        torch.as_tensor(spatial.grid_l31, dtype=torch.float32), boxes
    )
    return (
        F.normalize(final, p=2, dim=-1, eps=1.0e-12)
        .cpu()
        .numpy()
        .astype(np.float16),
        F.normalize(l31, p=2, dim=-1, eps=1.0e-12)
        .cpu()
        .numpy()
        .astype(np.float16),
    )


def pool_spatial_rois_with_l7(
    spatial: SpatialGridFeatures,
    normalized_boxes: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Pool final/l31 and optional l7 grids at exact normalized boxes."""

    final, l31 = pool_spatial_rois(spatial, normalized_boxes)
    if spatial.grid_l7 is None:
        return final, l31, None
    boxes = torch.as_tensor(np.asarray(normalized_boxes), dtype=torch.float32)
    l7 = area_weighted_roi_pool(
        torch.as_tensor(spatial.grid_l7, dtype=torch.float32), boxes
    )
    return (
        final,
        l31,
        F.normalize(l7, p=2, dim=-1, eps=1.0e-12).cpu().numpy().astype(np.float16),
    )


def repool_proposals(
    groups: Sequence[CachedTrainingGroup],
    proposals: Mapping[str, Sequence[ProposalCandidate]],
    targets: Mapping[str, ProposalTarget],
    spatial_grids: Mapping[str, SpatialGridFeatures],
    *,
    reconstruction_atol: float = 0.005,
    moved_roi_min_l2: float = 1.0e-5,
    refined_roi_changed_fraction_min: float = 0.95,
) -> RepooledGroups:
    """Build selector groups whose ROI and geometry match each proposal box."""

    if not 0.0 < refined_roi_changed_fraction_min <= 1.0:
        raise ValueError(
            "refined ROI changed-fraction minimum must be within (0,1]"
        )

    expected = {group.group_key for group in groups}
    for name, values in (
        ("proposals", proposals),
        ("targets", targets),
        ("spatial grids", spatial_grids),
    ):
        if set(values) != expected:
            raise ValueError(f"{name} do not cover the exact training groups")
    output: list[CachedTrainingGroup] = []
    original_errors: list[float] = []
    refined_deltas: list[float] = []
    source_parent_l23_count = 0
    exact_l7_count = 0
    recomputed_geometry_count = 0
    for group in groups:
        selected = tuple(proposals[group.group_key])
        if not selected:
            raise ValueError(f"proposal group {group.group_key} is empty")
        candidate_ids = tuple(item.candidate_id for item in selected)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"proposal group {group.group_key} has duplicate IDs")
        spatial = spatial_grids[group.group_key]
        if spatial.frame_size != group.frame_size:
            raise ValueError(f"spatial frame mismatch for {group.group_key}")
        source_indices = tuple(int(item.source_index) for item in selected)
        if any(
            source < 0 or source >= len(group.candidate_ids)
            for source in source_indices
        ):
            raise ValueError(
                f"proposal source index is invalid for {group.group_key}"
            )
        boxes_norm = tuple(item.bbox_xyxy_norm for item in selected)
        roi_final, roi_l31, repooled_l7 = pool_spatial_rois_with_l7(
            spatial, boxes_norm
        )
        roi_l7 = (
            repooled_l7
            if repooled_l7 is not None
            else (
                np.stack(
                    [np.asarray(group.roi_l7[source]) for source in source_indices]
                ).astype(np.float16)
                if group.roi_l7 is not None
                else None
            )
        )
        if repooled_l7 is not None:
            if group.roi_l7 is None:
                raise ValueError("l7 spatial grid requires parent roi_l7")
            exact_l7_count += len(selected)
        roi_l23 = (
            np.stack(
                [
                    np.asarray(group.roi_l23[source])
                    for source in source_indices
                ]
            ).astype(np.float16)
            if group.roi_l23 is not None
            else None
        )
        if roi_l23 is not None:
            source_parent_l23_count += len(selected)
        structure = np.zeros((len(selected), 12), dtype=np.float32)
        missing = np.zeros_like(structure)
        ious = np.zeros(len(selected), dtype=np.float32)
        for index, proposal in enumerate(selected):
            source = source_indices[index]
            structure[index] = np.asarray(group.structure[source], dtype=np.float32)
            missing[index] = np.asarray(group.missing[source], dtype=np.float32)
            box = np.asarray(proposal.bbox_xyxy_norm, dtype=np.float32)
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            if width <= 0.0 or height <= 0.0:
                raise ValueError("proposal has a non-positive normalized area")
            structure[index, 6:10] = box
            structure[index, 10] = math.log(max(width * height, 1.0e-8))
            structure[index, 11] = math.log(max(width / height, 1.0e-8))
            missing[index, 6:12] = 0.0
            recomputed_geometry_count += 1
            ious[index] = _proposal_iou(proposal, targets[group.group_key])
            if group.roi_l31 is None:
                raise ValueError("proposal repooling requires parent roi_l31")
            parent_final = np.asarray(group.roi_final[source], dtype=np.float32)
            parent_l31 = np.asarray(group.roi_l31[source], dtype=np.float32)
            feature_pairs = [
                (roi_final[index].astype(np.float32), parent_final),
                (roi_l31[index].astype(np.float32), parent_l31),
            ]
            if repooled_l7 is not None:
                feature_pairs.append(
                    (
                        roi_l7[index].astype(np.float32),
                        np.asarray(group.roi_l7[source], dtype=np.float32),
                    )
                )
            if proposal.refined:
                refined_deltas.append(
                    max(
                        float(np.linalg.norm(actual - parent))
                        for actual, parent in feature_pairs
                    )
                )
            else:
                original_errors.append(
                    max(
                        float(np.max(np.abs(actual - parent)))
                        for actual, parent in feature_pairs
                    )
                )
        output.append(
            CachedTrainingGroup(
                group_key=group.group_key,
                sample_id=group.sample_id,
                row_index=group.row_index,
                dataset=group.dataset,
                original_id=group.original_id,
                video_id=group.video_id,
                clean_question=group.clean_question,
                requested_timestamp=group.requested_timestamp,
                candidate_ids=candidate_ids,
                boxes_xyxy=tuple(item.bbox_xyxy for item in selected),
                ranks=tuple(item.rank for item in selected),
                q_last=group.q_last,
                q_mean=group.q_mean,
                roi_final=roi_final,
                roi_l7=roi_l7,
                roi_l31=roi_l31,
                roi_l23=roi_l23,
                structure=structure,
                missing=missing,
                ious=ious,
                native_scores=np.asarray(
                    [item.score for item in selected], dtype=np.float32
                ),
                frame_size=group.frame_size,
                parent_source_indices=tuple(
                    int(item.source_index) for item in selected
                ),
                refined_flags=tuple(bool(item.refined) for item in selected),
            )
        )
    max_original_error = max(original_errors, default=0.0)
    refined_changed = sum(value > moved_roi_min_l2 for value in refined_deltas)
    audit = {
        "group_count": len(output),
        "proposal_count": sum(len(group.candidate_ids) for group in output),
        "original_proposal_count": len(original_errors),
        "refined_proposal_count": len(refined_deltas),
        "recomputed_geometry_count": recomputed_geometry_count,
        "original_reconstruction_max_abs": max_original_error,
        "reconstruction_atol": reconstruction_atol,
        "original_reconstruction_passed": max_original_error <= reconstruction_atol,
        "refined_visual_changed_count": refined_changed,
        "refined_visual_changed_fraction": (
            refined_changed / len(refined_deltas) if refined_deltas else 0.0
        ),
        "moved_roi_min_l2": moved_roi_min_l2,
        "refined_visual_changed_fraction_min": refined_roi_changed_fraction_min,
        "refined_visual_change_passed": bool(refined_deltas)
        and refined_changed / len(refined_deltas)
        >= refined_roi_changed_fraction_min,
        "source_parent_l23_count": source_parent_l23_count,
        "source_parent_l23_coverage": source_parent_l23_count
        / sum(len(group.candidate_ids) for group in output),
        "exact_l7_count": exact_l7_count,
        "exact_l7_coverage": exact_l7_count
        / sum(len(group.candidate_ids) for group in output),
        "invalid_box_count": 0,
    }
    return RepooledGroups(groups=tuple(output), audit=audit)


def _proposal_iou(candidate: ProposalCandidate, target: ProposalTarget) -> float:
    if not target.has_target or target.gt_box_xyxy_norm is None:
        return 0.0
    left = candidate.bbox_xyxy_norm
    right = target.gt_box_xyxy_norm
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1.0e-12)
