"""GT-isolated targets and a query-conditioned local proposal adapter."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from refinerank.cache import ProposalTargetStore, sha256_file
from refinerank.coordinates import normalize_annotation_box_xyxy
from refinerank.geometry import (
    denormalize_box_xyxy,
    normalize_box_xyxy,
)
from refinerank.model import ranking_loss
from refinerank.train_data import CachedTrainingGroup


@dataclass(frozen=True)
class ProposalAdapterConfig:
    """Frozen architecture, bounded-coordinate, and shortlist configuration."""

    d_model: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    max_trainable_parameters: int = 1_500_000
    center_shift_scale: float = 0.5
    max_log_scale: float = math.log(2.0)
    regression_anchor_count: int = 8
    shortlist_k: int = 48
    native_anchor_count: int = 16
    quality_weight: float = 0.7
    diversity_weight: float = 0.3

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.hidden_dim <= 0:
            raise ValueError("proposal dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("proposal dropout must be within [0,1)")
        if self.center_shift_scale <= 0.0 or self.max_log_scale <= 0.0:
            raise ValueError("proposal delta bounds must be positive")
        if not 1 <= self.regression_anchor_count <= self.shortlist_k:
            raise ValueError("proposal regression anchor count is invalid")
        if not 1 <= self.native_anchor_count <= self.shortlist_k:
            raise ValueError("proposal native anchor count is invalid")
        if not math.isclose(
            self.quality_weight + self.diversity_weight,
            1.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError("proposal shortlist weights must sum to one")


@dataclass(frozen=True)
class ProposalAdapterInputMask:
    """Explicitly enable proposal-adapter branches before projection."""

    use_query: bool = True
    use_roi: bool = True
    use_structure: bool = True
    predict_box_delta: bool = True
    use_final: bool = True
    use_l31: bool = True


PROPOSAL_INPUT_MASKS = {
    "none": ProposalAdapterInputMask(
        use_query=False,
        use_roi=False,
        use_final=False,
        use_l31=False,
        use_structure=False,
        predict_box_delta=False,
    ),
    "quality_only": ProposalAdapterInputMask(predict_box_delta=False),
    "structure": ProposalAdapterInputMask(use_query=False, use_roi=False),
    "roi_no_query": ProposalAdapterInputMask(use_query=False),
    "full": ProposalAdapterInputMask(),
}


def proposal_input_mask(variant: str) -> ProposalAdapterInputMask:
    """Resolve one public Proposal Repair ablation variant."""

    try:
        return PROPOSAL_INPUT_MASKS[variant]
    except KeyError as error:
        choices = ", ".join(sorted(PROPOSAL_INPUT_MASKS))
        raise ValueError(
            f"unsupported proposal repair variant {variant!r}; choose {choices}"
        ) from error


@dataclass(frozen=True)
class ProposalTarget:
    """One requested-timestamp training target."""

    has_target: bool
    gt_box_xyxy_norm: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ProposalAdapterOutput:
    """Per-candidate quality logits and already-bounded box deltas."""

    original_quality_logits: Tensor
    refined_quality_logits: Tensor
    bounded_deltas: Tensor


@dataclass(frozen=True)
class ProposalCandidate:
    """One GT-free original or locally refined proposal."""

    candidate_id: str
    bbox_xyxy_norm: tuple[float, float, float, float]
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    rank: int
    source_index: int
    refined: bool


class QueryConditionedProposalAdapter(nn.Module):
    """Candidate-wise query/ROI fusion with bounded coordinate refinement."""

    def __init__(self, config: ProposalAdapterConfig = ProposalAdapterConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.query_projection = nn.Linear(3584, width)
        self.final_projection = nn.Linear(3584, width)
        self.l31_projection = nn.Linear(1280, width)
        self.structure_projection = nn.Linear(24, width)
        self.query_norm = nn.LayerNorm(width)
        self.candidate_norm = nn.LayerNorm(width)
        self.fusion = nn.Sequential(
            nn.Linear(width * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.quality_head = nn.Linear(config.hidden_dim, 2)
        self.delta_head = nn.Linear(config.hidden_dim, 4)
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if parameter_count > config.max_trainable_parameters:
            raise ValueError(
                f"proposal adapter has {parameter_count} parameters, above "
                f"{config.max_trainable_parameters}"
            )

    def forward(
        self,
        *,
        q_last: Tensor,
        roi_final: Tensor,
        roi_l31: Tensor,
        structure: Tensor,
        missing: Tensor,
        padding_mask: Tensor,
        input_mask: ProposalAdapterInputMask = ProposalAdapterInputMask(),
    ) -> ProposalAdapterOutput:
        """Predict quality and bounded deltas without candidate-order features."""

        _validate_adapter_inputs(
            q_last,
            roi_final,
            roi_l31,
            structure,
            missing,
            padding_mask,
        )
        query_input = q_last.float() * float(input_mask.use_query)
        use_final = input_mask.use_roi and input_mask.use_final
        use_l31 = input_mask.use_roi and input_mask.use_l31
        final_input = roi_final.float() * float(use_final)
        l31_input = roi_l31.float() * float(use_l31)
        structure_input = torch.cat((structure, missing), dim=-1) * float(
            input_mask.use_structure
        )
        query = self.query_norm(
            self.query_projection(F.normalize(query_input, dim=-1))
        ) * float(input_mask.use_query)
        candidate = self.candidate_norm(
            self.final_projection(F.normalize(final_input, dim=-1))
            * float(use_final)
            + self.l31_projection(F.normalize(l31_input, dim=-1))
            * float(use_l31)
            + self.structure_projection(structure_input)
            * float(input_mask.use_structure)
        )
        expanded_query = query[:, None, :].expand_as(candidate)
        fused = self.fusion(
            torch.cat((candidate, expanded_query, candidate * expanded_query), dim=-1)
        )
        qualities = self.quality_head(fused)
        raw_deltas = self.delta_head(fused)
        bounds = raw_deltas.new_tensor(
            [
                self.config.center_shift_scale,
                self.config.center_shift_scale,
                self.config.max_log_scale,
                self.config.max_log_scale,
            ]
        )
        bounded = (
            torch.tanh(raw_deltas)
            * bounds
            * float(input_mask.predict_box_delta)
        )
        floor = torch.finfo(qualities.dtype).min
        return ProposalAdapterOutput(
            original_quality_logits=qualities[..., 0].masked_fill(
                padding_mask, floor
            ),
            refined_quality_logits=qualities[..., 1].masked_fill(
                padding_mask, floor
            ),
            bounded_deltas=bounded.masked_fill(padding_mask[..., None], 0.0),
        )


def build_proposal_target_store(
    groups: Sequence[CachedTrainingGroup],
    *,
    train_split: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build exact-timestamp normalized GT targets in a separate train-only store."""

    loaded = json.loads(Path(train_split).read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise TypeError("proposal train split must be a JSON list")
    entries: dict[str, dict[str, Any]] = {}
    missing_by_dataset: Counter[str] = Counter()
    present_by_dataset: Counter[str] = Counter()
    for group in groups:
        if group.row_index < 0 or group.row_index >= len(loaded):
            raise IndexError(
                f"proposal group row index is outside split: {group.row_index}"
            )
        raw = loaded[group.row_index]
        if not isinstance(raw, Mapping):
            raise TypeError(f"proposal split row {group.row_index} must be a mapping")
        raw_id = str(raw.get("id") or raw.get("sample_id") or "")
        if raw_id != group.sample_id:
            raise ValueError(
                f"proposal split row {group.row_index} sample id mismatch"
            )
        gt_boxes = _extract_gt_boxes(raw)
        gt_box = gt_boxes.get(_timestamp_key(group.requested_timestamp))
        if gt_box is None:
            entries[group.group_key] = {
                "has_target": False,
                "gt_box_xyxy_norm": None,
            }
            missing_by_dataset[group.dataset] += 1
            continue
        normalized = normalize_annotation_box_xyxy(
            gt_box,
            dataset=group.dataset,
            image_size=group.frame_size,
        )
        entries[group.group_key] = {
            "has_target": True,
            "gt_box_xyxy_norm": normalized,
        }
        present_by_dataset[group.dataset] += 1
    split_hash = sha256_file(Path(train_split))
    store_hash = ProposalTargetStore(output_path, phase="train").write(
        entries,
        source_split_hash=split_hash,
    )
    return {
        "schema_version": 1,
        "phase": "train",
        "contains_gt": True,
        "group_count": len(entries),
        "target_count": sum(present_by_dataset.values()),
        "missing_target_count": sum(missing_by_dataset.values()),
        "target_count_by_dataset": dict(sorted(present_by_dataset.items())),
        "missing_target_count_by_dataset": dict(sorted(missing_by_dataset.items())),
        "source_split_hash": split_hash,
        "store_hash": store_hash,
    }


def load_proposal_targets(
    groups: Sequence[CachedTrainingGroup],
    *,
    target_store: Path,
    expected_source_split_hash: str,
) -> dict[str, ProposalTarget]:
    """Load a target store and require exact group coverage."""

    raw = ProposalTargetStore(target_store, phase="train").read(
        expected_source_split_hash=expected_source_split_hash
    )
    expected = {group.group_key for group in groups}
    if set(raw) != expected:
        raise ValueError("ProposalTargetStore coverage does not match training groups")
    return {
        group_key: ProposalTarget(
            has_target=bool(entry["has_target"]),
            gt_box_xyxy_norm=entry["gt_box_xyxy_norm"],
        )
        for group_key, entry in raw.items()
    }


def decode_bounded_box_deltas(boxes_xyxy: Tensor, deltas: Tensor) -> Tensor:
    """Apply bounded center/log-scale deltas in normalized coordinates."""

    if boxes_xyxy.shape != deltas.shape or boxes_xyxy.shape[-1] != 4:
        raise ValueError("proposal boxes and deltas must have identical (...,4) shape")
    if not torch.isfinite(boxes_xyxy).all() or not torch.isfinite(deltas).all():
        raise ValueError("proposal boxes and deltas must be finite")
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    width = x2 - x1
    height = y2 - y1
    if (
        (x1 < 0).any()
        or (y1 < 0).any()
        or (x2 > 1).any()
        or (y2 > 1).any()
        or (width <= 0).any()
        or (height <= 0).any()
    ):
        raise ValueError("proposal decode received invalid normalized boxes")
    dx, dy, dw, dh = deltas.unbind(dim=-1)
    center_x = (x1 + x2) * 0.5 + dx * width
    center_y = (y1 + y2) * 0.5 + dy * height
    new_width = width * torch.exp(dw)
    new_height = height * torch.exp(dh)
    decoded = torch.stack(
        (
            (center_x - 0.5 * new_width).clamp(0.0, 1.0),
            (center_y - 0.5 * new_height).clamp(0.0, 1.0),
            (center_x + 0.5 * new_width).clamp(0.0, 1.0),
            (center_y + 0.5 * new_height).clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    if ((decoded[..., 2:] - decoded[..., :2]) <= 0).any():
        raise ValueError("proposal decode produced a non-positive box")
    return decoded


def encode_bounded_box_targets(
    boxes_xyxy: Tensor,
    gt_boxes_xyxy: Tensor,
    *,
    config: ProposalAdapterConfig,
) -> Tensor:
    """Encode GT into the same clipped delta space predicted by the adapter."""

    if boxes_xyxy.shape[-1] != 4 or gt_boxes_xyxy.shape[-1] != 4:
        raise ValueError("proposal target boxes need four coordinates")
    if gt_boxes_xyxy.ndim == boxes_xyxy.ndim - 1:
        gt_boxes_xyxy = gt_boxes_xyxy.unsqueeze(-2).expand_as(boxes_xyxy)
    if gt_boxes_xyxy.shape != boxes_xyxy.shape:
        raise ValueError("proposal target box shape mismatch")
    _validate_normalized_box_tensor(boxes_xyxy, name="proposal boxes")
    _validate_normalized_box_tensor(gt_boxes_xyxy, name="proposal GT boxes")
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = gt_boxes_xyxy.unbind(dim=-1)
    width = x2 - x1
    height = y2 - y1
    gt_width = gx2 - gx1
    gt_height = gy2 - gy1
    dx = (((gx1 + gx2) - (x1 + x2)) * 0.5 / width).clamp(
        -config.center_shift_scale, config.center_shift_scale
    )
    dy = (((gy1 + gy2) - (y1 + y2)) * 0.5 / height).clamp(
        -config.center_shift_scale, config.center_shift_scale
    )
    dw = torch.log(gt_width / width).clamp(
        -config.max_log_scale, config.max_log_scale
    )
    dh = torch.log(gt_height / height).clamp(
        -config.max_log_scale, config.max_log_scale
    )
    return torch.stack((dx, dy, dw, dh), dim=-1)


def proposal_adapter_loss(
    output: ProposalAdapterOutput,
    *,
    original_boxes_xyxy: Tensor,
    original_ious: Tensor,
    gt_boxes_xyxy: Tensor,
    has_target: Tensor,
    padding_mask: Tensor,
    config: ProposalAdapterConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train quality, bounded deltas, and refined-box overlap on target groups."""

    if has_target.ndim != 1 or has_target.dtype != torch.bool:
        raise ValueError("proposal has_target must be bool with shape (B,)")
    if not has_target.any():
        raise ValueError("proposal batch has no requested-timestamp GT targets")
    batch_size, candidate_count = original_ious.shape
    if original_boxes_xyxy.shape != (batch_size, candidate_count, 4):
        raise ValueError("proposal original box shape mismatch")
    if gt_boxes_xyxy.shape != (batch_size, 4):
        raise ValueError("proposal GT box shape mismatch")
    valid_groups = has_target
    boxes = original_boxes_xyxy[valid_groups]
    ious = original_ious[valid_groups]
    mask = padding_mask[valid_groups]
    dummy_box = boxes.new_tensor((0.0, 0.0, 1.0, 1.0))
    boxes = torch.where(mask[..., None], dummy_box, boxes)
    gt = gt_boxes_xyxy[valid_groups]
    deltas = output.bounded_deltas[valid_groups]
    predicted_boxes = decode_bounded_box_deltas(boxes, deltas)
    expanded_gt = gt[:, None, :].expand_as(predicted_boxes)
    refined_ious = aligned_box_iou(predicted_boxes, expanded_gt).detach()
    original_total, original_parts = ranking_loss(
        output.original_quality_logits[valid_groups], ious, mask
    )
    refined_total, refined_parts = ranking_loss(
        output.refined_quality_logits[valid_groups], refined_ious, mask
    )
    target_deltas = encode_bounded_box_targets(boxes, gt, config=config)
    anchor_mask = torch.zeros_like(mask)
    for row in range(boxes.shape[0]):
        valid_indices = torch.where(~mask[row])[0]
        count = min(config.regression_anchor_count, int(valid_indices.numel()))
        selected = valid_indices[
            torch.topk(ious[row, valid_indices], k=count, largest=True).indices
        ]
        anchor_mask[row, selected] = True
    delta_loss = F.smooth_l1_loss(deltas[anchor_mask], target_deltas[anchor_mask])
    giou_loss = (1.0 - aligned_generalized_box_iou(
        predicted_boxes[anchor_mask], expanded_gt[anchor_mask]
    )).mean()
    total = original_total + refined_total + delta_loss + 0.5 * giou_loss
    return total, {
        "total": total.detach(),
        "original_listwise": original_parts["listwise"],
        "original_regression": original_parts["regression"],
        "refined_listwise": refined_parts["listwise"],
        "refined_regression": refined_parts["regression"],
        "delta": delta_loss.detach(),
        "giou": giou_loss.detach(),
        "target_group_count": valid_groups.sum().detach(),
        "anchor_count": anchor_mask.sum().detach(),
    }


def quality_diversity_shortlist(
    group: CachedTrainingGroup,
    *,
    original_scores: Sequence[float],
    refined_scores: Sequence[float],
    refined_boxes_xyxy_norm: Sequence[Sequence[float]],
    config: ProposalAdapterConfig,
) -> tuple[ProposalCandidate, ...]:
    """Keep native anchors, then GT-free quality/diversity MMR to K."""

    count = len(group.candidate_ids)
    if not (
        len(original_scores)
        == len(refined_scores)
        == len(refined_boxes_xyxy_norm)
        == count
    ):
        raise ValueError("proposal shortlist inputs do not align with candidates")
    width, height = group.frame_size
    if width <= 1 or height <= 1:
        raise ValueError("proposal shortlist requires the source frame size")
    originals: list[ProposalCandidate] = []
    refined: list[ProposalCandidate] = []
    for index, candidate_id in enumerate(group.candidate_ids):
        original_norm = tuple(float(value) for value in group.structure[index, 6:10])
        originals.append(
            _make_proposal_candidate(
                candidate_id=candidate_id,
                box_norm=original_norm,
                score=float(original_scores[index]),
                rank=group.ranks[index],
                source_index=index,
                refined=False,
                width=width,
                height=height,
                absolute_box=group.boxes_xyxy[index],
            )
        )
        refined.append(
            _make_proposal_candidate(
                candidate_id=f"{candidate_id}::refined",
                box_norm=tuple(
                    float(value) for value in refined_boxes_xyxy_norm[index]
                ),
                score=float(refined_scores[index]),
                rank=group.ranks[index],
                source_index=index,
                refined=True,
                width=width,
                height=height,
                absolute_box=None,
            )
        )
    anchor_count = min(config.native_anchor_count, count, config.shortlist_k)
    selected = list(originals[:anchor_count])
    pool = originals[anchor_count:] + refined
    while pool and len(selected) < config.shortlist_k:
        best_index = min(
            range(len(pool)),
            key=lambda index: (
                -_mmr_objective(pool[index], selected, config=config),
                pool[index].candidate_id,
            ),
        )
        selected.append(pool.pop(best_index))
    return tuple(selected)


def aligned_box_iou(left: Tensor, right: Tensor) -> Tensor:
    """Aligned IoU for identically shaped normalized boxes."""

    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("aligned IoU boxes must have identical (...,4) shape")
    intersection_lt = torch.maximum(left[..., :2], right[..., :2])
    intersection_rb = torch.minimum(left[..., 2:], right[..., 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    left_wh = left[..., 2:] - left[..., :2]
    right_wh = right[..., 2:] - right[..., :2]
    left_area = left_wh[..., 0] * left_wh[..., 1]
    right_area = right_wh[..., 0] * right_wh[..., 1]
    return intersection / (left_area + right_area - intersection).clamp_min(1.0e-8)


def aligned_generalized_box_iou(left: Tensor, right: Tensor) -> Tensor:
    """Aligned generalized IoU for normalized xyxy boxes."""

    iou = aligned_box_iou(left, right)
    enclosure_lt = torch.minimum(left[..., :2], right[..., :2])
    enclosure_rb = torch.maximum(left[..., 2:], right[..., 2:])
    enclosure_wh = (enclosure_rb - enclosure_lt).clamp_min(0.0)
    enclosure = enclosure_wh[..., 0] * enclosure_wh[..., 1]
    intersection_lt = torch.maximum(left[..., :2], right[..., :2])
    intersection_rb = torch.minimum(left[..., 2:], right[..., 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    left_wh = left[..., 2:] - left[..., :2]
    right_wh = right[..., 2:] - right[..., :2]
    union = (
        left_wh[..., 0] * left_wh[..., 1]
        + right_wh[..., 0] * right_wh[..., 1]
        - intersection
    )
    return iou - (enclosure - union) / enclosure.clamp_min(1.0e-8)


def proposal_config_dict(config: ProposalAdapterConfig) -> dict[str, Any]:
    """Return a manifest-safe config mapping."""

    return asdict(config)


def _extract_gt_boxes(
    raw: Mapping[str, Any],
) -> dict[str, tuple[float, float, float, float]]:
    struc_info = raw.get("struc_info")
    items = struc_info if isinstance(struc_info, list) else [struc_info]
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("bbox_dict"), Mapping
        ):
            continue
        output: dict[str, tuple[float, float, float, float]] = {}
        for timestamp, box in item["bbox_dict"].items():
            if (
                isinstance(box, Sequence)
                and not isinstance(box, (str, bytes))
                and len(box) == 4
            ):
                output[_timestamp_key(timestamp)] = tuple(float(value) for value in box)
        if output:
            return output
    return {}


def _timestamp_key(value: Any) -> str:
    return f"{float(value):.3f}"


def _validate_adapter_inputs(
    q_last: Tensor,
    roi_final: Tensor,
    roi_l31: Tensor,
    structure: Tensor,
    missing: Tensor,
    padding_mask: Tensor,
) -> None:
    if q_last.ndim != 2 or q_last.shape[-1] != 3584:
        raise ValueError("proposal q_last must have shape (B,3584)")
    if roi_final.ndim != 3 or roi_final.shape[-1] != 3584:
        raise ValueError("proposal roi_final must have shape (B,K,3584)")
    if roi_l31.shape != (*roi_final.shape[:2], 1280):
        raise ValueError("proposal roi_l31 must have shape (B,K,1280)")
    if (
        structure.shape != (*roi_final.shape[:2], 12)
        or missing.shape != structure.shape
    ):
        raise ValueError("proposal structure/missing must have shape (B,K,12)")
    if padding_mask.shape != roi_final.shape[:2] or padding_mask.dtype != torch.bool:
        raise ValueError("proposal padding mask must be bool with shape (B,K)")
    tensors = (q_last, roi_final, roi_l31, structure, missing)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("proposal adapter inputs must be finite")


def _validate_normalized_box_tensor(boxes: Tensor, *, name: str) -> None:
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} must be finite")
    if (
        (boxes[..., :2] < 0).any()
        or (boxes[..., 2:] > 1).any()
        or ((boxes[..., 2:] - boxes[..., :2]) <= 0).any()
    ):
        raise ValueError(f"{name} contains an invalid normalized box")


def _make_proposal_candidate(
    *,
    candidate_id: str,
    box_norm: tuple[float, ...],
    score: float,
    rank: int,
    source_index: int,
    refined: bool,
    width: int,
    height: int,
    absolute_box: Sequence[float] | None,
) -> ProposalCandidate:
    if len(box_norm) != 4 or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("proposal shortlist received invalid box or score")
    absolute = (
        tuple(float(value) for value in absolute_box)
        if absolute_box is not None
        else denormalize_box_xyxy(box_norm, width=width, height=height)
    )
    round_trip = normalize_box_xyxy(absolute, width=width, height=height)
    if not np.allclose(box_norm, round_trip, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("proposal coordinate round-trip failed")
    return ProposalCandidate(
        candidate_id=candidate_id,
        bbox_xyxy_norm=tuple(float(value) for value in box_norm),
        bbox_xyxy=absolute,
        score=score,
        rank=rank,
        source_index=source_index,
        refined=refined,
    )


def _mmr_objective(
    candidate: ProposalCandidate,
    selected: Sequence[ProposalCandidate],
    *,
    config: ProposalAdapterConfig,
) -> float:
    if selected:
        max_overlap = max(
            _box_iou_numpy(candidate.bbox_xyxy_norm, item.bbox_xyxy_norm)
            for item in selected
        )
        diversity = 1.0 - max_overlap
    else:
        diversity = 1.0
    return config.quality_weight * candidate.score + config.diversity_weight * diversity


def _box_iou_numpy(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return intersection / max(left_area + right_area - intersection, 1.0e-12)
