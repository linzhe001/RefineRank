from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from refinerank.cache import ProposalTargetStore, sha256_file
from refinerank.proposal import (
    ProposalAdapterConfig,
    ProposalAdapterInputMask,
    ProposalTarget,
    QueryConditionedProposalAdapter,
    build_proposal_target_store,
    decode_bounded_box_deltas,
    load_proposal_targets,
    proposal_adapter_loss,
    quality_diversity_shortlist,
)
from refinerank.train_data import CachedTrainingGroup


def _group(
    *,
    timestamp: float = 1.0,
    candidate_count: int = 4,
) -> CachedTrainingGroup:
    boxes = np.asarray(
        [
            (0.05, 0.05, 0.25, 0.25),
            (0.30, 0.10, 0.55, 0.40),
            (0.10, 0.50, 0.40, 0.85),
            (0.60, 0.55, 0.90, 0.90),
        ][:candidate_count],
        dtype=np.float32,
    )
    structure = np.zeros((candidate_count, 12), dtype=np.float32)
    structure[:, :6] = 0.5
    structure[:, 6:10] = boxes
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    structure[:, 10] = np.log(widths * heights)
    structure[:, 11] = np.log(widths / heights)
    return CachedTrainingGroup(
        group_key=f"000000:sample@{timestamp:.3f}",
        sample_id="sample",
        row_index=0,
        dataset="CoPESD",
        original_id="original",
        video_id="video-1",
        requested_timestamp=timestamp,
        candidate_ids=tuple(f"candidate-{index}" for index in range(candidate_count)),
        boxes_xyxy=tuple(
            tuple(float(value) for value in box * 100.0) for box in boxes
        ),
        ranks=tuple(range(candidate_count)),
        q_last=np.ones(3584, dtype=np.float16),
        roi_final=np.ones((candidate_count, 3584), dtype=np.float16),
        roi_l31=np.ones((candidate_count, 1280), dtype=np.float16),
        roi_l23=np.ones((candidate_count, 1280), dtype=np.float16),
        structure=structure,
        missing=np.zeros((candidate_count, 12), dtype=np.float32),
        ious=np.asarray([0.1, 0.8, 0.2, 0.3][:candidate_count], dtype=np.float32),
        native_scores=np.asarray(
            [0.9, 0.8, 0.7, 0.6][:candidate_count], dtype=np.float32
        ),
        frame_size=(100, 100),
    )


def _model_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    return {
        "q_last": torch.randn(2, 3584, generator=generator),
        "roi_final": torch.randn(2, 4, 3584, generator=generator),
        "roi_l31": torch.randn(2, 4, 1280, generator=generator),
        "structure": torch.randn(2, 4, 12, generator=generator),
        "missing": torch.zeros(2, 4, 12),
        "padding_mask": torch.tensor(
            [[False, False, False, False], [False, False, False, True]]
        ),
    }


def test_proposal_target_store_is_train_only_and_hash_checked(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    with pytest.raises(PermissionError, match="only for phase='train'"):
        ProposalTargetStore(path, phase="val")
    store = ProposalTargetStore(path, phase="train")
    store.write(
        {
            "present": {
                "has_target": True,
                "gt_box_xyxy_norm": (0.1, 0.2, 0.5, 0.7),
            },
            "missing": {"has_target": False, "gt_box_xyxy_norm": None},
        },
        source_split_hash="split-a",
    )

    loaded = store.read(expected_source_split_hash="split-a")

    assert loaded["present"]["gt_box_xyxy_norm"] == (0.1, 0.2, 0.5, 0.7)
    assert loaded["missing"]["has_target"] is False
    with pytest.raises(ValueError, match="source split hash mismatch"):
        store.read(expected_source_split_hash="split-b")


def test_target_builder_uses_exact_timestamp_and_keeps_feature_group_gt_free(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train.json"
    split.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "struc_info": [{"bbox_dict": {"1": [10, 20, 50, 70]}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    groups = (_group(timestamp=1.0), _group(timestamp=2.0))
    target_path = tmp_path / "proposal_targets.json"

    audit = build_proposal_target_store(
        groups,
        train_split=split,
        output_path=target_path,
    )
    targets = load_proposal_targets(
        groups,
        target_store=target_path,
        expected_source_split_hash=sha256_file(split),
    )

    assert audit["target_count"] == 1
    assert audit["missing_target_count"] == 1
    assert targets[groups[0].group_key].gt_box_xyxy_norm == (0.1, 0.2, 0.5, 0.7)
    assert targets[groups[1].group_key] == ProposalTarget(False, None)
    assert not hasattr(groups[0], "gt_box_xyxy_norm")


def test_cholec_target_builder_uses_fixed_annotation_canvas(tmp_path: Path) -> None:
    split = tmp_path / "train.json"
    split.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "struc_info": [
                        {"bbox_dict": {"1": [206.0, 9.0, 303.0, 83.0]}}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    group = replace(
        _group(timestamp=1.0),
        dataset="CholecTrack20",
        frame_size=(1920, 1080),
    )
    target_path = tmp_path / "proposal_targets.json"

    build_proposal_target_store(
        (group,),
        train_split=split,
        output_path=target_path,
    )
    target = load_proposal_targets(
        (group,),
        target_store=target_path,
        expected_source_split_hash=sha256_file(split),
    )[group.group_key]

    assert target.gt_box_xyxy_norm == pytest.approx(
        (206.0 / 854.0, 9.0 / 480.0, 303.0 / 854.0, 83.0 / 480.0)
    )


def test_proposal_adapter_parameter_cap_and_permutation_equivariance() -> None:
    config = ProposalAdapterConfig(dropout=0.0)
    model = QueryConditionedProposalAdapter(config).eval()
    inputs = _model_inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(inputs)
    for key in ("roi_final", "roi_l31", "structure", "missing", "padding_mask"):
        permuted[key] = inputs[key][:, permutation]

    with torch.inference_mode():
        expected = model(**inputs)
        actual = model(**permuted)

    assert sum(parameter.numel() for parameter in model.parameters()) <= 1_500_000
    torch.testing.assert_close(
        actual.original_quality_logits,
        expected.original_quality_logits[:, permutation],
    )
    torch.testing.assert_close(
        actual.refined_quality_logits,
        expected.refined_quality_logits[:, permutation],
    )
    torch.testing.assert_close(
        actual.bounded_deltas,
        expected.bounded_deltas[:, permutation],
    )


@pytest.mark.parametrize(
    ("masked_branch", "input_name"),
    (("final", "roi_final"), ("l31", "roi_l31")),
)
def test_individual_visual_branch_masks_are_invariant_to_masked_input(
    masked_branch: str,
    input_name: str,
) -> None:
    model = QueryConditionedProposalAdapter(
        ProposalAdapterConfig(dropout=0.0)
    ).eval()
    inputs = _model_inputs()
    modified = dict(inputs)
    modified[input_name] = inputs[input_name] * 37.0 - 11.0
    mask = ProposalAdapterInputMask(
        use_final=masked_branch != "final",
        use_l31=masked_branch != "l31",
    )

    with torch.inference_mode():
        expected = model(**inputs, input_mask=mask)
        actual = model(**modified, input_mask=mask)

    torch.testing.assert_close(
        actual.original_quality_logits, expected.original_quality_logits
    )
    torch.testing.assert_close(
        actual.refined_quality_logits, expected.refined_quality_logits
    )
    torch.testing.assert_close(actual.bounded_deltas, expected.bounded_deltas)


def test_legacy_roi_mask_still_disables_both_visual_branches() -> None:
    model = QueryConditionedProposalAdapter(
        ProposalAdapterConfig(dropout=0.0)
    ).eval()
    inputs = _model_inputs()
    modified = dict(inputs)
    modified["roi_final"] = inputs["roi_final"] * 17.0
    modified["roi_l31"] = inputs["roi_l31"] * -23.0
    mask = ProposalAdapterInputMask(use_roi=False)

    with torch.inference_mode():
        expected = model(**inputs, input_mask=mask)
        actual = model(**modified, input_mask=mask)

    torch.testing.assert_close(
        actual.original_quality_logits, expected.original_quality_logits
    )
    torch.testing.assert_close(actual.bounded_deltas, expected.bounded_deltas)



def test_bounded_delta_decode_is_valid_and_rejects_invalid_box() -> None:
    boxes = torch.tensor([[[0.1, 0.1, 0.3, 0.3]]])
    deltas = torch.tensor([[[0.5, -0.5, np.log(2.0), -np.log(2.0)]]])

    decoded = decode_bounded_box_deltas(boxes, deltas)

    assert torch.all(decoded >= 0.0)
    assert torch.all(decoded <= 1.0)
    assert torch.all(decoded[..., 2:] > decoded[..., :2])
    invalid = boxes.clone()
    invalid[..., 2] = invalid[..., 0]
    with pytest.raises(ValueError, match="invalid normalized boxes"):
        decode_bounded_box_deltas(invalid, deltas)


def test_proposal_loss_skips_missing_target_group_and_has_finite_gradients() -> None:
    config = ProposalAdapterConfig(dropout=0.0)
    model = QueryConditionedProposalAdapter(config)
    inputs = _model_inputs()
    output = model(**inputs)
    boxes = torch.tensor(
        [
            [
                [0.05, 0.05, 0.25, 0.25],
                [0.30, 0.10, 0.55, 0.40],
                [0.10, 0.50, 0.40, 0.85],
                [0.60, 0.55, 0.90, 0.90],
            ],
            [
                [0.05, 0.05, 0.25, 0.25],
                [0.30, 0.10, 0.55, 0.40],
                [0.10, 0.50, 0.40, 0.85],
                [0.60, 0.55, 0.90, 0.90],
            ],
        ]
    )
    ious = torch.tensor([[0.1, 0.8, 0.2, 0.3], [0.0, 0.0, 0.0, 0.0]])

    loss, parts = proposal_adapter_loss(
        output,
        original_boxes_xyxy=boxes,
        original_ious=ious,
        gt_boxes_xyxy=torch.tensor(
            [[0.25, 0.15, 0.60, 0.55], [0.0, 0.0, 0.0, 0.0]]
        ),
        has_target=torch.tensor([True, False]),
        padding_mask=inputs["padding_mask"],
        config=config,
    )

    assert torch.isfinite(loss)
    assert parts["target_group_count"].item() == 1
    assert parts["anchor_count"].item() == 4
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_quality_diversity_shortlist_keeps_native_anchors_and_caps_k() -> None:
    group = _group(candidate_count=4)
    config = ProposalAdapterConfig(
        dropout=0.0,
        regression_anchor_count=4,
        shortlist_k=6,
        native_anchor_count=2,
    )
    refined_boxes = np.asarray(group.structure[:, 6:10]).copy()
    refined_boxes[2] = (0.65, 0.05, 0.95, 0.35)

    selected = quality_diversity_shortlist(
        group,
        original_scores=(0.9, 0.8, 0.7, 0.6),
        refined_scores=(0.2, 0.3, 0.95, 0.4),
        refined_boxes_xyxy_norm=refined_boxes,
        config=config,
    )
    repeated = quality_diversity_shortlist(
        replace(group),
        original_scores=(0.9, 0.8, 0.7, 0.6),
        refined_scores=(0.2, 0.3, 0.95, 0.4),
        refined_boxes_xyxy_norm=refined_boxes,
        config=config,
    )

    assert tuple(item.candidate_id for item in selected[:2]) == (
        "candidate-0",
        "candidate-1",
    )
    assert len(selected) == 6
    assert any(item.candidate_id == "candidate-2::refined" for item in selected)
    assert selected == repeated
    assert all(
        0.0 <= value <= 1.0
        for item in selected
        for value in item.bbox_xyxy_norm
    )
