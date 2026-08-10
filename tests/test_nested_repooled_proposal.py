from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from refinerank.cache import (
    SpatialGridCacheReader,
    SpatialGridCacheWriter,
)
from refinerank.decode import build_source_indexed_stg_prediction_rows
from refinerank.proposal import ProposalCandidate, ProposalTarget
from refinerank.proposal_training import (
    fit_predict_proposal_split,
    predict_proposal_split_checkpoint,
)
from refinerank.repool import (
    pool_spatial_rois,
    pool_spatial_rois_with_l7,
    repool_proposals,
)
from refinerank.train_data import CachedTrainingGroup
from refinerank.training import OOFConfig
from refinerank.types import (
    CandidateGroup,
    FrozenCandidate,
    SpatialGridFeatures,
)


def _spatial() -> SpatialGridFeatures:
    l31 = np.zeros((1, 4, 4, 1280), dtype=np.float16)
    final = np.zeros((1, 2, 2, 3584), dtype=np.float16)
    for index in range(16):
        row, column = divmod(index, 4)
        l31[0, row, column, index] = 1.0
    for index in range(4):
        row, column = divmod(index, 2)
        final[0, row, column, index] = 1.0
    return SpatialGridFeatures(
        group_key="group@1.000",
        grid_l31=l31,
        grid_final=final,
        frame_size=(100, 100),
        grid_thw=(1, 4, 4),
    )


def _spatial_with_l7() -> SpatialGridFeatures:
    spatial = _spatial()
    l7 = np.flip(spatial.grid_l31, axis=2).copy()
    return SpatialGridFeatures(
        group_key=spatial.group_key,
        grid_l31=spatial.grid_l31,
        grid_final=spatial.grid_final,
        frame_size=spatial.frame_size,
        grid_thw=spatial.grid_thw,
        grid_l7=l7,
    )


def _cached_group() -> CachedTrainingGroup:
    box = (0.0, 0.0, 0.5, 0.5)
    roi_final, roi_l31 = pool_spatial_rois(_spatial(), (box,))
    structure = np.zeros((1, 12), dtype=np.float32)
    structure[0, :6] = np.arange(6, dtype=np.float32)
    structure[0, 6:10] = box
    structure[0, 10] = math.log(0.25)
    structure[0, 11] = 0.0
    return CachedTrainingGroup(
        group_key="group@1.000",
        sample_id="sample",
        row_index=0,
        dataset="CoPESD",
        original_id="original",
        video_id="video-1",
        clean_question="Where are the forceps?",
        requested_timestamp=1.0,
        candidate_ids=("parent",),
        boxes_xyxy=((0.0, 0.0, 50.0, 50.0),),
        ranks=(0,),
        q_last=np.ones(3584, dtype=np.float16),
        q_mean=np.arange(3584, dtype=np.float16),
        roi_final=roi_final,
        roi_l31=roi_l31,
        roi_l23=None,
        structure=structure,
        missing=np.zeros((1, 12), dtype=np.float32),
        ious=np.asarray((0.25,), dtype=np.float32),
        native_scores=np.asarray((0.9,), dtype=np.float32),
        frame_size=(100, 100),
    )


def _candidate_group() -> CandidateGroup:
    parent = _cached_group()
    return CandidateGroup(
        group_key=parent.group_key,
        sample_id=parent.sample_id,
        original_id=parent.original_id,
        video_id=parent.video_id,
        row_index=parent.row_index,
        dataset=parent.dataset,
        requested_timestamp=parent.requested_timestamp,
        clean_question="Where is the target?",
        frame_path=Path("unused.jpg"),
        frame_size=parent.frame_size,
        candidates=(
            FrozenCandidate(
                candidate_id="parent",
                candidate_time=1.0,
                bbox_xyxy=parent.boxes_xyxy[0],
                native_score=0.9,
                rank=0,
                source="cached_dino",
                structure=tuple(float(value) for value in parent.structure[0]),
                missing=(0.0,) * 12,
            ),
        ),
    )


def test_spatial_grid_cache_is_gt_free_hash_checked_and_source_checked(
    tmp_path: Path,
) -> None:
    group = _candidate_group()
    writer = SpatialGridCacheWriter(
        tmp_path / "spatial",
        {"candidate_pool_hash": "pool-a", "phase": "train"},
    )
    path = writer.write_group(group, _spatial())
    writer.write_index((group,), {group.group_key: path})

    reader = SpatialGridCacheReader(
        tmp_path / "spatial",
        expected_sources={"candidate_pool_hash": "pool-a"},
    )
    actual = reader.read_group(group.group_key)
    np.testing.assert_array_equal(actual.grid_l31, _spatial().grid_l31)
    with pytest.raises(ValueError, match="source mismatch"):
        SpatialGridCacheReader(
            tmp_path / "spatial",
            expected_sources={"candidate_pool_hash": "pool-b"},
        )
    with pytest.raises(ValueError, match="forbidden"):
        SpatialGridCacheWriter(tmp_path / "bad", {"gt": "must-not-enter"})
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.read_group(group.group_key)


def test_exact_repool_reconstructs_original_and_moves_refined_features() -> None:
    group = _cached_group()
    proposals = {
        group.group_key: (
            ProposalCandidate(
                candidate_id="parent",
                bbox_xyxy_norm=(0.0, 0.0, 0.5, 0.5),
                bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
                score=0.8,
                rank=0,
                source_index=0,
                refined=False,
            ),
            ProposalCandidate(
                candidate_id="parent::refined",
                bbox_xyxy_norm=(0.5, 0.5, 1.0, 1.0),
                bbox_xyxy=(50.0, 50.0, 100.0, 100.0),
                score=0.7,
                rank=0,
                source_index=0,
                refined=True,
            ),
        )
    }
    target = ProposalTarget(True, (0.5, 0.5, 1.0, 1.0))

    result = repool_proposals(
        (group,),
        proposals,
        {group.group_key: target},
        {group.group_key: _spatial()},
    )

    assert result.audit["original_reconstruction_passed"] is True
    assert result.audit["refined_visual_change_passed"] is True
    assert result.audit["refined_visual_changed_fraction_min"] == 0.95
    actual = result.groups[0]
    np.testing.assert_array_equal(actual.structure[1, :6], group.structure[0, :6])
    np.testing.assert_allclose(actual.structure[1, 6:10], (0.5, 0.5, 1.0, 1.0))
    assert actual.structure[1, 10] == pytest.approx(math.log(0.25))
    assert actual.structure[1, 11] == pytest.approx(0.0)
    assert actual.ious.tolist() == pytest.approx([0.0, 1.0])
    assert not np.array_equal(actual.roi_l31[0], actual.roi_l31[1])
    assert actual.parent_source_indices == (0, 0)
    assert actual.refined_flags == (False, True)
    assert actual.q_mean is not None
    np.testing.assert_array_equal(actual.q_mean, group.q_mean)
    assert actual.clean_question == group.clean_question


def test_exact_repool_uses_optional_l7_spatial_grid() -> None:
    spatial = _spatial_with_l7()
    group = _cached_group()
    _final, _l31, parent_l7 = pool_spatial_rois_with_l7(
        spatial, ((0.0, 0.0, 0.5, 0.5),)
    )
    assert parent_l7 is not None
    group = CachedTrainingGroup(
        **{
            **group.__dict__,
            "roi_l7": parent_l7,
        }
    )
    proposals = {
        group.group_key: (
            ProposalCandidate(
                candidate_id="parent",
                bbox_xyxy_norm=(0.0, 0.0, 0.5, 0.5),
                bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
                score=0.8,
                rank=0,
                source_index=0,
                refined=False,
            ),
            ProposalCandidate(
                candidate_id="parent::refined",
                bbox_xyxy_norm=(0.5, 0.5, 1.0, 1.0),
                bbox_xyxy=(50.0, 50.0, 100.0, 100.0),
                score=0.7,
                rank=0,
                source_index=0,
                refined=True,
            ),
        )
    }

    result = repool_proposals(
        (group,),
        proposals,
        {group.group_key: ProposalTarget(True, (0.5, 0.5, 1.0, 1.0))},
        {group.group_key: spatial},
    )

    assert result.audit["exact_l7_coverage"] == 1.0
    assert result.groups[0].roi_l7 is not None
    np.testing.assert_array_equal(result.groups[0].roi_l7[0], parent_l7[0])
    assert not np.array_equal(
        result.groups[0].roi_l7[0], result.groups[0].roi_l7[1]
    )


def test_exact_repool_copies_source_parent_l23_to_all_children() -> None:
    parent_l23 = np.arange(1280, dtype=np.float16)[None, :]
    group = CachedTrainingGroup(
        **{
            **_cached_group().__dict__,
            "roi_l23": parent_l23,
        }
    )
    proposals = {
        group.group_key: (
            ProposalCandidate(
                candidate_id="parent",
                bbox_xyxy_norm=(0.0, 0.0, 0.5, 0.5),
                bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
                score=0.8,
                rank=0,
                source_index=0,
                refined=False,
            ),
            ProposalCandidate(
                candidate_id="parent::refined",
                bbox_xyxy_norm=(0.5, 0.5, 1.0, 1.0),
                bbox_xyxy=(50.0, 50.0, 100.0, 100.0),
                score=0.7,
                rank=0,
                source_index=0,
                refined=True,
            ),
        )
    }

    result = repool_proposals(
        (group,),
        proposals,
        {group.group_key: ProposalTarget(True, (0.5, 0.5, 1.0, 1.0))},
        {group.group_key: _spatial()},
    )

    assert result.audit["source_parent_l23_coverage"] == 1.0
    assert result.groups[0].roi_l23 is not None
    np.testing.assert_array_equal(result.groups[0].roi_l23[0], parent_l23[0])
    np.testing.assert_array_equal(result.groups[0].roi_l23[1], parent_l23[0])


def test_spatial_pool_rejects_invalid_box() -> None:
    with pytest.raises(ValueError, match="invalid xyxy"):
        pool_spatial_rois(_spatial(), ((0.5, 0.5, 0.5, 1.0),))


def test_explicit_proposal_split_rejects_video_id_leakage() -> None:
    group = _cached_group()
    second = CachedTrainingGroup(
        **{
            **group.__dict__,
            "group_key": "group@2.000",
            "requested_timestamp": 2.0,
        }
    )
    targets = {
        group.group_key: ProposalTarget(True, (0.0, 0.0, 0.5, 0.5)),
        second.group_key: ProposalTarget(True, (0.0, 0.0, 0.5, 0.5)),
    }
    with pytest.raises(ValueError, match="video_id"):
        fit_predict_proposal_split(
            (group, second),
            targets,
            train_indices=(0,),
            test_indices=(1,),
            oof_config=OOFConfig(epochs=0),
            model_seed=1,
            batch_seed_base=1,
            device="cpu",
        )


def test_split_checkpoint_replays_exact_heldout_proposals(tmp_path: Path) -> None:
    base = _cached_group()
    datasets = ("CholecTrack20", "EgoSurgery", "CoPESD") * 2
    groups = tuple(
        CachedTrainingGroup(
            **{
                **base.__dict__,
                "group_key": f"split-{index}@1.000",
                "sample_id": f"split-{index}",
                "row_index": index,
                "dataset": dataset,
                "original_id": f"split-{index}",
                "video_id": f"video-{index}",
            }
        )
        for index, dataset in enumerate(datasets)
    )
    targets = {
        group.group_key: ProposalTarget(True, (0.0, 0.0, 0.5, 0.5))
        for group in groups
    }
    checkpoint = tmp_path / "fold.pt"
    fitted = fit_predict_proposal_split(
        groups,
        targets,
        train_indices=(0, 1, 2),
        test_indices=(3, 4, 5),
        oof_config=OOFConfig(
            epochs=1,
            groups_per_dataset=1,
            eval_batch_size=3,
        ),
        model_seed=43,
        batch_seed_base=47,
        split_checkpoint_path=checkpoint,
        split_checkpoint_metadata={
            "source_feature_contract_hash": "contract",
            "source_input_hashes": {"feature_manifest": "abc"},
            "outer_fold": 0,
        },
        device="cpu",
    )

    replayed = predict_proposal_split_checkpoint(
        groups,
        checkpoint,
        test_indices=(3, 4, 5),
        expected_source_feature_contract_hash="contract",
        expected_input_hashes={"feature_manifest": "abc"},
        device="cpu",
    )

    assert replayed.proposals == fitted.proposals
    assert replayed.original_quality_scores == fitted.original_quality_scores
    assert replayed.training_trace == fitted.training_trace


def test_public_cholec_output_preserves_decoded_image_canvas() -> None:
    annotation_box = (206.0, 9.0, 303.0, 83.0)
    image_box = (
        annotation_box[0] * 1920.0 / 854.0,
        annotation_box[1] * 1080.0 / 480.0,
        annotation_box[2] * 1920.0 / 854.0,
        annotation_box[3] * 1080.0 / 480.0,
    )
    group = CachedTrainingGroup(
        **{
            **_cached_group().__dict__,
            "dataset": "CholecTrack20",
            "frame_size": (1920, 1080),
            "boxes_xyxy": (image_box,),
        }
    )

    rows, audits = build_source_indexed_stg_prediction_rows(
        (group,),
        {group.group_key: (1.0,)},
    )

    assert rows[0]["prediction"] == (
        "1.0 seconds: [463.14, 20.25, 681.22, 186.75]"
    )
    assert audits[0]["output_coordinate_space"] == "decoded_image_canvas"
    assert audits[0]["image_frame_sizes"] == [(1920, 1080)]
    assert audits[0]["image_frame_sizes_by_timestamp"] == {
        "1.0": [1920, 1080]
    }


def test_public_output_snaps_small_detector_boundary_drift() -> None:
    group = CachedTrainingGroup(
        **{
            **_cached_group().__dict__,
            "dataset": "CholecTrack20",
            "frame_size": (1920, 1080),
            "boxes_xyxy": ((-2.0, 0.0, 1920.5, 1080.0),),
        }
    )

    rows, _audits = build_source_indexed_stg_prediction_rows(
        (group,),
        {group.group_key: (1.0,)},
    )

    assert rows[0]["prediction"] == "1.0 seconds: [0, 0, 1920, 1080]"


def test_public_cholec_output_rejects_incompatible_image_canvas() -> None:
    group = CachedTrainingGroup(
        **{
            **_cached_group().__dict__,
            "dataset": "CholecTrack20",
            "frame_size": (100, 100),
        }
    )

    with pytest.raises(ValueError, match="aspect ratio is incompatible"):
        build_source_indexed_stg_prediction_rows(
            (group,),
            {group.group_key: (1.0,)},
        )


def test_public_cholec_output_rejects_mixed_frame_sizes_per_row() -> None:
    first = CachedTrainingGroup(
        **{
            **_cached_group().__dict__,
            "dataset": "CholecTrack20",
            "frame_size": (854, 480),
        }
    )
    second = CachedTrainingGroup(
        **{
            **first.__dict__,
            "group_key": "group@2.000",
            "requested_timestamp": 2.0,
            "frame_size": (1920, 1080),
            "boxes_xyxy": ((0.0, 0.0, 960.0, 540.0),),
        }
    )

    with pytest.raises(ValueError, match="requires one decoded frame size"):
        build_source_indexed_stg_prediction_rows(
            (first, second),
            {first.group_key: (1.0,), second.group_key: (1.0,)},
        )


def test_public_output_keeps_duplicate_original_ids_source_indexed() -> None:
    first = _cached_group()
    second = CachedTrainingGroup(
        **{
            **first.__dict__,
            "group_key": "other@1.000",
            "sample_id": "other-sample",
            "row_index": 1,
        }
    )

    rows, _audits = build_source_indexed_stg_prediction_rows(
        (first, second),
        {first.group_key: (1.0,), second.group_key: (1.0,)},
    )

    assert [row["source_index"] for row in rows] == [0, 1]
    assert [row["id"] for row in rows] == ["sample", "other-sample"]


def test_public_output_rejects_duplicate_source_indices() -> None:
    first = _cached_group()
    second = CachedTrainingGroup(
        **{
            **first.__dict__,
            "group_key": "other@1.000",
            "sample_id": "other-sample",
        }
    )

    with pytest.raises(RuntimeError, match="duplicate source_index"):
        build_source_indexed_stg_prediction_rows(
            (first, second),
            {first.group_key: (1.0,), second.group_key: (1.0,)},
        )


def test_proposal_deployment_cli_uses_quality_scores_without_selector(
    tmp_path: Path,
) -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "refinerank" / "refinenet_cli.py"
    )
    spec = importlib.util.spec_from_file_location(
        "refinenet_cli_proposal_deployment",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load refinenet_cli")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    full_args = module._parser().parse_args(
        [
            "--exp-dir",
            str(tmp_path / "full"),
            "proposal-full",
            "--feature-cache",
            "features",
            "--candidate-store",
            "candidates.jsonl",
            "--label-store",
            "labels.json",
            "--operator-authorized",
        ]
    )
    predict_args = module._parser().parse_args(
        [
            "--exp-dir",
            str(tmp_path / "predict"),
            "proposal-predict",
            "--feature-cache",
            "public-features",
            "--candidate-store",
            "public-candidates.jsonl",
            "--spatial-grid-cache",
            "public-spatial",
            "--deployment-manifest",
            "deployment.json",
            "--proposal-checkpoint",
            "proposal.pt",
            "--require-public-test",
        ]
    )
    group = CachedTrainingGroup(
        **{
            **_cached_group().__dict__,
            "native_scores": (0.75,),
        }
    )

    assert full_args.command == "proposal-full"
    assert full_args.operator_authorized is True
    assert predict_args.command == "proposal-predict"
    assert predict_args.require_public_test is True
    assert module._proposal_quality_scores((group,)) == {
        group.group_key: (0.75,)
    }
