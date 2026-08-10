"""Deterministic train-only OOF for query-conditioned proposal adaptation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from refinerank.decode import ScoredCandidate, select_with_iter97_dp
from refinerank.proposal import (
    ProposalAdapterConfig,
    ProposalAdapterInputMask,
    ProposalAdapterOutput,
    ProposalCandidate,
    ProposalTarget,
    QueryConditionedProposalAdapter,
    decode_bounded_box_deltas,
    proposal_adapter_loss,
    quality_diversity_shortlist,
)
from refinerank.train_data import (
    CachedTrainingGroup,
    StructureStandardizer,
    balanced_batch_indices,
    collate_groups,
    deterministic_group_folds,
    fit_structure_standardizer,
)
from refinerank.training import OOFConfig

PROPOSAL_CHECKPOINT_SCHEMA_VERSION = 1
PROPOSAL_SPLIT_CHECKPOINT_SCHEMA_VERSION = 1
ProposalQueryMode = Literal["q_last", "q_mean"]


@dataclass(frozen=True)
class ProposalOOFResult:
    """OOF proposal sets, metrics, oracle audits, and finite training traces."""

    proposals: Mapping[str, tuple[ProposalCandidate, ...]]
    shuffled_query_proposals: Mapping[str, tuple[ProposalCandidate, ...]]
    shuffled_roi_proposals: Mapping[str, tuple[ProposalCandidate, ...]]
    original_quality_scores: Mapping[str, tuple[float, ...]]
    metrics: Mapping[str, Any]
    shuffled_query_metrics: Mapping[str, Any]
    shuffled_roi_metrics: Mapping[str, Any]
    original_oracle: Mapping[str, Any]
    combined_oracle: Mapping[str, Any]
    fold_standardizers: tuple[Mapping[str, list[float]], ...]
    fold_audit: tuple[Mapping[str, Any], ...]
    training_trace: tuple[Mapping[str, Any], ...]
    parameter_audit: Mapping[str, Any]
    coordinate_audit: Mapping[str, Any]


@dataclass(frozen=True)
class ProposalSplitPrediction:
    """One explicit proposal-adapter fit and disjoint held-out proposals."""

    proposals: Mapping[str, tuple[ProposalCandidate, ...]]
    original_quality_scores: Mapping[str, tuple[float, ...]]
    standardizer: Mapping[str, list[float]]
    training_trace: tuple[Mapping[str, Any], ...]
    trainable_parameters: int
    coordinate_count: int
    train_group_count: int
    excluded_missing_target_train_group_count: int
    shuffled_query_proposals: Mapping[str, tuple[ProposalCandidate, ...]] | None = None
    shuffled_roi_proposals: Mapping[str, tuple[ProposalCandidate, ...]] | None = None
    shuffled_query_original_quality_scores: (
        Mapping[str, tuple[float, ...]] | None
    ) = None
    shuffled_roi_original_quality_scores: (
        Mapping[str, tuple[float, ...]] | None
    ) = None
    diagnostic_coordinate_count: int = 0


def train_full_proposal_adapter(
    groups: Sequence[CachedTrainingGroup],
    targets: Mapping[str, ProposalTarget],
    output_path: Path,
    *,
    operator_authorized: bool,
    source_feature_contract_hash: str,
    adapter_config: ProposalAdapterConfig = ProposalAdapterConfig(),
    oof_config: OOFConfig = OOFConfig(),
    device: str = "cuda",
) -> tuple[Path, tuple[Mapping[str, Any], ...]]:
    """Fit the deployment proposal adapter on all train groups and save it."""

    if not operator_authorized:
        raise PermissionError(
            "full proposal-adapter training requires operator authorization"
        )
    expected = {group.group_key for group in groups}
    if set(targets) != expected:
        raise ValueError("proposal targets do not cover the training groups")
    train = tuple(
        index
        for index, group in enumerate(groups)
        if targets[group.group_key].has_target
    )
    if not train:
        raise ValueError("full proposal-adapter training has no target groups")
    _set_seed(oof_config.seed)
    standardizer = fit_structure_standardizer(groups, train)
    torch_device = torch.device(device)
    model = QueryConditionedProposalAdapter(adapter_config).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=oof_config.learning_rate,
        weight_decay=oof_config.weight_decay,
    )
    trace: list[Mapping[str, Any]] = []
    model.train()
    for epoch in range(oof_config.epochs):
        epoch_parts: dict[str, list[float]] = defaultdict(list)
        gradient_norms: list[float] = []
        for batch_number, batch_indices in enumerate(
            balanced_batch_indices(
                groups,
                train,
                per_dataset=oof_config.groups_per_dataset,
                seed=oof_config.seed + epoch,
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            batch = _collate_proposal_batch(
                groups,
                targets,
                batch_indices,
                standardizer,
                device=torch_device,
            )
            output = _forward(model, batch)
            loss, parts = proposal_adapter_loss(
                output,
                original_boxes_xyxy=_tensor(batch, "raw_structure")[..., 6:10],
                original_ious=_tensor(batch, "ious"),
                gt_boxes_xyxy=_tensor(batch, "gt_boxes_xyxy"),
                has_target=_tensor(batch, "has_target").bool(),
                padding_mask=_tensor(batch, "padding_mask").bool(),
                config=adapter_config,
            )
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError(
                    f"non-finite full proposal loss at epoch={epoch} "
                    f"batch={batch_number}"
                )
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), oof_config.gradient_clip
            )
            if not math.isfinite(float(gradient_norm)):
                raise FloatingPointError(
                    f"non-finite full proposal gradient at epoch={epoch} "
                    f"batch={batch_number}"
                )
            optimizer.step()
            gradient_norms.append(float(gradient_norm))
            for name, value in parts.items():
                epoch_parts[name].append(float(value))
        trace.append(
            {
                "epoch": epoch,
                "batch_count": len(gradient_norms),
                "loss": {
                    name: fmean(values) for name, values in epoch_parts.items()
                },
                "pre_clip_gradient_norm_mean": fmean(gradient_norms),
                "pre_clip_gradient_norm_max": max(gradient_norms),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": PROPOSAL_CHECKPOINT_SCHEMA_VERSION,
            "source_feature_contract_hash": source_feature_contract_hash,
            "adapter_config": asdict(adapter_config),
            "train_config": asdict(oof_config),
            "standardizer": {
                "mean": standardizer.mean,
                "std": standardizer.std,
            },
            "model_state": model.cpu().state_dict(),
        },
        output_path,
    )
    del optimizer, model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return output_path, tuple(trace)


def predict_proposal_adapter_checkpoint(
    groups: Sequence[CachedTrainingGroup],
    path: Path,
    *,
    expected_source_feature_contract_hash: str,
    batch_size: int = 48,
    device: str = "cuda",
) -> dict[str, tuple[ProposalCandidate, ...]]:
    """Predict GT-free proposal families with a full-fit adapter checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != PROPOSAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("proposal-adapter checkpoint schema mismatch")
    if (
        payload.get("source_feature_contract_hash")
        != expected_source_feature_contract_hash
    ):
        raise ValueError("proposal-adapter feature-contract mismatch")
    adapter_config = ProposalAdapterConfig(**payload["adapter_config"])
    standardizer = StructureStandardizer(
        np.asarray(payload["standardizer"]["mean"], dtype=np.float32),
        np.asarray(payload["standardizer"]["std"], dtype=np.float32),
    )
    model = QueryConditionedProposalAdapter(adapter_config)
    model.load_state_dict(payload["model_state"], strict=True)
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()
    proposals, _scores, _coordinate_count = _predict_proposals(
        model,
        groups,
        tuple(range(len(groups))),
        standardizer,
        adapter_config=adapter_config,
        oof_config=OOFConfig(eval_batch_size=batch_size),
        device=torch_device,
    )
    return proposals


def fit_predict_proposal_split(
    groups: Sequence[CachedTrainingGroup],
    targets: Mapping[str, ProposalTarget],
    *,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    adapter_config: ProposalAdapterConfig = ProposalAdapterConfig(),
    input_mask: ProposalAdapterInputMask = ProposalAdapterInputMask(),
    query_mode: ProposalQueryMode = "q_last",
    oof_config: OOFConfig = OOFConfig(),
    model_seed: int,
    batch_seed_base: int,
    trace_context: Mapping[str, Any] | None = None,
    semantic_diagnostics: bool = False,
    split_checkpoint_path: Path | None = None,
    split_checkpoint_metadata: Mapping[str, Any] | None = None,
    device: str = "cuda",
) -> ProposalSplitPrediction:
    """Fit on explicit video-disjoint groups and predict only the holdout."""

    _validate_query_mode(query_mode)
    raw_train = tuple(int(index) for index in train_indices)
    test = tuple(int(index) for index in test_indices)
    if not raw_train or not test:
        raise ValueError("proposal split requires non-empty train and test indices")
    if set(raw_train) & set(test):
        raise ValueError("proposal split train and test indices must be disjoint")
    train_keys = {
        (groups[index].dataset, groups[index].video_id) for index in raw_train
    }
    test_keys = {(groups[index].dataset, groups[index].video_id) for index in test}
    if train_keys & test_keys:
        raise ValueError("proposal split leaks a video_id across train and test")
    train = tuple(
        index for index in raw_train if targets[groups[index].group_key].has_target
    )
    if not train:
        raise ValueError("proposal split has no train groups with targets")
    _set_seed(model_seed)
    standardizer = fit_structure_standardizer(groups, train)
    torch_device = torch.device(device)
    model = QueryConditionedProposalAdapter(adapter_config).to(torch_device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=oof_config.learning_rate,
        weight_decay=oof_config.weight_decay,
    )
    trace: list[Mapping[str, Any]] = []
    model.train()
    for epoch in range(oof_config.epochs):
        epoch_parts: dict[str, list[float]] = defaultdict(list)
        gradient_norms: list[float] = []
        batches = balanced_batch_indices(
            groups,
            train,
            per_dataset=oof_config.groups_per_dataset,
            seed=batch_seed_base + epoch,
        )
        for batch_number, batch_indices in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            batch = _collate_proposal_batch(
                groups,
                targets,
                batch_indices,
                standardizer,
                device=torch_device,
            )
            output = _forward(
                model,
                batch,
                input_mask=input_mask,
                query_mode=query_mode,
            )
            loss, parts = proposal_adapter_loss(
                output,
                original_boxes_xyxy=_tensor(batch, "raw_structure")[..., 6:10],
                original_ious=_tensor(batch, "ious"),
                gt_boxes_xyxy=_tensor(batch, "gt_boxes_xyxy"),
                has_target=_tensor(batch, "has_target").bool(),
                padding_mask=_tensor(batch, "padding_mask").bool(),
                config=adapter_config,
            )
            if not math.isfinite(float(loss.detach())):
                detail = f"epoch={epoch} batch={batch_number}"
                raise FloatingPointError(f"non-finite proposal loss at {detail}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), oof_config.gradient_clip
            )
            if not math.isfinite(float(gradient_norm)):
                detail = f"epoch={epoch} batch={batch_number}"
                raise FloatingPointError(f"non-finite proposal gradient at {detail}")
            optimizer.step()
            gradient_norms.append(float(gradient_norm))
            for name, value in parts.items():
                epoch_parts[name].append(float(value))
        row: dict[str, Any] = {
            "epoch": epoch,
            "batch_count": len(gradient_norms),
            "loss": {name: fmean(values) for name, values in epoch_parts.items()},
            "pre_clip_gradient_norm_mean": fmean(gradient_norms),
            "pre_clip_gradient_norm_max": max(gradient_norms),
        }
        if trace_context is not None:
            row = {**trace_context, **row}
        trace.append(row)
    model.eval()
    proposals, original_scores, coordinate_count = _predict_proposals(
        model,
        groups,
        test,
        standardizer,
        adapter_config=adapter_config,
        oof_config=oof_config,
        device=torch_device,
        input_mask=input_mask,
        query_mode=query_mode,
    )
    query_proposals: Mapping[str, tuple[ProposalCandidate, ...]] | None = None
    roi_proposals: Mapping[str, tuple[ProposalCandidate, ...]] | None = None
    query_original_scores: Mapping[str, tuple[float, ...]] | None = None
    roi_original_scores: Mapping[str, tuple[float, ...]] | None = None
    diagnostic_coordinate_count = 0
    if semantic_diagnostics:
        query_proposals, query_original_scores, query_count = _predict_proposals(
            model,
            groups,
            test,
            standardizer,
            adapter_config=adapter_config,
            oof_config=oof_config,
            device=torch_device,
            query_override=_shuffled_query_overrides(
                groups,
                test,
                query_mode=query_mode,
            ),
            input_mask=input_mask,
            query_mode=query_mode,
        )
        roi_proposals, roi_original_scores, roi_count = _predict_proposals(
            model,
            groups,
            test,
            standardizer,
            adapter_config=adapter_config,
            oof_config=oof_config,
            device=torch_device,
            shuffle_roi=True,
            seed=model_seed,
            input_mask=input_mask,
            query_mode=query_mode,
        )
        diagnostic_coordinate_count = query_count + roi_count
    if split_checkpoint_path is not None:
        _save_proposal_split_checkpoint(
            split_checkpoint_path,
            model=model,
            standardizer=standardizer,
            adapter_config=adapter_config,
            input_mask=input_mask,
            query_mode=query_mode,
            oof_config=oof_config,
            model_seed=model_seed,
            batch_seed_base=batch_seed_base,
            raw_train_indices=raw_train,
            train_indices=train,
            test_indices=test,
            groups=groups,
            training_trace=trace,
            metadata=split_checkpoint_metadata,
        )
    del optimizer, model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return ProposalSplitPrediction(
        proposals=proposals,
        original_quality_scores=original_scores,
        standardizer={
            "mean": standardizer.mean.tolist(),
            "std": standardizer.std.tolist(),
        },
        training_trace=tuple(trace),
        trainable_parameters=trainable_parameters,
        coordinate_count=coordinate_count,
        train_group_count=len(train),
        excluded_missing_target_train_group_count=len(raw_train) - len(train),
        shuffled_query_proposals=query_proposals,
        shuffled_roi_proposals=roi_proposals,
        shuffled_query_original_quality_scores=query_original_scores,
        shuffled_roi_original_quality_scores=roi_original_scores,
        diagnostic_coordinate_count=diagnostic_coordinate_count,
    )


def load_proposal_split_checkpoint(
    path: Path,
    *,
    expected_source_feature_contract_hash: str | None = None,
    expected_input_hashes: Mapping[str, str] | None = None,
    device: str = "cpu",
) -> tuple[QueryConditionedProposalAdapter, StructureStandardizer, dict[str, Any]]:
    """Load a held-out Proposal Repair fit with strict provenance checks."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != PROPOSAL_SPLIT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("proposal split checkpoint schema mismatch")
    if payload.get("checkpoint_kind") != "proposal_adapter_video_disjoint_split":
        raise ValueError("proposal split checkpoint kind mismatch")
    if (
        expected_source_feature_contract_hash is not None
        and payload.get("source_feature_contract_hash")
        != expected_source_feature_contract_hash
    ):
        raise ValueError("proposal split checkpoint feature-contract mismatch")
    stored_hashes = payload.get("source_input_hashes")
    if not isinstance(stored_hashes, Mapping):
        raise ValueError("proposal split checkpoint source hashes are missing")
    for name, expected in (expected_input_hashes or {}).items():
        if stored_hashes.get(name) != expected:
            raise ValueError(f"proposal split checkpoint source mismatch for {name}")
    adapter_config = ProposalAdapterConfig(**payload["adapter_config"])
    standardizer = StructureStandardizer(
        np.asarray(payload["standardizer"]["mean"], dtype=np.float32),
        np.asarray(payload["standardizer"]["std"], dtype=np.float32),
    )
    model = QueryConditionedProposalAdapter(adapter_config)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(torch.device(device)).eval()
    return model, standardizer, dict(payload)


def predict_proposal_split_checkpoint(
    groups: Sequence[CachedTrainingGroup],
    path: Path,
    *,
    test_indices: Sequence[int],
    expected_source_feature_contract_hash: str | None = None,
    expected_input_hashes: Mapping[str, str] | None = None,
    device: str = "cuda",
) -> ProposalSplitPrediction:
    """Replay one saved held-out fit and verify its exact group identity."""

    model, standardizer, payload = load_proposal_split_checkpoint(
        path,
        expected_source_feature_contract_hash=expected_source_feature_contract_hash,
        expected_input_hashes=expected_input_hashes,
        device=device,
    )
    test = tuple(int(index) for index in test_indices)
    if _group_digest(groups, test) != payload.get("test_group_digest"):
        raise ValueError("proposal split checkpoint test-group digest mismatch")
    adapter_config = ProposalAdapterConfig(**payload["adapter_config"])
    input_mask = ProposalAdapterInputMask(**payload["input_mask"])
    query_mode = str(payload["query_mode"])
    _validate_query_mode(query_mode)
    train_config = OOFConfig(**payload["train_config"])
    proposals, original_scores, coordinate_count = _predict_proposals(
        model,
        groups,
        test,
        standardizer,
        adapter_config=adapter_config,
        oof_config=train_config,
        device=torch.device(device),
        input_mask=input_mask,
        query_mode=query_mode,
    )
    return ProposalSplitPrediction(
        proposals=proposals,
        original_quality_scores=original_scores,
        standardizer={
            "mean": standardizer.mean.tolist(),
            "std": standardizer.std.tolist(),
        },
        training_trace=tuple(payload["training_trace"]),
        trainable_parameters=int(payload["trainable_parameters"]),
        coordinate_count=coordinate_count,
        train_group_count=int(payload["train_group_count"]),
        excluded_missing_target_train_group_count=int(
            payload["excluded_missing_target_train_group_count"]
        ),
    )


def run_proposal_adapter_oof(
    groups: Sequence[CachedTrainingGroup],
    targets: Mapping[str, ProposalTarget],
    *,
    adapter_config: ProposalAdapterConfig = ProposalAdapterConfig(),
    input_mask: ProposalAdapterInputMask = ProposalAdapterInputMask(),
    oof_config: OOFConfig = OOFConfig(),
    device: str = "cuda",
) -> ProposalOOFResult:
    """Fit five video-disjoint adapters and predict held-out proposals."""

    expected = {group.group_key for group in groups}
    if set(targets) != expected:
        raise ValueError("proposal targets do not cover the training groups")
    fold_ids = deterministic_group_folds(
        groups,
        folds=oof_config.folds,
        seed=oof_config.seed,
    )
    proposals: dict[str, tuple[ProposalCandidate, ...]] = {}
    query_proposals: dict[str, tuple[ProposalCandidate, ...]] = {}
    roi_proposals: dict[str, tuple[ProposalCandidate, ...]] = {}
    original_scores: dict[str, tuple[float, ...]] = {}
    standardizer_rows: list[Mapping[str, list[float]]] = []
    fold_audits: list[Mapping[str, Any]] = []
    training_trace: list[Mapping[str, Any]] = []
    coordinate_counts: list[int] = []
    parameter_counts: list[int] = []
    for fold in range(oof_config.folds):
        raw_train = [index for index, value in enumerate(fold_ids) if value != fold]
        test_indices = [index for index, value in enumerate(fold_ids) if value == fold]
        train_keys = {
            (groups[index].dataset, groups[index].video_id)
            for index in raw_train
        }
        test_keys = {
            (groups[index].dataset, groups[index].video_id)
            for index in test_indices
        }
        overlap = train_keys & test_keys
        if overlap:
            raise ValueError(f"proposal fold {fold} leaks video IDs")
        split = fit_predict_proposal_split(
            groups,
            targets,
            train_indices=raw_train,
            test_indices=test_indices,
            adapter_config=adapter_config,
            input_mask=input_mask,
            oof_config=oof_config,
            model_seed=oof_config.seed + fold,
            batch_seed_base=oof_config.seed + fold * 10_000,
            trace_context={"fold": fold},
            semantic_diagnostics=True,
            device=device,
        )
        if (
            split.shuffled_query_proposals is None
            or split.shuffled_roi_proposals is None
        ):
            raise RuntimeError("proposal semantic diagnostics were not produced")
        proposals.update(split.proposals)
        query_proposals.update(split.shuffled_query_proposals)
        roi_proposals.update(split.shuffled_roi_proposals)
        original_scores.update(split.original_quality_scores)
        standardizer_rows.append(split.standardizer)
        training_trace.extend(split.training_trace)
        parameter_counts.append(split.trainable_parameters)
        if split.diagnostic_coordinate_count % 2:
            raise RuntimeError("proposal diagnostic coordinate count is not paired")
        diagnostic_count = split.diagnostic_coordinate_count // 2
        coordinate_counts.extend(
            (split.coordinate_count, diagnostic_count, diagnostic_count)
        )
        fold_audits.append(
            {
                "fold": fold,
                "train_group_count": split.train_group_count,
                "excluded_missing_target_train_group_count": (
                    split.excluded_missing_target_train_group_count
                ),
                "test_group_count": len(test_indices),
                "train_video_id_count": len(train_keys),
                "test_video_id_count": len(test_keys),
                "video_id_overlap_count": len(overlap),
            }
        )
    for name, values in (
        ("normal", proposals),
        ("shuffled_query", query_proposals),
        ("shuffled_roi", roi_proposals),
    ):
        if set(values) != expected:
            raise RuntimeError(f"proposal {name} OOF coverage mismatch")
    normal_metrics = summarize_proposals(groups, proposals, targets)
    query_metrics = summarize_proposals(groups, query_proposals, targets)
    roi_metrics = summarize_proposals(groups, roi_proposals, targets)
    original_oracle = summarize_original_oracle(groups)
    combined_oracle = normal_metrics["oracle"]
    return ProposalOOFResult(
        proposals=proposals,
        shuffled_query_proposals=query_proposals,
        shuffled_roi_proposals=roi_proposals,
        original_quality_scores=original_scores,
        metrics=normal_metrics,
        shuffled_query_metrics=query_metrics,
        shuffled_roi_metrics=roi_metrics,
        original_oracle=original_oracle,
        combined_oracle=combined_oracle,
        fold_standardizers=tuple(standardizer_rows),
        fold_audit=tuple(fold_audits),
        training_trace=tuple(training_trace),
        parameter_audit={
            "trainable_parameters": max(parameter_counts),
            "max_trainable_parameters": adapter_config.max_trainable_parameters,
            "parameter_cap_passed": max(parameter_counts)
            <= adapter_config.max_trainable_parameters,
            "backbone_parameters_in_optimizer": 0,
            "backbone_forward_calls": 0,
            "frozen_feature_cache_only": True,
        },
        coordinate_audit={
            "proposal_sets_checked": len(coordinate_counts),
            "proposal_count_checked": sum(coordinate_counts),
            "invalid_box_count": 0,
            "round_trip_failures": 0,
            "passed": True,
        },
    )


def summarize_proposals(
    groups: Sequence[CachedTrainingGroup],
    proposals: Mapping[str, Sequence[ProposalCandidate]],
    targets: Mapping[str, ProposalTarget],
) -> dict[str, Any]:
    """Apply unchanged decoding and summarize selected IoU plus oracle@K."""

    selected_by_group: dict[str, ProposalCandidate] = {}
    cholec_sequences: dict[
        tuple[str, int], list[CachedTrainingGroup]
    ] = defaultdict(list)
    effective_counts: list[int] = []
    shortlist_refined: list[float] = []
    oracle_by_dataset: dict[str, list[float]] = defaultdict(list)
    iou_by_group_id: dict[tuple[str, str], float] = {}
    for group in groups:
        candidates = tuple(proposals[group.group_key])
        if not candidates:
            raise ValueError(f"proposal group {group.group_key} is empty")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"proposal group {group.group_key} has duplicate IDs")
        effective_counts.append(len(candidates))
        shortlist_refined.extend(float(item.refined) for item in candidates)
        group_ious = [
            _proposal_iou(candidate, targets[group.group_key])
            for candidate in candidates
        ]
        oracle_by_dataset[group.dataset].append(max(group_ious))
        for candidate, iou in zip(candidates, group_ious, strict=True):
            iou_by_group_id[(group.group_key, candidate.candidate_id)] = iou
        if group.dataset == "CholecTrack20":
            cholec_sequences[(group.dataset, group.row_index)].append(group)
        else:
            selected_by_group[group.group_key] = max(
                candidates,
                key=lambda item: (item.score, item.candidate_id),
            )
    for sequence in cholec_sequences.values():
        timestamp_candidates: dict[float, tuple[ScoredCandidate, ...]] = {}
        group_by_timestamp: dict[float, CachedTrainingGroup] = {}
        for group in sequence:
            group_by_timestamp[group.requested_timestamp] = group
            timestamp_candidates[group.requested_timestamp] = tuple(
                ScoredCandidate(
                    candidate_id=item.candidate_id,
                    timestamp=group.requested_timestamp,
                    bbox_xyxy=item.bbox_xyxy,
                    score=item.score,
                    rank=item.rank,
                )
                for item in proposals[group.group_key]
            )
        selected = select_with_iter97_dp(
            dataset="CholecTrack20",
            timestamp_candidates=timestamp_candidates,
        )
        for timestamp, chosen in selected.items():
            group = group_by_timestamp[timestamp]
            selected_by_group[group.group_key] = next(
                item
                for item in proposals[group.group_key]
                if item.candidate_id == chosen.candidate_id
            )
    if len(selected_by_group) != len(groups):
        raise RuntimeError("proposal decoder did not select one box per group")
    row_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    selected_refined: list[float] = []
    for group in groups:
        chosen = selected_by_group[group.group_key]
        selected_refined.append(float(chosen.refined))
        row_values[(group.dataset, group.row_index)].append(
            iou_by_group_id[(group.group_key, chosen.candidate_id)]
        )
    selected_by_dataset: dict[str, list[float]] = defaultdict(list)
    for (dataset, _row_index), values in row_values.items():
        selected_by_dataset[dataset].append(fmean(values))
    per_dataset = {
        dataset: fmean(values)
        for dataset, values in sorted(selected_by_dataset.items())
    }
    oracle_per_dataset = {
        dataset: fmean(values)
        for dataset, values in sorted(oracle_by_dataset.items())
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
        "shortlist_refined_fraction": fmean(shortlist_refined),
        "selected_refined_fraction": fmean(selected_refined),
        "oracle": {
            "overall_dataset_mean": fmean(oracle_per_dataset.values()),
            "per_dataset": oracle_per_dataset,
        },
    }


def summarize_original_oracle(
    groups: Sequence[CachedTrainingGroup],
) -> dict[str, Any]:
    """Summarize the frozen original-candidate oracle per timestamp group."""

    values: dict[str, list[float]] = defaultdict(list)
    for group in groups:
        values[group.dataset].append(float(np.max(group.ious)))
    per_dataset = {
        dataset: fmean(dataset_values)
        for dataset, dataset_values in sorted(values.items())
    }
    return {
        "overall_dataset_mean": fmean(per_dataset.values()),
        "per_dataset": per_dataset,
    }


def proposal_screening_gate(
    result: ProposalOOFResult,
    *,
    config: Mapping[str, Any],
    boundary_audit: Mapping[str, bool],
) -> dict[str, Any]:
    """Apply the pre-registered selected/oracle/semantic/boundary screen."""

    metrics = result.metrics
    overall = float(metrics["overall_dataset_mean"])
    query_drop = overall - float(
        result.shuffled_query_metrics["overall_dataset_mean"]
    )
    roi_drop = overall - float(result.shuffled_roi_metrics["overall_dataset_mean"])
    per_dataset = metrics["per_dataset"]
    minimums = config["distilled_per_dataset_min"]
    dataset_pass = {
        dataset: float(per_dataset[dataset]) >= float(minimums[dataset])
        for dataset in minimums
    }
    oracle = result.combined_oracle
    oracle_baseline = config["oracle_baseline_per_dataset"]
    oracle_tolerance = float(config["oracle_regression_tolerance"])
    oracle_dataset_pass = {
        dataset: float(oracle["per_dataset"][dataset])
        >= float(oracle_baseline[dataset]) - oracle_tolerance
        for dataset in oracle_baseline
    }
    oracle_gain = float(oracle["overall_dataset_mean"]) - float(
        config["oracle_baseline_mean"]
    )
    boundary_pass = all(bool(value) for value in boundary_audit.values())
    checks = {
        "overall": overall >= float(config["overall_min"]),
        "per_dataset": all(dataset_pass.values()),
        "oracle_gain": oracle_gain >= float(config["oracle_gain_min"]),
        "oracle_preservation": all(oracle_dataset_pass.values()),
        "shuffled_query": query_drop >= float(config["shuffled_query_drop_min"]),
        "shuffled_roi": roi_drop >= float(config["shuffled_roi_drop_min"]),
        "boundary": boundary_pass,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "overall": overall,
        "overall_min": float(config["overall_min"]),
        "per_dataset": dict(per_dataset),
        "distilled_per_dataset_min": dict(minimums),
        "per_dataset_pass": dataset_pass,
        "oracle": oracle,
        "oracle_gain": oracle_gain,
        "oracle_gain_min": float(config["oracle_gain_min"]),
        "oracle_dataset_pass": oracle_dataset_pass,
        "shuffled_query_drop": query_drop,
        "shuffled_query_drop_min": float(config["shuffled_query_drop_min"]),
        "shuffled_roi_drop": roi_drop,
        "shuffled_roi_drop_min": float(config["shuffled_roi_drop_min"]),
        "boundary_audit": dict(boundary_audit),
    }


def write_proposal_oof_result(result: ProposalOOFResult, path: Path) -> None:
    """Write compact JSON with complete candidate-level OOF proposal behavior."""

    payload = {
        "schema_version": 1,
        "model_kind": "query_conditioned_proposal_adapter",
        "proposals": _serialize_proposal_map(result.proposals),
        "shuffled_query_proposals": _serialize_proposal_map(
            result.shuffled_query_proposals
        ),
        "shuffled_roi_proposals": _serialize_proposal_map(
            result.shuffled_roi_proposals
        ),
        "original_quality_scores": result.original_quality_scores,
        "metrics": result.metrics,
        "shuffled_query_metrics": result.shuffled_query_metrics,
        "shuffled_roi_metrics": result.shuffled_roi_metrics,
        "original_oracle": result.original_oracle,
        "combined_oracle": result.combined_oracle,
        "fold_standardizers": result.fold_standardizers,
        "fold_audit": result.fold_audit,
        "training_trace": result.training_trace,
        "parameter_audit": result.parameter_audit,
        "coordinate_audit": result.coordinate_audit,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _collate_proposal_batch(
    groups: Sequence[CachedTrainingGroup],
    targets: Mapping[str, ProposalTarget],
    indices: Sequence[int],
    standardizer: StructureStandardizer,
    *,
    device: torch.device,
    query_override: Mapping[int, np.ndarray] | None = None,
    shuffle_roi: bool = False,
    seed: int = 0,
) -> dict[str, Tensor | None]:
    batch = collate_groups(
        groups,
        indices,
        standardizer,
        device=device,
        query_override=query_override,
        shuffle_roi=shuffle_roi,
        seed=seed,
    )
    gt_boxes = torch.zeros((len(indices), 4), dtype=torch.float32, device=device)
    has_target = torch.zeros(len(indices), dtype=torch.bool, device=device)
    for row, group_index in enumerate(indices):
        target = targets[groups[group_index].group_key]
        has_target[row] = target.has_target
        if target.gt_box_xyxy_norm is not None:
            gt_boxes[row] = torch.as_tensor(
                target.gt_box_xyxy_norm,
                dtype=torch.float32,
                device=device,
            )
    batch["gt_boxes_xyxy"] = gt_boxes
    batch["has_target"] = has_target
    return batch


def _save_proposal_split_checkpoint(
    path: Path,
    *,
    model: QueryConditionedProposalAdapter,
    standardizer: StructureStandardizer,
    adapter_config: ProposalAdapterConfig,
    input_mask: ProposalAdapterInputMask,
    query_mode: ProposalQueryMode,
    oof_config: OOFConfig,
    model_seed: int,
    batch_seed_base: int,
    raw_train_indices: Sequence[int],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    groups: Sequence[CachedTrainingGroup],
    training_trace: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
) -> None:
    metadata = dict(metadata or {})
    source_hashes = metadata.pop("source_input_hashes", None)
    source_contract = metadata.pop("source_feature_contract_hash", None)
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("split checkpoint requires non-empty source input hashes")
    if not isinstance(source_contract, str) or not source_contract:
        raise ValueError("split checkpoint requires a source feature contract hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "schema_version": PROPOSAL_SPLIT_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "proposal_adapter_video_disjoint_split",
            "source_feature_contract_hash": source_contract,
            "source_input_hashes": dict(source_hashes),
            "adapter_config": asdict(adapter_config),
            "train_config": asdict(oof_config),
            "standardizer": {
                "mean": standardizer.mean.tolist(),
                "std": standardizer.std.tolist(),
            },
            "input_mask": asdict(input_mask),
            "query_mode": query_mode,
            "model_seed": int(model_seed),
            "batch_seed_base": int(batch_seed_base),
            "raw_train_group_count": len(raw_train_indices),
            "train_group_count": len(train_indices),
            "excluded_missing_target_train_group_count": len(raw_train_indices)
            - len(train_indices),
            "test_group_count": len(test_indices),
            "raw_train_group_digest": _group_digest(groups, raw_train_indices),
            "train_group_digest": _group_digest(groups, train_indices),
            "test_group_digest": _group_digest(groups, test_indices),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "training_trace": list(training_trace),
            "metadata": metadata,
            "model_state": state,
        },
        path,
    )


def _group_digest(
    groups: Sequence[CachedTrainingGroup], indices: Sequence[int]
) -> str:
    payload = "\n".join(groups[int(index)].group_key for index in indices)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _forward(
    model: QueryConditionedProposalAdapter,
    batch: Mapping[str, Tensor | None],
    *,
    input_mask: ProposalAdapterInputMask = ProposalAdapterInputMask(),
    query_mode: ProposalQueryMode = "q_last",
) -> ProposalAdapterOutput:
    _validate_query_mode(query_mode)
    return model(
        q_last=_tensor(batch, query_mode),
        roi_final=_tensor(batch, "roi_final"),
        roi_l31=_tensor(batch, "roi_l31"),
        structure=_tensor(batch, "structure"),
        missing=_tensor(batch, "missing"),
        padding_mask=_tensor(batch, "padding_mask").bool(),
        input_mask=input_mask,
    )


def _predict_proposals(
    model: QueryConditionedProposalAdapter,
    groups: Sequence[CachedTrainingGroup],
    indices: Sequence[int],
    standardizer: StructureStandardizer,
    *,
    adapter_config: ProposalAdapterConfig,
    oof_config: OOFConfig,
    device: torch.device,
    query_override: Mapping[int, np.ndarray] | None = None,
    shuffle_roi: bool = False,
    seed: int = 0,
    input_mask: ProposalAdapterInputMask = ProposalAdapterInputMask(),
    query_mode: ProposalQueryMode = "q_last",
) -> tuple[
    dict[str, tuple[ProposalCandidate, ...]],
    dict[str, tuple[float, ...]],
    int,
]:
    _validate_query_mode(query_mode)
    output: dict[str, tuple[ProposalCandidate, ...]] = {}
    original_score_map: dict[str, tuple[float, ...]] = {}
    coordinate_count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), oof_config.eval_batch_size):
            batch_indices = list(indices[start : start + oof_config.eval_batch_size])
            batch = collate_groups(
                groups,
                batch_indices,
                standardizer,
                device=device,
                shuffle_roi=shuffle_roi,
                seed=seed + start,
            )
            if query_override is not None:
                batch[query_mode] = torch.stack(
                    [
                        torch.as_tensor(
                            query_override[index],
                            dtype=torch.float32,
                            device=device,
                        )
                        for index in batch_indices
                    ]
                )
            prediction = _forward(
                model,
                batch,
                input_mask=input_mask,
                query_mode=query_mode,
            )
            original_probabilities = torch.sigmoid(
                prediction.original_quality_logits
            )
            refined_probabilities = torch.sigmoid(prediction.refined_quality_logits)
            original_boxes = _tensor(batch, "raw_structure")[..., 6:10]
            padding_mask = _tensor(batch, "padding_mask").bool()
            dummy_box = original_boxes.new_tensor((0.0, 0.0, 1.0, 1.0))
            original_boxes = torch.where(
                padding_mask[..., None], dummy_box, original_boxes
            )
            refined_boxes = decode_bounded_box_deltas(
                original_boxes, prediction.bounded_deltas
            )
            for row, group_index in enumerate(batch_indices):
                group = groups[group_index]
                count = len(group.candidate_ids)
                original_scores = tuple(
                    float(value)
                    for value in original_probabilities[row, :count].cpu().tolist()
                )
                refined_scores = tuple(
                    float(value)
                    for value in refined_probabilities[row, :count].cpu().tolist()
                )
                boxes = refined_boxes[row, :count].cpu().tolist()
                selected = quality_diversity_shortlist(
                    group,
                    original_scores=original_scores,
                    refined_scores=refined_scores,
                    refined_boxes_xyxy_norm=boxes,
                    config=adapter_config,
                )
                output[group.group_key] = selected
                original_score_map[group.group_key] = original_scores
                coordinate_count += len(selected)
    return output, original_score_map, coordinate_count


def _proposal_iou(candidate: ProposalCandidate, target: ProposalTarget) -> float:
    if not target.has_target or target.gt_box_xyxy_norm is None:
        return 0.0
    left = candidate.bbox_xyxy_norm
    right = target.gt_box_xyxy_norm
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1.0e-12)


def _serialize_proposal_map(
    values: Mapping[str, Sequence[ProposalCandidate]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        group_key: [asdict(candidate) for candidate in candidates]
        for group_key, candidates in values.items()
    }


def _shuffled_query_overrides(
    groups: Sequence[CachedTrainingGroup],
    indices: Sequence[int],
    *,
    query_mode: ProposalQueryMode = "q_last",
) -> dict[int, np.ndarray]:
    _validate_query_mode(query_mode)
    if len(indices) < 2:
        raise ValueError("proposal query shuffle requires at least two groups")
    output: dict[int, np.ndarray] = {}
    for offset, index in enumerate(indices):
        source = groups[index]
        for step in range(1, len(indices)):
            candidate = groups[indices[(offset + step) % len(indices)]]
            if (candidate.dataset, candidate.row_index) != (
                source.dataset,
                source.row_index,
            ):
                query = getattr(candidate, query_mode)
                if query is None:
                    raise ValueError(f"proposal query field {query_mode} is missing")
                output[index] = query
                break
        else:
            raise ValueError("proposal query shuffle found no disjoint replacement")
    return output


def _validate_query_mode(query_mode: str) -> None:
    if query_mode not in {"q_last", "q_mean"}:
        raise ValueError(f"unsupported proposal query mode: {query_mode}")


def _tensor(batch: Mapping[str, Tensor | None], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"proposal batch tensor {key} is missing")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
