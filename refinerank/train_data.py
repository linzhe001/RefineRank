"""Cached-group loading, fold assignment, standardization, and collation."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from refinerank.cache import (
    FeatureCacheReader,
    LabelStore,
    read_candidate_store,
)

DATASETS = ("CholecTrack20", "EgoSurgery", "CoPESD")


@dataclass(frozen=True)
class CachedTrainingGroup:
    """One aligned FeatureStore/CandidateStore/LabelStore training group."""

    group_key: str
    sample_id: str
    row_index: int
    dataset: str
    original_id: str
    requested_timestamp: float
    candidate_ids: tuple[str, ...]
    boxes_xyxy: tuple[tuple[float, float, float, float], ...]
    ranks: tuple[int, ...]
    q_last: np.ndarray
    roi_final: np.ndarray
    roi_l31: np.ndarray | None
    roi_l23: np.ndarray | None
    structure: np.ndarray
    missing: np.ndarray
    ious: np.ndarray
    native_scores: np.ndarray
    frame_size: tuple[int, int] = (1, 1)
    parent_source_indices: tuple[int, ...] | None = None
    refined_flags: tuple[bool, ...] | None = None
    video_id: str = ""
    clean_question: str = ""
    q_mean: np.ndarray | None = None
    roi_l7: np.ndarray | None = None
    roi_l15: np.ndarray | None = None


@dataclass(frozen=True)
class StructureStandardizer:
    """Fold-local structural mean/std fitted without missing values."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray, missing: np.ndarray) -> np.ndarray:
        standardized = (values - self.mean) / self.std
        return np.where(missing > 0.5, 0.0, standardized).astype(np.float32)


def load_training_groups(
    *,
    feature_cache: Path,
    candidate_store: Path,
    label_store: Path,
) -> tuple[CachedTrainingGroup, ...]:
    """Align all three stores and fail on any candidate-order mismatch."""

    labels = LabelStore(label_store, phase="train").read()
    return _load_aligned_groups(
        feature_cache=feature_cache,
        candidate_store=candidate_store,
        labels=labels,
    )


def load_inference_groups(
    *,
    feature_cache: Path,
    candidate_store: Path,
) -> tuple[CachedTrainingGroup, ...]:
    """Load GT-free cached groups for checkpoint inference."""

    return _load_aligned_groups(
        feature_cache=feature_cache,
        candidate_store=candidate_store,
        labels=None,
    )


def _load_aligned_groups(
    *,
    feature_cache: Path,
    candidate_store: Path,
    labels: Mapping[str, Mapping[str, Sequence[object]]] | None,
) -> tuple[CachedTrainingGroup, ...]:
    candidates = {
        group.group_key: group for group in read_candidate_store(candidate_store)
    }
    output: list[CachedTrainingGroup] = []
    for metadata, features in FeatureCacheReader(feature_cache).iter_groups():
        group = candidates.get(features.group_key)
        entry = labels.get(features.group_key) if labels is not None else None
        if group is None or (labels is not None and entry is None):
            raise KeyError(f"store alignment missing group {features.group_key}")
        ids = tuple(item.candidate_id for item in group.candidates)
        if ids != features.candidate_ids or (
            entry is not None and ids != tuple(entry["candidate_ids"])
        ):
            raise ValueError(f"candidate order mismatch for {features.group_key}")
        if tuple(group.frame_size) != tuple(features.frame_size):
            raise ValueError(f"frame size mismatch for {features.group_key}")
        candidate_video_id = group.video_id.strip()
        feature_video_id = str(metadata.get("video_id") or "").strip()
        if (
            candidate_video_id
            and feature_video_id
            and candidate_video_id != feature_video_id
        ):
            raise ValueError(f"video_id mismatch for {features.group_key}")
        video_id = candidate_video_id or feature_video_id
        if not video_id:
            raise ValueError(f"video_id missing for {features.group_key}")
        output.append(
            CachedTrainingGroup(
                group_key=features.group_key,
                sample_id=group.sample_id,
                row_index=group.row_index,
                dataset=str(metadata["dataset"]),
                original_id=str(metadata["original_id"]),
                video_id=video_id,
                clean_question=group.clean_question,
                requested_timestamp=float(metadata["requested_timestamp"]),
                candidate_ids=ids,
                boxes_xyxy=tuple(item.bbox_xyxy for item in group.candidates),
                ranks=tuple(item.rank for item in group.candidates),
                q_last=features.q_last,
                q_mean=features.q_mean,
                roi_final=features.roi_final,
                roi_l7=features.roi_l7,
                roi_l15=features.roi_l15,
                roi_l31=features.roi_l31,
                roi_l23=features.roi_l23,
                structure=features.structure,
                missing=features.missing,
                ious=np.asarray(
                    entry["ious"] if entry is not None else np.zeros(len(ids)),
                    dtype=np.float32,
                ),
                native_scores=np.asarray(
                    [item.native_score for item in group.candidates], dtype=np.float32
                ),
                frame_size=features.frame_size,
            )
        )
    if not output:
        raise ValueError("no aligned training groups found")
    return tuple(output)


def deterministic_group_folds(
    groups: Sequence[CachedTrainingGroup],
    *,
    folds: int = 5,
    seed: int = 20260719,
) -> tuple[int, ...]:
    """Assign `(dataset, video_id)` groups to balanced deterministic folds."""

    if folds < 2:
        raise ValueError("OOF requires at least two folds")
    keys_by_dataset: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for group in groups:
        if not group.video_id:
            raise ValueError(
                f"group {group.group_key} is missing video_id required for OOF"
            )
        keys_by_dataset[group.dataset].add((group.dataset, group.video_id))
    assignment: dict[tuple[str, str], int] = {}
    for dataset, keys in sorted(keys_by_dataset.items()):
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{seed}:{dataset}:{key[1]}".encode()
            ).hexdigest(),
        )
        for index, key in enumerate(ordered):
            assignment[key] = index % folds
    return tuple(assignment[(group.dataset, group.video_id)] for group in groups)


def fit_structure_standardizer(
    groups: Sequence[CachedTrainingGroup],
    indices: Sequence[int],
) -> StructureStandardizer:
    """Fit one structural standardizer from the current training fold only."""

    values = np.concatenate([groups[index].structure for index in indices], axis=0)
    missing = np.concatenate([groups[index].missing for index in indices], axis=0)
    observed = missing <= 0.5
    count = observed.sum(axis=0)
    if (count == 0).any():
        raise ValueError("a structural feature is missing for the entire training fold")
    mean = np.where(observed, values, 0.0).sum(axis=0) / count
    variance = np.where(observed, (values - mean) ** 2, 0.0).sum(axis=0) / count
    std = np.sqrt(variance)
    std = np.where(std < 1.0e-6, 1.0, std)
    return StructureStandardizer(mean.astype(np.float32), std.astype(np.float32))


def balanced_batch_indices(
    groups: Sequence[CachedTrainingGroup],
    indices: Sequence[int],
    *,
    per_dataset: int = 16,
    seed: int,
) -> Iterator[list[int]]:
    """Yield 48-group batches with 16 groups from each dataset."""

    rng = np.random.default_rng(seed)
    by_dataset = {
        dataset: [index for index in indices if groups[index].dataset == dataset]
        for dataset in DATASETS
    }
    missing = [dataset for dataset, values in by_dataset.items() if not values]
    if missing:
        raise ValueError(f"balanced sampler missing datasets: {missing}")
    batch_count = max(
        math.ceil(len(values) / per_dataset) for values in by_dataset.values()
    )
    for _ in range(batch_count):
        batch: list[int] = []
        for dataset in DATASETS:
            values = by_dataset[dataset]
            replace = len(values) < per_dataset
            batch.extend(
                int(value)
                for value in rng.choice(values, size=per_dataset, replace=replace)
            )
        rng.shuffle(batch)
        yield batch


def collate_groups(
    groups: Sequence[CachedTrainingGroup],
    indices: Sequence[int],
    standardizer: StructureStandardizer,
    *,
    device: torch.device,
    query_override: Mapping[int, np.ndarray] | None = None,
    shuffle_roi: bool = False,
    seed: int = 0,
) -> dict[str, Tensor | None]:
    """Pad variable-K groups and optionally apply semantic shuffle diagnostics."""

    selected = [groups[index] for index in indices]
    max_k = max(len(group.candidate_ids) for group in selected)
    batch_size = len(selected)

    def zeros(*shape: int) -> Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=device)

    q_last = zeros(batch_size, 3584)
    q_mean = (
        zeros(batch_size, 3584)
        if any(g.q_mean is not None for g in selected)
        else None
    )
    roi_final = zeros(batch_size, max_k, 3584)
    roi_l7 = (
        zeros(batch_size, max_k, 1280)
        if any(g.roi_l7 is not None for g in selected)
        else None
    )
    roi_l15 = (
        zeros(batch_size, max_k, 1280)
        if any(g.roi_l15 is not None for g in selected)
        else None
    )
    roi_l31 = (
        zeros(batch_size, max_k, 1280)
        if any(g.roi_l31 is not None for g in selected)
        else None
    )
    roi_l23 = (
        zeros(batch_size, max_k, 1280)
        if any(g.roi_l23 is not None for g in selected)
        else None
    )
    structure = zeros(batch_size, max_k, 12)
    raw_structure = zeros(batch_size, max_k, 12)
    missing = zeros(batch_size, max_k, 12)
    ious = zeros(batch_size, max_k)
    padding_mask = torch.ones((batch_size, max_k), dtype=torch.bool, device=device)
    parent_source_indices = torch.full(
        (batch_size, max_k), -1, dtype=torch.long, device=device
    )
    refined_flags = torch.zeros(
        (batch_size, max_k), dtype=torch.bool, device=device
    )
    rng = np.random.default_rng(seed)
    for batch_index, (source_index, group) in enumerate(
        zip(indices, selected, strict=True)
    ):
        count = len(group.candidate_ids)
        query = (query_override or {}).get(source_index, group.q_last)
        q_last[batch_index] = torch.as_tensor(query, dtype=torch.float32, device=device)
        if q_mean is not None:
            if group.q_mean is None:
                raise ValueError("mixed q_mean availability in one cache")
            q_mean[batch_index] = torch.as_tensor(
                group.q_mean, dtype=torch.float32, device=device
            )
        order = rng.permutation(count) if shuffle_roi else np.arange(count)
        roi_final[batch_index, :count] = torch.as_tensor(
            group.roi_final[order], dtype=torch.float32, device=device
        )
        if roi_l7 is not None:
            if group.roi_l7 is None:
                raise ValueError("mixed roi_l7 availability in one cache")
            roi_l7[batch_index, :count] = torch.as_tensor(
                group.roi_l7[order], dtype=torch.float32, device=device
            )
        if roi_l15 is not None:
            if group.roi_l15 is None:
                raise ValueError("mixed roi_l15 availability in one cache")
            roi_l15[batch_index, :count] = torch.as_tensor(
                group.roi_l15[order], dtype=torch.float32, device=device
            )
        if roi_l31 is not None:
            if group.roi_l31 is None:
                raise ValueError("mixed roi_l31 availability in one cache")
            roi_l31[batch_index, :count] = torch.as_tensor(
                group.roi_l31[order], dtype=torch.float32, device=device
            )
        if roi_l23 is not None:
            if group.roi_l23 is None:
                raise ValueError("mixed roi_l23 availability in one cache")
            roi_l23[batch_index, :count] = torch.as_tensor(
                group.roi_l23[order], dtype=torch.float32, device=device
            )
        standardized = standardizer.transform(group.structure, group.missing)
        structure[batch_index, :count] = torch.as_tensor(standardized, device=device)
        raw_structure[batch_index, :count] = torch.as_tensor(
            group.structure, device=device
        )
        missing[batch_index, :count] = torch.as_tensor(group.missing, device=device)
        ious[batch_index, :count] = torch.as_tensor(group.ious, device=device)
        parent_indices = group.parent_source_indices or tuple(range(count))
        child_flags = group.refined_flags or (False,) * count
        if len(parent_indices) != count or len(child_flags) != count:
            raise ValueError("parent/child metadata does not align with candidates")
        parent_source_indices[batch_index, :count] = torch.as_tensor(
            parent_indices, dtype=torch.long, device=device
        )
        refined_flags[batch_index, :count] = torch.as_tensor(
            child_flags, dtype=torch.bool, device=device
        )
        padding_mask[batch_index, :count] = False
    return {
        "q_last": q_last,
        "q_mean": q_mean,
        "roi_final": roi_final,
        "roi_l7": roi_l7,
        "roi_l15": roi_l15,
        "roi_l31": roi_l31,
        "roi_l23": roi_l23,
        "structure": structure,
        "raw_structure": raw_structure,
        "missing": missing,
        "ious": ious,
        "padding_mask": padding_mask,
        "parent_source_indices": parent_source_indices,
        "refined_flags": refined_flags,
    }
