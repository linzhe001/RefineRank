"""Streaming normalization and train-only cap gates for iter90 STG pools."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from PIL import Image

from refinerank.coordinates import annotation_box_to_image
from refinerank.geometry import normalize_box_xyxy
from refinerank.timeline import SampledFrameTimeline
from refinerank.types import CandidateGroup, FrozenCandidate

_STEP_RE = re.compile(r"every\s+(\d+(?:\.\d+)?)\s*seconds?", re.IGNORECASE)
_SPAN_RE = re.compile(
    r"(?:from|between)\s+(\d+(?:\.\d+)?)\s*(?:seconds?)?\s+"
    r"(?:to|and)\s+(\d+(?:\.\d+)?)\s*(?:seconds?)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ShortlistPolicy:
    """One deterministic GT-free candidate shortlist policy."""

    name: str
    strategy: str
    k: int
    native_anchor_count: int = 0
    diversity_alpha: float = 0.5

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("shortlist policy name must be non-empty")
        if self.strategy not in {"native", "diversity_mmr"}:
            raise ValueError(f"unsupported shortlist strategy {self.strategy!r}")
        if self.k <= 0:
            raise ValueError("shortlist policy k must be positive")
        if not 0.0 <= self.diversity_alpha <= 1.0:
            raise ValueError("shortlist diversity_alpha must be within [0, 1]")
        if self.strategy == "native" and self.native_anchor_count != 0:
            raise ValueError("native shortlist policy cannot define anchors")
        if self.strategy == "diversity_mmr" and not (
            1 <= self.native_anchor_count <= self.k
        ):
            raise ValueError(
                "diversity shortlist native_anchor_count must be within [1, k]"
            )


@dataclass(frozen=True)
class CandidateCapReport:
    """Per-dataset oracle retention and the resulting frozen policy/K."""

    pool_cap: int
    tolerance: float
    selected_k: int | None
    per_k: Mapping[str, Mapping[str, Mapping[str, float | int | bool]]]
    boundary_snapped_candidate_count: int = 0
    rejected_invalid_candidate_count: int = 0
    selected_policy: str | None = None
    per_policy: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return asdict(self)


@dataclass(frozen=True)
class _Context:
    row_index: int
    sample_id: str
    original_id: str
    video_id: str
    dataset: str
    question: str
    requested_timestamps: tuple[float, ...]
    gt_boxes: Mapping[str, tuple[float, float, float, float]]
    timeline: SampledFrameTimeline


@dataclass(frozen=True)
class _RawCandidate:
    candidate_id: str
    candidate_time: float
    bbox_xyxy: tuple[float, float, float, float]
    native_score: float
    rank: int
    source: str
    dino_box_score: float
    temporal_selector_score: float
    coverage_ratio: float
    smoothness: float
    missing: tuple[float, float, float, float]


@dataclass(frozen=True)
class _DetectorIdentity:
    caption: str
    box_threshold: float
    text_threshold: float
    query_plan_digest: str


@dataclass(frozen=True)
class _DetectorProgress:
    requested_timestamp: float
    detector_identity: _DetectorIdentity
    tube_count: int
    status: str


class _PoolAccumulator:
    def __init__(self, *, pool_cap: int, max_time_delta: float) -> None:
        self.pool_cap = pool_cap
        self.max_time_delta = max_time_delta
        self.within: list[tuple[tuple[Any, ...], int, _RawCandidate]] = []
        self.serial = 0

    def add(self, candidate: _RawCandidate, requested_timestamp: float) -> None:
        delta = abs(candidate.candidate_time - requested_timestamp)
        self.serial += 1
        if delta <= self.max_time_delta + 1.0e-6:
            key = _native_key(candidate, requested_timestamp)
            _push_top(self.within, key, self.serial, candidate, self.pool_cap)

    def active(self, requested_timestamp: float) -> list[_RawCandidate]:
        candidates = [entry[2] for entry in self.within]
        exact = [
            item for item in candidates if _delta(item, requested_timestamp) <= 1.0e-6
        ]
        if exact:
            return exact
        neighbor = [
            item for item in candidates if _delta(item, requested_timestamp) <= 1.0
        ]
        if neighbor:
            return neighbor
        return candidates


def build_candidate_stores(
    *,
    split_json: Path,
    tube_jsonl: Path,
    valdata_root: Path,
    pool_cap: int = 128,
    cap_candidates: Sequence[int] = (32, 64),
    shortlist_policies: Sequence[ShortlistPolicy] | None = None,
    oracle_tolerance: float = 0.005,
    expand_radius: float = 4.0,
    max_candidate_time_delta: float = 0.0,
    include_labels: bool = True,
    frozen_k: int | None = None,
    frozen_policy_name: str | None = None,
    max_split_rows: int | None = None,
    max_tube_lines: int | None = None,
    require_tube_frame_provenance: bool = False,
    tube_progress_jsonl: Path | None = None,
) -> tuple[
    tuple[CandidateGroup, ...],
    dict[str, dict[str, tuple[str, ...] | tuple[float, ...]]],
    CandidateCapReport,
]:
    """Build a GT-free CandidateStore and optional separate train labels."""

    if max_candidate_time_delta < 0.0:
        raise ValueError("max_candidate_time_delta must be non-negative")
    if max_candidate_time_delta > expand_radius:
        raise ValueError("max_candidate_time_delta cannot exceed expand_radius")
    policies = _resolve_shortlist_policies(cap_candidates, shortlist_policies)
    if pool_cap < max(policy.k for policy in policies):
        raise ValueError("pool_cap must be at least the largest cap candidate")
    frozen_policy = None
    if not include_labels or frozen_k is not None or frozen_policy_name is not None:
        frozen_policy = _resolve_frozen_policy(
            policies,
            frozen_k=frozen_k,
            frozen_policy_name=frozen_policy_name,
        )
    contexts = _load_contexts(
        split_json,
        valdata_root=valdata_root,
        include_gt=include_labels,
        max_rows=max_split_rows,
    )
    accumulators = {
        _group_key(context, timestamp): _PoolAccumulator(
            pool_cap=pool_cap,
            max_time_delta=max_candidate_time_delta,
        )
        for context in contexts.values()
        for timestamp in context.requested_timestamps
    }
    detector_progress: dict[str, _DetectorProgress] = {}
    if require_tube_frame_provenance:
        if tube_progress_jsonl is None:
            raise ValueError(
                "strict tube frame provenance requires tube_progress_jsonl"
            )
        detector_progress = _load_detector_progress(
            tube_progress_jsonl,
            contexts=contexts,
            expected_group_keys=set(accumulators),
        )
    detector_identities: dict[str, _DetectorIdentity] = {}
    tube_row_counts: dict[str, int] = defaultdict(int)
    with tube_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_tube_lines is not None and line_number > max_tube_lines:
                break
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"{tube_jsonl}:{line_number} must be a JSON object")
            if require_tube_frame_provenance:
                row_index = _strict_json_integer(
                    loaded.get("split_row_index"),
                    f"{tube_jsonl}:{line_number} split_row_index",
                )
            else:
                row_index = int(
                    _finite_float(loaded.get("split_row_index"), -1.0)[0]
                )
            context = contexts.get(row_index)
            if context is None:
                if require_tube_frame_provenance:
                    raise ValueError(
                        f"{tube_jsonl}:{line_number} split_row_index does not "
                        "match an STG split row"
                    )
                continue
            if str(loaded.get("sample_id") or "") != context.sample_id:
                raise ValueError(
                    f"{tube_jsonl}:{line_number} sample_id does not match split row"
                )
            tube = loaded.get("tube")
            if not isinstance(tube, Mapping):
                if require_tube_frame_provenance:
                    raise TypeError(
                        f"{tube_jsonl}:{line_number} strict tube must be a mapping"
                    )
                continue
            strict_timestamp = None
            if require_tube_frame_provenance:
                strict_timestamp, detector_identity = (
                    _validated_tube_requested_timestamp(
                        loaded,
                        context,
                        source=f"{tube_jsonl}:{line_number}",
                    )
                )
                strict_group_key = _group_key(context, strict_timestamp)
                previous_identity = detector_identities.setdefault(
                    strict_group_key, detector_identity
                )
                if previous_identity != detector_identity:
                    raise ValueError(
                        f"{tube_jsonl}:{line_number} detector identity changed "
                        f"within requested group {strict_group_key}"
                    )
                progress = detector_progress[strict_group_key]
                if progress.detector_identity != detector_identity:
                    raise ValueError(
                        f"{tube_jsonl}:{line_number} detector identity does not "
                        f"match progress for requested group {strict_group_key}"
                    )
                tube_row_counts[strict_group_key] += 1
            for candidate in _tube_candidates(
                loaded,
                tube,
                line_number=line_number,
                strict=require_tube_frame_provenance,
            ):
                target_timestamps = (
                    (strict_timestamp,)
                    if strict_timestamp is not None
                    else context.requested_timestamps
                )
                for timestamp in target_timestamps:
                    if (
                        require_tube_frame_provenance
                        and abs(candidate.candidate_time - timestamp) > 1.0e-6
                    ):
                        raise ValueError(
                            f"{tube_jsonl}:{line_number} candidate time does not "
                            "match requested-time frame provenance"
                        )
                    accumulators[_group_key(context, timestamp)].add(
                        candidate,
                        timestamp,
                    )

    if require_tube_frame_provenance:
        for group_key, progress in detector_progress.items():
            actual_count = tube_row_counts.get(group_key, 0)
            if actual_count != progress.tube_count:
                raise ValueError(
                    f"requested-time detector tube coverage mismatch for {group_key}: "
                    f"{actual_count} != {progress.tube_count}"
                )

    full_groups: list[CandidateGroup] = []
    full_labels: dict[str, dict[str, tuple[str, ...] | tuple[float, ...]]] = {}
    boundary_snapped_candidate_count = 0
    rejected_invalid_candidate_count = 0
    for context in contexts.values():
        for timestamp in context.requested_timestamps:
            group_key = _group_key(context, timestamp)
            raw_candidates = accumulators[group_key].active(timestamp)
            group, labels, boundary_counts = _finalize_group(
                context,
                timestamp,
                raw_candidates,
            )
            boundary_snapped_candidate_count += boundary_counts[0]
            rejected_invalid_candidate_count += boundary_counts[1]
            full_groups.append(group)
            full_labels[group_key] = labels

    if include_labels:
        report = choose_candidate_cap(
            full_groups,
            full_labels,
            candidates=cap_candidates,
            policies=policies,
            pool_cap=pool_cap,
            tolerance=oracle_tolerance,
            boundary_snapped_candidate_count=boundary_snapped_candidate_count,
            rejected_invalid_candidate_count=rejected_invalid_candidate_count,
        )
        if frozen_policy is not None:
            report = replace(
                report,
                selected_k=frozen_policy.k,
                selected_policy=frozen_policy.name,
            )
    else:
        report = CandidateCapReport(
            pool_cap=pool_cap,
            tolerance=oracle_tolerance,
            selected_k=frozen_policy.k if frozen_policy is not None else None,
            per_k={},
            boundary_snapped_candidate_count=boundary_snapped_candidate_count,
            rejected_invalid_candidate_count=rejected_invalid_candidate_count,
            selected_policy=frozen_policy.name if frozen_policy is not None else None,
            per_policy={},
        )
    if report.selected_k is None:
        return tuple(full_groups), full_labels, report
    selected_policy = _policy_by_name(policies, report.selected_policy)
    capped_groups: list[CandidateGroup] = []
    capped_labels: dict[str, dict[str, tuple[str, ...] | tuple[float, ...]]] = {}
    for group in full_groups:
        order = _shortlist_indices(group, selected_policy)
        candidates_selected = tuple(group.candidates[index] for index in order)
        capped_groups.append(
            CandidateGroup(
                **{
                    **group.__dict__,
                    "candidates": candidates_selected,
                }
            )
        )
        if include_labels:
            labels = full_labels[group.group_key]
            label_by_id = dict(
                zip(labels["candidate_ids"], labels["ious"], strict=True)
            )
            capped_labels[group.group_key] = {
                "candidate_ids": tuple(
                    item.candidate_id for item in candidates_selected
                ),
                "ious": tuple(
                    label_by_id[item.candidate_id] for item in candidates_selected
                ),
            }
    return tuple(capped_groups), capped_labels, report


def resolved_requested_frame_content_digest(
    split_json: Path,
    *,
    valdata_root: Path,
) -> str:
    """Hash the exact source-frame bytes resolved for every requested group."""

    contexts = _load_contexts(
        split_json,
        valdata_root=valdata_root,
        include_gt=False,
        max_rows=None,
    )
    rows: list[tuple[str, Path]] = []
    for context in contexts.values():
        for timestamp in context.requested_timestamps:
            resolved = context.timeline.resolve(timestamp)
            frame_path = resolved.frame_path.resolve()
            if not frame_path.is_file():
                raise FileNotFoundError(frame_path)
            rows.append((_group_key(context, timestamp), frame_path))
    frame_hashes: dict[Path, str] = {}
    digest = hashlib.sha256()
    for group_key, frame_path in sorted(rows):
        frame_hash = frame_hashes.get(frame_path)
        if frame_hash is None:
            frame_hash = _frame_sha256(frame_path)
            frame_hashes[frame_path] = frame_hash
        for value in (group_key, str(frame_path), frame_hash):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def choose_candidate_cap(
    groups: Sequence[CandidateGroup],
    labels: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    candidates: Sequence[int] = (32, 64),
    policies: Sequence[ShortlistPolicy] | None = None,
    pool_cap: int = 128,
    tolerance: float = 0.005,
    boundary_snapped_candidate_count: int = 0,
    rejected_invalid_candidate_count: int = 0,
) -> CandidateCapReport:
    """Select the first policy retaining per-dataset oracle within tolerance."""

    if not groups:
        raise ValueError("candidate cap requires at least one group")
    resolved_policies = _resolve_shortlist_policies(candidates, policies)
    per_k: dict[str, dict[str, dict[str, float | int | bool]]] = {}
    per_policy: dict[str, dict[str, Any]] = {}
    selected_k: int | None = None
    selected_policy: str | None = None
    for policy in resolved_policies:
        by_dataset: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for group in groups:
            entry = labels.get(group.group_key)
            if entry is None:
                raise KeyError(f"missing labels for group {group.group_key}")
            label_by_id = dict(zip(entry["candidate_ids"], entry["ious"], strict=True))
            full_oracle = max(
                float(label_by_id[item.candidate_id]) for item in group.candidates
            )
            shortlisted = shortlist_candidates(group, policy)
            k_oracle = max(
                float(label_by_id[item.candidate_id]) for item in shortlisted
            )
            by_dataset[group.dataset].append((full_oracle, k_oracle))
        dataset_rows: dict[str, dict[str, float | int | bool]] = {}
        for dataset, values in sorted(by_dataset.items()):
            full_mean = fmean(item[0] for item in values)
            k_mean = fmean(item[1] for item in values)
            dataset_rows[dataset] = {
                "group_count": len(values),
                "oracle_full": full_mean,
                "oracle_k": k_mean,
                "retention_loss": full_mean - k_mean,
                "passed": k_mean >= full_mean - tolerance,
            }
        if policy.strategy == "native":
            per_k[str(policy.k)] = dataset_rows
        per_policy[policy.name] = {
            "strategy": policy.strategy,
            "k": policy.k,
            "native_anchor_count": policy.native_anchor_count,
            "diversity_alpha": policy.diversity_alpha,
            "datasets": dataset_rows,
        }
        if selected_k is None and all(
            bool(row["passed"]) for row in dataset_rows.values()
        ):
            selected_k = policy.k
            selected_policy = policy.name
    return CandidateCapReport(
        pool_cap=pool_cap,
        tolerance=tolerance,
        selected_k=selected_k,
        per_k=per_k,
        boundary_snapped_candidate_count=boundary_snapped_candidate_count,
        rejected_invalid_candidate_count=rejected_invalid_candidate_count,
        selected_policy=selected_policy,
        per_policy=per_policy,
    )


def shortlist_candidates(
    group: CandidateGroup,
    policy: ShortlistPolicy,
) -> tuple[FrozenCandidate, ...]:
    """Select a stable GT-free shortlist from one candidate group."""

    return tuple(group.candidates[index] for index in _shortlist_indices(group, policy))


def _resolve_shortlist_policies(
    candidates: Sequence[int],
    policies: Sequence[ShortlistPolicy] | None,
) -> tuple[ShortlistPolicy, ...]:
    resolved = tuple(policies or ())
    if not resolved:
        resolved = tuple(
            ShortlistPolicy(name=f"native_k{k}", strategy="native", k=int(k))
            for k in candidates
        )
    if not resolved:
        raise ValueError("at least one shortlist policy is required")
    names = [policy.name for policy in resolved]
    if len(names) != len(set(names)):
        raise ValueError("shortlist policy names must be unique")
    return resolved


def _resolve_frozen_policy(
    policies: Sequence[ShortlistPolicy],
    *,
    frozen_k: int | None,
    frozen_policy_name: str | None,
) -> ShortlistPolicy:
    if frozen_policy_name is None:
        matches = [policy for policy in policies if policy.k == frozen_k]
        if len(matches) != 1:
            raise ValueError(
                "GT-free phase requires one unambiguous train-frozen policy"
            )
        return matches[0]
    policy = _policy_by_name(policies, frozen_policy_name)
    if frozen_k is not None and frozen_k != policy.k:
        raise ValueError("train-frozen shortlist policy and K do not match")
    return policy


def _policy_by_name(
    policies: Sequence[ShortlistPolicy],
    name: str | None,
) -> ShortlistPolicy:
    if name is None:
        raise ValueError("candidate cap did not freeze a shortlist policy")
    for policy in policies:
        if policy.name == name:
            return policy
    raise ValueError(f"unknown shortlist policy {name!r}")


def _shortlist_indices(
    group: CandidateGroup,
    policy: ShortlistPolicy,
) -> list[int]:
    native_order = sorted(
        range(len(group.candidates)),
        key=lambda index: _frozen_native_key(
            group.candidates[index], group.requested_timestamp
        ),
        reverse=True,
    )
    target_count = min(policy.k, len(native_order))
    if policy.strategy == "native" or target_count == len(native_order):
        return native_order[:target_count]

    selected = native_order[: min(policy.native_anchor_count, target_count)]
    remaining = set(native_order) - set(selected)
    scores = [candidate.native_score for candidate in group.candidates]
    score_min = min(scores)
    score_max = max(scores)

    def normalized_score(index: int) -> float:
        if score_max <= score_min:
            return 1.0
        return (scores[index] - score_min) / (score_max - score_min)

    while len(selected) < target_count:
        best_index: int | None = None
        best_key: tuple[Any, ...] | None = None
        for index in remaining:
            overlap = max(
                _box_iou(
                    group.candidates[index].bbox_xyxy,
                    group.candidates[chosen].bbox_xyxy,
                )
                for chosen in selected
            )
            utility = (
                policy.diversity_alpha * normalized_score(index)
                + (1.0 - policy.diversity_alpha) * (1.0 - overlap)
            )
            key = (
                utility,
                _frozen_native_key(
                    group.candidates[index], group.requested_timestamp
                ),
            )
            if best_key is None or key > best_key:
                best_index = index
                best_key = key
        if best_index is None:
            raise RuntimeError("diversity shortlist could not select a candidate")
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def _load_contexts(
    path: Path,
    *,
    valdata_root: Path,
    include_gt: bool,
    max_rows: int | None,
) -> dict[int, _Context]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise TypeError(f"{path} must contain a JSON list")
    contexts: dict[int, _Context] = {}
    for row_index, raw in enumerate(loaded):
        if max_rows is not None and len(contexts) >= max_rows:
            break
        if (
            not isinstance(raw, Mapping)
            or str(raw.get("qa_type") or "").lower() != "stg"
        ):
            continue
        question = _clean_question(raw)
        timestamps = _requested_timestamps(question)
        if not timestamps:
            raise ValueError(
                f"split row {row_index} has no visible requested timestamps"
            )
        sample_id = _required_id(raw, row_index)
        video = raw.get("video") or raw.get("frame_paths")
        if not isinstance(video, list) or not video:
            raise ValueError(f"split row {row_index} has no frame paths")
        paths = tuple(_remap_frame_path(item, valdata_root) for item in video)
        metadata = (
            raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {}
        )
        timeline = SampledFrameTimeline.from_row(
            raw,
            frame_paths=paths,
            require_paths_exist=False,
        )
        contexts[row_index] = _Context(
            row_index=row_index,
            sample_id=sample_id,
            original_id=_original_id(raw, sample_id),
            video_id=_video_id(metadata, row_index=row_index),
            dataset=_dataset(raw),
            question=question,
            requested_timestamps=timestamps,
            gt_boxes=_gt_boxes(raw) if include_gt else {},
            timeline=timeline,
        )
    if not contexts:
        raise ValueError(f"no STG rows found in {path}")
    return contexts


def _load_detector_progress(
    path: Path,
    *,
    contexts: Mapping[int, _Context],
    expected_group_keys: set[str],
) -> dict[str, _DetectorProgress]:
    if not path.is_file():
        raise FileNotFoundError(path)
    progress: dict[str, _DetectorProgress] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = f"{path}:{line_number}"
            loaded = json.loads(line)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"{source} must be a JSON object")
            row_index = _strict_json_integer(
                loaded.get("split_row_index"),
                f"{source} split_row_index",
            )
            context = contexts.get(row_index)
            if context is None:
                raise ValueError(
                    f"{source} split_row_index does not match an STG split row"
                )
            if str(loaded.get("sample_id") or "") != context.sample_id:
                raise ValueError(f"{source} sample_id does not match split row")
            requested, detector_identity = _validated_tube_requested_timestamp(
                loaded,
                context,
                source=source,
            )
            group_key = _group_key(context, requested)
            if loaded.get("group_key") != group_key:
                raise ValueError(f"{source} group_key does not match requested group")
            if group_key not in expected_group_keys:
                raise ValueError(f"{source} has an unexpected requested group")
            if group_key in progress:
                raise ValueError(f"{source} duplicates requested group {group_key}")
            if loaded.get("dataset") != context.dataset:
                raise ValueError(f"{source} dataset does not match split row")
            tube_count = _strict_json_integer(
                loaded.get("tube_count"), f"{source} tube_count"
            )
            if tube_count < 0:
                raise ValueError(f"{source} tube_count must be non-negative")
            status = loaded.get("status")
            if status not in {"PASS", "NO_DETECTIONS"}:
                raise ValueError(f"{source} detector status is invalid: {status!r}")
            if status == "NO_DETECTIONS" and tube_count != 0:
                raise ValueError(
                    f"{source} NO_DETECTIONS requires tube_count=0"
                )
            if status == "PASS" and tube_count == 0:
                raise ValueError(f"{source} PASS requires at least one tube")
            progress[group_key] = _DetectorProgress(
                requested_timestamp=requested,
                detector_identity=detector_identity,
                tube_count=tube_count,
                status=status,
            )
    missing = sorted(expected_group_keys - set(progress))
    unexpected = sorted(set(progress) - expected_group_keys)
    if missing or unexpected:
        raise ValueError(
            "requested-time detector progress coverage mismatch: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    return progress


def _tube_candidates(
    loaded: Mapping[str, Any],
    tube: Mapping[str, Any],
    *,
    line_number: int,
    strict: bool = False,
) -> list[_RawCandidate]:
    boxes = tube.get("timestamp_boxes")
    if not isinstance(boxes, list):
        if strict:
            raise TypeError(
                f"tube line {line_number} timestamp_boxes must be a list"
            )
        return []
    if strict and not boxes:
        raise ValueError(f"tube line {line_number} has no timestamp boxes")
    tube_id = str(tube.get("tube_id") or f"line{line_number}")
    rank_match = re.search(r"(\d+)", tube_id)
    if strict and "tube_rank" in tube:
        rank = _strict_json_integer(
            tube["tube_rank"], f"tube line {line_number} tube_rank"
        )
        if rank < 1:
            raise ValueError(f"tube line {line_number} tube_rank must be positive")
    else:
        explicit_rank, rank_missing = _finite_float(tube.get("tube_rank"))
        if not rank_missing:
            rank = max(1, int(explicit_rank))
        else:
            rank = (
                max(1, int(rank_match.group(1)))
                if rank_match
                else max(1, line_number)
            )
    if strict:
        temporal, temporal_missing = _strict_optional_json_number(
            tube, "temporal_selector_score", source=f"tube line {line_number}"
        )
        coverage, coverage_missing = _strict_optional_json_number(
            tube, "coverage_ratio", source=f"tube line {line_number}"
        )
        smoothness, smoothness_missing = _strict_optional_json_number(
            tube, "smoothness", source=f"tube line {line_number}"
        )
        selector_scores = {
            key: _strict_json_number(
                tube[key], f"tube line {line_number} {key}"
            )
            for key in ("tube_selector_score", "tube_selector_model_norm_score")
            if key in tube
        }
        tube_score = selector_scores.get(
            "tube_selector_score",
            selector_scores.get("tube_selector_model_norm_score", 0.0),
        )
        tube_prior, _ = _strict_optional_json_number(
            tube, "tube_prior_score", source=f"tube line {line_number}"
        )
        for name, value, missing_value in (
            ("temporal_selector_score", temporal, temporal_missing),
            ("coverage_ratio", coverage, coverage_missing),
            ("smoothness", smoothness, smoothness_missing),
        ):
            if not missing_value:
                _require_unit_interval(
                    value,
                    f"tube line {line_number} {name}",
                )
        for name, value in selector_scores.items():
            _require_unit_interval(value, f"tube line {line_number} {name}")
        if "tube_prior_score" in tube:
            _require_unit_interval(
                tube_prior,
                f"tube line {line_number} tube_prior_score",
            )
    else:
        temporal, temporal_missing = _finite_float(
            tube.get("temporal_selector_score")
        )
        coverage, coverage_missing = _finite_float(tube.get("coverage_ratio"))
        smoothness, smoothness_missing = _finite_float(tube.get("smoothness"))
        tube_score, _ = _finite_float(
            tube.get(
                "tube_selector_score", tube.get("tube_selector_model_norm_score")
            )
        )
        tube_prior, _ = _finite_float(tube.get("tube_prior_score"))
    output: list[_RawCandidate] = []
    for box_index, box_record in enumerate(boxes):
        if not isinstance(box_record, Mapping):
            if strict:
                raise TypeError(
                    f"tube line {line_number} box {box_index} must be a mapping"
                )
            continue
        box = box_record.get("bbox")
        if (
            not isinstance(box, Sequence)
            or isinstance(box, (str, bytes))
            or len(box) != 4
        ):
            raise ValueError(f"tube line {line_number} box {box_index} is not xyxy")
        if strict:
            bbox = tuple(
                _strict_json_number(
                    value,
                    f"tube line {line_number} box {box_index} bbox[{coordinate}]",
                )
                for coordinate, value in enumerate(box)
            )
            box_score = _strict_json_number(
                box_record.get("score"),
                f"tube line {line_number} box {box_index} score",
            )
            if not 0.0 <= box_score <= 1.0:
                raise ValueError(
                    f"tube line {line_number} box {box_index} score must be "
                    "within [0, 1]"
                )
            box_missing = False
        else:
            bbox = tuple(float(value) for value in box)
            if not all(math.isfinite(value) for value in bbox):
                raise ValueError(
                    f"tube line {line_number} box {box_index} is non-finite"
                )
            box_score, box_missing = _finite_float(box_record.get("score"))
        raw_candidate_time = box_record.get("time")
        if strict:
            candidate_time = _strict_json_number(
                raw_candidate_time,
                f"tube line {line_number} box {box_index} time",
            )
        else:
            candidate_time, _ = _finite_float(raw_candidate_time)
        source = str(box_record.get("candidate_source") or "tube_timestamp_box")
        native_score = (
            0.45 * box_score
            + 0.30 * tube_score
            + 0.10 * tube_prior
            + 0.08 * temporal
            + 0.05 * coverage
            + 0.02 / rank
        )
        candidate_id = f"{tube_id}:b{box_index:04d}@{_timestamp_key(candidate_time)}"
        output.append(
            _RawCandidate(
                candidate_id=candidate_id,
                candidate_time=candidate_time,
                bbox_xyxy=bbox,
                native_score=native_score,
                rank=rank,
                source=source,
                dino_box_score=box_score,
                temporal_selector_score=temporal,
                coverage_ratio=coverage,
                smoothness=smoothness,
                missing=(
                    float(box_missing),
                    float(temporal_missing),
                    float(coverage_missing),
                    float(smoothness_missing),
                ),
            )
        )
    return output


def _validated_tube_requested_timestamp(
    loaded: Mapping[str, Any],
    context: _Context,
    *,
    source: str,
) -> tuple[float, _DetectorIdentity]:
    schema_version = _strict_json_integer(
        loaded.get("schema_version"), f"{source} schema_version"
    )
    if schema_version != 2:
        raise ValueError(f"{source} schema_version must equal 2")
    identity = {
        "phase": "requested_time_dino",
        "metric_scope": "gt_free_candidate_generation",
        "candidate_source": "requested_time_dino",
    }
    for key, expected in identity.items():
        if loaded.get(key) != expected:
            raise ValueError(f"{source} {key} must equal {expected!r}")
    required = {
        "requested_timestamp",
        "resolved_frame_timestamp",
        "source_frame_id",
        "source_frame_path",
        "alignment_error_seconds",
        "detector_caption",
        "box_threshold",
        "text_threshold",
        "query_plan_digest",
        "timeline_digest",
    }
    missing = sorted(key for key in required if key not in loaded)
    if missing:
        raise ValueError(f"{source} missing tube frame provenance: {missing}")
    requested = _strict_json_number(
        loaded["requested_timestamp"], f"{source} requested_timestamp"
    )
    matched = next(
        (
            timestamp
            for timestamp in context.requested_timestamps
            if abs(timestamp - requested) <= 1.0e-6
        ),
        None,
    )
    if matched is None:
        raise ValueError(f"{source} requested timestamp is not part of the STG group")
    group_key = _group_key(context, matched)
    if loaded.get("context_key") != f"stg:{context.sample_id}:{context.row_index}":
        raise ValueError(f"{source} context_key does not match split row")
    if loaded.get("detection_cache_key") != f"requested_time_dino:{group_key}":
        raise ValueError(f"{source} detection_cache_key does not match requested group")
    if "group_key" in loaded and loaded.get("group_key") != group_key:
        raise ValueError(f"{source} group_key does not match requested group")
    resolved = context.timeline.resolve(matched)
    checks = {
        "resolved_frame_timestamp": resolved.resolved_frame_timestamp,
        "alignment_error_seconds": resolved.alignment_error_seconds,
    }
    for key, expected in checks.items():
        actual = _strict_json_number(loaded[key], f"{source} {key}")
        if abs(actual - expected) > 1.0e-9:
            raise ValueError(
                f"{source} {key} does not match resolved group frame: "
                f"{actual} != {expected}"
            )
    source_frame_id = loaded["source_frame_id"]
    if isinstance(source_frame_id, bool) or not isinstance(source_frame_id, int):
        raise TypeError(f"{source} source_frame_id must be an integer")
    if source_frame_id != resolved.source_frame_id:
        raise ValueError(f"{source} source_frame_id does not match group frame")
    source_frame_path = loaded["source_frame_path"]
    if not isinstance(source_frame_path, str) or not source_frame_path.strip():
        raise TypeError(f"{source} source_frame_path must be a non-empty string")
    tube_path = Path(source_frame_path)
    if tube_path.resolve() != resolved.frame_path.resolve():
        raise ValueError(f"{source} source_frame_path does not match group frame")
    timeline_digest = loaded["timeline_digest"]
    if not isinstance(timeline_digest, str) or not timeline_digest.strip():
        raise TypeError(f"{source} timeline_digest must be a non-empty string")
    if timeline_digest != resolved.timeline_digest:
        raise ValueError(f"{source} timeline_digest does not match group timeline")
    text_values: dict[str, str] = {}
    for key in ("detector_caption", "query_plan_digest"):
        value = loaded[key]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{source} {key} must be a non-empty string")
        text_values[key] = value
    threshold_values: dict[str, float] = {}
    for key in ("box_threshold", "text_threshold"):
        value = _strict_json_number(loaded[key], f"{source} {key}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{source} {key} must be within [0, 1]")
        threshold_values[key] = value
    return matched, _DetectorIdentity(
        caption=text_values["detector_caption"],
        box_threshold=threshold_values["box_threshold"],
        text_threshold=threshold_values["text_threshold"],
        query_plan_digest=text_values["query_plan_digest"],
    )


def _finalize_group(
    context: _Context,
    timestamp: float,
    raw_candidates: Sequence[_RawCandidate],
) -> tuple[
    CandidateGroup,
    dict[str, tuple[str, ...] | tuple[float, ...]],
    tuple[int, int],
]:
    resolved_frame = context.timeline.resolve(timestamp)
    frame_path = resolved_frame.frame_path
    if not frame_path.exists():
        raise FileNotFoundError(f"nearest sampled frame not found: {frame_path}")
    with Image.open(frame_path) as image:
        width, height = image.size
    if not raw_candidates:
        raw_candidates = _current_frame_anchor_candidates(
            timestamp=timestamp,
            width=width,
            height=height,
        )
    duration = max(1.0, context.timeline.frame_timestamps[-1])
    gt_box = context.gt_boxes.get(_timestamp_key(timestamp))
    gt_box_image = (
        annotation_box_to_image(
            gt_box,
            dataset=context.dataset,
            image_size=(width, height),
        )
        if gt_box is not None
        else None
    )
    candidates: list[FrozenCandidate] = []
    ious: list[float] = []
    seen: set[str] = set()
    boundary_snapped = 0
    rejected_invalid = 0
    for raw in sorted(
        raw_candidates,
        key=lambda item: _native_key(item, timestamp),
        reverse=True,
    ):
        if raw.candidate_id in seen:
            continue
        seen.add(raw.candidate_id)
        try:
            x1, y1, x2, y2 = normalize_box_xyxy(
                raw.bbox_xyxy,
                width=width,
                height=height,
            )
        except ValueError:
            rejected_invalid += 1
            continue
        if _box_exceeds_frame(raw.bbox_xyxy, width=width, height=height):
            boundary_snapped += 1
        area = max((x2 - x1) * (y2 - y1), 1.0e-8)
        aspect = max((x2 - x1) / (y2 - y1), 1.0e-8)
        structure = (
            raw.dino_box_score,
            raw.temporal_selector_score,
            raw.coverage_ratio,
            raw.smoothness,
            abs(raw.candidate_time - timestamp) / duration,
            1.0 if abs(raw.candidate_time - timestamp) <= 1.0e-6 else 0.0,
            x1,
            y1,
            x2,
            y2,
            math.log(area),
            math.log(aspect),
        )
        missing = (*raw.missing, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        candidates.append(
            FrozenCandidate(
                candidate_id=raw.candidate_id,
                candidate_time=raw.candidate_time,
                bbox_xyxy=raw.bbox_xyxy,
                native_score=raw.native_score,
                rank=raw.rank,
                source=raw.source,
                structure=structure,
                missing=missing,
            )
        )
        ious.append(
            _box_iou(raw.bbox_xyxy, gt_box_image)
            if gt_box_image is not None
            else 0.0
        )
    group_key = _group_key(context, timestamp)
    if not candidates:
        raise ValueError(
            f"candidate group {group_key} has no valid proposals; "
            f"rejected_invalid={rejected_invalid}"
        )
    group = CandidateGroup(
        group_key=group_key,
        sample_id=context.sample_id,
        original_id=context.original_id,
        video_id=context.video_id,
        row_index=context.row_index,
        dataset=context.dataset,
        requested_timestamp=timestamp,
        clean_question=context.question,
        frame_path=frame_path,
        frame_size=(width, height),
        candidates=tuple(candidates),
        source_frame_id=resolved_frame.source_frame_id,
        resolved_frame_timestamp=resolved_frame.resolved_frame_timestamp,
        alignment_error_seconds=resolved_frame.alignment_error_seconds,
        frame_alignment_exact=resolved_frame.exact,
        timeline_digest=resolved_frame.timeline_digest,
    )
    return (
        group,
        {
            "candidate_ids": tuple(item.candidate_id for item in candidates),
            "ious": tuple(ious),
        },
        (boundary_snapped, rejected_invalid),
    )


def _box_exceeds_frame(
    box: Sequence[float],
    *,
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1 < 0.0 or y1 < 0.0 or x2 > width or y2 > height


def _clean_question(raw: Mapping[str, Any]) -> str:
    question = raw.get("question")
    if isinstance(question, str) and question.strip():
        return question.replace("<video>", " ").strip()
    conversations = raw.get("conversations")
    if isinstance(conversations, list):
        for message in conversations:
            if isinstance(message, Mapping) and message.get("from") in {
                "human",
                "user",
            }:
                return " ".join(
                    str(message.get("value") or "").replace("<video>", " ").split()
                )
    raise ValueError("STG row has no visible question")


def _requested_timestamps(question: str) -> tuple[float, ...]:
    step_match = _STEP_RE.search(question)
    span_match = _SPAN_RE.search(question)
    if step_match is None or span_match is None:
        explicit = re.findall(r"(\d+(?:\.\d+)?)\s*seconds?\s*:", question, re.I)
        return tuple(dict.fromkeys(float(value) for value in explicit))
    step = float(step_match.group(1))
    start = float(span_match.group(1))
    end = float(span_match.group(2))
    if step <= 0 or end < start:
        raise ValueError(f"invalid requested timestamp span in question: {question}")
    values: list[float] = []
    current = start
    while current <= end + 1.0e-6 and len(values) < 256:
        values.append(round(current, 3))
        current += step
    if current <= end + 1.0e-6:
        raise ValueError("STG timestamp span exceeds the 256-item limit")
    return tuple(dict.fromkeys(values))


def _gt_boxes(raw: Mapping[str, Any]) -> dict[str, tuple[float, float, float, float]]:
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


def _dataset(raw: Mapping[str, Any]) -> str:
    for key in ("data_source", "dataset_name", "dataset"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("STG row has no dataset identifier")


def _required_id(raw: Mapping[str, Any], row_index: int) -> str:
    for key in ("id", "sample_id"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"split row {row_index} has no sample id")


def _original_id(raw: Mapping[str, Any], fallback: str) -> str:
    value = raw.get("original_id")
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _video_id(metadata: Mapping[str, Any], *, row_index: int) -> str:
    value = metadata.get("video_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"split row {row_index} has no metadata.video_id")
    return value.strip()


def _current_frame_anchor_candidates(
    *,
    timestamp: float,
    width: int,
    height: int,
) -> tuple[_RawCandidate, ...]:
    """Create GT-free current-frame anchors when no exact detector box exists."""

    normalized = (
        (0.00, 0.00, 1.00, 1.00),
        (0.00, 0.00, 0.50, 0.50),
        (0.25, 0.00, 0.75, 0.50),
        (0.50, 0.00, 1.00, 0.50),
        (0.00, 0.25, 0.50, 0.75),
        (0.25, 0.25, 0.75, 0.75),
        (0.50, 0.25, 1.00, 0.75),
        (0.00, 0.50, 0.50, 1.00),
        (0.25, 0.50, 0.75, 1.00),
        (0.50, 0.50, 1.00, 1.00),
        (0.125, 0.125, 0.375, 0.375),
        (0.375, 0.125, 0.625, 0.375),
        (0.625, 0.125, 0.875, 0.375),
        (0.125, 0.375, 0.375, 0.625),
        (0.375, 0.375, 0.625, 0.625),
        (0.625, 0.375, 0.875, 0.625),
        (0.125, 0.625, 0.375, 0.875),
        (0.375, 0.625, 0.625, 0.875),
        (0.625, 0.625, 0.875, 0.875),
    )
    output = []
    for rank, (x1, y1, x2, y2) in enumerate(normalized, start=1):
        output.append(
            _RawCandidate(
                candidate_id=f"current_frame_anchor_{rank:02d}@{_timestamp_key(timestamp)}",
                candidate_time=timestamp,
                bbox_xyxy=(x1 * width, y1 * height, x2 * width, y2 * height),
                native_score=1.0 / rank,
                rank=rank,
                source="current_frame_anchor",
                dino_box_score=0.0,
                temporal_selector_score=0.0,
                coverage_ratio=0.0,
                smoothness=0.0,
                missing=(1.0, 1.0, 1.0, 1.0),
            )
        )
    return tuple(output)


def _remap_frame_path(value: Any, valdata_root: Path) -> Path:
    text = str(value)
    prefix = "/root/data"
    if text.startswith(prefix):
        return valdata_root / text[len(prefix) :].lstrip("/")
    path = Path(text)
    return path if path.is_absolute() else valdata_root / path


def _group_key(context: _Context, timestamp: float) -> str:
    return f"{context.row_index:06d}:{context.sample_id}@{_timestamp_key(timestamp)}"


def _timestamp_key(value: Any) -> str:
    return f"{float(value):.3f}"


def _frame_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, default: float = 0.0) -> tuple[float, bool]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default, True
    return (parsed, False) if math.isfinite(parsed) else (default, True)


def _strict_json_number(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} must be a finite JSON number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite JSON number")
    return parsed


def _strict_json_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a JSON integer")
    return value


def _strict_optional_json_number(
    payload: Mapping[str, Any], key: str, *, source: str
) -> tuple[float, bool]:
    if key not in payload:
        return 0.0, True
    return _strict_json_number(payload[key], f"{source} {key}"), False


def _require_unit_interval(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")


def _delta(candidate: _RawCandidate, timestamp: float) -> float:
    return abs(candidate.candidate_time - timestamp)


def _native_key(candidate: _RawCandidate, timestamp: float) -> tuple[Any, ...]:
    delta = _delta(candidate, timestamp)
    tier = 2 if delta <= 1.0e-6 else 1 if delta <= 1.0 else 0
    return tier, candidate.native_score, -delta, -candidate.rank, candidate.candidate_id


def _frozen_native_key(candidate: FrozenCandidate, timestamp: float) -> tuple[Any, ...]:
    delta = abs(candidate.candidate_time - timestamp)
    tier = 2 if delta <= 1.0e-6 else 1 if delta <= 1.0 else 0
    return tier, candidate.native_score, -delta, -candidate.rank, candidate.candidate_id


def _push_top(
    heap: list[tuple[tuple[Any, ...], int, _RawCandidate]],
    key: tuple[Any, ...],
    serial: int,
    candidate: _RawCandidate,
    cap: int,
) -> None:
    entry = (key, serial, candidate)
    if len(heap) < cap:
        heapq.heappush(heap, entry)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _box_iou(left: Sequence[float], right: Sequence[float] | None) -> float:
    if right is None:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    inter = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (
        max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        - inter
    )
    return inter / union if union > 0 else 0.0
