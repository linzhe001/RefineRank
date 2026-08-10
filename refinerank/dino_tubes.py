#!/usr/bin/env python
"""Generate provenance-complete GroundingDINO tubes at requested STG times."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from refinerank.cache import read_candidate_store, sha256_file
from refinerank.groundingdino_runner import (
    GroundingDINORunner,
    refined_subboxes,
)
from refinerank.types import CandidateGroup

CODE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_GENERATOR_SOURCES = {
    "generator_script": Path(__file__).resolve(),
    "refinerank_source_tree": CODE_ROOT / "refinerank",
}

GENERIC_PHRASES = {
    "instrument",
    "surgical instrument",
    "laparoscopic instrument",
    "tool",
    "surgical tool",
    "object",
    "visible object",
}

_TARGET_PATTERNS = (
    re.compile(
        r"(?:bounding boxes? for|where is|track|locate)\s+(?:the\s+)?"
        r"(.+?)(?:\s+at every|\s+every|\s+from|\s+between|\s+at\s+\d|[?.]|$)",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class QueryPlanIndex:
    """Frozen QueryPlans addressable by exact row key or unique sample ID."""

    exact: Mapping[tuple[int, str], Mapping[str, Any]]
    by_sample: Mapping[str, tuple[Mapping[str, Any], ...]]

    def resolve(self, group: CandidateGroup) -> tuple[Mapping[str, Any], str]:
        """Resolve without using GT, falling back to visible question text."""

        exact = self.exact.get((group.row_index, group.sample_id))
        if exact is not None:
            return exact, "exact_cache_key"
        sample_plans = self.by_sample.get(group.sample_id, ())
        if len(sample_plans) == 1:
            return sample_plans[0], "unique_sample_id_cache_key"
        if len(sample_plans) > 1:
            raise ValueError(
                f"conflicting QueryPlans for sample ID {group.sample_id!r}"
            )
        return _visible_question_fallback_plan(group.clean_question), (
            "visible_question_fallback"
        )


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            yield row


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _query_plans(path: Path) -> QueryPlanIndex:
    exact: dict[tuple[int, str], Mapping[str, Any]] = {}
    by_sample_digest: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in _jsonl_rows(path):
        plan = row.get("query_plan")
        if not isinstance(plan, dict):
            continue
        sample_id = str(row["sample_id"])
        key = (int(row["split_row_index"]), sample_id)
        previous = exact.get(key)
        if previous is not None and previous != plan:
            raise ValueError(f"conflicting QueryPlans for {key}")
        exact[key] = plan
        by_sample_digest.setdefault(sample_id, {})[_digest(plan)] = plan
    return QueryPlanIndex(
        exact=exact,
        by_sample={
            sample_id: tuple(plans.values())
            for sample_id, plans in by_sample_digest.items()
        },
    )


def _visible_question_fallback_plan(question: str) -> dict[str, Any]:
    """Build a deterministic detector plan from visible question text only."""

    normalized = " ".join(question.replace("<video>", " ").split())
    target = ""
    for pattern in _TARGET_PATTERNS:
        match = pattern.search(normalized)
        if match is not None:
            target = " ".join(match.group(1).strip(" ,.:;!?\"'").lower().split())
            break
    if not target or len(target.split()) > 8:
        target = "surgical instrument"
    phrases = [target]
    if not target.startswith("surgical "):
        phrases.append(f"surgical {target}")
    phrases.append(f"visible {target}")
    return {
        "compressed_query": f"track {target}",
        "confidence": 0.5,
        "core_tokens": target.split(),
        "detector_phrases": phrases,
        "intent_terms": ["track", *target.split()],
        "negative_terms": ["background"],
        "short_reason": "deterministic fallback from visible question text",
        "source": "visible_question_fallback",
        "spatial_terms": ["visible"],
        "target_category_hint": target,
        "temporal_cues": [],
    }


def _detector_caption(plan: Mapping[str, Any]) -> str:
    raw = plan.get("detector_phrases")
    phrases = []
    if isinstance(raw, list):
        for value in raw:
            phrase = " ".join(str(value).strip().lower().split())
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    selected = [value for value in phrases if value not in GENERIC_PHRASES]
    selected = (selected or phrases)[:6]
    if not selected:
        hint = " ".join(str(plan.get("target_category_hint") or "").split())
        selected = [hint or "surgical instrument"]
    return " . ".join(selected) + " ."


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite JSON number")
    return parsed


def _validate_detector_controls(
    *, box_threshold: float, text_threshold: float, max_boxes: int
) -> None:
    for name, value in (
        ("box_threshold", box_threshold),
        ("text_threshold", text_threshold),
    ):
        parsed = _strict_json_number(value, name)
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if isinstance(max_boxes, bool) or not isinstance(max_boxes, int):
        raise TypeError("max_boxes must be a JSON integer")
    if max_boxes <= 0:
        raise ValueError("max_boxes must be positive")


def runtime_stack_identity(requested_device: str) -> dict[str, Any]:
    """Return the executable stack that can change detector group outputs."""

    import numpy
    import torch
    import torchvision
    from PIL import __version__ as pillow_version

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
    effective_device = requested_device
    if requested_device == "auto":
        effective_device = "cuda" if cuda_available else "cpu"
    devices = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(cuda_device_count)
    ]
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "Pillow": pillow_version,
        },
        "device": {
            "requested": requested_device,
            "effective": effective_device,
        },
        "accelerator": {
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "devices": devices,
            "driver": _nvidia_driver_identity(cuda_available),
            "toolkit": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def _nvidia_driver_identity(cuda_available: bool) -> dict[str, Any]:
    if not cuda_available:
        return {"query_status": "CUDA_UNAVAILABLE", "version": None}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"query_status": "NVIDIA_SMI_NOT_FOUND", "version": None}
    versions = sorted(
        {line.strip() for line in result.stdout.splitlines() if line.strip()}
    )
    if result.returncode != 0 or not versions:
        return {
            "query_status": f"NVIDIA_SMI_EXIT_{result.returncode}",
            "version": None,
        }
    return {"query_status": "PASS", "version": versions}


def _optional_nonnegative_json_integer(
    value: Any,
    *,
    default: int,
    label: str,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a JSON integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _prediction_list(prediction: Mapping[str, Any], key: str) -> list[Any]:
    raw = prediction.get(key)
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"GroundingDINO {key} must be a sequence")
    return list(raw)


def _validated_image_size(
    value: Any,
    *,
    expected: tuple[int, int],
    group_key: str,
) -> tuple[int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(f"GroundingDINO image_size is invalid for {group_key}")
    dimensions = []
    for index, dimension in enumerate(value):
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(
                f"GroundingDINO image_size[{index}] must be a JSON integer"
            )
        if dimension <= 0:
            raise ValueError(
                f"GroundingDINO image_size[{index}] must be positive"
            )
        dimensions.append(dimension)
    image_size = (dimensions[0], dimensions[1])
    if image_size != expected:
        raise ValueError(
            f"GroundingDINO image_size mismatch for {group_key}: "
            f"{image_size} != {expected}"
        )
    return image_size


def build_candidate_tube(
    *,
    group: CandidateGroup,
    ordinal: int,
    box: Sequence[float],
    score: float,
    phrase: str,
    candidate_source: str,
    detector_caption: str,
    box_threshold: float,
    text_threshold: float,
    query_plan_digest: str,
    query_plan_resolution: str = "exact_cache_key",
) -> dict[str, Any]:
    """Build one requested-time tube with complete frame provenance."""

    provenance = (
        group.source_frame_id,
        group.resolved_frame_timestamp,
        group.alignment_error_seconds,
        group.timeline_digest,
    )
    if any(value is None for value in provenance):
        raise ValueError(f"candidate group {group.group_key} has no frame provenance")
    timestamp = float(group.requested_timestamp)
    tube_id = f"T{ordinal:04d}_requested_time_dino"
    return {
        "schema_version": 2,
        "phase": "requested_time_dino",
        "metric_scope": "gt_free_candidate_generation",
        "sample_id": group.sample_id,
        "split_row_index": group.row_index,
        "candidate_source": "requested_time_dino",
        "context_key": f"stg:{group.sample_id}:{group.row_index}",
        "detection_cache_key": f"requested_time_dino:{group.group_key}",
        "requested_timestamp": timestamp,
        "resolved_frame_timestamp": group.resolved_frame_timestamp,
        "source_frame_id": group.source_frame_id,
        "source_frame_path": str(group.frame_path.resolve()),
        "alignment_error_seconds": group.alignment_error_seconds,
        "detector_caption": detector_caption,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "query_plan_digest": query_plan_digest,
        "query_plan_resolution": query_plan_resolution,
        "timeline_digest": group.timeline_digest,
        "tube": {
            "tube_id": tube_id,
            "tube_rank": ordinal,
            "window_id": f"requested@{timestamp:.3f}",
            "detector_phrase": phrase,
            "raw_detector_phrases": [phrase] if phrase else [],
            "query_plan_id": f"QP{group.row_index:05d}",
            "timestamp_boxes": [
                {
                    "time": timestamp,
                    "bbox": [float(value) for value in box],
                    "score": float(score),
                    "frame_position": 1,
                    "candidate_source": candidate_source,
                }
            ],
            "frame_candidates": [],
            "score_stats": {
                "mean": float(score),
                "max": float(score),
                "std": 0.0,
                "count": 1,
            },
            "coverage_ratio": 1.0,
            "smoothness": 1.0,
            "temporal_selector_score": 0.0,
            "window_frame_count": 1,
            "window_start_timestamp": timestamp,
            "window_end_timestamp": timestamp,
        },
    }


def _detect_group(
    group: CandidateGroup,
    plan: Mapping[str, Any],
    *,
    query_plan_resolution: str,
    runner: GroundingDINORunner,
    max_boxes: int,
    box_threshold: float,
    text_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_detector_controls(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        max_boxes=max_boxes,
    )
    if not group.frame_path.is_file():
        raise FileNotFoundError(group.frame_path)
    caption = _detector_caption(plan)
    plan_digest = _digest(plan)
    prediction = runner.predict_frames([group.frame_path], caption)[0]
    if not isinstance(prediction, Mapping):
        raise TypeError(f"GroundingDINO output must be a mapping for {group.group_key}")
    all_boxes = _prediction_list(prediction, "boxes_xyxy")
    all_scores = _prediction_list(prediction, "scores")
    all_phrases = _prediction_list(prediction, "phrases")
    if not (len(all_boxes) == len(all_scores) == len(all_phrases)):
        raise ValueError(f"GroundingDINO output mismatch for {group.group_key}")
    image_size = _validated_image_size(
        prediction.get("image_size", group.frame_size),
        expected=group.frame_size,
        group_key=group.group_key,
    )
    raw_detector_box_count = _optional_nonnegative_json_integer(
        prediction.get("raw_box_count"),
        default=len(all_boxes),
        label="GroundingDINO raw_box_count",
    )
    clipped_detector_box_count = _optional_nonnegative_json_integer(
        prediction.get("clipped_box_count"),
        default=0,
        label="GroundingDINO clipped_box_count",
    )
    discarded_detector_box_count = _optional_nonnegative_json_integer(
        prediction.get("discarded_invalid_box_count"),
        default=0,
        label="GroundingDINO discarded_invalid_box_count",
    )
    if raw_detector_box_count != len(all_boxes) + discarded_detector_box_count:
        raise ValueError(
            f"GroundingDINO box audit count mismatch for {group.group_key}"
        )
    boxes: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    for index, (raw_box, raw_score) in enumerate(
        zip(all_boxes[:max_boxes], all_scores[:max_boxes], strict=True)
    ):
        if (
            isinstance(raw_box, (str, bytes))
            or not isinstance(raw_box, Sequence)
            or len(raw_box) != 4
        ):
            raise ValueError(
                f"GroundingDINO box {index} is not xyxy for {group.group_key}"
            )
        box = tuple(
            _strict_json_number(
                coordinate,
                f"GroundingDINO box {index} coordinate {coordinate_index}",
            )
            for coordinate_index, coordinate in enumerate(raw_box)
        )
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"GroundingDINO box {index} must have positive xyxy area for "
                f"{group.group_key}"
            )
        width, height = image_size
        if x1 < 0.0 or y1 < 0.0 or x2 > width or y2 > height:
            raise ValueError(
                f"GroundingDINO box {index} must lie within source image "
                f"{width}x{height} for {group.group_key}"
            )
        score = _strict_json_number(
            raw_score, f"GroundingDINO box {index} score"
        )
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"GroundingDINO box {index} score must be within [0, 1]"
            )
        boxes.append(box)
        scores.append(score)
    phrases = all_phrases[:max_boxes]
    tubes = []
    ordinal = 0
    refined_count = 0
    for box, score, phrase in zip(boxes, scores, phrases, strict=True):
        ordinal += 1
        tubes.append(
            build_candidate_tube(
                group=group,
                ordinal=ordinal,
                box=box,
                score=float(score),
                phrase=str(phrase or ""),
                candidate_source="dino_raw_box",
                detector_caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                query_plan_digest=plan_digest,
                query_plan_resolution=query_plan_resolution,
            )
        )
        for name, refined_box, score_scale in refined_subboxes(
            box,
            image_size=image_size,
        ):
            ordinal += 1
            refined_count += 1
            tubes.append(
                build_candidate_tube(
                    group=group,
                    ordinal=ordinal,
                    box=refined_box,
                    score=float(score) * score_scale,
                    phrase=str(phrase or ""),
                    candidate_source=name,
                    detector_caption=caption,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    query_plan_digest=plan_digest,
                    query_plan_resolution=query_plan_resolution,
                )
            )
    return tubes, {
        "schema_version": 2,
        "phase": "requested_time_dino",
        "metric_scope": "gt_free_candidate_generation",
        "candidate_source": "requested_time_dino",
        "group_key": group.group_key,
        "row_index": group.row_index,
        "split_row_index": group.row_index,
        "sample_id": group.sample_id,
        "dataset": group.dataset,
        "context_key": f"stg:{group.sample_id}:{group.row_index}",
        "detection_cache_key": f"requested_time_dino:{group.group_key}",
        "requested_timestamp": group.requested_timestamp,
        "resolved_frame_timestamp": group.resolved_frame_timestamp,
        "source_frame_id": group.source_frame_id,
        "source_frame_path": str(group.frame_path.resolve()),
        "alignment_error_seconds": group.alignment_error_seconds,
        "timeline_digest": group.timeline_digest,
        "detector_caption": caption,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "query_plan_digest": plan_digest,
        "query_plan_resolution": query_plan_resolution,
        "raw_box_count": raw_detector_box_count,
        "clipped_box_count": clipped_detector_box_count,
        "discarded_invalid_box_count": discarded_detector_box_count,
        "retained_raw_box_count": len(boxes),
        "refined_box_count": refined_count,
        "tube_count": len(tubes),
        "status": "PASS" if tubes else "NO_DETECTIONS",
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def frame_content_digest(groups: Sequence[CandidateGroup]) -> str:
    """Hash every effective group-to-frame mapping and frame byte content."""

    digest = hashlib.sha256()
    frame_hashes: dict[Path, str] = {}
    for group in sorted(groups, key=lambda item: item.group_key):
        frame_path = group.frame_path.resolve()
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        frame_hash = frame_hashes.get(frame_path)
        if frame_hash is None:
            frame_hash = sha256_file(frame_path)
            frame_hashes[frame_path] = frame_hash
        for value in (group.group_key, str(frame_path), frame_hash):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    if not frame_hashes:
        raise ValueError("requested-time DINO has no source frames to hash")
    return digest.hexdigest()


def source_tree_digest(root: Path) -> str:
    """Hash effective detector source files while excluding VCS/runtime state."""

    source_root = Path(root)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    files = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and not any(part in ignored for part in path.parts)
    )
    if not files:
        raise ValueError(f"detector source tree is empty: {source_root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def local_generator_source_hashes(
    paths: Mapping[str, Path] = LOCAL_GENERATOR_SOURCES,
) -> dict[str, str]:
    """Hash every local source tree that defines committed group parts."""

    hashes: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        source = Path(path)
        if not name or not source.exists():
            raise FileNotFoundError(f"local generator source is missing: {name}={path}")
        hashes[name] = (
            sha256_file(source) if source.is_file() else source_tree_digest(source)
        )
    return hashes


def detector_input_hashes(
    *,
    candidate_store: Path,
    query_plan_cache: Path,
    groundingdino_config: Path,
    groundingdino_checkpoint: Path,
    groundingdino_repo: Path,
    groups: Sequence[CandidateGroup],
) -> dict[str, str]:
    """Hash every input that can affect requested-time detector group parts."""

    hashes = {
        "candidate_store": sha256_file(candidate_store),
        "query_plan_cache": sha256_file(query_plan_cache),
        "groundingdino_config": sha256_file(groundingdino_config),
        "groundingdino_checkpoint": sha256_file(groundingdino_checkpoint),
        "groundingdino_source_tree": source_tree_digest(groundingdino_repo),
        "source_frames": frame_content_digest(groups),
    }
    hashes.update(
        {
            f"local_source:{name}": digest
            for name, digest in local_generator_source_hashes().items()
        }
    )
    return hashes


def validate_input_identity_stable(
    started: Mapping[str, str], finished: Mapping[str, str]
) -> None:
    """Fail if detector inputs changed while any group was being processed."""

    if dict(finished) != dict(started):
        changed = sorted(
            key
            for key in set(started) | set(finished)
            if started.get(key) != finished.get(key)
        )
        raise RuntimeError(
            "requested-time DINO inputs changed during processing: "
            + ", ".join(changed)
        )


def _group_part_path(root: Path, group_key: str) -> Path:
    name = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
    return root / f"{name}.json"


def write_group_part(
    root: Path,
    *,
    group_key: str,
    tubes: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> Path:
    """Commit one detector group as the atomic resume unit."""

    payload = _group_part_payload(
        group_key=group_key,
        tubes=tubes,
        audit=audit,
    )
    path = _group_part_path(root, group_key)
    _write_json_atomic(path, payload)
    return path


def _group_part_payload(
    *,
    group_key: str,
    tubes: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one self-hashed detector group payload."""

    if str(audit.get("group_key") or "") != group_key:
        raise ValueError("requested-time DINO audit group key mismatch")
    expected_cache_key = f"requested_time_dino:{group_key}"
    for tube in tubes:
        if tube.get("detection_cache_key") != expected_cache_key:
            raise ValueError("requested-time DINO tube group key mismatch")
    tube_count = audit.get("tube_count")
    if isinstance(tube_count, bool) or not isinstance(tube_count, int):
        raise TypeError("requested-time DINO audit tube_count must be an integer")
    if tube_count != len(tubes):
        raise ValueError("requested-time DINO audit tube count mismatch")
    payload = {
        "schema_version": 2,
        "group_key": group_key,
        "tubes": list(tubes),
        "audit": dict(audit),
    }
    return {**payload, "payload_sha256": _digest(payload)}


def load_group_parts(
    root: Path,
    *,
    expected_group_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load and validate all atomically committed detector groups."""

    parts_root = Path(root)
    if not parts_root.is_dir():
        raise NotADirectoryError(parts_root)
    expected = set(expected_group_keys)
    committed: dict[str, dict[str, Any]] = {}
    for path in sorted(parts_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid requested-time DINO group part: {path}")
        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 2
        ):
            raise ValueError(f"invalid requested-time DINO group part: {path}")
        payload_sha256 = payload.get("payload_sha256")
        if not isinstance(payload_sha256, str) or not payload_sha256:
            raise ValueError(f"requested-time DINO group part lacks digest: {path}")
        digest_payload = {
            key: value for key, value in payload.items() if key != "payload_sha256"
        }
        if payload_sha256 != _digest(digest_payload):
            raise RuntimeError(
                f"requested-time DINO group part payload digest mismatch: {path}"
            )
        group_key = str(payload.get("group_key") or "")
        if group_key not in expected:
            raise ValueError(f"unexpected requested-time DINO group part: {group_key}")
        if path != _group_part_path(parts_root, group_key):
            raise ValueError(
                f"requested-time DINO group part filename mismatch: {path}"
            )
        if group_key in committed:
            raise ValueError(f"duplicate requested-time DINO group part: {group_key}")
        tubes = payload.get("tubes")
        audit = payload.get("audit")
        if not isinstance(tubes, list) or not isinstance(audit, dict):
            raise TypeError(f"requested-time DINO group part is incomplete: {path}")
        if str(audit.get("group_key") or "") != group_key:
            raise ValueError(f"requested-time DINO audit mismatch: {path}")
        tube_count = audit.get("tube_count")
        if isinstance(tube_count, bool) or not isinstance(tube_count, int):
            raise TypeError(f"requested-time DINO tube_count is invalid: {path}")
        if tube_count != len(tubes):
            raise ValueError(f"requested-time DINO tube count mismatch: {path}")
        expected_cache_key = f"requested_time_dino:{group_key}"
        if any(
            not isinstance(tube, dict)
            or tube.get("detection_cache_key") != expected_cache_key
            for tube in tubes
        ):
            raise ValueError(f"requested-time DINO tube mapping mismatch: {path}")
        committed[group_key] = payload
    return committed


def materialize_group_outputs(
    groups: Sequence[CandidateGroup],
    committed: Mapping[str, Mapping[str, Any]],
    *,
    output: Path,
    progress: Path,
) -> None:
    """Deterministically rebuild aggregate JSONL from committed group parts."""

    tube_rows: list[Mapping[str, Any]] = []
    audit_rows: list[Mapping[str, Any]] = []
    for group in groups:
        payload = committed.get(group.group_key)
        if payload is None:
            continue
        tubes = payload.get("tubes")
        audit = payload.get("audit")
        if not isinstance(tubes, list) or not isinstance(audit, Mapping):
            raise TypeError(f"invalid committed detector group {group.group_key}")
        tube_rows.extend(tubes)
        audit_rows.append(audit)
    _write_jsonl_atomic(output, tube_rows)
    _write_jsonl_atomic(progress, audit_rows)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_resume_identity(
    previous: Mapping[str, Any],
    *,
    input_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    output: Path,
    progress: Path,
    group_parts: Path,
    git_commit: str,
    runtime_identity: Mapping[str, Any],
) -> None:
    """Reject partial detector resumes across any generator identity change."""

    schema_version = previous.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 2
    ):
        raise ValueError("requested-time DINO resume manifest schema mismatch")
    if previous.get("input_hashes") != input_hashes:
        raise RuntimeError("refusing resume because upstream hashes changed")
    if previous.get("config") != config:
        raise RuntimeError("refusing resume because detector config changed")
    if previous.get("output") != str(output):
        raise RuntimeError("refusing resume because detector output path changed")
    if previous.get("progress") != str(progress):
        raise RuntimeError("refusing resume because detector progress path changed")
    if previous.get("group_parts") != str(group_parts):
        raise RuntimeError("refusing resume because group-parts path changed")
    if previous.get("git_commit") != git_commit:
        raise RuntimeError("refusing resume because generator commit changed")
    if previous.get("runtime_stack_identity") != runtime_identity:
        raise RuntimeError("refusing resume because runtime stack changed")
    status = previous.get("status")
    if status not in {"running", "partial", "completed"}:
        raise ValueError(f"requested-time DINO resume status is invalid: {status!r}")
    if status == "completed":
        if previous.get("full_run_completed") is not True:
            raise ValueError("completed requested-time DINO manifest is inconsistent")
        actual_hashes = {
            "output_sha256": sha256_file(output),
            "progress_sha256": sha256_file(progress),
            "group_parts_sha256": source_tree_digest(group_parts),
        }
        changed = [
            key for key, value in actual_hashes.items() if previous.get(key) != value
        ]
        if changed:
            raise RuntimeError(
                "refusing resume because completed detector artifacts changed: "
                + ", ".join(changed)
            )


def validate_bootstrap_resume_state(
    *,
    manifest: Path,
    output: Path,
    progress: Path,
    group_parts: Path,
) -> None:
    """Allow only the empty-directory window before the initial manifest."""

    if manifest.exists():
        raise ValueError("bootstrap resume requires a missing manifest")
    unexpected = [path for path in (output, progress) if path.exists()]
    if unexpected:
        raise RuntimeError(
            "cannot bootstrap detector resume with unbound aggregate state: "
            f"{unexpected[0]}"
        )
    if not group_parts.is_dir():
        raise FileNotFoundError(
            "bootstrap detector resume requires the existing group-parts directory"
        )
    if any(group_parts.iterdir()):
        raise RuntimeError(
            "cannot bootstrap detector resume from non-empty unbound group parts"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-store", type=Path, required=True)
    parser.add_argument("--query-plan-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--group-parts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--groundingdino-repo", type=Path, required=True)
    parser.add_argument("--groundingdino-config", type=Path, required=True)
    parser.add_argument("--groundingdino-checkpoint", type=Path, required=True)
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-boxes", type=int, default=12)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_detector_controls(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_boxes=args.max_boxes,
    )
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    runtime_identity = runtime_stack_identity(args.device)
    groups = read_candidate_store(args.candidate_store)
    plans = _query_plans(args.query_plan_cache)
    input_hashes = detector_input_hashes(
        candidate_store=args.candidate_store,
        query_plan_cache=args.query_plan_cache,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        groundingdino_repo=args.groundingdino_repo,
        groups=groups,
    )
    config = {
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "max_boxes": args.max_boxes,
        "device": args.device,
    }
    git_commit = _git_commit()
    if args.resume:
        if args.manifest.is_file():
            previous = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(previous, dict):
                raise TypeError("requested-time DINO manifest must be a JSON object")
            validate_resume_identity(
                previous,
                input_hashes=input_hashes,
                config=config,
                output=args.output,
                progress=args.progress,
                group_parts=args.group_parts,
                git_commit=git_commit,
                runtime_identity=runtime_identity,
            )
            if previous.get("status") == "completed":
                print(
                    json.dumps(previous, indent=2, ensure_ascii=False, sort_keys=True)
                )
                return 0
        else:
            validate_bootstrap_resume_state(
                manifest=args.manifest,
                output=args.output,
                progress=args.progress,
                group_parts=args.group_parts,
            )
    else:
        for path in (args.output, args.progress, args.group_parts, args.manifest):
            if path.exists():
                raise FileExistsError(path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.progress.parent.mkdir(parents=True, exist_ok=True)
        args.group_parts.mkdir(parents=True, exist_ok=False)
    committed = load_group_parts(
        args.group_parts,
        expected_group_keys=[group.group_key for group in groups],
    )
    done = set(committed)
    pending = [group for group in groups if group.group_key not in done]
    if args.max_groups is not None:
        pending = pending[: args.max_groups]
    base_manifest = {
        "schema_version": 2,
        "run_type": "requested_time_groundingdino",
        "started_at": started_at,
        "git_commit": git_commit,
        "runtime_stack_identity": runtime_identity,
        "candidate_store": str(args.candidate_store),
        "query_plan_cache": str(args.query_plan_cache),
        "output": str(args.output),
        "progress": str(args.progress),
        "group_parts": str(args.group_parts),
        "input_hashes": input_hashes,
        "config": config,
        "group_count": len(groups),
        "query_plan_cache_exact_key_count": len(plans.exact),
        "status": "running",
        "completed_group_count": len(done),
    }
    if pending:
        _write_json_atomic(args.manifest, base_manifest)
        runner = GroundingDINORunner(
            repo_path=args.groundingdino_repo,
            config_path=args.groundingdino_config,
            checkpoint_path=args.groundingdino_checkpoint,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
        try:
            for index, group in enumerate(pending, start=1):
                plan, query_plan_resolution = plans.resolve(group)
                tubes, audit = _detect_group(
                    group,
                    plan,
                    query_plan_resolution=query_plan_resolution,
                    runner=runner,
                    max_boxes=args.max_boxes,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
                write_group_part(
                    args.group_parts,
                    group_key=group.group_key,
                    tubes=tubes,
                    audit=audit,
                )
                committed[group.group_key] = _group_part_payload(
                    group_key=group.group_key,
                    tubes=tubes,
                    audit=audit,
                )
                done.add(group.group_key)
                if index == 1 or index % 10 == 0 or index == len(pending):
                    _write_json_atomic(
                        args.manifest,
                        {**base_manifest, "completed_group_count": len(done)},
                    )
                    print(
                        f"requested-time DINO {index}/{len(pending)} "
                        f"group={group.group_key} tubes={len(tubes)}",
                        flush=True,
                    )
        finally:
            runner.close()
    materialize_group_outputs(
        groups,
        committed,
        output=args.output,
        progress=args.progress,
    )
    finished_input_hashes = detector_input_hashes(
        candidate_store=args.candidate_store,
        query_plan_cache=args.query_plan_cache,
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        groundingdino_repo=args.groundingdino_repo,
        groups=groups,
    )
    validate_input_identity_stable(input_hashes, finished_input_hashes)
    finished_runtime_identity = runtime_stack_identity(args.device)
    if finished_runtime_identity != runtime_identity:
        raise RuntimeError(
            "requested-time DINO runtime stack changed during processing"
        )
    progress_rows = [
        committed[group.group_key]["audit"]
        for group in groups
        if group.group_key in committed
    ]
    complete = len(done) == len(groups)
    manifest = {
        **base_manifest,
        "finished_at": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "completed_group_count": len(done),
        "processed_this_invocation": len(pending),
        "progress_row_count": len(progress_rows),
        "no_detection_group_count": sum(
            int(row["status"] == "NO_DETECTIONS") for row in progress_rows
        ),
        "query_plan_resolution_counts": {
            mode: sum(
                int(row["query_plan_resolution"] == mode)
                for row in progress_rows
            )
            for mode in sorted(
                {str(row["query_plan_resolution"]) for row in progress_rows}
            )
        },
        "total_tube_count": sum(int(row["tube_count"]) for row in progress_rows),
        "full_run_completed": complete,
        "status": "completed" if complete else "partial",
        "output_sha256": sha256_file(args.output),
        "progress_sha256": sha256_file(args.progress),
        "group_parts_sha256": source_tree_digest(args.group_parts),
        "contains_gt": False,
    }
    _write_json_atomic(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
