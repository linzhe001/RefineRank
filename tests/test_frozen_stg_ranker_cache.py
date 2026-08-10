from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from refinerank.cache import (
    FeatureCacheReader,
    FeatureCacheWriter,
    LabelStore,
    feature_contract_digest,
    read_candidate_store,
    write_candidate_store,
)
from refinerank.candidates import (
    ShortlistPolicy,
    _requested_timestamps,
    build_candidate_stores,
    choose_candidate_cap,
    shortlist_candidates,
)
from refinerank.timeline import SampledFrameTimeline
from refinerank.train_data import load_inference_groups
from refinerank.types import CandidateGroup, FrozenCandidate, GroupFeatures


def _load_cache_builder_script():
    path = (
        Path(__file__).resolve().parents[1] / "refinerank" / "cache_builder.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_cache_builder_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate(index: int) -> FrozenCandidate:
    return FrozenCandidate(
        candidate_id=f"candidate-{index:03d}",
        candidate_time=1.0,
        bbox_xyxy=(float(index), 0.0, float(index + 1), 1.0),
        native_score=float(100 - index),
        rank=index + 1,
        source="cached_dino",
        structure=tuple(float(index) for _ in range(12)),
        missing=(0.0,) * 12,
    )


def _group(candidate_count: int = 2, *, dataset: str = "CoPESD") -> CandidateGroup:
    return CandidateGroup(
        group_key=f"group-{dataset}",
        sample_id="sample-1",
        original_id="original-1",
        video_id="video-1",
        row_index=1,
        dataset=dataset,
        requested_timestamp=1.0,
        clean_question="Where is the tool at 1 second?",
        frame_path=Path("unused.jpg"),
        frame_size=(128, 64),
        candidates=tuple(_candidate(index) for index in range(candidate_count)),
    )


def _features(group: CandidateGroup) -> GroupFeatures:
    count = len(group.candidates)
    return GroupFeatures(
        group_key=group.group_key,
        candidate_ids=tuple(item.candidate_id for item in group.candidates),
        q_last=np.zeros(3584, dtype=np.float16),
        roi_l23=np.zeros((count, 1280), dtype=np.float16),
        roi_l31=np.zeros((count, 1280), dtype=np.float16),
        roi_final=np.zeros((count, 3584), dtype=np.float16),
        structure=np.asarray([item.structure for item in group.candidates]),
        missing=np.asarray([item.missing for item in group.candidates]),
        frame_size=group.frame_size,
        grid_thw=(1, 8, 16),
    )


def _provenance_group(tmp_path: Path) -> CandidateGroup:
    frame = tmp_path / "000001.jpg"
    Image.new("RGB", (128, 64)).save(frame)
    return replace(
        _group(),
        frame_path=frame,
        source_frame_id=1,
        resolved_frame_timestamp=1.0,
        alignment_error_seconds=0.0,
        frame_alignment_exact=True,
        timeline_digest="timeline-sha256",
    )


def _strict_candidate_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    frame = tmp_path / "000100.jpg"
    Image.new("RGB", (100, 50)).save(frame)
    row = {
        "id": "sample-1",
        "original_id": "original-1",
        "qa_type": "stg",
        "data_source": "CoPESD",
        "video": [str(frame)],
        "sampled_video_frames": [100],
        "metadata": {
            "fps": 1.0,
            "video_id": "video-1",
            "input_video_start_frame": 100,
        },
        "question": "boxes sampled every 1 seconds from 0 to 0 seconds",
    }
    split = tmp_path / "split.json"
    split.write_text(json.dumps([row]), encoding="utf-8")
    timeline = SampledFrameTimeline.from_row(row)
    tube = {
        "schema_version": 2,
        "phase": "requested_time_dino",
        "metric_scope": "gt_free_candidate_generation",
        "split_row_index": 0,
        "sample_id": "sample-1",
        "candidate_source": "requested_time_dino",
        "context_key": "stg:sample-1:0",
        "detection_cache_key": "requested_time_dino:000000:sample-1@0.000",
        "requested_timestamp": 0.0,
        "resolved_frame_timestamp": 0.0,
        "source_frame_id": 100,
        "source_frame_path": str(frame),
        "alignment_error_seconds": 0.0,
        "detector_caption": "surgical tool .",
        "box_threshold": 0.05,
        "text_threshold": 0.10,
        "query_plan_digest": "abc123",
        "timeline_digest": timeline.digest,
        "tube": {
            "tube_id": "T01",
            "timestamp_boxes": [
                {
                    "bbox": [1, 1, 20, 20],
                    "score": 0.9,
                    "time": 0.0,
                    "candidate_source": "dino_raw_box",
                }
            ],
        },
    }
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        json.dumps(
            {
                **{key: value for key, value in tube.items() if key != "tube"},
                "group_key": "000000:sample-1@0.000",
                "row_index": 0,
                "dataset": "CoPESD",
                "tube_count": 1,
                "status": "PASS",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return split, frame, progress, tube


def test_label_store_is_train_only(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="only for phase='train'"):
        LabelStore(tmp_path / "labels.json", phase="val")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_timestamp", float("nan")),
        ("resolved_frame_timestamp", float("inf")),
        ("alignment_error_seconds", float("nan")),
    ],
)
def test_candidate_group_rejects_nonfinite_timeline_values(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    group = _provenance_group(tmp_path)

    with pytest.raises(ValueError, match="must be finite"):
        replace(group, **{field: value})


def test_candidate_store_rejects_non_boolean_exact_flag(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    write_candidate_store((_provenance_group(tmp_path),), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frame_alignment_exact"] = "false"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a JSON boolean"):
        read_candidate_store(path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_candidate_store_schema_version_requires_json_integer(
    tmp_path: Path, schema_version: object
) -> None:
    path = tmp_path / "candidates.jsonl"
    write_candidate_store((_provenance_group(tmp_path),), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match="schema_version must be a JSON integer"):
        read_candidate_store(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_frame_id", 1.9, "must be a JSON integer"),
        ("timeline_digest", None, "provenance is incomplete"),
    ],
)
def test_candidate_store_v2_requires_typed_complete_provenance(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "candidates.jsonl"
    write_candidate_store((_provenance_group(tmp_path),), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        read_candidate_store(path)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("sample_id",), 1, "sample_id must be a non-empty JSON string"),
        (("row_index",), True, "row_index must be a JSON integer"),
        (("requested_timestamp",), "1", "requested_timestamp must be a finite"),
        (("frame_size", 0), 128.0, r"frame_size\[0\] must be a JSON integer"),
        (
            ("candidates", 0, "candidate_id"),
            1,
            "candidate_id must be a non-empty JSON string",
        ),
        (
            ("candidates", 0, "candidate_time"),
            "1",
            "candidate_time must be a finite JSON number",
        ),
        (
            ("candidates", 0, "bbox_xyxy", 0),
            True,
            r"bbox_xyxy\[0\] must be a finite JSON number",
        ),
        (
            ("candidates", 0, "native_score"),
            float("nan"),
            "native_score must be a finite JSON number",
        ),
        (
            ("candidates", 0, "rank"),
            1.0,
            "rank must be a JSON integer",
        ),
        (
            ("candidates", 0, "source"),
            None,
            "source must be a non-empty JSON string",
        ),
        (
            ("candidates", 0, "structure", 0),
            "0",
            r"structure\[0\] must be a finite JSON number",
        ),
        (
            ("candidates", 0, "missing", 0),
            True,
            r"missing\[0\] must be a finite JSON number",
        ),
    ],
)
def test_candidate_store_v2_rejects_coerced_group_and_candidate_fields(
    tmp_path: Path,
    field_path: tuple[object, ...],
    value: object,
    message: str,
) -> None:
    path = tmp_path / "candidates.jsonl"
    write_candidate_store((_provenance_group(tmp_path),), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    target: object = payload
    for component in field_path[:-1]:
        target = target[component]  # type: ignore[index]
    target[field_path[-1]] = value  # type: ignore[index]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        read_candidate_store(path)


def test_candidate_store_v1_rejects_v2_provenance_fields(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    write_candidate_store((_provenance_group(tmp_path),), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v1 must not contain v2 provenance"):
        read_candidate_store(path)


def test_strict_candidate_store_requires_detector_progress(tmp_path: Path) -> None:
    split, frame, _progress, payload = _strict_candidate_inputs(tmp_path)
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires tube_progress_jsonl"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
        )


def test_cache_builder_requires_completed_hash_bound_detector_manifest(
    tmp_path: Path,
) -> None:
    script = _load_cache_builder_script()
    tubes = tmp_path / "tubes.jsonl"
    progress = tmp_path / "progress.jsonl"
    tubes.write_text(json.dumps({"tube": 1}) + "\n", encoding="utf-8")
    progress.write_text(json.dumps({"group": 1}) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 2,
        "run_type": "requested_time_groundingdino",
        "status": "completed",
        "full_run_completed": True,
        "contains_gt": False,
        "output": str(tubes),
        "progress": str(progress),
        "output_sha256": script.sha256_file(tubes),
        "progress_sha256": script.sha256_file(progress),
        "group_count": 1,
        "completed_group_count": 1,
        "progress_row_count": 1,
        "total_tube_count": 1,
        "runtime_stack_identity": {
            "python": {},
            "platform": "test",
            "packages": {},
            "accelerator": {},
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validated = script._validate_detector_manifest(
        manifest_path=manifest_path,
        tube_jsonl=tubes,
        progress_jsonl=progress,
    )
    assert validated["group_count"] == 1

    manifest["full_run_completed"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="completed GT-free schema-2"):
        script._validate_detector_manifest(
            manifest_path=manifest_path,
            tube_jsonl=tubes,
            progress_jsonl=progress,
        )


def test_strict_candidate_store_allows_audited_no_detections_only(
    tmp_path: Path,
) -> None:
    split, frame, progress, _payload = _strict_candidate_inputs(tmp_path)
    audit = json.loads(progress.read_text(encoding="utf-8"))
    audit.update({"status": "NO_DETECTIONS", "tube_count": 0})
    progress.write_text(json.dumps(audit) + "\n", encoding="utf-8")
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text("", encoding="utf-8")

    groups, labels, _report = build_candidate_stores(
        split_json=split,
        tube_jsonl=tubes,
        tube_progress_jsonl=progress,
        valdata_root=frame.parent,
        include_labels=False,
        frozen_k=32,
        require_tube_frame_provenance=True,
    )

    assert labels == {}
    assert {candidate.source for candidate in groups[0].candidates} == {
        "current_frame_anchor"
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1, "schema_version must equal 2"),
        ("phase", "legacy", "phase must equal"),
        ("metric_scope", "unknown", "metric_scope must equal"),
        ("candidate_source", "legacy", "candidate_source must equal"),
    ],
)
def test_strict_candidate_store_requires_requested_time_dino_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    payload[field] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            tube_progress_jsonl=progress,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
        )


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("missing_tube", TypeError, "strict tube must be a mapping"),
        ("malformed_boxes", TypeError, "timestamp_boxes must be a list"),
        ("null_caption", TypeError, "detector_caption must be a non-empty string"),
        ("missing_time", ValueError, "time must be a finite JSON number"),
    ],
)
def test_strict_candidate_store_rejects_malformed_detector_rows(
    tmp_path: Path,
    mutation: str,
    error: type[Exception],
    message: str,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    if mutation == "missing_tube":
        payload["tube"] = None
    elif mutation == "malformed_boxes":
        tube = payload["tube"]
        if not isinstance(tube, dict):
            raise AssertionError("strict test tube fixture is not a mapping")
        tube["timestamp_boxes"] = None
    elif mutation == "null_caption":
        payload["detector_caption"] = None
    elif mutation == "missing_time":
        tube = payload["tube"]
        if not isinstance(tube, dict):
            raise AssertionError("strict test tube fixture is not a mapping")
        boxes = tube["timestamp_boxes"]
        if not isinstance(boxes, list) or not isinstance(boxes[0], dict):
            raise AssertionError("strict test timestamp box fixture is invalid")
        boxes[0].pop("time")
    else:
        raise AssertionError(f"unknown test mutation: {mutation}")
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(error, match=message):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


@pytest.mark.parametrize("value", [None, True, 0.0, "0", float("nan")])
def test_strict_candidate_store_requires_integer_split_row_index(
    tmp_path: Path, value: object
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    payload["split_row_index"] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match="split_row_index must be a JSON integer"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


def test_strict_candidate_store_rejects_unknown_split_row_index(
    tmp_path: Path,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    payload["split_row_index"] = 99
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match an STG split row"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


@pytest.mark.parametrize(
    ("field", "value", "box_field"),
    [
        ("requested_timestamp", False, False),
        ("requested_timestamp", "0", False),
        ("resolved_frame_timestamp", False, False),
        ("alignment_error_seconds", "0", False),
        ("box_threshold", False, False),
        ("text_threshold", "0.1", False),
        ("time", False, True),
        ("time", "0", True),
    ],
)
def test_strict_candidate_store_requires_json_number_provenance(
    tmp_path: Path,
    field: str,
    value: object,
    box_field: bool,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    if box_field:
        tube = payload["tube"]
        if not isinstance(tube, dict):
            raise AssertionError("strict test tube fixture is not a mapping")
        boxes = tube["timestamp_boxes"]
        if not isinstance(boxes, list) or not isinstance(boxes[0], dict):
            raise AssertionError("strict test timestamp box fixture is invalid")
        boxes[0][field] = value
    else:
        payload[field] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must be a finite JSON number"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


def test_strict_candidate_store_requires_bounded_detector_thresholds(
    tmp_path: Path,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    payload["box_threshold"] = 1.1
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"box_threshold must be within \[0, 1\]"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


@pytest.mark.parametrize("value", [True, "1"])
def test_strict_candidate_store_rejects_coerced_bbox_coordinates(
    tmp_path: Path, value: object
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    tube = payload["tube"]
    if not isinstance(tube, dict):
        raise AssertionError("strict test tube fixture is not a mapping")
    boxes = tube["timestamp_boxes"]
    if not isinstance(boxes, list) or not isinstance(boxes[0], dict):
        raise AssertionError("strict test timestamp box fixture is invalid")
    bbox = boxes[0]["bbox"]
    if not isinstance(bbox, list):
        raise AssertionError("strict test bbox fixture is invalid")
    bbox[0] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        TypeError, match=r"bbox\[0\] must be a finite JSON number"
    ):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


@pytest.mark.parametrize(
    "value",
    [None, "not-a-number", True, float("nan"), float("inf")],
)
def test_strict_candidate_store_requires_finite_box_score(
    tmp_path: Path, value: object
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    tube = payload["tube"]
    if not isinstance(tube, dict):
        raise AssertionError("strict test tube fixture is not a mapping")
    boxes = tube["timestamp_boxes"]
    if not isinstance(boxes, list) or not isinstance(boxes[0], dict):
        raise AssertionError("strict test timestamp box fixture is invalid")
    if value is None:
        boxes[0].pop("score")
    else:
        boxes[0]["score"] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="score must be a finite"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_strict_candidate_store_requires_bounded_box_score(
    tmp_path: Path, value: float
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    tube = payload["tube"]
    if not isinstance(tube, dict):
        raise AssertionError("strict test tube fixture is not a mapping")
    boxes = tube["timestamp_boxes"]
    if not isinstance(boxes, list) or not isinstance(boxes[0], dict):
        raise AssertionError("strict test timestamp box fixture is invalid")
    boxes[0]["score"] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"score must be within \[0, 1\]"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            tube_progress_jsonl=progress,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tube_rank", 1.0, "tube_rank must be a JSON integer"),
        (
            "temporal_selector_score",
            "0.5",
            "temporal_selector_score must be a finite JSON number",
        ),
        ("coverage_ratio", True, "coverage_ratio must be a finite JSON number"),
        ("smoothness", "1", "smoothness must be a finite JSON number"),
        (
            "tube_selector_score",
            False,
            "tube_selector_score must be a finite JSON number",
        ),
        (
            "tube_selector_model_norm_score",
            "0.5",
            "tube_selector_model_norm_score must be a finite JSON number",
        ),
        (
            "tube_prior_score",
            float("inf"),
            "tube_prior_score must be a finite JSON number",
        ),
    ],
)
def test_strict_candidate_store_rejects_malformed_present_detector_scores(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    tube = payload["tube"]
    if not isinstance(tube, dict):
        raise AssertionError("strict test tube fixture is not a mapping")
    tube[field] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temporal_selector_score", -0.01),
        ("coverage_ratio", 1.01),
        ("smoothness", -0.01),
        ("tube_selector_score", 1.01),
        ("tube_selector_model_norm_score", -0.01),
        ("tube_prior_score", 1.01),
    ],
)
def test_strict_candidate_store_requires_bounded_detector_scores(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    tube = payload["tube"]
    if not isinstance(tube, dict):
        raise AssertionError("strict test tube fixture is not a mapping")
    tube[field] = value
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{field} must be within \[0, 1\]"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("detector_caption", "different target ."),
        ("box_threshold", 0.06),
        ("text_threshold", 0.11),
        ("query_plan_digest", "different-plan"),
    ],
)
def test_strict_candidate_store_requires_one_detector_identity_per_group(
    tmp_path: Path, field: str, changed: object
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    second = json.loads(json.dumps(payload))
    second[field] = changed
    tube = second["tube"]
    if not isinstance(tube, dict):
        raise AssertionError("strict test tube fixture is not a mapping")
    tube["tube_id"] = "T02"
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(
        json.dumps(payload) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="detector identity changed"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tubes,
            valdata_root=frame.parent,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


def test_requested_timestamps_do_not_invent_unaligned_span_end() -> None:
    question = "boxes sampled every 8 seconds from 66 to 78 seconds"

    assert _requested_timestamps(question) == (66.0, 74.0)


def test_requested_timestamps_reject_span_overflow() -> None:
    question = "boxes sampled every 1 second from 0 to 256 seconds"

    with pytest.raises(ValueError, match="exceeds the 256-item limit"):
        _requested_timestamps(question)


def test_val_candidate_store_is_invariant_to_stripped_gt(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (64, 32)).save(frame)
    base_row = {
        "id": "sample-1",
        "original_id": "original-1",
        "qa_type": "stg",
        "data_source": "CoPESD",
        "video": [str(frame.with_name("000001.jpg"))],
        "sampled_video_frames": [1],
        "metadata": {
            "fps": 1.0,
            "video_id": "video-1",
            "input_video_start_frame": 1,
        },
        "conversations": [
            {
                "from": "human",
                "value": "boxes sampled every 1 seconds from 0 to 0 seconds",
            }
        ],
    }
    frame.rename(frame.with_name("000001.jpg"))
    with_gt = {**base_row, "struc_info": [{"bbox_dict": {"0": [1, 1, 9, 9]}}]}
    without_gt = {**base_row, "struc_info": []}
    split_with_gt = tmp_path / "with_gt.json"
    split_without_gt = tmp_path / "without_gt.json"
    split_with_gt.write_text(json.dumps([with_gt]), encoding="utf-8")
    split_without_gt.write_text(json.dumps([without_gt]), encoding="utf-8")
    tube = tmp_path / "tubes.jsonl"
    tube.write_text(
        json.dumps(
            {
                "split_row_index": 0,
                "sample_id": "sample-1",
                "tube": {
                    "tube_id": "T01",
                    "timestamp_boxes": [
                        {"bbox": [1, 1, 20, 20], "score": 0.7, "time": 0.0},
                        {"bbox": [1, 1, 90, 20], "score": 0.6, "time": 0.0},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    left = build_candidate_stores(
        split_json=split_with_gt,
        tube_jsonl=tube,
        valdata_root=tmp_path,
        include_labels=False,
        frozen_k=32,
    )
    right = build_candidate_stores(
        split_json=split_without_gt,
        tube_jsonl=tube,
        valdata_root=tmp_path,
        include_labels=False,
        frozen_k=32,
    )

    assert left[0] == right[0]
    assert left[1] == right[1] == {}
    assert len(left[0][0].candidates) == 1
    assert left[2].rejected_invalid_candidate_count == 1


def test_candidate_store_rejects_distant_boxes_and_uses_current_frame_anchors(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50)).save(frame)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "original_id": "original-1",
                    "qa_type": "stg",
                    "data_source": "CoPESD",
                    "video": [str(frame.with_name("000000.jpg"))],
                    "sampled_video_frames": [0],
                    "metadata": {
                        "fps": 1.0,
                        "video_id": "video-1",
                        "input_video_start_frame": 0,
                    },
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "boxes sampled every 1 seconds from 0 to 0 seconds"
                            ),
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    frame.rename(frame.with_name("000000.jpg"))
    tube = tmp_path / "tubes.jsonl"
    tube.write_text(
        json.dumps(
            {
                "split_row_index": 0,
                "sample_id": "sample-1",
                "tube": {
                    "tube_id": "T01",
                    "timestamp_boxes": [
                        {"bbox": [1, 1, 20, 20], "score": 0.9, "time": 30.0}
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    groups, labels, _report = build_candidate_stores(
        split_json=split,
        tube_jsonl=tube,
        valdata_root=tmp_path,
        include_labels=False,
        frozen_k=32,
    )

    assert labels == {}
    assert all(item.candidate_time == 0.0 for item in groups[0].candidates)
    assert {item.source for item in groups[0].candidates} == {
        "current_frame_anchor"
    }
    assert all("T01" not in item.candidate_id for item in groups[0].candidates)


def test_candidate_store_rejects_detector_group_frame_mismatch(
    tmp_path: Path,
) -> None:
    split, frame, progress, payload = _strict_candidate_inputs(tmp_path)
    payload["source_frame_id"] = 101
    tube = tmp_path / "tubes.jsonl"
    tube.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source_frame_id"):
        build_candidate_stores(
            split_json=split,
            tube_jsonl=tube,
            valdata_root=tmp_path,
            include_labels=False,
            frozen_k=32,
            require_tube_frame_provenance=True,
            tube_progress_jsonl=progress,
        )


def test_train_candidate_store_respects_explicit_frozen_shortlist(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "000000.jpg"
    Image.new("RGB", (100, 50)).save(frame)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "original_id": "original-1",
                    "qa_type": "stg",
                    "data_source": "CoPESD",
                    "video": [str(frame)],
                    "sampled_video_frames": [0],
                    "metadata": {
                        "fps": 1.0,
                        "video_id": "video-1",
                        "input_video_start_frame": 0,
                    },
                    "question": "boxes sampled every 1 seconds from 0 to 0 seconds",
                    "struc_info": [{"bbox_dict": {"0": [1, 1, 20, 20]}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(
        "\n".join(
            json.dumps(
                {
                    "split_row_index": 0,
                    "sample_id": "sample-1",
                    "tube": {
                        "tube_id": f"T{index:02d}",
                        "timestamp_boxes": [
                            {
                                "bbox": [index, 1, index + 10, 20],
                                "score": 1.0 - index / 100.0,
                                "time": 0.0,
                            }
                        ],
                    },
                }
            )
            for index in range(1, 13)
        )
        + "\n",
        encoding="utf-8",
    )
    policies = (
        ShortlistPolicy(name="native_k4", strategy="native", k=4),
        ShortlistPolicy(name="native_k8", strategy="native", k=8),
    )

    groups, labels, report = build_candidate_stores(
        split_json=split,
        tube_jsonl=tubes,
        valdata_root=tmp_path,
        pool_cap=16,
        cap_candidates=(4, 8),
        shortlist_policies=policies,
        include_labels=True,
        frozen_k=8,
        frozen_policy_name="native_k8",
    )

    assert report.selected_k == 8
    assert report.selected_policy == "native_k8"
    assert len(groups[0].candidates) == 8
    assert len(labels[groups[0].group_key]["candidate_ids"]) == 8


def test_cholec_candidate_labels_convert_annotation_box_to_image_space(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "cholec_1080.png"
    Image.new("RGB", (1920, 1080)).save(frame)
    annotation_box = (206.0, 9.0, 303.0, 83.0)
    image_box = (
        annotation_box[0] * 1920.0 / 854.0,
        annotation_box[1] * 1080.0 / 480.0,
        annotation_box[2] * 1920.0 / 854.0,
        annotation_box[3] * 1080.0 / 480.0,
    )
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "original_id": "original-1",
                    "qa_type": "stg",
                    "data_source": "CholecTrack20",
                    "video": [str(frame.with_name("000001.png"))],
                    "sampled_video_frames": [1],
                    "metadata": {
                        "fps": 1.0,
                        "video_id": "VID111",
                        "input_video_start_frame": 1,
                    },
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "boxes sampled every 1 seconds from 0 to 0 seconds"
                            ),
                        }
                    ],
                    "struc_info": [{"bbox_dict": {"0": annotation_box}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    frame.rename(frame.with_name("000001.png"))
    tubes = tmp_path / "tubes.jsonl"
    tubes.write_text(
        json.dumps(
            {
                "split_row_index": 0,
                "sample_id": "sample-1",
                "tube": {
                    "tube_id": "T01",
                    "timestamp_boxes": [
                        {"bbox": image_box, "score": 0.9, "time": 0.0}
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    groups, labels, _report = build_candidate_stores(
        split_json=split,
        tube_jsonl=tubes,
        valdata_root=tmp_path,
        include_labels=True,
    )

    assert groups[0].frame_size == (1920, 1080)
    assert labels[groups[0].group_key]["ious"] == pytest.approx((1.0,))


def test_feature_contract_hash_ignores_phase_and_candidate_pool_only() -> None:
    train = {
        "model_config_hash": "model",
        "prompt_template": "prompt",
        "candidate_pool_hash": "train-pool",
        "phase": "train",
    }
    val = {**train, "candidate_pool_hash": "val-pool", "phase": "val"}

    assert feature_contract_digest(train) == feature_contract_digest(val)
    assert feature_contract_digest(train) != feature_contract_digest(
        {**train, "prompt_template": "changed"}
    )


def test_candidate_and_feature_stores_are_gt_free_and_hash_checked(
    tmp_path: Path,
) -> None:
    group = _group()
    candidate_path = tmp_path / "candidates.jsonl"
    write_candidate_store((group,), candidate_path)
    assert read_candidate_store(candidate_path) == (group,)
    lowered = candidate_path.read_text(encoding="utf-8").lower()
    assert '"gt"' not in lowered
    assert '"ious"' not in lowered

    writer = FeatureCacheWriter(tmp_path / "features", {"source": "train-cache"})
    feature_path = writer.write_group(group, _features(group))
    writer.write_index((group,), {group.group_key: feature_path})
    rows = list(FeatureCacheReader(tmp_path / "features").iter_groups())
    assert len(rows) == 1
    assert rows[0][1].missing.shape == (2, 12)
    assert rows[0][1].q_mean is None
    assert rows[0][1].roi_l7 is None
    assert rows[0][1].roi_l15 is None
    with pytest.raises(ValueError, match="source mismatch"):
        FeatureCacheReader(
            tmp_path / "features",
            expected_sources={"source": "different-cache"},
        )


def test_feature_cache_round_trips_q_mean_l7_and_l15(tmp_path: Path) -> None:
    group = _group()
    count = len(group.candidates)
    features = replace(
        _features(group),
        q_mean=np.ones(3584, dtype=np.float16),
        roi_l7=np.ones((count, 1280), dtype=np.float16),
        roi_l15=np.full((count, 1280), 2.0, dtype=np.float16),
    )
    writer = FeatureCacheWriter(
        tmp_path / "features",
        {"visual_layers": [7, 15, 23, 31], "query_features": ["q_last", "q_mean"]},
    )
    feature_path = writer.write_group(group, features)
    writer.write_index((group,), {group.group_key: feature_path})

    loaded = FeatureCacheReader(tmp_path / "features").read_group(group.group_key)

    np.testing.assert_array_equal(loaded.q_mean, features.q_mean)
    np.testing.assert_array_equal(loaded.roi_l7, features.roi_l7)
    np.testing.assert_array_equal(loaded.roi_l15, features.roi_l15)


def test_moved_roi_probe_crosses_one_spatial_cell_for_small_box() -> None:
    script = _load_cache_builder_script()
    box = (
        0.3951088905334473,
        0.3377507245099103,
        0.420263131459554,
        0.3740850660536024,
    )

    moved = script._moved_box(box, grid_thw=(1, 10, 18))

    assert moved[0] - box[0] == pytest.approx(1.0 / 18.0)
    assert moved[2] - box[2] == pytest.approx(1.0 / 18.0)
    assert moved[2] - moved[0] == pytest.approx(box[2] - box[0])
    assert all(0.0 <= value <= 1.0 for value in moved)


def test_store_alignment_rejects_mismatched_video_identity(tmp_path: Path) -> None:
    candidate_group = _group()
    feature_group = replace(candidate_group, video_id="different-video")
    candidate_path = tmp_path / "candidates.jsonl"
    write_candidate_store((candidate_group,), candidate_path)
    writer = FeatureCacheWriter(tmp_path / "features", {"source": "train-cache"})
    feature_path = writer.write_group(feature_group, _features(feature_group))
    writer.write_index((feature_group,), {feature_group.group_key: feature_path})

    with pytest.raises(ValueError, match="video_id mismatch"):
        load_inference_groups(
            feature_cache=tmp_path / "features",
            candidate_store=candidate_path,
        )


def test_feature_manifest_rejects_forbidden_gt_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden GT/TAL key"):
        FeatureCacheWriter(tmp_path / "features", {"gt": "must-not-enter"})


def test_missing_mask_rejects_non_binary_values() -> None:
    candidate = _candidate(0)

    with pytest.raises(ValueError, match="missing mask"):
        FrozenCandidate(
            **{
                **candidate.__dict__,
                "missing": (0.5,) + candidate.missing[1:],
            }
        )


@pytest.mark.parametrize(
    ("best_index", "expected_k"),
    [(10, 32), (40, 64), (70, None)],
)
def test_adaptive_candidate_cap_retains_oracle(
    best_index: int,
    expected_k: int | None,
) -> None:
    groups = tuple(
        _group(80, dataset=dataset)
        for dataset in ("CholecTrack20", "EgoSurgery", "CoPESD")
    )
    labels = {
        group.group_key: {
            "candidate_ids": tuple(item.candidate_id for item in group.candidates),
            "ious": tuple(
                1.0 if index == best_index else 0.0
                for index in range(len(group.candidates))
            ),
        }
        for group in groups
    }

    report = choose_candidate_cap(groups, labels)

    assert report.selected_k == expected_k


def test_gt_free_diversity_shortlist_retains_distinct_lower_score_box() -> None:
    candidates = (
        FrozenCandidate(
            **{
                **_candidate(0).__dict__,
                "bbox_xyxy": (0.0, 0.0, 10.0, 10.0),
                "native_score": 1.0,
            }
        ),
        FrozenCandidate(
            **{
                **_candidate(1).__dict__,
                "bbox_xyxy": (0.0, 0.0, 10.0, 10.0),
                "native_score": 0.9,
            }
        ),
        FrozenCandidate(
            **{
                **_candidate(2).__dict__,
                "bbox_xyxy": (40.0, 0.0, 50.0, 10.0),
                "native_score": 0.8,
            }
        ),
    )
    policy = ShortlistPolicy(
        name="diversity_mmr_k2",
        strategy="diversity_mmr",
        k=2,
        native_anchor_count=1,
        diversity_alpha=0.5,
    )
    group = CandidateGroup(
        **{
            **_group().__dict__,
            "candidates": candidates,
        }
    )

    selected = shortlist_candidates(group, policy)
    permuted = CandidateGroup(
        **{
            **group.__dict__,
            "candidates": (candidates[2], candidates[0], candidates[1]),
        }
    )

    assert tuple(item.candidate_id for item in selected) == (
        "candidate-000",
        "candidate-002",
    )
    assert tuple(
        item.candidate_id for item in shortlist_candidates(permuted, policy)
    ) == tuple(item.candidate_id for item in selected)


def test_policy_cap_prefers_passing_diversity_then_keeps_native_control() -> None:
    policies = (
        ShortlistPolicy(
            name="diversity_mmr_k2",
            strategy="diversity_mmr",
            k=2,
            native_anchor_count=1,
            diversity_alpha=0.5,
        ),
        ShortlistPolicy(name="native_k3_control", strategy="native", k=3),
    )
    groups = tuple(
        CandidateGroup(
            **{
                **_group(dataset=dataset).__dict__,
                "candidates": (
                    FrozenCandidate(
                        **{
                            **_candidate(0).__dict__,
                            "bbox_xyxy": (0.0, 0.0, 10.0, 10.0),
                            "native_score": 1.0,
                        }
                    ),
                    FrozenCandidate(
                        **{
                            **_candidate(1).__dict__,
                            "bbox_xyxy": (0.0, 0.0, 10.0, 10.0),
                            "native_score": 0.9,
                        }
                    ),
                    FrozenCandidate(
                        **{
                            **_candidate(2).__dict__,
                            "bbox_xyxy": (40.0, 0.0, 50.0, 10.0),
                            "native_score": 0.8,
                        }
                    ),
                ),
            }
        )
        for dataset in ("CholecTrack20", "EgoSurgery", "CoPESD")
    )
    labels = {
        group.group_key: {
            "candidate_ids": tuple(item.candidate_id for item in group.candidates),
            "ious": (0.0, 0.0, 1.0),
        }
        for group in groups
    }

    report = choose_candidate_cap(groups, labels, policies=policies)

    assert report.selected_policy == "diversity_mmr_k2"
    assert report.selected_k == 2
    assert report.per_policy["diversity_mmr_k2"]["datasets"]["CoPESD"][
        "passed"
    ]
    assert report.per_policy["native_k3_control"]["datasets"]["CoPESD"][
        "passed"
    ]


def test_diversity_shortlist_rejects_missing_native_anchor() -> None:
    with pytest.raises(ValueError, match="native_anchor_count"):
        ShortlistPolicy(
            name="invalid",
            strategy="diversity_mmr",
            k=48,
            native_anchor_count=0,
        )
