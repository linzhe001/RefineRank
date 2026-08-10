#!/usr/bin/env python
"""Build GT-isolated STG candidate and frozen MedVLM feature stores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from refinerank.cache import (  # noqa: E402
    FeatureCacheReader,
    FeatureCacheWriter,
    LabelStore,
    SpatialGridCacheWriter,
    sha256_file,
    write_candidate_store,
)
from refinerank.candidates import (  # noqa: E402
    ShortlistPolicy,
    build_candidate_stores,
)
from refinerank.features import (  # noqa: E402
    PROMPT_TEMPLATE,
    FrozenMedVLMFeatureExtractor,
)
from refinerank.repool import (  # noqa: E402
    pool_spatial_rois_with_l7,
)
from refinerank.run_manifest import (  # noqa: E402
    RunContext,
    write_run_manifest,
)


def main(argv: list[str] | None = None) -> int:
    """Run candidate-cap or frozen-feature cache construction."""

    args = _parser().parse_args(argv)
    config = _load_config(args.config)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    args.exp_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = args.exp_dir / "stdout+stderr.log"
    resolved_path = args.exp_dir / "resolved_config.json"
    resolved = {"command": args.command, "config": config, "args": vars(args)}
    resolved["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in resolved["args"].items()
    }
    _write_json(resolved_path, resolved)
    exit_code = 1
    error: str | None = None
    artifacts: list[Path] = []
    checkpoint_path: Path | None = None
    summary: dict[str, Any] = {}
    try:
        if args.command == "candidates":
            summary, artifacts = _run_candidates(args, config)
            exit_code = 0 if summary["cap_report"]["selected_k"] is not None else 2
        elif args.command == "features":
            summary, artifacts, checkpoint_path = _run_features(args, config)
            exit_code = 0
        elif args.command == "spatial-features":
            summary, artifacts, checkpoint_path = _run_spatial_features(
                args, config
            )
            exit_code = 0
        else:
            raise ValueError(f"unsupported command {args.command}")
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        stdout_path.write_text(error + "\n", encoding="utf-8")
        raise
    finally:
        if not stdout_path.exists():
            stdout_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        write_run_manifest(
            RunContext(
                run_type=f"frozen_stg_ranker_cache_{args.command}",
                command=" ".join(sys.argv),
                config_path=args.config,
                resolved_config_path=resolved_path,
                exp_dir=args.exp_dir,
                stdout_log_path=stdout_path,
                metric_scope="local_proxy_not_official",
                started_at=started_at,
                duration_seconds=time.perf_counter() - started,
                exit_code=exit_code,
                eval_artifact_paths=tuple(artifacts),
                checkpoint_path=checkpoint_path,
                error=error,
                extra={"phase": args.command, **summary},
            ),
            args.exp_dir / "run_manifest.json",
            workspace_root=REPO_ROOT,
        )
    return exit_code


def _run_candidates(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    data = config["data"]
    phase = args.phase
    if phase == "val":
        _require_passed_gate(args.gate_report)
    split_key = "train_split" if phase == "train" else "val_split"
    tube_key = "train_tube_candidates" if phase == "train" else "val_tube_candidates"
    pool_cap_key = (
        "candidate_pool_cap" if phase == "train" else "val_candidate_pool_cap"
    )
    detector_manifest: dict[str, Any] | None = None
    if args.require_tube_frame_provenance:
        if args.tube_progress_jsonl is None or args.detector_manifest is None:
            raise ValueError(
                "strict candidate construction requires --tube-progress-jsonl "
                "and --detector-manifest"
            )
        detector_manifest = _validate_detector_manifest(
            manifest_path=args.detector_manifest,
            tube_jsonl=args.tube_jsonl or Path(data[tube_key]),
            progress_jsonl=args.tube_progress_jsonl,
        )
    policies = _shortlist_policies(data)
    groups, labels, report = build_candidate_stores(
        split_json=args.split_json or Path(data[split_key]),
        tube_jsonl=args.tube_jsonl or Path(data[tube_key]),
        valdata_root=args.data_root or Path(data["valdata_root"]),
        pool_cap=args.pool_cap or int(data[pool_cap_key]),
        cap_candidates=tuple(policy.k for policy in policies),
        shortlist_policies=policies,
        oracle_tolerance=float(data["oracle_tolerance"]),
        expand_radius=float(data["expand_radius"]),
        max_candidate_time_delta=(
            args.max_candidate_time_delta
            if args.max_candidate_time_delta is not None
            else float(data["max_candidate_time_delta"])
        ),
        include_labels=phase == "train",
        frozen_k=args.shortlist_k,
        frozen_policy_name=args.shortlist_policy,
        max_split_rows=args.max_stg_rows,
        max_tube_lines=args.max_tube_lines,
        require_tube_frame_provenance=args.require_tube_frame_provenance,
        tube_progress_jsonl=args.tube_progress_jsonl,
    )
    candidate_path = args.exp_dir / "candidate_store.jsonl"
    report_path = args.exp_dir / "candidate_cap_report.json"
    candidate_hash = write_candidate_store(groups, candidate_path)
    _write_json(report_path, report.to_dict())
    summary = {
        "phase": phase,
        "candidate_store": str(candidate_path),
        "candidate_store_hash": candidate_hash,
        "group_count": len(groups),
        "cap_report": report.to_dict(),
        "selected_shortlist_policy": report.selected_policy,
        "max_candidate_time_delta": (
            args.max_candidate_time_delta
            if args.max_candidate_time_delta is not None
            else float(data["max_candidate_time_delta"])
        ),
        "val_inputs_read": phase == "val",
        "public_inputs_read": phase == "public",
        "require_tube_frame_provenance": args.require_tube_frame_provenance,
    }
    if detector_manifest is not None:
        summary.update(
            {
                "detector_manifest": str(args.detector_manifest),
                "detector_manifest_hash": sha256_file(args.detector_manifest),
                "tube_progress_jsonl": str(args.tube_progress_jsonl),
                "tube_progress_hash": sha256_file(args.tube_progress_jsonl),
                "detector_group_count": detector_manifest["group_count"],
            }
        )
    artifacts = [candidate_path, report_path]
    if phase == "train":
        label_path = args.exp_dir / "train_label_store.json"
        label_hash = LabelStore(label_path, phase="train").write(labels)
        summary["label_store"] = str(label_path)
        summary["label_store_hash"] = label_hash
        artifacts.insert(1, label_path)
    return summary, artifacts


def _validate_detector_manifest(
    *,
    manifest_path: Path,
    tube_jsonl: Path,
    progress_jsonl: Path,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("requested-time detector manifest must be a mapping")
    expected = {
        "schema_version": 2,
        "run_type": "requested_time_groundingdino",
        "status": "completed",
        "full_run_completed": True,
        "contains_gt": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "requested-time detector manifest is not a completed GT-free schema-2 run"
        )
    for key, path in (("output", tube_jsonl), ("progress", progress_jsonl)):
        value = payload.get(key)
        if not isinstance(value, str) or Path(value).resolve() != path.resolve():
            raise ValueError(f"requested-time detector {key} path mismatch")
    if payload.get("output_sha256") != sha256_file(tube_jsonl):
        raise ValueError("requested-time detector tube hash mismatch")
    if payload.get("progress_sha256") != sha256_file(progress_jsonl):
        raise ValueError("requested-time detector progress hash mismatch")
    group_count = payload.get("group_count")
    completed_count = payload.get("completed_group_count")
    progress_count = payload.get("progress_row_count")
    if (
        isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count <= 0
        or completed_count != group_count
        or progress_count != group_count
        or _nonempty_line_count(progress_jsonl) != group_count
    ):
        raise ValueError("requested-time detector group coverage is incomplete")
    if payload.get("total_tube_count") != _nonempty_line_count(tube_jsonl):
        raise ValueError("requested-time detector tube count mismatch")
    runtime = payload.get("runtime_stack_identity")
    if not isinstance(runtime, Mapping):
        raise ValueError("requested-time detector manifest lacks runtime identity")
    runtime_sections = ("python", "platform", "packages", "accelerator")
    if not all(key in runtime for key in runtime_sections):
        raise ValueError("requested-time detector runtime identity is incomplete")
    return payload


def _nonempty_line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _run_features(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[Path], Path]:
    from refinerank.cache import read_candidate_store

    if args.candidate_store is None:
        raise ValueError("features requires --candidate-store")
    if args.phase == "val":
        _require_passed_gate(args.gate_report)
    groups = read_candidate_store(args.candidate_store)
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    model_config = config["model"]
    expected_k = int(args.shortlist_k)
    configured_policy = _configured_shortlist_policy(
        config["data"],
        args.shortlist_policy,
    )
    if expected_k != configured_policy.k:
        raise ValueError("configured shortlist policy and K do not match")
    if any(len(group.candidates) > expected_k for group in groups):
        raise ValueError("CandidateStore contains more candidates than frozen K")
    model_path = Path(model_config["medvlm_path"])
    dino_checkpoint = Path(model_config["groundingdino_checkpoint"])
    checkpoint_hashes_before = _model_hashes(model_path)
    dino_hash_before = sha256_file(dino_checkpoint)
    visual_layers = tuple(int(value) for value in model_config["visual_layers"])
    manifest = {
        "model_path": str(model_path),
        "model_config_hash": sha256_file(model_path / "config.json"),
        "model_checkpoint_hashes": checkpoint_hashes_before,
        "groundingdino_checkpoint": str(dino_checkpoint),
        "groundingdino_checkpoint_hash": dino_hash_before,
        "processor_hashes": _processor_hashes(model_path),
        "prompt_template": PROMPT_TEMPLATE,
        "candidate_pool_hash": sha256_file(args.candidate_store),
        "visual_layers": list(visual_layers),
        "query_features": ["q_last", "q_mean"],
        "shortlist_k": expected_k,
        "shortlist_policy": args.shortlist_policy,
        "coordinate_space": (
            "source_absolute_xyxy_output_with_normalized_pooling_"
            "and_2.5px_boundary_snap"
        ),
        "visual_dtype": "float16_l2_normalized",
        "structure_dtype": "float32_fold_standardized_at_train_time",
        "phase": args.phase if args.max_groups is None else f"{args.phase}_smoke",
    }
    cache_root = args.exp_dir / "feature_cache"
    writer = FeatureCacheWriter(cache_root, manifest)
    extractor = FrozenMedVLMFeatureExtractor.from_pretrained(
        model_path,
        device=str(model_config["device"]),
        dtype=str(model_config["dtype"]),
        visual_layers=visual_layers,
    )
    group_paths: dict[str, Path] = {}
    for index, group in enumerate(groups, start=1):
        features = extractor.extract_group(group)
        if len(group.candidates) > 1 and not _candidate_roi_varies(features.roi_final):
            raise RuntimeError(
                f"candidate-level roi_final features do not vary in {group.group_key}"
            )
        group_paths[group.group_key] = writer.write_group(group, features)
        print(
            f"[frozen-stg-cache] extracted {index}/{len(groups)} {group.group_key}",
            flush=True,
        )
    index_path = writer.write_index(groups, group_paths)
    checkpoint_hashes_after = _model_hashes(model_path)
    dino_hash_after = sha256_file(dino_checkpoint)
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("MedVLM checkpoint hashes changed during extraction")
    if dino_hash_before != dino_hash_after:
        raise RuntimeError("GroundingDINO checkpoint hash changed during extraction")
    summary_path = args.exp_dir / "feature_cache_summary.json"
    summary = {
        "feature_cache": str(cache_root),
        "manifest_hash": writer.manifest["manifest_hash"],
        "group_count": len(groups),
        "medvlm_checkpoint_hashes_unchanged": True,
        "groundingdino_checkpoint_hash_unchanged": True,
        "val_inputs_read": args.phase == "val",
        "public_inputs_read": args.phase == "public",
    }
    _write_json(summary_path, summary)
    return summary, [cache_root / "manifest.json", index_path, summary_path], model_path


def _run_spatial_features(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[Path], Path]:
    """Build raw l31/final grids and prove reference ROI reconstruction."""

    from refinerank.cache import read_candidate_store

    if args.phase not in {"train", "public"}:
        raise PermissionError("spatial-grid extraction supports train or public")
    groups = read_candidate_store(args.candidate_store)
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    configured_policy = _configured_shortlist_policy(
        config["data"], args.shortlist_policy
    )
    if int(args.shortlist_k) != configured_policy.k:
        raise ValueError("configured spatial shortlist policy and K do not match")
    if any(len(group.candidates) > configured_policy.k for group in groups):
        raise ValueError("CandidateStore contains more candidates than frozen K")
    reference = FeatureCacheReader(args.reference_feature_cache)
    if reference.manifest.get("phase") != args.phase:
        raise PermissionError(
            "spatial-grid reference FeatureStore phase must match extraction phase"
        )
    candidate_hash = sha256_file(args.candidate_store)
    if reference.manifest.get("candidate_pool_hash") != candidate_hash:
        raise ValueError("reference FeatureStore candidate pool hash mismatch")
    model_config = config["model"]
    model_path = Path(model_config["medvlm_path"])
    dino_checkpoint = Path(model_config["groundingdino_checkpoint"])
    checkpoint_hashes_before = _model_hashes(model_path)
    dino_hash_before = sha256_file(dino_checkpoint)
    visual_layers = tuple(int(value) for value in model_config["visual_layers"])
    manifest = {
        "model_path": str(model_path),
        "model_config_hash": sha256_file(model_path / "config.json"),
        "model_checkpoint_hashes": checkpoint_hashes_before,
        "groundingdino_checkpoint": str(dino_checkpoint),
        "groundingdino_checkpoint_hash": dino_hash_before,
        "processor_hashes": _processor_hashes(model_path),
        "prompt_template": PROMPT_TEMPLATE,
        "candidate_pool_hash": candidate_hash,
        "reference_feature_manifest_hash": reference.manifest["manifest_hash"],
        "visual_layers": list(visual_layers),
        "shortlist_k": int(args.shortlist_k),
        "shortlist_policy": args.shortlist_policy,
        "coordinate_space": "normalized_source_xyxy_repooling",
        "visual_dtype": "float16_raw_spatial_grid",
        "phase": (
            args.phase if args.max_groups is None else f"{args.phase}_smoke"
        ),
    }
    cache_root = args.exp_dir / "spatial_grid_cache"
    writer = SpatialGridCacheWriter(cache_root, manifest)
    extractor = FrozenMedVLMFeatureExtractor.from_pretrained(
        model_path,
        device=str(model_config["device"]),
        dtype=str(model_config["dtype"]),
        visual_layers=visual_layers,
    )
    reconstruction_atol = float(
        config.get("spatial_cache", {}).get("reconstruction_atol", 0.005)
    )
    moved_roi_min_l2 = float(
        config.get("spatial_cache", {}).get("moved_roi_min_l2", 1.0e-5)
    )
    moved_fraction_min = float(
        config.get("spatial_cache", {}).get(
            "moved_roi_changed_fraction_min", 0.99
        )
    )
    if not 0.0 < moved_fraction_min <= 1.0:
        raise ValueError("moved ROI changed-fraction minimum must be within (0,1]")
    group_paths: dict[str, Path] = {}
    final_errors: list[float] = []
    l31_errors: list[float] = []
    l7_errors: list[float] = []
    fresh_final_errors: list[float] = []
    fresh_l31_errors: list[float] = []
    fresh_l7_errors: list[float] = []
    moved_deltas: list[float] = []
    for index, group in enumerate(groups, start=1):
        pooled, spatial = extractor.extract_group_with_spatial(group)
        expected = reference.read_group(group.group_key)
        if pooled.candidate_ids != expected.candidate_ids:
            raise ValueError("spatial extraction candidate order mismatch")
        if pooled.roi_l31 is None or expected.roi_l31 is None:
            raise ValueError("spatial reconstruction requires roi_l31")
        if 7 in visual_layers and (
            pooled.roi_l7 is None
            or expected.roi_l7 is None
            or spatial.grid_l7 is None
        ):
            raise ValueError("l7 spatial reconstruction requires roi_l7 and grid_l7")
        fresh_final_errors.append(
            float(
                np.max(
                    np.abs(
                        pooled.roi_final.astype(np.float32)
                        - expected.roi_final.astype(np.float32)
                    )
                )
            )
        )
        fresh_l31_errors.append(
            float(
                np.max(
                    np.abs(
                        pooled.roi_l31.astype(np.float32)
                        - expected.roi_l31.astype(np.float32)
                    )
                )
            )
        )
        if pooled.roi_l7 is not None and expected.roi_l7 is not None:
            fresh_l7_errors.append(
                float(
                    np.max(
                        np.abs(
                            pooled.roi_l7.astype(np.float32)
                            - expected.roi_l7.astype(np.float32)
                        )
                    )
                )
            )
        boxes = tuple(item.structure[6:10] for item in group.candidates)
        repooled_final, repooled_l31, repooled_l7 = pool_spatial_rois_with_l7(
            spatial, boxes
        )
        final_errors.append(
            float(
                np.max(
                    np.abs(
                        repooled_final.astype(np.float32)
                        - expected.roi_final.astype(np.float32)
                    )
                )
            )
        )
        l31_errors.append(
            float(
                np.max(
                    np.abs(
                        repooled_l31.astype(np.float32)
                        - expected.roi_l31.astype(np.float32)
                    )
                )
            )
        )
        if repooled_l7 is not None and expected.roi_l7 is not None:
            l7_errors.append(
                float(
                    np.max(
                        np.abs(
                            repooled_l7.astype(np.float32)
                            - expected.roi_l7.astype(np.float32)
                        )
                    )
                )
            )
        moved_box = _moved_box(boxes[0], grid_thw=spatial.grid_thw)
        moved_final, moved_l31, moved_l7 = pool_spatial_rois_with_l7(
            spatial, (moved_box,)
        )
        moved_deltas.append(
            max(
                float(
                    np.linalg.norm(
                        moved_final[0].astype(np.float32)
                        - repooled_final[0].astype(np.float32)
                    )
                ),
                float(
                    np.linalg.norm(
                        moved_l31[0].astype(np.float32)
                        - repooled_l31[0].astype(np.float32)
                    )
                ),
                *(
                    [
                        float(
                            np.linalg.norm(
                                moved_l7[0].astype(np.float32)
                                - repooled_l7[0].astype(np.float32)
                            )
                        )
                    ]
                    if moved_l7 is not None and repooled_l7 is not None
                    else []
                ),
            )
        )
        group_paths[group.group_key] = writer.write_group(group, spatial)
        print(
            f"[frozen-spatial-cache] extracted {index}/{len(groups)} "
            f"{group.group_key}",
            flush=True,
        )
    max_reconstruction = max((*final_errors, *l31_errors, *l7_errors))
    max_fresh = max((*fresh_final_errors, *fresh_l31_errors, *fresh_l7_errors))
    moved_min = min(moved_deltas)
    moved_changed_count = sum(
        value > moved_roi_min_l2 for value in moved_deltas
    )
    moved_changed_fraction = moved_changed_count / len(moved_deltas)
    if max_reconstruction > reconstruction_atol or max_fresh > reconstruction_atol:
        raise RuntimeError(
            "spatial-grid reconstruction exceeds the configured tolerance"
        )
    if moved_changed_fraction < moved_fraction_min:
        raise RuntimeError(
            "moved ROI feature-change coverage is below the configured minimum"
        )
    index_path = writer.write_index(groups, group_paths)
    if checkpoint_hashes_before != _model_hashes(model_path):
        raise RuntimeError("MedVLM checkpoint hashes changed during spatial extraction")
    if dino_hash_before != sha256_file(dino_checkpoint):
        raise RuntimeError(
            "GroundingDINO checkpoint hash changed during spatial extraction"
        )
    summary = {
        "spatial_grid_cache": str(cache_root),
        "manifest_hash": writer.manifest["manifest_hash"],
        "reference_feature_manifest_hash": reference.manifest["manifest_hash"],
        "group_count": len(groups),
        "reconstruction_max_abs": max_reconstruction,
        "l7_reconstruction_max_abs": max(l7_errors, default=None),
        "l7_spatial_grid_present": bool(l7_errors),
        "fresh_forward_reference_max_abs": max_fresh,
        "reconstruction_atol": reconstruction_atol,
        "moved_roi_l2_min": moved_min,
        "moved_roi_min_l2": moved_roi_min_l2,
        "moved_roi_changed_count": moved_changed_count,
        "moved_roi_changed_fraction": moved_changed_fraction,
        "moved_roi_changed_fraction_min": moved_fraction_min,
        "coordinate_reconstruction_passed": True,
        "medvlm_checkpoint_hashes_unchanged": True,
        "groundingdino_checkpoint_hash_unchanged": True,
        "val_inputs_read": False,
        "public_inputs_read": args.phase == "public",
    }
    summary_path = args.exp_dir / "spatial_grid_cache_summary.json"
    _write_json(summary_path, summary)
    return summary, [cache_root / "manifest.json", index_path, summary_path], model_path


def _model_hashes(model_path: Path) -> dict[str, str]:
    files = sorted(model_path.glob("model-*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no MedVLM checkpoint shards under {model_path}")
    return {path.name: sha256_file(path) for path in files}


def _processor_hashes(model_path: Path) -> dict[str, str]:
    names = (
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    )
    paths = [model_path / name for name in names]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"processor file missing: {missing[0]}")
    return {path.name: sha256_file(path) for path in paths}


def _candidate_roi_varies(values: Any) -> bool:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    return bool(np.max(np.linalg.norm(array - array[:1], axis=1)) > 1.0e-5)


def _moved_box(
    box: Any,
    *,
    grid_thw: tuple[int, int, int],
) -> tuple[float, float, float, float]:
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("moved ROI smoke requires one normalized xyxy box")
    if len(grid_thw) != 3 or any(int(value) <= 0 for value in grid_thw):
        raise ValueError("moved ROI smoke requires positive grid_thw dimensions")
    _grid_t, grid_height, grid_width = (int(value) for value in grid_thw)
    width = float(values[2] - values[0])
    height = float(values[3] - values[1])
    x_step = max(0.25 * width, 1.0 / grid_width)
    y_step = max(0.25 * height, 1.0 / grid_height)
    moves = (
        (x_step, 0.0),
        (-x_step, 0.0),
        (0.0, y_step),
        (0.0, -y_step),
    )
    for dx, dy in moves:
        moved = values + np.asarray((dx, dy, dx, dy))
        if moved[0] >= 0.0 and moved[1] >= 0.0 and moved[2] <= 1.0 and moved[3] <= 1.0:
            return tuple(float(value) for value in moved)
    center = (values[:2] + values[2:]) * 0.5
    half = (values[2:] - values[:2]) * 0.45
    shrunk = np.concatenate((center - half, center + half))
    return tuple(float(value) for value in shrunk)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("ranker config root must be a mapping")
    for section in ("model", "data", "proposal_adapter", "train", "decoder"):
        if not isinstance(loaded.get(section), dict):
            raise ValueError(f"ranker config section {section!r} is required")
    return loaded


def _shortlist_policies(data: dict[str, Any]) -> tuple[ShortlistPolicy, ...]:
    raw_policies = data.get("shortlist_policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        legacy = data.get("cap_candidates")
        if not isinstance(legacy, list) or not legacy:
            raise ValueError("data.shortlist_policies must be a non-empty list")
        return tuple(
            ShortlistPolicy(
                name=f"native_k{int(value)}",
                strategy="native",
                k=int(value),
            )
            for value in legacy
        )
    policies: list[ShortlistPolicy] = []
    for index, raw in enumerate(raw_policies):
        if not isinstance(raw, dict):
            raise TypeError(f"data.shortlist_policies[{index}] must be a mapping")
        policies.append(
            ShortlistPolicy(
                name=str(raw.get("name") or ""),
                strategy=str(raw.get("strategy") or ""),
                k=int(raw.get("k", 0)),
                native_anchor_count=int(raw.get("native_anchor_count", 0)),
                diversity_alpha=float(raw.get("diversity_alpha", 0.5)),
            )
        )
    return tuple(policies)


def _configured_shortlist_policy(
    data: dict[str, Any],
    policy_name: str,
) -> ShortlistPolicy:
    policies = _shortlist_policies(data)
    for policy in policies:
        if policy.name == policy_name:
            return policy
    raise ValueError(f"unknown configured shortlist policy {policy_name!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _require_passed_gate(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise PermissionError("val cache construction requires --gate-report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("decision") != "PASS_FULL_TRAIN":
        raise PermissionError("val cache construction requires PASS_FULL_TRAIN")
    if report.get("selected_visual_variant") not in {
        "final",
        "final_l31",
        "final_l31_l23",
    }:
        raise ValueError("gate report does not freeze a visual variant")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/refinerank.yaml",
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidates = subparsers.add_parser("candidates")
    candidates.add_argument(
        "--phase", choices=("train", "val", "public"), default="train"
    )
    candidates.add_argument("--shortlist-k", type=int)
    candidates.add_argument("--shortlist-policy")
    candidates.add_argument("--gate-report", type=Path)
    candidates.add_argument("--max-stg-rows", type=int)
    candidates.add_argument("--max-tube-lines", type=int)
    candidates.add_argument("--split-json", type=Path)
    candidates.add_argument("--tube-jsonl", type=Path)
    candidates.add_argument("--tube-progress-jsonl", type=Path)
    candidates.add_argument("--detector-manifest", type=Path)
    candidates.add_argument("--data-root", type=Path)
    candidates.add_argument("--pool-cap", type=int)
    candidates.add_argument("--max-candidate-time-delta", type=float)
    candidates.add_argument("--require-tube-frame-provenance", action="store_true")
    features = subparsers.add_parser("features")
    features.add_argument(
        "--phase", choices=("train", "val", "public"), default="train"
    )
    features.add_argument("--gate-report", type=Path)
    features.add_argument("--candidate-store", type=Path, required=True)
    features.add_argument("--shortlist-k", type=int, required=True)
    features.add_argument("--shortlist-policy", required=True)
    features.add_argument("--max-groups", type=int)
    spatial = subparsers.add_parser("spatial-features")
    spatial.add_argument("--phase", choices=("train", "public"), default="train")
    spatial.add_argument("--candidate-store", type=Path, required=True)
    spatial.add_argument("--reference-feature-cache", type=Path, required=True)
    spatial.add_argument("--shortlist-k", type=int, required=True)
    spatial.add_argument("--shortlist-policy", required=True)
    spatial.add_argument("--max-groups", type=int)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
