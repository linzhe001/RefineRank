"""Train RefineNet on all train groups or run public STG prediction.

Trimmed from the research repo's ``scripts/train_frozen_stg_ranker.py``.
Only the final-submission commands are kept:

- ``proposal-full``    : fit the QueryConditionedProposalAdapter on all train
                         groups and write the deployment checkpoint.
- ``proposal-predict`` : score candidates, repool exact ROIs, and decode
                         MedVidBench public STG prediction rows.

All ablation/comparison subcommands (oof, dual-oof, distill-oof,
nested/hierarchical variants, ablation-holdout, full, predict) are
intentionally excluded from this package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

from refinerank.cache import (  # noqa: E402
    FeatureCacheReader,
    SpatialGridCacheReader,
    feature_contract_digest,
)
from refinerank.decode import (  # noqa: E402
    build_source_indexed_stg_prediction_rows,
)
from refinerank.proposal import (  # noqa: E402
    ProposalAdapterConfig,
    ProposalTarget,
    build_proposal_target_store,
    load_proposal_targets,
)
from refinerank.proposal_training import (  # noqa: E402
    predict_proposal_adapter_checkpoint,
    train_full_proposal_adapter,
)
from refinerank.repool import repool_proposals  # noqa: E402
from refinerank.run_manifest import RunContext, write_run_manifest  # noqa: E402
from refinerank.train_data import (  # noqa: E402
    CachedTrainingGroup,
    load_inference_groups,
    load_training_groups,
)
from refinerank.training import OOFConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Execute one unique, manifest-backed RefineNet phase."""

    args = _parser().parse_args(argv)
    config = _load_config(args.config)
    args.exp_dir.mkdir(parents=True, exist_ok=False)
    resolved_path = args.exp_dir / "resolved_config.json"
    stdout_path = args.exp_dir / "stdout+stderr.log"
    resolved_config = {
        "command": args.command,
        "config": config,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    _write_json(resolved_path, resolved_config)
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    exit_code = 1
    error: str | None = None
    checkpoint_path: Path | None = None
    artifacts: list[Path] = []
    summary: dict[str, Any] = {}
    try:
        if args.command == "proposal-full":
            exit_code, summary, artifacts, checkpoint_path = _run_proposal_full(
                args, config
            )
        elif args.command == "proposal-predict":
            exit_code, summary, artifacts = _run_proposal_predict(args, config)
            checkpoint_path = args.proposal_checkpoint
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
                run_type=f"refinerank_{args.command}",
                command=" ".join(sys.argv),
                config_path=args.config,
                resolved_config_path=resolved_path,
                exp_dir=args.exp_dir,
                stdout_log_path=stdout_path,
                metric_scope=(
                    "train_only_oof"
                    if args.command == "proposal-full"
                    else "online_public_prediction_only_inference"
                ),
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


def _run_proposal_full(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any], list[Path], Path]:
    """Fit the selected Proposal Repair quality model on all train groups."""

    if not args.operator_authorized:
        raise PermissionError("proposal full fit requires --operator-authorized")
    feature_reader = FeatureCacheReader(args.feature_cache)
    if feature_reader.manifest.get("phase") != "train":
        raise PermissionError("proposal full fit requires train features")
    if feature_reader.manifest.get("contains_gt") is not False:
        raise ValueError("proposal full fit FeatureStore must be explicitly GT-free")
    train_split = args.train_split or Path(config["data"]["train_split"])
    tracked_inputs = {
        "feature_manifest": args.feature_cache / "manifest.json",
        "feature_index": args.feature_cache / "index.json",
        "candidate_store": args.candidate_store,
        "label_store": args.label_store,
        "train_split": train_split,
    }
    hashes_before = {name: _sha256(path) for name, path in tracked_inputs.items()}
    groups = load_training_groups(
        feature_cache=args.feature_cache,
        candidate_store=args.candidate_store,
        label_store=args.label_store,
    )
    target_path = args.exp_dir / "proposal_target_store.json"
    target_audit = build_proposal_target_store(
        groups,
        train_split=train_split,
        output_path=target_path,
    )
    target_audit_path = args.exp_dir / "proposal_target_audit.json"
    _write_json(target_audit_path, target_audit)
    source_split_hash = str(target_audit.get("source_split_hash") or "")
    if not source_split_hash:
        raise ValueError("proposal target audit lacks source_split_hash")
    targets = load_proposal_targets(
        groups,
        target_store=target_path,
        expected_source_split_hash=source_split_hash,
    )
    contract_hash = feature_contract_digest(feature_reader.manifest)
    proposal_checkpoint = args.exp_dir / "checkpoints/proposal_adapter_full.pt"
    proposal_checkpoint, proposal_trace = train_full_proposal_adapter(
        groups,
        targets,
        proposal_checkpoint,
        operator_authorized=True,
        source_feature_contract_hash=contract_hash,
        adapter_config=ProposalAdapterConfig(**config["proposal_adapter"]),
        oof_config=OOFConfig(**config["train"]),
        device=args.device,
    )
    hashes_after = {name: _sha256(path) for name, path in tracked_inputs.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("proposal full fit mutated frozen inputs")
    deployment = {
        "schema_version": 1,
        "decision": "PROPOSAL_REPAIR_FULL_FIT_COMPLETE",
        "model_kind": "proposal_repair_quality",
        "selector_kind": "repair_quality",
        "repair_variant": "full",
        "query_mode": "q_last",
        "decoder_mode": "fixed_current",
        "operator_authorized": True,
        "fit_protocol": "fixed_fold_selected_then_all_train_groups_full_fit",
        "source_selection_report": "docs/40_iterations/iter130.md",
        "proposal_target_store": str(target_path),
        "feature_contract_hash": contract_hash,
        "train_group_count": len(groups),
        "proposal_checkpoint": str(proposal_checkpoint),
        "proposal_checkpoint_sha256": _sha256(proposal_checkpoint),
        "proposal_training_trace": proposal_trace,
        "input_hashes": hashes_before,
        "frozen_inputs_unchanged": True,
        "val_inputs_read": 0,
        "public_inputs_read": 0,
        "official_leaderboard_claim": False,
    }
    deployment_path = args.exp_dir / "deployment_manifest.json"
    _write_json(deployment_path, deployment)
    return (
        0,
        deployment,
        [target_path, target_audit_path, deployment_path, proposal_checkpoint],
        proposal_checkpoint,
    )


def _run_proposal_predict(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[int, dict[str, Any], list[Path]]:
    """Run Proposal Repair quality and the fixed decoder on public STG groups."""

    deployment = json.loads(args.deployment_manifest.read_text(encoding="utf-8"))
    if deployment.get("decision") != "PROPOSAL_REPAIR_FULL_FIT_COMPLETE":
        raise PermissionError("proposal prediction needs a completed deployment")
    if deployment.get("model_kind") != "proposal_repair_quality":
        raise ValueError("deployment manifest is not Proposal Repair quality")
    feature_reader = FeatureCacheReader(args.feature_cache)
    spatial_reader = SpatialGridCacheReader(args.spatial_grid_cache)
    if feature_reader.manifest.get("phase") != "public":
        raise PermissionError("proposal public prediction needs public features")
    if feature_reader.manifest.get("contains_gt") is not False:
        raise ValueError("proposal public FeatureStore must be explicitly GT-free")
    if spatial_reader.manifest.get("phase") != "public":
        raise PermissionError("proposal public prediction needs public spatial grids")
    if spatial_reader.manifest.get("contains_gt") is not False:
        raise ValueError("proposal public SpatialGridStore must be explicitly GT-free")
    contract_hash = feature_contract_digest(feature_reader.manifest)
    if contract_hash != deployment.get("feature_contract_hash"):
        raise ValueError("public feature contract differs from full-fit train contract")
    tracked_inputs = {
        "feature_manifest": args.feature_cache / "manifest.json",
        "feature_index": args.feature_cache / "index.json",
        "spatial_manifest": args.spatial_grid_cache / "manifest.json",
        "spatial_index": args.spatial_grid_cache / "index.json",
        "candidate_store": args.candidate_store,
        "deployment_manifest": args.deployment_manifest,
        "proposal_checkpoint": args.proposal_checkpoint,
    }
    hashes_before = {name: _sha256(path) for name, path in tracked_inputs.items()}
    groups = load_inference_groups(
        feature_cache=args.feature_cache,
        candidate_store=args.candidate_store,
    )
    proposals = predict_proposal_adapter_checkpoint(
        groups,
        args.proposal_checkpoint,
        expected_source_feature_contract_hash=contract_hash,
        batch_size=int(config["train"]["eval_batch_size"]),
        device=args.device,
    )
    targets = {
        group.group_key: ProposalTarget(has_target=False, gt_box_xyxy_norm=None)
        for group in groups
    }
    repooled = repool_proposals(
        groups,
        proposals,
        targets,
        spatial_reader.read_all(),
        reconstruction_atol=float(config["spatial_cache"]["reconstruction_atol"]),
        moved_roi_min_l2=float(config["spatial_cache"]["moved_roi_min_l2"]),
        refined_roi_changed_fraction_min=float(
            config["spatial_cache"]["refined_roi_changed_fraction_min"]
        ),
    )
    scores = _proposal_quality_scores(repooled.groups)
    rows, audits = build_source_indexed_stg_prediction_rows(
        repooled.groups,
        scores,
    )
    if args.require_public_test and len(rows) != 780:
        raise ValueError(f"public STG row count must be 780, got {len(rows)}")
    prediction_path = args.exp_dir / "final_repair_public_stg_predictions.json"
    audit_path = args.exp_dir / "prediction_audit.json"
    _write_json(prediction_path, rows)
    _write_json(audit_path, audits)
    hashes_after = {name: _sha256(path) for name, path in tracked_inputs.items()}
    if hashes_before != hashes_after:
        raise RuntimeError("proposal public prediction mutated frozen inputs")
    selected_ids = [
        candidate_id
        for audit in audits
        for candidate_id in audit["selected_candidate_ids"].values()
    ]
    selected_refined_count = sum(
        candidate_id.endswith("::refined") for candidate_id in selected_ids
    )
    summary = {
        "decision": "PROPOSAL_REPAIR_PUBLIC_STG_PREDICTION_COMPLETE",
        "metric_scope": "online_public_prediction_only_inference",
        "row_count": len(rows),
        "timestamp_count": len(repooled.groups),
        "source_index_unique": len({row["source_index"] for row in rows})
        == len(rows),
        "feature_contract_hash": contract_hash,
        "repooled_audit": repooled.audit,
        "selected_candidate_count": len(selected_ids),
        "selected_refined_count": selected_refined_count,
        "selected_refined_fraction": (
            selected_refined_count / len(selected_ids) if selected_ids else 0.0
        ),
        "input_hashes": hashes_before,
        "frozen_inputs_unchanged": True,
        "contains_gt": False,
        "official_leaderboard_claim": False,
    }
    summary_path = args.exp_dir / "prediction_summary.json"
    _write_json(summary_path, summary)
    return 0, summary, [prediction_path, audit_path, summary_path]


def _proposal_quality_scores(
    groups: Sequence[CachedTrainingGroup],
) -> dict[str, tuple[float, ...]]:
    """Expose learned Proposal Repair quality as the fixed decoder score."""

    scores: dict[str, tuple[float, ...]] = {}
    for group in groups:
        values = tuple(float(value) for value in group.native_scores)
        if len(values) != len(group.candidate_ids):
            raise ValueError(f"proposal score shape mismatch for {group.group_key}")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("proposal quality scores must be calibrated to [0, 1]")
        scores[group.group_key] = values
    return scores


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("ranker config root must be a mapping")
    return loaded


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _add_store_arguments(parser: argparse.ArgumentParser, *, labels: bool) -> None:
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--candidate-store", type=Path, required=True)
    if labels:
        parser.add_argument("--label-store", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/refinerank.yaml",
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    subparsers = parser.add_subparsers(dest="command", required=True)

    proposal_full = subparsers.add_parser("proposal-full")
    _add_store_arguments(proposal_full, labels=True)
    proposal_full.add_argument(
        "--train-split",
        type=Path,
        help="train split JSON; defaults to data.train_split in the config",
    )
    proposal_full.add_argument("--operator-authorized", action="store_true")

    proposal_predict = subparsers.add_parser("proposal-predict")
    _add_store_arguments(proposal_predict, labels=False)
    proposal_predict.add_argument("--spatial-grid-cache", type=Path, required=True)
    proposal_predict.add_argument("--deployment-manifest", type=Path, required=True)
    proposal_predict.add_argument("--proposal-checkpoint", type=Path, required=True)
    proposal_predict.add_argument("--require-public-test", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
