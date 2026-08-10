#!/usr/bin/env python
"""RefineRank user-facing entry point.

Subcommands:

- ``dino-tubes`` : generate GroundingDINO tubes at requested STG times.
- ``cache``      : build candidate store, frozen MedVLM feature cache, and
                   raw spatial-grid cache (three steps, sequentially).
- ``train``      : fit RefineNet (QueryConditionedProposalAdapter) on all
                   train groups; checkpoint lands under ``checkpoints/refinenet/``.
- ``predict``    : run RefineNet + parameter-free decoder on public STG
                   groups and write MedVidBench prediction rows.
- ``test``       : run the core pytest suite.

All heavy lifting lives in the ``refinerank`` package; this file only wires
defaults from ``configs/refinerank.yaml`` into the underlying handlers.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = REPO_ROOT / "configs/refinerank.yaml"


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("config root must be a mapping")
    return loaded


def _shortlist_defaults(
    config: dict[str, Any], shortlist_k: int | None
) -> tuple[int, str]:
    """Resolve (k, policy name) from config; an explicit k wins."""

    data = config["data"]
    adapter = config.get("proposal_adapter", {})
    k = int(shortlist_k or adapter.get("shortlist_k") or 48)
    policies = data.get("shortlist_policies") or []
    for policy in policies:
        if int(policy.get("k", 0)) == k:
            return k, str(policy["name"])
    if policies:
        first = policies[0]
        return int(first["k"]), str(first["name"])
    return k, f"native_k{k}"


def _run_dino_tubes(args: argparse.Namespace) -> int:
    from refinerank import dino_tubes

    config = _load_config(args.config)
    model = config.get("model", {})
    exp_dir = (
        args.exp_dir
        or REPO_ROOT / "runs" / f"dino_tubes_{args.phase}_{_utc_stamp()}"
    )
    repo = args.groundingdino_repo or model.get("groundingdino_repo")
    dino_config = args.groundingdino_config or model.get("groundingdino_config")
    checkpoint = args.groundingdino_checkpoint or model.get("groundingdino_checkpoint")
    if repo is None or dino_config is None or checkpoint is None:
        raise SystemExit(
            "dino-tubes needs --groundingdino-repo/--groundingdino-config/"
            "--groundingdino-checkpoint (or the model.* equivalents in the config)"
        )
    argv = [
        "--candidate-store", str(args.candidate_store),
        "--query-plan-cache", str(args.query_plan_cache),
        "--output", str(exp_dir / "tubes.jsonl"),
        "--progress", str(exp_dir / "progress.jsonl"),
        "--group-parts", str(exp_dir / "group_parts"),
        "--manifest", str(exp_dir / "manifest.json"),
        "--groundingdino-repo", str(repo),
        "--groundingdino-config", str(dino_config),
        "--groundingdino-checkpoint", str(checkpoint),
        "--box-threshold", str(args.box_threshold),
        "--text-threshold", str(args.text_threshold),
        "--device", args.device,
    ]
    if args.max_groups is not None:
        argv += ["--max-groups", str(args.max_groups)]
    if args.resume:
        argv.append("--resume")
    print(f"[interface] dino-tubes phase={args.phase} exp_dir={exp_dir}", flush=True)
    return dino_tubes.main(argv)


def _run_cache(args: argparse.Namespace) -> int:
    from refinerank import cache_builder

    config = _load_config(args.config)
    shortlist_k, shortlist_policy = _shortlist_defaults(config, args.shortlist_k)
    exp_dir = args.exp_dir or REPO_ROOT / "runs" / f"cache_{args.phase}"
    candidate_store = exp_dir / "candidates" / "candidate_store.jsonl"
    feature_cache = exp_dir / "features" / "feature_cache"
    steps = [
        (
            "candidates",
            exp_dir / "candidates",
            [
                "candidates",
                "--phase", args.phase,
                "--shortlist-k", str(shortlist_k),
                "--shortlist-policy", shortlist_policy,
            ],
        ),
        (
            "features",
            exp_dir / "features",
            [
                "features",
                "--phase", args.phase,
                "--candidate-store", str(candidate_store),
                "--shortlist-k", str(shortlist_k),
                "--shortlist-policy", shortlist_policy,
            ],
        ),
        (
            "spatial-features",
            exp_dir / "spatial_features",
            [
                "spatial-features",
                "--phase", args.phase,
                "--candidate-store", str(candidate_store),
                "--reference-feature-cache", str(feature_cache),
                "--shortlist-k", str(shortlist_k),
                "--shortlist-policy", shortlist_policy,
            ],
        ),
    ]
    for name, step_exp_dir, step_argv in steps:
        argv = [
            "--config", str(args.config),
            "--exp-dir", str(step_exp_dir),
            *step_argv,
        ]
        if args.max_groups is not None and name != "candidates":
            argv += ["--max-groups", str(args.max_groups)]
        print(f"[interface] cache step={name} exp_dir={step_exp_dir}", flush=True)
        exit_code = cache_builder.main(argv)
        if exit_code != 0:
            print(f"[interface] cache step={name} failed with {exit_code}", flush=True)
            return exit_code
    print(f"[interface] cache complete: {exp_dir}", flush=True)
    return 0


def _cache_paths(cache_dir: Path, phase: str) -> dict[str, Path]:
    paths = {
        "candidate_store": cache_dir / "candidates" / "candidate_store.jsonl",
        "feature_cache": cache_dir / "features" / "feature_cache",
        "spatial_grid_cache": cache_dir / "spatial_features" / "spatial_grid_cache",
    }
    if phase == "train":
        paths["label_store"] = cache_dir / "candidates" / "train_label_store.json"
    return paths


def _run_train(args: argparse.Namespace) -> int:
    from refinerank import refinenet_cli

    config = _load_config(args.config)
    cache_dir = args.cache_dir or REPO_ROOT / "runs" / "cache_train"
    stores = _cache_paths(cache_dir, "train")
    feature_cache = args.feature_cache or stores["feature_cache"]
    candidate_store = args.candidate_store or stores["candidate_store"]
    label_store = args.label_store or stores["label_store"]
    train_split = args.train_split or Path(config["data"]["train_split"])
    exp_dir = (
        args.exp_dir
        or REPO_ROOT / "checkpoints" / "refinenet" / f"run_{_utc_stamp()}"
    )
    argv = [
        "--config", str(args.config),
        "--exp-dir", str(exp_dir),
        "--device", args.device,
        "proposal-full",
        "--feature-cache", str(feature_cache),
        "--candidate-store", str(candidate_store),
        "--label-store", str(label_store),
        "--train-split", str(train_split),
        "--operator-authorized",
    ]
    print(f"[interface] train exp_dir={exp_dir}", flush=True)
    return refinenet_cli.main(argv)


def _resolve_checkpoint(checkpoint: Path | None) -> tuple[Path, Path]:
    """Return (deployment_manifest, proposal_checkpoint).

    Supports the flat release layout
    (``checkpoints/refinenet/{deployment_manifest.json,proposal_adapter_full.pt}``)
    as well as training-run dirs
    (``checkpoints/refinenet/run_*/{deployment_manifest.json,checkpoints/proposal_adapter_full.pt}``).
    """

    if checkpoint is None:
        root = REPO_ROOT / "checkpoints" / "refinenet"
        flat_manifest = root / "deployment_manifest.json"
        flat_weights = root / "proposal_adapter_full.pt"
        if flat_manifest.is_file() and flat_weights.is_file():
            return flat_manifest, flat_weights
        runs = sorted(
            (
                path
                for path in root.glob("run_*")
                if (path / "deployment_manifest.json").is_file()
            ),
            reverse=True,
        ) if root.is_dir() else []
        if not runs:
            raise SystemExit(
                "no trained RefineNet checkpoint found under checkpoints/refinenet/; "
                "run `python interface.py train` first or pass --checkpoint"
            )
        checkpoint = runs[0]
    if checkpoint.is_dir():
        manifest = checkpoint / "deployment_manifest.json"
        weights = checkpoint / "checkpoints" / "proposal_adapter_full.pt"
        if not weights.is_file():
            weights = checkpoint / "proposal_adapter_full.pt"
    else:
        weights = checkpoint
        sibling = checkpoint.parent / "deployment_manifest.json"
        manifest = (
            sibling
            if sibling.is_file()
            else checkpoint.parent.parent / "deployment_manifest.json"
        )
    if not manifest.is_file() or not weights.is_file():
        raise SystemExit(
            f"cannot resolve deployment manifest / weights from {checkpoint}"
        )
    return manifest, weights


def _run_predict(args: argparse.Namespace) -> int:
    from refinerank import refinenet_cli

    cache_dir = args.cache_dir or REPO_ROOT / "runs" / "cache_public"
    stores = _cache_paths(cache_dir, "public")
    feature_cache = args.feature_cache or stores["feature_cache"]
    candidate_store = args.candidate_store or stores["candidate_store"]
    spatial_grid_cache = args.spatial_grid_cache or stores["spatial_grid_cache"]
    manifest, weights = _resolve_checkpoint(args.checkpoint)
    exp_dir = (
        args.exp_dir
        or REPO_ROOT / "runs" / f"predict_public_{_utc_stamp()}"
    )
    argv = [
        "--config", str(args.config),
        "--exp-dir", str(exp_dir),
        "--device", args.device,
        "proposal-predict",
        "--feature-cache", str(feature_cache),
        "--candidate-store", str(candidate_store),
        "--spatial-grid-cache", str(spatial_grid_cache),
        "--deployment-manifest", str(manifest),
        "--proposal-checkpoint", str(weights),
    ]
    if args.require_public_test:
        argv.append("--require-public-test")
    print(f"[interface] predict exp_dir={exp_dir} checkpoint={weights}", flush=True)
    return refinenet_cli.main(argv)


def _run_test(args: argparse.Namespace) -> int:
    if args.pytest_args:
        command = [sys.executable, "-m", "pytest", *args.pytest_args]
    else:
        command = [sys.executable, "-m", "pytest", "tests/", "-q"]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return completed.returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    dino = subparsers.add_parser(
        "dino-tubes",
        help="generate GroundingDINO tubes at requested STG times",
    )
    add_config(dino)
    dino.add_argument("--phase", choices=("train", "public"), required=True)
    dino.add_argument("--candidate-store", type=Path, required=True)
    dino.add_argument("--query-plan-cache", type=Path, required=True)
    dino.add_argument("--exp-dir", type=Path)
    dino.add_argument("--groundingdino-repo", type=Path)
    dino.add_argument("--groundingdino-config", type=Path)
    dino.add_argument("--groundingdino-checkpoint", type=Path)
    dino.add_argument("--box-threshold", type=float, default=0.05)
    dino.add_argument("--text-threshold", type=float, default=0.10)
    dino.add_argument("--device", default="cuda")
    dino.add_argument("--max-groups", type=int)
    dino.add_argument("--resume", action="store_true")
    dino.set_defaults(handler=_run_dino_tubes)

    cache = subparsers.add_parser(
        "cache",
        help="build candidate store + feature cache + spatial-grid cache",
    )
    add_config(cache)
    cache.add_argument("--phase", choices=("train", "public"), required=True)
    cache.add_argument("--exp-dir", type=Path)
    cache.add_argument("--shortlist-k", type=int)
    cache.add_argument("--max-groups", type=int)
    cache.set_defaults(handler=_run_cache)

    train = subparsers.add_parser(
        "train",
        help="fit RefineNet on all train groups (proposal-full)",
    )
    add_config(train)
    train.add_argument("--exp-dir", type=Path)
    train.add_argument("--cache-dir", type=Path)
    train.add_argument("--feature-cache", type=Path)
    train.add_argument("--candidate-store", type=Path)
    train.add_argument("--label-store", type=Path)
    train.add_argument("--train-split", type=Path)
    train.add_argument("--device", default="cuda")
    train.set_defaults(handler=_run_train)

    predict = subparsers.add_parser(
        "predict",
        help="run RefineNet + fixed decoder on public STG groups",
    )
    add_config(predict)
    predict.add_argument(
        "--checkpoint",
        type=Path,
        help="run dir under checkpoints/refinenet/ or a proposal_adapter_full.pt; "
        "defaults to the latest run",
    )
    predict.add_argument("--exp-dir", type=Path)
    predict.add_argument("--cache-dir", type=Path)
    predict.add_argument("--feature-cache", type=Path)
    predict.add_argument("--candidate-store", type=Path)
    predict.add_argument("--spatial-grid-cache", type=Path)
    predict.add_argument("--device", default="cuda")
    predict.add_argument(
        "--require-public-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enforce the 780-row MedVidBench public STG contract (default: on)",
    )
    predict.set_defaults(handler=_run_predict)

    test = subparsers.add_parser("test", help="run the core pytest suite")
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test.set_defaults(handler=_run_test)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
