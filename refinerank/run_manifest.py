"""Run manifest and git snapshot helpers for smoke and training runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GitSnapshot:
    """Serializable git state for a smoke or training run."""

    commit: str | None
    branch: str | None
    is_dirty: bool
    status_short: str
    dirty_patch_path: Path | None = None


@dataclass(frozen=True)
class RunContext:
    """Inputs needed to write a run_manifest.json artifact."""

    run_type: str
    command: str
    config_path: Path
    resolved_config_path: Path
    exp_dir: Path
    stdout_log_path: Path
    metric_scope: str
    started_at: str
    duration_seconds: float
    exit_code: int
    eval_artifact_paths: tuple[Path, ...] = ()
    checkpoint_path: Path | None = None
    error: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def collect_git_snapshot(
    workspace_root: Path,
    output_dir: Path | None = None,
) -> GitSnapshot:
    """Collect git commit, branch, status, and optional dirty patch."""

    root = Path(workspace_root)
    commit = _git(root, ["rev-parse", "HEAD"], allow_fail=True)
    branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"], allow_fail=True)
    status_short = _git(root, ["status", "--short"], allow_fail=True) or ""
    dirty_patch_path = None
    if status_short and output_dir is not None:
        git_status_dir = Path(output_dir) / "git_status"
        git_status_dir.mkdir(parents=True, exist_ok=True)
        patch_text = _git(root, ["diff"], allow_fail=True) or ""
        if patch_text:
            dirty_patch_path = git_status_dir / "dirty.patch"
            dirty_patch_path.write_text(patch_text, encoding="utf-8")
    return GitSnapshot(
        commit=commit,
        branch=branch,
        is_dirty=bool(status_short.strip()),
        status_short=status_short,
        dirty_patch_path=dirty_patch_path,
    )


def write_run_manifest(
    context: RunContext,
    output_path: Path,
    *,
    workspace_root: Path = Path("."),
) -> Path:
    """Write a run manifest plus git snapshot JSON."""

    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    git_snapshot = collect_git_snapshot(workspace_root, manifest_path.parent)
    git_status_dir = manifest_path.parent / "git_status"
    git_status_dir.mkdir(parents=True, exist_ok=True)
    git_json_path = git_status_dir / "git.json"
    git_json_path.write_text(
        json.dumps(
            {
                "commit": git_snapshot.commit,
                "branch": git_snapshot.branch,
                "is_dirty": git_snapshot.is_dirty,
                "status_short": git_snapshot.status_short,
                "dirty_patch_path": _path_for_json(
                    git_snapshot.dirty_patch_path,
                    workspace_root,
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    payload = {
        "artifact_contract_version": "1.0",
        "run_type": context.run_type,
        "command": context.command,
        "config_path": _path_for_json(context.config_path, workspace_root),
        "resolved_config_path": _path_for_json(
            context.resolved_config_path,
            workspace_root,
        ),
        "exp_dir": _path_for_json(context.exp_dir, workspace_root),
        "stdout_log_path": _path_for_json(context.stdout_log_path, workspace_root),
        "git_snapshot_path": _path_for_json(git_json_path, workspace_root),
        "git_commit": git_snapshot.commit,
        "git_is_dirty": git_snapshot.is_dirty,
        "checkpoint_path": _path_for_json(context.checkpoint_path, workspace_root),
        "eval_artifact_paths": [
            _path_for_json(path, workspace_root) for path in context.eval_artifact_paths
        ],
        "metric_scope": context.metric_scope,
        "official_evaluator_verified": context.metric_scope == "official",
        "started_at": context.started_at,
        "duration_seconds": context.duration_seconds,
        "exit_code": context.exit_code,
        "error": context.error,
        "extra": dict(context.extra),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _git(root: Path, args: list[str], *, allow_fail: bool) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=not allow_fail,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        if allow_fail:
            return None
        raise
    text = completed.stdout.strip()
    return text if text else None


def _path_for_json(path: Path | None, workspace_root: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(Path(workspace_root).resolve()))
    except ValueError:
        return str(resolved)
