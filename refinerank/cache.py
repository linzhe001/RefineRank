"""Hash-checked CandidateStore, FeatureStore, and train-only LabelStore."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from refinerank.types import (
    CandidateGroup,
    FrozenCandidate,
    GroupFeatures,
    SpatialGridFeatures,
)

FEATURE_CACHE_SCHEMA_VERSION = 1
SPATIAL_GRID_CACHE_SCHEMA_VERSION = 1
CANDIDATE_STORE_SCHEMA_VERSION = 2
_READABLE_CANDIDATE_STORE_SCHEMA_VERSIONS = {1, 2}
_CANDIDATE_STORE_PROVENANCE_KEYS = (
    "source_frame_id",
    "resolved_frame_timestamp",
    "alignment_error_seconds",
    "frame_alignment_exact",
    "timeline_digest",
)
LABEL_STORE_SCHEMA_VERSION = 1
PROPOSAL_TARGET_STORE_SCHEMA_VERSION = 1
_FORBIDDEN_FEATURE_KEYS = {
    "answer",
    "bbox_dict",
    "gnd",
    "ground_truth",
    "gt",
    "iou",
    "ious",
    "label",
    "labels",
    "struc_info",
    "tal",
    "uai_box",
}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible manifest, excluding a stored digest field."""

    normalized = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def feature_contract_digest(payload: Mapping[str, Any]) -> str:
    """Hash train/val-invariant frozen feature semantics for checkpoints."""

    excluded = {"manifest_hash", "candidate_pool_hash", "phase"}
    normalized = {key: value for key, value in payload.items() if key not in excluded}
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_candidate_store(groups: Sequence[CandidateGroup], path: Path) -> str:
    """Write GT-free candidate groups as deterministic JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            payload = _candidate_group_payload(group)
            _reject_forbidden_feature_keys(payload)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(output_path)


def read_candidate_store(path: Path) -> tuple[CandidateGroup, ...]:
    """Read and validate deterministic GT-free candidate JSONL."""

    groups: list[CandidateGroup] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            _reject_forbidden_feature_keys(payload)
            schema_version = payload.get("schema_version")
            if isinstance(schema_version, bool) or not isinstance(
                schema_version, int
            ):
                raise TypeError(
                    f"{path}:{line_number} candidate schema_version must be "
                    "a JSON integer"
                )
            if schema_version not in _READABLE_CANDIDATE_STORE_SCHEMA_VERSIONS:
                raise ValueError(f"{path}:{line_number} candidate schema mismatch")
            if schema_version == 1:
                unexpected = sorted(
                    key
                    for key in _CANDIDATE_STORE_PROVENANCE_KEYS
                    if key in payload
                )
                if unexpected:
                    raise ValueError(
                        f"{path}:{line_number} candidate v1 must not contain "
                        f"v2 provenance fields: {unexpected}"
                    )
            else:
                _validate_candidate_store_v2_provenance(
                    payload,
                    source=f"{path}:{line_number}",
                )
            if schema_version == 2:
                groups.append(
                    _candidate_group_from_v2_payload(
                        payload,
                        source=f"{path}:{line_number}",
                    )
                )
            else:
                groups.append(_candidate_group_from_payload(payload))
    if not groups:
        raise ValueError(f"candidate store is empty: {path}")
    return tuple(groups)


class LabelStore:
    """Train-only IoU labels kept physically separate from frozen features."""

    def __init__(self, path: Path, *, phase: str) -> None:
        if phase != "train":
            raise PermissionError("LabelStore is available only for phase='train'")
        self.path = Path(path)

    def write(
        self,
        entries: Mapping[
            str,
            Mapping[str, Sequence[str] | Sequence[float]],
        ],
    ) -> str:
        """Write candidate-aligned IoU labels."""

        for group_key, entry in entries.items():
            candidate_ids = tuple(str(value) for value in entry["candidate_ids"])
            ious = tuple(float(value) for value in entry["ious"])
            if not candidate_ids or len(candidate_ids) != len(ious):
                raise ValueError(f"LabelStore group {group_key} alignment mismatch")
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError(f"LabelStore group {group_key} has duplicate ids")
            if any(
                not np.isfinite(value) or value < 0.0 or value > 1.0 for value in ious
            ):
                raise ValueError(f"LabelStore group {group_key} has invalid IoU")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": LABEL_STORE_SCHEMA_VERSION,
            "phase": "train",
            "groups": entries,
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return sha256_file(self.path)

    def read(self) -> dict[str, dict[str, tuple[Any, ...]]]:
        """Read and validate candidate-aligned train labels."""

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != LABEL_STORE_SCHEMA_VERSION:
            raise ValueError("LabelStore schema mismatch")
        if payload.get("phase") != "train":
            raise PermissionError("LabelStore payload is not train-only")
        groups = payload.get("groups")
        if not isinstance(groups, Mapping):
            raise TypeError("LabelStore groups must be a mapping")
        output: dict[str, dict[str, tuple[Any, ...]]] = {}
        for group_key, entry in groups.items():
            if not isinstance(entry, Mapping):
                raise TypeError(f"LabelStore group {group_key} must be a mapping")
            candidate_ids = tuple(
                str(value) for value in entry.get("candidate_ids", ())
            )
            ious = tuple(float(value) for value in entry.get("ious", ()))
            if not candidate_ids or len(candidate_ids) != len(ious):
                raise ValueError(f"LabelStore group {group_key} alignment mismatch")
            output[str(group_key)] = {"candidate_ids": candidate_ids, "ious": ious}
        return output


class ProposalTargetStore:
    """Train-only requested-timestamp GT boxes for proposal supervision."""

    def __init__(self, path: Path, *, phase: str) -> None:
        if phase != "train":
            raise PermissionError(
                "ProposalTargetStore is available only for phase='train'"
            )
        self.path = Path(path)

    def write(
        self,
        entries: Mapping[str, Mapping[str, Any]],
        *,
        source_split_hash: str,
    ) -> str:
        """Write normalized GT boxes without touching the GT-free FeatureStore."""

        normalized: dict[str, dict[str, Any]] = {}
        for group_key, entry in entries.items():
            has_target = entry.get("has_target")
            if not isinstance(has_target, bool):
                raise TypeError(
                    f"ProposalTargetStore group {group_key} needs bool has_target"
                )
            raw_box = entry.get("gt_box_xyxy_norm")
            if has_target:
                if not isinstance(raw_box, Sequence) or isinstance(
                    raw_box, (str, bytes)
                ):
                    raise TypeError(
                        f"ProposalTargetStore group {group_key} needs a GT box"
                    )
                box = tuple(float(value) for value in raw_box)
                _validate_normalized_target_box(box, group_key=str(group_key))
                serialized_box: list[float] | None = list(box)
            else:
                if raw_box is not None:
                    raise ValueError(
                        f"ProposalTargetStore group {group_key} marks absent GT"
                    )
                serialized_box = None
            normalized[str(group_key)] = {
                "has_target": has_target,
                "gt_box_xyxy_norm": serialized_box,
            }
        if not normalized:
            raise ValueError("ProposalTargetStore cannot be empty")
        if not source_split_hash:
            raise ValueError("ProposalTargetStore source split hash is required")
        payload = {
            "schema_version": PROPOSAL_TARGET_STORE_SCHEMA_VERSION,
            "phase": "train",
            "contains_gt": True,
            "source_split_hash": source_split_hash,
            "groups": normalized,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return sha256_file(self.path)

    def read(
        self,
        *,
        expected_source_split_hash: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Read and validate train-only proposal targets."""

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PROPOSAL_TARGET_STORE_SCHEMA_VERSION:
            raise ValueError("ProposalTargetStore schema mismatch")
        if payload.get("phase") != "train" or payload.get("contains_gt") is not True:
            raise PermissionError("ProposalTargetStore payload is not train-only GT")
        source_hash = payload.get("source_split_hash")
        if (
            expected_source_split_hash is not None
            and source_hash != expected_source_split_hash
        ):
            raise ValueError("ProposalTargetStore source split hash mismatch")
        groups = payload.get("groups")
        if not isinstance(groups, Mapping) or not groups:
            raise TypeError("ProposalTargetStore groups must be a non-empty mapping")
        output: dict[str, dict[str, Any]] = {}
        for group_key, entry in groups.items():
            if not isinstance(entry, Mapping):
                raise TypeError(
                    f"ProposalTargetStore group {group_key} must be a mapping"
                )
            has_target = entry.get("has_target")
            if not isinstance(has_target, bool):
                raise TypeError(
                    f"ProposalTargetStore group {group_key} needs bool has_target"
                )
            raw_box = entry.get("gt_box_xyxy_norm")
            box: tuple[float, float, float, float] | None
            if has_target:
                if not isinstance(raw_box, Sequence) or isinstance(
                    raw_box, (str, bytes)
                ):
                    raise TypeError(
                        f"ProposalTargetStore group {group_key} needs a GT box"
                    )
                parsed = tuple(float(value) for value in raw_box)
                _validate_normalized_target_box(parsed, group_key=str(group_key))
                box = parsed
            else:
                if raw_box is not None:
                    raise ValueError(
                        f"ProposalTargetStore group {group_key} marks absent GT"
                    )
                box = None
            output[str(group_key)] = {
                "has_target": has_target,
                "gt_box_xyxy_norm": box,
            }
        return output


class FeatureCacheWriter:
    """Write one compressed, GT-free frozen-feature file per candidate group."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.groups_dir = self.root / "groups"
        self.root.mkdir(parents=True, exist_ok=True)
        self.groups_dir.mkdir(parents=True, exist_ok=True)
        base_manifest = dict(manifest)
        base_manifest["schema_version"] = FEATURE_CACHE_SCHEMA_VERSION
        base_manifest["contains_gt"] = False
        _reject_forbidden_feature_keys(base_manifest)
        base_manifest["manifest_hash"] = manifest_digest(base_manifest)
        self.manifest = base_manifest
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != base_manifest:
                raise ValueError("feature cache manifest mismatch; refusing resume")
        else:
            manifest_path.write_text(
                json.dumps(base_manifest, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )

    def write_group(self, group: CandidateGroup, features: GroupFeatures) -> Path:
        """Write one group after candidate-id and shape validation."""

        features.validate()
        expected_ids = tuple(item.candidate_id for item in group.candidates)
        if (
            features.group_key != group.group_key
            or features.candidate_ids != expected_ids
        ):
            raise ValueError("feature group does not align with CandidateStore")
        output_path = self.groups_dir / f"{_safe_group_name(group.group_key)}.npz"
        if output_path.exists():
            with np.load(output_path, allow_pickle=False) as arrays:
                stored_ids = tuple(str(value) for value in arrays["candidate_ids"])
            if stored_ids != expected_ids:
                raise ValueError("existing feature group candidate order mismatch")
            return output_path
        arrays: dict[str, np.ndarray] = {
            "candidate_ids": np.asarray(features.candidate_ids),
            "q_last": np.asarray(features.q_last, dtype=np.float16),
            "roi_final": np.asarray(features.roi_final, dtype=np.float16),
            "structure": np.asarray(features.structure, dtype=np.float32),
            "missing": np.asarray(features.missing, dtype=np.float32),
            "frame_size": np.asarray(features.frame_size, dtype=np.int32),
            "grid_thw": np.asarray(features.grid_thw, dtype=np.int32),
        }
        if features.q_mean is not None:
            arrays["q_mean"] = np.asarray(features.q_mean, dtype=np.float16)
        if features.roi_l7 is not None:
            arrays["roi_l7"] = np.asarray(features.roi_l7, dtype=np.float16)
        if features.roi_l15 is not None:
            arrays["roi_l15"] = np.asarray(features.roi_l15, dtype=np.float16)
        if features.roi_l23 is not None:
            arrays["roi_l23"] = np.asarray(features.roi_l23, dtype=np.float16)
        if features.roi_l31 is not None:
            arrays["roi_l31"] = np.asarray(features.roi_l31, dtype=np.float16)
        np.savez_compressed(output_path, **arrays)
        return output_path

    def write_index(
        self,
        groups: Sequence[CandidateGroup],
        group_paths: Mapping[str, Path],
    ) -> Path:
        """Write the GT-free cache index after all requested groups finish."""

        rows = []
        for group in groups:
            path = group_paths.get(group.group_key)
            if path is None or not path.exists():
                raise FileNotFoundError(f"feature file missing for {group.group_key}")
            with np.load(path, allow_pickle=False) as arrays:
                grid_thw = [int(value) for value in arrays["grid_thw"]]
            rows.append(
                {
                    "group_key": group.group_key,
                    "sample_id": group.sample_id,
                    "original_id": group.original_id,
                    "video_id": group.video_id,
                    "row_index": group.row_index,
                    "dataset": group.dataset,
                    "requested_timestamp": group.requested_timestamp,
                    "candidate_count": len(group.candidates),
                    "frame_size": list(group.frame_size),
                    "grid_thw": grid_thw,
                    "feature_path": str(path.relative_to(self.root)),
                    "feature_hash": sha256_file(path),
                }
            )
        payload = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "contains_gt": False,
            "manifest_hash": self.manifest["manifest_hash"],
            "groups": rows,
        }
        _reject_forbidden_feature_keys(payload)
        index_path = self.root / "index.json"
        index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return index_path


class FeatureCacheReader:
    """Read hash-checked frozen features with fail-closed source validation."""

    def __init__(
        self,
        root: Path,
        *,
        expected_manifest_hash: str | None = None,
        expected_sources: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError("feature cache schema mismatch")
        actual_hash = manifest_digest(self.manifest)
        if actual_hash != self.manifest.get("manifest_hash"):
            raise ValueError("feature cache manifest hash is invalid")
        if expected_manifest_hash is not None and actual_hash != expected_manifest_hash:
            raise ValueError("feature cache manifest hash mismatch")
        for key, value in (expected_sources or {}).items():
            if self.manifest.get(key) != value:
                raise ValueError(f"feature cache source mismatch for {key}")
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.index.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError("feature cache index schema mismatch")
        if self.index.get("contains_gt") is not False:
            raise ValueError("feature cache index must explicitly exclude GT")
        if self.index.get("manifest_hash") != actual_hash:
            raise ValueError("feature cache index points to a different manifest")
        if not isinstance(self.index.get("groups"), list) or not self.index["groups"]:
            raise ValueError("feature cache index has no groups")
        self._rows = {
            str(row["group_key"]): row for row in self.index.get("groups", [])
        }
        if len(self._rows) != len(self.index["groups"]):
            raise ValueError("feature cache index has duplicate group keys")

    def iter_groups(self) -> Iterable[tuple[Mapping[str, Any], GroupFeatures]]:
        """Yield index metadata and verified group tensors."""

        for row in self.index.get("groups", []):
            yield row, self.read_group(str(row["group_key"]))

    def read_group(self, group_key: str) -> GroupFeatures:
        """Read one verified group without materializing the entire cache."""

        row = self._rows.get(group_key)
        if row is None:
            raise KeyError(f"feature cache has no group {group_key}")
        path = self.root / row["feature_path"]
        if sha256_file(path) != row["feature_hash"]:
            raise ValueError(f"feature hash mismatch for {group_key}")
        with np.load(path, allow_pickle=False) as arrays:
            features = GroupFeatures(
                group_key=group_key,
                candidate_ids=tuple(str(value) for value in arrays["candidate_ids"]),
                q_last=arrays["q_last"],
                q_mean=arrays["q_mean"] if "q_mean" in arrays else None,
                roi_l7=arrays["roi_l7"] if "roi_l7" in arrays else None,
                roi_l15=arrays["roi_l15"] if "roi_l15" in arrays else None,
                roi_l23=arrays["roi_l23"] if "roi_l23" in arrays else None,
                roi_l31=arrays["roi_l31"] if "roi_l31" in arrays else None,
                roi_final=arrays["roi_final"],
                structure=arrays["structure"],
                missing=arrays["missing"],
                frame_size=tuple(int(value) for value in arrays["frame_size"]),
                grid_thw=tuple(int(value) for value in arrays["grid_thw"]),
            )
        features.validate()
        return features


class SpatialGridCacheWriter:
    """Write GT-free FP16 l31/final grids independently of LabelStore."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.groups_dir = self.root / "groups"
        self.root.mkdir(parents=True, exist_ok=True)
        self.groups_dir.mkdir(parents=True, exist_ok=True)
        base_manifest = dict(manifest)
        base_manifest["schema_version"] = SPATIAL_GRID_CACHE_SCHEMA_VERSION
        base_manifest["contains_gt"] = False
        _reject_forbidden_feature_keys(base_manifest)
        base_manifest["manifest_hash"] = manifest_digest(base_manifest)
        self.manifest = base_manifest
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != base_manifest:
                raise ValueError("spatial-grid manifest mismatch; refusing resume")
        else:
            manifest_path.write_text(
                json.dumps(base_manifest, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )

    def write_group(
        self,
        group: CandidateGroup,
        features: SpatialGridFeatures,
    ) -> Path:
        """Write one raw grid after source-group and shape validation."""

        features.validate()
        if features.group_key != group.group_key:
            raise ValueError("spatial-grid group does not align with CandidateStore")
        if features.frame_size != group.frame_size:
            raise ValueError(
                "spatial-grid frame size does not align with CandidateStore"
            )
        output_path = self.groups_dir / f"{_safe_group_name(group.group_key)}.npz"
        if output_path.exists():
            with np.load(output_path, allow_pickle=False) as arrays:
                stored_thw = tuple(int(value) for value in arrays["grid_thw"])
            if stored_thw != features.grid_thw:
                raise ValueError("existing spatial-grid dimensions do not match")
            return output_path
        np.savez_compressed(
            output_path,
            grid_l31=np.asarray(features.grid_l31, dtype=np.float16),
            grid_final=np.asarray(features.grid_final, dtype=np.float16),
            frame_size=np.asarray(features.frame_size, dtype=np.int32),
            grid_thw=np.asarray(features.grid_thw, dtype=np.int32),
            **(
                {"grid_l7": np.asarray(features.grid_l7, dtype=np.float16)}
                if features.grid_l7 is not None
                else {}
            ),
        )
        return output_path

    def write_index(
        self,
        groups: Sequence[CandidateGroup],
        group_paths: Mapping[str, Path],
    ) -> Path:
        """Write the complete hash-checked spatial-grid index."""

        rows = []
        for group in groups:
            path = group_paths.get(group.group_key)
            if path is None or not path.exists():
                raise FileNotFoundError(f"spatial grid missing for {group.group_key}")
            with np.load(path, allow_pickle=False) as arrays:
                grid_thw = [int(value) for value in arrays["grid_thw"]]
            rows.append(
                {
                    "group_key": group.group_key,
                    "sample_id": group.sample_id,
                    "original_id": group.original_id,
                    "video_id": group.video_id,
                    "row_index": group.row_index,
                    "dataset": group.dataset,
                    "requested_timestamp": group.requested_timestamp,
                    "frame_size": list(group.frame_size),
                    "grid_thw": grid_thw,
                    "spatial_path": str(path.relative_to(self.root)),
                    "spatial_hash": sha256_file(path),
                }
            )
        payload = {
            "schema_version": SPATIAL_GRID_CACHE_SCHEMA_VERSION,
            "contains_gt": False,
            "manifest_hash": self.manifest["manifest_hash"],
            "groups": rows,
        }
        _reject_forbidden_feature_keys(payload)
        index_path = self.root / "index.json"
        index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return index_path


class SpatialGridCacheReader:
    """Read hash-checked frozen spatial grids and reject source drift."""

    def __init__(
        self,
        root: Path,
        *,
        expected_manifest_hash: str | None = None,
        expected_sources: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != SPATIAL_GRID_CACHE_SCHEMA_VERSION:
            raise ValueError("spatial-grid cache schema mismatch")
        if self.manifest.get("contains_gt") is not False:
            raise ValueError("spatial-grid manifest must explicitly exclude GT")
        actual_hash = manifest_digest(self.manifest)
        if actual_hash != self.manifest.get("manifest_hash"):
            raise ValueError("spatial-grid manifest hash is invalid")
        if expected_manifest_hash is not None and actual_hash != expected_manifest_hash:
            raise ValueError("spatial-grid manifest hash mismatch")
        for key, value in (expected_sources or {}).items():
            if self.manifest.get(key) != value:
                raise ValueError(f"spatial-grid source mismatch for {key}")
        self.index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        if self.index.get("schema_version") != SPATIAL_GRID_CACHE_SCHEMA_VERSION:
            raise ValueError("spatial-grid index schema mismatch")
        if self.index.get("contains_gt") is not False:
            raise ValueError("spatial-grid index must explicitly exclude GT")
        if self.index.get("manifest_hash") != actual_hash:
            raise ValueError("spatial-grid index points to a different manifest")
        rows = self.index.get("groups")
        if not isinstance(rows, list) or not rows:
            raise ValueError("spatial-grid index has no groups")
        self._rows = {str(row["group_key"]): row for row in rows}
        if len(self._rows) != len(rows):
            raise ValueError("spatial-grid index has duplicate group keys")

    def read_group(self, group_key: str) -> SpatialGridFeatures:
        """Read one verified l31/final spatial-grid pair."""

        row = self._rows.get(group_key)
        if row is None:
            raise KeyError(f"spatial-grid cache has no group {group_key}")
        path = self.root / row["spatial_path"]
        if sha256_file(path) != row["spatial_hash"]:
            raise ValueError(f"spatial-grid hash mismatch for {group_key}")
        with np.load(path, allow_pickle=False) as arrays:
            features = SpatialGridFeatures(
                group_key=group_key,
                grid_l31=arrays["grid_l31"],
                grid_final=arrays["grid_final"],
                frame_size=tuple(int(value) for value in arrays["frame_size"]),
                grid_thw=tuple(int(value) for value in arrays["grid_thw"]),
                grid_l7=arrays["grid_l7"] if "grid_l7" in arrays else None,
            )
        features.validate()
        return features

    def read_all(self) -> dict[str, SpatialGridFeatures]:
        """Materialize all grids once for repeated five-outer repooling."""

        return {group_key: self.read_group(group_key) for group_key in self._rows}


def _candidate_group_payload(group: CandidateGroup) -> dict[str, Any]:
    has_frame_provenance = group.source_frame_id is not None
    payload = {
        "schema_version": (
            CANDIDATE_STORE_SCHEMA_VERSION if has_frame_provenance else 1
        ),
        "group_key": group.group_key,
        "sample_id": group.sample_id,
        "original_id": group.original_id,
        "video_id": group.video_id,
        "row_index": group.row_index,
        "dataset": group.dataset,
        "requested_timestamp": group.requested_timestamp,
        "clean_question": group.clean_question,
        "frame_path": str(group.frame_path),
        "frame_size": group.frame_size,
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "candidate_time": item.candidate_time,
                "bbox_xyxy": item.bbox_xyxy,
                "native_score": item.native_score,
                "rank": item.rank,
                "source": item.source,
                "structure": item.structure,
                "missing": item.missing,
            }
            for item in group.candidates
        ],
    }
    if has_frame_provenance:
        payload.update(
            {
                "source_frame_id": group.source_frame_id,
                "resolved_frame_timestamp": group.resolved_frame_timestamp,
                "alignment_error_seconds": group.alignment_error_seconds,
                "frame_alignment_exact": group.frame_alignment_exact,
                "timeline_digest": group.timeline_digest,
            }
        )
    return payload


def _validate_candidate_store_v2_provenance(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> None:
    required = _CANDIDATE_STORE_PROVENANCE_KEYS
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise ValueError(f"{source} candidate v2 provenance is incomplete: {missing}")
    source_frame_id = payload["source_frame_id"]
    if isinstance(source_frame_id, bool) or not isinstance(source_frame_id, int):
        raise TypeError(f"{source} source_frame_id must be a JSON integer")
    for key in ("resolved_frame_timestamp", "alignment_error_seconds"):
        _strict_json_number(payload[key], f"{source} {key}")
    if not isinstance(payload["frame_alignment_exact"], bool):
        raise TypeError(f"{source} frame_alignment_exact must be a JSON boolean")
    timeline_digest = payload["timeline_digest"]
    if not isinstance(timeline_digest, str) or not timeline_digest.strip():
        raise TypeError(f"{source} timeline_digest must be a non-empty string")


def _candidate_group_from_v2_payload(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> CandidateGroup:
    text_fields = {
        key: _strict_nonempty_json_string(payload.get(key), f"{source} {key}")
        for key in (
            "group_key",
            "sample_id",
            "original_id",
            "video_id",
            "dataset",
            "clean_question",
            "frame_path",
            "timeline_digest",
        )
    }
    row_index = _strict_json_integer(payload.get("row_index"), f"{source} row_index")
    if row_index < 0:
        raise ValueError(f"{source} row_index must be non-negative")
    frame_size = _strict_json_integer_tuple(
        payload.get("frame_size"),
        length=2,
        label=f"{source} frame_size",
    )
    if any(value <= 0 for value in frame_size):
        raise ValueError(f"{source} frame_size values must be positive")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise TypeError(f"{source} candidates must be a non-empty JSON array")
    candidates = tuple(
        _frozen_candidate_from_v2_payload(
            item,
            source=f"{source} candidates[{index}]",
        )
        for index, item in enumerate(raw_candidates)
    )
    exact_alignment = payload.get("frame_alignment_exact")
    if not isinstance(exact_alignment, bool):
        raise TypeError(f"{source} frame_alignment_exact must be a JSON boolean")
    return CandidateGroup(
        group_key=text_fields["group_key"],
        sample_id=text_fields["sample_id"],
        original_id=text_fields["original_id"],
        video_id=text_fields["video_id"],
        row_index=row_index,
        dataset=text_fields["dataset"],
        requested_timestamp=_strict_json_number(
            payload.get("requested_timestamp"),
            f"{source} requested_timestamp",
        ),
        clean_question=text_fields["clean_question"],
        frame_path=Path(text_fields["frame_path"]),
        frame_size=frame_size,
        candidates=candidates,
        source_frame_id=_strict_json_integer(
            payload.get("source_frame_id"),
            f"{source} source_frame_id",
        ),
        resolved_frame_timestamp=_strict_json_number(
            payload.get("resolved_frame_timestamp"),
            f"{source} resolved_frame_timestamp",
        ),
        alignment_error_seconds=_strict_json_number(
            payload.get("alignment_error_seconds"),
            f"{source} alignment_error_seconds",
        ),
        frame_alignment_exact=exact_alignment,
        timeline_digest=text_fields["timeline_digest"],
    )


def _frozen_candidate_from_v2_payload(
    payload: Any,
    *,
    source: str,
) -> FrozenCandidate:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{source} must be a JSON object")
    rank = _strict_json_integer(payload.get("rank"), f"{source} rank")
    if rank <= 0:
        raise ValueError(f"{source} rank must be positive")
    return FrozenCandidate(
        candidate_id=_strict_nonempty_json_string(
            payload.get("candidate_id"), f"{source} candidate_id"
        ),
        candidate_time=_strict_json_number(
            payload.get("candidate_time"), f"{source} candidate_time"
        ),
        bbox_xyxy=_strict_json_number_tuple(
            payload.get("bbox_xyxy"),
            length=4,
            label=f"{source} bbox_xyxy",
        ),
        native_score=_strict_json_number(
            payload.get("native_score"), f"{source} native_score"
        ),
        rank=rank,
        source=_strict_nonempty_json_string(
            payload.get("source"), f"{source} source"
        ),
        structure=_strict_json_number_tuple(
            payload.get("structure"),
            length=12,
            label=f"{source} structure",
        ),
        missing=_strict_json_number_tuple(
            payload.get("missing"),
            length=12,
            label=f"{source} missing",
        ),
    )


def _strict_nonempty_json_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty JSON string")
    return value


def _strict_json_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a JSON integer")
    return value


def _strict_json_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite JSON number")
    return parsed


def _strict_json_number_tuple(
    value: Any,
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise TypeError(f"{label} must be a {length}-item JSON array")
    return tuple(
        _strict_json_number(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _strict_json_integer_tuple(
    value: Any,
    *,
    length: int,
    label: str,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise TypeError(f"{label} must be a {length}-item JSON array")
    return tuple(
        _strict_json_integer(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _candidate_group_from_payload(payload: Mapping[str, Any]) -> CandidateGroup:
    candidates = tuple(
        FrozenCandidate(
            candidate_id=str(item["candidate_id"]),
            candidate_time=float(item["candidate_time"]),
            bbox_xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
            native_score=float(item["native_score"]),
            rank=int(item["rank"]),
            source=str(item["source"]),
            structure=tuple(float(value) for value in item["structure"]),
            missing=tuple(float(value) for value in item["missing"]),
        )
        for item in payload["candidates"]
    )
    exact_alignment = payload.get("frame_alignment_exact")
    if exact_alignment is not None and not isinstance(exact_alignment, bool):
        raise TypeError("candidate frame_alignment_exact must be a JSON boolean")
    return CandidateGroup(
        group_key=str(payload["group_key"]),
        sample_id=str(payload["sample_id"]),
        original_id=str(payload["original_id"]),
        video_id=str(payload.get("video_id") or ""),
        row_index=int(payload["row_index"]),
        dataset=str(payload["dataset"]),
        requested_timestamp=float(payload["requested_timestamp"]),
        clean_question=str(payload["clean_question"]),
        frame_path=Path(payload["frame_path"]),
        frame_size=tuple(int(value) for value in payload["frame_size"]),
        candidates=candidates,
        source_frame_id=(
            int(payload["source_frame_id"])
            if payload.get("source_frame_id") is not None
            else None
        ),
        resolved_frame_timestamp=(
            float(payload["resolved_frame_timestamp"])
            if payload.get("resolved_frame_timestamp") is not None
            else None
        ),
        alignment_error_seconds=(
            float(payload["alignment_error_seconds"])
            if payload.get("alignment_error_seconds") is not None
            else None
        ),
        frame_alignment_exact=exact_alignment,
        timeline_digest=(
            str(payload["timeline_digest"])
            if payload.get("timeline_digest") is not None
            else None
        ),
    )


def _reject_forbidden_feature_keys(payload: Any, *, path: str = "root") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_FEATURE_KEYS:
                raise ValueError(
                    f"forbidden GT/TAL key in feature cache at {path}.{key}"
                )
            _reject_forbidden_feature_keys(value, path=f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _reject_forbidden_feature_keys(value, path=f"{path}[{index}]")


def _safe_group_name(group_key: str) -> str:
    return hashlib.sha256(group_key.encode()).hexdigest()[:24]


def _validate_normalized_target_box(
    box: Sequence[float], *, group_key: str
) -> None:
    if len(box) != 4 or not all(np.isfinite(value) for value in box):
        raise ValueError(f"ProposalTargetStore group {group_key} has invalid GT box")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
        raise ValueError(
            f"ProposalTargetStore group {group_key} GT box is outside [0,1]"
        )
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"ProposalTargetStore group {group_key} GT box has non-positive area"
        )
