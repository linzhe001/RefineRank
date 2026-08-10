"""Resolve requested times against sampled source-frame identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedSampledFrame:
    """One requested timestamp aligned to a sampled source frame."""

    requested_timestamp: float
    frame_path: Path
    source_frame_id: int
    resolved_frame_timestamp: float
    alignment_error_seconds: float
    exact: bool
    timeline_digest: str


@dataclass(frozen=True)
class SampledFrameTimeline:
    """Validated mapping from clip-relative time to sampled frame identity."""

    frame_paths: tuple[Path, ...]
    sampled_video_frames: tuple[int, ...]
    sampling_fps: float
    start_frame: int
    require_paths_exist: bool = True
    source_frame_cadence: int = field(init=False)
    source_frames_per_second: float = field(init=False)
    frame_timestamps: tuple[float, ...] = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        paths = tuple(Path(path) for path in self.frame_paths)
        frame_ids = tuple(self.sampled_video_frames)
        if not paths:
            raise ValueError("sampled frame timeline must contain at least one frame")
        if len(paths) != len(frame_ids):
            raise ValueError(
                "frame path and sampled_video_frames lengths do not match: "
                f"{len(paths)} != {len(frame_ids)}"
            )
        if not math.isfinite(self.sampling_fps) or self.sampling_fps <= 0.0:
            raise ValueError("sampling_fps must be finite and positive")
        if isinstance(self.start_frame, bool) or not isinstance(self.start_frame, int):
            raise TypeError("start_frame must be an integer")
        for index, frame_id in enumerate(frame_ids):
            if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                raise TypeError(f"sampled_video_frames[{index}] must be an integer")
        if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
            raise ValueError("sampled source frame IDs must be strictly increasing")
        if self.start_frame != frame_ids[0]:
            raise ValueError(
                "timeline start_frame must equal the first sampled source frame: "
                f"{self.start_frame} != {frame_ids[0]}"
            )

        missing: list[Path] = []
        for index, (path, frame_id) in enumerate(zip(paths, frame_ids, strict=True)):
            path_frame_id = source_frame_id_from_path(path)
            if path_frame_id != frame_id:
                raise ValueError(
                    f"frame path identity mismatch at index {index}: "
                    f"{path_frame_id} != {frame_id}"
                )
            if self.require_paths_exist and not path.is_file():
                missing.append(path)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} sampled frame path(s) are missing; first: {missing[0]}"
            )

        deltas = [right - left for left, right in zip(frame_ids, frame_ids[1:])]
        cadence = _primary_source_frame_cadence(deltas, self.sampling_fps)
        source_fps = cadence * float(self.sampling_fps)
        timestamps = tuple(
            (frame_id - self.start_frame) / source_fps for frame_id in frame_ids
        )
        digest = _timeline_digest(
            paths=paths,
            frame_ids=frame_ids,
            sampling_fps=float(self.sampling_fps),
            start_frame=self.start_frame,
            source_frame_cadence=cadence,
        )
        object.__setattr__(self, "frame_paths", paths)
        object.__setattr__(self, "sampled_video_frames", frame_ids)
        object.__setattr__(self, "sampling_fps", float(self.sampling_fps))
        object.__setattr__(self, "source_frame_cadence", cadence)
        object.__setattr__(self, "source_frames_per_second", source_fps)
        object.__setattr__(self, "frame_timestamps", timestamps)
        object.__setattr__(self, "digest", digest)

    @property
    def sampling_period_seconds(self) -> float:
        """Return the nominal interval between sampled frames."""

        return 1.0 / self.sampling_fps

    def resolve(self, timestamp: float) -> ResolvedSampledFrame:
        """Resolve an exact or half-period-nearest requested timestamp."""

        requested = float(timestamp)
        if not math.isfinite(requested):
            raise ValueError("requested timestamp must be finite")
        index = min(
            range(len(self.frame_timestamps)),
            key=lambda item: (
                abs(self.frame_timestamps[item] - requested),
                self.sampled_video_frames[item],
            ),
        )
        resolved = self.frame_timestamps[index]
        error = abs(resolved - requested)
        tolerance = self.sampling_period_seconds / 2.0
        if error > tolerance + 1.0e-9:
            raise ValueError(
                f"requested timestamp {requested:g}s is {error:.6f}s from the "
                f"nearest sampled frame, exceeding half-period {tolerance:.6f}s"
            )
        return ResolvedSampledFrame(
            requested_timestamp=requested,
            frame_path=self.frame_paths[index],
            source_frame_id=self.sampled_video_frames[index],
            resolved_frame_timestamp=resolved,
            alignment_error_seconds=error,
            exact=error <= 1.0e-9,
            timeline_digest=self.digest,
        )

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        frame_paths: Sequence[Path | str] | None = None,
        require_paths_exist: bool = True,
    ) -> SampledFrameTimeline:
        """Build a timeline from one MedVidU-style row."""

        raw_paths = frame_paths
        if raw_paths is None:
            value = row.get("video") or row.get("frame_paths")
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError("row must contain frame paths")
            raw_paths = value
        paths = tuple(Path(str(path)) for path in raw_paths)
        raw_frames = row.get("sampled_video_frames")
        if not isinstance(raw_frames, Sequence) or isinstance(
            raw_frames, (str, bytes)
        ):
            raise ValueError("row must contain sampled_video_frames")
        frames = tuple(raw_frames)
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        sampling_fps = _finite_positive_float(metadata.get("fps"), "metadata.fps")
        start_frame = _required_int(
            metadata.get("input_video_start_frame", frames[0] if frames else None),
            "metadata.input_video_start_frame",
        )
        return cls(
            frame_paths=paths,
            sampled_video_frames=frames,  # type: ignore[arg-type]
            sampling_fps=sampling_fps,
            start_frame=start_frame,
            require_paths_exist=require_paths_exist,
        )


def _primary_source_frame_cadence(
    deltas: Sequence[int], sampling_fps: float
) -> int:
    if not deltas:
        cadence = round(1.0 / sampling_fps)
        return max(1, cadence)
    if len(deltas) == 1:
        cadence = deltas[0]
        expected = max(1, round(1.0 / sampling_fps))
        if cadence != expected:
            raise ValueError(
                "source-frame cadence is underdetermined from one non-nominal "
                f"interval: observed {cadence}, expected {expected}"
            )
        return cadence
    counts = Counter(deltas)
    modal_count = max(counts.values())
    primary = min(delta for delta, count in counts.items() if count == modal_count)
    supported_divisors = [
        delta
        for delta, count in counts.items()
        if primary % delta == 0 and count * 10 >= len(deltas)
    ]
    return min(supported_divisors, default=primary)


def source_frame_id_from_path(path: Path | str) -> int:
    """Parse a source frame ID from numeric or dataset-prefixed filenames."""

    frame_path = Path(path)
    token = frame_path.stem.rsplit("_", maxsplit=1)[-1]
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            f"frame path has no numeric identity suffix: {frame_path}"
        ) from exc


def _timeline_digest(
    *,
    paths: Sequence[Path],
    frame_ids: Sequence[int],
    sampling_fps: float,
    start_frame: int,
    source_frame_cadence: int,
) -> str:
    payload = {
        "frame_names": [path.name for path in paths],
        "sampled_video_frames": list(frame_ids),
        "sampling_fps": sampling_fps,
        "source_frame_cadence": source_frame_cadence,
        "start_frame": start_frame,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return parsed


def _required_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if str(value).strip() not in {str(parsed), f"{parsed}.0"}:
        raise ValueError(f"{field_name} must be an integer")
    return parsed
