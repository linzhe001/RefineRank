"""Build strict MedVidBench submissions with an STG-only prediction overlay."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUBMISSION_FIELDS = frozenset({"id", "qa_type", "prediction"})
STG_OVERLAY_FIELDS = frozenset(
    {"source_index", "id", "qa_type", "prediction"}
)
PUBLIC_TEST_QA_TYPE_COUNTS = {
    "cvs_assessment": 648,
    "dense_captioning_gemini": 728,
    "dense_captioning_gpt": 751,
    "next_action": 670,
    "region_caption_gemini": 168,
    "region_caption_gpt": 138,
    "skill_assessment": 160,
    "stg": 780,
    "tal": 1637,
    "video_summary_gemini": 302,
    "video_summary_gpt": 263,
}
PUBLIC_TEST_ROW_COUNT = sum(PUBLIC_TEST_QA_TYPE_COUNTS.values())


@dataclass(frozen=True)
class MedVidBenchSubmissionBuildResult:
    """Summary of a strict STG-overlay submission build."""

    output_path: Path
    source_count: int
    stg_overlay_count: int
    non_stg_preserved_count: int
    qa_type_counts: Mapping[str, int]
    public_test_contract_enforced: bool

    def to_json(self) -> dict[str, object]:
        """Serialize the build result for a run manifest."""

        return {
            "output_path": str(self.output_path),
            "source_count": self.source_count,
            "stg_overlay_count": self.stg_overlay_count,
            "non_stg_preserved_count": self.non_stg_preserved_count,
            "qa_type_counts": dict(self.qa_type_counts),
            "public_test_contract_enforced": self.public_test_contract_enforced,
            "required_submission_fields": sorted(SUBMISSION_FIELDS),
            "stg_overlay_fields": sorted(STG_OVERLAY_FIELDS),
            "non_stg_value_preservation_passed": True,
            "metric_scope": "online_public_prediction_only_packaging",
        }


def build_stg_overlay_submission(
    *,
    source_json: Path,
    uai_predictions_json: Path,
    stg_predictions_json: Path,
    output_path: Path,
    require_public_test: bool = False,
) -> MedVidBenchSubmissionBuildResult:
    """Overlay source-indexed STG predictions onto full uAI predictions.

    The source index is required because MedVidBench contains repeated ``id``
    and ``(id, qa_type)`` pairs. The generated file strips that internal field
    and contains only the three fields accepted by the online leaderboard.
    """

    source_rows = _load_json_array(source_json, "source")
    uai_rows = _load_json_array(uai_predictions_json, "uAI predictions")
    stg_rows = _load_json_array(stg_predictions_json, "STG predictions")
    source_qa_counts = _validate_source_rows(source_rows)
    _validate_full_predictions(source_rows, uai_rows)
    stg_by_source_index = _validate_stg_overlay(source_rows, stg_rows)

    if require_public_test:
        _validate_public_test_contract(source_rows, source_qa_counts)

    output_rows: list[dict[str, str]] = []
    non_stg_preserved_count = 0
    for source_index, (source, uai_prediction) in enumerate(
        zip(source_rows, uai_rows, strict=True)
    ):
        if source["qa_type"] == "stg":
            overlay = stg_by_source_index[source_index]
            output_rows.append(
                {
                    "id": overlay["id"],
                    "qa_type": overlay["qa_type"],
                    "prediction": overlay["prediction"],
                }
            )
        else:
            output_rows.append(dict(uai_prediction))
            non_stg_preserved_count += 1

    _validate_full_predictions(source_rows, output_rows)
    for source_index, (source, uai_prediction, output_prediction) in enumerate(
        zip(source_rows, uai_rows, output_rows, strict=True)
    ):
        if source["qa_type"] != "stg" and output_prediction != uai_prediction:
            raise RuntimeError(
                f"non-STG prediction changed at source index {source_index}"
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return MedVidBenchSubmissionBuildResult(
        output_path=output,
        source_count=len(source_rows),
        stg_overlay_count=len(stg_rows),
        non_stg_preserved_count=non_stg_preserved_count,
        qa_type_counts=dict(sorted(source_qa_counts.items())),
        public_test_contract_enforced=require_public_test,
    )


def _load_json_array(path: Path, name: str) -> list[Mapping[str, Any]]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise TypeError(f"{name} file must be a JSON array: {path}")
    if not loaded:
        raise ValueError(f"{name} file is empty: {path}")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(loaded):
        if not isinstance(row, Mapping):
            raise TypeError(f"{name} row {index} must be a JSON object")
        rows.append(row)
    return rows


def _validate_source_rows(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    qa_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        _required_text(row.get("id"), f"source row {index} id")
        qa_type = _required_text(
            row.get("qa_type"), f"source row {index} qa_type"
        )
        qa_counts[qa_type] += 1
    if qa_counts["stg"] == 0:
        raise ValueError("source file contains no stg rows")
    return qa_counts


def _validate_full_predictions(
    source_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(prediction_rows) != len(source_rows):
        raise ValueError(
            "full prediction count does not match source count: "
            f"{len(prediction_rows)} != {len(source_rows)}"
        )
    for index, (source, prediction) in enumerate(
        zip(source_rows, prediction_rows, strict=True)
    ):
        _validate_exact_fields(prediction, SUBMISSION_FIELDS, "prediction", index)
        prediction_id = _required_text(
            prediction.get("id"), f"prediction row {index} id"
        )
        prediction_qa_type = _required_text(
            prediction.get("qa_type"), f"prediction row {index} qa_type"
        )
        _required_prediction(prediction.get("prediction"), index)
        if prediction_id != source["id"]:
            raise ValueError(
                f"prediction/source id mismatch at row {index}: "
                f"{prediction_id!r} != {source['id']!r}"
            )
        if prediction_qa_type != source["qa_type"]:
            raise ValueError(
                f"prediction/source qa_type mismatch at row {index}: "
                f"{prediction_qa_type!r} != {source['qa_type']!r}"
            )


def _validate_stg_overlay(
    source_rows: Sequence[Mapping[str, Any]],
    stg_rows: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    expected_indices = {
        index for index, row in enumerate(source_rows) if row["qa_type"] == "stg"
    }
    indexed: dict[int, Mapping[str, Any]] = {}
    for row_index, prediction in enumerate(stg_rows):
        _validate_exact_fields(
            prediction, STG_OVERLAY_FIELDS, "STG overlay", row_index
        )
        source_index = prediction.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise TypeError(
                f"STG overlay row {row_index} source_index must be an integer"
            )
        if source_index in indexed:
            raise ValueError(f"duplicate STG source_index: {source_index}")
        if source_index not in expected_indices:
            raise ValueError(
                f"STG overlay source_index {source_index} does not select an stg row"
            )
        source = source_rows[source_index]
        prediction_id = _required_text(
            prediction.get("id"), f"STG overlay row {row_index} id"
        )
        prediction_qa_type = _required_text(
            prediction.get("qa_type"), f"STG overlay row {row_index} qa_type"
        )
        _required_prediction(prediction.get("prediction"), row_index)
        if prediction_id != source["id"] or prediction_qa_type != "stg":
            raise ValueError(
                f"STG overlay row {row_index} does not match source index "
                f"{source_index}"
            )
        indexed[source_index] = prediction

    actual_indices = set(indexed)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(
            "STG overlay coverage does not match source STG rows: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return indexed


def _validate_public_test_contract(
    source_rows: Sequence[Mapping[str, Any]],
    qa_counts: Mapping[str, int],
) -> None:
    if len(source_rows) != PUBLIC_TEST_ROW_COUNT:
        raise ValueError(
            "public-test source count mismatch: "
            f"{len(source_rows)} != {PUBLIC_TEST_ROW_COUNT}"
        )
    if dict(qa_counts) != PUBLIC_TEST_QA_TYPE_COUNTS:
        raise ValueError(
            "public-test qa_type counts do not match the locked manifest: "
            f"{dict(sorted(qa_counts.items()))}"
        )


def _validate_exact_fields(
    row: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
    index: int,
) -> None:
    fields = frozenset(row)
    if fields != expected:
        raise ValueError(
            f"{name} row {index} fields must be {sorted(expected)}; "
            f"got {sorted(fields)}"
        )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_prediction(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prediction row {index} must contain non-empty text")
    return value
