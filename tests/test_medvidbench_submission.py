from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from refinerank.medvidbench_submission import (  # noqa: E402
    build_stg_overlay_submission,
)


def test_build_stg_overlay_preserves_non_stg_and_handles_duplicate_ids(
    tmp_path: Path,
) -> None:
    source = [
        {"id": "duplicate", "qa_type": "stg"},
        {"id": "tal-id", "qa_type": "tal"},
        {"id": "duplicate", "qa_type": "stg"},
        {"id": "caption-id", "qa_type": "video_summary_gpt"},
    ]
    uai = [
        _prediction("duplicate", "stg", "uai stg 0"),
        _prediction("tal-id", "tal", "1.0-2.0 seconds."),
        _prediction("duplicate", "stg", "uai stg 2"),
        _prediction("caption-id", "video_summary_gpt", "uAI summary"),
    ]
    stg = [
        _stg_prediction(2, "duplicate", "iter115 box 2"),
        _stg_prediction(0, "duplicate", "iter115 box 0"),
    ]
    source_path, uai_path, stg_path, output_path = _write_inputs(
        tmp_path, source, uai, stg
    )

    result = build_stg_overlay_submission(
        source_json=source_path,
        uai_predictions_json=uai_path,
        stg_predictions_json=stg_path,
        output_path=output_path,
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output[0] == _prediction("duplicate", "stg", "iter115 box 0")
    assert output[1] == uai[1]
    assert output[2] == _prediction("duplicate", "stg", "iter115 box 2")
    assert output[3] == uai[3]
    assert all(set(row) == {"id", "qa_type", "prediction"} for row in output)
    assert result.stg_overlay_count == 2
    assert result.non_stg_preserved_count == 2
    assert result.to_json()["non_stg_value_preservation_passed"] is True


def test_build_stg_overlay_fails_when_stg_occurrence_is_missing(
    tmp_path: Path,
) -> None:
    source = [
        {"id": "duplicate", "qa_type": "stg"},
        {"id": "duplicate", "qa_type": "stg"},
    ]
    uai = [
        _prediction("duplicate", "stg", "uai 0"),
        _prediction("duplicate", "stg", "uai 1"),
    ]
    stg = [_stg_prediction(0, "duplicate", "iter115 0")]
    source_path, uai_path, stg_path, output_path = _write_inputs(
        tmp_path, source, uai, stg
    )

    with pytest.raises(ValueError, match="coverage"):
        build_stg_overlay_submission(
            source_json=source_path,
            uai_predictions_json=uai_path,
            stg_predictions_json=stg_path,
            output_path=output_path,
        )


def test_build_stg_overlay_fails_on_full_prediction_order_mismatch(
    tmp_path: Path,
) -> None:
    source = [
        {"id": "stg-id", "qa_type": "stg"},
        {"id": "tal-id", "qa_type": "tal"},
    ]
    uai = [
        _prediction("tal-id", "tal", "uai tal"),
        _prediction("stg-id", "stg", "uai stg"),
    ]
    stg = [_stg_prediction(0, "stg-id", "iter115")]
    source_path, uai_path, stg_path, output_path = _write_inputs(
        tmp_path, source, uai, stg
    )

    with pytest.raises(ValueError, match="mismatch at row 0"):
        build_stg_overlay_submission(
            source_json=source_path,
            uai_predictions_json=uai_path,
            stg_predictions_json=stg_path,
            output_path=output_path,
        )


def test_build_stg_overlay_fails_on_extra_submission_field(
    tmp_path: Path,
) -> None:
    source = [{"id": "stg-id", "qa_type": "stg"}]
    uai = [_prediction("stg-id", "stg", "uai") | {"metadata": {}}]
    stg = [_stg_prediction(0, "stg-id", "iter115")]
    source_path, uai_path, stg_path, output_path = _write_inputs(
        tmp_path, source, uai, stg
    )

    with pytest.raises(ValueError, match="fields must be"):
        build_stg_overlay_submission(
            source_json=source_path,
            uai_predictions_json=uai_path,
            stg_predictions_json=stg_path,
            output_path=output_path,
        )


def test_build_stg_overlay_public_contract_rejects_local_subset(
    tmp_path: Path,
) -> None:
    source = [{"id": "stg-id", "qa_type": "stg"}]
    uai = [_prediction("stg-id", "stg", "uai")]
    stg = [_stg_prediction(0, "stg-id", "iter115")]
    source_path, uai_path, stg_path, output_path = _write_inputs(
        tmp_path, source, uai, stg
    )

    with pytest.raises(ValueError, match="public-test source count"):
        build_stg_overlay_submission(
            source_json=source_path,
            uai_predictions_json=uai_path,
            stg_predictions_json=stg_path,
            output_path=output_path,
            require_public_test=True,
        )


def test_overlay_cli_records_selected_stg_route_label(tmp_path: Path) -> None:
    source = [{"id": "stg-id", "qa_type": "stg"}]
    uai = [_prediction("stg-id", "stg", "uai")]
    stg = [_stg_prediction(0, "stg-id", "repair")]
    source_path, uai_path, stg_path, output_path = _write_inputs(
        tmp_path, source, uai, stg
    )
    manifest_path = tmp_path / "manifest.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "refinerank.medvidbench_overlay",
            "--source-json",
            str(source_path),
            "--uai-predictions-json",
            str(uai_path),
            "--stg-predictions-json",
            str(stg_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--stg-route-label",
            "proposal_repair_quality_final",
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["route_policy"]["stg"] == "proposal_repair_quality_final"


def _write_inputs(
    tmp_path: Path,
    source: list[dict[str, object]],
    uai: list[dict[str, object]],
    stg: list[dict[str, object]],
) -> tuple[Path, Path, Path, Path]:
    source_path = tmp_path / "source.json"
    uai_path = tmp_path / "uai.json"
    stg_path = tmp_path / "stg.json"
    output_path = tmp_path / "submission.json"
    for path, rows in (
        (source_path, source),
        (uai_path, uai),
        (stg_path, stg),
    ):
        path.write_text(json.dumps(rows), encoding="utf-8")
    return source_path, uai_path, stg_path, output_path


def _prediction(sample_id: str, qa_type: str, text: str) -> dict[str, object]:
    return {"id": sample_id, "qa_type": qa_type, "prediction": text}


def _stg_prediction(
    source_index: int,
    sample_id: str,
    text: str,
) -> dict[str, object]:
    return {
        "source_index": source_index,
        "id": sample_id,
        "qa_type": "stg",
        "prediction": text,
    }
