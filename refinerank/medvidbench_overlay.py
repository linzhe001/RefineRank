#!/usr/bin/env python
"""Build a strict MedVidBench submission by replacing only STG predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from refinerank.medvidbench_submission import build_stg_overlay_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--uai-predictions-json", required=True, type=Path)
    parser.add_argument("--stg-predictions-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--stg-route-label",
        default="iter115_hierarchical_parent_child_ranker",
        help="Stable label recorded for the STG overlay route.",
    )
    parser.add_argument(
        "--require-public-test",
        action="store_true",
        help="Require the locked 6245-row, 780-STG public-test composition.",
    )
    args = parser.parse_args(argv)

    result = build_stg_overlay_submission(
        source_json=args.source_json,
        uai_predictions_json=args.uai_predictions_json,
        stg_predictions_json=args.stg_predictions_json,
        output_path=args.output,
        require_public_test=args.require_public_test,
    )
    payload = result.to_json()
    payload["input_sha256"] = {
        "source_json": _sha256(args.source_json),
        "uai_predictions_json": _sha256(args.uai_predictions_json),
        "stg_predictions_json": _sha256(args.stg_predictions_json),
    }
    payload["output_sha256"] = _sha256(args.output)
    if args.manifest is not None:
        _write_manifest(args.manifest, args, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    result: dict[str, object],
) -> None:
    manifest = {
        "schema_version": 1,
        "kind": "medvidbench_stg_only_overlay_submission",
        "inputs": {
            "source_json": str(args.source_json),
            "uai_predictions_json": str(args.uai_predictions_json),
            "stg_predictions_json": str(args.stg_predictions_json),
        },
        "outputs": {"submission_json": str(args.output)},
        "route_policy": {
            "stg": args.stg_route_label,
            "non_stg": "preserve_uai_prediction_value",
        },
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
