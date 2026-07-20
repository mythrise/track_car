from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ibr1_experiment.artifacts import (
    DIAGNOSTICS_MANIFEST_FILENAME,
    DIAGNOSTICS_SUMMARY_FILENAME,
    EVAL_GEOMETRY_FILENAME,
    EXPECTED_EVAL_RECORDS,
    EXPECTED_GRADIENT_RECORDS,
    EXPECTED_OPTIMIZER_RECORDS,
    EXPECTED_TRAINING_RECORDS,
    GRADIENT_GEOMETRY_FILENAME,
    IBR1ArtifactContractError,
    OPTIMIZER_GEOMETRY_FILENAME,
    TRAINING_GEOMETRY_FILENAME,
    write_diagnostics_bundle,
)
from ibr1_experiment.authority import canonical_json_bytes, canonical_json_sha256


def _records(count: int, kind: str) -> list[dict[str, object]]:
    return [{"kind": kind, "position": index, "value": 0.0} for index in range(count)]


def _document(analysis_class: str, count: int | None = None) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 1,
        "analysis_class": analysis_class,
        "family_id": "IBR1",
        "internal_test": "sealed",
        "internal_test_opened": False,
    }
    if count is not None:
        document["records"] = _records(count, analysis_class)
    return document


def _inputs() -> dict[str, object]:
    summary = _document("ibr1_diagnostics_summary")
    summary["training_geometry"] = {"I2_pass": True}
    summary["eval_geometry"] = {"records": EXPECTED_EVAL_RECORDS}
    return {
        "training_records": _records(EXPECTED_TRAINING_RECORDS, "training"),
        "eval_records": _records(EXPECTED_EVAL_RECORDS, "eval"),
        "gradient_document": _document(
            "ibr1_gradient_geometry", EXPECTED_GRADIENT_RECORDS
        ),
        "optimizer_document": _document(
            "ibr1_optimizer_geometry", EXPECTED_OPTIMIZER_RECORDS
        ),
        "summary_document": summary,
        "lifecycle_bindings": {
            "checkpoint_identity": {"verified": True},
            "eval_order_guard_receipt": {"verified": True},
            "final_assembly_receipt": {"verified": True},
            "predictor_identity": {"verified": True},
            "u_pre_identity": {"verified": True},
        },
    }


def _load_canonical(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    document = json.loads(payload)
    assert payload == canonical_json_bytes(document) + b"\n"
    return document


def test_writes_complete_canonical_bundle_and_manifest_last(tmp_path: Path) -> None:
    output = tmp_path / "diagnostics"
    result = write_diagnostics_bundle(output, **_inputs())

    assert sorted(path.name for path in output.iterdir()) == sorted(
        [
            TRAINING_GEOMETRY_FILENAME,
            EVAL_GEOMETRY_FILENAME,
            GRADIENT_GEOMETRY_FILENAME,
            OPTIMIZER_GEOMETRY_FILENAME,
            DIAGNOSTICS_SUMMARY_FILENAME,
            DIAGNOSTICS_MANIFEST_FILENAME,
        ]
    )
    training_lines = (output / TRAINING_GEOMETRY_FILENAME).read_bytes().splitlines()
    eval_lines = (output / EVAL_GEOMETRY_FILENAME).read_bytes().splitlines()
    assert len(training_lines) == EXPECTED_TRAINING_RECORDS
    assert len(eval_lines) == EXPECTED_EVAL_RECORDS
    assert all(line == canonical_json_bytes(json.loads(line)) for line in training_lines)
    assert all(line == canonical_json_bytes(json.loads(line)) for line in eval_lines)

    summary = _load_canonical(output / DIAGNOSTICS_SUMMARY_FILENAME)
    summary_payload = dict(summary)
    summary_hash = summary_payload.pop("receipt_payload_sha256")
    assert summary_hash == canonical_json_sha256(summary_payload)
    assert summary["formal_training_authorized"] is False

    manifest = _load_canonical(output / DIAGNOSTICS_MANIFEST_FILENAME)
    manifest_payload = dict(manifest)
    manifest_hash = manifest_payload.pop("receipt_payload_sha256")
    assert manifest_hash == canonical_json_sha256(manifest_payload)
    assert manifest["manifest_written_after_all_bound_artifacts"] is True
    for filename, binding in manifest["artifacts"].items():
        payload = (output / filename).read_bytes()
        assert binding["sha256"] == hashlib.sha256(payload).hexdigest()
        assert binding["bytes"] == len(payload)
    assert result["manifest"]["sha256"] == hashlib.sha256(
        (output / DIAGNOSTICS_MANIFEST_FILENAME).read_bytes()
    ).hexdigest()
    assert result["formal_training_authorized"] is False


def test_rejects_existing_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "diagnostics"
    output.mkdir()

    with pytest.raises(IBR1ArtifactContractError, match="fresh and exclusive"):
        write_diagnostics_bundle(output, **_inputs())


def test_rejects_wrong_cardinality_before_creating_directory(tmp_path: Path) -> None:
    inputs = _inputs()
    inputs["training_records"] = inputs["training_records"][:-1]
    output = tmp_path / "diagnostics"

    with pytest.raises(IBR1ArtifactContractError, match="cardinality mismatch"):
        write_diagnostics_bundle(output, **inputs)

    assert not output.exists()


def test_rejects_nonfinite_or_authorizing_nested_content(tmp_path: Path) -> None:
    nonfinite = _inputs()
    nonfinite["eval_records"][0]["value"] = float("nan")
    with pytest.raises(Exception, match="finite canonical-JSON"):
        write_diagnostics_bundle(tmp_path / "nan", **nonfinite)

    authorizing = _inputs()
    authorizing["lifecycle_bindings"]["predictor_identity"] = {
        "nested": {"formal_training_authorized": True}
    }
    with pytest.raises(Exception, match="authorizing field"):
        write_diagnostics_bundle(tmp_path / "authorizing", **authorizing)

    assert not (tmp_path / "nan").exists()
    assert not (tmp_path / "authorizing").exists()


def test_requires_complete_lifecycle_identity_bindings(tmp_path: Path) -> None:
    inputs = _inputs()
    del inputs["lifecycle_bindings"]["u_pre_identity"]

    with pytest.raises(IBR1ArtifactContractError, match="binding keys"):
        write_diagnostics_bundle(tmp_path / "diagnostics", **inputs)
