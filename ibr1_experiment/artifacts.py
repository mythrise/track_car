"""Canonical, fail-closed writers for preregistered IBR1 diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

from .authority import canonical_json_bytes, canonical_json_sha256
from .model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


TRAINING_GEOMETRY_FILENAME = "training_geometry.jsonl"
EVAL_GEOMETRY_FILENAME = "eval_geometry.jsonl"
GRADIENT_GEOMETRY_FILENAME = "gradient_geometry.json"
OPTIMIZER_GEOMETRY_FILENAME = "optimizer_geometry.json"
DIAGNOSTICS_SUMMARY_FILENAME = "diagnostics_summary.json"
DIAGNOSTICS_MANIFEST_FILENAME = "diagnostics_manifest.json"

EXPECTED_TRAINING_RECORDS = 2 * 256
EXPECTED_EVAL_RECORDS = 3 * 2 * 512 * 8 * 2
EXPECTED_GRADIENT_RECORDS = 128
EXPECTED_OPTIMIZER_RECORDS = 2 * 128

_REQUIRED_LIFECYCLE_BINDINGS = {
    "checkpoint_identity",
    "eval_order_guard_receipt",
    "final_assembly_receipt",
    "predictor_identity",
    "u_pre_identity",
}


class IBR1ArtifactContractError(RuntimeError):
    """Raised before an incomplete diagnostics bundle can claim authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1ArtifactContractError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _records(
    value: Any,
    *,
    label: str,
    expected: int,
) -> tuple[dict[str, Any], ...]:
    _require(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray)),
        f"{label} must be a sequence",
    )
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        _require(isinstance(record, Mapping), f"{label}[{index}] must be a mapping")
        normalized_record = dict(record)
        canonical_json_bytes(normalized_record)
        normalized.append(normalized_record)
    _require(
        len(normalized) == expected,
        f"{label} cardinality mismatch: expected {expected}, observed {len(normalized)}",
    )
    return tuple(normalized)


def _document(
    value: Any,
    *,
    label: str,
    analysis_class: str,
    expected_records: int | None = None,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    document = dict(value)
    _require(
        document.get("analysis_class") == analysis_class,
        f"{label} analysis_class mismatch",
    )
    _require(document.get("family_id") == IBR1_FAMILY_ID, f"{label} family drift")
    _require(
        document.get("internal_test") == "sealed"
        and document.get("internal_test_opened") is False,
        f"{label} internal-test policy drift",
    )
    if expected_records is not None:
        records = document.get("records")
        _require(
            isinstance(records, Sequence)
            and not isinstance(records, (str, bytes, bytearray))
            and len(records) == expected_records,
            f"{label} record cardinality mismatch",
        )
    canonical_json_bytes(document)
    return document


def _lifecycle_bindings(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "lifecycle_bindings must be a mapping")
    bindings = dict(value)
    _require(
        set(bindings) == _REQUIRED_LIFECYCLE_BINDINGS,
        "lifecycle binding keys differ from the frozen identity contract",
    )
    for name, binding in bindings.items():
        _require(isinstance(binding, Mapping), f"{name} must be a mapping")
    canonical_json_bytes(bindings)
    return bindings


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    _require(
        "receipt_payload_sha256" not in document,
        "document already contains receipt_payload_sha256",
    )
    document["receipt_payload_sha256"] = canonical_json_sha256(document)
    return document


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(record)) + b"\n" for record in records)


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(document)) + b"\n"


def _exclusive_write(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as exc:
        raise IBR1ArtifactContractError(
            f"refusing to overwrite diagnostics artifact: {path}"
        ) from exc
    except OSError as exc:
        raise IBR1ArtifactContractError(
            f"cannot write diagnostics artifact: {path}"
        ) from exc


def _artifact_entry(
    *,
    filename: str,
    payload: bytes,
    artifact_format: str,
    records: int | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "filename": filename,
        "sha256": _sha256(payload),
        "bytes": len(payload),
        "format": artifact_format,
    }
    if records is not None:
        entry["records"] = records
    return entry


def write_diagnostics_bundle(
    output_dir: str | Path,
    *,
    training_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    gradient_document: Mapping[str, Any],
    optimizer_document: Mapping[str, Any],
    summary_document: Mapping[str, Any],
    lifecycle_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the complete diagnostics bundle, with the manifest written last."""

    training = _records(
        training_records,
        label="training geometry",
        expected=EXPECTED_TRAINING_RECORDS,
    )
    evaluation = _records(
        eval_records,
        label="EVAL geometry",
        expected=EXPECTED_EVAL_RECORDS,
    )
    gradient = _document(
        gradient_document,
        label="gradient geometry",
        analysis_class="ibr1_gradient_geometry",
        expected_records=EXPECTED_GRADIENT_RECORDS,
    )
    optimizer = _document(
        optimizer_document,
        label="optimizer geometry",
        analysis_class="ibr1_optimizer_geometry",
        expected_records=EXPECTED_OPTIMIZER_RECORDS,
    )
    summary_input = _document(
        summary_document,
        label="diagnostics summary",
        analysis_class="ibr1_diagnostics_summary",
    )
    _require(
        isinstance(summary_input.get("training_geometry"), Mapping)
        and isinstance(summary_input.get("eval_geometry"), Mapping),
        "diagnostics summary is missing training/eval summaries",
    )
    bindings = _lifecycle_bindings(lifecycle_bindings)

    summary = _self_hashed(
        {
            **summary_input,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "formal_training_authorized": False,
        }
    )
    payloads = {
        TRAINING_GEOMETRY_FILENAME: _jsonl_bytes(training),
        EVAL_GEOMETRY_FILENAME: _jsonl_bytes(evaluation),
        GRADIENT_GEOMETRY_FILENAME: _json_bytes(gradient),
        OPTIMIZER_GEOMETRY_FILENAME: _json_bytes(optimizer),
        DIAGNOSTICS_SUMMARY_FILENAME: _json_bytes(summary),
    }

    destination = Path(output_dir).expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise IBR1ArtifactContractError(
            "diagnostics output directory must be fresh and exclusive"
        ) from exc
    except OSError as exc:
        raise IBR1ArtifactContractError(
            f"cannot create diagnostics output directory: {destination}"
        ) from exc

    artifact_entries: dict[str, dict[str, Any]] = {}
    record_counts = {
        TRAINING_GEOMETRY_FILENAME: EXPECTED_TRAINING_RECORDS,
        EVAL_GEOMETRY_FILENAME: EXPECTED_EVAL_RECORDS,
        GRADIENT_GEOMETRY_FILENAME: EXPECTED_GRADIENT_RECORDS,
        OPTIMIZER_GEOMETRY_FILENAME: EXPECTED_OPTIMIZER_RECORDS,
    }
    for filename in (
        TRAINING_GEOMETRY_FILENAME,
        EVAL_GEOMETRY_FILENAME,
        GRADIENT_GEOMETRY_FILENAME,
        OPTIMIZER_GEOMETRY_FILENAME,
        DIAGNOSTICS_SUMMARY_FILENAME,
    ):
        payload = payloads[filename]
        _exclusive_write(destination / filename, payload)
        artifact_entries[filename] = _artifact_entry(
            filename=filename,
            payload=payload,
            artifact_format="jsonl" if filename.endswith(".jsonl") else "json",
            records=record_counts.get(filename),
        )

    manifest = _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": "ibr1_diagnostics_manifest",
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "artifacts": artifact_entries,
            "lifecycle_bindings": bindings,
            "manifest_written_after_all_bound_artifacts": True,
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    manifest_payload = _json_bytes(manifest)
    _exclusive_write(destination / DIAGNOSTICS_MANIFEST_FILENAME, manifest_payload)

    return {
        "output_dir": str(destination),
        "artifacts": artifact_entries,
        "manifest": {
            "filename": DIAGNOSTICS_MANIFEST_FILENAME,
            "sha256": _sha256(manifest_payload),
            "receipt_payload_sha256": manifest["receipt_payload_sha256"],
        },
        "formal_training_authorized": False,
    }


__all__ = [
    "DIAGNOSTICS_MANIFEST_FILENAME",
    "DIAGNOSTICS_SUMMARY_FILENAME",
    "EVAL_GEOMETRY_FILENAME",
    "EXPECTED_EVAL_RECORDS",
    "EXPECTED_GRADIENT_RECORDS",
    "EXPECTED_OPTIMIZER_RECORDS",
    "EXPECTED_TRAINING_RECORDS",
    "GRADIENT_GEOMETRY_FILENAME",
    "IBR1ArtifactContractError",
    "OPTIMIZER_GEOMETRY_FILENAME",
    "TRAINING_GEOMETRY_FILENAME",
    "write_diagnostics_bundle",
]
