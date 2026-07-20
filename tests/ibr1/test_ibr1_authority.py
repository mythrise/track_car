from __future__ import annotations

import copy
import contextvars
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from f2_experiment.assembly import (
    F2AssemblyContractError,
    _resolve_frozen_token_ledger,
)
from f2_experiment.cli import SOURCE_FILES, TRANSITIVE_SOURCE_FILES
from f2_experiment.reproducibility import CUDA_REPRODUCIBILITY_SETTINGS

import ibr1_experiment.authority as authority
import ibr1_experiment.cal_pair as cal_pair


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _copy_file(source_root: Path, destination_root: Path, relative: Path) -> None:
    source = source_root / relative
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _clone_receipt_project(destination: Path) -> Path:
    for directory in ("ibr1_experiment", "tests/ibr1"):
        for path in sorted((PROJECT_ROOT / directory).glob("*.py")):
            if path.name.startswith("._"):
                continue
            _copy_file(PROJECT_ROOT, destination, path.relative_to(PROJECT_ROOT))
    for relative in (*SOURCE_FILES, *TRANSITIVE_SOURCE_FILES):
        _copy_file(PROJECT_ROOT, destination, Path(relative.as_posix()))
    for directory in (
        "experiments/windows_cuda_ibr1/preregistration",
        "experiments/windows_cuda_ibr1/primary",
    ):
        for path in sorted((PROJECT_ROOT / directory).iterdir()):
            if path.is_file():
                _copy_file(PROJECT_ROOT, destination, path.relative_to(PROJECT_ROOT))
    for relative in (
        Path("experiments/windows_cuda_f2/f2_smoke_negative_result_seal_v1.json"),
        Path("experiments/windows_cuda_f2/smoke_cuda_v1/smoke_summary.json"),
    ):
        _copy_file(PROJECT_ROOT, destination, relative)
    return destination.resolve()


@pytest.fixture(scope="session")
def support_observation() -> dict[str, Any]:
    binding = authority.build_support_binding(PROJECT_ROOT)
    return copy.deepcopy(binding["observation"])


def _support_observer(observation: dict[str, Any]):
    def observe(_root: Path) -> dict[str, Any]:
        return copy.deepcopy(observation)

    return observe


def _asset_observation(root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_class": "f2_assembly_frozen_asset_binding",
        "project_root": str(root),
        "train": {
            "relative_path": "data/collected_v1/datasets/train.jsonl",
            "rows": 13_746,
            "sha256": authority.FROZEN_TRAIN_SHA256,
        },
        "vision_cache": {
            "cache_root": str(root / "data/collected_v1/vision_cache"),
            "image_base_root": str(root),
            "cache_manifest_sha256": authority.FROZEN_CACHE_MANIFEST_SHA256,
            "cache_provenance_sha256": authority.FROZEN_CACHE_PROVENANCE_SHA256,
            "token_payload_sha256": authority.FROZEN_TOKEN_PAYLOAD_SHA256,
            "token_payload_verified": False,
            "dino_model_sha256": authority.FROZEN_DINO_SHA256,
            "siglip_model_sha256": authority.FROZEN_SIGLIP_SHA256,
            "recorded_path_root": "frozen-producer-root",
            "effective_path_root": str(root),
            "path_relocated": True,
            "metadata_only": True,
        },
        "base_hf": {
            "path": str(root / "base-hf"),
            "artifact_sha256": authority.FROZEN_BASE_HF_ARTIFACT_SHA256,
        },
        "qwen": {
            "path": str(root / "qwen"),
            "artifact_sha256": authority.FROZEN_QWEN_SHA256,
        },
        "prompt_erratum": {
            "relative_path": (
                "data/collected_v1/audits/prompt_normalization_erratum_v4.json"
            ),
            "sha256": authority.FROZEN_PROMPT_ERRATUM_SHA256,
        },
        "token_ledger_sha256": "a" * 64,
        "token_ledger_file_count": 36_946,
        "internal_test_opened": False,
    }


def _asset_observer(root: Path) -> dict[str, Any]:
    return _asset_observation(root)


def _cuda_receipt() -> dict[str, Any]:
    return {
        **dict(CUDA_REPRODUCIBILITY_SETTINGS),
        "torch_version": "test-torch",
        "cuda_runtime": "test-cuda",
    }


def _cal_context() -> dict[str, Any]:
    return {
        "seed": 0,
        "device": "cuda:0",
        "package": "SA-Hstar",
        "probe_surface": "base.proj",
        "initialization": "sealed F2 seed-0 tensor identity",
        "checkpoint_init_sha256": (
            "74f838e314dd6f3b208dbed23e0e7a92dcdde09bb7d79b0ce8c1147d9b251e54"
        ),
        "cuda_reproducibility": _cuda_receipt(),
    }


def _rehash(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result.pop("receipt_payload_sha256", None)
    result["receipt_payload_sha256"] = authority.canonical_json_sha256(result)
    return result


def _write_canonical(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(authority.canonical_json_bytes(document) + b"\n")


F2_MEDIANS = {
    "L_cot": 13.100267518450678,
    "L_future": 0.7527186011353795,
    "L_verify": 0.5114651815582008,
}


def _callback_transcript(bootstrap: dict[str, Any]) -> dict[str, Any]:
    execution_records = bootstrap["support_binding"]["observation"]["supports"][
        "CAL"
    ]["execution_binding"]["records"]
    previous = "0" * 64
    records: list[dict[str, Any]] = []
    for expected in execution_records:
        position = expected["position"]
        record = {
            "position": position,
            "row_identity": {
                "original_row_index": expected["original_row_index"],
                "sequence_id": expected["sequence_id"],
                "frame_idx": expected["frame_idx"],
                "mirrored": expected["mirrored"],
                "logged_prev_action": expected["prev_action"],
            },
            "reset_reasons": expected["reset_reasons"],
            "subordinate_f2": {
                "step0_parity": True,
                "prev_free": True,
                "track_grad_norm": 0.0,
                "aux_grad_norms": dict(F2_MEDIANS),
            },
            "ibr1": {
                "geometry_dtype": "torch.float32",
                "zero_init_persistence": True,
                "post_decode_abs_max": 0.75,
                "controlled_tensor_shape": [8, 2],
                "controlled_cells": 16,
                "realized_delta_reconstruction_error": 5e-7,
                "prev_free_observation_graph": True,
            },
        }
        record_sha = authority.canonical_json_sha256(
            {"previous_sha256": previous, "record": record}
        )
        records.append(
            {
                **record,
                "previous_sha256": previous,
                "record_sha256": record_sha,
            }
        )
        previous = record_sha
    return {
        "schema_version": 1,
        "analysis_class": authority.CAL_CALLBACK_TRANSCRIPT_CLASS,
        "rows": 512,
        "chain_algorithm": (
            "sha256(canonical_json({previous_sha256,"
            "record_without_chain_fields}))"
        ),
        "initial_sha256": "0" * 64,
        "final_sha256": previous,
        "records_sha256": authority.canonical_json_sha256(records),
        "records": records,
    }


def _raw_f2_document(
    bootstrap_path: Path,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    asset = bootstrap["asset_binding"]["observation"]
    execution_records = bootstrap["support_binding"]["observation"]["supports"][
        "CAL"
    ]["execution_binding"]["records"]
    reset_indices = sorted(
        record["original_row_index"]
        for record in execution_records
        if record["reset_reasons"]
    )
    reset_count = len(reset_indices)
    return {
        "schema_version": 1,
        "analysis_class": "f2_cal_zero_update_audit_receipt",
        "architecture_lock": "L1+D2+AP2+F2",
        "package": "SA-Hstar",
        "support": "CAL",
        "rows": 512,
        "optimizer_updates": 0,
        "assembly_receipt_sha256": hashlib.sha256(
            bootstrap_path.read_bytes()
        ).hexdigest(),
        "assembly_receipt_payload_sha256": bootstrap[
            "receipt_payload_sha256"
        ],
        "token_ledger_binding": {
            "anchor": "trust_on_first_read_at_freeze",
            "sha256": asset["token_ledger_sha256"],
            "file_count": asset["token_ledger_file_count"],
        },
        "cal_context": _cal_context(),
        "amendment_binding": {
            "amendment_id": "f2-adjudication-amendment-1",
            "sha256": (
                "2adb79ec3cd5f7d077eec23f10fac1da71eb3bd86135ea9c2837db90b065d40c"
            ),
        },
        "step0_parity": {"checked_rows": 512, "failures": 0},
        "prev_free_graph_audit": {"checked_rows": 512, "failures": 0},
        "ap2_zero_init_proof": {
            "checked_rows": 512,
            "violations": 0,
            "track_grad_norm_max": 0.0,
        },
        "static_reset_receipt": {
            "expected": reset_count,
            "observed": reset_count,
            "original_indices_sha256": authority.canonical_json_sha256(
                reset_indices
            ),
            "strafe_intersection": [],
        },
        "gradient_calibration": {
            "per_aux_grad_norm_median": dict(F2_MEDIANS),
        },
        "lambda_calibration": {
            "proposed_lambda": dict(authority.FROZEN_AUX_COEFFICIENTS)
        },
        "formal_training_authorized": False,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }


def _numeric_document(
    bootstrap_path: Path,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    bootstrap_binding = {
        "filename": bootstrap_path.name,
        "sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
        "receipt_payload_sha256": bootstrap["receipt_payload_sha256"],
        "analysis_class": authority.ASSEMBLY_RECEIPT_CLASS,
    }
    return _rehash(
        {
            "schema_version": 1,
            "analysis_class": authority.CAL_NUMERIC_EVIDENCE_CLASS,
            "family_id": authority.IBR1_FAMILY_ID,
            "architecture_lock": authority.IBR1_ARCHITECTURE_LOCK,
            "support": "CAL",
            "rows": 512,
            "optimizer_updates": 0,
            "geometry_dtype": "torch.float32",
            "bootstrap_binding": bootstrap_binding,
            "source_binding": copy.deepcopy(bootstrap["source_binding"]),
            "cal_context": _cal_context(),
            "zero_init_persistence": {
                "checked_rows": 512,
                "checked_cells": 8192,
                "per_row_shape": [8, 2],
                "failures": 0,
                "contract": "torch.equal_same_dtype_device",
            },
            "post_decode_range": {
                "checked_rows": 512,
                "checked_cells": 8192,
                "per_row_shape": [8, 2],
                "violations": 0,
                "abs_max": 0.75,
            },
            "realized_delta_reconstruction": {
                "checked_rows": 512,
                "checked_cells": 8192,
                "per_row_shape": [8, 2],
                "failures": 0,
                "error_max": 5e-7,
                "threshold": 1e-6,
            },
            "prev_free_observation_graph": {
                "checked_rows": 512,
                "failures": 0,
            },
            "auxiliary_reachability": {
                "checked_rows": 512,
                "failures": 0,
                "per_aux_grad_norm_median": dict(F2_MEDIANS),
            },
            "callback_transcript": _callback_transcript(bootstrap),
            "row_callback_count": 512,
            "lambda_proposal": dict(authority.FROZEN_AUX_COEFFICIENTS),
            "proposal_role": (
                "identity_no_drift_audit_not_coefficient_selection"
            ),
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )


def _freeze_bootstrap(
    root: Path, support_observation: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = root / "experiments/windows_cuda_ibr1/assembly_bootstrap_test.json"
    authority.freeze_assembly_receipt(
        root,
        path,
        phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        support_observer=_support_observer(support_observation),
        asset_observer=_asset_observer,
    )
    return path, authority.verify_assembly_receipt(
        root,
        path,
        required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        support_observer=_support_observer(support_observation),
        asset_observer=_asset_observer,
    )


def _install_live_bootstrap_verifier(
    monkeypatch: pytest.MonkeyPatch,
    bootstrap: dict[str, Any],
):
    original = authority.verify_assembly_receipt
    support = copy.deepcopy(bootstrap["support_binding"]["observation"])
    assets = copy.deepcopy(bootstrap["asset_binding"]["observation"])

    def verify(
        project_root: str | Path,
        receipt_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.setdefault("support_observer", _support_observer(support))
        kwargs.setdefault("asset_observer", lambda _root: copy.deepcopy(assets))
        return original(project_root, receipt_path, **kwargs)

    monkeypatch.setattr(authority, "verify_assembly_receipt", verify)
    return verify


def _write_cal_inputs(
    root: Path,
    bootstrap_path: Path,
    bootstrap: dict[str, Any],
    *,
    directory_name: str = "cal_manual",
) -> dict[str, Path]:
    directory = root / "experiments/windows_cuda_ibr1" / directory_name
    directory.mkdir(parents=True, exist_ok=False)
    raw_path = directory / "cal_audit_receipt_v1.json"
    numeric_path = directory / "ibr1_cal_numeric_evidence.json"
    _write_canonical(raw_path, _raw_f2_document(bootstrap_path, bootstrap))
    _write_canonical(numeric_path, _numeric_document(bootstrap_path, bootstrap))
    return {"raw": raw_path, "numeric": numeric_path}


def _build_manual_cal_artifacts(
    root: Path,
    bootstrap_path: Path,
    bootstrap: dict[str, Any],
    *,
    directory_name: str = "cal_manual",
) -> dict[str, Path]:
    paths = _write_cal_inputs(
        root,
        bootstrap_path,
        bootstrap,
        directory_name=directory_name,
    )
    core_path = paths["raw"].parent / "ibr1_cal_core_receipt.json"
    core = authority.build_cal_core_receipt(
        root,
        bootstrap_receipt_path=bootstrap_path,
        raw_f2_kernel_receipt_path=paths["raw"],
        numeric_evidence_receipt_path=paths["numeric"],
    )
    _write_canonical(core_path, core)
    envelope_path = paths["raw"].parent / "ibr1_cal_envelope.json"
    envelope = authority.build_cal_envelope(
        root,
        core_receipt_path=core_path,
        bootstrap_receipt_path=bootstrap_path,
    )
    _write_canonical(envelope_path, envelope)
    return {**paths, "core": core_path, "envelope": envelope_path}


def _write_fake_witness(
    *,
    bootstrap_path: Path,
    bootstrap: dict[str, Any],
    paths: dict[str, Path],
    role: str,
    pid: int,
    parent_challenge: str | None,
    parent_pid: int | None,
) -> Path:
    raw = json.loads(paths["raw"].read_text(encoding="utf-8"))
    numeric = json.loads(paths["numeric"].read_text(encoding="utf-8"))
    core = json.loads(paths["core"].read_text(encoding="utf-8"))
    envelope = json.loads(paths["envelope"].read_text(encoding="utf-8"))
    transcript = numeric["callback_transcript"]

    def receipt_binding(path: Path, document: dict[str, Any]) -> dict[str, str]:
        return {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "receipt_payload_sha256": document["receipt_payload_sha256"],
            "analysis_class": document["analysis_class"],
        }

    orchestration = {
        "analysis_class": "ibr1_cal_worker_parent_challenge",
        "parent_challenge": parent_challenge,
        "parent_pid": parent_pid,
        "child_pid": pid,
    }
    document = _rehash(
        {
            "schema_version": 1,
            "analysis_class": authority.CAL_EXECUTION_WITNESS_CLASS,
            "family_id": authority.IBR1_FAMILY_ID,
            "architecture_lock": authority.IBR1_ARCHITECTURE_LOCK,
            "role": role,
            "process_identity": {
                "pid": pid,
                "process_start_token": f"manual-process-{pid}",
                "module_import_token": f"manual-import-{pid}",
            },
            "orchestration_binding": orchestration,
            "audit_clock": {
                "started_ns": pid * 10,
                "ended_ns": pid * 10 + 1,
                "callback_count": 512,
                "first_position": 0,
                "last_position": 511,
                "ordered_positions_sha256": authority.canonical_json_sha256(
                    list(range(512))
                ),
            },
            "bootstrap_binding": {
                "filename": bootstrap_path.name,
                "sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
                "receipt_payload_sha256": bootstrap["receipt_payload_sha256"],
                "analysis_class": authority.ASSEMBLY_RECEIPT_CLASS,
            },
            "runner_binding": {
                "entrypoint": "run_ibr1_cal_audit_once",
                "source_path": "ibr1_experiment/calibration.py",
                "source_sha256": bootstrap["source_binding"][
                    "ibr1_source_sha256"
                ]["ibr1_experiment/calibration.py"],
            },
            "production_bindings": authority._expected_cal_production_bindings(
                bootstrap, actual_context=_cal_context()
            ),
            "callback_transcript_binding": {
                "container_analysis_class": authority.CAL_NUMERIC_EVIDENCE_CLASS,
                "container_receipt_payload_sha256": numeric[
                    "receipt_payload_sha256"
                ],
                "analysis_class": authority.CAL_CALLBACK_TRANSCRIPT_CLASS,
                "rows": 512,
                "records_sha256": transcript["records_sha256"],
                "final_sha256": transcript["final_sha256"],
            },
            "artifacts": {
                "raw_f2_kernel": {
                    "filename": paths["raw"].name,
                    "sha256": hashlib.sha256(paths["raw"].read_bytes()).hexdigest(),
                    "canonical_payload_sha256": authority.canonical_json_sha256(raw),
                    "analysis_class": raw["analysis_class"],
                },
                "numeric_evidence": receipt_binding(paths["numeric"], numeric),
                "core": receipt_binding(paths["core"], core),
                "envelope": receipt_binding(paths["envelope"], envelope),
            },
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    path = paths["raw"].parent / "ibr1_cal_execution_witness.json"
    _write_canonical(path, document)
    return path


def _run_fake_cal_process_pair(
    root: Path,
    bootstrap_path: Path,
) -> dict[str, Path]:
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    main_paths = _build_manual_cal_artifacts(
        root, bootstrap_path, bootstrap, directory_name="cal_main"
    )
    reproduction_paths = _build_manual_cal_artifacts(
        root, bootstrap_path, bootstrap, directory_name="cal_reproduction"
    )

    def witness(
        paths: dict[str, Path], role: str, pid: int
    ) -> Path:
        raw = json.loads(paths["raw"].read_text(encoding="utf-8"))
        numeric = json.loads(paths["numeric"].read_text(encoding="utf-8"))
        core = json.loads(paths["core"].read_text(encoding="utf-8"))
        envelope = json.loads(paths["envelope"].read_text(encoding="utf-8"))
        transcript = numeric["callback_transcript"]

        def receipt_binding(path: Path, document: dict[str, Any]) -> dict[str, str]:
            return {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "receipt_payload_sha256": document["receipt_payload_sha256"],
                "analysis_class": document["analysis_class"],
            }

        document = _rehash(
            {
                "schema_version": 1,
                "analysis_class": authority.CAL_EXECUTION_WITNESS_CLASS,
                "family_id": authority.IBR1_FAMILY_ID,
                "architecture_lock": authority.IBR1_ARCHITECTURE_LOCK,
                "role": role,
                "process_identity": {
                    "pid": pid,
                    "process_start_token": f"manual-process-{pid}",
                    "module_import_token": f"manual-import-{pid}",
                },
                "orchestration_binding": {
                    "analysis_class": "ibr1_cal_worker_parent_challenge",
                    "parent_challenge": None,
                    "parent_pid": None,
                    "child_pid": pid,
                },
                "audit_clock": {
                    "started_ns": pid * 10,
                    "ended_ns": pid * 10 + 1,
                    "callback_count": 512,
                    "first_position": 0,
                    "last_position": 511,
                    "ordered_positions_sha256": authority.canonical_json_sha256(
                        list(range(512))
                    ),
                },
                "bootstrap_binding": {
                    "filename": bootstrap_path.name,
                    "sha256": hashlib.sha256(
                        bootstrap_path.read_bytes()
                    ).hexdigest(),
                    "receipt_payload_sha256": bootstrap[
                        "receipt_payload_sha256"
                    ],
                    "analysis_class": authority.ASSEMBLY_RECEIPT_CLASS,
                },
                "runner_binding": {
                    "entrypoint": "run_ibr1_cal_audit_once",
                    "source_path": "ibr1_experiment/calibration.py",
                    "source_sha256": bootstrap["source_binding"][
                        "ibr1_source_sha256"
                    ]["ibr1_experiment/calibration.py"],
                },
                "production_bindings": authority._expected_cal_production_bindings(
                    bootstrap, actual_context=_cal_context()
                ),
                "callback_transcript_binding": {
                    "container_analysis_class": (
                        authority.CAL_NUMERIC_EVIDENCE_CLASS
                    ),
                    "container_receipt_payload_sha256": numeric[
                        "receipt_payload_sha256"
                    ],
                    "analysis_class": authority.CAL_CALLBACK_TRANSCRIPT_CLASS,
                    "rows": 512,
                    "records_sha256": transcript["records_sha256"],
                    "final_sha256": transcript["final_sha256"],
                },
                "artifacts": {
                    "raw_f2_kernel": {
                        "filename": paths["raw"].name,
                        "sha256": hashlib.sha256(
                            paths["raw"].read_bytes()
                        ).hexdigest(),
                        "canonical_payload_sha256": (
                            authority.canonical_json_sha256(raw)
                        ),
                        "analysis_class": raw["analysis_class"],
                    },
                    "numeric_evidence": receipt_binding(paths["numeric"], numeric),
                    "core": receipt_binding(paths["core"], core),
                    "envelope": receipt_binding(paths["envelope"], envelope),
                },
                "formal_training_authorized": False,
                "internal_test": "sealed",
                "internal_test_opened": False,
            }
        )
        path = paths["raw"].parent / "ibr1_cal_execution_witness.json"
        _write_canonical(path, document)
        return path

    main_witness = witness(main_paths, "main", 101)
    reproduction_witness = witness(reproduction_paths, "reproduction", 202)

    def artifact(directory: Path, filename: str) -> Path:
        path = directory / filename
        assert path.is_file()
        return path

    main = main_paths["raw"].parent
    reproduction = reproduction_paths["raw"].parent
    return {
        "main_raw": main_paths["raw"],
        "reproduction_raw": reproduction_paths["raw"],
        "main_numeric": main_paths["numeric"],
        "reproduction_numeric": reproduction_paths["numeric"],
        "main_core": main_paths["core"],
        "reproduction_core": reproduction_paths["core"],
        "main_envelope": main_paths["envelope"],
        "reproduction_envelope": reproduction_paths["envelope"],
        "main_witness": artifact(main, main_witness.name),
        "reproduction_witness": artifact(
            reproduction, reproduction_witness.name
        ),
    }


def _pair_kwargs(artifacts: dict[str, Path], bootstrap_path: Path) -> dict[str, Path]:
    return {
        "main_raw_f2_kernel_path": artifacts["main_raw"],
        "reproduction_raw_f2_kernel_path": artifacts["reproduction_raw"],
        "main_numeric_evidence_path": artifacts["main_numeric"],
        "reproduction_numeric_evidence_path": artifacts["reproduction_numeric"],
        "main_core_path": artifacts["main_core"],
        "reproduction_core_path": artifacts["reproduction_core"],
        "main_envelope_path": artifacts["main_envelope"],
        "reproduction_envelope_path": artifacts["reproduction_envelope"],
        "main_execution_witness_path": artifacts["main_witness"],
        "reproduction_execution_witness_path": artifacts[
            "reproduction_witness"
        ],
        "bootstrap_receipt_path": bootstrap_path,
    }


def _unsafe_canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unsafe_rehash(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    result.pop("receipt_payload_sha256", None)
    result["receipt_payload_sha256"] = hashlib.sha256(
        _unsafe_canonical_bytes(result)
    ).hexdigest()
    return result


def _write_unsafe_canonical(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_unsafe_canonical_bytes(document) + b"\n")


def test_canonical_json_is_finite_and_exclusive(tmp_path: Path) -> None:
    assert authority.canonical_json_bytes({"z": 1, "a": "辅助"}) == (
        b'{"a":"\xe8\xbe\x85\xe5\x8a\xa9","z":1}'
    )
    destination = tmp_path / "receipt.json"
    document = {"finite": 1.0}
    authority.exclusive_write_json(destination, document)
    assert destination.read_bytes() == authority.canonical_json_bytes(document) + b"\n"
    with pytest.raises(authority.IBR1AuthorityError, match="overwrite"):
        authority.exclusive_write_json(destination, document)
    nonfinite = tmp_path / "nonfinite.json"
    with pytest.raises(authority.IBR1AuthorityError, match="finite"):
        authority.exclusive_write_json(nonfinite, {"bad": float("nan")})
    assert not nonfinite.exists()


def test_complete_authority_chain_and_f2_update0_evidence() -> None:
    chain = authority.verify_primary_authority(PROJECT_ROOT)
    assert [item["role"] for item in chain["ordered_contract_chain"]] == [
        "primary",
        "primary_amendment_1",
        "diagnostics_schema",
        "diagnostics_schema_amendment_1",
    ]
    assert chain["effective_overrides"]["authoritative_geometry_dtype"] == (
        "torch.float32"
    )
    assert chain["effective_overrides"]["optimizer_geometry_expected_records"] == 256
    assert chain["formal_training_authorized"] is False
    negative = authority.verify_f2_negative_evidence(PROJECT_ROOT)
    assert negative["negative_seal"]["sha256"] == authority.F2_NEGATIVE_SEAL_SHA256
    assert set(negative["sealed_smoke_summary"]["update0_state_sha256"].values()) == {
        negative["sealed_smoke_summary"]["checkpoint_init_sha256"]
    }


@pytest.mark.parametrize("mutation", ["base_only", "reorder", "sha", "payload"])
def test_recorded_authority_chain_drift_fails_closed(mutation: str) -> None:
    chain = authority.verify_authority_chain(PROJECT_ROOT)
    drifted = copy.deepcopy(chain)
    if mutation == "base_only":
        drifted["ordered_contract_chain"] = drifted["ordered_contract_chain"][:1]
    elif mutation == "reorder":
        drifted["ordered_contract_chain"][1:3] = reversed(
            drifted["ordered_contract_chain"][1:3]
        )
    elif mutation == "sha":
        drifted["ordered_contract_chain"][1]["sha256"] = "0" * 64
    else:
        drifted["ordered_contract_chain"][1]["receipt_payload_sha256"] = "0" * 64
    drifted = _rehash(drifted)
    with pytest.raises(authority.IBR1AuthorityError, match="incomplete|reordered|drifted"):
        authority.verify_authority_chain(PROJECT_ROOT, drifted)


def test_missing_diagnostics_amendment_and_byte_drift_fail_closed(tmp_path: Path) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    amendment = root / Path(authority.DIAGNOSTICS_AMENDMENT1_RELATIVE.as_posix())
    amendment.unlink()
    with pytest.raises(authority.IBR1AuthorityError, match="missing"):
        authority.verify_authority_chain(root)

    root = _clone_receipt_project(tmp_path / "project_bytes")
    primary_amendment = root / Path(authority.PRIMARY_AMENDMENT1_RELATIVE.as_posix())
    primary_amendment.write_bytes(primary_amendment.read_bytes() + b" ")
    with pytest.raises(authority.IBR1AuthorityError, match="SHA drifted"):
        authority.verify_authority_chain(root)


def test_predecessor_semantic_drift_fails_even_if_new_hash_is_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    path = root / Path(authority.PRIMARY_AMENDMENT1_RELATIVE.as_posix())
    document = json.loads(path.read_text(encoding="utf-8"))
    document["predecessor"]["path"] = "wrong-primary.json"
    document = _rehash(document)
    _write_canonical(path, document)
    replacement_spec = authority.FrozenJsonSpec(
        role=authority.PRIMARY_AMENDMENT1_SPEC.role,
        relative_path=authority.PRIMARY_AMENDMENT1_SPEC.relative_path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        payload_sha256=document["receipt_payload_sha256"],
        analysis_class=authority.PRIMARY_AMENDMENT1_SPEC.analysis_class,
    )
    monkeypatch.setattr(authority, "PRIMARY_AMENDMENT1_SPEC", replacement_spec)
    with pytest.raises(authority.IBR1AuthorityError, match="predecessor"):
        authority.verify_authority_chain(root)


def test_bootstrap_is_fresh_sealed_and_freeze_null(
    tmp_path: Path, support_observation: dict[str, Any]
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    document = authority.build_assembly_receipt(
        root,
        phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        support_observer=_support_observer(support_observation),
        asset_observer=_asset_observer,
    )
    assert document["analysis_class"] == authority.ASSEMBLY_RECEIPT_CLASS
    assert document["phase"] == authority.ASSEMBLY_PHASE_BOOTSTRAP
    assert document["lambda_freeze_binding"] is None
    assert document["formal_training_authorized"] is False
    assert document["source_binding"]["analysis_class"] == authority.SOURCE_BINDING_CLASS
    assert document["test_binding"]["analysis_class"] == authority.TEST_BINDING_CLASS
    assert document["support_binding"]["analysis_class"] == authority.SUPPORT_BINDING_CLASS
    assert document["asset_binding"]["analysis_class"] == authority.ASSET_BINDING_CLASS
    asset_binding = document["asset_binding"]
    asset_observation = asset_binding["observation"]
    assert asset_binding["token_ledger_sha256"] == asset_observation[
        "token_ledger_sha256"
    ]
    assert asset_binding["token_ledger_file_count"] == asset_observation[
        "token_ledger_file_count"
    ]
    eval_support = document["support_binding"]["observation"]["supports"]["EVAL-FIX"]
    assert eval_support["rows"] == 512
    assert eval_support["ordered_original_indices_sha256"] == (
        "5123a14dc526dfcef96e73ee838e33b265dee0bff0efe66e36e806540e1922ec"
    )
    with pytest.raises(authority.IBR1AuthorityError, match="file-only"):
        authority.build_assembly_receipt(
            root,
            phase=authority.ASSEMBLY_PHASE_FINAL,
            support_observer=_support_observer(support_observation),
            asset_observer=_asset_observer,
        )
    with pytest.raises(authority.IBR1AuthorityError, match="file-only"):
        authority.freeze_assembly_receipt(
            root,
            root / "final.json",
            phase=authority.ASSEMBLY_PHASE_FINAL,
            lambda_adoption_freeze_path=root / "existing-freeze.json",
            support_observer=_support_observer(support_observation),
            asset_observer=_asset_observer,
        )
    with pytest.raises(authority.IBR1AuthorityError, match="freeze=null"):
        authority.build_assembly_receipt(
            root,
            phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
            lambda_adoption_freeze_path=root / "fake.json",
            support_observer=_support_observer(support_observation),
            asset_observer=_asset_observer,
        )


def test_bootstrap_token_ledger_anchor_is_consumable_by_f2_cal_and_fail_closed(
    tmp_path: Path,
    support_observation: dict[str, Any],
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap = authority.build_assembly_receipt(
        root,
        phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        support_observer=_support_observer(support_observation),
        asset_observer=_asset_observer,
    )
    frozen_sha = bootstrap["asset_binding"]["token_ledger_sha256"]
    frozen_count = bootstrap["asset_binding"]["token_ledger_file_count"]
    ledger = SimpleNamespace(
        ledger_sha256=frozen_sha,
        token_files=frozen_count,
    )

    assert (
        _resolve_frozen_token_ledger(root, bootstrap, token_ledger=ledger)
        is ledger
    )

    nested_only = copy.deepcopy(bootstrap)
    nested_only["asset_binding"].pop("token_ledger_sha256")
    nested_only["asset_binding"].pop("token_ledger_file_count")
    with pytest.raises(
        F2AssemblyContractError,
        match="does not freeze a token ledger anchor",
    ):
        _resolve_frozen_token_ledger(root, nested_only, token_ledger=ledger)

    mismatched = copy.deepcopy(bootstrap)
    mismatched["asset_binding"]["token_ledger_sha256"] = "0" * 64
    with pytest.raises(F2AssemblyContractError, match="TOKEN_LEDGER_MISMATCH"):
        _resolve_frozen_token_ledger(root, mismatched, token_ledger=ledger)

    count_mismatch = copy.deepcopy(bootstrap)
    count_mismatch["asset_binding"]["token_ledger_file_count"] -= 1
    with pytest.raises(F2AssemblyContractError, match="file count differs"):
        _resolve_frozen_token_ledger(
            root,
            count_mismatch,
            token_ledger=ledger,
        )


def test_static_bootstrap_rejects_divergent_token_ledger_compatibility_anchor(
    tmp_path: Path,
    support_observation: dict[str, Any],
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    path, bootstrap = _freeze_bootstrap(root, support_observation)
    bootstrap["asset_binding"]["token_ledger_sha256"] = "0" * 64
    bootstrap["asset_binding"] = _rehash(bootstrap["asset_binding"])
    bootstrap = _rehash(bootstrap)
    _write_canonical(path, bootstrap)

    with pytest.raises(
        authority.IBR1AuthorityError,
        match="compatibility anchor differs",
    ):
        authority._verify_static_assembly(
            path,
            expected_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )


def test_static_bootstrap_rejects_equal_numeric_token_ledger_count_type_drift(
    tmp_path: Path,
    support_observation: dict[str, Any],
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    path, bootstrap = _freeze_bootstrap(root, support_observation)
    bootstrap["asset_binding"]["observation"][
        "token_ledger_file_count"
    ] = 36_946.0
    bootstrap["asset_binding"] = _rehash(bootstrap["asset_binding"])
    bootstrap = _rehash(bootstrap)
    _write_canonical(path, bootstrap)

    with pytest.raises(
        authority.IBR1AuthorityError,
        match="observation token ledger cardinality drifted",
    ):
        authority._verify_static_assembly(
            path,
            expected_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )


def test_cal_execution_binding_single_field_drift_is_rejected(
    tmp_path: Path,
    support_observation: dict[str, Any],
) -> None:
    drifted = copy.deepcopy(support_observation)
    execution = drifted["supports"]["CAL"]["execution_binding"]
    record = execution["records"][17]
    record["sequence_id"] = record["sequence_id"] + "-drift"
    payload = dict(record)
    payload.pop("row_sha256")
    record["row_sha256"] = authority.canonical_json_sha256(payload)
    execution["records_sha256"] = authority.canonical_json_sha256(
        execution["records"]
    )
    with pytest.raises(authority.IBR1AuthorityError, match="aggregate SHA"):
        authority.build_support_binding(
            tmp_path,
            observer=_support_observer(drifted),
        )


def test_live_orchestrator_public_command_is_not_injectable() -> None:
    signature = inspect.signature(cal_pair.run_live_cal_pair_and_freeze)
    assert list(signature.parameters) == [
        "project_root",
        "bootstrap_receipt_path",
        "output_dir",
        "freeze_output_path",
        "final_output_path",
    ]
    with pytest.raises(TypeError, match="popen_factory"):
        cal_pair.run_live_cal_pair_and_freeze(
            ".",
            bootstrap_receipt_path="bootstrap.json",
            output_dir="pair",
            freeze_output_path="freeze.json",
            final_output_path="final.json",
            popen_factory=lambda *_args, **_kwargs: None,
        )


def test_manual_cal_pair_is_forensic_and_cannot_freeze(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    artifacts = _run_fake_cal_process_pair(root, bootstrap_path)
    pair_kwargs = _pair_kwargs(artifacts, bootstrap_path)
    pair = authority.verify_cal_pair(root, **pair_kwargs)
    assert pair["raw_f2_byte_identical"] is True
    assert pair["numeric_evidence_byte_identical"] is True
    assert pair["core_byte_identical"] is True
    assert pair["envelope_byte_identical"] is True
    assert pair["analysis_class"] == authority.CAL_PAIR_FORENSIC_CLASS
    assert pair["authority_eligible"] is False
    assert pair["distinct_processes_verified"] is False
    assert pair["recorded_process_identities_differ"] is True
    assert pair["process_identity"]["main"]["pid"] != pair["process_identity"][
        "reproduction"
    ]["pid"]
    assert pair["lambda_proposal"] == dict(authority.FROZEN_AUX_COEFFICIENTS)
    numeric = json.loads(artifacts["main_numeric"].read_text(encoding="utf-8"))
    for field in (
        "zero_init_persistence",
        "post_decode_range",
        "realized_delta_reconstruction",
    ):
        assert numeric[field]["checked_cells"] == 8192
        assert numeric[field]["per_row_shape"] == [8, 2]
    witness = json.loads(artifacts["main_witness"].read_text(encoding="utf-8"))
    assert witness["audit_clock"]["callback_count"] == 512
    assert witness["audit_clock"]["first_position"] == 0
    assert witness["audit_clock"]["last_position"] == 511

    freeze_path = root / "experiments/windows_cuda_ibr1/lambda_adoption_freeze.json"
    with pytest.raises(authority.IBR1AuthorityError, match="file-only"):
        authority.build_lambda_adoption_freeze(root, **pair_kwargs)
    with pytest.raises(authority.IBR1AuthorityError, match="file-only"):
        authority.freeze_lambda_adoption_freeze(
            root,
            freeze_path,
            **pair_kwargs,
        )
    assert not freeze_path.exists()

    assert not hasattr(authority, "build_cal_execution_receipt")
    import_attempt = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ibr1_experiment.authority import build_cal_execution_receipt",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_attempt.returncode != 0

    artifacts["reproduction_witness"].write_bytes(
        artifacts["main_witness"].read_bytes()
    )
    with pytest.raises(authority.IBR1AuthorityError, match="role drifted"):
        authority.verify_cal_pair(root, **pair_kwargs)


def test_private_session_reuses_nested_bootstrap_and_rotates_fresh_ledger(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested CAL verification must be O(1) in ledger traversals."""

    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    original_verify = authority.verify_assembly_receipt
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    artifacts = _run_fake_cal_process_pair(root, bootstrap_path)
    pair_kwargs = _pair_kwargs(artifacts, bootstrap_path)
    monkeypatch.setattr(authority, "verify_assembly_receipt", original_verify)

    support = copy.deepcopy(support_observation)
    assets = copy.deepcopy(bootstrap["asset_binding"]["observation"])
    ledger_calls: list[Path] = []

    monkeypatch.setattr(
        authority,
        "_default_support_observer",
        lambda _root: copy.deepcopy(support),
    )
    monkeypatch.setattr(
        authority,
        "verify_frozen_assets",
        lambda _root, verify_token_payload=False: copy.deepcopy(assets),
    )

    def counted_ledger(project_root: Path) -> SimpleNamespace:
        ledger_calls.append(project_root)
        return SimpleNamespace(
            ledger_sha256=assets["token_ledger_sha256"],
            token_files=assets["token_ledger_file_count"],
        )

    monkeypatch.setattr(authority, "build_train_token_ledger", counted_ledger)

    with authority._authoritative_verification_session(
        authority._VERIFICATION_SESSION_SECRET,
        root,
    ):
        authority._verify_assembly_receipt_fresh(
            root,
            bootstrap_path,
            required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )
        authority.verify_cal_pair(root, **pair_kwargs)
        for _ in range(100):
            authority.verify_assembly_receipt(
                root,
                bootstrap_path,
                required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
            )
        assert len(ledger_calls) == 1

        authority._verify_assembly_receipt_fresh(
            root,
            bootstrap_path,
            required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )
        assert len(ledger_calls) == 2


def test_final_issue_and_repeated_transition_verification_keep_constant_ledger_cost(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    assets = copy.deepcopy(bootstrap["asset_binding"]["observation"])
    monkeypatch.setattr(
        authority,
        "_default_support_observer",
        lambda _root: copy.deepcopy(support_observation),
    )
    monkeypatch.setattr(
        authority,
        "verify_frozen_assets",
        lambda _root, verify_token_payload=False: copy.deepcopy(assets),
    )
    ledger_calls = 0

    def counted_ledger(_root: Path) -> SimpleNamespace:
        nonlocal ledger_calls
        ledger_calls += 1
        return SimpleNamespace(
            ledger_sha256=assets["token_ledger_sha256"],
            token_files=assets["token_ledger_file_count"],
        )

    monkeypatch.setattr(authority, "build_train_token_ledger", counted_ledger)
    bootstrap_binding = authority._receipt_binding(root, bootstrap_path, bootstrap)
    freeze_path = root / "experiments/windows_cuda_ibr1/fake_live_freeze.json"
    freeze = _rehash(
        {
            "schema_version": 1,
            "analysis_class": authority.LAMBDA_ADOPTION_FREEZE_CLASS,
            "evidence": {"bootstrap_assembly_receipt": bootstrap_binding},
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    _write_canonical(freeze_path, freeze)
    monkeypatch.setattr(
        authority,
        "verify_lambda_adoption_freeze",
        lambda _root, _path: copy.deepcopy(freeze),
    )

    final_path = root / "experiments/windows_cuda_ibr1/fake_final.json"
    with authority._authoritative_verification_session(
        authority._VERIFICATION_SESSION_SECRET,
        root,
    ):
        authority._verify_assembly_receipt_fresh(
            root,
            bootstrap_path,
            required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )
        assert ledger_calls == 1

        final = authority._build_assembly_receipt_document(
            root,
            phase=authority.ASSEMBLY_PHASE_FINAL,
            lambda_adoption_freeze_path=freeze_path,
            _reuse_verified_bootstrap=True,
        )
        _write_canonical(final_path, final)
        assert ledger_calls == 1

        authority._verify_assembly_receipt_fresh(
            root,
            final_path,
            required_phase=authority.ASSEMBLY_PHASE_FINAL,
        )
        assert ledger_calls == 2
        for _ in range(100):
            authority.verify_assembly_receipt(
                root,
                final_path,
                required_phase=authority.ASSEMBLY_PHASE_FINAL,
            )
        assert ledger_calls == 2


def test_private_session_is_fail_closed_on_drift_observer_swap_and_context_replay(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    original_verify = authority.verify_assembly_receipt
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    monkeypatch.setattr(authority, "verify_assembly_receipt", original_verify)
    assets = copy.deepcopy(bootstrap["asset_binding"]["observation"])
    monkeypatch.setattr(
        authority,
        "_default_support_observer",
        lambda _root: copy.deepcopy(support_observation),
    )
    monkeypatch.setattr(
        authority,
        "verify_frozen_assets",
        lambda _root, verify_token_payload=False: copy.deepcopy(assets),
    )
    calls = 0
    live_ledger_sha = [assets["token_ledger_sha256"]]

    def counted_ledger(_root: Path) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            ledger_sha256=live_ledger_sha[0],
            token_files=assets["token_ledger_file_count"],
        )

    monkeypatch.setattr(authority, "build_train_token_ledger", counted_ledger)

    with pytest.raises(authority.IBR1AuthorityError, match="private capability"):
        with authority._authoritative_verification_session(object(), root):
            pass

    leaked_session: object | None = None
    with authority._authoritative_verification_session(
        authority._VERIFICATION_SESSION_SECRET,
        root,
    ):
        leaked_session = authority._CURRENT_VERIFICATION_SESSION.get()
        authority._verify_assembly_receipt_fresh(
            root,
            bootstrap_path,
            required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )
        assert calls == 1

        # A copied ContextVar cannot authorize a different thread.
        copied = contextvars.copy_context()
        errors: list[BaseException] = []

        def cross_thread() -> None:
            try:
                copied.run(
                    authority.verify_assembly_receipt,
                    root,
                    bootstrap_path,
                    required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
                )
            except BaseException as exc:  # noqa: BLE001 - assertion transport
                errors.append(exc)

        thread = threading.Thread(target=cross_thread)
        thread.start()
        thread.join()
        assert errors and isinstance(errors[0], authority.IBR1AuthorityError)

        # A custom observer is never allowed to reuse the default snapshot.
        authority.verify_assembly_receipt(
            root,
            bootstrap_path,
            required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
            support_observer=lambda _root: copy.deepcopy(support_observation),
            asset_observer=lambda _root: copy.deepcopy(assets),
        )
        assert calls == 1
        authority.verify_assembly_receipt(
            root,
            bootstrap_path,
            required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )
        assert calls == 2

        live_ledger_sha[0] = "b" * 64
        with pytest.raises(authority.IBR1AuthorityError):
            authority._verify_assembly_receipt_fresh(
                root,
                bootstrap_path,
                required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
            )
        live_ledger_sha[0] = assets["token_ledger_sha256"]
        authority._verify_assembly_receipt_fresh(
            root,
            bootstrap_path,
            required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )

        source_path = root / "ibr1_experiment" / "authority.py"
        source_bytes = source_path.read_bytes()
        source_path.write_bytes(source_bytes + b"\n")
        try:
            with pytest.raises(authority.IBR1AuthorityError):
                authority._verify_assembly_receipt_fresh(
                    root,
                    bootstrap_path,
                    required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
                )
            # The failed fresh epoch burns the old snapshot; replay is not
            # possible even though the receipt bytes themselves are unchanged.
            with pytest.raises(authority.IBR1AuthorityError):
                authority.verify_assembly_receipt(
                    root,
                    bootstrap_path,
                    required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
                )
        finally:
            source_path.write_bytes(source_bytes)

    assert leaked_session is not None
    replay_token = authority._CURRENT_VERIFICATION_SESSION.set(leaked_session)
    try:
        with pytest.raises(authority.IBR1AuthorityError, match="lifetime drifted"):
            authority.verify_assembly_receipt(
                root,
                bootstrap_path,
                required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
            )
    finally:
        authority._CURRENT_VERIFICATION_SESSION.reset(replay_token)

def test_injected_orchestrator_seam_is_fixed_command_but_non_authoritative(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def artifact_binding(path: Path, document: dict[str, Any]) -> dict[str, str]:
        return {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "receipt_payload_sha256": document["receipt_payload_sha256"],
            "analysis_class": document["analysis_class"],
        }

    class FakeProcess:
        def __init__(
            self,
            command: list[str],
            *,
            pid: int,
            stdout_text: str,
        ) -> None:
            self.args = command
            self.pid = pid
            self.returncode: int | None = None
            self._stdout = stdout_text

        def communicate(self) -> tuple[str, str]:
            self.returncode = 0
            return self._stdout, ""

    def popen_factory(command: list[str], **kwargs: Any) -> FakeProcess:
        command = list(command)
        calls.append((command, dict(kwargs)))

        def option(name: str) -> str:
            return command[command.index(name) + 1]

        role = option("--role")
        output = Path(option("--output-dir")).resolve()
        challenge = option("--parent-challenge")
        parent_pid = int(option("--parent-pid"))
        pid = 910_001 if role == "main" else 910_002
        directory_name = output.relative_to(
            root / "experiments/windows_cuda_ibr1"
        ).as_posix()
        paths = _build_manual_cal_artifacts(
            root,
            bootstrap_path,
            bootstrap,
            directory_name=directory_name,
        )
        witness_path = _write_fake_witness(
            bootstrap_path=bootstrap_path,
            bootstrap=bootstrap,
            paths=paths,
            role=role,
            pid=pid,
            parent_challenge=challenge,
            parent_pid=parent_pid,
        )
        paths["witness"] = witness_path
        raw = json.loads(paths["raw"].read_text(encoding="utf-8"))
        numeric = json.loads(paths["numeric"].read_text(encoding="utf-8"))
        core = json.loads(paths["core"].read_text(encoding="utf-8"))
        envelope = json.loads(paths["envelope"].read_text(encoding="utf-8"))
        witness = json.loads(paths["witness"].read_text(encoding="utf-8"))
        orchestration = {
            "analysis_class": "ibr1_cal_worker_parent_challenge",
            "parent_challenge": challenge,
            "parent_pid": parent_pid,
            "child_pid": pid,
        }
        calibration_result = {
            "role": role,
            "orchestration_binding": orchestration,
            "raw_f2_kernel": {
                "filename": paths["raw"].name,
                "sha256": hashlib.sha256(paths["raw"].read_bytes()).hexdigest(),
                "canonical_payload_sha256": authority.canonical_json_sha256(raw),
                "analysis_class": raw["analysis_class"],
            },
            "numeric_evidence": artifact_binding(paths["numeric"], numeric),
            "core": artifact_binding(paths["core"], core),
            "envelope": artifact_binding(paths["envelope"], envelope),
            "execution_witness": artifact_binding(paths["witness"], witness),
            "formal_training_authorized": False,
        }
        payload = {
            "schema_version": 1,
            "analysis_class": "ibr1_cal_worker_result",
            "role": role,
            "parent_challenge": challenge,
            "parent_pid": parent_pid,
            "child_pid": pid,
            "output_dir": str(output),
            "runtime": {
                "python_executable": str(cal_pair.OFFICIAL_PYTHON_EXECUTABLE),
                "torch_version": cal_pair.OFFICIAL_TORCH_VERSION,
                "cuda_runtime": cal_pair.OFFICIAL_CUDA_RUNTIME,
                "device": cal_pair.OFFICIAL_DEVICE,
            },
            "calibration_result": calibration_result,
        }
        stdout_text = authority.canonical_json_bytes(payload).decode("utf-8") + "\n"
        return FakeProcess(command, pid=pid, stdout_text=stdout_text)

    live_root = root / "experiments/windows_cuda_ibr1/live_pair"
    freeze_path = live_root / "lambda_adoption_freeze.json"
    final_path = live_root / "assembly_final.json"
    result = cal_pair._run_live_cal_pair_and_freeze(
        root,
        bootstrap_receipt_path=bootstrap_path,
        output_dir=live_root,
        freeze_output_path=freeze_path,
        final_output_path=final_path,
        popen_factory=popen_factory,
    )
    assert result["analysis_class"] == cal_pair.TEST_ONLY_PAIR_CLASS
    assert result["authority_eligible"] is False
    assert result["final_authority_capability"] is None
    assert result["freeze_written"] is False
    assert result["final_written"] is False
    assert result["worker_pids"] == {"main": 910_001, "reproduction": 910_002}
    assert len(calls) == 2
    challenges = {
        command[command.index("--parent-challenge") + 1]
        for command, _kwargs in calls
    }
    assert len(challenges) == 1
    for command, kwargs in calls:
        assert command[:3] == [
            str(cal_pair.OFFICIAL_PYTHON_EXECUTABLE),
            "-m",
            "ibr1_experiment.cal_worker",
        ]
        assert kwargs["cwd"] == str(root)
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
        assert kwargs["text"] is True
    assert not freeze_path.exists()
    assert not final_path.exists()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_sha", "token ledger SHA"),
        ("missing_count", "cardinality drifted"),
        ("count_mismatch", "cardinality drifted"),
    ],
)
def test_cal_pair_verifies_bootstrap_before_output_or_worker_burn(
    tmp_path: Path,
    support_observation: dict[str, Any],
    mutation: str,
    match: str,
) -> None:
    root = _clone_receipt_project(tmp_path / f"project_{mutation}")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    asset_binding = bootstrap["asset_binding"]
    if mutation == "missing_sha":
        asset_binding.pop("token_ledger_sha256")
    elif mutation == "missing_count":
        asset_binding.pop("token_ledger_file_count")
    else:
        asset_binding["token_ledger_file_count"] -= 1
    bootstrap["asset_binding"] = _rehash(asset_binding)
    bootstrap = _rehash(bootstrap)
    _write_canonical(bootstrap_path, bootstrap)

    output = root / "experiments/windows_cuda_ibr1/invalid_live_pair"
    freeze_path = output.parent / f"{mutation}_freeze.json"
    final_path = output.parent / f"{mutation}_final.json"
    popen_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def forbidden_popen(*args: Any, **kwargs: Any) -> None:
        popen_calls.append((args, kwargs))
        raise AssertionError("worker must not start for an invalid bootstrap")

    with pytest.raises(authority.IBR1AuthorityError, match=match):
        cal_pair._run_live_cal_pair_and_freeze(
            root,
            bootstrap_receipt_path=bootstrap_path,
            output_dir=output,
            freeze_output_path=freeze_path,
            final_output_path=final_path,
            popen_factory=forbidden_popen,
        )

    assert popen_calls == []
    assert not output.exists()
    assert not freeze_path.exists()
    assert not final_path.exists()


@pytest.mark.parametrize(
    "mutation",
    ["step0", "prev_free", "zero_init", "median", "proposal", "device", "cuda"],
)
def test_raw_f2_numeric_contract_drift_stops_core_construction(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = _clone_receipt_project(tmp_path / f"project_{mutation}")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _write_cal_inputs(root, bootstrap_path, bootstrap)
    raw = json.loads(paths["raw"].read_text(encoding="utf-8"))
    if mutation == "step0":
        raw["step0_parity"]["failures"] = 1
    elif mutation == "prev_free":
        raw["prev_free_graph_audit"]["failures"] = 1
    elif mutation == "zero_init":
        raw["ap2_zero_init_proof"]["track_grad_norm_max"] = 0.01
    elif mutation == "median":
        raw["gradient_calibration"]["per_aux_grad_norm_median"]["L_cot"] = 1.0
    elif mutation == "proposal":
        raw["lambda_calibration"]["proposed_lambda"]["L_cot"] = 0.02
    elif mutation == "device":
        raw["cal_context"]["device"] = "cpu"
    else:
        raw["cal_context"]["cuda_reproducibility"] = {
            "deterministic_algorithms": True
        }
    _write_canonical(paths["raw"], raw)
    with pytest.raises(authority.IBR1AuthorityError):
        authority.build_cal_core_receipt(
            root,
            bootstrap_receipt_path=bootstrap_path,
            raw_f2_kernel_receipt_path=paths["raw"],
            numeric_evidence_receipt_path=paths["numeric"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["range", "reconstruction", "dtype", "init", "source", "controlled_cells"],
)
def test_ibr1_numeric_evidence_drift_stops_core_construction(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = _clone_receipt_project(tmp_path / f"project_{mutation}")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _write_cal_inputs(root, bootstrap_path, bootstrap)
    numeric = json.loads(paths["numeric"].read_text(encoding="utf-8"))
    if mutation == "range":
        numeric["post_decode_range"]["violations"] = 1
    elif mutation == "reconstruction":
        numeric["realized_delta_reconstruction"]["error_max"] = 2e-6
    elif mutation == "dtype":
        numeric["geometry_dtype"] = "torch.float16"
    elif mutation == "init":
        numeric["cal_context"]["checkpoint_init_sha256"] = "0" * 64
    elif mutation == "source":
        numeric["source_binding"]["receipt_payload_sha256"] = "0" * 64
    else:
        numeric["post_decode_range"]["checked_cells"] = 512
    _write_canonical(paths["numeric"], _rehash(numeric))
    with pytest.raises(authority.IBR1AuthorityError):
        authority.build_cal_core_receipt(
            root,
            bootstrap_receipt_path=bootstrap_path,
            raw_f2_kernel_receipt_path=paths["raw"],
            numeric_evidence_receipt_path=paths["numeric"],
        )


def test_caller_cannot_supply_checks_evidence_or_proposal(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _write_cal_inputs(root, bootstrap_path, bootstrap)
    with pytest.raises(TypeError, match="unexpected keyword argument 'checks'"):
        authority.build_cal_core_receipt(
            root,
            bootstrap_receipt_path=bootstrap_path,
            raw_f2_kernel_receipt_path=paths["raw"],
            numeric_evidence_receipt_path=paths["numeric"],
            checks={name: {"passed": True} for name in authority.CAL_REQUIRED_CHECKS},
            audit_evidence={"caller": "self-asserted-pass"},
            lambda_proposal=dict(authority.FROZEN_AUX_COEFFICIENTS),
        )


@pytest.mark.parametrize("operation", ["build", "verify"])
def test_live_source_drift_during_core_operation_fails_closed(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = _clone_receipt_project(tmp_path / f"project_{operation}")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    normal_verify = _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _write_cal_inputs(root, bootstrap_path, bootstrap)
    core_path = paths["raw"].parent / "ibr1_cal_core_receipt.json"
    if operation == "verify":
        core = authority.build_cal_core_receipt(
            root,
            bootstrap_receipt_path=bootstrap_path,
            raw_f2_kernel_receipt_path=paths["raw"],
            numeric_evidence_receipt_path=paths["numeric"],
        )
        _write_canonical(core_path, core)

    calls = 0

    def drifted_verify(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        observed = copy.deepcopy(normal_verify(*args, **kwargs))
        if calls == 2:
            observed["source_binding"]["receipt_payload_sha256"] = "0" * 64
            observed = _rehash(observed)
        return observed

    monkeypatch.setattr(authority, "verify_assembly_receipt", drifted_verify)
    with pytest.raises(authority.IBR1AuthorityError, match="drifted"):
        if operation == "build":
            authority.build_cal_core_receipt(
                root,
                bootstrap_receipt_path=bootstrap_path,
                raw_f2_kernel_receipt_path=paths["raw"],
                numeric_evidence_receipt_path=paths["numeric"],
            )
        else:
            authority.verify_cal_core_receipt(
                root,
                core_path,
                expected_bootstrap_receipt_path=bootstrap_path,
                raw_f2_kernel_receipt_path=paths["raw"],
                numeric_evidence_receipt_path=paths["numeric"],
            )


@pytest.mark.parametrize("target", ["numeric", "core"])
def test_nested_formal_training_authorization_in_artifacts_fails_closed(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root = _clone_receipt_project(tmp_path / f"project_{target}")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _build_manual_cal_artifacts(
        root, bootstrap_path, bootstrap, directory_name=f"nested_{target}"
    )
    path = paths[target]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["nested_audit"] = {"formal_training_authorized": True}
    _write_unsafe_canonical(path, _unsafe_rehash(document))
    with pytest.raises(authority.IBR1AuthorityError, match="authorizing field"):
        if target == "numeric":
            authority.build_cal_core_receipt(
                root,
                bootstrap_receipt_path=bootstrap_path,
                raw_f2_kernel_receipt_path=paths["raw"],
                numeric_evidence_receipt_path=paths["numeric"],
            )
        else:
            authority.verify_cal_core_receipt(
                root,
                paths["core"],
                expected_bootstrap_receipt_path=bootstrap_path,
                raw_f2_kernel_receipt_path=paths["raw"],
                numeric_evidence_receipt_path=paths["numeric"],
            )


@pytest.mark.parametrize("target", ["witness", "freeze"])
def test_nested_internal_test_open_in_sealed_artifacts_fails_closed(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root = _clone_receipt_project(tmp_path / f"project_{target}")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _build_manual_cal_artifacts(
        root, bootstrap_path, bootstrap, directory_name=f"nested_{target}"
    )
    receipt_path = paths["raw"].parent / f"nested_{target}.json"
    analysis_class = (
        authority.CAL_EXECUTION_WITNESS_CLASS
        if target == "witness"
        else authority.LAMBDA_ADOPTION_FREEZE_CLASS
    )
    document = _unsafe_rehash(
        {
            "schema_version": 1,
            "analysis_class": analysis_class,
            "nested_audit": {"internal_test_opened": True},
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    _write_unsafe_canonical(receipt_path, document)
    with pytest.raises(authority.IBR1AuthorityError, match="opens"):
        if target == "freeze":
            authority.verify_lambda_adoption_freeze(root, receipt_path)
        else:
            authority.verify_cal_execution_witness(
                root,
                receipt_path,
                expected_role="main",
                bootstrap_receipt_path=bootstrap_path,
                raw_f2_kernel_receipt_path=paths["raw"],
                numeric_evidence_receipt_path=paths["numeric"],
                core_receipt_path=paths["core"],
                envelope_receipt_path=paths["envelope"],
            )


def test_suspicious_authority_alias_is_rejected_recursively() -> None:
    with pytest.raises(authority.IBR1AuthorityError, match="suspicious"):
        authority.canonical_json_bytes(
            {"metadata": {"formal-training-authorized": False}}
        )


def test_deep_list_mapping_internal_test_escalation_is_rejected() -> None:
    with pytest.raises(authority.IBR1AuthorityError, match="non-sealed"):
        authority.canonical_json_bytes(
            {"layers": [{"records": [{"internal_test": "opened"}]}]}
        )


def test_execution_witness_requires_exact_artifact_binding(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    artifacts = _run_fake_cal_process_pair(root, bootstrap_path)
    witness = json.loads(artifacts["main_witness"].read_text(encoding="utf-8"))
    witness["artifacts"]["numeric_evidence"]["sha256"] = "0" * 64
    _write_canonical(artifacts["main_witness"], _rehash(witness))
    with pytest.raises(authority.IBR1AuthorityError, match="artifact binding"):
        authority.verify_cal_execution_witness(
            root,
            artifacts["main_witness"],
            expected_role="main",
            bootstrap_receipt_path=bootstrap_path,
            raw_f2_kernel_receipt_path=artifacts["main_raw"],
            numeric_evidence_receipt_path=artifacts["main_numeric"],
            core_receipt_path=artifacts["main_core"],
            envelope_receipt_path=artifacts["main_envelope"],
        )


def test_legacy_self_hashed_witness_without_transcript_binding_is_rejected(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    artifacts = _run_fake_cal_process_pair(root, bootstrap_path)
    witness = json.loads(artifacts["main_witness"].read_text(encoding="utf-8"))
    witness.pop("callback_transcript_binding")
    legacy_path = artifacts["main_witness"].with_name("legacy_witness.json")
    _write_canonical(legacy_path, _rehash(witness))
    with pytest.raises(authority.IBR1AuthorityError, match="callback-transcript"):
        authority.verify_cal_execution_witness(
            root,
            legacy_path,
            expected_role="main",
            bootstrap_receipt_path=bootstrap_path,
            raw_f2_kernel_receipt_path=artifacts["main_raw"],
            numeric_evidence_receipt_path=artifacts["main_numeric"],
            core_receipt_path=artifacts["main_core"],
            envelope_receipt_path=artifacts["main_envelope"],
        )


def test_witness_production_callable_or_source_binding_drift_is_rejected(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    artifacts = _run_fake_cal_process_pair(root, bootstrap_path)
    witness = json.loads(artifacts["main_witness"].read_text(encoding="utf-8"))
    witness["production_bindings"]["subordinate_kernel"]["callable"] = (
        "tests.fake_runner"
    )
    witness["production_bindings"]["row_auditor_factory"]["source_sha256"] = (
        "0" * 64
    )
    _write_canonical(artifacts["main_witness"], _rehash(witness))
    with pytest.raises(authority.IBR1AuthorityError, match="production"):
        authority.verify_cal_execution_witness(
            root,
            artifacts["main_witness"],
            expected_role="main",
            bootstrap_receipt_path=bootstrap_path,
            raw_f2_kernel_receipt_path=artifacts["main_raw"],
            numeric_evidence_receipt_path=artifacts["main_numeric"],
            core_receipt_path=artifacts["main_core"],
            envelope_receipt_path=artifacts["main_envelope"],
        )


def test_callback_transcript_single_record_drift_is_rejected_after_rehash(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _write_cal_inputs(root, bootstrap_path, bootstrap)
    numeric = json.loads(paths["numeric"].read_text(encoding="utf-8"))
    transcript = numeric["callback_transcript"]
    transcript["records"][17]["ibr1"]["controlled_cells"] = 15
    previous = "0" * 64
    rebuilt: list[dict[str, Any]] = []
    for raw_entry in transcript["records"]:
        record = dict(raw_entry)
        record.pop("previous_sha256")
        record.pop("record_sha256")
        record_sha = authority.canonical_json_sha256(
            {"previous_sha256": previous, "record": record}
        )
        rebuilt.append(
            {
                **record,
                "previous_sha256": previous,
                "record_sha256": record_sha,
            }
        )
        previous = record_sha
    transcript["records"] = rebuilt
    transcript["final_sha256"] = previous
    transcript["records_sha256"] = authority.canonical_json_sha256(rebuilt)
    _write_canonical(paths["numeric"], _rehash(numeric))
    with pytest.raises(authority.IBR1AuthorityError, match="identity/shape"):
        authority.verify_cal_numeric_evidence(
            root,
            paths["numeric"],
            bootstrap_receipt_path=bootstrap_path,
        )


def test_callback_transcript_exact_support_field_drift_is_rejected_after_rehash(
    tmp_path: Path,
    support_observation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    bootstrap_path, bootstrap = _freeze_bootstrap(root, support_observation)
    _install_live_bootstrap_verifier(monkeypatch, bootstrap)
    paths = _write_cal_inputs(root, bootstrap_path, bootstrap)
    numeric = json.loads(paths["numeric"].read_text(encoding="utf-8"))
    transcript = numeric["callback_transcript"]
    transcript["records"][17]["row_identity"]["sequence_id"] += "-drift"
    previous = "0" * 64
    rebuilt: list[dict[str, Any]] = []
    for raw_entry in transcript["records"]:
        record = dict(raw_entry)
        record.pop("previous_sha256")
        record.pop("record_sha256")
        record_sha = authority.canonical_json_sha256(
            {"previous_sha256": previous, "record": record}
        )
        rebuilt.append(
            {
                **record,
                "previous_sha256": previous,
                "record_sha256": record_sha,
            }
        )
        previous = record_sha
    transcript["records"] = rebuilt
    transcript["final_sha256"] = previous
    transcript["records_sha256"] = authority.canonical_json_sha256(rebuilt)
    _write_canonical(paths["numeric"], _rehash(numeric))
    with pytest.raises(authority.IBR1AuthorityError, match="row identity"):
        authority.verify_cal_numeric_evidence(
            root,
            paths["numeric"],
            bootstrap_receipt_path=bootstrap_path,
        )


def test_full_payload_reread_and_internal_open_fail_closed(
    tmp_path: Path, support_observation: dict[str, Any]
) -> None:
    root = _clone_receipt_project(tmp_path / "project")
    observed = _asset_observation(root)
    observed["vision_cache"]["token_payload_verified"] = True
    with pytest.raises(authority.IBR1AuthorityError, match="must not traverse"):
        authority.build_asset_binding(root, observer=lambda _root: observed)
    observed = _asset_observation(root)
    observed["internal_test_opened"] = True
    with pytest.raises(authority.IBR1AuthorityError, match="opened"):
        authority.build_assembly_receipt(
            root,
            phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
            support_observer=_support_observer(support_observation),
            asset_observer=lambda _root: observed,
        )


@pytest.mark.parametrize(
    "ledger_file_count",
    [36_946.0, True, "36946"],
    ids=["float", "bool", "string"],
)
def test_asset_binding_rejects_non_exact_token_ledger_file_count_type(
    tmp_path: Path,
    ledger_file_count: Any,
) -> None:
    root = tmp_path / "project"
    observed = _asset_observation(root)
    observed["token_ledger_file_count"] = ledger_file_count

    with pytest.raises(authority.IBR1AuthorityError, match="cardinality drifted"):
        authority.build_asset_binding(root, observer=lambda _root: observed)
