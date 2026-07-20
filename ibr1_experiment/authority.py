"""Fail-closed authority foundation for the preregistered IBR1 smoke.

This module owns IBR1 identities only.  It may reuse the frozen F2 support
and asset *verification kernels*, but it never promotes an F2 top-level
receipt class into IBR1 authority.  In particular, CAL output is normalized
into an IBR1 core receipt and then bound by an IBR1 envelope before a fresh
IBR1 lambda-adoption freeze can be issued.

The module is deliberately receipt-only: it does not construct a model, run
CAL, evaluate a row, create an optimizer, or read the sealed internal-test
subtree.  The only data traversal performed by the default observers is the
frozen train support and its train-split token ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import median
from typing import Any, Literal

from f2_experiment.assembly_data import (
    FROZEN_BASE_HF_ARTIFACT_SHA256,
    FROZEN_CACHE_MANIFEST_SHA256,
    FROZEN_CACHE_PROVENANCE_SHA256,
    FROZEN_DINO_SHA256,
    FROZEN_PROMPT_ERRATUM_SHA256,
    FROZEN_QWEN_SHA256,
    FROZEN_SIGLIP_SHA256,
    FROZEN_TOKEN_PAYLOAD_SHA256,
    build_train_token_ledger,
    ordered_support_rows,
    support_reset_plan,
    verify_frozen_assets,
)
from f2_experiment.cli import (
    source_bindings as f2_source_bindings,
    transitive_source_bindings,
)
from f2_experiment.reproducibility import (
    F2CudaReproducibilityError,
    validate_cuda_reproducibility_receipt,
)
from f2_experiment.support import (
    FROZEN_TRAIN_RELATIVE,
    FROZEN_TRAIN_ROWS,
    FROZEN_TRAIN_SHA256,
    SUPPORT_EXPECTATIONS,
    build_frozen_support,
    parse_train_jsonl,
)

from .model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID
from .runtime_contract import (
    OFFICIAL_CUDA_RUNTIME,
    OFFICIAL_DEVICE,
    OFFICIAL_PYTHON_EXECUTABLE,
    OFFICIAL_TORCH_VERSION,
)


INTERNAL_TEST_POLICY = "sealed"

ASSEMBLY_SCHEMA_VERSION = 1
ASSEMBLY_RECEIPT_VERSION = 1
ASSEMBLY_RECEIPT_CLASS = "ibr1_assembly_source_receipt"
ASSEMBLY_PHASE_BOOTSTRAP = "bootstrap"
ASSEMBLY_PHASE_FINAL = "final"
AssemblyPhase = Literal["bootstrap", "final"]

SOURCE_BINDING_CLASS = "ibr1_source_binding"
TEST_BINDING_CLASS = "ibr1_test_binding"
SUPPORT_BINDING_CLASS = "ibr1_frozen_support_binding"
ASSET_BINDING_CLASS = "ibr1_frozen_asset_binding"
AUTHORITY_CHAIN_CLASS = "ibr1_effective_preregistration_chain"
F2_NEGATIVE_EVIDENCE_CLASS = "ibr1_f2_negative_evidence_binding"

CAL_CORE_RECEIPT_CLASS = "ibr1_cal_core_receipt"
CAL_ENVELOPE_RECEIPT_CLASS = "ibr1_cal_zero_update_audit_envelope"
CAL_NUMERIC_EVIDENCE_CLASS = "ibr1_cal_numeric_evidence_receipt"
CAL_EXECUTION_WITNESS_CLASS = "ibr1_cal_execution_witness"
CAL_CALLBACK_TRANSCRIPT_CLASS = "ibr1_cal_callback_transcript"
CAL_EXECUTION_BINDING_CLASS = "ibr1_cal_exact_execution_binding"
CAL_PAIR_FORENSIC_CLASS = "ibr1_cal_pair_forensic_evidence"
CAL_LIVE_PROCESS_ATTESTATION_CLASS = "ibr1_cal_live_process_attestation"
# Compatibility alias for callers that imported the first 2B1 draft name.
CAL_EXECUTION_RECEIPT_CLASS = CAL_EXECUTION_WITNESS_CLASS
LAMBDA_ADOPTION_FREEZE_CLASS = "ibr1_lambda_adoption_freeze"

CAL_SUPPORT = "CAL"
CAL_ROWS = 512
CAL_CONTROLLED_SHAPE = [8, 2]
CAL_CONTROLLED_CELLS = CAL_ROWS * CAL_CONTROLLED_SHAPE[0] * CAL_CONTROLLED_SHAPE[1]
CAL_EXECUTION_BINDING_RECORDS_SHA256 = (
    "8e9f2441b90e67f2526f47a91271b2a2999acf91142dc456a9b99c7fee636631"
)
CAL_SEED = 0
CAL_DEVICE = "cuda:0"
CAL_PACKAGE = "SA-Hstar"
CAL_PROBE_SURFACE = "base.proj"
CAL_REQUIRED_CHECKS = (
    "f2_step0_parity",
    "f2_prev_free_graph",
    "f2_ap2_zero_init",
    "f2_auxiliary_medians",
    "ibr1_zero_init_persistence",
    "ibr1_post_decode_range",
    "ibr1_realized_delta_reconstruction",
    "ibr1_prev_free_observation_graph",
    "ibr1_auxiliary_reachability",
    "authority_bindings",
)

FROZEN_AUX_COEFFICIENTS: Mapping[str, float] = {
    "L_cot": 0.0195,
    "L_future": 0.34,
    "L_verify": 0.5,
}

PRIMARY_RELATIVE = PurePosixPath(
    "experiments/windows_cuda_ibr1/preregistration/"
    "ibr1_primary_preregistration_v1.json"
)
PRIMARY_SHA256 = "b08d1d001b2178d13abd30dc94f1a2d24c574965ad200b2dff1f87320c8b2007"
PRIMARY_PAYLOAD_SHA256 = (
    "5de7cdf41f21f4db03d43d12ea6e0f5e8ea0c5d7e730bdac9df93cf5d7d3b9b8"
)

PRIMARY_AMENDMENT1_RELATIVE = PurePosixPath(
    "experiments/windows_cuda_ibr1/preregistration/"
    "ibr1_preregistration_amendment1_v1.json"
)
PRIMARY_AMENDMENT1_SHA256 = (
    "e054023f3f86a1e7718a2f76717f31ef9b5b58a4633293725ce989c3e000942b"
)
PRIMARY_AMENDMENT1_PAYLOAD_SHA256 = (
    "d0b2a581ec86a40eff0f0c58ed55a0783ef47a577f66ca82e0a8630db64d3cbc"
)

DIAGNOSTICS_SCHEMA_RELATIVE = PurePosixPath(
    "experiments/windows_cuda_ibr1/preregistration/"
    "ibr1_diagnostics_schema_v1.json"
)
DIAGNOSTICS_SCHEMA_SHA256 = (
    "bc4835044a974e7e0215407d2efed4aa692003545d1a920060dd4b2902afcacb"
)
DIAGNOSTICS_SCHEMA_PAYLOAD_SHA256 = (
    "29907fbb76fa08a96f3c1c835077b3961d414c693384a374eeec3dcb16161a4b"
)

DIAGNOSTICS_AMENDMENT1_RELATIVE = PurePosixPath(
    "experiments/windows_cuda_ibr1/preregistration/"
    "ibr1_diagnostics_schema_amendment1_v1.json"
)
DIAGNOSTICS_AMENDMENT1_SHA256 = (
    "444a5668733eb27cf777b24a28bff9fd744034129af0857284007b78734dfaac"
)
DIAGNOSTICS_AMENDMENT1_PAYLOAD_SHA256 = (
    "546a12c9b8db134da8a4305c9778d48f205037d1ed621666f1d95dc32ca44707"
)

F2_NEGATIVE_SEAL_RELATIVE = PurePosixPath(
    "experiments/windows_cuda_f2/f2_smoke_negative_result_seal_v1.json"
)
F2_NEGATIVE_SEAL_SHA256 = (
    "b85585c8232f65c75d5958abb7d51d7624db4031c9adec86ee570d0a5b7378e7"
)

FROZEN_TRAIN_TOKEN_FILES = 36_946

class IBR1AuthorityError(RuntimeError):
    """Raised whenever an IBR1 authority contract must fail closed."""


@dataclass(frozen=True)
class FrozenJsonSpec:
    """One immutable self-hashed preregistration JSON identity."""

    role: str
    relative_path: PurePosixPath
    sha256: str
    payload_sha256: str
    analysis_class: str


PRIMARY_SPEC = FrozenJsonSpec(
    role="primary",
    relative_path=PRIMARY_RELATIVE,
    sha256=PRIMARY_SHA256,
    payload_sha256=PRIMARY_PAYLOAD_SHA256,
    analysis_class="ibr1_primary_preregistration_index",
)
PRIMARY_AMENDMENT1_SPEC = FrozenJsonSpec(
    role="primary_amendment_1",
    relative_path=PRIMARY_AMENDMENT1_RELATIVE,
    sha256=PRIMARY_AMENDMENT1_SHA256,
    payload_sha256=PRIMARY_AMENDMENT1_PAYLOAD_SHA256,
    analysis_class="ibr1_primary_preregistration_amendment",
)
DIAGNOSTICS_SCHEMA_SPEC = FrozenJsonSpec(
    role="diagnostics_schema",
    relative_path=DIAGNOSTICS_SCHEMA_RELATIVE,
    sha256=DIAGNOSTICS_SCHEMA_SHA256,
    payload_sha256=DIAGNOSTICS_SCHEMA_PAYLOAD_SHA256,
    analysis_class="ibr1_diagnostics_schema_preregistration",
)
DIAGNOSTICS_AMENDMENT1_SPEC = FrozenJsonSpec(
    role="diagnostics_schema_amendment_1",
    relative_path=DIAGNOSTICS_AMENDMENT1_RELATIVE,
    sha256=DIAGNOSTICS_AMENDMENT1_SHA256,
    payload_sha256=DIAGNOSTICS_AMENDMENT1_PAYLOAD_SHA256,
    analysis_class="ibr1_diagnostics_schema_amendment",
)

COMPONENT_SPECS: Mapping[str, FrozenJsonSpec] = {
    "ablation_and_claim": FrozenJsonSpec(
        role="ablation_and_claim",
        relative_path=PurePosixPath(
            "experiments/windows_cuda_ibr1/preregistration/"
            "ibr1_ablation_and_claim_v1.json"
        ),
        sha256="03fe3cfb71d33d8787335648e383fef090ea2431d026ed0c5e3f850fdc3bb5e9",
        payload_sha256=(
            "78ba176f80c53ef79014d517bedfb20cfdb517621d527a52a3603c4f0f488a7f"
        ),
        analysis_class="ibr1_ablation_and_claim_preregistration",
    ),
    "architecture": FrozenJsonSpec(
        role="architecture",
        relative_path=PurePosixPath(
            "experiments/windows_cuda_ibr1/preregistration/"
            "ibr1_architecture_preregistration_v1.json"
        ),
        sha256="cf138d189738f51953d7ba217355e2ea58753aa20766decdddc44840b805b806",
        payload_sha256=(
            "9055a6f0b14335eb7b22e7ae931f9e8b5e48ccc00d95518b0b496af74310e7cb"
        ),
        analysis_class="ibr1_architecture_preregistration",
    ),
    "cal_protocol": FrozenJsonSpec(
        role="cal_protocol",
        relative_path=PurePosixPath(
            "experiments/windows_cuda_ibr1/preregistration/"
            "ibr1_cal_protocol_v1.json"
        ),
        sha256="98aeb565ada02fb2d2308872e54f047f052d09aad7bb328d2f7fd7433d7dca40",
        payload_sha256=(
            "ea7f2a0e8d9f13299b23040cc2bfe859442919146eee342e3d1b480e1a06ecbf"
        ),
        analysis_class="ibr1_cal_protocol_preregistration",
    ),
    "gate_registry": FrozenJsonSpec(
        role="gate_registry",
        relative_path=PurePosixPath(
            "experiments/windows_cuda_ibr1/preregistration/"
            "ibr1_gate_registry_v1.json"
        ),
        sha256="235dbc0c8ec09f15630f7402c54ce42d9184b65f6e3a99eb09da0ddd764cb509",
        payload_sha256=(
            "5c23ca85afdca8849efe6982e647a05ce0c7fce145e21512d69302baeba93bd4"
        ),
        analysis_class="ibr1_gate_registry",
    ),
    "gradient_policy": FrozenJsonSpec(
        role="gradient_policy",
        relative_path=PurePosixPath(
            "experiments/windows_cuda_ibr1/preregistration/"
            "ibr1_gradient_policy_v1.json"
        ),
        sha256="7c08c352c06bbc9ed82cd93799a500c6a5f3a56ccb558ff2116622895ca451e5",
        payload_sha256=(
            "2bd16a6f14e1c07199906cacc602196c3fc30dba7f089fb6f2d77bcc47cdae96"
        ),
        analysis_class="ibr1_gradient_policy_preregistration",
    ),
    "negative_result_adoption": FrozenJsonSpec(
        role="negative_result_adoption",
        relative_path=PurePosixPath(
            "experiments/windows_cuda_ibr1/preregistration/"
            "ibr1_negative_result_adoption_v1.json"
        ),
        sha256="5fb4709ef07df805c193b32eee17c9447885d2e2031d477acbc961124c5f0ca5",
        payload_sha256=(
            "7ed9620b4452dbc00d84a5197ada70dd844c567405d4b3cadd32a2c251ebf91b"
        ),
        analysis_class="ibr1_negative_result_adoption",
    ),
}

PRIMARY_PROVENANCE: Mapping[str, tuple[PurePosixPath, str]] = {
    "command": (
        PurePosixPath(
            "experiments/windows_cuda_ibr1/primary/"
            "20260720_ibr1_primary_command.json"
        ),
        "456e0df0b79f85fd485c68cfe6a65d10851fe3354f581c7f1be14937e245a8bf",
    ),
    "prompt": (
        PurePosixPath(
            "experiments/windows_cuda_ibr1/primary/"
            "20260720_ibr1_primary_prompt.md"
        ),
        "47da19cff8be7fc7a5bff3ef494cec8f0a3c8b2a710362bc5a2aba590c033735",
    ),
    "raw": (
        PurePosixPath(
            "experiments/windows_cuda_ibr1/primary/"
            "20260720_ibr1_primary_raw.md"
        ),
        "60eab140b6b1b7b95c52b830a3d985e54440c490f2da802fa203e1084f131457",
    ),
    "verdict": (
        PurePosixPath(
            "experiments/windows_cuda_ibr1/primary/"
            "20260720_ibr1_primary_verdict.md"
        ),
        "4aa55aa49ec619d0d4e603872394fdc9d39897a0f086c698dc281a0ca59d98a4",
    ),
}

PRIMARY_AMENDMENT1_VERDICT_RELATIVE = PurePosixPath(
    "experiments/windows_cuda_ibr1/primary/"
    "20260720_ibr1_primary_amendment1_verdict.md"
)
PRIMARY_AMENDMENT1_VERDICT_SHA256 = (
    "d67dbb5f60f3f886da41a32dc4d4df5de9c9b2ebdd660a16f8031c16b9e933f7"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1AuthorityError(message)


def _reject_nested_authority_escalation(
    value: Any, label: str, path: str = "$"
) -> None:
    """Reject nested authorization/internal-test escalation at any depth."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = "".join(character for character in key.lower() if character.isalnum())
            item_path = f"{path}.{key}"
            if normalized == "formaltrainingauthorized":
                _require(
                    key == "formal_training_authorized" and item is False,
                    f"{label} contains suspicious or authorizing field at {item_path}",
                )
            elif normalized == "internaltestopened":
                _require(
                    key == "internal_test_opened" and item is False,
                    f"{label} opens or aliases the internal test at {item_path}",
                )
            elif normalized == "internaltest":
                _require(
                    key == "internal_test" and item == INTERNAL_TEST_POLICY,
                    f"{label} carries a non-sealed internal-test field at {item_path}",
                )
            _reject_nested_authority_escalation(item, label, item_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_nested_authority_escalation(
                item, label, f"{path}[{index}]"
            )


def _valid_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase SHA-256 hex",
    )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return sorted compact UTF-8 JSON and reject every nonfinite value."""

    _reject_nested_authority_escalation(value, "canonical authority document")
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise IBR1AuthorityError(
            "value is not finite canonical-JSON serializable"
        ) from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _with_payload_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    _require(
        "receipt_payload_sha256" not in document,
        "receipt payload already contains receipt_payload_sha256",
    )
    document["receipt_payload_sha256"] = canonical_json_sha256(document)
    return document


def _verify_payload_self_hash(value: Mapping[str, Any], label: str) -> str:
    _reject_nested_authority_escalation(value, label)
    payload = dict(value)
    stored = payload.pop("receipt_payload_sha256", None)
    _valid_sha256(stored, f"{label} receipt_payload_sha256")
    _require(
        stored == canonical_json_sha256(payload),
        f"{label} payload self-hash differs from its content",
    )
    return str(stored)


def _sha256_file(path: Path, label: str) -> str:
    _require(path.is_file(), f"{label} is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IBR1AuthorityError(f"{label} is unreadable: {path}") from exc
    _require(isinstance(document, dict), f"{label} must be a JSON object")
    return document


def _load_canonical_receipt(path: Path, label: str) -> dict[str, Any]:
    document = _load_json(path, label)
    expected = canonical_json_bytes(document) + b"\n"
    _require(
        path.read_bytes() == expected,
        f"{label} is not sorted compact finite canonical JSON plus LF",
    )
    _verify_payload_self_hash(document, label)
    return document


def exclusive_write_json(path: str | Path, value: Any) -> str:
    """Write canonical JSON plus LF with O_EXCL; never overwrite evidence."""

    destination = Path(path).expanduser().resolve()
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o644)
    except FileExistsError as exc:
        raise IBR1AuthorityError(
            f"refusing to overwrite frozen authority evidence: {destination}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _root_relative(root: Path, path: Path, label: str) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise IBR1AuthorityError(
            f"{label} must stay inside the project root: {resolved}"
        ) from exc
    return relative.as_posix()


def _resolve_bound_path(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} has no path")
    portable = PurePosixPath(value)
    _require(
        "\\" not in value
        and not portable.is_absolute()
        and not PureWindowsPath(value).is_absolute()
        and portable.as_posix() == value
        and all(part not in (".", "..") for part in portable.parts),
        f"{label} must be a clean project-relative POSIX path",
    )
    path = (root / Path(*portable.parts)).resolve()
    _root_relative(root, path, label)
    return path


def _sealed(value: Mapping[str, Any], label: str) -> None:
    _require(
        value.get("internal_test") == INTERNAL_TEST_POLICY
        and value.get("internal_test_opened") is False,
        f"{label} internal-test seal is broken",
    )


def _formal_forbidden(value: Mapping[str, Any], label: str) -> None:
    _require(
        value.get("formal_training_authorized") is False,
        f"{label} must keep formal_training_authorized=false",
    )


def _binding_for_spec(
    root: Path, spec: FrozenJsonSpec
) -> tuple[dict[str, Any], dict[str, str]]:
    path = _resolve_bound_path(root, spec.relative_path.as_posix(), spec.role)
    file_sha = _sha256_file(path, spec.role)
    _require(file_sha == spec.sha256, f"{spec.role} whole-file SHA drifted")
    document = _load_json(path, spec.role)
    _require(
        document.get("schema_version") == 1
        and document.get("analysis_class") == spec.analysis_class,
        f"{spec.role} schema or analysis class drifted",
    )
    _require(
        document.get("family_id") == IBR1_FAMILY_ID
        or spec is PRIMARY_SPEC,
        f"{spec.role} family identity drifted",
    )
    _sealed(document, spec.role)
    payload_sha = _verify_payload_self_hash(document, spec.role)
    _require(
        payload_sha == spec.payload_sha256,
        f"{spec.role} payload SHA differs from the frozen authority",
    )
    return document, {
        "role": spec.role,
        "path": spec.relative_path.as_posix(),
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
        "analysis_class": spec.analysis_class,
    }


def _verify_primary_provenance(
    root: Path, primary: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    recorded = primary.get("primary_provenance")
    _require(isinstance(recorded, Mapping), "PRIMARY provenance is malformed")
    _require(
        set(recorded) == set(PRIMARY_PROVENANCE),
        "PRIMARY provenance key set drifted",
    )
    bindings: dict[str, dict[str, str]] = {}
    for name, (relative, expected_sha) in PRIMARY_PROVENANCE.items():
        item = recorded.get(name)
        _require(isinstance(item, Mapping), f"PRIMARY {name} binding is malformed")
        _require(
            item.get("path") == relative.as_posix()
            and item.get("sha256") == expected_sha,
            f"PRIMARY {name} binding drifted",
        )
        path = _resolve_bound_path(root, relative.as_posix(), f"PRIMARY {name}")
        actual_sha = _sha256_file(path, f"PRIMARY {name}")
        _require(actual_sha == expected_sha, f"PRIMARY {name} bytes drifted")
        bindings[name] = {
            "path": relative.as_posix(),
            "sha256": actual_sha,
        }
    command = _load_json(
        _resolve_bound_path(
            root, PRIMARY_PROVENANCE["command"][0].as_posix(), "PRIMARY command"
        ),
        "PRIMARY command",
    )
    _sealed(command, "PRIMARY command")
    _require(
        command.get("external_model_api_called") is False
        and command.get("historical_fable_primary_superseded") is False,
        "PRIMARY authority provenance was overstated",
    )
    return bindings


def _verify_primary_components(
    root: Path, primary: Mapping[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    recorded = primary.get("component_bindings")
    _require(isinstance(recorded, Mapping), "PRIMARY component bindings malformed")
    _require(
        set(recorded) == set(COMPONENT_SPECS),
        "PRIMARY component binding key set drifted",
    )
    bindings: dict[str, dict[str, str]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name, spec in COMPONENT_SPECS.items():
        document, binding = _binding_for_spec(root, spec)
        item = recorded.get(name)
        _require(isinstance(item, Mapping), f"PRIMARY component {name} malformed")
        _require(
            item.get("path") == binding["path"]
            and item.get("sha256") == binding["sha256"]
            and item.get("payload_sha256")
            == binding["receipt_payload_sha256"],
            f"PRIMARY component {name} binding drifted",
        )
        bindings[name] = binding
        documents[name] = document
    return bindings, documents


def verify_f2_negative_evidence(project_root: str | Path) -> dict[str, Any]:
    """Verify adoption -> F2 seal -> sealed summary -> update-0 init chain."""

    root = Path(project_root).expanduser().resolve()
    adoption, adoption_binding = _binding_for_spec(
        root, COMPONENT_SPECS["negative_result_adoption"]
    )
    parent = adoption.get("adopted_parent")
    _require(isinstance(parent, Mapping), "negative-result adoption has no parent")
    _require(
        parent.get("path") == F2_NEGATIVE_SEAL_RELATIVE.as_posix()
        and parent.get("sha256") == F2_NEGATIVE_SEAL_SHA256
        and parent.get("decision") == "FAIL_STOP"
        and parent.get("valid_input") is True
        and parent.get("engineering_failure") is False
        and parent.get("scientific_negative_result") is True,
        "negative-result adoption does not bind the sealed valid F2 failure",
    )
    prohibitions = adoption.get("adopted_prohibitions")
    _require(
        isinstance(prohibitions, Mapping)
        and all(value is True for value in prohibitions.values()),
        "F2 negative-result prohibitions are incomplete",
    )

    seal_path = _resolve_bound_path(root, str(parent["path"]), "F2 negative seal")
    seal_sha = _sha256_file(seal_path, "F2 negative seal")
    _require(seal_sha == F2_NEGATIVE_SEAL_SHA256, "F2 negative seal bytes drifted")
    seal = _load_json(seal_path, "F2 negative seal")
    _sealed(seal, "F2 negative seal")
    _formal_forbidden(seal.get("gate_outcomes", {}).get("combined", {}), "F2 result")
    run = seal.get("run")
    combined = seal.get("gate_outcomes", {}).get("combined")
    policy = seal.get("seal_policy")
    _require(
        seal.get("analysis_class") == "f2_authoritative_smoke_negative_result_seal"
        and isinstance(run, Mapping)
        and run.get("valid_input") is True
        and run.get("engineering_failure") is False
        and run.get("scientific_negative_result") is True
        and isinstance(combined, Mapping)
        and combined.get("status") == "FAIL"
        and combined.get("decision") == "STOP"
        and isinstance(policy, Mapping)
        and policy.get("same_F2_retry_forbidden") is True
        and policy.get("seed_selection_forbidden") is True
        and policy.get("gate_threshold_tuning_forbidden") is True
        and policy.get("formal_9_run_forbidden") is True,
        "F2 negative seal status or no-retry policy drifted",
    )

    summary_binding = seal.get("evidence", {}).get("smoke_summary")
    _require(isinstance(summary_binding, Mapping), "F2 seal has no smoke summary")
    summary_path = _resolve_bound_path(
        root, summary_binding.get("path"), "sealed F2 smoke summary"
    )
    summary_sha = _sha256_file(summary_path, "sealed F2 smoke summary")
    _require(
        summary_sha == summary_binding.get("sha256"),
        "sealed F2 smoke summary bytes drifted",
    )
    summary = _load_json(summary_path, "sealed F2 smoke summary")
    _sealed(summary, "sealed F2 smoke summary")
    _formal_forbidden(summary, "sealed F2 smoke summary")
    init_sha = _valid_sha256(
        summary.get("checkpoint_init_sha256"), "sealed F2 init SHA"
    )
    update0 = summary.get("checkpoints", {}).get("update0")
    _require(
        summary.get("analysis_class") == "f2_production_smoke_summary"
        and summary.get("seed") == 0
        and summary.get("passed") is False
        and summary.get("status") == "FAIL"
        and summary.get("decision") == "STOP"
        and isinstance(update0, Mapping),
        "sealed F2 smoke summary identity drifted",
    )
    for arm in ("S-CTRL", "S-SELF"):
        arm_binding = update0.get(arm)
        _require(
            isinstance(arm_binding, Mapping)
            and arm_binding.get("state_sha256") == init_sha,
            f"sealed F2 {arm} update-0 state differs from the init SHA",
        )
    seal_authority = seal.get("authority")
    summary_assembly = summary.get("assembly_receipt")
    _require(
        isinstance(seal_authority, Mapping)
        and isinstance(summary_assembly, Mapping)
        and seal_authority.get("assembly_receipt_sha256")
        == summary_assembly.get("sha256")
        and seal_authority.get("assembly_receipt_payload_sha256")
        == summary_assembly.get("payload_sha256"),
        "F2 seal and summary bind different final assemblies",
    )
    document = {
        "schema_version": 1,
        "analysis_class": F2_NEGATIVE_EVIDENCE_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "negative_result_adoption": adoption_binding,
        "negative_seal": {
            "path": F2_NEGATIVE_SEAL_RELATIVE.as_posix(),
            "sha256": seal_sha,
        },
        "sealed_smoke_summary": {
            "path": _root_relative(root, summary_path, "sealed F2 summary"),
            "sha256": summary_sha,
            "seed": 0,
            "checkpoint_init_sha256": init_sha,
            "update0_state_sha256": {
                arm: update0[arm]["state_sha256"] for arm in ("S-CTRL", "S-SELF")
            },
        },
        "decision": "FAIL_STOP",
        "valid_scientific_negative_result": True,
        "formal_training_authorized": False,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    return _with_payload_self_hash(document)


def verify_authority_chain(
    project_root: str | Path,
    recorded_chain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify PRIMARY plus both ordered amendment chains and F2 adoption.

    Supplying ``recorded_chain`` additionally proves that an embedded chain
    is complete and in the frozen order.  A PRIMARY base document by itself
    is intentionally insufficient authority.
    """

    root = Path(project_root).expanduser().resolve()
    primary, primary_binding = _binding_for_spec(root, PRIMARY_SPEC)
    _require(
        primary.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK
        and primary.get("family_id") == IBR1_FAMILY_ID
        and primary.get("formal_training_authorized") is False
        and primary.get("candidate_cap") == 1,
        "PRIMARY family, architecture, cap, or formal policy drifted",
    )
    authority = primary.get("authority")
    _require(
        isinstance(authority, Mapping)
        and authority.get("class") == "user_delegated_local_agent_team_primary"
        and authority.get("external_model_review_completed") is False
        and authority.get("historical_fable_primary_superseded") is False,
        "PRIMARY authority provenance drifted",
    )
    components, component_documents = _verify_primary_components(root, primary)
    provenance = _verify_primary_provenance(root, primary)

    primary_amendment, primary_amendment_binding = _binding_for_spec(
        root, PRIMARY_AMENDMENT1_SPEC
    )
    predecessor = primary_amendment.get("predecessor")
    _require(
        isinstance(predecessor, Mapping)
        and predecessor.get("path") == PRIMARY_RELATIVE.as_posix()
        and predecessor.get("sha256") == PRIMARY_SHA256
        and predecessor.get("payload_sha256") == PRIMARY_PAYLOAD_SHA256,
        "PRIMARY amendment predecessor drifted",
    )
    replacement = primary_amendment.get("replacement_contract")
    _require(
        isinstance(replacement, Mapping)
        and replacement.get("authoritative_geometry_dtype") == "torch.float32"
        and replacement.get("prev_and_latent_same_device_required") is True
        and replacement.get("prev_and_latent_same_dtype_required") is True
        and replacement.get("float16_bfloat16_mixed_dtype_or_amp")
        == "FAIL_CLOSED_NO_SILENT_CAST",
        "PRIMARY amendment effective dtype/exactness contract drifted",
    )
    verdict = primary_amendment.get("verdict_binding")
    _require(
        isinstance(verdict, Mapping)
        and verdict.get("path") == PRIMARY_AMENDMENT1_VERDICT_RELATIVE.as_posix()
        and verdict.get("sha256") == PRIMARY_AMENDMENT1_VERDICT_SHA256,
        "PRIMARY amendment verdict binding drifted",
    )
    amendment_verdict_path = _resolve_bound_path(
        root,
        PRIMARY_AMENDMENT1_VERDICT_RELATIVE.as_posix(),
        "PRIMARY amendment verdict",
    )
    _require(
        _sha256_file(amendment_verdict_path, "PRIMARY amendment verdict")
        == PRIMARY_AMENDMENT1_VERDICT_SHA256,
        "PRIMARY amendment verdict bytes drifted",
    )

    diagnostics, diagnostics_binding = _binding_for_spec(
        root, DIAGNOSTICS_SCHEMA_SPEC
    )
    diagnostics_authority = diagnostics.get("authority")
    _require(
        isinstance(diagnostics_authority, Mapping)
        and diagnostics_authority.get("primary_index_path")
        == PRIMARY_RELATIVE.as_posix()
        and diagnostics_authority.get("primary_index_sha256") == PRIMARY_SHA256
        and diagnostics_authority.get("amendment1_path")
        == PRIMARY_AMENDMENT1_RELATIVE.as_posix()
        and diagnostics_authority.get("amendment1_sha256")
        == PRIMARY_AMENDMENT1_SHA256,
        "diagnostics schema does not bind PRIMARY plus amendment 1",
    )
    diagnostics_amendment, diagnostics_amendment_binding = _binding_for_spec(
        root, DIAGNOSTICS_AMENDMENT1_SPEC
    )
    diagnostics_predecessor = diagnostics_amendment.get("predecessor")
    _require(
        isinstance(diagnostics_predecessor, Mapping)
        and diagnostics_predecessor.get("path")
        == DIAGNOSTICS_SCHEMA_RELATIVE.as_posix()
        and diagnostics_predecessor.get("sha256") == DIAGNOSTICS_SCHEMA_SHA256
        and diagnostics_predecessor.get("payload_sha256")
        == DIAGNOSTICS_SCHEMA_PAYLOAD_SHA256,
        "diagnostics amendment predecessor drifted",
    )
    quantile = diagnostics_amendment.get("quantile_contract")
    cardinality = diagnostics_amendment.get("optimizer_cardinality")
    _require(
        diagnostics_amendment.get("new_thresholds") is False
        and isinstance(quantile, Mapping)
        and "4096 axis cells including zeros"
        in str(quantile.get("deciding_universe_per_arm"))
        and isinstance(cardinality, Mapping)
        and cardinality.get("expected_records") == 256
        and cardinality.get("updates_per_arm") == 128
        and cardinality.get("G6_contribution_geometry_remains_CTRL_only") is True,
        "diagnostics amendment effective universe/cardinality drifted",
    )

    cal_protocol = component_documents["cal_protocol"]
    proposal_policy = cal_protocol.get("lambda_policy")
    _require(
        cal_protocol.get("bootstrap_required") is True
        and cal_protocol.get("optimizer_updates") == 0
        and cal_protocol.get("rows") == CAL_ROWS
        and cal_protocol.get("seed") == CAL_SEED
        and cal_protocol.get("device") == CAL_DEVICE
        and isinstance(proposal_policy, Mapping)
        and proposal_policy.get("proposal_must_equal_inherited_values_exactly")
        is True,
        "CAL preregistration no-drift policy changed",
    )
    _validate_lambda_values(
        proposal_policy.get("inherited_values"), "CAL inherited lambda"
    )
    gate_registry = component_documents["gate_registry"]
    _formal_forbidden(gate_registry, "IBR1 gate registry")
    f2_negative = verify_f2_negative_evidence(root)

    ordered = [
        primary_binding,
        primary_amendment_binding,
        diagnostics_binding,
        diagnostics_amendment_binding,
    ]
    chain = _with_payload_self_hash(
        {
            "schema_version": 1,
            "analysis_class": AUTHORITY_CHAIN_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "ordered_contract_chain": ordered,
            "component_bindings": components,
            "primary_provenance": provenance,
            "effective_overrides": {
                "zero_init_persistence": replacement.get(
                    "zero_init_persistence"
                ),
                "authoritative_geometry_dtype": replacement.get(
                    "authoritative_geometry_dtype"
                ),
                "mixed_dtype_or_amp": replacement.get(
                    "float16_bfloat16_mixed_dtype_or_amp"
                ),
                "overshoot_deciding_universe": quantile.get(
                    "deciding_universe_per_arm"
                ),
                "optimizer_geometry_expected_records": cardinality.get(
                    "expected_records"
                ),
                "g6_contribution_geometry": "IBR1-CTRL only",
            },
            "f2_negative_evidence_payload_sha256": f2_negative[
                "receipt_payload_sha256"
            ],
            "formal_training_authorized": False,
            "internal_test": INTERNAL_TEST_POLICY,
            "internal_test_opened": False,
        }
    )
    if recorded_chain is not None:
        _verify_payload_self_hash(recorded_chain, "recorded authority chain")
        _require(
            dict(recorded_chain) == chain,
            "recorded authority chain is incomplete, reordered, or drifted",
        )
    return chain


def verify_primary_authority(
    project_root: str | Path,
    recorded_chain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility name for the complete, amendment-effective authority."""

    return verify_authority_chain(project_root, recorded_chain)


def _python_tree_bindings(root: Path, relative_dir: str, label: str) -> dict[str, str]:
    directory = (root / relative_dir).resolve()
    _require(directory.is_dir(), f"{label} directory is missing: {directory}")
    files = sorted(
        path
        for path in directory.glob("*.py")
        if path.is_file() and not path.name.startswith("._")
    )
    _require(bool(files), f"{label} has no Python files")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def build_source_binding(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    document = {
        "schema_version": 1,
        "analysis_class": SOURCE_BINDING_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "ibr1_source_sha256": _python_tree_bindings(
            root, "ibr1_experiment", "IBR1 source"
        ),
        "inherited_f2_source_sha256": f2_source_bindings(root),
        "transitive_source_sha256": transitive_source_bindings(root),
        "formal_training_authorized": False,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    return _with_payload_self_hash(document)


def build_test_binding(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    document = {
        "schema_version": 1,
        "analysis_class": TEST_BINDING_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "tests_sha256": _python_tree_bindings(root, "tests/ibr1", "IBR1 tests"),
        "formal_training_authorized": False,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    return _with_payload_self_hash(document)


def _cal_execution_row_payload(
    *,
    position: int,
    original_row_index: int,
    row: Mapping[str, Any],
    reset_reasons: Sequence[str],
) -> dict[str, Any]:
    sequence_id = row.get("sequence_id")
    frame_idx = row.get("frame_idx")
    mirrored = row.get("mirrored")
    previous = row.get("prev_action")
    _require(
        isinstance(sequence_id, str)
        and bool(sequence_id)
        and isinstance(frame_idx, int)
        and not isinstance(frame_idx, bool)
        and isinstance(mirrored, bool),
        f"frozen CAL execution row identity is malformed at position {position}",
    )
    _require(
        isinstance(previous, Sequence)
        and not isinstance(previous, (str, bytes))
        and len(previous) >= 3,
        f"frozen CAL previous action is malformed at position {position}",
    )
    previous_axes: list[float] = []
    for axis, value in enumerate(previous[:3]):
        _require(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value)),
            f"frozen CAL previous action axis {axis} is malformed at position {position}",
        )
        previous_axes.append(float(value))
    reasons = list(reset_reasons)
    _require(
        all(isinstance(reason, str) and bool(reason) for reason in reasons),
        f"frozen CAL reset reasons are malformed at position {position}",
    )
    return {
        "position": position,
        "original_row_index": original_row_index,
        "sequence_id": sequence_id,
        "frame_idx": frame_idx,
        "mirrored": mirrored,
        "prev_action": previous_axes,
        "reset_reasons": reasons,
    }


def _build_cal_execution_binding(
    rows: Sequence[Mapping[str, Any]],
    receipt: Any,
) -> dict[str, Any]:
    ordered = ordered_support_rows(rows, receipt, CAL_SUPPORT)
    reset_plan = support_reset_plan(rows, receipt, CAL_SUPPORT)
    _require(
        len(ordered) == CAL_ROWS and len(reset_plan) == CAL_ROWS,
        "frozen CAL exact execution binding must contain 512 rows",
    )
    records: list[dict[str, Any]] = []
    for position, ((original_row_index, row), reasons) in enumerate(
        zip(ordered, reset_plan, strict=True)
    ):
        payload = _cal_execution_row_payload(
            position=position,
            original_row_index=original_row_index,
            row=row,
            reset_reasons=reasons,
        )
        records.append(
            {
                **payload,
                "row_sha256": canonical_json_sha256(payload),
            }
        )
    return {
        "schema_version": 1,
        "analysis_class": CAL_EXECUTION_BINDING_CLASS,
        "support": CAL_SUPPORT,
        "rows": CAL_ROWS,
        "row_sha_algorithm": "sha256(canonical_json(row_without_row_sha256))",
        "records_sha256": canonical_json_sha256(records),
        "records": records,
    }


def _validate_cal_execution_binding(
    value: Any,
    *,
    ordered_original_indices: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    _require(isinstance(value, Mapping), "CAL exact execution binding is missing")
    records = value.get("records")
    _require(
        value.get("schema_version") == 1
        and value.get("analysis_class") == CAL_EXECUTION_BINDING_CLASS
        and value.get("support") == CAL_SUPPORT
        and value.get("rows") == CAL_ROWS
        and value.get("row_sha_algorithm")
        == "sha256(canonical_json(row_without_row_sha256))"
        and isinstance(records, Sequence)
        and not isinstance(records, (str, bytes))
        and len(records) == CAL_ROWS,
        "CAL exact execution binding identity/cardinality drifted",
    )
    normalized: list[dict[str, Any]] = []
    for position, raw_record in enumerate(records):
        _require(
            isinstance(raw_record, Mapping),
            f"CAL exact execution binding record {position} is malformed",
        )
        record = dict(raw_record)
        row_sha = record.pop("row_sha256", None)
        _require(
            set(record)
            == {
                "position",
                "original_row_index",
                "sequence_id",
                "frame_idx",
                "mirrored",
                "prev_action",
                "reset_reasons",
            }
            and record.get("position") == position
            and record.get("original_row_index")
            == ordered_original_indices[position]
            and isinstance(record.get("sequence_id"), str)
            and bool(record.get("sequence_id"))
            and isinstance(record.get("frame_idx"), int)
            and not isinstance(record.get("frame_idx"), bool)
            and isinstance(record.get("mirrored"), bool),
            f"CAL exact execution binding row identity drifted at position {position}",
        )
        previous = record.get("prev_action")
        reasons = record.get("reset_reasons")
        _require(
            isinstance(previous, Sequence)
            and not isinstance(previous, (str, bytes))
            and len(previous) == 3
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in previous
            )
            and isinstance(reasons, Sequence)
            and not isinstance(reasons, (str, bytes))
            and all(isinstance(reason, str) and bool(reason) for reason in reasons)
            and row_sha == canonical_json_sha256(record),
            f"CAL exact execution binding payload/SHA drifted at position {position}",
        )
        normalized.append({**record, "row_sha256": str(row_sha)})
    _require(
        value.get("records_sha256") == canonical_json_sha256(normalized)
        and value.get("records_sha256")
        == CAL_EXECUTION_BINDING_RECORDS_SHA256,
        "CAL exact execution binding aggregate SHA drifted",
    )
    return tuple(normalized)


def _default_support_observer(root: Path) -> Mapping[str, Any]:
    train_path = root / FROZEN_TRAIN_RELATIVE
    receipt = build_frozen_support(train_path)
    rows = parse_train_jsonl(train_path.read_bytes())
    cal_execution_binding = _build_cal_execution_binding(rows, receipt)
    supports: dict[str, Any] = {}
    for name in ("CAL", "SMK-TRAIN", "EVAL-FIX"):
        ordered = [
            index for block in receipt.supports[name] for index in block.row_indices
        ]
        supports[name] = {
            "rows": len(ordered),
            "blocks": len(receipt.supports[name]),
            "ordered_original_indices": ordered,
            "ordered_original_indices_sha256": canonical_json_sha256(ordered),
            "row_set_sha256": receipt.row_sha256[name],
            "coverage": receipt.coverage[name].to_dict(),
        }
        if name == CAL_SUPPORT:
            supports[name]["execution_binding"] = cal_execution_binding
    return {
        "train": {
            "path": FROZEN_TRAIN_RELATIVE.as_posix(),
            "rows": receipt.train_rows,
            "sha256": receipt.train_sha256,
        },
        "supports": supports,
        "union": {
            "rows": len(receipt.union_row_indices),
            "sha256": receipt.union_sha256,
        },
        "inherited_support_contract_payload_sha256": canonical_json_sha256(
            receipt.to_dict()
        ),
    }


def _validate_support_observation(observed: Mapping[str, Any]) -> None:
    train = observed.get("train")
    supports = observed.get("supports")
    union = observed.get("union")
    _require(
        isinstance(train, Mapping)
        and train.get("path") == FROZEN_TRAIN_RELATIVE.as_posix()
        and train.get("rows") == FROZEN_TRAIN_ROWS
        and train.get("sha256") == FROZEN_TRAIN_SHA256,
        "fresh support binding train identity drifted",
    )
    _require(
        isinstance(supports, Mapping)
        and set(supports) == {"CAL", "SMK-TRAIN", "EVAL-FIX"},
        "fresh support binding support set drifted",
    )
    all_indices: list[int] = []
    for name in ("CAL", "SMK-TRAIN", "EVAL-FIX"):
        support = supports[name]
        expectation = SUPPORT_EXPECTATIONS[name]
        _require(isinstance(support, Mapping), f"{name} support binding malformed")
        indices = support.get("ordered_original_indices")
        _require(
            isinstance(indices, Sequence)
            and not isinstance(indices, (str, bytes))
            and len(indices) == expectation.rows
            and all(
                isinstance(index, int) and not isinstance(index, bool)
                for index in indices
            )
            and len(set(indices)) == len(indices),
            f"{name} ordered original-index coverage drifted",
        )
        ordered = list(indices)
        _require(
            support.get("rows") == expectation.rows
            and support.get("blocks") == expectation.blocks
            and support.get("ordered_original_indices_sha256")
            == canonical_json_sha256(ordered)
            and support.get("ordered_original_indices_sha256")
            == expectation.sha256
            and support.get("row_set_sha256") == expectation.sha256,
            f"{name} frozen ordered support identity drifted",
        )
        if name == CAL_SUPPORT:
            _validate_cal_execution_binding(
                support.get("execution_binding"),
                ordered_original_indices=ordered,
            )
        all_indices.extend(ordered)
    _require(
        len(all_indices) == len(set(all_indices)),
        "CAL/SMK-TRAIN/EVAL-FIX fresh supports are not disjoint",
    )
    _require(
        isinstance(union, Mapping)
        and union.get("rows") == len(all_indices)
        and union.get("sha256") == canonical_json_sha256(sorted(all_indices))
        and union.get("sha256")
        == "906f990a34ed9bcc6c852f7295293b467628ab56c465d41450d4ae9715aa19be",
        "fresh support union identity drifted",
    )
    _valid_sha256(
        observed.get("inherited_support_contract_payload_sha256"),
        "inherited support contract payload SHA",
    )


def build_support_binding(
    project_root: str | Path,
    *,
    observer: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    if observer is None:
        observer = _default_support_observer
    observed = observer(root)
    _require(isinstance(observed, Mapping), "support observer returned no mapping")
    _validate_support_observation(observed)
    return _with_payload_self_hash(
        {
            "schema_version": 1,
            "analysis_class": SUPPORT_BINDING_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "observation": dict(observed),
            "formal_training_authorized": False,
            "internal_test": INTERNAL_TEST_POLICY,
            "internal_test_opened": False,
        }
    )


def _default_asset_observer(root: Path) -> Mapping[str, Any]:
    assets = verify_frozen_assets(root, verify_token_payload=False)
    ledger = build_train_token_ledger(root)
    return {
        **dict(assets),
        "token_ledger_sha256": ledger.ledger_sha256,
        "token_ledger_file_count": ledger.token_files,
    }


def _validate_asset_observation(observed: Mapping[str, Any]) -> None:
    train = observed.get("train")
    cache = observed.get("vision_cache")
    base_hf = observed.get("base_hf")
    qwen = observed.get("qwen")
    erratum = observed.get("prompt_erratum")
    _require(
        observed.get("internal_test_opened") is False,
        "asset verification opened the sealed internal test",
    )
    _require(
        isinstance(train, Mapping)
        and train.get("relative_path") == FROZEN_TRAIN_RELATIVE.as_posix()
        and train.get("rows") == FROZEN_TRAIN_ROWS
        and train.get("sha256") == FROZEN_TRAIN_SHA256,
        "asset train binding drifted",
    )
    expected_cache = {
        "cache_manifest_sha256": FROZEN_CACHE_MANIFEST_SHA256,
        "cache_provenance_sha256": FROZEN_CACHE_PROVENANCE_SHA256,
        "token_payload_sha256": FROZEN_TOKEN_PAYLOAD_SHA256,
        "dino_model_sha256": FROZEN_DINO_SHA256,
        "siglip_model_sha256": FROZEN_SIGLIP_SHA256,
    }
    _require(isinstance(cache, Mapping), "asset vision-cache binding malformed")
    for field, expected in expected_cache.items():
        _require(cache.get(field) == expected, f"asset cache {field} drifted")
    _require(
        cache.get("token_payload_verified") is False,
        "asset binding must not traverse the full token payload/internal test",
    )
    _require(
        isinstance(base_hf, Mapping)
        and base_hf.get("artifact_sha256") == FROZEN_BASE_HF_ARTIFACT_SHA256,
        "base HF artifact identity drifted",
    )
    _require(
        isinstance(qwen, Mapping)
        and qwen.get("artifact_sha256") == FROZEN_QWEN_SHA256,
        "Qwen artifact identity drifted",
    )
    _require(
        isinstance(erratum, Mapping)
        and erratum.get("sha256") == FROZEN_PROMPT_ERRATUM_SHA256,
        "prompt erratum identity drifted",
    )
    _valid_sha256(observed.get("token_ledger_sha256"), "token ledger SHA")
    _require(
        observed.get("token_ledger_file_count") == FROZEN_TRAIN_TOKEN_FILES,
        "train-split token ledger cardinality drifted",
    )


def build_asset_binding(
    project_root: str | Path,
    *,
    observer: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    if observer is None:
        observer = _default_asset_observer
    observed = observer(root)
    _require(isinstance(observed, Mapping), "asset observer returned no mapping")
    _validate_asset_observation(observed)
    ledger_sha = str(observed["token_ledger_sha256"])
    ledger_file_count = int(observed["token_ledger_file_count"])
    return _with_payload_self_hash(
        {
            "schema_version": 1,
            "analysis_class": ASSET_BINDING_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "verification_mode": (
                "manifest_metadata_plus_train_split_per_file_ledger; "
                "no_full_payload_reread"
            ),
            # The subordinate F2 CAL kernel consumes the inherited assembly
            # contract at ``asset_binding.token_ledger_*``. Freeze the exact
            # same anchor at that compatibility surface while retaining the
            # complete IBR1 observation below; verification requires the two
            # copies to remain identical, so this cannot become a second
            # independently mutable authority.
            "token_ledger_sha256": ledger_sha,
            "token_ledger_file_count": ledger_file_count,
            "observation": dict(observed),
            "formal_training_authorized": False,
            "internal_test": INTERNAL_TEST_POLICY,
            "internal_test_opened": False,
        }
    )


def _validate_lambda_values(value: Any, label: str) -> dict[str, float]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(
        set(value) == set(FROZEN_AUX_COEFFICIENTS),
        f"{label} must cover exactly the three frozen auxiliary losses",
    )
    normalized: dict[str, float] = {}
    for name, expected in FROZEN_AUX_COEFFICIENTS.items():
        observed = value[name]
        _require(
            not isinstance(observed, bool) and isinstance(observed, (int, float)),
            f"{label}.{name} must be numeric",
        )
        numeric = float(observed)
        _require(
            math.isfinite(numeric) and numeric == expected,
            f"{label}.{name} must equal inherited value {expected!r} exactly",
        )
        normalized[name] = numeric
    return normalized


def _receipt_binding(root: Path, path: Path, document: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": _root_relative(root, path, "receipt binding"),
        "sha256": _sha256_file(path, "receipt binding"),
        "receipt_payload_sha256": _verify_payload_self_hash(
            document, "receipt binding"
        ),
        "analysis_class": str(document.get("analysis_class")),
    }


def _verify_nested_binding(
    value: Any, *, expected_class: str, label: str
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is malformed")
    _require(
        value.get("analysis_class") == expected_class,
        f"{label} analysis class drifted",
    )
    _sealed(value, label)
    _formal_forbidden(value, label)
    _verify_payload_self_hash(value, label)
    return value


def _verify_asset_ledger_anchor(binding: Mapping[str, Any]) -> None:
    """Require the IBR1/F2 token-ledger compatibility anchor to be singular."""

    observation = binding.get("observation")
    _require(
        isinstance(observation, Mapping),
        "assembly asset binding observation is missing",
    )
    ledger_sha = _valid_sha256(
        binding.get("token_ledger_sha256"),
        "assembly asset token ledger SHA",
    )
    ledger_file_count = binding.get("token_ledger_file_count")
    _require(
        isinstance(ledger_file_count, int)
        and not isinstance(ledger_file_count, bool)
        and ledger_file_count == FROZEN_TRAIN_TOKEN_FILES,
        "assembly asset token ledger cardinality drifted",
    )
    _require(
        observation.get("token_ledger_sha256") == ledger_sha
        and observation.get("token_ledger_file_count") == ledger_file_count,
        "assembly asset token ledger compatibility anchor differs from "
        "the verified IBR1 observation",
    )


def _verify_static_assembly(
    path: Path, *, expected_phase: AssemblyPhase | None = None
) -> dict[str, Any]:
    document = _load_canonical_receipt(path, "IBR1 assembly receipt")
    _require(
        document.get("schema_version") == ASSEMBLY_SCHEMA_VERSION
        and document.get("receipt_version") == ASSEMBLY_RECEIPT_VERSION
        and document.get("analysis_class") == ASSEMBLY_RECEIPT_CLASS
        and document.get("family_id") == IBR1_FAMILY_ID
        and document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK,
        "IBR1 assembly identity drifted",
    )
    _sealed(document, "IBR1 assembly receipt")
    _formal_forbidden(document, "IBR1 assembly receipt")
    phase = document.get("phase")
    _require(
        phase in (ASSEMBLY_PHASE_BOOTSTRAP, ASSEMBLY_PHASE_FINAL),
        "IBR1 assembly phase is invalid",
    )
    if expected_phase is not None:
        _require(phase == expected_phase, "IBR1 assembly phase differs from expected")
    freeze = document.get("lambda_freeze_binding")
    if phase == ASSEMBLY_PHASE_BOOTSTRAP:
        _require(freeze is None, "bootstrap assembly must bind freeze=null")
    else:
        _require(isinstance(freeze, Mapping), "final assembly must bind adoption freeze")
    _verify_nested_binding(
        document.get("source_binding"),
        expected_class=SOURCE_BINDING_CLASS,
        label="assembly source binding",
    )
    _verify_nested_binding(
        document.get("test_binding"),
        expected_class=TEST_BINDING_CLASS,
        label="assembly test binding",
    )
    _verify_nested_binding(
        document.get("support_binding"),
        expected_class=SUPPORT_BINDING_CLASS,
        label="assembly support binding",
    )
    asset_binding = _verify_nested_binding(
        document.get("asset_binding"),
        expected_class=ASSET_BINDING_CLASS,
        label="assembly asset binding",
    )
    _verify_asset_ledger_anchor(asset_binding)
    _verify_nested_binding(
        document.get("authority_chain"),
        expected_class=AUTHORITY_CHAIN_CLASS,
        label="assembly authority chain",
    )
    _verify_nested_binding(
        document.get("f2_negative_evidence"),
        expected_class=F2_NEGATIVE_EVIDENCE_CLASS,
        label="assembly F2 negative evidence",
    )
    return document


def _bootstrap_compatibility_fields(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: document[field]
        for field in (
            "source_binding",
            "test_binding",
            "support_binding",
            "asset_binding",
            "authority_chain",
            "f2_negative_evidence",
        )
    }


def _build_assembly_receipt_document(
    project_root: str | Path,
    *,
    phase: AssemblyPhase,
    lambda_adoption_freeze_path: str | Path | None = None,
    support_observer: Callable[[Path], Mapping[str, Any]] | None = None,
    asset_observer: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive an assembly document for trusted builders and verification."""

    root = Path(project_root).expanduser().resolve()
    _require(
        phase in (ASSEMBLY_PHASE_BOOTSTRAP, ASSEMBLY_PHASE_FINAL),
        "assembly phase must be 'bootstrap' or 'final'",
    )
    if phase == ASSEMBLY_PHASE_BOOTSTRAP:
        _require(
            lambda_adoption_freeze_path is None,
            "bootstrap assembly must be built with freeze=null",
        )
    else:
        _require(
            lambda_adoption_freeze_path is not None,
            "final assembly requires an IBR1 lambda-adoption freeze",
        )

    source_binding = build_source_binding(root)
    test_binding = build_test_binding(root)
    support_binding = build_support_binding(root, observer=support_observer)
    asset_binding = build_asset_binding(root, observer=asset_observer)
    authority_chain = verify_authority_chain(root)
    f2_negative = verify_f2_negative_evidence(root)
    _require(
        authority_chain.get("f2_negative_evidence_payload_sha256")
        == f2_negative.get("receipt_payload_sha256"),
        "authority chain and assembly bind different F2 negative evidence",
    )

    freeze_binding: dict[str, str] | None = None
    if lambda_adoption_freeze_path is not None:
        freeze_path = Path(lambda_adoption_freeze_path).expanduser().resolve()
        _root_relative(root, freeze_path, "lambda-adoption freeze")
        freeze = verify_lambda_adoption_freeze(root, freeze_path)
        bootstrap_binding = freeze["evidence"]["bootstrap_assembly_receipt"]
        bootstrap_path = _resolve_bound_path(
            root, bootstrap_binding["path"], "freeze bootstrap assembly"
        )
        bootstrap = _verify_static_assembly(
            bootstrap_path, expected_phase=ASSEMBLY_PHASE_BOOTSTRAP
        )
        current_fields = {
            "source_binding": source_binding,
            "test_binding": test_binding,
            "support_binding": support_binding,
            "asset_binding": asset_binding,
            "authority_chain": authority_chain,
            "f2_negative_evidence": f2_negative,
        }
        _require(
            _bootstrap_compatibility_fields(bootstrap) == current_fields,
            "live final assembly differs from the CAL bootstrap assembly",
        )
        freeze_binding = _receipt_binding(root, freeze_path, freeze)

    document = {
        "schema_version": ASSEMBLY_SCHEMA_VERSION,
        "receipt_version": ASSEMBLY_RECEIPT_VERSION,
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "phase": phase,
        "project_root": str(root),
        "source_binding": source_binding,
        "test_binding": test_binding,
        "support_binding": support_binding,
        "asset_binding": asset_binding,
        "authority_chain": authority_chain,
        "f2_negative_evidence": f2_negative,
        "lambda_freeze_binding": freeze_binding,
        "candidate_cap": 1,
        "formal_training_authorized": False,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    return _with_payload_self_hash(document)


def build_assembly_receipt(
    project_root: str | Path,
    *,
    phase: AssemblyPhase,
    lambda_adoption_freeze_path: str | Path | None = None,
    support_observer: Callable[[Path], Mapping[str, Any]] | None = None,
    asset_observer: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build bootstrap authority; reject every file-only final construction."""

    _require(
        phase != ASSEMBLY_PHASE_FINAL,
        "file-only build-final is forbidden; final authority requires the "
        "same-call live CAL capability",
    )
    return _build_assembly_receipt_document(
        project_root,
        phase=phase,
        lambda_adoption_freeze_path=lambda_adoption_freeze_path,
        support_observer=support_observer,
        asset_observer=asset_observer,
    )


def freeze_assembly_receipt(
    project_root: str | Path,
    output: str | Path,
    *,
    phase: AssemblyPhase,
    lambda_adoption_freeze_path: str | Path | None = None,
    support_observer: Callable[[Path], Mapping[str, Any]] | None = None,
    asset_observer: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    document = build_assembly_receipt(
        project_root,
        phase=phase,
        lambda_adoption_freeze_path=lambda_adoption_freeze_path,
        support_observer=support_observer,
        asset_observer=asset_observer,
    )
    destination = Path(output).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    _root_relative(root, destination, "assembly receipt output")
    file_sha = exclusive_write_json(destination, document)
    return {
        "path": str(destination),
        "sha256": file_sha,
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        "phase": phase,
    }


def _freeze_final_assembly_receipt_from_live_cal(
    project_root: str | Path,
    output: str | Path,
    *,
    lambda_adoption_freeze_path: str | Path,
    live_pair_proof: Any,
) -> dict[str, str]:
    """Consume one genuine live-pair proof while issuing the final receipt."""

    root = Path(project_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    freeze_path = Path(lambda_adoption_freeze_path).expanduser().resolve()
    _root_relative(root, destination, "final assembly receipt output")
    _root_relative(root, freeze_path, "lambda-adoption freeze")
    _require(not destination.exists(), "final assembly receipt already exists")

    # Import lazily to avoid an authority/cal-pair import cycle.  The cal-pair
    # module validates the exact private class/secret and burns the proof before
    # any file-derived assembly work begins.
    from . import cal_pair

    cal_pair._consume_live_pair_proof_for_final(
        live_pair_proof,
        project_root=root,
        freeze_path=freeze_path,
    )
    document = _build_assembly_receipt_document(
        root,
        phase=ASSEMBLY_PHASE_FINAL,
        lambda_adoption_freeze_path=freeze_path,
    )
    file_sha = exclusive_write_json(destination, document)
    return {
        "path": str(destination),
        "sha256": file_sha,
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        "phase": ASSEMBLY_PHASE_FINAL,
    }


def verify_assembly_receipt(
    project_root: str | Path,
    receipt_path: str | Path,
    *,
    required_phase: AssemblyPhase | None = None,
    support_observer: Callable[[Path], Mapping[str, Any]] | None = None,
    asset_observer: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(receipt_path).expanduser().resolve()
    _root_relative(root, path, "assembly receipt")
    document = _verify_static_assembly(path, expected_phase=required_phase)
    freeze = document.get("lambda_freeze_binding")
    freeze_path: Path | None = None
    if isinstance(freeze, Mapping):
        freeze_path = _resolve_bound_path(root, freeze.get("path"), "adoption freeze")
        _require(
            freeze.get("analysis_class") == LAMBDA_ADOPTION_FREEZE_CLASS
            and freeze.get("sha256") == _sha256_file(freeze_path, "adoption freeze"),
            "assembly adoption-freeze binding drifted",
        )
    expected = _build_assembly_receipt_document(
        root,
        phase=document["phase"],
        lambda_adoption_freeze_path=freeze_path,
        support_observer=support_observer,
        asset_observer=asset_observer,
    )
    _require(
        document == expected,
        "assembly receipt differs from the live fresh IBR1 authority",
    )
    return document


def _bootstrap_binding(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    bootstrap = _verify_static_assembly(
        path, expected_phase=ASSEMBLY_PHASE_BOOTSTRAP
    )
    return bootstrap, _receipt_binding(root, path, bootstrap)


def _validate_cal_context(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "CAL context must be a mapping")
    checkpoint_sha = _valid_sha256(
        value.get("checkpoint_init_sha256"), "CAL checkpoint init SHA"
    )
    _require(
        value.get("seed") == CAL_SEED
        and value.get("device") == CAL_DEVICE
        and value.get("package") == CAL_PACKAGE
        and value.get("probe_surface") == CAL_PROBE_SURFACE,
        "CAL seed/device/package/probe context drifted",
    )
    try:
        reproducibility = validate_cuda_reproducibility_receipt(
            value.get("cuda_reproducibility")
        )
    except F2CudaReproducibilityError as exc:
        raise IBR1AuthorityError(
            f"CAL CUDA reproducibility binding is invalid: {exc}"
        ) from exc
    normalized = dict(value)
    normalized["checkpoint_init_sha256"] = checkpoint_sha
    normalized["cuda_reproducibility"] = reproducibility
    canonical_json_bytes(normalized)
    return normalized


def _bootstrap_authority_payloads(bootstrap: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: bootstrap[field]["receipt_payload_sha256"]
        for name, field in (
            ("source", "source_binding"),
            ("tests", "test_binding"),
            ("support", "support_binding"),
            ("assets", "asset_binding"),
            ("preregistration", "authority_chain"),
            ("f2_negative_evidence", "f2_negative_evidence"),
        )
    }


def _round_sig(value: float, digits: int = 3) -> float:
    _require(digits > 0, "significant digits must be positive")
    _require(math.isfinite(value), "cannot round a nonfinite value")
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return round(value, digits - 1 - exponent)


def _load_canonical_finite_json(path: Path, label: str) -> dict[str, Any]:
    document = _load_json(path, label)
    _reject_nested_authority_escalation(document, label)
    _require(
        path.read_bytes() == canonical_json_bytes(document) + b"\n",
        f"{label} is not canonical finite JSON plus LF",
    )
    return document


def _portable_filename(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and bool(value)
        and Path(value).name == value
        and PurePosixPath(value).as_posix() == value,
        f"{label} is not a plain portable filename",
    )
    return value


def _finite_positive(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    numeric = float(value)
    _require(math.isfinite(numeric) and numeric > 0.0, f"{label} must be positive finite")
    return numeric


def _derive_f2_lambda(medians: Mapping[str, float]) -> dict[str, float]:
    _require(
        set(medians) == set(FROZEN_AUX_COEFFICIENTS),
        "F2 auxiliary medians do not cover the frozen losses",
    )
    minimum = min(medians.values())
    return {
        name: _round_sig(min(0.5 * minimum / medians[name], 1.0), 3)
        for name in FROZEN_AUX_COEFFICIENTS
    }


def _live_bootstrap_snapshot(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    document = verify_assembly_receipt(
        root, path, required_phase=ASSEMBLY_PHASE_BOOTSTRAP
    )
    return document, _receipt_binding(root, path, document)


def _sealed_init_sha(bootstrap: Mapping[str, Any]) -> str:
    return _valid_sha256(
        bootstrap.get("f2_negative_evidence", {})
        .get("sealed_smoke_summary", {})
        .get("checkpoint_init_sha256"),
        "bootstrap sealed F2 init SHA",
    )


def _verify_f2_raw_cal_receipt(
    root: Path,
    path: Path,
    *,
    bootstrap: Mapping[str, Any],
    bootstrap_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _root_relative(root, path, "raw F2 CAL kernel")
    document = _load_canonical_finite_json(path, "raw F2 CAL kernel")
    _require(
        document.get("analysis_class") == "f2_cal_zero_update_audit_receipt"
        and document.get("architecture_lock") == "L1+D2+AP2+F2"
        and document.get("package") == CAL_PACKAGE
        and document.get("support") == CAL_SUPPORT
        and document.get("rows") == CAL_ROWS
        and document.get("optimizer_updates") == 0,
        "raw F2 CAL kernel identity/zero-update contract drifted",
    )
    _sealed(document, "raw F2 CAL kernel")
    context = _validate_cal_context(document.get("cal_context"))
    _require(
        context["checkpoint_init_sha256"] == _sealed_init_sha(bootstrap),
        "raw F2 CAL init SHA differs from sealed F2 update-0 evidence",
    )
    amendment = document.get("amendment_binding")
    _require(
        isinstance(amendment, Mapping)
        and amendment.get("amendment_id") == "f2-adjudication-amendment-1"
        and amendment.get("sha256")
        == "2adb79ec3cd5f7d077eec23f10fac1da71eb3bd86135ea9c2837db90b065d40c",
        "raw F2 CAL amendment binding drifted",
    )
    _require(
        document.get("assembly_receipt_sha256") == bootstrap_binding.get("sha256")
        and document.get("assembly_receipt_payload_sha256")
        == bootstrap_binding.get("receipt_payload_sha256"),
        "raw F2 CAL kernel binds a different bootstrap assembly",
    )
    ledger = document.get("token_ledger_binding")
    asset = bootstrap["asset_binding"]["observation"]
    _require(
        isinstance(ledger, Mapping)
        and ledger.get("anchor") == "trust_on_first_read_at_freeze"
        and ledger.get("sha256") == asset.get("token_ledger_sha256")
        and ledger.get("file_count") == asset.get("token_ledger_file_count"),
        "raw F2 CAL token ledger differs from bootstrap assets",
    )
    step0 = document.get("step0_parity")
    prev_free = document.get("prev_free_graph_audit")
    zero_init = document.get("ap2_zero_init_proof")
    _require(
        isinstance(step0, Mapping)
        and step0.get("checked_rows") == CAL_ROWS
        and step0.get("failures") == 0,
        "raw F2 CAL step0 parity did not cover 512 rows with zero failures",
    )
    _require(
        isinstance(prev_free, Mapping)
        and prev_free.get("checked_rows") == CAL_ROWS
        and prev_free.get("failures") == 0,
        "raw F2 CAL prev-free audit did not cover 512 rows with zero failures",
    )
    _require(
        isinstance(zero_init, Mapping)
        and zero_init.get("checked_rows") == CAL_ROWS
        and zero_init.get("violations") == 0
        and float(zero_init.get("track_grad_norm_max", math.nan)) == 0.0,
        "raw F2 CAL AP2 zero-init proof failed",
    )
    static_reset = document.get("static_reset_receipt")
    _require(
        isinstance(static_reset, Mapping)
        and static_reset.get("expected")
        == SUPPORT_EXPECTATIONS[CAL_SUPPORT].static_resets
        and static_reset.get("observed")
        == SUPPORT_EXPECTATIONS[CAL_SUPPORT].static_resets
        and isinstance(static_reset.get("original_indices_sha256"), str),
        "raw F2 CAL static-reset receipt drifted",
    )
    gradient = document.get("gradient_calibration")
    raw_medians = gradient.get("per_aux_grad_norm_median") if isinstance(gradient, Mapping) else None
    _require(isinstance(raw_medians, Mapping), "raw F2 CAL auxiliary medians are missing")
    medians = {
        name: _finite_positive(raw_medians.get(name), f"raw F2 median {name}")
        for name in FROZEN_AUX_COEFFICIENTS
    }
    derived_proposal = _derive_f2_lambda(medians)
    _validate_lambda_values(derived_proposal, "recomputed F2 proposal")
    calibration = document.get("lambda_calibration")
    recorded_proposal = calibration.get("proposed_lambda") if isinstance(calibration, Mapping) else None
    _require(
        _validate_lambda_values(recorded_proposal, "raw F2 recorded proposal")
        == derived_proposal,
        "raw F2 recorded proposal differs from the frozen median formula",
    )
    binding = {
        "filename": path.name,
        "sha256": _sha256_file(path, "raw F2 CAL kernel"),
        "canonical_payload_sha256": canonical_json_sha256(document),
        "analysis_class": "f2_cal_zero_update_audit_receipt",
    }
    derived = {
        "cal_context": context,
        "per_aux_grad_norm_median": medians,
        "lambda_proposal": derived_proposal,
        "step0": dict(step0),
        "prev_free": dict(prev_free),
        "ap2_zero_init": dict(zero_init),
        "token_ledger_binding": dict(ledger),
        "static_reset_receipt": dict(static_reset),
    }
    return document, binding, derived


def _validate_callback_transcript(
    value: Any,
    *,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "IBR1 callback transcript is missing")
    records = value.get("records")
    _require(
        value.get("schema_version") == 1
        and value.get("analysis_class") == CAL_CALLBACK_TRANSCRIPT_CLASS
        and value.get("rows") == CAL_ROWS
        and value.get("initial_sha256") == "0" * 64
        and value.get("chain_algorithm")
        == (
            "sha256(canonical_json({previous_sha256,"
            "record_without_chain_fields}))"
        )
        and isinstance(records, Sequence)
        and not isinstance(records, (str, bytes))
        and len(records) == CAL_ROWS,
        "IBR1 callback transcript identity/cardinality drifted",
    )
    cal_support = (
        bootstrap.get("support_binding", {})
        .get("observation", {})
        .get("supports", {})
        .get(CAL_SUPPORT, {})
    )
    support_indices = (
        cal_support.get("ordered_original_indices")
        if isinstance(cal_support, Mapping)
        else None
    )
    _require(
        isinstance(support_indices, Sequence)
        and not isinstance(support_indices, (str, bytes))
        and len(support_indices) == CAL_ROWS,
        "bootstrap CAL support order is unavailable",
    )
    execution_records = _validate_cal_execution_binding(
        cal_support.get("execution_binding")
        if isinstance(cal_support, Mapping)
        else None,
        ordered_original_indices=list(support_indices),
    )
    previous_sha = "0" * 64
    aux_norms: dict[str, list[float]] = {
        name: [] for name in FROZEN_AUX_COEFFICIENTS
    }
    post_abs_max = 0.0
    reconstruction_error_max = 0.0
    reset_original_indices: list[int] = []
    for position, raw_entry in enumerate(records):
        _require(
            isinstance(raw_entry, Mapping),
            f"callback transcript record {position} is malformed",
        )
        entry = dict(raw_entry)
        stored_previous = entry.pop("previous_sha256", None)
        stored_record_sha = entry.pop("record_sha256", None)
        _require(
            set(entry)
            == {
                "position",
                "row_identity",
                "reset_reasons",
                "subordinate_f2",
                "ibr1",
            }
            and entry.get("position") == position
            and stored_previous == previous_sha
            and stored_record_sha
            == canonical_json_sha256(
                {"previous_sha256": previous_sha, "record": entry}
            ),
            f"callback transcript hash/position chain broke at {position}",
        )
        previous_sha = str(stored_record_sha)
        row_identity = entry.get("row_identity")
        expected_execution = execution_records[position]
        expected_row_identity = {
            "original_row_index": expected_execution["original_row_index"],
            "sequence_id": expected_execution["sequence_id"],
            "frame_idx": expected_execution["frame_idx"],
            "mirrored": expected_execution["mirrored"],
            "logged_prev_action": expected_execution["prev_action"],
        }
        _require(
            isinstance(row_identity, Mapping)
            and set(row_identity)
            == {
                "original_row_index",
                "sequence_id",
                "frame_idx",
                "mirrored",
                "logged_prev_action",
            }
            and dict(row_identity) == expected_row_identity
            and isinstance(row_identity.get("sequence_id"), str)
            and bool(row_identity.get("sequence_id"))
            and isinstance(row_identity.get("frame_idx"), int)
            and not isinstance(row_identity.get("frame_idx"), bool)
            and isinstance(row_identity.get("mirrored"), bool),
            f"callback transcript row identity drifted at {position}",
        )
        logged_prev = row_identity.get("logged_prev_action")
        _require(
            isinstance(logged_prev, Sequence)
            and not isinstance(logged_prev, (str, bytes))
            and len(logged_prev) == 3
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in logged_prev
            ),
            f"callback transcript logged previous action drifted at {position}",
        )
        reasons = entry.get("reset_reasons")
        _require(
            isinstance(reasons, Sequence)
            and not isinstance(reasons, (str, bytes))
            and all(isinstance(reason, str) and bool(reason) for reason in reasons)
            and list(reasons) == expected_execution["reset_reasons"],
            f"callback transcript exact execution binding drifted at {position}",
        )
        if reasons:
            reset_original_indices.append(int(support_indices[position]))
        subordinate = entry.get("subordinate_f2")
        _require(
            isinstance(subordinate, Mapping)
            and set(subordinate)
            == {
                "step0_parity",
                "prev_free",
                "track_grad_norm",
                "aux_grad_norms",
            }
            and subordinate.get("step0_parity") is True
            and subordinate.get("prev_free") is True
            and not isinstance(subordinate.get("track_grad_norm"), bool)
            and isinstance(subordinate.get("track_grad_norm"), (int, float))
            and float(subordinate.get("track_grad_norm")) == 0.0,
            f"callback transcript subordinate evidence failed at {position}",
        )
        raw_aux = subordinate.get("aux_grad_norms")
        _require(
            isinstance(raw_aux, Mapping)
            and set(raw_aux) == set(FROZEN_AUX_COEFFICIENTS),
            f"callback transcript auxiliary blocks drifted at {position}",
        )
        for name in FROZEN_AUX_COEFFICIENTS:
            observed = raw_aux[name]
            _require(
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and math.isfinite(float(observed))
                and float(observed) >= 0.0,
                f"callback transcript {name} norm failed at {position}",
            )
            aux_norms[name].append(float(observed))
        ibr1 = entry.get("ibr1")
        _require(
            isinstance(ibr1, Mapping)
            and set(ibr1)
            == {
                "geometry_dtype",
                "zero_init_persistence",
                "post_decode_abs_max",
                "controlled_tensor_shape",
                "controlled_cells",
                "realized_delta_reconstruction_error",
                "prev_free_observation_graph",
            }
            and ibr1.get("geometry_dtype") == "torch.float32"
            and ibr1.get("zero_init_persistence") is True
            and ibr1.get("controlled_tensor_shape") == CAL_CONTROLLED_SHAPE
            and ibr1.get("controlled_cells")
            == CAL_CONTROLLED_SHAPE[0] * CAL_CONTROLLED_SHAPE[1]
            and ibr1.get("prev_free_observation_graph") is True,
            f"callback transcript IBR1 identity/shape failed at {position}",
        )
        post_value = ibr1.get("post_decode_abs_max")
        reconstruction_value = ibr1.get("realized_delta_reconstruction_error")
        _require(
            not isinstance(post_value, bool)
            and isinstance(post_value, (int, float))
            and math.isfinite(float(post_value))
            and 0.0 <= float(post_value) <= 1.0,
            f"callback transcript post-decode range failed at {position}",
        )
        _require(
            not isinstance(reconstruction_value, bool)
            and isinstance(reconstruction_value, (int, float))
            and math.isfinite(float(reconstruction_value))
            and 0.0 <= float(reconstruction_value) <= 1e-6,
            f"callback transcript reconstruction failed at {position}",
        )
        post_abs_max = max(post_abs_max, float(post_value))
        reconstruction_error_max = max(
            reconstruction_error_max, float(reconstruction_value)
        )
    _require(
        value.get("final_sha256") == previous_sha
        and value.get("records_sha256") == canonical_json_sha256(list(records)),
        "IBR1 callback transcript final/aggregate SHA drifted",
    )
    medians = {
        name: float(median(values)) for name, values in aux_norms.items()
    }
    _require(
        all(value > 0.0 for value in medians.values()),
        "IBR1 callback transcript contains an unreachable auxiliary median",
    )
    return {
        "per_aux_grad_norm_median": medians,
        "post_decode_abs_max": post_abs_max,
        "reconstruction_error_max": reconstruction_error_max,
        "reset_original_indices": reset_original_indices,
        "records_sha256": str(value["records_sha256"]),
        "final_sha256": str(value["final_sha256"]),
        "rows": CAL_ROWS,
    }


def _verify_ibr1_numeric_evidence(
    root: Path,
    path: Path,
    *,
    bootstrap: Mapping[str, Any],
    bootstrap_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _root_relative(root, path, "IBR1 CAL numeric evidence")
    document = _load_canonical_receipt(path, "IBR1 CAL numeric evidence")
    _require(
        document.get("analysis_class") == CAL_NUMERIC_EVIDENCE_CLASS
        and document.get("family_id") == IBR1_FAMILY_ID
        and document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK
        and document.get("support") == CAL_SUPPORT
        and document.get("rows") == CAL_ROWS
        and document.get("optimizer_updates") == 0
        and document.get("geometry_dtype") == "torch.float32"
        and document.get("row_callback_count") == CAL_ROWS,
        "IBR1 numeric evidence identity/dtype/cardinality drifted",
    )
    _sealed(document, "IBR1 CAL numeric evidence")
    _formal_forbidden(document, "IBR1 CAL numeric evidence")
    expected_bootstrap_binding = {
        "filename": Path(str(bootstrap_binding["path"])).name,
        "sha256": bootstrap_binding["sha256"],
        "receipt_payload_sha256": bootstrap_binding[
            "receipt_payload_sha256"
        ],
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
    }
    _require(
        document.get("bootstrap_binding") == expected_bootstrap_binding,
        "IBR1 numeric evidence binds a different bootstrap assembly",
    )
    _require(
        document.get("source_binding") == bootstrap["source_binding"],
        "IBR1 numeric evidence source binding drifted",
    )
    context = _validate_cal_context(document.get("cal_context"))
    _require(
        context["checkpoint_init_sha256"] == _sealed_init_sha(bootstrap),
        "IBR1 numeric evidence init SHA differs from sealed F2 update-0 evidence",
    )
    persistence = document.get("zero_init_persistence")
    post_decode = document.get("post_decode_range")
    reconstruction = document.get("realized_delta_reconstruction")
    prev_free = document.get("prev_free_observation_graph")
    auxiliary = document.get("auxiliary_reachability")
    _require(
        isinstance(persistence, Mapping)
        and persistence.get("checked_rows") == CAL_ROWS
        and persistence.get("checked_cells") == CAL_CONTROLLED_CELLS
        and persistence.get("per_row_shape") == CAL_CONTROLLED_SHAPE
        and persistence.get("failures") == 0,
        "IBR1 numeric zero-init persistence failed",
    )
    _require(
        isinstance(post_decode, Mapping)
        and post_decode.get("checked_rows") == CAL_ROWS
        and post_decode.get("checked_cells") == CAL_CONTROLLED_CELLS
        and post_decode.get("per_row_shape") == CAL_CONTROLLED_SHAPE
        and post_decode.get("violations") == 0,
        "IBR1 numeric post-decode range check failed",
    )
    abs_max = post_decode.get("abs_max")
    _require(
        not isinstance(abs_max, bool)
        and isinstance(abs_max, (int, float))
        and math.isfinite(float(abs_max))
        and 0.0 <= float(abs_max) <= 1.0,
        "IBR1 numeric post-decode abs_max is invalid",
    )
    _require(
        isinstance(reconstruction, Mapping)
        and reconstruction.get("checked_rows") == CAL_ROWS
        and reconstruction.get("checked_cells") == CAL_CONTROLLED_CELLS
        and reconstruction.get("per_row_shape") == CAL_CONTROLLED_SHAPE
        and reconstruction.get("failures") == 0,
        "IBR1 numeric realized-delta reconstruction failed",
    )
    error_max = reconstruction.get("error_max")
    _require(
        not isinstance(error_max, bool)
        and isinstance(error_max, (int, float))
        and math.isfinite(float(error_max))
        and 0.0 <= float(error_max) <= 1e-6,
        "IBR1 numeric reconstruction error exceeds 1e-6",
    )
    _require(
        isinstance(prev_free, Mapping)
        and prev_free.get("checked_rows") == CAL_ROWS
        and prev_free.get("failures") == 0,
        "IBR1 numeric prev-free observation audit failed",
    )
    _require(
        isinstance(auxiliary, Mapping)
        and auxiliary.get("checked_rows") == CAL_ROWS
        and auxiliary.get("failures") == 0,
        "IBR1 numeric auxiliary reachability audit failed",
    )
    raw_aux_medians = auxiliary.get("per_aux_grad_norm_median")
    _require(
        isinstance(raw_aux_medians, Mapping),
        "IBR1 numeric auxiliary medians are missing",
    )
    aux_medians = {
        name: _finite_positive(
            raw_aux_medians.get(name), f"IBR1 numeric median {name}"
        )
        for name in FROZEN_AUX_COEFFICIENTS
    }
    transcript = _validate_callback_transcript(
        document.get("callback_transcript"), bootstrap=bootstrap
    )
    _require(
        transcript["per_aux_grad_norm_median"] == aux_medians,
        "IBR1 numeric auxiliary medians differ from callback transcript",
    )
    _require(
        float(abs_max) == transcript["post_decode_abs_max"],
        "IBR1 numeric post-decode aggregate differs from callback transcript",
    )
    _require(
        float(error_max) == transcript["reconstruction_error_max"],
        "IBR1 numeric reconstruction aggregate differs from callback transcript",
    )
    proposal = _validate_lambda_values(
        document.get("lambda_proposal"), "IBR1 numeric lambda proposal"
    )
    _require(
        _derive_f2_lambda(transcript["per_aux_grad_norm_median"]) == proposal,
        "IBR1 numeric proposal differs from callback transcript medians",
    )
    _require(
        document.get("proposal_role")
        == "identity_no_drift_audit_not_coefficient_selection",
        "IBR1 numeric proposal role drifted",
    )
    binding = {
        "filename": path.name,
        "sha256": _sha256_file(path, "IBR1 CAL numeric evidence"),
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": CAL_NUMERIC_EVIDENCE_CLASS,
    }
    derived = {
        "cal_context": context,
        "zero_init_persistence": dict(persistence),
        "post_decode_range": dict(post_decode),
        "realized_delta_reconstruction": dict(reconstruction),
        "prev_free_observation_graph": dict(prev_free),
        "auxiliary_reachability": dict(auxiliary),
        "per_aux_grad_norm_median": aux_medians,
        "lambda_proposal": proposal,
        "row_callback_count": CAL_ROWS,
        "callback_transcript": transcript,
    }
    return document, binding, derived


def verify_cal_numeric_evidence(
    project_root: str | Path,
    numeric_evidence_path: str | Path,
    *,
    bootstrap_receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    bootstrap_path = Path(bootstrap_receipt_path).expanduser().resolve()
    bootstrap, bootstrap_binding = _live_bootstrap_snapshot(root, bootstrap_path)
    document, _binding, _derived = _verify_ibr1_numeric_evidence(
        root,
        Path(numeric_evidence_path).expanduser().resolve(),
        bootstrap=bootstrap,
        bootstrap_binding=bootstrap_binding,
    )
    after, after_binding = _live_bootstrap_snapshot(root, bootstrap_path)
    _require(
        bootstrap == after and bootstrap_binding == after_binding,
        "live bootstrap authority drifted while verifying numeric evidence",
    )
    return document


def _derived_core_checks(
    bootstrap: Mapping[str, Any],
    raw: Mapping[str, Any],
    numeric: Mapping[str, Any],
) -> dict[str, Any]:
    asset = bootstrap["asset_binding"]["observation"]
    return {
        "f2_step0_parity": {
            "passed": True,
            "checked_rows": raw["step0"]["checked_rows"],
            "failures": raw["step0"]["failures"],
        },
        "f2_prev_free_graph": {
            "passed": True,
            "checked_rows": raw["prev_free"]["checked_rows"],
            "failures": raw["prev_free"]["failures"],
        },
        "f2_ap2_zero_init": {
            "passed": True,
            "checked_rows": raw["ap2_zero_init"]["checked_rows"],
            "violations": raw["ap2_zero_init"]["violations"],
            "track_grad_norm_max": raw["ap2_zero_init"]["track_grad_norm_max"],
        },
        "f2_auxiliary_medians": {
            "passed": True,
            "per_aux_grad_norm_median": raw["per_aux_grad_norm_median"],
        },
        "ibr1_zero_init_persistence": {
            "passed": True,
            **numeric["zero_init_persistence"],
        },
        "ibr1_post_decode_range": {
            "passed": True,
            **numeric["post_decode_range"],
        },
        "ibr1_realized_delta_reconstruction": {
            "passed": True,
            **numeric["realized_delta_reconstruction"],
        },
        "ibr1_prev_free_observation_graph": {
            "passed": True,
            **numeric["prev_free_observation_graph"],
        },
        "ibr1_auxiliary_reachability": {
            "passed": True,
            **numeric["auxiliary_reachability"],
        },
        "authority_bindings": {
            "passed": True,
            "bootstrap_authority_payload_sha256": _bootstrap_authority_payloads(
                bootstrap
            ),
            "token_ledger_sha256": asset["token_ledger_sha256"],
            "token_ledger_file_count": asset["token_ledger_file_count"],
            "cuda_reproducibility": raw["cal_context"]["cuda_reproducibility"],
        },
    }


def _core_document_from_evidence(
    *,
    bootstrap: Mapping[str, Any],
    bootstrap_binding: Mapping[str, Any],
    raw_binding: Mapping[str, Any],
    raw_derived: Mapping[str, Any],
    numeric_binding: Mapping[str, Any],
    numeric_derived: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        raw_derived["cal_context"] == numeric_derived["cal_context"],
        "raw F2 and IBR1 numeric CAL contexts differ",
    )
    _require(
        raw_derived["per_aux_grad_norm_median"]
        == numeric_derived["per_aux_grad_norm_median"],
        "raw F2 and IBR1 numeric auxiliary medians differ",
    )
    _require(
        raw_derived["lambda_proposal"] == numeric_derived["lambda_proposal"],
        "raw F2 and IBR1 numeric lambda proposals differ",
    )
    transcript_resets = numeric_derived["callback_transcript"][
        "reset_original_indices"
    ]
    raw_static_reset = raw_derived["static_reset_receipt"]
    _require(
        len(transcript_resets) == raw_static_reset["observed"]
        and canonical_json_sha256(sorted(transcript_resets))
        == raw_static_reset["original_indices_sha256"],
        "callback transcript reset reasons differ from raw F2 static resets",
    )
    proposal = _validate_lambda_values(
        raw_derived["lambda_proposal"], "mechanically derived CAL proposal"
    )
    document = {
        "schema_version": 1,
        "analysis_class": CAL_CORE_RECEIPT_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "support": CAL_SUPPORT,
        "rows": CAL_ROWS,
        "optimizer_updates": 0,
        "bootstrap_assembly_receipt": dict(bootstrap_binding),
        "live_bootstrap_authority": {
            "verified_before_and_after": True,
            "receipt_payload_sha256": bootstrap["receipt_payload_sha256"],
            "source_binding_payload_sha256": bootstrap["source_binding"][
                "receipt_payload_sha256"
            ],
            "authority_chain_payload_sha256": bootstrap["authority_chain"][
                "receipt_payload_sha256"
            ],
        },
        "raw_f2_kernel_receipt_binding": dict(raw_binding),
        "ibr1_numeric_evidence_binding": dict(numeric_binding),
        "callback_transcript_binding": {
            "numeric_receipt_payload_sha256": numeric_binding[
                "receipt_payload_sha256"
            ],
            "analysis_class": CAL_CALLBACK_TRANSCRIPT_CLASS,
            "rows": CAL_ROWS,
            "records_sha256": numeric_derived["callback_transcript"][
                "records_sha256"
            ],
            "final_sha256": numeric_derived["callback_transcript"][
                "final_sha256"
            ],
        },
        "cal_context": raw_derived["cal_context"],
        "checks": _derived_core_checks(bootstrap, raw_derived, numeric_derived),
        "gradient_calibration": {
            "formula": (
                "lambda_i=round_sig3(min(0.5*min_j(median_j)/median_i,1.0))"
            ),
            "per_aux_grad_norm_median": raw_derived[
                "per_aux_grad_norm_median"
            ],
            "lambda_proposal": proposal,
        },
        "lambda_proposal": proposal,
        "proposal_role": "identity_no_drift_audit_not_coefficient_selection",
        "passed": True,
        "formal_training_authorized": False,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    return _with_payload_self_hash(document)


def build_cal_core_receipt(
    project_root: str | Path,
    *,
    bootstrap_receipt_path: str | Path,
    raw_f2_kernel_receipt_path: str | Path,
    numeric_evidence_receipt_path: str | Path,
) -> dict[str, Any]:
    """Derive IBR1 CAL authority only from two bound numeric artifacts."""

    root = Path(project_root).expanduser().resolve()
    bootstrap_path = Path(bootstrap_receipt_path).expanduser().resolve()
    before, before_binding = _live_bootstrap_snapshot(root, bootstrap_path)
    _raw, raw_binding, raw_derived = _verify_f2_raw_cal_receipt(
        root,
        Path(raw_f2_kernel_receipt_path).expanduser().resolve(),
        bootstrap=before,
        bootstrap_binding=before_binding,
    )
    _numeric, numeric_binding, numeric_derived = _verify_ibr1_numeric_evidence(
        root,
        Path(numeric_evidence_receipt_path).expanduser().resolve(),
        bootstrap=before,
        bootstrap_binding=before_binding,
    )
    after, after_binding = _live_bootstrap_snapshot(root, bootstrap_path)
    _require(
        before == after and before_binding == after_binding,
        "live bootstrap authority/source drifted during CAL core construction",
    )
    return _core_document_from_evidence(
        bootstrap=before,
        bootstrap_binding=before_binding,
        raw_binding=raw_binding,
        raw_derived=raw_derived,
        numeric_binding=numeric_binding,
        numeric_derived=numeric_derived,
    )


def verify_cal_core_receipt(
    project_root: str | Path,
    core_receipt_path: str | Path,
    *,
    expected_bootstrap_receipt_path: str | Path | None = None,
    raw_f2_kernel_receipt_path: str | Path | None = None,
    numeric_evidence_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(core_receipt_path).expanduser().resolve()
    _root_relative(root, path, "CAL core receipt")
    document = _load_canonical_receipt(path, "CAL core receipt")
    _require(
        document.get("analysis_class") == CAL_CORE_RECEIPT_CLASS
        and document.get("family_id") == IBR1_FAMILY_ID
        and document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK
        and document.get("support") == CAL_SUPPORT
        and document.get("rows") == CAL_ROWS
        and document.get("optimizer_updates") == 0
        and document.get("passed") is True,
        "CAL core identity/zero-update contract drifted",
    )
    _sealed(document, "CAL core receipt")
    _formal_forbidden(document, "CAL core receipt")
    bootstrap_binding = document.get("bootstrap_assembly_receipt")
    _require(isinstance(bootstrap_binding, Mapping), "CAL core bootstrap binding malformed")
    bootstrap_path = _resolve_bound_path(
        root, bootstrap_binding.get("path"), "CAL core bootstrap assembly"
    )
    if expected_bootstrap_receipt_path is not None:
        expected = Path(expected_bootstrap_receipt_path).expanduser().resolve()
        _require(bootstrap_path == expected, "CAL core binds a different bootstrap")
    before, before_binding = _live_bootstrap_snapshot(root, bootstrap_path)
    _require(
        dict(bootstrap_binding) == before_binding,
        "CAL core bootstrap binding differs from live authority",
    )
    raw_binding = document.get("raw_f2_kernel_receipt_binding")
    numeric_binding = document.get("ibr1_numeric_evidence_binding")
    _require(
        isinstance(raw_binding, Mapping) and isinstance(numeric_binding, Mapping),
        "CAL core artifact bindings are malformed",
    )
    raw_filename = _portable_filename(raw_binding.get("filename"), "raw F2 filename")
    numeric_filename = _portable_filename(
        numeric_binding.get("filename"), "IBR1 numeric filename"
    )
    raw_path = (
        path.parent / raw_filename
        if raw_f2_kernel_receipt_path is None
        else Path(raw_f2_kernel_receipt_path).expanduser().resolve()
    )
    numeric_path = (
        path.parent / numeric_filename
        if numeric_evidence_receipt_path is None
        else Path(numeric_evidence_receipt_path).expanduser().resolve()
    )
    _require(raw_path.name == raw_filename, "CAL core raw F2 filename mismatch")
    _require(numeric_path.name == numeric_filename, "CAL core numeric filename mismatch")
    _raw, observed_raw_binding, raw_derived = _verify_f2_raw_cal_receipt(
        root,
        raw_path,
        bootstrap=before,
        bootstrap_binding=before_binding,
    )
    _numeric, observed_numeric_binding, numeric_derived = _verify_ibr1_numeric_evidence(
        root,
        numeric_path,
        bootstrap=before,
        bootstrap_binding=before_binding,
    )
    expected_document = _core_document_from_evidence(
        bootstrap=before,
        bootstrap_binding=before_binding,
        raw_binding=observed_raw_binding,
        raw_derived=raw_derived,
        numeric_binding=observed_numeric_binding,
        numeric_derived=numeric_derived,
    )
    after, after_binding = _live_bootstrap_snapshot(root, bootstrap_path)
    _require(
        before == after and before_binding == after_binding,
        "live bootstrap authority/source drifted during CAL core verification",
    )
    transcript_binding = document.get("callback_transcript_binding")
    _require(
        transcript_binding
        == expected_document.get("callback_transcript_binding"),
        "CAL core callback-transcript binding drifted",
    )
    _require(document == expected_document, "CAL core is not mechanically derived")
    return document


def build_cal_envelope(
    project_root: str | Path,
    *,
    core_receipt_path: str | Path,
    bootstrap_receipt_path: str | Path,
) -> dict[str, Any]:
    """Bind one canonical IBR1 CAL core into an IBR1 authority envelope."""

    root = Path(project_root).expanduser().resolve()
    core_path = Path(core_receipt_path).expanduser().resolve()
    bootstrap_path = Path(bootstrap_receipt_path).expanduser().resolve()
    core = verify_cal_core_receipt(
        root,
        core_path,
        expected_bootstrap_receipt_path=bootstrap_path,
    )
    _bootstrap, bootstrap_binding = _bootstrap_binding(root, bootstrap_path)
    _require(
        core_path.name == PurePosixPath(core_path.name).as_posix(),
        "CAL core filename is not portable",
    )
    document = {
        "schema_version": 1,
        "analysis_class": CAL_ENVELOPE_RECEIPT_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "support": CAL_SUPPORT,
        "rows": CAL_ROWS,
        "optimizer_updates": 0,
        "bootstrap_assembly_receipt": bootstrap_binding,
        "core_receipt_binding": {
            "filename": core_path.name,
            "sha256": _sha256_file(core_path, "CAL core receipt"),
            "receipt_payload_sha256": core["receipt_payload_sha256"],
            "analysis_class": CAL_CORE_RECEIPT_CLASS,
        },
        "cal_context": core["cal_context"],
        "lambda_proposal": core["lambda_proposal"],
        "passed": True,
        "formal_training_authorized": False,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    return _with_payload_self_hash(document)


def verify_cal_envelope(
    project_root: str | Path,
    envelope_receipt_path: str | Path,
    *,
    core_receipt_path: str | Path | None = None,
    expected_bootstrap_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    envelope_path = Path(envelope_receipt_path).expanduser().resolve()
    _root_relative(root, envelope_path, "CAL envelope")
    envelope = _load_canonical_receipt(envelope_path, "CAL envelope")
    _require(
        envelope.get("analysis_class") == CAL_ENVELOPE_RECEIPT_CLASS
        and envelope.get("family_id") == IBR1_FAMILY_ID
        and envelope.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK
        and envelope.get("support") == CAL_SUPPORT
        and envelope.get("rows") == CAL_ROWS
        and envelope.get("optimizer_updates") == 0
        and envelope.get("passed") is True,
        "CAL envelope identity/zero-update contract drifted",
    )
    _sealed(envelope, "CAL envelope")
    _formal_forbidden(envelope, "CAL envelope")
    proposal = _validate_lambda_values(
        envelope.get("lambda_proposal"), "CAL envelope proposal"
    )
    context = _validate_cal_context(envelope.get("cal_context"))
    binding = envelope.get("core_receipt_binding")
    _require(isinstance(binding, Mapping), "CAL envelope core binding malformed")
    filename = binding.get("filename")
    _require(
        isinstance(filename, str)
        and bool(filename)
        and Path(filename).name == filename
        and PurePosixPath(filename).as_posix() == filename,
        "CAL envelope core filename is not a plain portable filename",
    )
    inferred_core = envelope_path.parent / filename
    core_path = (
        inferred_core
        if core_receipt_path is None
        else Path(core_receipt_path).expanduser().resolve()
    )
    _require(core_path.name == filename, "CAL envelope/core filename mismatch")
    core = verify_cal_core_receipt(
        root,
        core_path,
        expected_bootstrap_receipt_path=expected_bootstrap_receipt_path,
    )
    _require(
        binding.get("analysis_class") == CAL_CORE_RECEIPT_CLASS
        and binding.get("sha256") == _sha256_file(core_path, "CAL core receipt")
        and binding.get("receipt_payload_sha256")
        == core.get("receipt_payload_sha256"),
        "CAL envelope core receipt binding drifted",
    )
    _require(
        core.get("lambda_proposal") == proposal
        and core.get("cal_context") == context,
        "CAL envelope and core carry different proposal/context",
    )
    bootstrap_binding = envelope.get("bootstrap_assembly_receipt")
    _require(isinstance(bootstrap_binding, Mapping), "CAL envelope bootstrap malformed")
    bootstrap_path = _resolve_bound_path(
        root, bootstrap_binding.get("path"), "CAL envelope bootstrap"
    )
    if expected_bootstrap_receipt_path is not None:
        expected = Path(expected_bootstrap_receipt_path).expanduser().resolve()
        _require(bootstrap_path == expected, "CAL envelope binds a different bootstrap")
    _bootstrap, observed_bootstrap = _bootstrap_binding(root, bootstrap_path)
    _require(
        dict(bootstrap_binding) == observed_bootstrap,
        "CAL envelope bootstrap binding drifted",
    )
    core_bootstrap = core.get("bootstrap_assembly_receipt")
    _require(
        isinstance(core_bootstrap, Mapping)
        and dict(core_bootstrap) == dict(bootstrap_binding),
        "CAL envelope and core do not close over the same bootstrap binding",
    )
    # Re-run the core verifier with the envelope's resolved bootstrap path;
    # this closes the optional-argument hole where a valid core could be bound
    # to one bootstrap while the envelope named another.
    verify_cal_core_receipt(
        root,
        core_path,
        expected_bootstrap_receipt_path=bootstrap_path,
    )
    return envelope


def _process_token(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= 512,
        f"{label} must be a nonempty bounded string",
    )
    return value


def _raw_cal_artifact_binding(
    root: Path, path: Path, document: Mapping[str, Any]
) -> dict[str, str]:
    _root_relative(root, path, "raw CAL artifact")
    return {
        "filename": path.name,
        "sha256": _sha256_file(path, "raw CAL artifact"),
        "canonical_payload_sha256": canonical_json_sha256(document),
        "analysis_class": str(document["analysis_class"]),
    }


def _witness_receipt_artifact_binding(
    root: Path, path: Path, document: Mapping[str, Any]
) -> dict[str, str]:
    _root_relative(root, path, "CAL witness artifact")
    return {
        "filename": path.name,
        "sha256": _sha256_file(path, "CAL witness artifact"),
        "receipt_payload_sha256": str(document["receipt_payload_sha256"]),
        "analysis_class": str(document["analysis_class"]),
    }


def _expected_cal_production_bindings(
    bootstrap: Mapping[str, Any],
    *,
    actual_context: Mapping[str, Any],
) -> dict[str, Any]:
    source = bootstrap["source_binding"]
    ibr1_sources = source["ibr1_source_sha256"]
    f2_sources = source["inherited_f2_source_sha256"]
    return {
        "subordinate_kernel": {
            "callable": "f2_experiment.assembly.run_cal_audit",
            "source_path": "f2_experiment/assembly.py",
            "source_sha256": f2_sources["f2_experiment/assembly.py"],
        },
        "f2_model_kernel": {
            "class": "f2_experiment.assembly_model.CalRowAuditor",
            "source_path": "f2_experiment/assembly_model.py",
            "source_sha256": f2_sources["f2_experiment/assembly_model.py"],
        },
        "row_auditor_factory": {
            "callable": (
                "ibr1_experiment.calibration_model.build_ibr1_cal_row_auditor"
            ),
            "source_path": "ibr1_experiment/calibration_model.py",
            "source_sha256": ibr1_sources[
                "ibr1_experiment/calibration_model.py"
            ],
        },
        "row_auditor": {
            "class": "ibr1_experiment.calibration_model.IBR1ModelCalRowAuditor",
            "context_callable": (
                "ibr1_experiment.calibration_model."
                "IBR1ModelCalRowAuditor.context_receipt"
            ),
        },
        "ibr1_assembly": {
            "source_path": "ibr1_experiment/assembly_model.py",
            "source_sha256": ibr1_sources["ibr1_experiment/assembly_model.py"],
        },
        "actual_context": dict(actual_context),
    }


def verify_cal_execution_witness(
    project_root: str | Path,
    execution_witness_path: str | Path,
    *,
    expected_role: Literal["main", "reproduction"],
    bootstrap_receipt_path: str | Path,
    raw_f2_kernel_receipt_path: str | Path,
    numeric_evidence_receipt_path: str | Path,
    core_receipt_path: str | Path,
    envelope_receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(execution_witness_path).expanduser().resolve()
    _root_relative(root, path, "CAL execution witness")
    document = _load_canonical_receipt(path, "CAL execution witness")
    _require(
        document.get("analysis_class") == CAL_EXECUTION_WITNESS_CLASS
        and document.get("family_id") == IBR1_FAMILY_ID
        and document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK
        and document.get("role") == expected_role,
        "CAL execution witness identity/role drifted",
    )
    _sealed(document, "CAL execution witness")
    _formal_forbidden(document, "CAL execution witness")
    identity = document.get("process_identity")
    _require(isinstance(identity, Mapping), "CAL process identity is malformed")
    _require(
        isinstance(identity.get("pid"), int)
        and not isinstance(identity.get("pid"), bool)
        and identity.get("pid") > 0,
        "CAL process identity PID is invalid",
    )
    _process_token(identity.get("process_start_token"), "process-start token")
    _process_token(identity.get("module_import_token"), "module-import token")
    orchestration = document.get("orchestration_binding")
    _require(
        isinstance(orchestration, Mapping)
        and set(orchestration)
        == {
            "analysis_class",
            "parent_challenge",
            "parent_pid",
            "child_pid",
        }
        and orchestration.get("analysis_class")
        == "ibr1_cal_worker_parent_challenge"
        and orchestration.get("child_pid") == identity.get("pid"),
        "CAL witness parent orchestration binding is malformed",
    )
    parent_challenge = orchestration.get("parent_challenge")
    parent_pid = orchestration.get("parent_pid")
    _require(
        (
            parent_challenge is None
            and parent_pid is None
        )
        or (
            isinstance(parent_challenge, str)
            and len(parent_challenge) == 64
            and all(
                character in "0123456789abcdef"
                for character in parent_challenge
            )
            and isinstance(parent_pid, int)
            and not isinstance(parent_pid, bool)
            and parent_pid > 0
            and parent_pid != identity.get("pid")
        ),
        "CAL witness parent challenge/PID binding drifted",
    )
    clock = document.get("audit_clock")
    _require(isinstance(clock, Mapping), "CAL witness audit clock is malformed")
    started = clock.get("started_ns")
    ended = clock.get("ended_ns")
    _require(
        isinstance(started, int)
        and not isinstance(started, bool)
        and isinstance(ended, int)
        and not isinstance(ended, bool)
        and 0 <= started < ended
        and clock.get("callback_count") == CAL_ROWS
        and clock.get("first_position") == 0
        and clock.get("last_position") == CAL_ROWS - 1
        and clock.get("ordered_positions_sha256")
        == canonical_json_sha256(list(range(CAL_ROWS))),
        "CAL witness does not prove one ordered 512-row audit session",
    )
    bootstrap_path = Path(bootstrap_receipt_path).expanduser().resolve()
    bootstrap, bootstrap_binding = _live_bootstrap_snapshot(root, bootstrap_path)
    expected_bootstrap = {
        "filename": bootstrap_path.name,
        "sha256": bootstrap_binding["sha256"],
        "receipt_payload_sha256": bootstrap_binding[
            "receipt_payload_sha256"
        ],
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
    }
    _require(
        document.get("bootstrap_binding") == expected_bootstrap,
        "CAL witness bootstrap binding drifted",
    )
    calibration_source = "ibr1_experiment/calibration.py"
    runner = document.get("runner_binding")
    _require(
        isinstance(runner, Mapping)
        and runner.get("entrypoint") == "run_ibr1_cal_audit_once"
        and runner.get("source_path") == calibration_source
        and runner.get("source_sha256")
        == bootstrap["source_binding"]["ibr1_source_sha256"].get(
            calibration_source
        ),
        "CAL witness is not bound to the frozen calibration runner",
    )
    raw_path = Path(raw_f2_kernel_receipt_path).expanduser().resolve()
    numeric_path = Path(numeric_evidence_receipt_path).expanduser().resolve()
    core_path = Path(core_receipt_path).expanduser().resolve()
    envelope_path = Path(envelope_receipt_path).expanduser().resolve()
    raw, _raw_binding, raw_derived = _verify_f2_raw_cal_receipt(
        root,
        raw_path,
        bootstrap=bootstrap,
        bootstrap_binding=bootstrap_binding,
    )
    numeric, _numeric_binding, numeric_derived = _verify_ibr1_numeric_evidence(
        root,
        numeric_path,
        bootstrap=bootstrap,
        bootstrap_binding=bootstrap_binding,
    )
    core = verify_cal_core_receipt(
        root,
        core_path,
        expected_bootstrap_receipt_path=bootstrap_path,
        raw_f2_kernel_receipt_path=raw_path,
        numeric_evidence_receipt_path=numeric_path,
    )
    envelope = verify_cal_envelope(
        root,
        envelope_path,
        core_receipt_path=core_path,
        expected_bootstrap_receipt_path=bootstrap_path,
    )
    _require(
        raw_derived["cal_context"] == numeric_derived["cal_context"],
        "CAL witness raw/numeric contexts differ",
    )
    _require(
        document.get("production_bindings")
        == _expected_cal_production_bindings(
            bootstrap, actual_context=raw_derived["cal_context"]
        ),
        "CAL witness production kernel/model bindings drifted",
    )
    transcript = numeric_derived["callback_transcript"]
    expected_transcript_binding = {
        "container_analysis_class": CAL_NUMERIC_EVIDENCE_CLASS,
        "container_receipt_payload_sha256": numeric[
            "receipt_payload_sha256"
        ],
        "analysis_class": CAL_CALLBACK_TRANSCRIPT_CLASS,
        "rows": CAL_ROWS,
        "records_sha256": transcript["records_sha256"],
        "final_sha256": transcript["final_sha256"],
    }
    _require(
        document.get("callback_transcript_binding")
        == expected_transcript_binding,
        "CAL witness callback-transcript binding drifted",
    )
    artifacts = document.get("artifacts")
    _require(
        isinstance(artifacts, Mapping)
        and set(artifacts)
        == {"raw_f2_kernel", "numeric_evidence", "core", "envelope"}
        and artifacts.get("raw_f2_kernel")
        == _raw_cal_artifact_binding(root, raw_path, raw)
        and artifacts.get("numeric_evidence")
        == _witness_receipt_artifact_binding(root, numeric_path, numeric)
        and artifacts.get("core")
        == _witness_receipt_artifact_binding(root, core_path, core)
        and artifacts.get("envelope")
        == _witness_receipt_artifact_binding(root, envelope_path, envelope),
        "CAL execution witness artifact binding drifted",
    )
    return document


def _cal_artifact_binding(
    root: Path, path: Path, document: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "path": _root_relative(root, path, "CAL artifact"),
        "sha256": _sha256_file(path, "CAL artifact"),
        "receipt_payload_sha256": str(document["receipt_payload_sha256"]),
        "analysis_class": str(document["analysis_class"]),
    }


def verify_cal_pair(
    project_root: str | Path,
    *,
    main_raw_f2_kernel_path: str | Path,
    reproduction_raw_f2_kernel_path: str | Path,
    main_numeric_evidence_path: str | Path,
    reproduction_numeric_evidence_path: str | Path,
    main_core_path: str | Path,
    reproduction_core_path: str | Path,
    main_envelope_path: str | Path,
    reproduction_envelope_path: str | Path,
    main_execution_witness_path: str | Path,
    reproduction_execution_witness_path: str | Path,
    bootstrap_receipt_path: str | Path,
) -> dict[str, Any]:
    """Forensically compare two CAL artifact trees.

    Files can establish byte identity and internally recorded process fields,
    but they cannot prove that two operating-system processes were live and
    observed by the caller.  Consequently this API is permanently
    non-authoritative and cannot advance the lambda freeze.
    """

    root = Path(project_root).expanduser().resolve()
    main_raw = Path(main_raw_f2_kernel_path).expanduser().resolve()
    repro_raw = Path(reproduction_raw_f2_kernel_path).expanduser().resolve()
    main_numeric = Path(main_numeric_evidence_path).expanduser().resolve()
    repro_numeric = Path(reproduction_numeric_evidence_path).expanduser().resolve()
    main_core = Path(main_core_path).expanduser().resolve()
    repro_core = Path(reproduction_core_path).expanduser().resolve()
    main_envelope = Path(main_envelope_path).expanduser().resolve()
    repro_envelope = Path(reproduction_envelope_path).expanduser().resolve()
    main_witness = Path(main_execution_witness_path).expanduser().resolve()
    repro_witness = Path(reproduction_execution_witness_path).expanduser().resolve()
    bootstrap = Path(bootstrap_receipt_path).expanduser().resolve()
    for label, path in (
        ("main raw F2 CAL", main_raw),
        ("reproduction raw F2 CAL", repro_raw),
        ("main IBR1 numeric evidence", main_numeric),
        ("reproduction IBR1 numeric evidence", repro_numeric),
        ("main CAL core", main_core),
        ("reproduction CAL core", repro_core),
        ("main CAL envelope", main_envelope),
        ("reproduction CAL envelope", repro_envelope),
        ("main CAL execution witness", main_witness),
        ("reproduction CAL execution witness", repro_witness),
        ("CAL bootstrap", bootstrap),
    ):
        _root_relative(root, path, label)
    for label, main_path, repro_path in (
        ("raw F2 CAL", main_raw, repro_raw),
        ("IBR1 numeric evidence", main_numeric, repro_numeric),
        ("CAL core", main_core, repro_core),
        ("CAL envelope", main_envelope, repro_envelope),
        ("CAL execution witness", main_witness, repro_witness),
    ):
        _require(main_path != repro_path, f"{label} paths must be distinct")
        _require(
            main_path.name == repro_path.name,
            f"{label} pair must use the same portable filename",
        )
    bootstrap_document, bootstrap_binding = _live_bootstrap_snapshot(root, bootstrap)
    main_raw_document, _main_raw_binding, main_raw_derived = _verify_f2_raw_cal_receipt(
        root,
        main_raw,
        bootstrap=bootstrap_document,
        bootstrap_binding=bootstrap_binding,
    )
    repro_raw_document, _repro_raw_binding, repro_raw_derived = _verify_f2_raw_cal_receipt(
        root,
        repro_raw,
        bootstrap=bootstrap_document,
        bootstrap_binding=bootstrap_binding,
    )
    _require(
        main_raw.read_bytes() == repro_raw.read_bytes()
        and main_raw_document == repro_raw_document
        and main_raw_derived == repro_raw_derived,
        "CAL raw F2 main/reproduction artifacts are not byte-identical",
    )
    main_numeric_document, _main_numeric_binding, main_numeric_derived = (
        _verify_ibr1_numeric_evidence(
            root,
            main_numeric,
            bootstrap=bootstrap_document,
            bootstrap_binding=bootstrap_binding,
        )
    )
    repro_numeric_document, _repro_numeric_binding, repro_numeric_derived = (
        _verify_ibr1_numeric_evidence(
            root,
            repro_numeric,
            bootstrap=bootstrap_document,
            bootstrap_binding=bootstrap_binding,
        )
    )
    _require(
        main_numeric.read_bytes() == repro_numeric.read_bytes()
        and main_numeric_document == repro_numeric_document
        and main_numeric_derived == repro_numeric_derived,
        "IBR1 numeric main/reproduction artifacts are not byte-identical",
    )
    main_core_document = verify_cal_core_receipt(
        root,
        main_core,
        expected_bootstrap_receipt_path=bootstrap,
        raw_f2_kernel_receipt_path=main_raw,
        numeric_evidence_receipt_path=main_numeric,
    )
    repro_core_document = verify_cal_core_receipt(
        root,
        repro_core,
        expected_bootstrap_receipt_path=bootstrap,
        raw_f2_kernel_receipt_path=repro_raw,
        numeric_evidence_receipt_path=repro_numeric,
    )
    _require(
        main_core.read_bytes() == repro_core.read_bytes(),
        "CAL main/reproduction core receipts are not byte-identical",
    )
    main_envelope_document = verify_cal_envelope(
        root,
        main_envelope,
        core_receipt_path=main_core,
        expected_bootstrap_receipt_path=bootstrap,
    )
    repro_envelope_document = verify_cal_envelope(
        root,
        repro_envelope,
        core_receipt_path=repro_core,
        expected_bootstrap_receipt_path=bootstrap,
    )
    _require(
        main_envelope.read_bytes() == repro_envelope.read_bytes(),
        "CAL main/reproduction envelopes are not byte-identical",
    )
    _require(
        main_core_document == repro_core_document
        and main_envelope_document == repro_envelope_document,
        "CAL pair parsed identities differ despite byte comparison",
    )
    main_witness_document = verify_cal_execution_witness(
        root,
        main_witness,
        expected_role="main",
        bootstrap_receipt_path=bootstrap,
        raw_f2_kernel_receipt_path=main_raw,
        numeric_evidence_receipt_path=main_numeric,
        core_receipt_path=main_core,
        envelope_receipt_path=main_envelope,
    )
    repro_witness_document = verify_cal_execution_witness(
        root,
        repro_witness,
        expected_role="reproduction",
        bootstrap_receipt_path=bootstrap,
        raw_f2_kernel_receipt_path=repro_raw,
        numeric_evidence_receipt_path=repro_numeric,
        core_receipt_path=repro_core,
        envelope_receipt_path=repro_envelope,
    )
    main_identity = main_witness_document["process_identity"]
    repro_identity = repro_witness_document["process_identity"]
    _require(
        main_identity.get("pid") != repro_identity.get("pid")
        and main_identity.get("process_start_token")
        != repro_identity.get("process_start_token")
        and main_identity.get("module_import_token")
        != repro_identity.get("module_import_token")
        and dict(main_identity) != dict(repro_identity),
        "CAL main/reproduction were not proven to be distinct processes",
    )
    _validate_lambda_values(
        main_core_document.get("lambda_proposal"), "CAL pair proposal"
    )
    return {
        "schema_version": 1,
        "analysis_class": CAL_PAIR_FORENSIC_CLASS,
        "bootstrap_assembly_receipt": bootstrap_binding,
        "main": {
            "raw_f2_kernel": {
                "path": _root_relative(root, main_raw, "main raw F2 CAL"),
                **_raw_cal_artifact_binding(root, main_raw, main_raw_document),
            },
            "numeric_evidence": _cal_artifact_binding(
                root, main_numeric, main_numeric_document
            ),
            "core": _cal_artifact_binding(root, main_core, main_core_document),
            "envelope": _cal_artifact_binding(
                root, main_envelope, main_envelope_document
            ),
            "execution_witness": _cal_artifact_binding(
                root, main_witness, main_witness_document
            ),
        },
        "reproduction": {
            "raw_f2_kernel": {
                "path": _root_relative(root, repro_raw, "reproduction raw F2 CAL"),
                **_raw_cal_artifact_binding(root, repro_raw, repro_raw_document),
            },
            "numeric_evidence": _cal_artifact_binding(
                root, repro_numeric, repro_numeric_document
            ),
            "core": _cal_artifact_binding(root, repro_core, repro_core_document),
            "envelope": _cal_artifact_binding(
                root, repro_envelope, repro_envelope_document
            ),
            "execution_witness": _cal_artifact_binding(
                root, repro_witness, repro_witness_document
            ),
        },
        "raw_f2_byte_identical": True,
        "numeric_evidence_byte_identical": True,
        "core_byte_identical": True,
        "envelope_byte_identical": True,
        "recorded_process_identities_differ": True,
        "distinct_processes_verified": False,
        "authority_eligible": False,
        "process_identity": {
            "main": dict(main_identity),
            "reproduction": dict(repro_identity),
        },
        "lambda_proposal": dict(FROZEN_AUX_COEFFICIENTS),
        "cal_context": main_core_document["cal_context"],
        "formal_training_authorized": False,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }


def build_lambda_adoption_freeze(
    project_root: str | Path,
    *,
    bootstrap_receipt_path: str | Path,
    main_raw_f2_kernel_path: str | Path,
    reproduction_raw_f2_kernel_path: str | Path,
    main_numeric_evidence_path: str | Path,
    reproduction_numeric_evidence_path: str | Path,
    main_core_path: str | Path,
    reproduction_core_path: str | Path,
    main_envelope_path: str | Path,
    reproduction_envelope_path: str | Path,
    main_execution_witness_path: str | Path,
    reproduction_execution_witness_path: str | Path,
) -> dict[str, Any]:
    raise IBR1AuthorityError(
        "file-only CAL paths are forensic and cannot build a lambda-adoption "
        "freeze; use the fixed live CAL pair orchestrator"
    )


def freeze_lambda_adoption_freeze(
    project_root: str | Path,
    output: str | Path,
    *,
    bootstrap_receipt_path: str | Path,
    main_raw_f2_kernel_path: str | Path,
    reproduction_raw_f2_kernel_path: str | Path,
    main_numeric_evidence_path: str | Path,
    reproduction_numeric_evidence_path: str | Path,
    main_core_path: str | Path,
    reproduction_core_path: str | Path,
    main_envelope_path: str | Path,
    reproduction_envelope_path: str | Path,
    main_execution_witness_path: str | Path,
    reproduction_execution_witness_path: str | Path,
) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    _root_relative(root, destination, "lambda-adoption freeze output")
    document = build_lambda_adoption_freeze(
        root,
        bootstrap_receipt_path=bootstrap_receipt_path,
        main_raw_f2_kernel_path=main_raw_f2_kernel_path,
        reproduction_raw_f2_kernel_path=reproduction_raw_f2_kernel_path,
        main_numeric_evidence_path=main_numeric_evidence_path,
        reproduction_numeric_evidence_path=reproduction_numeric_evidence_path,
        main_core_path=main_core_path,
        reproduction_core_path=reproduction_core_path,
        main_envelope_path=main_envelope_path,
        reproduction_envelope_path=reproduction_envelope_path,
        main_execution_witness_path=main_execution_witness_path,
        reproduction_execution_witness_path=reproduction_execution_witness_path,
    )
    file_sha = exclusive_write_json(destination, document)
    return {
        "path": str(destination),
        "sha256": file_sha,
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": LAMBDA_ADOPTION_FREEZE_CLASS,
    }


def _paths_from_freeze_evidence(
    root: Path, evidence: Mapping[str, Any]
) -> dict[str, Path]:
    bootstrap = evidence.get("bootstrap_assembly_receipt")
    main = evidence.get("cal_main")
    reproduction = evidence.get("cal_reproduction")
    _require(
        isinstance(bootstrap, Mapping)
        and isinstance(main, Mapping)
        and isinstance(reproduction, Mapping),
        "lambda-adoption freeze evidence is malformed",
    )
    main_raw = main.get("raw_f2_kernel")
    main_numeric = main.get("numeric_evidence")
    main_core = main.get("core")
    main_envelope = main.get("envelope")
    main_witness = main.get("execution_witness")
    repro_raw = reproduction.get("raw_f2_kernel")
    repro_numeric = reproduction.get("numeric_evidence")
    repro_core = reproduction.get("core")
    repro_envelope = reproduction.get("envelope")
    repro_witness = reproduction.get("execution_witness")
    _require(
        all(
            isinstance(value, Mapping)
            for value in (
                main_raw,
                main_numeric,
                main_core,
                main_envelope,
                main_witness,
                repro_raw,
                repro_numeric,
                repro_core,
                repro_envelope,
                repro_witness,
            )
        ),
        "lambda-adoption freeze CAL bindings are malformed",
    )
    return {
        "bootstrap": _resolve_bound_path(
            root, bootstrap.get("path"), "freeze bootstrap"
        ),
        "main_raw": _resolve_bound_path(
            root, main_raw.get("path"), "freeze main raw F2 CAL"
        ),
        "main_numeric": _resolve_bound_path(
            root, main_numeric.get("path"), "freeze main numeric evidence"
        ),
        "main_core": _resolve_bound_path(
            root, main_core.get("path"), "freeze main core"
        ),
        "main_envelope": _resolve_bound_path(
            root, main_envelope.get("path"), "freeze main envelope"
        ),
        "main_witness": _resolve_bound_path(
            root, main_witness.get("path"), "freeze main execution witness"
        ),
        "reproduction_raw": _resolve_bound_path(
            root, repro_raw.get("path"), "freeze reproduction raw F2 CAL"
        ),
        "reproduction_numeric": _resolve_bound_path(
            root,
            repro_numeric.get("path"),
            "freeze reproduction numeric evidence",
        ),
        "reproduction_core": _resolve_bound_path(
            root, repro_core.get("path"), "freeze reproduction core"
        ),
        "reproduction_envelope": _resolve_bound_path(
            root, repro_envelope.get("path"), "freeze reproduction envelope"
        ),
        "reproduction_witness": _resolve_bound_path(
            root,
            repro_witness.get("path"),
            "freeze reproduction execution witness",
        ),
    }


def _worker_result_artifact_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(binding)
    bound_path = normalized.pop("path", None)
    _require(isinstance(bound_path, str), "live worker artifact path is missing")
    normalized["filename"] = PurePosixPath(bound_path).name
    return normalized


def _validate_live_orchestration(
    root: Path,
    *,
    evidence: Mapping[str, Any],
    paths: Mapping[str, Path],
    observed: Mapping[str, Any],
) -> None:
    attestation = evidence.get("live_orchestration")
    _require(isinstance(attestation, Mapping), "live CAL orchestration is missing")
    challenge = attestation.get("parent_challenge")
    parent_pid = attestation.get("parent_pid")
    executable = str(OFFICIAL_PYTHON_EXECUTABLE)
    _require(
        attestation.get("schema_version") == 1
        and attestation.get("analysis_class") == CAL_LIVE_PROCESS_ATTESTATION_CLASS
        and isinstance(challenge, str)
        and len(challenge) == 64
        and all(character in "0123456789abcdef" for character in challenge)
        and isinstance(parent_pid, int)
        and not isinstance(parent_pid, bool)
        and parent_pid > 0
        and attestation.get("python_executable") == executable
        and attestation.get("worker_module") == "ibr1_experiment.cal_worker",
        "live CAL parent challenge/executable identity drifted",
    )
    bootstrap, _binding = _live_bootstrap_snapshot(root, paths["bootstrap"])
    source_sha = bootstrap["source_binding"]["ibr1_source_sha256"]
    expected_source = {
        "ibr1_experiment/cal_pair.py": source_sha.get(
            "ibr1_experiment/cal_pair.py"
        ),
        "ibr1_experiment/cal_worker.py": source_sha.get(
            "ibr1_experiment/cal_worker.py"
        ),
        "ibr1_experiment/runtime_contract.py": source_sha.get(
            "ibr1_experiment/runtime_contract.py"
        ),
    }
    _require(
        attestation.get("source_binding") == expected_source
        and all(
            isinstance(value, str) and len(value) == 64
            for value in expected_source.values()
        ),
        "live CAL orchestrator/worker source binding drifted",
    )
    workers = attestation.get("workers")
    _require(
        isinstance(workers, Mapping)
        and set(workers) == {"main", "reproduction"},
        "live CAL worker set drifted",
    )
    worker_pids: list[int] = []
    for role in ("main", "reproduction"):
        worker = workers[role]
        _require(
            isinstance(worker, Mapping)
            and set(worker)
            == {
                "role",
                "pid",
                "args",
                "exit_code",
                "stdout",
                "stdout_sha256",
                "stderr",
                "stderr_sha256",
                "result",
                "witness_orchestration_binding",
            },
            f"live {role} worker attestation is malformed",
        )
        pid = worker.get("pid")
        role_key = "main" if role == "main" else "reproduction"
        role_evidence = observed[role_key]
        cal_context = observed.get("cal_context")
        cuda_context = (
            cal_context.get("cuda_reproducibility")
            if isinstance(cal_context, Mapping)
            else None
        )
        _require(
            isinstance(cal_context, Mapping)
            and cal_context.get("device") == OFFICIAL_DEVICE
            and isinstance(cuda_context, Mapping)
            and cuda_context.get("torch_version") == OFFICIAL_TORCH_VERSION
            and cuda_context.get("cuda_runtime") == OFFICIAL_CUDA_RUNTIME,
            "live CAL torch/CUDA/device runtime identity drifted",
        )
        output_dir = paths[f"{role_key}_raw"].parent
        expected_args = [
            executable,
            "-m",
            "ibr1_experiment.cal_worker",
            "--project-root",
            str(root),
            "--role",
            role,
            "--bootstrap-receipt",
            str(paths["bootstrap"]),
            "--output-dir",
            str(output_dir),
            "--parent-challenge",
            challenge,
            "--parent-pid",
            str(parent_pid),
        ]
        expected_orchestration = {
            "analysis_class": "ibr1_cal_worker_parent_challenge",
            "parent_challenge": challenge,
            "parent_pid": parent_pid,
            "child_pid": pid,
        }
        _require(
            isinstance(pid, int)
            and not isinstance(pid, bool)
            and pid > 0
            and pid != parent_pid
            and worker.get("role") == role
            and worker.get("args") == expected_args
            and worker.get("exit_code") == 0
            and worker.get("witness_orchestration_binding")
            == expected_orchestration,
            f"live {role} worker command/challenge/PID/exit drifted",
        )
        worker_pids.append(pid)
        stdout = worker.get("stdout")
        stderr = worker.get("stderr")
        _require(
            isinstance(stdout, str)
            and isinstance(stderr, str)
            and worker.get("stdout_sha256")
            == hashlib.sha256(stdout.encode("utf-8")).hexdigest()
            and worker.get("stderr_sha256")
            == hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            f"live {role} worker captured output SHA drifted",
        )
        lines = stdout.splitlines()
        _require(
            len(lines) == 1 and bool(lines[0]),
            f"live {role} worker stdout is not one result",
        )
        try:
            stdout_result = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise IBR1AuthorityError(
                f"live {role} worker stdout is not JSON"
            ) from exc
        _require(
            isinstance(stdout_result, Mapping)
            and lines[0] == canonical_json_bytes(stdout_result).decode("utf-8")
            and stdout == lines[0] + "\n"
            and worker.get("result") == stdout_result,
            f"live {role} worker stdout/result binding drifted",
        )
        expected_calibration_result = {
            "role": role,
            "orchestration_binding": expected_orchestration,
            "raw_f2_kernel": _worker_result_artifact_binding(
                role_evidence["raw_f2_kernel"]
            ),
            "numeric_evidence": _worker_result_artifact_binding(
                role_evidence["numeric_evidence"]
            ),
            "core": _worker_result_artifact_binding(role_evidence["core"]),
            "envelope": _worker_result_artifact_binding(
                role_evidence["envelope"]
            ),
            "execution_witness": _worker_result_artifact_binding(
                role_evidence["execution_witness"]
            ),
            "formal_training_authorized": False,
        }
        expected_runtime = {
            "python_executable": executable,
            "torch_version": OFFICIAL_TORCH_VERSION,
            "cuda_runtime": OFFICIAL_CUDA_RUNTIME,
            "device": OFFICIAL_DEVICE,
        }
        _require(
            stdout_result.get("schema_version") == 1
            and stdout_result.get("analysis_class") == "ibr1_cal_worker_result"
            and stdout_result.get("role") == role
            and stdout_result.get("parent_challenge") == challenge
            and stdout_result.get("parent_pid") == parent_pid
            and stdout_result.get("child_pid") == pid
            and stdout_result.get("output_dir") == str(output_dir)
            and stdout_result.get("runtime") == expected_runtime
            and stdout_result.get("calibration_result")
            == expected_calibration_result,
            f"live {role} worker canonical result drifted",
        )
        witness_path = paths[f"{role_key}_witness"]
        witness = _load_canonical_receipt(
            witness_path, f"live {role} execution witness"
        )
        _require(
            witness.get("process_identity", {}).get("pid") == pid
            and witness.get("orchestration_binding") == expected_orchestration,
            f"live {role} witness challenge/PID drifted",
        )
    _require(
        len(set(worker_pids)) == 2,
        "live main/reproduction worker PIDs are not distinct",
    )


def verify_lambda_adoption_freeze(
    project_root: str | Path, freeze_path: str | Path
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = Path(freeze_path).expanduser().resolve()
    _root_relative(root, path, "lambda-adoption freeze")
    document = _load_canonical_receipt(path, "lambda-adoption freeze")
    _require(
        document.get("analysis_class") == LAMBDA_ADOPTION_FREEZE_CLASS
        and document.get("family_id") == IBR1_FAMILY_ID
        and document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK
        and document.get("mechanism") == "inheritance_identity_no_drift_audit"
        and document.get("generation_contract")
        == "parent_owned_live_popen_challenge_v1"
        and document.get("adoption_status") == "FROZEN_EXACT_INHERITANCE"
        and document.get("difference_action")
        == "STOP_NO_SMOKE_NO_LAMBDA_CHANGE",
        "lambda-adoption freeze identity/policy drifted",
    )
    _sealed(document, "lambda-adoption freeze")
    _formal_forbidden(document, "lambda-adoption freeze")
    frozen = _validate_lambda_values(
        document.get("frozen_values"), "lambda-adoption frozen values"
    )
    proposal = _validate_lambda_values(
        document.get("proposal"), "lambda-adoption proposal"
    )
    _require(frozen == proposal, "lambda adoption is not exact inheritance")
    evidence = document.get("evidence")
    _require(isinstance(evidence, Mapping), "lambda-adoption evidence malformed")
    _require(
        evidence.get("raw_f2_byte_identical") is True
        and evidence.get("numeric_evidence_byte_identical") is True
        and evidence.get("core_byte_identical") is True
        and evidence.get("envelope_byte_identical") is True
        and evidence.get("distinct_processes_verified") is True
        and evidence.get("authority_eligible") is True
        and isinstance(evidence.get("process_identity"), Mapping),
        "lambda-adoption reproduction contract is incomplete",
    )
    paths = _paths_from_freeze_evidence(root, evidence)
    observed = verify_cal_pair(
        root,
        main_raw_f2_kernel_path=paths["main_raw"],
        reproduction_raw_f2_kernel_path=paths["reproduction_raw"],
        main_numeric_evidence_path=paths["main_numeric"],
        reproduction_numeric_evidence_path=paths["reproduction_numeric"],
        main_core_path=paths["main_core"],
        reproduction_core_path=paths["reproduction_core"],
        main_envelope_path=paths["main_envelope"],
        reproduction_envelope_path=paths["reproduction_envelope"],
        main_execution_witness_path=paths["main_witness"],
        reproduction_execution_witness_path=paths["reproduction_witness"],
        bootstrap_receipt_path=paths["bootstrap"],
    )
    _require(
        evidence.get("bootstrap_assembly_receipt")
        == observed["bootstrap_assembly_receipt"]
        and evidence.get("cal_main") == observed["main"]
        and evidence.get("cal_reproduction") == observed["reproduction"]
        and evidence.get("process_identity") == observed["process_identity"],
        "lambda-adoption evidence bindings drifted from actual bytes",
    )
    _require(
        evidence.get("file_pair_forensic")
        == {
            "analysis_class": observed["analysis_class"],
            "authority_eligible": False,
            "distinct_processes_verified": False,
            "recorded_process_identities_differ": observed[
                "recorded_process_identities_differ"
            ],
        }
        and observed.get("authority_eligible") is False
        and observed.get("distinct_processes_verified") is False,
        "lambda-adoption file-pair evidence was not kept forensic",
    )
    _validate_live_orchestration(
        root,
        evidence=evidence,
        paths=paths,
        observed=observed,
    )
    return document


__all__ = [
    "ASSEMBLY_PHASE_BOOTSTRAP",
    "ASSEMBLY_PHASE_FINAL",
    "ASSEMBLY_RECEIPT_CLASS",
    "ASSET_BINDING_CLASS",
    "AUTHORITY_CHAIN_CLASS",
    "CAL_CORE_RECEIPT_CLASS",
    "CAL_CALLBACK_TRANSCRIPT_CLASS",
    "CAL_EXECUTION_BINDING_CLASS",
    "CAL_EXECUTION_BINDING_RECORDS_SHA256",
    "CAL_ENVELOPE_RECEIPT_CLASS",
    "CAL_EXECUTION_WITNESS_CLASS",
    "CAL_EXECUTION_RECEIPT_CLASS",
    "CAL_NUMERIC_EVIDENCE_CLASS",
    "CAL_LIVE_PROCESS_ATTESTATION_CLASS",
    "CAL_PAIR_FORENSIC_CLASS",
    "CAL_REQUIRED_CHECKS",
    "F2_NEGATIVE_EVIDENCE_CLASS",
    "F2_NEGATIVE_SEAL_SHA256",
    "FROZEN_AUX_COEFFICIENTS",
    "IBR1AuthorityError",
    "LAMBDA_ADOPTION_FREEZE_CLASS",
    "SOURCE_BINDING_CLASS",
    "SUPPORT_BINDING_CLASS",
    "TEST_BINDING_CLASS",
    "build_assembly_receipt",
    "build_asset_binding",
    "build_cal_core_receipt",
    "build_cal_envelope",
    "build_lambda_adoption_freeze",
    "build_source_binding",
    "build_support_binding",
    "build_test_binding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "exclusive_write_json",
    "freeze_assembly_receipt",
    "freeze_lambda_adoption_freeze",
    "verify_assembly_receipt",
    "verify_authority_chain",
    "verify_cal_core_receipt",
    "verify_cal_envelope",
    "verify_cal_execution_witness",
    "verify_cal_numeric_evidence",
    "verify_cal_pair",
    "verify_f2_negative_evidence",
    "verify_lambda_adoption_freeze",
    "verify_primary_authority",
]
