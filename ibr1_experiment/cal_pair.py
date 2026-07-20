"""Parent-owned live CAL pair orchestration and lambda-freeze issuance.

File paths alone are never sufficient authority.  This module starts the two
official workers itself, observes their live ``Popen`` objects, binds both to
one random parent challenge, and consumes a non-serializable proof in the same
call that writes the lambda-adoption freeze.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any
from weakref import WeakKeyDictionary

from . import authority
from .runtime_contract import (
    OFFICIAL_CUDA_RUNTIME,
    OFFICIAL_DEVICE,
    OFFICIAL_PYTHON_EXECUTABLE,
    OFFICIAL_TORCH_VERSION,
    require_official_python,
)


LIVE_ATTESTATION_CLASS = authority.CAL_LIVE_PROCESS_ATTESTATION_CLASS
WORKER_RESULT_CLASS = "ibr1_cal_worker_result"
WORKER_MODULE = "ibr1_experiment.cal_worker"
TEST_ONLY_PAIR_CLASS = "ibr1_cal_pair_test_only_evidence"

RAW_FILENAME = "cal_audit_receipt_v1.json"
NUMERIC_FILENAME = "ibr1_cal_numeric_evidence.json"
CORE_FILENAME = "ibr1_cal_core_receipt.json"
ENVELOPE_FILENAME = "ibr1_cal_envelope.json"
WITNESS_FILENAME = "ibr1_cal_execution_witness.json"
ROLE_FILENAMES = {
    "raw": RAW_FILENAME,
    "numeric": NUMERIC_FILENAME,
    "core": CORE_FILENAME,
    "envelope": ENVELOPE_FILENAME,
    "witness": WITNESS_FILENAME,
}

_LIVE_PROOF_SECRET = object()
_FINAL_CAPABILITY_SECRET = object()
_AUTHORITATIVE_RUN_SECRET = object()
_OFFICIAL_POPEN = subprocess.Popen


@dataclass(slots=True)
class _FinalCapabilityState:
    root: Path
    freeze_path: Path
    final_path: Path
    final_binding_bytes: bytes
    attestation_bytes: bytes
    consumed: bool = False
    smoke_claimed: bool = False


_FINAL_CAPABILITY_REGISTRY: WeakKeyDictionary[
    Any, _FinalCapabilityState
] = WeakKeyDictionary()


class IBR1CalPairError(authority.IBR1AuthorityError):
    """Raised when live subprocess evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1CalPairError(message)


def _root_relative(root: Path, path: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise IBR1CalPairError(f"{label} lies outside the project root") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IBR1CalPairError(f"{label} is not canonical JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    _require(
        payload == authority.canonical_json_bytes(value) + b"\n",
        f"{label} bytes are not canonical",
    )
    return value


def _worker_command(
    *,
    executable: Path,
    root: Path,
    role: str,
    bootstrap: Path,
    output: Path,
    challenge: str,
    parent_pid: int,
) -> list[str]:
    return [
        str(executable),
        "-m",
        WORKER_MODULE,
        "--project-root",
        str(root),
        "--role",
        role,
        "--bootstrap-receipt",
        str(bootstrap),
        "--output-dir",
        str(output),
        "--parent-challenge",
        challenge,
        "--parent-pid",
        str(parent_pid),
    ]


def _artifact_paths(directory: Path) -> dict[str, Path]:
    expected_names = set(ROLE_FILENAMES.values())
    _require(directory.is_dir(), f"CAL worker output directory is missing: {directory}")
    actual_names = {path.name for path in directory.iterdir()}
    _require(
        actual_names == expected_names,
        "CAL worker output does not contain exactly the five official artifacts",
    )
    return {name: directory / filename for name, filename in ROLE_FILENAMES.items()}


def _parse_worker_stdout(
    stdout: str,
    *,
    role: str,
    challenge: str,
    parent_pid: int,
    child_pid: int,
    output: Path,
) -> dict[str, Any]:
    _require(isinstance(stdout, str), f"{role} CAL worker stdout is not text")
    lines = stdout.splitlines()
    _require(len(lines) == 1 and bool(lines[0]), f"{role} CAL worker stdout is not unique")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise IBR1CalPairError(f"{role} CAL worker stdout is not JSON") from exc
    _require(isinstance(payload, dict), f"{role} CAL worker result is malformed")
    canonical_line = authority.canonical_json_bytes(payload).decode("utf-8")
    _require(
        lines[0] == canonical_line and stdout == canonical_line + "\n",
        f"{role} CAL worker stdout is not one canonical result",
    )
    _require(
        payload.get("schema_version") == 1
        and payload.get("analysis_class") == WORKER_RESULT_CLASS
        and payload.get("role") == role
        and payload.get("parent_challenge") == challenge
        and payload.get("parent_pid") == parent_pid
        and payload.get("child_pid") == child_pid
        and payload.get("output_dir") == str(output)
        and payload.get("runtime")
        == {
            "python_executable": str(OFFICIAL_PYTHON_EXECUTABLE),
            "torch_version": OFFICIAL_TORCH_VERSION,
            "cuda_runtime": OFFICIAL_CUDA_RUNTIME,
            "device": OFFICIAL_DEVICE,
        }
        and isinstance(payload.get("calibration_result"), Mapping),
        f"{role} CAL worker result identity/challenge drifted",
    )
    result = payload["calibration_result"]
    expected_orchestration = {
        "analysis_class": "ibr1_cal_worker_parent_challenge",
        "parent_challenge": challenge,
        "parent_pid": parent_pid,
        "child_pid": child_pid,
    }
    _require(
        result.get("role") == role
        and result.get("orchestration_binding") == expected_orchestration
        and result.get("formal_training_authorized") is False
        and all(isinstance(result.get(name), Mapping) for name in (
            "raw_f2_kernel",
            "numeric_evidence",
            "core",
            "envelope",
            "execution_witness",
        )),
        f"{role} CAL worker calibration result drifted",
    )
    return payload


def _run_worker(
    *,
    popen_factory: Callable[..., Any],
    executable: Path,
    root: Path,
    role: str,
    bootstrap: Path,
    output: Path,
    challenge: str,
    parent_pid: int,
) -> tuple[Any, dict[str, Any], dict[str, Path]]:
    _require(not output.exists(), f"{role} CAL output directory is not fresh")
    command = _worker_command(
        executable=executable,
        root=root,
        role=role,
        bootstrap=bootstrap,
        output=output,
        challenge=challenge,
        parent_pid=parent_pid,
    )
    process = popen_factory(
        command,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    child_pid = getattr(process, "pid", None)
    _require(
        isinstance(child_pid, int) and not isinstance(child_pid, bool) and child_pid > 0,
        f"{role} CAL worker has no live Popen PID",
    )
    _require(child_pid != parent_pid, f"{role} CAL worker PID equals the parent PID")
    _require(
        list(getattr(process, "args", [])) == command,
        f"{role} CAL worker Popen args differ from the official command",
    )
    stdout, stderr = process.communicate()
    returncode = getattr(process, "returncode", None)
    _require(isinstance(stdout, str) and isinstance(stderr, str), "CAL worker pipes are not text")
    _require(
        returncode == 0,
        f"{role} CAL worker exited {returncode}; stderr={stderr!r}",
    )
    payload = _parse_worker_stdout(
        stdout,
        role=role,
        challenge=challenge,
        parent_pid=parent_pid,
        child_pid=child_pid,
        output=output,
    )
    paths = _artifact_paths(output)
    witness = _load_canonical(paths["witness"], f"{role} CAL execution witness")
    expected_orchestration = {
        "analysis_class": "ibr1_cal_worker_parent_challenge",
        "parent_challenge": challenge,
        "parent_pid": parent_pid,
        "child_pid": child_pid,
    }
    _require(
        witness.get("role") == role
        and witness.get("process_identity", {}).get("pid") == child_pid
        and witness.get("orchestration_binding") == expected_orchestration,
        f"{role} child witness does not bind the live parent challenge/PID",
    )
    record = {
        "role": role,
        "pid": child_pid,
        "args": command,
        "exit_code": returncode,
        "stdout": stdout,
        "stdout_sha256": _sha256_text(stdout),
        "stderr": stderr,
        "stderr_sha256": _sha256_text(stderr),
        "result": payload,
        "witness_orchestration_binding": expected_orchestration,
    }
    return process, record, paths


class _LivePairProof:
    __slots__ = (
        "__weakref__",
        "_secret",
        "root",
        "freeze_path",
        "forensic",
        "attestation",
        "processes",
        "stage",
    )

    def __init__(
        self,
        secret: object,
        *,
        root: Path,
        freeze_path: Path,
        forensic: Mapping[str, Any],
        attestation: Mapping[str, Any],
        processes: Mapping[str, Any],
    ) -> None:
        _require(secret is _LIVE_PROOF_SECRET, "live CAL proof constructor is private")
        self._secret = secret
        self.root = root
        self.freeze_path = freeze_path
        self.forensic = dict(forensic)
        self.attestation = dict(attestation)
        self.processes = dict(processes)
        self.stage = "fresh"

    def __reduce__(self) -> Any:
        raise TypeError("live CAL pair proof is intentionally non-serializable")

    def __getstate__(self) -> Any:
        raise TypeError("live CAL pair proof is intentionally non-serializable")


class FinalAuthorityCapability:
    """One-process, one-use continuation from live CAL/final into production."""

    __slots__ = (
        "__weakref__",
        "_secret",
        "_root",
        "_freeze_path",
        "_final_path",
        "_final_binding",
        "_attestation",
        "_consumed",
        "_smoke_claimed",
    )

    def __init__(
        self,
        secret: object,
        *,
        root: Path,
        freeze_path: Path,
        final_path: Path,
        final_binding: Mapping[str, str],
        attestation: Mapping[str, Any],
    ) -> None:
        _require(
            secret is _FINAL_CAPABILITY_SECRET,
            "final authority capability constructor is private",
        )
        self._secret = secret
        self._root = root
        self._freeze_path = freeze_path
        self._final_path = final_path
        self._final_binding = dict(final_binding)
        self._attestation = dict(attestation)
        self._consumed = False
        self._smoke_claimed = False

    def __reduce__(self) -> Any:
        raise TypeError("final authority capability is intentionally non-serializable")

    def __getstate__(self) -> Any:
        raise TypeError("final authority capability is intentionally non-serializable")


def _mint_final_authority_capability(
    secret: object,
    *,
    root: Path,
    freeze_path: Path,
    final_path: Path,
    final_binding: Mapping[str, str],
    attestation: Mapping[str, Any],
) -> FinalAuthorityCapability:
    """Mint the only registry-backed continuation from a successful live run."""

    _require(
        secret is _FINAL_CAPABILITY_SECRET,
        "final authority capability mint is private",
    )
    capability = FinalAuthorityCapability(
        secret,
        root=root,
        freeze_path=freeze_path,
        final_path=final_path,
        final_binding=final_binding,
        attestation=attestation,
    )
    _FINAL_CAPABILITY_REGISTRY[capability] = _FinalCapabilityState(
        root=root,
        freeze_path=freeze_path,
        final_path=final_path,
        final_binding_bytes=authority.canonical_json_bytes(
            dict(final_binding)
        ),
        attestation_bytes=authority.canonical_json_bytes(
            dict(attestation)
        ),
    )
    return capability


def _consume_live_pair_proof_for_final(
    proof: Any,
    *,
    project_root: Path,
    freeze_path: Path,
) -> None:
    """Burn the genuine parent-owned proof immediately before final build."""

    _require(
        type(proof) is _LivePairProof
        and proof._secret is _LIVE_PROOF_SECRET
        and proof.stage == "fresh",
        "final assembly requires one fresh genuine live CAL capability",
    )
    proof.stage = "final_consumed"
    _require(
        project_root == proof.root
        and freeze_path == proof.freeze_path
        and proof.attestation.get("parent_pid") == os.getpid(),
        "live CAL capability root/path/parent process drifted",
    )
    freeze = _load_canonical(freeze_path, "live lambda-adoption freeze")
    evidence = freeze.get("evidence")
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("live_orchestration") == proof.attestation,
        "live CAL capability does not bind the freeze being finalized",
    )
    workers = proof.attestation.get("workers")
    _require(isinstance(workers, Mapping), "live CAL capability workers are malformed")
    for role in ("main", "reproduction"):
        process = proof.processes.get(role)
        worker = workers.get(role)
        _require(
            process is not None
            and isinstance(worker, Mapping)
            and getattr(process, "pid", None) == worker.get("pid")
            and getattr(process, "returncode", None) == worker.get("exit_code") == 0,
            f"{role} live Popen capability drifted before final assembly",
        )


def consume_final_authority_capability(
    capability: FinalAuthorityCapability,
    *,
    project_root: str | Path,
    final_receipt_path: str | Path,
) -> dict[str, Any]:
    """Consume the real in-memory continuation for a production lifecycle."""

    state = (
        _FINAL_CAPABILITY_REGISTRY.get(capability)
        if type(capability) is FinalAuthorityCapability
        else None
    )
    _require(
        type(capability) is FinalAuthorityCapability
        and state is not None
        and state.consumed is False,
        "production lifecycle requires one fresh final authority capability",
    )
    assert state is not None
    final_binding = json.loads(state.final_binding_bytes.decode("utf-8"))
    attestation = json.loads(state.attestation_bytes.decode("utf-8"))
    capability._consumed = True
    state.consumed = True
    root = Path(project_root).expanduser().resolve()
    final_path = Path(final_receipt_path).expanduser().resolve()
    _require(
        root == state.root
        and final_path == state.final_path
        and attestation.get("parent_pid") == os.getpid(),
        "final authority capability root/path/parent process drifted",
    )
    document = authority.verify_assembly_receipt(
        root,
        final_path,
        required_phase=authority.ASSEMBLY_PHASE_FINAL,
    )
    observed_binding = {
        "path": str(final_path),
        "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": authority.ASSEMBLY_RECEIPT_CLASS,
        "phase": authority.ASSEMBLY_PHASE_FINAL,
    }
    _require(
        observed_binding == final_binding,
        "final authority capability receipt bytes drifted",
    )
    workers = attestation.get("workers")
    _require(isinstance(workers, Mapping), "final authority capability workers malformed")
    return {
        "analysis_class": "ibr1_final_authority_live_capability",
        "final_assembly_receipt": observed_binding,
        "lambda_adoption_freeze_path": str(state.freeze_path),
        "parent_pid": attestation["parent_pid"],
        "parent_challenge": attestation["parent_challenge"],
        "worker_pids": {
            role: workers[role]["pid"] for role in ("main", "reproduction")
        },
        "authority_eligible": True,
        "formal_training_authorized": False,
    }


def claim_consumed_final_authority_for_smoke(
    capability: FinalAuthorityCapability,
    *,
    project_root: str | Path,
    final_receipt_path: str | Path,
) -> dict[str, Any]:
    """Burn the consumed live continuation when constructing the smoke plan.

    A final receipt on disk is deliberately insufficient.  The production
    smoke plan must receive the exact non-serializable object that the same
    parent process just consumed, and that object can authorize one plan only.
    """

    state = (
        _FINAL_CAPABILITY_REGISTRY.get(capability)
        if type(capability) is FinalAuthorityCapability
        else None
    )
    _require(
        type(capability) is FinalAuthorityCapability
        and state is not None
        and state.consumed is True
        and state.smoke_claimed is False,
        "production smoke plan requires the freshly consumed final authority "
        "capability",
    )
    assert state is not None
    final_binding = json.loads(state.final_binding_bytes.decode("utf-8"))
    attestation = json.loads(state.attestation_bytes.decode("utf-8"))
    capability._smoke_claimed = True
    state.smoke_claimed = True
    root = Path(project_root).expanduser().resolve()
    final_path = Path(final_receipt_path).expanduser().resolve()
    _require(
        root == state.root
        and final_path == state.final_path
        and attestation.get("parent_pid") == os.getpid(),
        "consumed final authority capability root/path/parent process drifted",
    )
    document = authority.verify_assembly_receipt(
        root,
        final_path,
        required_phase=authority.ASSEMBLY_PHASE_FINAL,
    )
    observed_binding = {
        "path": str(final_path),
        "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": authority.ASSEMBLY_RECEIPT_CLASS,
        "phase": authority.ASSEMBLY_PHASE_FINAL,
    }
    _require(
        observed_binding == final_binding,
        "consumed final authority capability receipt bytes drifted",
    )
    return {
        "analysis_class": "ibr1_consumed_final_authority_smoke_claim",
        "final_assembly_receipt": observed_binding,
        "parent_pid": attestation["parent_pid"],
        "parent_challenge": attestation["parent_challenge"],
        "authority_eligible": True,
        "single_use": True,
        "formal_training_authorized": False,
        "internal_test": authority.INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }


def _self_hashed(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result["receipt_payload_sha256"] = authority.canonical_json_sha256(result)
    return result


def _build_freeze_from_live_proof(proof: _LivePairProof) -> dict[str, Any]:
    _require(
        isinstance(proof, _LivePairProof)
        and proof._secret is _LIVE_PROOF_SECRET
        and proof.stage == "fresh",
        "lambda freeze requires one fresh non-serializable live CAL proof",
    )
    forensic = proof.forensic
    _require(
        forensic.get("analysis_class") == authority.CAL_PAIR_FORENSIC_CLASS
        and forensic.get("authority_eligible") is False
        and forensic.get("distinct_processes_verified") is False
        and forensic.get("recorded_process_identities_differ") is True,
        "live proof does not contain a valid forensic CAL pair",
    )
    workers = proof.attestation.get("workers")
    _require(isinstance(workers, Mapping), "live proof worker attestation is malformed")
    for role in ("main", "reproduction"):
        process = proof.processes.get(role)
        worker = workers.get(role)
        _require(
            process is not None
            and isinstance(worker, Mapping)
            and getattr(process, "pid", None) == worker.get("pid")
            and getattr(process, "returncode", None) == worker.get("exit_code") == 0,
            f"{role} live Popen proof drifted before freeze issuance",
        )
    evidence = {
        "bootstrap_assembly_receipt": forensic["bootstrap_assembly_receipt"],
        "cal_main": forensic["main"],
        "cal_reproduction": forensic["reproduction"],
        "raw_f2_byte_identical": forensic["raw_f2_byte_identical"],
        "numeric_evidence_byte_identical": forensic[
            "numeric_evidence_byte_identical"
        ],
        "core_byte_identical": forensic["core_byte_identical"],
        "envelope_byte_identical": forensic["envelope_byte_identical"],
        "file_pair_forensic": {
            "analysis_class": forensic["analysis_class"],
            "authority_eligible": False,
            "distinct_processes_verified": False,
            "recorded_process_identities_differ": forensic[
                "recorded_process_identities_differ"
            ],
        },
        "authority_eligible": True,
        "distinct_processes_verified": True,
        "process_identity": forensic["process_identity"],
        "live_orchestration": proof.attestation,
    }
    return _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": authority.LAMBDA_ADOPTION_FREEZE_CLASS,
            "family_id": authority.IBR1_FAMILY_ID,
            "architecture_lock": authority.IBR1_ARCHITECTURE_LOCK,
            "mechanism": "inheritance_identity_no_drift_audit",
            "generation_contract": "parent_owned_live_popen_challenge_v1",
            "frozen_values": dict(authority.FROZEN_AUX_COEFFICIENTS),
            "proposal": forensic["lambda_proposal"],
            "evidence": evidence,
            "adoption_status": "FROZEN_EXACT_INHERITANCE",
            "difference_action": "STOP_NO_SMOKE_NO_LAMBDA_CHANGE",
            "formal_training_authorized": False,
            "internal_test": authority.INTERNAL_TEST_POLICY,
            "internal_test_opened": False,
        }
    )


def _run_live_cal_pair_and_freeze(
    project_root: str | Path,
    *,
    bootstrap_receipt_path: str | Path,
    output_dir: str | Path,
    freeze_output_path: str | Path,
    final_output_path: str | Path,
    popen_factory: Callable[..., Any],
    _authority_secret: object | None = None,
) -> dict[str, Any]:
    authoritative = _authority_secret is _AUTHORITATIVE_RUN_SECRET
    if authoritative:
        _require(
            popen_factory is _OFFICIAL_POPEN,
            "authoritative CAL requires the captured subprocess.Popen class",
        )
        require_official_python()
    else:
        _require(
            popen_factory is not _OFFICIAL_POPEN,
            "the injectable CAL seam is test-only; use the public production entry",
        )
    root = Path(project_root).expanduser().resolve()
    bootstrap = Path(bootstrap_receipt_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    freeze_output = Path(freeze_output_path).expanduser().resolve()
    final_output = Path(final_output_path).expanduser().resolve()
    _root_relative(root, bootstrap, "bootstrap receipt")
    _root_relative(root, output, "live CAL pair output")
    _root_relative(root, freeze_output, "lambda freeze output")
    _root_relative(root, final_output, "final assembly output")
    _require(not output.exists(), "live CAL pair output directory must be fresh")
    _require(not freeze_output.exists(), "lambda freeze output already exists")
    _require(not final_output.exists(), "final assembly output already exists")
    bootstrap_document = authority.verify_assembly_receipt(
        root,
        bootstrap,
        required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
    )
    output.mkdir(parents=True, exist_ok=False)
    _require(not any(output.iterdir()), "fresh live CAL pair output is not empty")
    challenge = secrets.token_hex(32)
    parent_pid = os.getpid()
    executable = OFFICIAL_PYTHON_EXECUTABLE
    process_objects: dict[str, Any] = {}
    worker_records: dict[str, dict[str, Any]] = {}
    role_paths: dict[str, dict[str, Path]] = {}
    for role in ("main", "reproduction"):
        process, record, paths = _run_worker(
            popen_factory=popen_factory,
            executable=executable,
            root=root,
            role=role,
            bootstrap=bootstrap,
            output=output / role,
            challenge=challenge,
            parent_pid=parent_pid,
        )
        process_objects[role] = process
        worker_records[role] = record
        role_paths[role] = paths
    _require(
        worker_records["main"]["pid"] != worker_records["reproduction"]["pid"],
        "main and reproduction live Popen PIDs are identical",
    )

    forensic = authority.verify_cal_pair(
        root,
        main_raw_f2_kernel_path=role_paths["main"]["raw"],
        reproduction_raw_f2_kernel_path=role_paths["reproduction"]["raw"],
        main_numeric_evidence_path=role_paths["main"]["numeric"],
        reproduction_numeric_evidence_path=role_paths["reproduction"]["numeric"],
        main_core_path=role_paths["main"]["core"],
        reproduction_core_path=role_paths["reproduction"]["core"],
        main_envelope_path=role_paths["main"]["envelope"],
        reproduction_envelope_path=role_paths["reproduction"]["envelope"],
        main_execution_witness_path=role_paths["main"]["witness"],
        reproduction_execution_witness_path=role_paths["reproduction"]["witness"],
        bootstrap_receipt_path=bootstrap,
    )
    source_sha = bootstrap_document["source_binding"]["ibr1_source_sha256"]
    attestation = {
        "schema_version": 1,
        "analysis_class": (
            LIVE_ATTESTATION_CLASS if authoritative else TEST_ONLY_PAIR_CLASS
        ),
        "parent_pid": parent_pid,
        "parent_challenge": challenge,
        "python_executable": str(executable),
        "worker_module": WORKER_MODULE,
        "source_binding": {
            "ibr1_experiment/cal_pair.py": source_sha[
                "ibr1_experiment/cal_pair.py"
            ],
            "ibr1_experiment/cal_worker.py": source_sha[
                "ibr1_experiment/cal_worker.py"
            ],
            "ibr1_experiment/runtime_contract.py": source_sha[
                "ibr1_experiment/runtime_contract.py"
            ],
        },
        "workers": worker_records,
    }
    worker_pids = {
        role: worker_records[role]["pid"] for role in ("main", "reproduction")
    }
    if not authoritative:
        return {
            "schema_version": 1,
            "analysis_class": TEST_ONLY_PAIR_CLASS,
            "parent_challenge": challenge,
            "parent_pid": parent_pid,
            "worker_pids": worker_pids,
            "forensic_pair_analysis_class": forensic["analysis_class"],
            "recorded_process_identities_differ": forensic[
                "recorded_process_identities_differ"
            ],
            "freeze_written": False,
            "final_written": False,
            "final_authority_capability": None,
            "authority_eligible": False,
            "formal_training_authorized": False,
        }

    proof = _LivePairProof(
        _LIVE_PROOF_SECRET,
        root=root,
        freeze_path=freeze_output,
        forensic=forensic,
        attestation=attestation,
        processes=process_objects,
    )
    freeze = _build_freeze_from_live_proof(proof)
    freeze_sha = authority.exclusive_write_json(freeze_output, freeze)
    authority.verify_lambda_adoption_freeze(root, freeze_output)
    final_binding = authority._freeze_final_assembly_receipt_from_live_cal(
        root,
        final_output,
        lambda_adoption_freeze_path=freeze_output,
        live_pair_proof=proof,
    )
    authority.verify_assembly_receipt(
        root,
        final_output,
        required_phase=authority.ASSEMBLY_PHASE_FINAL,
    )
    _require(
        proof.stage == "final_consumed",
        "live CAL proof was not consumed exactly once by final assembly",
    )
    final_capability = _mint_final_authority_capability(
        _FINAL_CAPABILITY_SECRET,
        root=root,
        freeze_path=freeze_output,
        final_path=final_output,
        final_binding=final_binding,
        attestation=attestation,
    )
    proof.stage = "successor_minted"
    return {
        "path": str(freeze_output),
        "sha256": freeze_sha,
        "receipt_payload_sha256": freeze["receipt_payload_sha256"],
        "analysis_class": authority.LAMBDA_ADOPTION_FREEZE_CLASS,
        "parent_challenge": challenge,
        "parent_pid": parent_pid,
        "worker_pids": worker_pids,
        "final_assembly": final_binding,
        "final_authority_capability": final_capability,
        "authority_eligible": True,
        "formal_training_authorized": False,
    }


def run_live_cal_pair_and_freeze(
    project_root: str | Path,
    *,
    bootstrap_receipt_path: str | Path,
    output_dir: str | Path,
    freeze_output_path: str | Path,
    final_output_path: str | Path,
) -> dict[str, Any]:
    """Run two fixed workers, then issue freeze/final plus one continuation."""

    return _run_live_cal_pair_and_freeze(
        project_root,
        bootstrap_receipt_path=bootstrap_receipt_path,
        output_dir=output_dir,
        freeze_output_path=freeze_output_path,
        final_output_path=final_output_path,
        popen_factory=_OFFICIAL_POPEN,
        _authority_secret=_AUTHORITATIVE_RUN_SECRET,
    )


__all__ = [
    "FinalAuthorityCapability",
    "IBR1CalPairError",
    "claim_consumed_final_authority_for_smoke",
    "consume_final_authority_capability",
    "run_live_cal_pair_and_freeze",
]
