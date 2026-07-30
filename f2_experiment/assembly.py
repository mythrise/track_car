"""Lifecycle orchestration for the F2 production assembly smoke.

This module owns the fail-closed assembly lifecycle mandated by handoff
section 14.3 and the 2026-07-18 Fable-5 merged adjudication receipt (as
amended by the preregistered f2-adjudication-amendment-1):

``freeze assembly/source receipt v4 -> update-0 EVAL-FIX (S-SELF) ->
paired 128 updates on SMK-TRAIN -> update-128 EVAL-FIX (both arms) ->
G6-G9 receipts -> combined smoke gate receipt -> external review``

It provides the assembly source receipt v4 (12 f2 sources, 15 third-party
transitive dependencies, data/weight/cache/controller bindings), the CAL
zero-update audit with the frozen lambda proposal mechanism, the EVAL-FIX
dual-mode executor, the paired-smoke orchestration around
:func:`f2_experiment.runner.run_paired_smoke`, and fail-closed checkpoint
save/load.  Every receipt is written with ``O_EXCL`` exclusive semantics into
``experiments/collected_v1_main/f2_smoke``-style directories; nothing here
can ever overwrite support receipt v3 or any earlier artifact.

Model- and data-side construction (real checkpoint/cache loaders, package
factories, G6 instrumentation) belongs to ``assembly_data`` /
``assembly_model``.  This module only consumes their narrow integration
seams and fails closed while they are absent, so the lifecycle contract is
testable with fakes and the real smoke stays forbidden until the full
assembly is bound by a fresh receipt (handoff section 16.3).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Literal

import torch

from .assembly_data import F2AssemblyContractError
from .cli import (
    NAMESPACE_PACKAGES,
    exclusive_write_json,
    source_bindings,
    transitive_source_bindings,
)
from .controller import (
    ActionFilterConfig,
    ActionFilterController,
    DEFAULT_CONFIG,
    bind_controller_identity,
)
from .evaluation import (
    G6Update,
    GateReceipt,
    aggregate_row_losses,
    build_smoke_gate_receipt,
    evaluate_g6,
    evaluate_g7,
    evaluate_g8,
    evaluate_g9,
)
from .model import AP2_HORIZON, AP2Prediction, ap2_track_loss
from .runner import (
    ARM_ORDER,
    S_CTRL,
    S_SELF,
    ArmCallbacks,
    ArmName,
    OptimizerUpdateEvent,
    RunnerG7Update,
    RunnerRow,
    RunnerTelemetryHooks,
    checkpoint_init_sha256,
    run_paired_smoke,
)
from .reproducibility import (
    F2CudaReproducibilityError,
    validate_cuda_reproducibility_receipt,
)
from .support import (
    ARCHITECTURE_LOCK,
    FROZEN_TRAIN_RELATIVE,
    FROZEN_TRAIN_ROWS,
    FROZEN_TRAIN_SHA256,
    INTERNAL_TEST_POLICY,
    SUPPORT_EXPECTATIONS,
    build_frozen_support,
    canonical_json_sha256,
    continues_sequence,
    parse_train_jsonl,
    verify_approval_files,
)


ASSEMBLY_SCHEMA_VERSION = 1
ASSEMBLY_RECEIPT_VERSION = 4
ASSEMBLY_RECEIPT_CLASS = "f2_assembly_source_receipt"
CAL_AUDIT_RECEIPT_CLASS = "f2_cal_zero_update_audit_receipt"
EVAL_FIX_RECEIPT_CLASS = "f2_eval_fix_snapshot_receipt"
CHECKPOINT_RECEIPT_CLASS = "f2_arm_checkpoint_receipt"
GATE_INPUTS_CLASS = "f2_smoke_gate_inputs"
SMOKE_SUMMARY_CLASS = "f2_production_smoke_summary"

CAL_SUPPORT = "CAL"
EVAL_SUPPORT = "EVAL-FIX"
SMOKE_SUPPORT = "SMK-TRAIN"

ADJUDICATION_RELATIVE = Path(
    "experiments/collected_v1_main/external_reviews/"
    "20260718_fable5_f2_merged_adjudication_receipt.json"
)
ADJUDICATION_SHA256 = (
    "c56f72c7010ac4c4be809eee105497e94813e30bd8bafd5a542db75967d1a453"
)
# PRIMARY amendment 1 (preregistered before the CAL audit, before any smoke
# optimizer update and before any dev evaluation): amends
# rulings.b_lambda_policy because AP2 zero-init makes d(track)/d(base.proj)
# identically zero at zero updates.
ADJUDICATION_AMENDMENT1_RELATIVE = Path(
    "experiments/collected_v1_main/external_reviews/"
    "20260718_fable5_f2_adjudication_amendment1_receipt.json"
)
ADJUDICATION_AMENDMENT1_SHA256 = (
    "2adb79ec3cd5f7d077eec23f10fac1da71eb3bd86135ea9c2837db90b065d40c"
)
ADJUDICATION_AMENDMENT1_ID = "f2-adjudication-amendment-1"
ADJUDICATION_AMENDMENT1_AMENDS = "rulings.b_lambda_policy"

# PRIMARY lambda-freeze receipt (ruling b freeze protocol).  The freeze
# receipt is re-issued whenever CAL is legitimately re-run, so its identity
# is bound per-receipt (path+SHA inside assembly receipt v4 successors) and
# validated by chain consistency at smoke time -- never by hardcoded lambda
# values in this module.
LAMBDA_FREEZE_CLASS = "f2_lambda_freeze"
LEGACY_SEEDED_LAMBDA_FREEZE_SHA256 = (
    "4fe38c150e81382600dfc5ff62a228e6b60b9432ef301b7092186ee7d112f128"
)

# Forensic rebuild outputs are never authoritative (GPT-5.6 sol P1-4): they
# carry their own analysis classes and can never authorize formal training.
FORENSIC_GATE_CLASS = "f2_preformal_smoke_gate_forensic_rebuild"
FORENSIC_GATES_CLASS = "f2_preformal_smoke_gates_forensic_rebuild"
PROMPT_ERRATUM_RELATIVE = Path(
    "data/collected_v1/audits/prompt_normalization_erratum_v4.json"
)
PROMPT_ERRATUM_SHA256 = (
    "baa9c322366e40377858cdedc9618dcc08e419df7991ae4bd3e7ca499facdbec"
)

SMOKE_PACKAGE = "SA-Hstar"
G6_BLOCK_MODE = "bstar"
G6_PROBE_SURFACE = "base.proj"
G7_G9_ARM_POLICY = "both_arms_AND"

# PRIMARY incremental adjudications (2026-07-18/19): the cache manifest and
# provenance whole-file SHAs are bound, a full-cache token payload re-read is
# forbidden (it would traverse the sealed internal-test subtree), and byte
# identity for every token file the lifecycle can touch is anchored by the
# train-split per-file TokenHashLedger frozen into the receipt.
CACHE_BINDING_MODE = (
    "manifest_sha_plus_train_split_per_file_ledger_no_sealed_reread"
)
CACHE_BINDING_REASON = (
    "internal-test seal: a full-cache token payload re-read would traverse "
    "the sealed internal-test subtree; instead the train-split per-file "
    "token hash ledger (assembly_data.build_train_token_ledger) freezes one "
    "SHA-256 per reachable token file and every load byte-verifies against "
    "it, while token_payload_sha256 stays the frozen manifest literal"
)
# The ledger anchor is trust-on-first-read at freeze: the per-file hashes
# are first observed at build-assembly-receipt time and frozen; every later
# lifecycle stage must rebuild the ledger to the exact same ledger_sha256.
TOKEN_LEDGER_ANCHOR = "trust_on_first_read_at_freeze"
TOKEN_LEDGER_FIELDS = ("token_ledger_sha256", "token_ledger_file_count")
FROZEN_TRAIN_TOKEN_IMAGES = 18_473
FROZEN_TRAIN_TOKEN_FILES = 36_946

LAMBDA_AUX_LOSSES = ("L_cot", "L_future", "L_verify")
LAMBDA_TARGET_FRACTION = 0.5
LAMBDA_UPPER_BOUND = 1.0
LAMBDA_SIGNIFICANT_DIGITS = 3
# Amended mechanism (amendment 1): aux-relative calibration; the original
# track-relative numerator is structurally zero under AP2 zero-init.
LAMBDA_MECHANISM = (
    "lambda_i = 0.5 * min_j(median||g_aux_j||) / median||g_aux_i||, "
    "each capped at 1.0"
)

GATE_CONTRACT_CHANGES = (
    "G7Update.abs_tanh_s_prev",
    "evaluate_g7.prev_scale_saturation_rate",
)

OPTIMIZER_CONTRACT = {
    "optimizer": "AdamW",
    "base_lr": 2e-5,
    "head_lr": 3e-4,
    "weight_decay": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "grad_clip_norm": 1.0,
    "source": "20260718 Fable-5 merged adjudication ruling f",
}

G_LEGACY_MAP = {
    "G1_step0_parity": (
        "HS6 + CAL step0 parity check (assert_step0_controlled_axis_persistence "
        "on CAL, corrigendum-2 frozen support=CAL) + prev-free graph assertion"
    ),
    "G2_reachability": (
        "CAL zero-update audit reachability section + G6 aux>=127/128, "
        "track>=120/128 at smoke"
    ),
    "G3_subordination_ratio": (
        "CAL per-aux gradient-norm statistics + G6 median aux/track <=1.5 "
        "(bstar) / per-aux <=0.75 (fallback); track-side evidence is handed "
        "to smoke G6 per f2-adjudication-amendment-1 (AP2 zero-init makes "
        "d(track)/d(base.proj) identically zero at zero updates)"
    ),
    "G4_alignment": (
        "G6 median cos(total, track) >= 0.6 and signed projection positive "
        ">= 108/120"
    ),
    "G5_anti_goodhart": (
        "Carried as G6 contract clause: lambda scaling must not change cos "
        "sign/direction; enforced by dedicated unit test (lambda-scaling "
        "invariance of cos)"
    ),
}

LAMBDA_POLICY = {
    "mechanism": LAMBDA_MECHANISM,
    "aux_losses": list(LAMBDA_AUX_LOSSES),
    "significant_digits": LAMBDA_SIGNIFICANT_DIGITS,
    "calibration_support": CAL_SUPPORT,
    "track_zero_assertion": (
        "the track-loss gradient on base.proj must be exactly zero on every "
        "CAL row at zero updates (AP2 zero-init structural converse); track "
        "reachability and aux/track subordination are enforced by smoke G6"
    ),
    "freeze_protocol": (
        "values freeze in the CAL audit receipt before any smoke optimizer "
        "update and before any dev evaluation is seen; changing them after "
        "update-0 EVAL-FIX is a protocol violation (HS12)"
    ),
    "source": (
        "20260718 Fable-5 merged adjudication ruling b as amended by "
        "f2-adjudication-amendment-1 (preregistered)"
    ),
    "amendment_id": ADJUDICATION_AMENDMENT1_ID,
}

EVAL_MODE_CONTRACT = {
    "support": EVAL_SUPPORT,
    "modes": ["logged", "self"],
    "grad": "torch.no_grad",
    "module_mode": (
        "eval() on all assembled modules; owned by the predictor builder "
        "(F2 modules have no dropout/BN, the LLM is frozen)"
    ),
    "reset_rows": (
        "prev_fy = logged prev (forward,yaw); controller state re-initialized "
        "from the logged prev_action"
    ),
    "self_recurrence": (
        "non-reset rows reuse the previous row's controller-filtered sent "
        "action (forward,yaw), detached; byte-aligned with runner branch2"
    ),
    "row_loss": "ap2_track_loss(prediction, target).total on raw actions",
    "strata": ["overall", "change", "turn", "other"],
}

LIFECYCLE_ORDER = (
    "freeze_assembly_receipt",
    "update0_eval_fix_S-SELF_logged_self",
    "paired_128_updates_SMK-TRAIN",
    "update128_eval_fix_both_arms_logged_self",
    "build_G6_G7_G8_G9_receipts",
    "build_combined_smoke_gate_receipt",
    "external_review",
)

CHECKPOINT_REQUIRED_KEYS = (
    "model",
    "optimizer",
    "u_pre",
    "arm",
    "assembly_receipt_sha256",
    "checkpoint_init_sha256",
)

EvalMode = Literal["logged", "self"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise F2AssemblyContractError(message)


def _sha256_file(path: Path, label: str) -> str:
    if not path.is_file():
        raise F2AssemblyContractError(f"{label} is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round_sig(value: float, digits: int = LAMBDA_SIGNIFICANT_DIGITS) -> float:
    """Round to ``digits`` significant digits (adjudication ruling b)."""

    if not isinstance(digits, int) or digits <= 0:
        raise F2AssemblyContractError("significant digits must be positive")
    number = float(value)
    if not math.isfinite(number):
        raise F2AssemblyContractError("cannot round a nonfinite value")
    if number == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(number)))
    return round(number, digits - 1 - exponent)


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2AssemblyContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise F2AssemblyContractError(f"{label} must be finite and nonnegative")
    return number


def _ensure_empty_directory(path: Path, label: str) -> Path:
    """Refuse to reuse a directory that already holds artifacts."""

    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        _require(resolved.is_dir(), f"{label} exists and is not a directory")
        _require(
            not any(resolved.iterdir()),
            f"{label} is not empty; failed lifecycles are preserved as "
            "evidence and must never be overwritten or resumed in place",
        )
    else:
        resolved.mkdir(parents=True)
    return resolved


def _write_receipt(path: Path, payload: Any) -> str:
    try:
        return exclusive_write_json(path, payload)
    except FileExistsError as exc:
        raise F2AssemblyContractError(
            f"refusing to overwrite an existing receipt: {path}"
        ) from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise F2AssemblyContractError(f"{label} is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise F2AssemblyContractError(f"{label} is unreadable: {path}") from exc
    if not isinstance(document, dict):
        raise F2AssemblyContractError(f"{label} must be a JSON object: {path}")
    return document


def _integration_attr(module_name: str, attribute: str) -> Any:
    """Resolve an assembly_data/assembly_model seam, failing closed if absent."""

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise F2AssemblyContractError(
            "F2 assembly integration is incomplete: "
            f"{module_name} is unavailable ({attribute} is required); the real "
            "smoke stays forbidden until the full assembly is bound"
        ) from exc
    value = getattr(module, attribute, None)
    if value is None:
        raise F2AssemblyContractError(
            "F2 assembly integration is incomplete: "
            f"{module_name}.{attribute} is not provided"
        )
    return value


def _default_asset_binding(project_root: Path) -> Mapping[str, Any]:
    """Manifest-binding asset verification (no token payload re-read).

    Calls ``assembly_data.verify_frozen_assets`` with
    ``verify_token_payload=False`` per the PRIMARY incremental adjudication:
    the internal-test seal forbids re-reading the token payload subtree.
    """

    verify_frozen_assets = _integration_attr(
        "f2_experiment.assembly_data", "verify_frozen_assets"
    )
    binding = verify_frozen_assets(project_root, verify_token_payload=False)
    _require(
        isinstance(binding, Mapping) and bool(binding),
        "frozen asset verification returned no binding",
    )
    build_ledger = _integration_attr(
        "f2_experiment.assembly_data", "build_train_token_ledger"
    )
    token_ledger = build_ledger(project_root)
    ledger_sha, token_files = _ledger_identity(token_ledger)
    _require(
        token_files == FROZEN_TRAIN_TOKEN_FILES,
        "train-split token ledger file count differs from the frozen "
        f"contract: observed {token_files}, expected "
        f"{FROZEN_TRAIN_TOKEN_FILES}",
    )
    return {
        **dict(binding),
        "token_ledger_sha256": ledger_sha,
        "token_ledger_file_count": token_files,
    }


def _assert_sealed_cache_binding(asset_binding: Mapping[str, Any]) -> None:
    vision_cache = asset_binding.get("vision_cache")
    if isinstance(vision_cache, Mapping) and vision_cache.get(
        "token_payload_verified"
    ):
        raise F2AssemblyContractError(
            "cache binding must not re-read the token payload "
            "(internal-test seal); use manifest-binding mode with "
            "verify_token_payload=False"
        )


def _assert_token_ledger_anchor(asset_binding: Mapping[str, Any]) -> None:
    ledger_sha = asset_binding.get("token_ledger_sha256")
    token_files = asset_binding.get("token_ledger_file_count")
    _require(
        isinstance(ledger_sha, str)
        and len(ledger_sha) == 64
        and all(char in "0123456789abcdef" for char in ledger_sha),
        "asset binding must freeze a lowercase token_ledger_sha256 anchor",
    )
    _require(
        isinstance(token_files, int)
        and not isinstance(token_files, bool)
        and token_files > 0,
        "asset binding must freeze a positive token_ledger_file_count",
    )


def _default_support_rows_loader(
    project_root: Path, support_name: str, token_ledger: Any
) -> tuple[Sequence[RunnerRow], Sequence[Mapping[str, Any]], frozenset[int]]:
    """Load ``(runner_rows, raw_rows, strafe_reset_indices)`` for a support.

    Wires the real ``assembly_data`` interface: frozen support receipt
    (HS1-HS5), block-major row ordering, the fail-closed cache loader via
    ``build_runner_rows`` (token bytes verified against the receipt-frozen
    ``token_ledger``), and the receipt-derived STRAFE reset set.
    """

    build_rows = _integration_attr(
        "f2_experiment.assembly_data", "build_runner_rows"
    )
    ordered_rows = _integration_attr(
        "f2_experiment.assembly_data", "ordered_support_rows"
    )
    reset_sets = _integration_attr(
        "f2_experiment.assembly_data", "smoke_reset_sets"
    )
    cache_roots = _integration_attr(
        "f2_experiment.assembly_data", "frozen_cache_roots"
    )
    root = Path(project_root).expanduser().resolve()
    train_path = root / FROZEN_TRAIN_RELATIVE
    receipt = build_frozen_support(train_path)
    rows = parse_train_jsonl(train_path.read_bytes())
    base_root, cache_root = cache_roots(root)
    runner_rows = build_rows(
        rows=rows,
        receipt=receipt,
        support_name=support_name,
        base_root=base_root,
        cache_root=cache_root,
        token_ledger=token_ledger,
    )
    raw_rows = tuple(row for _index, row in ordered_rows(rows, receipt, support_name))
    strafe, _expected = reset_sets(receipt, support_name)
    return runner_rows, raw_rows, frozenset(strafe)


def _ledger_identity(token_ledger: Any) -> tuple[str, int]:
    ledger_sha = getattr(token_ledger, "ledger_sha256", None)
    token_files = getattr(token_ledger, "token_files", None)
    _require(
        isinstance(ledger_sha, str)
        and len(ledger_sha) == 64
        and all(char in "0123456789abcdef" for char in ledger_sha),
        "token ledger has no valid ledger_sha256",
    )
    _require(
        isinstance(token_files, int)
        and not isinstance(token_files, bool)
        and token_files > 0,
        "token ledger has no valid token file count",
    )
    return ledger_sha, token_files


def _resolve_frozen_token_ledger(
    root: Path,
    receipt_document: Mapping[str, Any],
    *,
    token_ledger: Any | None = None,
) -> Any:
    """Rebuild the train-split token ledger and require the frozen anchor.

    The receipt freezes ``token_ledger_sha256``/``token_ledger_file_count``
    (trust-on-first-read at freeze); any later rebuild that does not
    reproduce the exact ledger SHA fails closed before any token is loaded.
    """

    binding = receipt_document.get("asset_binding")
    _require(
        isinstance(binding, Mapping),
        "assembly receipt carries no asset binding",
    )
    frozen_sha = binding.get("token_ledger_sha256")
    frozen_count = binding.get("token_ledger_file_count")
    _require(
        isinstance(frozen_sha, str) and len(frozen_sha) == 64,
        "assembly receipt does not freeze a token ledger anchor; CAL and "
        "smoke are forbidden until build-assembly-receipt freezes one",
    )
    if token_ledger is None:
        build_ledger = _integration_attr(
            "f2_experiment.assembly_data", "build_train_token_ledger"
        )
        token_ledger = build_ledger(root)
    ledger_sha, token_files = _ledger_identity(token_ledger)
    _require(
        ledger_sha == frozen_sha,
        "TOKEN_LEDGER_MISMATCH: the rebuilt train-split token ledger "
        f"({ledger_sha}) differs from the receipt-frozen anchor "
        f"({frozen_sha}); the cache is not the bound one and the lifecycle "
        "is forbidden",
    )
    _require(
        token_files == frozen_count,
        "token ledger file count differs from the receipt-frozen anchor",
    )
    return token_ledger


# ---------------------------------------------------------------------------
# Assembly source receipt v4
# ---------------------------------------------------------------------------


def _test_bindings(root: Path) -> dict[str, str]:
    tests_dir = root / "tests" / "f2"
    _require(tests_dir.is_dir(), f"tests/f2 directory is missing under {root}")
    files = sorted(
        path
        for path in tests_dir.glob("*.py")
        if path.is_file() and not path.name.startswith("._")
    )
    _require(bool(files), "tests/f2 contains no python test files")
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in files
    }


def _verify_namespace_packages(root: Path) -> list[str]:
    for package in NAMESPACE_PACKAGES:
        directory = root.joinpath(*package.split("."))
        _require(
            directory.is_dir(),
            f"namespace package directory is missing: {directory}",
        )
        marker = directory / "__init__.py"
        _require(
            not marker.exists(),
            f"{package} must stay a fileless namespace package; found {marker}",
        )
    return list(NAMESPACE_PACKAGES)


def _frozen_file_binding(
    root: Path, relative: Path, expected_sha256: str, label: str
) -> dict[str, str]:
    actual = _sha256_file(root / relative, label)
    _require(
        actual == expected_sha256,
        f"{label} SHA mismatch: observed {actual}, frozen {expected_sha256}",
    )
    return {"path": relative.as_posix(), "sha256": actual}


def _resolve_bound_path(root: Path, value: Any, label: str) -> Path:
    _require(
        isinstance(value, str) and bool(value),
        f"{label} binding has no path",
    )
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _verify_static_assembly_receipt_evidence(
    path: Path, label: str
) -> tuple[dict[str, Any], str]:
    """Verify immutable receipt identity without comparing it to the live tree."""

    document = _load_json(path, label)
    _require(
        document.get("analysis_class") == ASSEMBLY_RECEIPT_CLASS,
        f"{label} has the wrong analysis class",
    )
    _require(
        document.get("schema_version") == ASSEMBLY_SCHEMA_VERSION
        and document.get("receipt_version") == ASSEMBLY_RECEIPT_VERSION,
        f"{label} schema/receipt version mismatch",
    )
    _require(
        document.get("architecture_lock") == ARCHITECTURE_LOCK,
        f"{label} architecture lock mismatch",
    )
    _require(
        document.get("internal_test") == INTERNAL_TEST_POLICY
        and document.get("internal_test_opened") is False,
        f"{label} internal-test seal is broken",
    )
    payload = dict(document)
    stored_payload_sha = payload.pop("receipt_payload_sha256", None)
    _require(
        stored_payload_sha == canonical_json_sha256(payload),
        f"{label} payload SHA does not match its own content",
    )
    return document, str(stored_payload_sha)


def _verify_lambda_freeze_evidence(
    root: Path,
    freeze_document: Mapping[str, Any],
    *,
    expected_cal_path: Path | None = None,
    freeze_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the immutable main/repro/bootstrap evidence bound by a freeze."""

    legacy_seeded_freeze = (
        freeze_file_sha256 == LEGACY_SEEDED_LAMBDA_FREEZE_SHA256
    )
    _require(
        freeze_document.get("schema_version") == 1,
        "lambda freeze receipt schema version mismatch",
    )
    _require(
        freeze_document.get("internal_test_opened") is False,
        "lambda freeze receipt internal-test seal is broken",
    )
    _require(
        freeze_document.get("internal_test") == INTERNAL_TEST_POLICY
        or (
            legacy_seeded_freeze
            and freeze_document.get("internal_test") is None
        ),
        "lambda freeze receipt must record internal_test=sealed; only the "
        "exact legacy seeded freeze identity may omit this field",
    )
    evidence = freeze_document.get("evidence")
    _require(
        isinstance(evidence, Mapping),
        "lambda freeze receipt evidence is malformed",
    )
    adopted = evidence.get("cal_audit_receipt")
    reproduction = evidence.get("deterministic_reproduction")
    bootstrap = evidence.get("bootstrap_assembly_receipt")
    _require(
        isinstance(adopted, Mapping),
        "lambda freeze receipt has no main CAL evidence",
    )
    _require(
        isinstance(reproduction, Mapping),
        "lambda freeze receipt has no deterministic CAL reproduction evidence",
    )
    _require(
        isinstance(bootstrap, Mapping),
        "lambda freeze receipt has no bootstrap assembly evidence",
    )

    cal_path = _resolve_bound_path(
        root, adopted.get("path"), "lambda freeze main CAL evidence"
    ).expanduser().resolve()
    if expected_cal_path is not None:
        _require(
            cal_path == expected_cal_path.expanduser().resolve(),
            "the PRIMARY lambda freeze receipt does not adopt this CAL audit "
            "receipt path (authority chain broken)",
        )
    cal_file_sha = _sha256_file(cal_path, "lambda freeze main CAL receipt")
    _require(
        adopted.get("sha256") == cal_file_sha,
        "the PRIMARY lambda freeze receipt does not adopt this CAL audit "
        "receipt: evidence SHA differs from the actual main CAL bytes",
    )
    cal_document = _load_json(cal_path, "lambda freeze main CAL receipt")
    _require(
        cal_document.get("analysis_class") == CAL_AUDIT_RECEIPT_CLASS
        and cal_document.get("support") == CAL_SUPPORT
        and cal_document.get("rows") == SUPPORT_EXPECTATIONS[CAL_SUPPORT].rows
        and cal_document.get("optimizer_updates") == 0,
        "lambda freeze main CAL receipt is not a valid zero-update CAL audit",
    )
    _require(
        cal_document.get("internal_test") == INTERNAL_TEST_POLICY
        and cal_document.get("internal_test_opened") is False,
        "lambda freeze main CAL receipt internal-test seal is broken",
    )
    cal_context = cal_document.get("cal_context")
    _require(
        isinstance(cal_context, Mapping),
        "lambda freeze main CAL context is malformed",
    )
    for field in ("seed", "device", "checkpoint_init_sha256"):
        _require(
            field in adopted
            and adopted.get(field) == cal_context.get(field),
            "lambda freeze main CAL evidence differs from the actual CAL "
            f"context at {field!r}",
        )

    reproduction_path = _resolve_bound_path(
        root,
        reproduction.get("path"),
        "lambda freeze CAL reproduction evidence",
    ).expanduser().resolve()
    _require(
        reproduction_path != cal_path,
        "lambda freeze CAL reproduction must be a distinct receipt path",
    )
    reproduction_sha = _sha256_file(
        reproduction_path, "lambda freeze CAL reproduction receipt"
    )
    _require(
        reproduction.get("sha256") == reproduction_sha,
        "lambda freeze CAL reproduction evidence SHA differs from its bytes",
    )
    _require(
        reproduction_sha == cal_file_sha
        and reproduction_path.read_bytes() == cal_path.read_bytes(),
        "lambda freeze CAL reproduction is not byte-identical to the main "
        "CAL receipt; freeze and smoke are forbidden",
    )

    bootstrap_path = _resolve_bound_path(
        root,
        bootstrap.get("path"),
        "lambda freeze bootstrap assembly evidence",
    ).expanduser().resolve()
    bootstrap_sha = _sha256_file(
        bootstrap_path, "lambda freeze bootstrap assembly receipt"
    )
    _require(
        bootstrap.get("sha256") == bootstrap_sha,
        "lambda freeze bootstrap assembly evidence SHA differs from its bytes",
    )
    _, bootstrap_payload_sha = _verify_static_assembly_receipt_evidence(
        bootstrap_path, "lambda freeze bootstrap assembly receipt"
    )
    _require(
        cal_document.get("assembly_receipt_sha256") == bootstrap_sha
        and cal_document.get("assembly_receipt_payload_sha256")
        == bootstrap_payload_sha,
        "lambda freeze bootstrap assembly receipt differs from the receipt "
        "bound by the main CAL audit",
    )
    recorded_bootstrap_payload_sha = bootstrap.get("receipt_payload_sha256")
    _require(
        recorded_bootstrap_payload_sha == bootstrap_payload_sha
        or (
            legacy_seeded_freeze
            and recorded_bootstrap_payload_sha is None
        ),
        "lambda freeze bootstrap payload evidence differs from the verified "
        "bootstrap receipt; only the exact legacy seeded freeze identity may "
        "omit this field",
    )

    return {
        "cal_audit_receipt": {
            "path": cal_path,
            "sha256": cal_file_sha,
            "document": cal_document,
        },
        "deterministic_reproduction": {
            "path": reproduction_path,
            "sha256": reproduction_sha,
            "byte_identical": True,
        },
        "bootstrap_assembly_receipt": {
            "path": bootstrap_path,
            "sha256": bootstrap_sha,
            "receipt_payload_sha256": bootstrap_payload_sha,
        },
    }


def _lambda_freeze_binding(
    root: Path, lambda_freeze_receipt_path: str | Path
) -> dict[str, str]:
    """Bind a PRIMARY lambda freeze after verifying its immutable evidence."""

    given = Path(lambda_freeze_receipt_path)
    path = given if given.is_absolute() else (root / given)
    file_sha = _sha256_file(path, "lambda freeze receipt")
    document = _load_json(path, "lambda freeze receipt")
    _require(
        document.get("analysis_class") == LAMBDA_FREEZE_CLASS,
        "lambda freeze receipt has the wrong analysis class",
    )
    _verify_lambda_freeze_evidence(
        root, document, freeze_file_sha256=file_sha
    )
    try:
        recorded_path = path.resolve().relative_to(root).as_posix()
    except ValueError:
        recorded_path = (
            str(path.resolve()) if given.is_absolute() else given.as_posix()
        )
    return {
        "path": recorded_path,
        "sha256": file_sha,
        "analysis_class": LAMBDA_FREEZE_CLASS,
    }


def _amendment_binding(root: Path) -> dict[str, str]:
    binding = _frozen_file_binding(
        root,
        ADJUDICATION_AMENDMENT1_RELATIVE,
        ADJUDICATION_AMENDMENT1_SHA256,
        "adjudication amendment 1 receipt",
    )
    return {
        **binding,
        "amendment_id": ADJUDICATION_AMENDMENT1_ID,
        "amends": ADJUDICATION_AMENDMENT1_AMENDS,
    }


def build_assembly_receipt(
    project_root: str | Path,
    *,
    asset_binding: Mapping[str, Any] | None = None,
    support_receipt_path: str | Path | None = None,
    lambda_freeze_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the source receipt v4 payload (never writes anything).

    ``asset_binding`` defaults to the ``assembly_data.verify_frozen_assets``
    integration seam (real checkpoint / vision-cache / token / Qwen SHA
    bindings); it fails closed while that module is not integrated.

    ``lambda_freeze_receipt_path`` binds the PRIMARY lambda-freeze receipt
    into the approvals chain.  It may be omitted only for the CAL-bootstrap
    receipt (the freeze does not exist before CAL runs);
    :func:`run_production_smoke` refuses any receipt without this binding.
    """

    root = Path(project_root).expanduser().resolve()
    sources = source_bindings(root)
    transitive = transitive_source_bindings(root)
    tests = _test_bindings(root)
    namespaces = _verify_namespace_packages(root)
    approvals = verify_approval_files(root)
    adjudication = _frozen_file_binding(
        root,
        ADJUDICATION_RELATIVE,
        ADJUDICATION_SHA256,
        "merged adjudication receipt",
    )
    amendment = _amendment_binding(root)
    lambda_freeze_binding: dict[str, str] | None = None
    if lambda_freeze_receipt_path is not None:
        lambda_freeze_binding = _lambda_freeze_binding(
            root, lambda_freeze_receipt_path
        )
        approvals = {
            **approvals,
            "fable_f2_lambda_freeze": lambda_freeze_binding["sha256"],
        }
    train_path = root / FROZEN_TRAIN_RELATIVE
    train_sha = _sha256_file(train_path, "frozen train JSONL")
    _require(
        train_sha == FROZEN_TRAIN_SHA256,
        "frozen train JSONL SHA mismatch against the HS1 contract",
    )
    prompt_binding = _frozen_file_binding(
        root,
        PROMPT_ERRATUM_RELATIVE,
        PROMPT_ERRATUM_SHA256,
        "prompt normalization erratum v4",
    )
    if asset_binding is None:
        asset_binding = _default_asset_binding(root)
    _require(
        isinstance(asset_binding, Mapping) and bool(asset_binding),
        "asset binding must be a nonempty mapping",
    )
    _assert_sealed_cache_binding(asset_binding)
    _assert_token_ledger_anchor(asset_binding)
    support_receipt_binding: dict[str, Any] | None = None
    if support_receipt_path is not None:
        support_path = Path(support_receipt_path).expanduser().resolve()
        support_document = _load_json(support_path, "support receipt v4")
        _require(
            support_document.get("source_sha256") == sources,
            "support receipt v4 source bindings differ from the live tree; "
            "rebuild both receipts together",
        )
        support_receipt_binding = {
            "path": str(support_path),
            "sha256": _sha256_file(support_path, "support receipt v4"),
        }
    document: dict[str, Any] = {
        "schema_version": ASSEMBLY_SCHEMA_VERSION,
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        "receipt_version": ASSEMBLY_RECEIPT_VERSION,
        "architecture_lock": ARCHITECTURE_LOCK,
        "project_root": str(root),
        "source_sha256": sources,
        "transitive_source_sha256": transitive,
        "namespace_packages": namespaces,
        "tests_sha256": tests,
        "approval_sha256": approvals,
        "adjudication_binding": adjudication,
        "adjudication_amendment_binding": amendment,
        "lambda_freeze_binding": lambda_freeze_binding,
        "data_binding": {
            "train": {
                "path": FROZEN_TRAIN_RELATIVE.as_posix(),
                "rows": FROZEN_TRAIN_ROWS,
                "sha256": train_sha,
            },
            "prompt_normalization_erratum_v4": prompt_binding,
        },
        "asset_binding": dict(asset_binding),
        "cache_binding_mode": CACHE_BINDING_MODE,
        "cache_binding_reason": CACHE_BINDING_REASON,
        "controller_binding": bind_controller_identity(
            sources["f2_experiment/controller.py"]
        ),
        "support_receipt_binding": support_receipt_binding,
        "gate_contract_changes": list(GATE_CONTRACT_CHANGES),
        "optimizer_contract": dict(OPTIMIZER_CONTRACT),
        "probe_surface": G6_PROBE_SURFACE,
        "block_mode": G6_BLOCK_MODE,
        "smoke_package": SMOKE_PACKAGE,
        "g7_g9_arm_policy": G7_G9_ARM_POLICY,
        "eval_mode_contract": dict(EVAL_MODE_CONTRACT),
        "lambda_policy": dict(LAMBDA_POLICY),
        "lifecycle_order": list(LIFECYCLE_ORDER),
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    document["receipt_payload_sha256"] = canonical_json_sha256(document)
    return document


def freeze_assembly_receipt(
    project_root: str | Path,
    output: str | Path,
    *,
    asset_binding: Mapping[str, Any] | None = None,
    support_receipt_path: str | Path | None = None,
    lambda_freeze_receipt_path: str | Path | None = None,
) -> dict[str, str]:
    """Exclusive-write the receipt v4; refuses any existing destination."""

    document = build_assembly_receipt(
        project_root,
        asset_binding=asset_binding,
        support_receipt_path=support_receipt_path,
        lambda_freeze_receipt_path=lambda_freeze_receipt_path,
    )
    file_sha = _write_receipt(Path(output), document)
    return {
        "path": str(Path(output).expanduser().resolve()),
        "sha256": file_sha,
        "receipt_payload_sha256": document["receipt_payload_sha256"],
    }


def verify_assembly_receipt(
    project_root: str | Path,
    receipt_path: str | Path,
    *,
    asset_verifier: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-verify every recomputable binding of a frozen receipt v4.

    Every ``run_*`` lifecycle entry point calls this first (HS12: a real
    assembly not bound by a fresh receipt forbids any smoke).  Heavy asset
    payload verification stays with ``assembly_data.verify_frozen_assets``;
    pass it as ``asset_verifier`` to also re-check the asset binding here.
    """

    root = Path(project_root).expanduser().resolve()
    path = Path(receipt_path).expanduser().resolve()
    document = _load_json(path, "assembly receipt")
    _require(
        document.get("analysis_class") == ASSEMBLY_RECEIPT_CLASS,
        "assembly receipt has the wrong analysis class",
    )
    _require(
        document.get("schema_version") == ASSEMBLY_SCHEMA_VERSION
        and document.get("receipt_version") == ASSEMBLY_RECEIPT_VERSION,
        "assembly receipt schema/receipt version mismatch",
    )
    _require(
        document.get("architecture_lock") == ARCHITECTURE_LOCK,
        "assembly receipt architecture lock mismatch",
    )
    _require(
        document.get("internal_test") == INTERNAL_TEST_POLICY
        and document.get("internal_test_opened") is False,
        "assembly receipt internal-test seal is broken",
    )
    payload = dict(document)
    stored_payload_sha = payload.pop("receipt_payload_sha256", None)
    _require(
        stored_payload_sha == canonical_json_sha256(payload),
        "assembly receipt payload SHA does not match its own content",
    )
    _require(
        document.get("source_sha256") == source_bindings(root),
        "assembly receipt source bindings do not match the tree",
    )
    _require(
        document.get("transitive_source_sha256")
        == transitive_source_bindings(root),
        "assembly receipt transitive source bindings do not match the tree",
    )
    _require(
        document.get("tests_sha256") == _test_bindings(root),
        "assembly receipt test bindings do not match the tree",
    )
    _require(
        document.get("namespace_packages") == _verify_namespace_packages(root),
        "assembly receipt namespace package declaration mismatch",
    )
    verify_approval_files(root)
    _require(
        document.get("adjudication_binding")
        == _frozen_file_binding(
            root,
            ADJUDICATION_RELATIVE,
            ADJUDICATION_SHA256,
            "merged adjudication receipt",
        ),
        "assembly receipt adjudication binding mismatch",
    )
    _require(
        document.get("adjudication_amendment_binding") == _amendment_binding(root),
        "assembly receipt adjudication amendment binding mismatch",
    )
    recorded_freeze = document.get("lambda_freeze_binding")
    if recorded_freeze is not None:
        _require(
            isinstance(recorded_freeze, Mapping),
            "assembly receipt lambda freeze binding is malformed",
        )
        freeze_path = _resolve_bound_path(
            root, recorded_freeze.get("path"), "lambda freeze"
        )
        freeze_sha = _sha256_file(freeze_path, "lambda freeze receipt")
        _require(
            freeze_sha == recorded_freeze.get("sha256"),
            "lambda freeze receipt bytes differ from the assembly receipt "
            "binding",
        )
        freeze_document = _load_json(freeze_path, "lambda freeze receipt")
        _require(
            freeze_document.get("analysis_class") == LAMBDA_FREEZE_CLASS,
            "bound lambda freeze receipt has the wrong analysis class",
        )
        _require(
            document.get("approval_sha256", {}).get("fable_f2_lambda_freeze")
            == freeze_sha,
            "lambda freeze receipt is missing from the approvals binding",
        )
    data_binding = document.get("data_binding")
    _require(
        isinstance(data_binding, Mapping),
        "assembly receipt data binding is malformed",
    )
    train_sha = _sha256_file(root / FROZEN_TRAIN_RELATIVE, "frozen train JSONL")
    _require(
        train_sha == FROZEN_TRAIN_SHA256
        and data_binding.get("train")
        == {
            "path": FROZEN_TRAIN_RELATIVE.as_posix(),
            "rows": FROZEN_TRAIN_ROWS,
            "sha256": FROZEN_TRAIN_SHA256,
        },
        "assembly receipt train binding mismatch",
    )
    _require(
        data_binding.get("prompt_normalization_erratum_v4")
        == _frozen_file_binding(
            root,
            PROMPT_ERRATUM_RELATIVE,
            PROMPT_ERRATUM_SHA256,
            "prompt normalization erratum v4",
        ),
        "assembly receipt prompt erratum binding mismatch",
    )
    _require(
        document.get("controller_binding")
        == bind_controller_identity(
            source_bindings(root)["f2_experiment/controller.py"]
        ),
        "assembly receipt controller binding mismatch",
    )
    _require(
        document.get("cache_binding_mode") == CACHE_BINDING_MODE,
        "assembly receipt cache binding mode must be "
        f"{CACHE_BINDING_MODE!r} (internal-test seal adjudication)",
    )
    asset_binding = document.get("asset_binding")
    if isinstance(asset_binding, Mapping):
        _assert_sealed_cache_binding(asset_binding)
        _assert_token_ledger_anchor(asset_binding)
    else:
        raise F2AssemblyContractError(
            "assembly receipt asset binding is malformed"
        )
    if asset_verifier is not None:
        observed_assets = asset_verifier(root)
        _require(
            isinstance(observed_assets, Mapping)
            and dict(observed_assets) == dict(document.get("asset_binding", {})),
            "assembly receipt asset binding does not match the live assets",
        )
    return document


# ---------------------------------------------------------------------------
# Frozen support ordering and reset derivation
# ---------------------------------------------------------------------------


def _validate_support_rows(
    rows: Sequence[RunnerRow], *, support_name: str, label: str
) -> tuple[RunnerRow, ...]:
    expectation = SUPPORT_EXPECTATIONS.get(support_name)
    _require(
        expectation is not None,
        f"{label}: unknown frozen support {support_name!r}",
    )
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
        f"{label} must be an ordered sequence",
    )
    frozen = tuple(rows)
    _require(
        len(frozen) == expectation.rows,
        f"{label} must hold exactly {expectation.rows} rows, "
        f"got {len(frozen)}",
    )
    _require(
        all(isinstance(row, RunnerRow) for row in frozen),
        f"{label} rows must all be RunnerRow instances",
    )
    indices = [row.original_row_index for row in frozen]
    _require(
        all(right > left for left, right in zip(indices, indices[1:])),
        f"{label} rows must be strictly ordered by original_row_index",
    )
    return frozen


def build_support_reset_plan(
    rows: Sequence[RunnerRow],
    strafe_reset_original_indices: frozenset[int],
) -> tuple[tuple[str, ...], ...]:
    """Frozen reset predicate, byte-aligned with the paired runner."""

    plan: list[tuple[str, ...]] = []
    previous: RunnerRow | None = None
    for row in rows:
        reasons: list[str] = []
        if previous is None:
            reasons.append("stream_first")
        elif not continues_sequence(
            previous.sequence_mapping(), row.sequence_mapping()
        ):
            reasons.append("sequence_discontinuity")
        if row.original_row_index in strafe_reset_original_indices:
            reasons.append("strafe_reset")
        plan.append(tuple(reasons))
        previous = row
    return tuple(plan)


def derive_static_reset_receipt(
    rows: Sequence[RunnerRow],
    *,
    support_name: str,
    strafe_reset_original_indices: frozenset[int],
) -> tuple[tuple[tuple[str, ...], ...], frozenset[int]]:
    """Derive the reset plan and enforce the frozen static-reset count."""

    plan = build_support_reset_plan(rows, strafe_reset_original_indices)
    observed = frozenset(
        row.original_row_index
        for row, reasons in zip(rows, plan)
        if reasons
    )
    expected_count = SUPPORT_EXPECTATIONS[support_name].static_resets
    _require(
        len(observed) == expected_count,
        f"{support_name} static resets must equal the frozen count "
        f"{expected_count}, derived {len(observed)}",
    )
    return plan, observed


# ---------------------------------------------------------------------------
# CAL zero-update audit (adjudication rulings a/b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalRowAudit:
    """Per-row CAL evidence produced by the model-side auditor seam.

    ``step0_parity`` covers HS6 (raw two-axis persistence and zero strafe);
    ``prev_free`` covers the prev-free graph assertion; the norms are the
    base.proj probe-surface gradient norms for the frozen aux blocks and the
    track loss.
    """

    step0_parity: bool
    prev_free: bool
    aux_grad_norms: Mapping[str, float]
    track_grad_norm: float


def _validate_cal_context_receipt(
    value: Any,
    *,
    package: str,
    probe_surface: str,
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping),
        "CAL context provider must return a mapping",
    )
    seed = value.get("seed")
    device = value.get("device")
    context_package = value.get("package")
    context_probe = value.get("probe_surface")
    initialization = value.get("initialization")
    checkpoint_sha = value.get("checkpoint_init_sha256")
    cuda_reproducibility = value.get("cuda_reproducibility")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "CAL context seed must be a nonnegative integer",
    )
    _require(
        isinstance(device, str) and bool(device),
        "CAL context device must be a nonempty string",
    )
    _require(
        context_package == package,
        "CAL context package differs from the requested audit package",
    )
    _require(
        context_probe == probe_surface,
        "CAL context probe surface differs from the assembly receipt",
    )
    _require(
        isinstance(initialization, str) and bool(initialization),
        "CAL context initialization contract must be a nonempty string",
    )
    _require(
        isinstance(checkpoint_sha, str)
        and len(checkpoint_sha) == 64
        and all(char in "0123456789abcdef" for char in checkpoint_sha),
        "CAL context checkpoint_init_sha256 must be lowercase SHA-256 hex",
    )
    if device.startswith("cuda"):
        try:
            cuda_reproducibility = validate_cuda_reproducibility_receipt(
                cuda_reproducibility
            )
        except F2CudaReproducibilityError as exc:
            raise F2AssemblyContractError(str(exc)) from exc
    else:
        _require(
            cuda_reproducibility is None,
            "non-CUDA CAL context must not claim CUDA reproducibility settings",
        )
    normalized = {
        "seed": seed,
        "device": device,
        "package": context_package,
        "probe_surface": context_probe,
        "initialization": initialization,
        "checkpoint_init_sha256": checkpoint_sha,
    }
    if cuda_reproducibility is not None:
        normalized["cuda_reproducibility"] = cuda_reproducibility
    return normalized


def run_cal_audit(
    project_root: str | Path,
    *,
    receipt_path: str | Path,
    output_dir: str | Path,
    rows_loader: Callable[..., Any] | None = None,
    row_auditor: Callable[..., CalRowAudit] | None = None,
    token_ledger: Any | None = None,
    cal_context_provider: Callable[[], Mapping[str, Any]] | None = None,
    package: str = SMOKE_PACKAGE,
    verifier: Callable[..., Mapping[str, Any]] = verify_assembly_receipt,
) -> dict[str, Any]:
    """CAL 512-row zero-update audit: HS6 parity, prev-free graph audit,
    gradient calibration, and the frozen lambda proposal.

    No optimizer object is ever constructed here; the receipt records the
    zero-update proof, the G1-G5 legacy mapping (adjudication ruling a) and
    the lambda proposal (ruling b as amended by f2-adjudication-amendment-1:
    aux-relative calibration, freeze authority stays with Fable).  Per the
    amendment, the track-loss gradient on base.proj must be exactly zero on
    every CAL row (AP2 zero-init structural converse); any nonzero value is
    an ``AP2_ZERO_INIT_VIOLATION`` and fails closed.  Track reachability and
    the aux/track subordination bounds are enforced by the smoke G6 gate.
    """

    root = Path(project_root).expanduser().resolve()
    receipt_document = verifier(root, receipt_path)
    receipt_file_sha = _sha256_file(
        Path(receipt_path).expanduser().resolve(), "assembly receipt"
    )
    output = _ensure_empty_directory(Path(output_dir), "CAL audit output dir")
    if rows_loader is None:
        rows_loader = _default_support_rows_loader
    if token_ledger is None:
        token_ledger = getattr(rows_loader, "token_ledger", None)
    token_ledger = _resolve_frozen_token_ledger(
        root, receipt_document, token_ledger=token_ledger
    )
    ledger_sha, token_files = _ledger_identity(token_ledger)
    rows, _raw_rows, strafe = rows_loader(root, CAL_SUPPORT, token_ledger)
    rows = _validate_support_rows(
        rows, support_name=CAL_SUPPORT, label="CAL rows"
    )
    strafe_set = frozenset(int(value) for value in strafe)
    plan, observed = derive_static_reset_receipt(
        rows,
        support_name=CAL_SUPPORT,
        strafe_reset_original_indices=strafe_set,
    )
    if row_auditor is None:
        row_auditor = _integration_attr(
            "f2_experiment.assembly_model", "audit_cal_row"
        )
        if cal_context_provider is None:
            cal_context_provider = _integration_attr(
                "f2_experiment.assembly_model", "active_cal_context_receipt"
            )
    elif cal_context_provider is None:
        cal_context_provider = getattr(row_auditor, "context_receipt", None)
    _require(
        callable(cal_context_provider),
        "CAL context provider is required so seed/device/package/checkpoint "
        "initialization are frozen into the receipt",
    )

    aux_norms: dict[str, list[float]] = {name: [] for name in LAMBDA_AUX_LOSSES}
    cal_context: dict[str, Any] | None = None
    probe_surface = str(
        receipt_document.get("probe_surface", G6_PROBE_SURFACE)
    )
    for position, (row, reasons) in enumerate(zip(rows, plan)):
        audit = row_auditor(row, reasons, position)
        _require(
            isinstance(audit, CalRowAudit),
            f"CAL auditor must return CalRowAudit at position {position}",
        )
        if position == 0:
            cal_context = _validate_cal_context_receipt(
                cal_context_provider(),
                package=package,
                probe_surface=probe_surface,
            )
        _require(
            audit.step0_parity,
            f"HS6_STEP0_PARITY failed at CAL position {position}",
        )
        _require(
            audit.prev_free,
            f"PREV_GRAPH_LEAK at CAL position {position}",
        )
        _require(
            isinstance(audit.aux_grad_norms, Mapping)
            and set(audit.aux_grad_norms) == set(LAMBDA_AUX_LOSSES),
            f"CAL aux gradient blocks must be exactly {list(LAMBDA_AUX_LOSSES)!r} "
            f"at position {position}",
        )
        # Amendment 1: AP2 zero-init structural converse.  The track loss
        # cannot reach base.proj before the first optimizer step, so the
        # observed gradient norm must be exactly zero on every row; any
        # nonzero value disproves the frozen zero-init contract.
        track_norm = _finite_nonnegative(
            audit.track_grad_norm, f"CAL track grad norm[{position}]"
        )
        _require(
            track_norm == 0.0,
            "AP2_ZERO_INIT_VIOLATION: track gradient on base.proj must be "
            f"exactly zero at zero updates; observed {track_norm!r} at CAL "
            f"position {position}",
        )
        for name in LAMBDA_AUX_LOSSES:
            aux_norms[name].append(
                _finite_nonnegative(
                    audit.aux_grad_norms[name],
                    f"CAL {name} grad norm[{position}]",
                )
            )

    aux_medians: dict[str, float] = {}
    for name in LAMBDA_AUX_LOSSES:
        aux_median = float(median(aux_norms[name]))
        _require(
            aux_median > 0.0,
            f"CAL {name} gradient median is zero; the aux block is "
            "unreachable on the calibration support",
        )
        aux_medians[name] = aux_median
    _require(cal_context is not None, "CAL context was never initialized")
    final_cal_context = _validate_cal_context_receipt(
        cal_context_provider(), package=package, probe_surface=probe_surface
    )
    _require(
        final_cal_context == cal_context,
        "CAL context changed during the zero-update audit",
    )
    # Amended mechanism (f2-adjudication-amendment-1):
    # lambda_i = 0.5 * min_j(median||g_aux_j||) / median||g_aux_i||,
    # each capped at 1.0, rounded to 3 significant digits.  The weakest
    # auxiliary receives exactly 0.5.
    aux_median_min = min(aux_medians.values())
    proposed_lambda = {
        name: _round_sig(
            min(
                LAMBDA_TARGET_FRACTION * aux_median_min / aux_median,
                LAMBDA_UPPER_BOUND,
            )
        )
        for name, aux_median in aux_medians.items()
    }

    receipt = {
        "schema_version": 1,
        "analysis_class": CAL_AUDIT_RECEIPT_CLASS,
        "architecture_lock": ARCHITECTURE_LOCK,
        "package": package,
        "support": CAL_SUPPORT,
        "rows": len(rows),
        "assembly_receipt_sha256": receipt_file_sha,
        "assembly_receipt_payload_sha256": receipt_document[
            "receipt_payload_sha256"
        ],
        "token_ledger_binding": {
            "anchor": TOKEN_LEDGER_ANCHOR,
            "sha256": ledger_sha,
            "file_count": token_files,
        },
        "cal_context": cal_context,
        "optimizer_updates": 0,
        "zero_update_proof": (
            "no optimizer object is constructed anywhere inside run_cal_audit"
        ),
        "step0_parity": {
            "support": CAL_SUPPORT,
            "checked_rows": len(rows),
            "failures": 0,
            "contract": (
                "assert_step0_controlled_axis_persistence + zero strafe "
                "(corrigendum-2 frozen parity support = CAL)"
            ),
        },
        "prev_free_graph_audit": {
            "checked_rows": len(rows),
            "failures": 0,
            "contract": "assert_prev_free_tensors on method/base features",
        },
        "ap2_zero_init_proof": {
            "checked_rows": len(rows),
            "violations": 0,
            "track_grad_norm_max": 0.0,
            "contract": (
                "track-loss gradient on base.proj is exactly zero on every "
                "CAL row at zero updates (AP2 zero-init structural converse, "
                "f2-adjudication-amendment-1); nonzero -> "
                "AP2_ZERO_INIT_VIOLATION fail-closed"
            ),
            "track_reachability_enforcement": (
                "smoke G6 gate (track reachable >= 120/128, gradient window "
                "u_pre 8..127, bstar median aux/track <= 1.5, per-aux "
                "fallback <= 0.75)"
            ),
        },
        "amendment_binding": {
            "path": ADJUDICATION_AMENDMENT1_RELATIVE.as_posix(),
            "sha256": ADJUDICATION_AMENDMENT1_SHA256,
            "amendment_id": ADJUDICATION_AMENDMENT1_ID,
            "amends": ADJUDICATION_AMENDMENT1_AMENDS,
            "preregistration_status": (
                "amended before the CAL audit was executed, before any smoke "
                "optimizer update, and before any dev evaluation was seen"
            ),
        },
        "static_reset_receipt": {
            "expected": SUPPORT_EXPECTATIONS[CAL_SUPPORT].static_resets,
            "observed": len(observed),
            "original_indices_sha256": canonical_json_sha256(sorted(observed)),
            "strafe_intersection": sorted(
                strafe_set & {row.original_row_index for row in rows}
            ),
        },
        "gradient_calibration": {
            "probe_surface": receipt_document.get(
                "probe_surface", G6_PROBE_SURFACE
            ),
            "per_aux_grad_norm_median": aux_medians,
            "aux_grad_norm_median_min": aux_median_min,
        },
        "lambda_calibration": {
            "mechanism": LAMBDA_MECHANISM,
            "amendment_id": ADJUDICATION_AMENDMENT1_ID,
            "target_fraction": LAMBDA_TARGET_FRACTION,
            "upper_bound": LAMBDA_UPPER_BOUND,
            "significant_digits": LAMBDA_SIGNIFICANT_DIGITS,
            "proposed_lambda": proposed_lambda,
            "status": "proposal",
            "freeze_authority": (
                "Fable 5 PRIMARY; values freeze before any smoke optimizer "
                "update (adjudication ruling b as amended by "
                "f2-adjudication-amendment-1)"
            ),
        },
        "g_legacy_map": dict(G_LEGACY_MAP),
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    receipt_path_out = output / "cal_audit_receipt_v1.json"
    file_sha = _write_receipt(receipt_path_out, receipt)
    return {"receipt": receipt, "path": str(receipt_path_out), "sha256": file_sha}


# ---------------------------------------------------------------------------
# EVAL-FIX dual-mode executor
# ---------------------------------------------------------------------------


def _validate_eval_prediction(prediction: Any, label: str) -> AP2Prediction:
    _require(
        isinstance(prediction, AP2Prediction),
        f"{label} must be an AP2Prediction",
    )
    _require(
        prediction.raw_actions.shape == (1, AP2_HORIZON, 3),
        f"{label}.raw_actions shape mismatch",
    )
    _require(
        prediction.delta_fy.shape == (1, AP2_HORIZON, 2),
        f"{label}.delta_fy shape mismatch",
    )
    for name in ("raw_actions", "delta_fy"):
        tensor = getattr(prediction, name)
        _require(
            bool(torch.isfinite(tensor).all().item()),
            f"{label}.{name} is nonfinite",
        )
    _require(
        int(torch.count_nonzero(prediction.raw_actions[..., 1]).item()) == 0,
        f"{label} predicted nonzero strafe",
    )
    return prediction


def _default_eval_loss(prediction: AP2Prediction, target: torch.Tensor) -> float:
    target = target.to(
        device=prediction.raw_actions.device,
        dtype=prediction.raw_actions.dtype,
    )
    return float(ap2_track_loss(prediction, target).total.detach().cpu().item())


def run_eval_fix(
    *,
    eval_rows: Sequence[RunnerRow],
    raw_rows: Sequence[Mapping[str, Any]],
    mode: EvalMode,
    predictor: Callable[..., AP2Prediction],
    strafe_reset_original_indices: frozenset[int],
    support_name: str = EVAL_SUPPORT,
    controller_config: ActionFilterConfig = DEFAULT_CONFIG,
    loss_fn: Callable[[AP2Prediction, torch.Tensor], float] | None = None,
) -> dict[str, Any]:
    """Run one EVAL-FIX pass in ``logged`` or ``self`` mode.

    ``self`` mode reproduces the runner branch2 recurrence byte-for-byte:
    reset rows re-seed prev and the controller from the logged prev action;
    all other rows reuse the previous row's controller-filtered sent action.
    """

    _require(mode in ("logged", "self"), f"unknown EVAL mode {mode!r}")
    rows = _validate_support_rows(
        eval_rows, support_name=support_name, label=f"{support_name} rows"
    )
    _require(
        isinstance(raw_rows, Sequence) and len(raw_rows) == len(rows),
        "raw JSONL rows must parallel the runner rows one-to-one",
    )
    strafe_set = frozenset(int(value) for value in strafe_reset_original_indices)
    plan, observed = derive_static_reset_receipt(
        rows,
        support_name=support_name,
        strafe_reset_original_indices=strafe_set,
    )
    if loss_fn is None:
        loss_fn = _default_eval_loss
    controller = ActionFilterController(controller_config)
    controller_state = None
    prev_fy: tuple[float, float] | None = None
    row_losses: list[float] = []
    with torch.no_grad():
        for position, (row, reasons) in enumerate(zip(rows, plan)):
            reset = bool(reasons)
            logged_fy = (row.logged_prev_action[0], row.logged_prev_action[2])
            if mode == "logged":
                prev = logged_fy
            elif reset:
                controller_state = controller.reset(row.logged_prev_action)
                prev = logged_fy
            else:
                _require(
                    prev_fy is not None and controller_state is not None,
                    f"self-mode recurrence reached position {position} "
                    "without an initializing reset",
                )
                prev = prev_fy
            prev_tensor = torch.tensor(
                [[prev[0], prev[1]]], dtype=torch.float32
            )
            prediction = _validate_eval_prediction(
                predictor(
                    row,
                    prev_tensor,
                    mode=mode,
                    reset=reset,
                    position=position,
                ),
                f"EVAL[{mode}][{position}]",
            )
            loss = loss_fn(prediction, row.target_actions.unsqueeze(0))
            loss_value = float(loss)
            _require(
                math.isfinite(loss_value) and loss_value >= 0.0,
                f"EVAL[{mode}][{position}] loss must be finite and nonnegative",
            )
            row_losses.append(loss_value)
            if mode == "self":
                k0 = prediction.raw_actions[0, 0, (0, 2)]
                _require(
                    bool(torch.isfinite(k0).all().item()),
                    f"EVAL_NONFINITE: self-mode k0 at position {position}",
                )
                controller_state, transition = controller.step(
                    controller_state,
                    (float(k0[0].item()), float(k0[1].item())),
                )
                prev_fy = transition.next_prev_fy
    summary = aggregate_row_losses(row_losses, raw_rows)
    return {
        "schema_version": 1,
        "analysis_class": EVAL_FIX_RECEIPT_CLASS,
        "architecture_lock": ARCHITECTURE_LOCK,
        "support": support_name,
        "rows": len(rows),
        "mode": mode,
        "static_resets": {
            "expected": SUPPORT_EXPECTATIONS[support_name].static_resets,
            "observed": len(observed),
        },
        "controller_config": controller_config.to_dict(),
        "eval_mode_contract": dict(EVAL_MODE_CONTRACT),
        "row_losses": row_losses,
        "summary": summary.to_dict(),
    }


def evaluate_snapshot(
    *,
    eval_rows: Sequence[RunnerRow],
    raw_rows: Sequence[Mapping[str, Any]],
    predictor: Callable[..., AP2Prediction],
    strafe_reset_original_indices: frozenset[int],
    arm: str,
    snapshot_label: str,
    support_name: str = EVAL_SUPPORT,
    controller_config: ActionFilterConfig = DEFAULT_CONFIG,
    loss_fn: Callable[[AP2Prediction, torch.Tensor], float] | None = None,
) -> dict[str, Any]:
    """Run both EVAL-FIX modes; the result feeds ``evaluate_g8`` directly."""

    results = {
        mode: run_eval_fix(
            eval_rows=eval_rows,
            raw_rows=raw_rows,
            mode=mode,
            predictor=predictor,
            strafe_reset_original_indices=strafe_reset_original_indices,
            support_name=support_name,
            controller_config=controller_config,
            loss_fn=loss_fn,
        )
        for mode in ("logged", "self")
    }
    return {
        "arm": arm,
        "snapshot": snapshot_label,
        "logged": results["logged"]["summary"],
        "self": results["self"]["summary"],
        "detail": results,
    }


# ---------------------------------------------------------------------------
# Fail-closed checkpoint save/load (no mid-run resume)
# ---------------------------------------------------------------------------


def save_arm_checkpoint(
    path: str | Path,
    *,
    arm: str,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Any,
    u_pre: int,
    assembly_receipt_sha256: str,
) -> dict[str, Any]:
    """Save one arm checkpoint plus an exclusive-write sidecar receipt."""

    destination = Path(path).expanduser().resolve()
    _require(
        not destination.exists(),
        f"refusing to overwrite an existing checkpoint: {destination}",
    )
    _require(arm in ARM_ORDER, f"unknown arm {arm!r}")
    _require(
        isinstance(u_pre, int) and not isinstance(u_pre, bool) and u_pre >= 0,
        "u_pre must be a nonnegative integer",
    )
    state_sha = checkpoint_init_sha256(model_state)
    payload = {
        "model": dict(model_state),
        "optimizer": optimizer_state,
        "u_pre": int(u_pre),
        "arm": arm,
        "assembly_receipt_sha256": assembly_receipt_sha256,
        "checkpoint_init_sha256": state_sha,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "xb") as handle:
        torch.save(payload, handle)
    file_sha = _sha256_file(destination, "arm checkpoint")
    sidecar_path = destination.with_name(destination.name + ".receipt.json")
    sidecar = {
        "schema_version": 1,
        "analysis_class": CHECKPOINT_RECEIPT_CLASS,
        "architecture_lock": ARCHITECTURE_LOCK,
        "arm": arm,
        "u_pre": int(u_pre),
        "checkpoint_file": destination.name,
        "checkpoint_file_sha256": file_sha,
        "checkpoint_init_sha256": state_sha,
        "assembly_receipt_sha256": assembly_receipt_sha256,
        "resume_policy": (
            "no mid-run resume; a failed smoke burns its directory and any "
            "rerun starts from a fresh directory and fresh receipts"
        ),
    }
    _write_receipt(sidecar_path, sidecar)
    return {
        "path": str(destination),
        "file_sha256": file_sha,
        "state_sha256": state_sha,
        "sidecar": str(sidecar_path),
    }


def load_arm_checkpoint_verified(
    path: str | Path,
    *,
    expected_assembly_receipt_sha256: str,
    expected_arm: str | None = None,
    expected_u_pre: int | None = None,
    expected_state_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load a checkpoint; any missing key or SHA mismatch fails closed."""

    destination = Path(path).expanduser().resolve()
    _require(destination.is_file(), f"arm checkpoint is missing: {destination}")
    sidecar_path = destination.with_name(destination.name + ".receipt.json")
    sidecar = _load_json(sidecar_path, "arm checkpoint sidecar receipt")
    file_sha = _sha256_file(destination, "arm checkpoint")
    _require(
        sidecar.get("checkpoint_file_sha256") == file_sha,
        "arm checkpoint bytes do not match the sidecar receipt",
    )
    try:
        payload = torch.load(
            destination, map_location="cpu", weights_only=True
        )
    except Exception as exc:  # noqa: BLE001 - any load failure fails closed
        raise F2AssemblyContractError(
            f"arm checkpoint cannot be deserialized: {destination}"
        ) from exc
    _require(isinstance(payload, dict), "arm checkpoint payload must be a dict")
    missing = [key for key in CHECKPOINT_REQUIRED_KEYS if key not in payload]
    _require(
        not missing,
        f"arm checkpoint is missing required keys: {missing!r}",
    )
    _require(
        payload["assembly_receipt_sha256"] == expected_assembly_receipt_sha256,
        "arm checkpoint is bound to a different assembly receipt",
    )
    recomputed = checkpoint_init_sha256(payload["model"])
    _require(
        recomputed == payload["checkpoint_init_sha256"]
        and recomputed == sidecar.get("checkpoint_init_sha256"),
        "arm checkpoint tensor state SHA mismatch",
    )
    if expected_arm is not None:
        _require(
            payload["arm"] == expected_arm,
            f"arm checkpoint belongs to {payload['arm']!r}, "
            f"expected {expected_arm!r}",
        )
    if expected_u_pre is not None:
        _require(
            payload["u_pre"] == expected_u_pre,
            f"arm checkpoint u_pre {payload['u_pre']!r} does not match "
            f"expected {expected_u_pre!r}",
        )
    if expected_state_keys is not None:
        _require(
            set(payload["model"]) == set(expected_state_keys),
            "arm checkpoint tensor set does not exactly match the model",
        )
    return payload


# ---------------------------------------------------------------------------
# G7 prev-scale presence (HS8) and both-arm gate policy
# ---------------------------------------------------------------------------


def assert_g7_updates_carry_prev_scale(
    updates: Sequence[RunnerG7Update],
) -> None:
    """HS8 guard: every update must carry ``abs_tanh_s_prev`` in [0,1]."""

    _require(bool(updates), "G7 updates are empty")
    for index, update in enumerate(updates):
        value = getattr(update, "abs_tanh_s_prev", None)
        if value is None:
            raise F2AssemblyContractError(
                f"HS8_G7_PREV_SCALE_MISSING: update {index} lacks "
                "abs_tanh_s_prev"
            )
        number = float(value)
        _require(
            math.isfinite(number) and 0.0 <= number <= 1.0,
            f"HS8_G7_PREV_SCALE_INVALID: update {index} abs_tanh_s_prev "
            "must lie in [0,1]",
        )


def _combined_arm_receipt(receipts: Mapping[str, GateReceipt]) -> GateReceipt:
    """Both-arm AND policy: a failing arm's receipt always represents the
    gate in the combined receipt, so the combination can never be weaker
    than either arm (plan section 4 note)."""

    if not receipts[S_SELF].passed:
        return receipts[S_SELF]
    if not receipts[S_CTRL].passed:
        return receipts[S_CTRL]
    return receipts[S_SELF]


def _g6_update_to_dict(update: G6Update) -> dict[str, Any]:
    return {
        "u_pre": update.u_pre,
        "aux_reachable": update.aux_reachable,
        "track_reachable": update.track_reachable,
        "cosine_total_track": update.cosine_total_track,
        "signed_projection": update.signed_projection,
        "aux_track_ratio": update.aux_track_ratio,
        "per_aux_ratios": (
            None
            if update.per_aux_ratios is None
            else dict(update.per_aux_ratios)
        ),
    }


def _evaluate_g6_with_fallback(
    updates: Sequence[G6Update],
    *,
    block_mode: str,
    fallback_evidence: Any,
) -> tuple[GateReceipt, GateReceipt, dict[str, Any]]:
    """Evaluate deciding B* plus non-deciding per-aux evidence together."""

    primary = evaluate_g6(updates, block_mode=block_mode)
    _require(
        block_mode == "bstar",
        "production G6 must use bstar as the deciding block mode",
    )
    _require(
        isinstance(fallback_evidence, Mapping),
        "G6 fallback evidence provider must return a mapping",
    )
    _require(
        fallback_evidence.get("deciding_block_mode") == "bstar"
        and fallback_evidence.get("block_mode") == "bstar",
        "G6 fallback evidence must declare bstar as the deciding mode",
    )
    series = fallback_evidence.get("per_aux_ratio_series")
    _require(
        isinstance(series, Sequence) and not isinstance(series, (str, bytes)),
        "G6 fallback evidence must contain an ordered per_aux_ratio_series",
    )
    ratios_by_clock: dict[int, Mapping[str, float]] = {}
    for index, entry in enumerate(series):
        _require(
            isinstance(entry, Mapping),
            f"G6 fallback evidence entry {index} must be a mapping",
        )
        u_pre = entry.get("u_pre")
        ratios = entry.get("ratios")
        _require(
            isinstance(u_pre, int)
            and not isinstance(u_pre, bool)
            and u_pre not in ratios_by_clock,
            f"G6 fallback evidence entry {index} has an invalid/duplicate clock",
        )
        _require(
            isinstance(ratios, Mapping) and bool(ratios),
            f"G6 fallback evidence entry {index} has no per-aux ratios",
        )
        ratios_by_clock[u_pre] = ratios
    expected_clocks = {update.u_pre for update in updates}
    _require(
        set(ratios_by_clock) == expected_clocks,
        "G6 fallback evidence clocks must exactly match the 128 smoke updates",
    )
    fallback_updates = [
        G6Update(
            u_pre=update.u_pre,
            aux_reachable=update.aux_reachable,
            track_reachable=update.track_reachable,
            cosine_total_track=update.cosine_total_track,
            signed_projection=update.signed_projection,
            per_aux_ratios=dict(ratios_by_clock[update.u_pre]),
        )
        for update in updates
    ]
    fallback = evaluate_g6(fallback_updates, block_mode="per_aux")
    document = primary.to_dict()
    document["contract"] = {
        **document["contract"],
        "deciding_block_mode": "bstar",
        "fallback_role": "evaluated_and_reported_non_deciding",
    }
    document["fallback_per_aux"] = {
        "role": "reported_non_deciding",
        "passed": fallback.passed,
        "status": "PASS" if fallback.passed else "FAIL",
        "checks": {name: dict(value) for name, value in fallback.checks.items()},
        "metrics": dict(fallback.metrics),
        "thresholds": dict(fallback.thresholds),
        "contract": dict(fallback.contract),
    }
    return primary, fallback, document


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Production smoke orchestration (handoff section 14.3 order)
# ---------------------------------------------------------------------------


def _compare_asset_bindings(observed: Any, recorded: Any) -> None:
    """P1-3: the live asset verification must match the receipt binding.

    The comparison is field-by-field over the union of top-level sections;
    any missing or differing field fails closed before any plan (and thus
    any base-checkpoint path) is constructed from the receipt.
    """

    _require(
        isinstance(observed, Mapping) and bool(observed),
        "live asset verification returned no binding",
    )
    _require(
        isinstance(recorded, Mapping) and bool(recorded),
        "assembly receipt carries no asset binding",
    )
    for field in sorted(set(observed) | set(recorded)):
        if (
            field not in observed
            or field not in recorded
            or _json_safe(observed[field]) != _json_safe(recorded[field])
        ):
            raise F2AssemblyContractError(
                "ASSET_BINDING_MISMATCH: live asset verification differs "
                f"from the receipt asset binding at field {field!r}; the "
                "assembly is not the bound one and smoke is forbidden"
            )


def verify_cal_lambda_authority(
    project_root: str | Path,
    *,
    cal_audit_receipt_path: str | Path,
    receipt_document: Mapping[str, Any],
    receipt_file_sha256: str,
    aux_coefficients: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """P1-1: machine-verify the CAL -> lambda-freeze -> receipt authority chain.

    Checks, all fail-closed: the CAL receipt class and zero-update/row
    contract, its amendment binding, the PRIMARY lambda-freeze receipt bound
    inside the current assembly receipt (bytes re-hashed), that the freeze
    adopts exactly this CAL receipt plus a distinct byte-identical
    reproduction, that both runs bind the same sealed bootstrap receipt and
    copied seed/device/init metadata, that the CAL receipt is either bound to
    the current assembly receipt directly or traceable through the bound
    freeze (supersede chain), and that CAL proposal, freeze values, and the
    ``assembly_model.FROZEN_AUX_COEFFICIENTS`` source literals agree value by
    value.  No lambda value is ever hardcoded here.
    """

    root = Path(project_root).expanduser().resolve()
    cal_path = Path(cal_audit_receipt_path).expanduser().resolve()
    cal_document = _load_json(cal_path, "CAL audit receipt")
    cal_file_sha = _sha256_file(cal_path, "CAL audit receipt")
    _require(
        cal_document.get("analysis_class") == CAL_AUDIT_RECEIPT_CLASS,
        "CAL audit receipt has the wrong analysis class",
    )
    _require(
        cal_document.get("optimizer_updates") == 0,
        "CAL audit receipt must prove zero optimizer updates",
    )
    _require(
        cal_document.get("rows") == SUPPORT_EXPECTATIONS[CAL_SUPPORT].rows,
        "CAL audit receipt row count differs from the frozen CAL support",
    )
    _require(
        cal_document.get("package")
        == receipt_document.get("smoke_package", SMOKE_PACKAGE),
        "CAL audit receipt package differs from the smoke package",
    )
    cal_context = _validate_cal_context_receipt(
        cal_document.get("cal_context"),
        package=str(receipt_document.get("smoke_package", SMOKE_PACKAGE)),
        probe_surface=str(
            receipt_document.get("probe_surface", G6_PROBE_SURFACE)
        ),
    )
    cal_ledger = cal_document.get("token_ledger_binding")
    asset_binding = receipt_document.get("asset_binding")
    _require(
        isinstance(cal_ledger, Mapping)
        and isinstance(asset_binding, Mapping)
        and cal_ledger.get("anchor") == TOKEN_LEDGER_ANCHOR
        and cal_ledger.get("sha256")
        == asset_binding.get("token_ledger_sha256")
        and cal_ledger.get("file_count")
        == asset_binding.get("token_ledger_file_count"),
        "CAL token ledger binding differs from the assembly receipt anchor",
    )
    cal_amendment = cal_document.get("amendment_binding")
    _require(
        isinstance(cal_amendment, Mapping)
        and cal_amendment.get("sha256") == ADJUDICATION_AMENDMENT1_SHA256,
        "CAL audit receipt amendment binding mismatch",
    )
    cal_lambda = cal_document.get("lambda_calibration")
    _require(
        isinstance(cal_lambda, Mapping)
        and cal_lambda.get("mechanism") == LAMBDA_MECHANISM,
        "CAL audit receipt lambda mechanism differs from the amended "
        "mechanism",
    )
    proposed = cal_lambda.get("proposed_lambda") if isinstance(cal_lambda, Mapping) else None
    _require(
        isinstance(proposed, Mapping)
        and set(proposed) == set(LAMBDA_AUX_LOSSES),
        "CAL audit receipt lambda proposal does not cover the frozen aux "
        "losses",
    )

    recorded_freeze = receipt_document.get("lambda_freeze_binding")
    _require(
        isinstance(recorded_freeze, Mapping),
        "assembly receipt does not bind a PRIMARY lambda freeze receipt; "
        "smoke is forbidden until the freeze is bound into the approvals "
        "chain",
    )
    freeze_path = _resolve_bound_path(
        root, recorded_freeze.get("path"), "lambda freeze"
    )
    freeze_file_sha = _sha256_file(freeze_path, "lambda freeze receipt")
    _require(
        freeze_file_sha == recorded_freeze.get("sha256"),
        "lambda freeze receipt bytes differ from the assembly receipt "
        "binding",
    )
    freeze_document = _load_json(freeze_path, "lambda freeze receipt")
    _require(
        freeze_document.get("analysis_class") == LAMBDA_FREEZE_CLASS,
        "lambda freeze receipt has the wrong analysis class",
    )
    _require(
        str(freeze_document.get("mechanism", "")).startswith(LAMBDA_MECHANISM),
        "lambda freeze mechanism differs from the amended mechanism",
    )
    verified_freeze_evidence = _verify_lambda_freeze_evidence(
        root,
        freeze_document,
        expected_cal_path=cal_path,
        freeze_file_sha256=freeze_file_sha,
    )
    evidence = freeze_document.get("evidence")
    adopted = (
        evidence.get("cal_audit_receipt") if isinstance(evidence, Mapping) else None
    )
    _require(
        isinstance(adopted, Mapping) and adopted.get("sha256") == cal_file_sha,
        "the PRIMARY lambda freeze receipt does not adopt this CAL audit "
        "receipt (authority chain broken)",
    )
    direct = (
        cal_document.get("assembly_receipt_sha256") == receipt_file_sha256
        or cal_document.get("assembly_receipt_payload_sha256")
        == receipt_document.get("receipt_payload_sha256")
    )
    chain = "direct" if direct else "via_lambda_freeze_supersede"

    frozen_values = freeze_document.get("frozen_values")
    _require(
        isinstance(frozen_values, Mapping)
        and set(frozen_values) == set(LAMBDA_AUX_LOSSES),
        "lambda freeze receipt frozen_values do not cover the frozen aux "
        "losses",
    )
    if aux_coefficients is None:
        try:
            model_module = importlib.import_module(
                "f2_experiment.assembly_model"
            )
        except ImportError as exc:
            raise F2AssemblyContractError(
                "assembly_model is unavailable; FROZEN_AUX_COEFFICIENTS "
                "cannot be checked and smoke is forbidden"
            ) from exc
        aux_coefficients = getattr(
            model_module, "FROZEN_AUX_COEFFICIENTS", None
        )
    _require(
        isinstance(aux_coefficients, Mapping) and bool(aux_coefficients),
        "assembly_model.FROZEN_AUX_COEFFICIENTS is not frozen; smoke is "
        "forbidden before the PRIMARY lambda freeze is written as source "
        "literals",
    )
    _require(
        set(aux_coefficients) == set(LAMBDA_AUX_LOSSES),
        "FROZEN_AUX_COEFFICIENTS keys differ from the frozen aux losses",
    )
    frozen_lambda: dict[str, float] = {}
    for name in LAMBDA_AUX_LOSSES:
        proposal_value = float(proposed[name])
        freeze_value = float(frozen_values[name])
        literal_value = float(aux_coefficients[name])
        _require(
            proposal_value == freeze_value == literal_value,
            f"lambda value mismatch for {name}: CAL proposal "
            f"{proposal_value!r}, PRIMARY freeze {freeze_value!r}, source "
            f"literal {literal_value!r}; smoke is forbidden",
        )
        frozen_lambda[name] = literal_value
    return {
        "cal_audit_receipt": {"path": str(cal_path), "sha256": cal_file_sha},
        "lambda_freeze_receipt": {
            "path": str(freeze_path),
            "sha256": freeze_file_sha,
        },
        "cal_reproduction_receipt": {
            "path": str(
                verified_freeze_evidence["deterministic_reproduction"]["path"]
            ),
            "sha256": verified_freeze_evidence[
                "deterministic_reproduction"
            ]["sha256"],
            "byte_identical": True,
        },
        "bootstrap_assembly_receipt": {
            "path": str(
                verified_freeze_evidence["bootstrap_assembly_receipt"]["path"]
            ),
            "sha256": verified_freeze_evidence[
                "bootstrap_assembly_receipt"
            ]["sha256"],
            "receipt_payload_sha256": verified_freeze_evidence[
                "bootstrap_assembly_receipt"
            ]["receipt_payload_sha256"],
        },
        "assembly_receipt_chain": chain,
        "frozen_lambda": frozen_lambda,
        "mechanism": LAMBDA_MECHANISM,
        "amendment_id": ADJUDICATION_AMENDMENT1_ID,
        "cal_context": cal_context,
        "token_ledger_binding": dict(cal_ledger),
        "internal_test_opened": False,
    }


@dataclass(frozen=True)
class SmokeArmAssembly:
    """One arm's runner callbacks plus its EVAL-FIX/checkpoint accessors.

    ``checkpoint_payload`` must return a live ``{"model": state_dict,
    "optimizer": optimizer_state}`` mapping each time it is called; the
    orchestrator snapshots it at update 0 and update 128.
    """

    callbacks: ArmCallbacks
    eval_predictor: Callable[..., AP2Prediction]
    checkpoint_payload: Callable[[], Mapping[str, Any]]
    eval_loss_fn: Callable[[AP2Prediction, torch.Tensor], float] | None = None


@dataclass(frozen=True)
class SmokeAssemblyPlan:
    """Everything the lifecycle needs beyond the frozen receipts."""

    smoke_rows: tuple[RunnerRow, ...]
    eval_rows: tuple[RunnerRow, ...]
    eval_raw_rows: tuple[Mapping[str, Any], ...]
    strafe_reset_original_indices: frozenset[int]
    expected_static_reset_original_indices: frozenset[int]
    arms: Mapping[ArmName, SmokeArmAssembly]
    g6_update: Callable[[OptimizerUpdateEvent], G6Update]
    g6_fallback_evidence: Callable[[], Mapping[str, Any]] | None = None
    controller_config: ActionFilterConfig = DEFAULT_CONFIG
    # Reproducibility metadata recorded into the smoke summary (P3): the
    # paired-arm init seed and the torch device the plan was built for.
    seed: int | None = None
    device: str | None = None
    checkpoint_init_sha256: str | None = None
    cuda_reproducibility: Mapping[str, Any] | None = None


def _default_plan_builder(
    project_root: Path, receipt_document: Mapping[str, Any]
) -> SmokeAssemblyPlan:
    """Integration seam: ``assembly_model.build_production_smoke_plan``.

    The seam performs the plan-section-4 wiring (frozen support build,
    JSONL->RunnerRow loaders, bit-identical paired arms, real callbacks and
    the G6 instrument) and returns a :class:`SmokeAssemblyPlan`.
    """

    builder = _integration_attr(
        "f2_experiment.assembly_model", "build_production_smoke_plan"
    )
    plan = builder(project_root, receipt_document)
    _require(
        isinstance(plan, SmokeAssemblyPlan),
        "production plan builder must return a SmokeAssemblyPlan",
    )
    return plan


def _canonical_torch_device(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be nonempty")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise F2AssemblyContractError(f"{label} is invalid: {value!r}") from exc
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    return str(device)


def _verify_smoke_plan_cal_context(
    plan: SmokeAssemblyPlan,
    cal_context: Mapping[str, Any],
) -> None:
    _require(
        plan.seed == cal_context.get("seed"),
        "smoke plan seed differs from the authoritative CAL context",
    )
    plan_device = _canonical_torch_device(plan.device, "smoke plan device")
    cal_device = _canonical_torch_device(
        cal_context.get("device"), "CAL context device"
    )
    _require(
        plan_device == cal_device,
        "smoke plan device differs from the authoritative CAL context",
    )
    _require(
        plan.checkpoint_init_sha256
        == cal_context.get("checkpoint_init_sha256"),
        "smoke plan checkpoint init SHA differs from the authoritative CAL "
        "context",
    )
    if cal_device.startswith("cuda"):
        _require(
            plan.cuda_reproducibility
            == cal_context.get("cuda_reproducibility"),
            "smoke plan CUDA reproducibility settings differ from the "
            "authoritative CAL context",
        )
    else:
        _require(
            plan.cuda_reproducibility is None,
            "non-CUDA smoke plan must not claim CUDA reproducibility settings",
        )


def _checkpoint_payload_of(arm_assembly: SmokeArmAssembly, arm: str) -> dict[str, Any]:
    payload = arm_assembly.checkpoint_payload()
    _require(
        isinstance(payload, Mapping)
        and "model" in payload
        and "optimizer" in payload,
        f"{arm} checkpoint payload must provide 'model' and 'optimizer'",
    )
    return {"model": payload["model"], "optimizer": payload["optimizer"]}


def run_production_smoke(
    project_root: str | Path,
    *,
    receipt_path: str | Path,
    output_dir: str | Path,
    cal_audit_receipt_path: str | Path,
    plan_builder: Callable[..., SmokeAssemblyPlan] | None = None,
    verifier: Callable[..., Mapping[str, Any]] = verify_assembly_receipt,
    asset_verifier: Callable[[Path], Any] | None = None,
    aux_coefficients: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run the full 128-update causal smoke lifecycle in the frozen order.

    The CAL audit receipt is mandatory (P1-1): the CAL -> lambda-freeze ->
    assembly-receipt authority chain is machine-verified before any model
    is constructed, and the live asset verification must match the receipt
    asset binding field by field (P1-3) before the plan may load the bound
    base checkpoint.

    Any gate FAIL still writes every receipt, seals the negative result with
    ``formal_training_authorized=false`` and returns; it never retries,
    never tunes a gate.  Contract violations raise and leave the partially
    written directory behind as burn evidence; reruns must use a fresh
    directory.
    """

    root = Path(project_root).expanduser().resolve()
    output = _ensure_empty_directory(Path(output_dir), "smoke output dir")
    receipt_document = verifier(root, receipt_path)
    receipt_file_sha = _sha256_file(
        Path(receipt_path).expanduser().resolve(), "assembly receipt"
    )
    if asset_verifier is None:
        asset_verifier = _default_asset_binding
    observed_assets = asset_verifier(root)
    _compare_asset_bindings(
        observed_assets, receipt_document.get("asset_binding")
    )
    cal_authority = verify_cal_lambda_authority(
        root,
        cal_audit_receipt_path=cal_audit_receipt_path,
        receipt_document=receipt_document,
        receipt_file_sha256=receipt_file_sha,
        aux_coefficients=aux_coefficients,
    )
    if plan_builder is None:
        plan_builder = _default_plan_builder
    plan = plan_builder(root, receipt_document)
    _require(
        isinstance(plan, SmokeAssemblyPlan),
        "plan builder must return a SmokeAssemblyPlan",
    )
    _verify_smoke_plan_cal_context(plan, cal_authority["cal_context"])
    smoke_rows = _validate_support_rows(
        plan.smoke_rows, support_name=SMOKE_SUPPORT, label="SMK-TRAIN rows"
    )
    eval_rows = _validate_support_rows(
        plan.eval_rows, support_name=EVAL_SUPPORT, label="EVAL-FIX rows"
    )
    _require(
        len(plan.eval_raw_rows) == len(eval_rows),
        "EVAL-FIX raw rows must parallel the runner rows",
    )
    _require(
        set(plan.arms) == set(ARM_ORDER),
        f"smoke plan arms must be exactly {ARM_ORDER!r}",
    )
    _require(
        callable(plan.g6_fallback_evidence),
        "production smoke plan must expose G6 per-aux fallback evidence",
    )
    strafe_set = frozenset(
        int(value) for value in plan.strafe_reset_original_indices
    )
    _smoke_plan, smoke_observed = derive_static_reset_receipt(
        smoke_rows,
        support_name=SMOKE_SUPPORT,
        strafe_reset_original_indices=strafe_set,
    )
    _require(
        smoke_observed
        == frozenset(
            int(value)
            for value in plan.expected_static_reset_original_indices
        ),
        "derived SMK-TRAIN static resets differ from the frozen expectation",
    )
    derive_static_reset_receipt(
        eval_rows,
        support_name=EVAL_SUPPORT,
        strafe_reset_original_indices=strafe_set,
    )

    written: dict[str, str] = {}

    def _emit(name: str, payload: Any) -> str:
        sha = _write_receipt(output / name, payload)
        written[name] = sha
        return sha

    # (1) update-0 checkpoints, bit-identical init proof.
    checkpoints: dict[str, dict[str, Any]] = {"update0": {}, "update128": {}}
    for arm in ARM_ORDER:
        payload = _checkpoint_payload_of(plan.arms[arm], arm)
        checkpoints["update0"][arm] = save_arm_checkpoint(
            output / f"checkpoint_update0_{arm}.pt",
            arm=arm,
            model_state=payload["model"],
            optimizer_state=payload["optimizer"],
            u_pre=0,
            assembly_receipt_sha256=receipt_file_sha,
        )
    _require(
        checkpoints["update0"][S_CTRL]["state_sha256"]
        == checkpoints["update0"][S_SELF]["state_sha256"],
        "paired arms are not bit-identical at update 0",
    )

    # (2) update-0 EVAL-FIX for S-SELF in logged/self modes.
    snapshot0_self = evaluate_snapshot(
        eval_rows=eval_rows,
        raw_rows=plan.eval_raw_rows,
        predictor=plan.arms[S_SELF].eval_predictor,
        strafe_reset_original_indices=strafe_set,
        arm=S_SELF,
        snapshot_label="update0",
        controller_config=plan.controller_config,
        loss_fn=plan.arms[S_SELF].eval_loss_fn,
    )
    _emit("eval_fix_update0_S-SELF.json", snapshot0_self)

    # (3) paired S-CTRL and S-SELF 128 updates on SMK-TRAIN.
    result = run_paired_smoke(
        smoke_rows,
        callbacks={arm: plan.arms[arm].callbacks for arm in ARM_ORDER},
        hooks=RunnerTelemetryHooks(g6_update=plan.g6_update),
        strafe_reset_original_indices=strafe_set,
        expected_static_reset_original_indices=smoke_observed,
        controller_config=plan.controller_config,
        require_audit_counters=True,
    )
    _emit("count_receipt.json", result.count_receipt.to_dict())

    # (4) update-128 checkpoints.
    for arm in ARM_ORDER:
        payload = _checkpoint_payload_of(plan.arms[arm], arm)
        checkpoints["update128"][arm] = save_arm_checkpoint(
            output / f"checkpoint_update128_{arm}.pt",
            arm=arm,
            model_state=payload["model"],
            optimizer_state=payload["optimizer"],
            u_pre=128,
            assembly_receipt_sha256=receipt_file_sha,
        )

    # (5) update-128 EVAL-FIX for both arms in logged/self modes.
    snapshot128: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        snapshot128[arm] = evaluate_snapshot(
            eval_rows=eval_rows,
            raw_rows=plan.eval_raw_rows,
            predictor=plan.arms[arm].eval_predictor,
            strafe_reset_original_indices=strafe_set,
            arm=arm,
            snapshot_label="update128",
            controller_config=plan.controller_config,
            loss_fn=plan.arms[arm].eval_loss_fn,
        )
        _emit(f"eval_fix_update128_{arm}.json", snapshot128[arm])

    # (6) G6-G9 receipts, then the combined receipt.
    block_mode = str(receipt_document.get("block_mode", G6_BLOCK_MODE))
    g6_fallback_evidence = plan.g6_fallback_evidence()
    g6_receipt, g6_fallback_receipt, g6_document = (
        _evaluate_g6_with_fallback(
            list(result.arms[S_CTRL].g6_updates),
            block_mode=block_mode,
            fallback_evidence=g6_fallback_evidence,
        )
    )
    _emit("g6_receipt.json", g6_document)
    g7_receipts: dict[str, GateReceipt] = {}
    g9_receipts: dict[str, GateReceipt] = {}
    for arm in ARM_ORDER:
        g7_updates = result.arms[arm].g7_updates
        assert_g7_updates_carry_prev_scale(g7_updates)
        g7_receipts[arm] = evaluate_g7(
            [update.gate_update() for update in g7_updates]
        )
        _emit(f"g7_receipt_{arm}.json", g7_receipts[arm].to_dict())
        g9_receipts[arm] = evaluate_g9(**result.arms[arm].g9.gate_kwargs())
        _emit(f"g9_receipt_{arm}.json", g9_receipts[arm].to_dict())
    g8_receipt = evaluate_g8(
        s_self_update0=snapshot0_self,
        s_self_update128=snapshot128[S_SELF],
        s_ctrl_update128=snapshot128[S_CTRL],
    )
    _emit("g8_receipt.json", g8_receipt.to_dict())
    g7_final = _combined_arm_receipt(g7_receipts)
    g9_final = _combined_arm_receipt(g9_receipts)
    combined = {
        **build_smoke_gate_receipt(
            g6_receipt, g7_final, g8_receipt, g9_final
        ),
        "provenance": {
            "source": "run_production_smoke",
            "authoritative": True,
            "assembly_receipt_sha256": receipt_file_sha,
            "cal_audit_receipt_sha256": cal_authority["cal_audit_receipt"][
                "sha256"
            ],
            "lambda_freeze_receipt_sha256": cal_authority[
                "lambda_freeze_receipt"
            ]["sha256"],
        },
    }
    _emit("combined_smoke_gate_receipt.json", combined)

    # (7) machine-readable gate inputs for forensic rebuilds.
    gate_inputs = {
        "schema_version": 1,
        "analysis_class": GATE_INPUTS_CLASS,
        "architecture_lock": ARCHITECTURE_LOCK,
        "block_mode": block_mode,
        "g6_updates": [
            _g6_update_to_dict(update)
            for update in result.arms[S_CTRL].g6_updates
        ],
        "g6_fallback_evidence": _json_safe(g6_fallback_evidence),
        "g7_updates": {
            arm: [update.to_dict() for update in result.arms[arm].g7_updates]
            for arm in ARM_ORDER
        },
        "g9_gate_kwargs": {
            arm: _json_safe(result.arms[arm].g9.gate_kwargs())
            for arm in ARM_ORDER
        },
        "eval_snapshots": {
            "update0": {
                S_SELF: {
                    "logged": snapshot0_self["logged"],
                    "self": snapshot0_self["self"],
                }
            },
            "update128": {
                arm: {
                    "logged": snapshot128[arm]["logged"],
                    "self": snapshot128[arm]["self"],
                }
                for arm in ARM_ORDER
            },
        },
    }
    _emit("gate_inputs.json", gate_inputs)

    passed = bool(combined["passed"])
    summary = {
        "schema_version": 1,
        "analysis_class": SMOKE_SUMMARY_CLASS,
        "architecture_lock": ARCHITECTURE_LOCK,
        "assembly_receipt": {
            "path": str(Path(receipt_path).expanduser().resolve()),
            "sha256": receipt_file_sha,
            "payload_sha256": receipt_document["receipt_payload_sha256"],
        },
        "cal_audit_receipt": cal_authority["cal_audit_receipt"],
        "cal_lambda_authority": cal_authority,
        "smoke_package": receipt_document.get("smoke_package", SMOKE_PACKAGE),
        "lifecycle_order": list(LIFECYCLE_ORDER),
        "g7_g9_arm_policy": G7_G9_ARM_POLICY,
        "seed": plan.seed,
        "device": plan.device,
        "cuda_reproducibility": plan.cuda_reproducibility,
        "checkpoint_init_sha256": result.checkpoint_init_sha256,
        "checkpoints": checkpoints,
        "static_reset_original_indices": sorted(
            result.static_reset_original_indices
        ),
        "gates": {
            "G6": g6_receipt.passed,
            "G7": {arm: g7_receipts[arm].passed for arm in ARM_ORDER},
            "G8": g8_receipt.passed,
            "G9": {arm: g9_receipts[arm].passed for arm in ARM_ORDER},
            "combined": passed,
        },
        "g6_fallback_report": {
            "deciding": False,
            "passed": g6_fallback_receipt.passed,
            "status": "PASS" if g6_fallback_receipt.passed else "FAIL",
            "ratio_median": g6_fallback_receipt.metrics.get("ratio_median"),
        },
        "artifact_sha256": written,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "decision": "GO" if passed else "STOP",
        "formal_training_authorized": passed,
        "next_step": (
            "external_review"
            if passed
            else "seal_negative_result_no_retry_no_gate_tuning"
        ),
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    _write_receipt(output / "smoke_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Forensic gate rebuild + CLI-facing EVAL command
# ---------------------------------------------------------------------------


def _snapshot_from(
    default: Mapping[str, Any], override_path: str | Path | None, label: str
) -> Mapping[str, Any]:
    if override_path is None:
        return default
    document = _load_json(Path(override_path).expanduser().resolve(), label)
    _require(
        "logged" in document and "self" in document,
        f"{label} must contain 'logged' and 'self' summaries",
    )
    return document


def build_gate_receipts_from_artifacts(
    smoke_dir: str | Path,
    *,
    output_dir: str | Path,
    eval0_path: str | Path | None = None,
    eval128_self_path: str | Path | None = None,
    eval128_ctrl_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild G6-G9 and the combined receipt from archived gate inputs.

    This is the post-failure forensic path; it never reruns training and
    always exclusive-writes into a fresh output directory.  Its outputs are
    NEVER authoritative (P1-4): every receipt carries a dedicated
    ``*_forensic_rebuild`` analysis class, the combined receipt always
    records ``formal_training_authorized=false`` regardless of the gate
    outcome, and any substituted EVAL snapshot is recorded in provenance.
    Only :func:`run_production_smoke` can produce an authorizing receipt.
    """

    smoke = Path(smoke_dir).expanduser().resolve()
    gate_inputs = _load_json(smoke / "gate_inputs.json", "smoke gate inputs")
    _require(
        gate_inputs.get("analysis_class") == GATE_INPUTS_CLASS,
        "gate inputs artifact has the wrong analysis class",
    )
    output = _ensure_empty_directory(
        Path(output_dir), "gate rebuild output dir"
    )
    written: dict[str, str] = {}

    def _emit(name: str, payload: Any) -> str:
        sha = _write_receipt(output / name, payload)
        written[name] = sha
        return sha

    def _forensic_gate_dict(receipt: GateReceipt) -> dict[str, Any]:
        payload = receipt.to_dict()
        payload["analysis_class"] = FORENSIC_GATE_CLASS
        payload["forensic_rebuild_not_authoritative"] = True
        return payload

    g6_receipt, _g6_fallback_receipt, g6_document = (
        _evaluate_g6_with_fallback(
            [
                G6Update(
                    u_pre=update["u_pre"],
                    aux_reachable=update["aux_reachable"],
                    track_reachable=update["track_reachable"],
                    cosine_total_track=update.get("cosine_total_track"),
                    signed_projection=update.get("signed_projection"),
                    aux_track_ratio=update.get("aux_track_ratio"),
                    per_aux_ratios=update.get("per_aux_ratios"),
                )
                for update in gate_inputs["g6_updates"]
            ],
            block_mode=str(gate_inputs["block_mode"]),
            fallback_evidence=gate_inputs.get("g6_fallback_evidence"),
        )
    )
    g6_document["analysis_class"] = FORENSIC_GATE_CLASS
    g6_document["forensic_rebuild_not_authoritative"] = True
    _emit("g6_receipt.json", g6_document)
    g7_receipts: dict[str, GateReceipt] = {}
    g9_receipts: dict[str, GateReceipt] = {}
    for arm in ARM_ORDER:
        g7_updates = gate_inputs["g7_updates"][arm]
        for index, update in enumerate(g7_updates):
            _require(
                isinstance(update, Mapping)
                and update.get("abs_tanh_s_prev") is not None,
                f"HS8_G7_PREV_SCALE_MISSING: archived update {index} for "
                f"{arm} lacks abs_tanh_s_prev",
            )
        g7_receipts[arm] = evaluate_g7(g7_updates)
        _emit(f"g7_receipt_{arm}.json", _forensic_gate_dict(g7_receipts[arm]))
        g9_receipts[arm] = evaluate_g9(**gate_inputs["g9_gate_kwargs"][arm])
        _emit(f"g9_receipt_{arm}.json", _forensic_gate_dict(g9_receipts[arm]))
    snapshots = gate_inputs["eval_snapshots"]
    g8_receipt = evaluate_g8(
        s_self_update0=_snapshot_from(
            snapshots["update0"][S_SELF], eval0_path, "update-0 snapshot"
        ),
        s_self_update128=_snapshot_from(
            snapshots["update128"][S_SELF],
            eval128_self_path,
            "update-128 S-SELF snapshot",
        ),
        s_ctrl_update128=_snapshot_from(
            snapshots["update128"][S_CTRL],
            eval128_ctrl_path,
            "update-128 S-CTRL snapshot",
        ),
    )
    _emit("g8_receipt.json", _forensic_gate_dict(g8_receipt))
    combined = {
        **build_smoke_gate_receipt(
            g6_receipt,
            _combined_arm_receipt(g7_receipts),
            g8_receipt,
            _combined_arm_receipt(g9_receipts),
        ),
        # P1-4: a forensic rebuild can never authorize formal training, no
        # matter what the recomputed gates say.
        "analysis_class": FORENSIC_GATES_CLASS,
        "formal_training_authorized": False,
        "forensic_rebuild_not_authoritative": True,
        "provenance": {
            "source": "build_gate_receipts_from_artifacts",
            "authoritative": False,
            "smoke_dir": str(smoke),
            "eval_snapshot_overrides": {
                "update0_S-SELF": (
                    None if eval0_path is None else str(eval0_path)
                ),
                "update128_S-SELF": (
                    None
                    if eval128_self_path is None
                    else str(eval128_self_path)
                ),
                "update128_S-CTRL": (
                    None
                    if eval128_ctrl_path is None
                    else str(eval128_ctrl_path)
                ),
            },
        },
    }
    _emit("combined_smoke_gate_receipt.json", combined)
    return {
        "passed": bool(combined["passed"]),
        "status": combined["status"],
        "formal_training_authorized": False,
        "forensic_rebuild_not_authoritative": True,
        "artifact_sha256": written,
        "output_dir": str(output),
    }


def run_eval_snapshot_command(
    project_root: str | Path,
    *,
    receipt_path: str | Path,
    arm: str,
    snapshot: int,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    verifier: Callable[..., Mapping[str, Any]] = verify_assembly_receipt,
    rows_loader: Callable[..., Any] | None = None,
    token_ledger: Any | None = None,
    predictor_builder: Callable[..., Callable[..., AP2Prediction]] | None = None,
    controller_config: ActionFilterConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Standalone EVAL-FIX snapshot command (post-failure forensics only)."""

    _require(arm in ARM_ORDER, f"unknown arm {arm!r}")
    _require(snapshot in (0, 128), "snapshot must be 0 or 128")
    root = Path(project_root).expanduser().resolve()
    receipt_document = verifier(root, receipt_path)
    receipt_file_sha = _sha256_file(
        Path(receipt_path).expanduser().resolve(), "assembly receipt"
    )
    if rows_loader is None:
        rows_loader = _default_support_rows_loader
    if token_ledger is None:
        token_ledger = getattr(rows_loader, "token_ledger", None)
    token_ledger = _resolve_frozen_token_ledger(
        root, receipt_document, token_ledger=token_ledger
    )
    ledger_sha, token_files = _ledger_identity(token_ledger)
    output = _ensure_empty_directory(Path(output_dir), "EVAL output dir")
    payload = load_arm_checkpoint_verified(
        checkpoint_path,
        expected_assembly_receipt_sha256=receipt_file_sha,
        expected_arm=arm,
        expected_u_pre=snapshot,
    )
    eval_rows, raw_rows, strafe = rows_loader(
        root, EVAL_SUPPORT, token_ledger
    )
    if predictor_builder is None:
        predictor_builder = _integration_attr(
            "f2_experiment.assembly_model",
            "build_eval_row_predictor_from_checkpoint",
        )
    predictor = predictor_builder(root, receipt_document, arm, payload)
    snapshot_document = evaluate_snapshot(
        eval_rows=eval_rows,
        raw_rows=raw_rows,
        predictor=predictor,
        strafe_reset_original_indices=frozenset(int(v) for v in strafe),
        arm=arm,
        snapshot_label=f"update{snapshot}",
        controller_config=controller_config,
    )
    snapshot_document["assembly_receipt_sha256"] = receipt_file_sha
    snapshot_document["token_ledger_binding"] = {
        "anchor": TOKEN_LEDGER_ANCHOR,
        "sha256": ledger_sha,
        "file_count": token_files,
    }
    name = f"eval_fix_update{snapshot}_{arm}.json"
    file_sha = _write_receipt(output / name, snapshot_document)
    return {
        "path": str(output / name),
        "sha256": file_sha,
        "arm": arm,
        "snapshot": snapshot,
    }


__all__ = [
    "ADJUDICATION_AMENDMENT1_ID",
    "ADJUDICATION_AMENDMENT1_RELATIVE",
    "ADJUDICATION_AMENDMENT1_SHA256",
    "ADJUDICATION_RELATIVE",
    "ADJUDICATION_SHA256",
    "ASSEMBLY_RECEIPT_CLASS",
    "ASSEMBLY_RECEIPT_VERSION",
    "ASSEMBLY_SCHEMA_VERSION",
    "CACHE_BINDING_MODE",
    "CACHE_BINDING_REASON",
    "CAL_AUDIT_RECEIPT_CLASS",
    "CAL_SUPPORT",
    "EVAL_MODE_CONTRACT",
    "EVAL_SUPPORT",
    "FORENSIC_GATES_CLASS",
    "FORENSIC_GATE_CLASS",
    "G6_BLOCK_MODE",
    "G6_PROBE_SURFACE",
    "G7_G9_ARM_POLICY",
    "GATE_CONTRACT_CHANGES",
    "G_LEGACY_MAP",
    "LAMBDA_AUX_LOSSES",
    "LAMBDA_FREEZE_CLASS",
    "LAMBDA_MECHANISM",
    "LAMBDA_POLICY",
    "LIFECYCLE_ORDER",
    "OPTIMIZER_CONTRACT",
    "SMOKE_PACKAGE",
    "SMOKE_SUPPORT",
    "CalRowAudit",
    "F2AssemblyContractError",
    "SmokeArmAssembly",
    "SmokeAssemblyPlan",
    "assert_g7_updates_carry_prev_scale",
    "build_assembly_receipt",
    "build_gate_receipts_from_artifacts",
    "build_support_reset_plan",
    "derive_static_reset_receipt",
    "evaluate_snapshot",
    "freeze_assembly_receipt",
    "load_arm_checkpoint_verified",
    "run_cal_audit",
    "run_eval_fix",
    "run_eval_snapshot_command",
    "run_production_smoke",
    "save_arm_checkpoint",
    "verify_assembly_receipt",
    "verify_cal_lambda_authority",
]
