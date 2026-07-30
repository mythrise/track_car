"""Command line entry points for the isolated F2 experiment.

The receipt commands (``build-support``, ``build-smoke-plan``,
``audit-contract``, ``build-assembly-receipt``) deliberately stop before
model construction or optimization: they read only SHA-pinned sources, the
frozen training JSONL, and the Fable approval artifacts.  The ``run-*``
commands are the explicit, adjudicated exception: they drive the frozen
assembly lifecycle through :mod:`f2_experiment.assembly` (imported lazily so
the receipt-only path stays light).  Validation and the sealed internal test
are never command-line inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from .controller import bind_controller_identity
from .support import (
    ARCHITECTURE_LOCK,
    CANDIDATE_CAP,
    FROZEN_TRAIN_RELATIVE,
    INTERNAL_TEST_POLICY,
    SMOKE_UPDATES,
    FrozenSupportReceipt,
    build_frozen_support,
    canonical_json_bytes,
    canonical_json_sha256,
    verify_approval_files,
)


class F2CliError(RuntimeError):
    """Raised when a receipt-only F2 command cannot fail closed."""


SOURCE_FILES = (
    PurePosixPath("f2_experiment/__init__.py"),
    PurePosixPath("f2_experiment/support.py"),
    PurePosixPath("f2_experiment/controller.py"),
    PurePosixPath("f2_experiment/model.py"),
    PurePosixPath("f2_experiment/evaluation.py"),
    PurePosixPath("f2_experiment/runner.py"),
    PurePosixPath("f2_experiment/opentrack_adapter.py"),
    PurePosixPath("f2_experiment/cli.py"),
    PurePosixPath("f2_experiment/assembly_data.py"),
    PurePosixPath("f2_experiment/assembly_model.py"),
    PurePosixPath("f2_experiment/assembly.py"),
    PurePosixPath("f2_experiment/reproducibility.py"),
)

# Transitive dependencies of the OpenTrack adapter and the production
# assembly (import closure through third_party), bound byte-for-byte by the
# assembly source receipt v4 (handoff blocker 4).
TRANSITIVE_SOURCE_FILES = (
    PurePosixPath("third_party/OpenTrackVLA/model.py"),
    PurePosixPath("third_party/OpenTrackVLA/cache_gridpool.py"),
    PurePosixPath("third_party/OpenTrackVLA/experiment_binding.py"),
    PurePosixPath("third_party/OpenTrackVLA/experiment_logging.py"),
    PurePosixPath("third_party/OpenTrackVLA/local_weights.py"),
    PurePosixPath("third_party/OpenTrackVLA/harness/__init__.py"),
    PurePosixPath("third_party/OpenTrackVLA/harness/base_repro/__init__.py"),
    PurePosixPath("third_party/OpenTrackVLA/harness/base_repro/polar_cot.py"),
    PurePosixPath("third_party/OpenTrackVLA/harness/base_repro/tim.py"),
    PurePosixPath("third_party/OpenTrackVLA/harness/core/__init__.py"),
    PurePosixPath("third_party/OpenTrackVLA/harness/core/event_bank.py"),
    PurePosixPath("third_party/OpenTrackVLA/harness/core/orchestrator.py"),
    PurePosixPath("third_party/OpenTrackVLA/open_trackvla_hf/__init__.py"),
    PurePosixPath("third_party/OpenTrackVLA/open_trackvla_hf/configuration_open_trackvla.py"),
    PurePosixPath("third_party/OpenTrackVLA/open_trackvla_hf/modeling_open_trackvla.py"),
)

# Package directories that intentionally carry no __init__.py; the receipt
# records their fileless identity instead of a byte binding.
NAMESPACE_PACKAGES = ("third_party", "third_party.OpenTrackVLA")


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise F2CliError(f"required F2 source is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_bindings(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    bindings = {
        relative.as_posix(): _sha256_file(root / relative)
        for relative in SOURCE_FILES
    }
    return bindings


def transitive_source_bindings(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    return {
        relative.as_posix(): _sha256_file(root / relative)
        for relative in TRANSITIVE_SOURCE_FILES
    }


def _ordered_rows(receipt: FrozenSupportReceipt, support_name: str) -> list[int]:
    try:
        blocks = receipt.supports[support_name]
        expected_rows = receipt.row_indices[support_name]
    except KeyError as exc:
        raise F2CliError(f"support receipt is missing {support_name}") from exc
    ordered = [index for block in blocks for index in block.row_indices]
    if len(ordered) != len(set(ordered)):
        raise F2CliError(f"{support_name} ordered rows are not unique")
    if set(ordered) != set(expected_rows):
        raise F2CliError(f"{support_name} ordered rows differ from frozen receipt")
    return ordered


def build_smoke_plan(receipt: FrozenSupportReceipt) -> dict[str, Any]:
    smoke_rows = _ordered_rows(receipt, "SMK-TRAIN")
    eval_rows = _ordered_rows(receipt, "EVAL-FIX")
    if len(smoke_rows) != 256 or len(eval_rows) != 512:
        raise F2CliError("frozen F2 smoke/eval supports have unexpected sizes")
    update_pairs = [smoke_rows[index : index + 2] for index in range(0, 256, 2)]
    if len(update_pairs) != SMOKE_UPDATES or any(len(pair) != 2 for pair in update_pairs):
        raise F2CliError("SMK-TRAIN cannot form exactly 128 optimizer updates")
    plan = {
        "schema_version": 1,
        "analysis_class": "f2_preformal_smoke_plan",
        "architecture_lock": ARCHITECTURE_LOCK,
        "candidate_cap": CANDIDATE_CAP,
        "internal_test": INTERNAL_TEST_POLICY,
        "train_sha256": receipt.train_sha256,
        "support_union_sha256": receipt.union_sha256,
        "arms": ["S-CTRL_logged_only", "S-SELF_frozen_paired_policy"],
        "smoke": {
            "support": "SMK-TRAIN",
            "ordered_row_indices": smoke_rows,
            "ordered_row_indices_sha256": canonical_json_sha256(smoke_rows),
            "rows": 256,
            "optimizer_updates": SMOKE_UPDATES,
            "gradient_accumulation_rows": 2,
            "update_pairs": update_pairs,
            "update_pairs_sha256": canonical_json_sha256(update_pairs),
            "warmup_updates": 16,
            "clock": "u_pre=0..127",
        },
        "evaluation": {
            "support": "EVAL-FIX",
            "ordered_row_indices": eval_rows,
            "ordered_row_indices_sha256": canonical_json_sha256(eval_rows),
            "rows": 512,
            "checkpoints": [0, 128],
            "modes": ["logged", "self"],
            "strata": ["overall", "change", "turn", "other"],
        },
        "budget": {
            "per_arm": {
                "rows": 256,
                "backbone_forwards": 256,
                "head_forwards": 512,
                "backwards": 256,
                "optimizer_steps": 128,
                "controller_steps": 256,
            },
            "arm_identical": True,
        },
        "forbidden": [
            "offset_search",
            "random_sampler",
            "internal_test_access",
            "formal_training_before_G6_G7_G8_G9_pass",
        ],
    }
    plan["plan_payload_sha256"] = canonical_json_sha256(plan)
    return plan


def build_support_document(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    train_path = (root / FROZEN_TRAIN_RELATIVE).resolve()
    expected_train_path = (root / "data/collected_v1/datasets/train.jsonl").resolve()
    if train_path != expected_train_path:
        raise F2CliError("frozen train path binding changed")
    approvals = verify_approval_files(root)
    receipt = build_frozen_support(train_path)
    bindings = source_bindings(root)
    controller_binding = bind_controller_identity(
        bindings["f2_experiment/controller.py"]
    )
    support = receipt.to_dict()
    document = {
        "schema_version": 1,
        "analysis_class": "f2_preformal_support_receipt",
        "architecture_lock": ARCHITECTURE_LOCK,
        "project_root": str(root),
        "train_relative_path": FROZEN_TRAIN_RELATIVE.as_posix(),
        "approval_sha256": approvals,
        "source_sha256": bindings,
        "controller_binding": controller_binding,
        "support": support,
        "smoke_plan": build_smoke_plan(receipt),
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    document["receipt_payload_sha256"] = canonical_json_sha256(document)
    return document


def exclusive_write_json(path: str | Path, value: Any) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build SHA-bound F2 receipts, or drive the frozen assembly "
            "lifecycle (run-* commands exit 2 when a preregistered gate "
            "fails after sealing the negative result)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build-support", "build-smoke-plan", "audit-contract"):
        receipt_parser = subparsers.add_parser(name)
        receipt_parser.add_argument("--project-root", default=".")
        receipt_parser.add_argument("--output", required=True)

    assembly_parser = subparsers.add_parser("build-assembly-receipt")
    assembly_parser.add_argument("--project-root", default=".")
    assembly_parser.add_argument("--output", required=True)
    assembly_parser.add_argument("--support-receipt", default=None)
    assembly_parser.add_argument("--lambda-freeze-receipt", default=None)

    cal_parser = subparsers.add_parser("run-cal-audit")
    cal_parser.add_argument("--project-root", default=".")
    cal_parser.add_argument("--receipt", required=True)
    cal_parser.add_argument("--output-dir", required=True)

    eval_parser = subparsers.add_parser("run-eval-fix")
    eval_parser.add_argument("--project-root", default=".")
    eval_parser.add_argument("--receipt", required=True)
    eval_parser.add_argument("--arm", required=True, choices=("S-CTRL", "S-SELF"))
    eval_parser.add_argument(
        "--snapshot", required=True, type=int, choices=(0, 128)
    )
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--output-dir", required=True)

    smoke_parser = subparsers.add_parser("run-smoke")
    smoke_parser.add_argument("--project-root", default=".")
    smoke_parser.add_argument("--receipt", required=True)
    smoke_parser.add_argument("--output-dir", required=True)
    # P1-1: the CAL audit receipt is mandatory; the CAL -> lambda-freeze ->
    # assembly-receipt authority chain is verified before any smoke.
    smoke_parser.add_argument("--cal-receipt", required=True)

    gates_parser = subparsers.add_parser("build-gate-receipts")
    gates_parser.add_argument("--smoke-dir", required=True)
    gates_parser.add_argument("--output-dir", required=True)
    gates_parser.add_argument("--eval0", default=None)
    gates_parser.add_argument("--eval128-self", default=None)
    gates_parser.add_argument("--eval128-ctrl", default=None)
    return parser


def _print_result(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _require_windows_cuda(
    command: str,
    *,
    platform_name: str | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any] | None:
    """Require CUDA and establish its deterministic Windows runtime policy."""

    platform = os.name if platform_name is None else platform_name
    if platform != "nt":
        return None

    from .reproducibility import (
        F2CudaReproducibilityError,
        configure_cuda_reproducibility,
        prepare_cublas_workspace_config,
    )

    try:
        # This must happen before importing torch when the CLI owns process
        # startup; configure_cuda_reproducibility repeats it idempotently.
        prepare_cublas_workspace_config()
    except F2CudaReproducibilityError as exc:
        raise F2CliError(str(exc)) from exc
    if torch_module is None:
        import torch as torch_module

    if not torch_module.cuda.is_available():
        raise F2CliError(
            f"{command} requires CUDA on native Windows; CPU fallback is "
            "forbidden by the Windows experiment handoff"
        )
    try:
        return configure_cuda_reproducibility(torch_module)
    except F2CudaReproducibilityError as exc:
        raise F2CliError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("build-support", "build-smoke-plan", "audit-contract"):
        root = Path(args.project_root).expanduser().resolve()
        if args.command == "build-support":
            document = build_support_document(root)
        elif args.command == "build-smoke-plan":
            document = build_support_document(root)["smoke_plan"]
        else:
            bindings = source_bindings(root)
            document = {
                "schema_version": 1,
                "analysis_class": "f2_source_contract_audit",
                "architecture_lock": ARCHITECTURE_LOCK,
                "source_sha256": bindings,
                "source_contract_sha256": canonical_json_sha256(bindings),
                "controller_binding": bind_controller_identity(
                    bindings["f2_experiment/controller.py"]
                ),
                "internal_test": INTERNAL_TEST_POLICY,
                "internal_test_opened": False,
            }
        output_sha256 = exclusive_write_json(args.output, document)
        _print_result(
            {"output": str(Path(args.output).resolve()), "sha256": output_sha256}
        )
        return 0

    # Configure native Windows CUDA before importing torch-heavy assembly
    # modules. Receipt-only paths above remain light and device-independent.
    if args.command in ("run-cal-audit", "run-smoke"):
        _require_windows_cuda(args.command)

    # Assembly lifecycle commands import lazily so that the receipt-only
    # path never pulls torch or third_party modules.
    from . import assembly as assembly_lifecycle

    if args.command == "build-assembly-receipt":
        result = assembly_lifecycle.freeze_assembly_receipt(
            Path(args.project_root).expanduser().resolve(),
            args.output,
            support_receipt_path=args.support_receipt,
            lambda_freeze_receipt_path=args.lambda_freeze_receipt,
        )
        _print_result(result)
        return 0
    if args.command == "run-cal-audit":
        result = assembly_lifecycle.run_cal_audit(
            Path(args.project_root).expanduser().resolve(),
            receipt_path=args.receipt,
            output_dir=args.output_dir,
        )
        _print_result({"path": result["path"], "sha256": result["sha256"]})
        return 0
    if args.command == "run-eval-fix":
        result = assembly_lifecycle.run_eval_snapshot_command(
            Path(args.project_root).expanduser().resolve(),
            receipt_path=args.receipt,
            arm=args.arm,
            snapshot=args.snapshot,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
        _print_result(result)
        return 0
    if args.command == "run-smoke":
        summary = assembly_lifecycle.run_production_smoke(
            Path(args.project_root).expanduser().resolve(),
            receipt_path=args.receipt,
            output_dir=args.output_dir,
            cal_audit_receipt_path=args.cal_receipt,
        )
        _print_result(
            {
                "status": summary["status"],
                "formal_training_authorized": summary[
                    "formal_training_authorized"
                ],
                "output_dir": str(Path(args.output_dir).resolve()),
            }
        )
        return 0 if summary["passed"] else 2
    result = assembly_lifecycle.build_gate_receipts_from_artifacts(
        args.smoke_dir,
        output_dir=args.output_dir,
        eval0_path=args.eval0,
        eval128_self_path=args.eval128_self,
        eval128_ctrl_path=args.eval128_ctrl,
    )
    _print_result(
        {
            "status": result["status"],
            "output_dir": result["output_dir"],
            # P1-4: a forensic rebuild can never authorize formal training.
            "formal_training_authorized": False,
            "forensic_rebuild_not_authoritative": True,
        }
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
