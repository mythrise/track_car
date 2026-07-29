import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from f2_experiment.cli import (
    NAMESPACE_PACKAGES,
    SOURCE_FILES,
    TRANSITIVE_SOURCE_FILES,
    F2CliError,
    build_parser,
    build_smoke_plan,
    build_support_document,
    exclusive_write_json,
    main,
    _require_windows_cuda,
    source_bindings,
    transitive_source_bindings,
)
from f2_experiment.support import FrozenSupportReceipt, build_frozen_support


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _real_receipt() -> FrozenSupportReceipt:
    return build_frozen_support(
        PROJECT_ROOT / "data/collected_v1/datasets/train.jsonl"
    )


def test_real_support_document_verifies_fable_approvals_and_frozen_receipts():
    document = build_support_document(PROJECT_ROOT)
    assert document["architecture_lock"] == "L1+D2+AP2+F2"
    assert len(document["approval_sha256"]) == 6
    assert document["support"]["train"] == {
        "rows": 13_746,
        "sha256": "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36",
    }
    assert document["support"]["eligible_pool"] == {
        "total": 173,
        "nonmirrored": 110,
        "mirrored": 63,
    }
    assert document["internal_test"] == "sealed"
    assert document["internal_test_opened"] is False
    assert len(document["receipt_payload_sha256"]) == 64
    assert set(document["source_sha256"]) == {
        "f2_experiment/__init__.py",
        "f2_experiment/support.py",
        "f2_experiment/controller.py",
        "f2_experiment/model.py",
        "f2_experiment/evaluation.py",
        "f2_experiment/runner.py",
        "f2_experiment/opentrack_adapter.py",
        "f2_experiment/cli.py",
        "f2_experiment/assembly_data.py",
        "f2_experiment/assembly_model.py",
        "f2_experiment/assembly.py",
        "f2_experiment/reproducibility.py",
    }


def test_smoke_plan_preserves_selected_block_order_and_exact_budget():
    receipt = _real_receipt()
    plan = build_smoke_plan(receipt)
    smoke_rows = plan["smoke"]["ordered_row_indices"]
    eval_rows = plan["evaluation"]["ordered_row_indices"]
    expected_smoke = [
        index for block in receipt.supports["SMK-TRAIN"] for index in block.row_indices
    ]
    assert smoke_rows == expected_smoke
    assert len(smoke_rows) == len(set(smoke_rows)) == 256
    assert len(eval_rows) == len(set(eval_rows)) == 512
    assert not set(smoke_rows) & set(eval_rows)
    assert len(plan["smoke"]["update_pairs"]) == 128
    assert plan["budget"]["per_arm"] == {
        "rows": 256,
        "backbone_forwards": 256,
        "head_forwards": 512,
        "backwards": 256,
        "optimizer_steps": 128,
        "controller_steps": 256,
    }


def test_smoke_plan_fails_closed_when_support_size_is_mutated():
    receipt = _real_receipt()
    mutated_rows = dict(receipt.row_indices)
    mutated_rows["SMK-TRAIN"] = mutated_rows["SMK-TRAIN"][:-1]
    mutated = FrozenSupportReceipt(
        **{**receipt.__dict__, "row_indices": mutated_rows}
    )
    with pytest.raises(F2CliError, match="differ from frozen receipt"):
        build_smoke_plan(mutated)


def test_exclusive_write_json_is_canonical_and_never_overwrites(tmp_path):
    destination = tmp_path / "receipt.json"
    output_sha = exclusive_write_json(destination, {"z": 1, "a": "辅助"})
    payload = destination.read_bytes()
    assert payload == b'{"a":"\xe8\xbe\x85\xe5\x8a\xa9","z":1}\n'
    assert len(output_sha) == 64
    with pytest.raises(FileExistsError):
        exclusive_write_json(destination, {"different": True})
    assert destination.read_bytes() == payload


def test_audit_contract_cli_has_no_validation_or_test_input(tmp_path):
    output = tmp_path / "audit.json"
    assert main(
        [
            "audit-contract",
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(output),
        ]
    ) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["analysis_class"] == "f2_source_contract_audit"
    assert document["internal_test"] == "sealed"
    assert document["internal_test_opened"] is False
    assert "f2_experiment/model.py" in document["source_sha256"]


def test_source_binding_fails_closed_on_missing_required_file(tmp_path):
    with pytest.raises(F2CliError, match="required F2 source is missing"):
        source_bindings(tmp_path)
    with pytest.raises(F2CliError, match="required F2 source is missing"):
        transitive_source_bindings(tmp_path)


def test_source_contract_covers_init_assembly_and_transitive_dependencies():
    names = {str(path) for path in SOURCE_FILES}
    assert "f2_experiment/__init__.py" in names
    assert {
        "f2_experiment/assembly_data.py",
        "f2_experiment/assembly_model.py",
        "f2_experiment/assembly.py",
        "f2_experiment/reproducibility.py",
    } <= names
    assert len(SOURCE_FILES) == 12
    assert len(TRANSITIVE_SOURCE_FILES) == 15
    for relative in TRANSITIVE_SOURCE_FILES:
        assert str(relative).startswith("third_party/OpenTrackVLA/")
        assert (PROJECT_ROOT / relative).is_file(), relative
    bindings = transitive_source_bindings(PROJECT_ROOT)
    assert set(bindings) == {str(path) for path in TRANSITIVE_SOURCE_FILES}
    assert all(len(sha) == 64 for sha in bindings.values())
    for package in NAMESPACE_PACKAGES:
        directory = PROJECT_ROOT.joinpath(*package.split("."))
        assert directory.is_dir()
        assert not (directory / "__init__.py").exists()


def test_package_docstring_is_generalized_to_main_v1_formal_isolation():
    import f2_experiment

    docstring = f2_experiment.__doc__
    assert "must not be imported by any" in docstring
    assert "main-v1 formal lifecycle (v8/v9 and successors)" in docstring
    assert "must not be imported by formal v8." not in docstring


def test_parser_exposes_assembly_lifecycle_subcommands():
    parser = build_parser()
    args = parser.parse_args(
        ["build-assembly-receipt", "--output", "receipt_v1.json"]
    )
    assert args.command == "build-assembly-receipt"
    assert args.support_receipt is None
    args = parser.parse_args(
        ["run-cal-audit", "--receipt", "r.json", "--output-dir", "cal_v1"]
    )
    assert args.command == "run-cal-audit"
    args = parser.parse_args(
        [
            "run-eval-fix",
            "--receipt",
            "r.json",
            "--arm",
            "S-SELF",
            "--snapshot",
            "128",
            "--checkpoint",
            "c.pt",
            "--output-dir",
            "eval128",
        ]
    )
    assert args.arm == "S-SELF"
    assert args.snapshot == 128
    args = parser.parse_args(
        [
            "run-smoke",
            "--receipt",
            "r.json",
            "--output-dir",
            "smoke_v1",
            "--cal-receipt",
            "cal_audit_receipt_v1.json",
        ]
    )
    assert args.cal_receipt == "cal_audit_receipt_v1.json"
    # P1-1: the CAL audit receipt is mandatory for run-smoke.
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["run-smoke", "--receipt", "r.json", "--output-dir", "smoke_v1"]
        )
    args = parser.parse_args(
        [
            "build-assembly-receipt",
            "--output",
            "receipt_v9.json",
            "--lambda-freeze-receipt",
            "lambda_freeze.json",
        ]
    )
    assert args.lambda_freeze_receipt == "lambda_freeze.json"
    args = parser.parse_args(
        ["build-gate-receipts", "--smoke-dir", "smoke_v1", "--output-dir", "g"]
    )
    assert args.eval0 is None
    assert args.eval128_self is None
    assert args.eval128_ctrl is None
    with pytest.raises(SystemExit):
        parser.parse_args(["run-eval-fix", "--receipt", "r.json"])


def _fake_torch_for_cuda_reproducibility(*, available=True, initialized=False):
    state = {
        "initialized": initialized,
        "deterministic": False,
        "warn_only": True,
        "precision": "high",
        "flash": True,
        "mem_efficient": True,
        "cudnn_sdp": True,
        "math": True,
    }
    cuda_backend = SimpleNamespace()
    cuda_backend.matmul = SimpleNamespace(allow_tf32=True)
    cuda_backend.enable_flash_sdp = lambda value: state.__setitem__("flash", value)
    cuda_backend.flash_sdp_enabled = lambda: state["flash"]
    cuda_backend.enable_mem_efficient_sdp = lambda value: state.__setitem__(
        "mem_efficient", value
    )
    cuda_backend.mem_efficient_sdp_enabled = lambda: state["mem_efficient"]
    cuda_backend.enable_cudnn_sdp = lambda value: state.__setitem__(
        "cudnn_sdp", value
    )
    cuda_backend.cudnn_sdp_enabled = lambda: state["cudnn_sdp"]
    cuda_backend.enable_math_sdp = lambda value: state.__setitem__("math", value)
    cuda_backend.math_sdp_enabled = lambda: state["math"]
    cudnn_backend = SimpleNamespace(
        benchmark=True,
        deterministic=False,
        allow_tf32=True,
    )
    fake = SimpleNamespace(
        __version__="2.6.0+cu124",
        version=SimpleNamespace(cuda="12.4"),
        cuda=SimpleNamespace(is_available=lambda: available),
        backends=SimpleNamespace(cuda=cuda_backend, cudnn=cudnn_backend),
    )
    fake.cuda.is_initialized = lambda: state["initialized"]
    fake.cuda.set_initialized = lambda value: state.__setitem__(
        "initialized", value
    )

    def use_deterministic_algorithms(enabled, *, warn_only=False):
        state["deterministic"] = enabled
        state["warn_only"] = warn_only

    fake.use_deterministic_algorithms = use_deterministic_algorithms
    fake.are_deterministic_algorithms_enabled = lambda: state["deterministic"]
    fake.is_deterministic_algorithms_warn_only_enabled = lambda: state["warn_only"]
    fake.set_float32_matmul_precision = lambda value: state.__setitem__(
        "precision", value
    )
    fake.get_float32_matmul_precision = lambda: state["precision"]
    return fake


def test_native_windows_production_commands_forbid_cpu_fallback(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(F2CliError, match="CPU fallback is forbidden"):
        _require_windows_cuda(
            "run-cal-audit",
            platform_name="nt",
            torch_module=_fake_torch_for_cuda_reproducibility(available=False),
        )
    _require_windows_cuda("run-cal-audit", platform_name="posix")


def test_native_windows_production_commands_freeze_cuda_runtime(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    receipt = _require_windows_cuda(
        "run-cal-audit",
        platform_name="nt",
        torch_module=_fake_torch_for_cuda_reproducibility(),
    )
    assert receipt is not None
    assert receipt["contract_id"] == "windows_cuda_deterministic_v1"
    assert receipt["cublas_workspace_config"] == ":4096:8"
    assert receipt["deterministic_algorithms"] is True
    assert receipt["deterministic_warn_only"] is False
    assert receipt["cudnn_deterministic"] is True
    assert receipt["cudnn_benchmark"] is False
    assert receipt["matmul_allow_tf32"] is False
    assert receipt["cudnn_allow_tf32"] is False
    assert receipt["sdpa_backend"] == "math_only"
    assert receipt["sdpa_flash_enabled"] is False
    assert receipt["sdpa_mem_efficient_enabled"] is False
    assert receipt["sdpa_cudnn_enabled"] is False
    assert receipt["sdpa_math_enabled"] is True


def test_native_windows_cuda_runtime_rejects_conflicting_cublas(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(F2CliError, match="conflicts with the frozen"):
        _require_windows_cuda(
            "run-smoke",
            platform_name="nt",
            torch_module=_fake_torch_for_cuda_reproducibility(),
        )


def test_native_windows_cuda_runtime_rejects_late_configuration(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(F2CliError, match="initialized before"):
        _require_windows_cuda(
            "run-cal-audit",
            platform_name="nt",
            torch_module=_fake_torch_for_cuda_reproducibility(initialized=True),
        )


def test_native_windows_cuda_runtime_allows_idempotent_recheck(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake = _fake_torch_for_cuda_reproducibility()
    first = _require_windows_cuda(
        "run-cal-audit", platform_name="nt", torch_module=fake
    )
    fake.cuda.set_initialized(True)
    second = _require_windows_cuda(
        "run-smoke", platform_name="nt", torch_module=fake
    )
    assert first == second
