from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import torch

import f2_experiment.assembly as f2_assembly
import f2_experiment.assembly_model as f2_assembly_model
from f2_experiment.assembly import CalRowAudit

import ibr1_experiment.authority as authority
import ibr1_experiment.calibration as calibration_under_test
import ibr1_experiment.calibration_model as calibration_model


MEDIANS = {
    "L_cot": 25.641025641025642,
    "L_future": 1.4705882352941178,
    "L_verify": 1.0,
}


@dataclass(frozen=True)
class FakeRow:
    original_row_index: int
    sequence_id: str
    frame_idx: int
    mirrored: bool
    logged_prev_action: tuple[float, float, float]


class FakeRowAuditor:
    def __init__(
        self,
        module: Any,
        *,
        dtype: str = "torch.float32",
        abs_max: float = 0.75,
        reconstruction_error: float = 0.0,
        controlled_shape: tuple[int, int] = (8, 2),
        controlled_cells: int = 16,
        medians: dict[str, float] | None = None,
    ) -> None:
        self.module = module
        self.dtype = dtype
        self.abs_max = abs_max
        self.reconstruction_error = reconstruction_error
        self.controlled_shape = controlled_shape
        self.controlled_cells = controlled_cells
        self.medians = dict(MEDIANS if medians is None else medians)

    def __call__(self, _row: FakeRow, _reasons: Any, _position: int) -> Any:
        return self.module.IBR1CalRowAudit(
            subordinate_audit=CalRowAudit(
                step0_parity=True,
                prev_free=True,
                aux_grad_norms=dict(self.medians),
                track_grad_norm=0.0,
            ),
            geometry_dtype=self.dtype,
            zero_init_persistence=True,
            post_decode_abs_max=self.abs_max,
            controlled_tensor_shape=self.controlled_shape,
            controlled_cells=self.controlled_cells,
            realized_delta_reconstruction_error=self.reconstruction_error,
            prev_free_observation_graph=True,
        )


def _driver(*, count: int = 512, fail: bool = False):
    def run(callback: Any) -> None:
        if fail:
            raise RuntimeError("dummy driver failed")
        for position in range(count):
            reasons = ("stream_first",) if position == 0 else ()
            callback(
                FakeRow(
                    original_row_index=10_000 + position,
                    sequence_id="dummy-sequence",
                    frame_idx=position,
                    mirrored=False,
                    logged_prev_action=(0.2, 0.0, -0.1),
                ),
                reasons,
                position,
            )

    return run


def _fresh_module() -> Any:
    module = importlib.reload(calibration_under_test)
    registry = module._interpreter_registry()
    with registry["lock"]:
        registry["used"] = False
    return module


def _source_binding() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]

    def sha(relative: str) -> str:
        return hashlib.sha256((project_root / relative).read_bytes()).hexdigest()

    return {
        "ibr1_source_sha256": {
            path: sha(path)
            for path in (
                "ibr1_experiment/assembly_model.py",
                "ibr1_experiment/calibration.py",
                "ibr1_experiment/calibration_model.py",
            )
        },
        "inherited_f2_source_sha256": {
            path: sha(path)
            for path in (
                "f2_experiment/assembly.py",
                "f2_experiment/assembly_model.py",
            )
        },
    }


def _fake_bootstrap(module: Any, path: Path) -> dict[str, Any]:
    source = _source_binding()
    source.update(
        {
            "analysis_class": authority.SOURCE_BINDING_CLASS,
            "receipt_payload_sha256": "1" * 64,
        }
    )
    return {
        "analysis_class": authority.ASSEMBLY_RECEIPT_CLASS,
        "phase": authority.ASSEMBLY_PHASE_BOOTSTRAP,
        "source_binding": source,
        "support_binding": {
            "observation": {
                "supports": {
                    "CAL": {
                        "ordered_original_indices": list(range(512)),
                    }
                }
            }
        },
        "receipt_payload_sha256": "2" * 64,
        "test_path": path.name,
        "module": module.__name__,
    }


def test_public_entrypoint_has_no_injectable_runner_or_auditor() -> None:
    signature = inspect.signature(calibration_under_test.run_ibr1_cal_audit_once)
    assert list(signature.parameters) == [
        "project_root",
        "role",
        "bootstrap_receipt_path",
        "output_dir",
    ]
    with pytest.raises(TypeError, match="subordinate_runner"):
        calibration_under_test.run_ibr1_cal_audit_once(
            ".",
            role="main",
            bootstrap_receipt_path="bootstrap.json",
            output_dir="out",
            subordinate_runner=lambda: None,
        )


def test_private_test_only_path_emits_no_authority_artifact(
    tmp_path: Path,
) -> None:
    module = _fresh_module()
    output = tmp_path / "test_only"
    result = module._run_ibr1_cal_audit_test_only(
        tmp_path,
        output_dir=output,
        row_driver=_driver(),
        row_auditor=FakeRowAuditor(module),
    )
    assert result["analysis_class"] == module.TEST_ONLY_EVIDENCE_CLASS
    assert result["authority_eligible"] is False
    assert {path.name for path in output.iterdir()} == {
        module.TEST_ONLY_EVIDENCE_FILENAME
    }
    forbidden = {
        module.RAW_F2_CAL_FILENAME,
        module.NUMERIC_EVIDENCE_FILENAME,
        module.CORE_RECEIPT_FILENAME,
        module.ENVELOPE_RECEIPT_FILENAME,
        module.EXECUTION_WITNESS_FILENAME,
    }
    assert forbidden.isdisjoint(path.name for path in output.iterdir())
    receipt = json.loads(
        (output / module.TEST_ONLY_EVIDENCE_FILENAME).read_text(encoding="utf-8")
    )
    assert receipt["test_only"] is True
    assert receipt["authority_eligible"] is False
    assert receipt["execution_witness_emitted"] is False
    transcript = receipt["callback_transcript"]
    assert transcript["rows"] == 512
    assert len(transcript["records"]) == 512
    assert transcript["records"][0]["position"] == 0
    assert transcript["records"][-1]["position"] == 511


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("short", "exactly 512"),
        ("driver", "dummy driver failed"),
        ("dtype", "not FP32"),
        ("range", "post-decode range"),
        ("reconstruction", "reconstruction failed"),
        ("shape", "8 horizons x 2 axes"),
        ("cells", "8 horizons x 2 axes"),
    ],
)
def test_private_failures_burn_session_and_emit_no_receipt(
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    module = _fresh_module()
    output = tmp_path / failure
    auditor = FakeRowAuditor(
        module,
        dtype="torch.float16" if failure == "dtype" else "torch.float32",
        abs_max=1.01 if failure == "range" else 0.75,
        reconstruction_error=1.1e-6 if failure == "reconstruction" else 0.0,
        controlled_shape=(4, 4) if failure == "shape" else (8, 2),
        controlled_cells=1 if failure == "cells" else 16,
    )
    with pytest.raises(Exception, match=match):
        module._run_ibr1_cal_audit_test_only(
            tmp_path,
            output_dir=output,
            row_driver=_driver(
                count=511 if failure == "short" else 512,
                fail=failure == "driver",
            ),
            row_auditor=auditor,
        )
    assert not (output / module.TEST_ONLY_EVIDENCE_FILENAME).exists()
    with pytest.raises(module.IBR1CalibrationContractError, match="single-use"):
        module._run_ibr1_cal_audit_test_only(
            tmp_path,
            output_dir=tmp_path / f"{failure}_retry",
            row_driver=_driver(),
            row_auditor=FakeRowAuditor(module),
        )


def test_registry_survives_sys_modules_eviction_and_reimport(tmp_path: Path) -> None:
    module = _fresh_module()
    with pytest.raises(RuntimeError, match="dummy driver failed"):
        module._run_ibr1_cal_audit_test_only(
            tmp_path,
            output_dir=tmp_path / "failed",
            row_driver=_driver(fail=True),
            row_auditor=FakeRowAuditor(module),
        )
    name = "ibr1_experiment.calibration"
    original = sys.modules.pop(name)
    try:
        reimported = importlib.import_module(name)
        with pytest.raises(
            reimported.IBR1CalibrationContractError, match="single-use"
        ):
            reimported._run_ibr1_cal_audit_test_only(
                tmp_path,
                output_dir=tmp_path / "reimported",
                row_driver=_driver(),
                row_auditor=FakeRowAuditor(reimported),
            )
    finally:
        sys.modules[name] = original


def test_registry_survives_alias_import(tmp_path: Path) -> None:
    module = _fresh_module()
    with pytest.raises(RuntimeError, match="dummy driver failed"):
        module._run_ibr1_cal_audit_test_only(
            tmp_path,
            output_dir=tmp_path / "failed",
            row_driver=_driver(fail=True),
            row_auditor=FakeRowAuditor(module),
        )
    alias_name = "ibr1_experiment._calibration_alias_test"
    spec = importlib.util.spec_from_file_location(alias_name, module.__file__)
    assert spec is not None and spec.loader is not None
    alias = importlib.util.module_from_spec(spec)
    sys.modules[alias_name] = alias
    try:
        spec.loader.exec_module(alias)
        with pytest.raises(alias.IBR1CalibrationContractError, match="single-use"):
            alias._run_ibr1_cal_audit_test_only(
                tmp_path,
                output_dir=tmp_path / "alias",
                row_driver=_driver(),
                row_auditor=FakeRowAuditor(alias),
            )
    finally:
        sys.modules.pop(alias_name, None)


def test_registry_replaces_inherited_wrong_pid() -> None:
    module = _fresh_module()
    registry = module._interpreter_registry()
    registry["pid"] = os.getpid() + 1
    replacement = module._interpreter_registry()
    assert replacement is not registry
    assert replacement["pid"] == os.getpid()
    assert replacement["used"] is False


def test_public_path_rejects_subordinate_kernel_identity_drift_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _fresh_module()
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text("{}", encoding="utf-8")
    bootstrap = _fake_bootstrap(module, bootstrap_path)
    monkeypatch.setattr(
        authority,
        "verify_assembly_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(bootstrap),
    )
    def spoofed_runner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    spoofed_runner.__module__ = "f2_experiment.assembly"
    spoofed_runner.__qualname__ = "run_cal_audit"
    monkeypatch.setattr(f2_assembly, "run_cal_audit", spoofed_runner)
    output = tmp_path / "identity_drift"
    with pytest.raises(module.IBR1CalibrationContractError, match="identity drifted"):
        module.run_ibr1_cal_audit_once(
            tmp_path,
            role="main",
            bootstrap_receipt_path=bootstrap_path,
            output_dir=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("target", ["factory", "class"])
def test_public_path_rejects_spoofed_model_component_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    module = _fresh_module()
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text("{}", encoding="utf-8")
    bootstrap = _fake_bootstrap(module, bootstrap_path)
    monkeypatch.setattr(
        authority,
        "verify_assembly_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(bootstrap),
    )
    if target == "factory":
        def spoofed_factory(_root: Path) -> object:
            return object()

        spoofed_factory.__module__ = "ibr1_experiment.calibration_model"
        spoofed_factory.__qualname__ = "build_ibr1_cal_row_auditor"
        monkeypatch.setattr(
            calibration_model,
            "build_ibr1_cal_row_auditor",
            spoofed_factory,
        )
    else:
        spoofed_class = type("IBR1ModelCalRowAuditor", (), {})
        spoofed_class.__module__ = "ibr1_experiment.calibration_model"
        spoofed_class.__qualname__ = "IBR1ModelCalRowAuditor"
        monkeypatch.setattr(
            calibration_model,
            "IBR1ModelCalRowAuditor",
            spoofed_class,
        )

    output = tmp_path / f"spoofed_{target}"
    with pytest.raises(module.IBR1CalibrationContractError, match="identity drifted"):
        module.run_ibr1_cal_audit_once(
            tmp_path,
            role="main",
            bootstrap_receipt_path=bootstrap_path,
            output_dir=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("auditor", "method_name"),
    [
        ("ibr1", "__init__"),
        ("ibr1", "_assert_init_binding"),
        ("ibr1", "context_receipt"),
        ("ibr1", "_ibr1_geometry"),
        ("ibr1", "__call__"),
        ("f2", "__init__"),
        ("f2", "context_receipt"),
        ("f2", "_probe_grad_norm"),
        ("f2", "__call__"),
    ],
)
def test_public_path_rejects_replaced_direct_auditor_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auditor: str,
    method_name: str,
) -> None:
    module = _fresh_module()
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text("{}", encoding="utf-8")
    bootstrap = _fake_bootstrap(module, bootstrap_path)
    monkeypatch.setattr(
        authority,
        "verify_assembly_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(bootstrap),
    )
    auditor_class = (
        calibration_model.IBR1ModelCalRowAuditor
        if auditor == "ibr1"
        else f2_assembly_model.CalRowAuditor
    )
    original_method = vars(auditor_class)[method_name]

    def spoofed_method(*_args: Any, **_kwargs: Any) -> object:
        return object()

    spoofed_method.__name__ = original_method.__name__
    spoofed_method.__module__ = original_method.__module__
    spoofed_method.__qualname__ = original_method.__qualname__
    monkeypatch.setattr(auditor_class, method_name, spoofed_method)

    output = tmp_path / f"replaced_{auditor}_{method_name}"
    with pytest.raises(
        module.IBR1CalibrationContractError,
        match="real method/code/module/file binding changed",
    ):
        module.run_ibr1_cal_audit_once(
            tmp_path,
            role="main",
            bootstrap_receipt_path=bootstrap_path,
            output_dir=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("auditor", ["ibr1", "f2"])
def test_public_path_rejects_in_place_auditor_call_code_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auditor: str,
) -> None:
    module = _fresh_module()
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text("{}", encoding="utf-8")
    bootstrap = _fake_bootstrap(module, bootstrap_path)
    monkeypatch.setattr(
        authority,
        "verify_assembly_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(bootstrap),
    )
    auditor_class = (
        calibration_model.IBR1ModelCalRowAuditor
        if auditor == "ibr1"
        else f2_assembly_model.CalRowAuditor
    )
    call_method = vars(auditor_class)["__call__"]

    def spoofed_call(*_args: Any, **_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr(call_method, "__code__", spoofed_call.__code__)

    output = tmp_path / f"replaced_{auditor}_call_code"
    with pytest.raises(
        module.IBR1CalibrationContractError,
        match="real method/code/module/file binding changed",
    ):
        module.run_ibr1_cal_audit_once(
            tmp_path,
            role="main",
            bootstrap_receipt_path=bootstrap_path,
            output_dir=output,
        )
    assert not output.exists()


def test_public_fixed_production_factory_fails_closed_on_cpu_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _fresh_module()
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text("{}", encoding="utf-8")
    bootstrap = _fake_bootstrap(module, bootstrap_path)
    monkeypatch.setattr(
        authority,
        "verify_assembly_receipt",
        lambda *_args, **_kwargs: copy.deepcopy(bootstrap),
    )
    loaded = False

    def forbidden_load():
        nonlocal loaded
        loaded = True
        raise AssertionError("production base loader must not run on CPU fallback")

    monkeypatch.setattr(calibration_model, "default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(calibration_model, "load_base_checkpoint", forbidden_load)
    output = tmp_path / "cpu_stop"
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="requires cuda:0",
    ):
        module.run_ibr1_cal_audit_once(
            tmp_path,
            role="main",
            bootstrap_receipt_path=bootstrap_path,
            output_dir=output,
        )
    assert loaded is False
    assert output.is_dir()
    assert not any(output.iterdir())


def test_authority_numeric_verifier_rejects_test_only_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _fresh_module()
    output = tmp_path / "test_only"
    module._run_ibr1_cal_audit_test_only(
        tmp_path,
        output_dir=output,
        row_driver=_driver(),
        row_auditor=FakeRowAuditor(module),
    )
    bootstrap_path = tmp_path / "bootstrap.json"
    bootstrap_path.write_text("{}", encoding="utf-8")
    bootstrap = _fake_bootstrap(module, bootstrap_path)
    binding = {
        "path": bootstrap_path.name,
        "sha256": hashlib.sha256(bootstrap_path.read_bytes()).hexdigest(),
        "receipt_payload_sha256": bootstrap["receipt_payload_sha256"],
        "analysis_class": authority.ASSEMBLY_RECEIPT_CLASS,
    }
    monkeypatch.setattr(
        authority,
        "_live_bootstrap_snapshot",
        lambda *_args: (copy.deepcopy(bootstrap), dict(binding)),
    )
    with pytest.raises(authority.IBR1AuthorityError, match="identity"):
        authority.verify_cal_numeric_evidence(
            tmp_path,
            output / module.TEST_ONLY_EVIDENCE_FILENAME,
            bootstrap_receipt_path=bootstrap_path,
        )
