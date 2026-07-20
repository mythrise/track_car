from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from ibr1_experiment import cli
from ibr1_experiment import runtime_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_official_runtime_contract_is_exact_and_selects_cuda_zero() -> None:
    selected: list[int] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def set_device(index: int) -> None:
            selected.append(index)

        @staticmethod
        def current_device() -> int:
            return 0

    fake_torch = SimpleNamespace(
        __version__="2.6.0+cu124",
        version=SimpleNamespace(cuda="12.4"),
        cuda=FakeCuda(),
    )
    assert runtime_contract.require_official_python(
        runtime_contract.OFFICIAL_PYTHON_EXECUTABLE
    ) == runtime_contract.OFFICIAL_PYTHON_EXECUTABLE
    assert runtime_contract.require_official_torch_cuda(fake_torch) == {
        "python_executable": str(runtime_contract.OFFICIAL_PYTHON_EXECUTABLE),
        "torch_version": "2.6.0+cu124",
        "cuda_runtime": "12.4",
        "device": "cuda:0",
    }
    assert selected == [0]


@pytest.mark.parametrize(
    ("torch_version", "cuda_runtime"),
    [("2.6.0", "12.4"), ("2.6.0+cu124", "12.1")],
)
def test_official_runtime_contract_rejects_version_drift(
    torch_version: str,
    cuda_runtime: str,
) -> None:
    fake_torch = SimpleNamespace(
        __version__=torch_version,
        version=SimpleNamespace(cuda=cuda_runtime),
    )
    with pytest.raises(runtime_contract.IBR1RuntimeContractError):
        runtime_contract.require_official_torch_cuda(fake_torch)


def test_cli_import_is_stdlib_only_before_command_dispatch() -> None:
    script = (
        "import json,sys; import ibr1_experiment.cli; "
        "print(json.dumps({name:(name in sys.modules) for name in "
        "['torch','ibr1_experiment.model','ibr1_experiment.authority',"
        "'ibr1_experiment.cal_pair']}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(completed.stdout) == {
        "torch": False,
        "ibr1_experiment.model": False,
        "ibr1_experiment.authority": False,
        "ibr1_experiment.cal_pair": False,
    }


def test_pre_torch_environment_rejects_conflicting_cublas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(cli.IBR1CliError, match="conflicts"):
        cli._prepare_pre_torch_environment()


def _runtime() -> dict[str, Any]:
    return {
        "python_executable": "python.exe",
        "platform": "nt",
        "torch_version": "2.6.0+cu124",
        "cuda_runtime": "12.4",
        "cuda_available": True,
        "device": "cuda:0",
        "device_name": "GPU",
        "cuda_reproducibility": {"contract_id": "test"},
    }


def test_bootstrap_dispatch_configures_runtime_before_authority_import(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    events: list[Any] = []

    def configure(*, require_cuda: bool) -> dict[str, Any]:
        events.append(("runtime", require_cuda))
        return _runtime()

    authority = SimpleNamespace(
        ASSEMBLY_PHASE_BOOTSTRAP="bootstrap",
        freeze_assembly_receipt=lambda root, output, *, phase: events.append(
            ("freeze", root, output, phase)
        )
        or {
            "path": str(output),
            "sha256": "a" * 64,
            "receipt_payload_sha256": "b" * 64,
            "phase": phase,
        },
    )

    def import_module(name: str) -> Any:
        events.append(("import", name))
        assert name == "ibr1_experiment.authority"
        return authority

    monkeypatch.setattr(cli, "_configure_runtime", configure)
    monkeypatch.setattr(cli, "_import_module", import_module)
    output = tmp_path / "bootstrap.json"
    assert (
        cli.main(
            [
                "build-bootstrap",
                "--project-root",
                str(PROJECT_ROOT),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert events[0] == ("runtime", True)
    assert events[1] == ("import", "ibr1_experiment.authority")
    assert events[2][0] == "freeze"
    payload = json.loads(capsys.readouterr().out)
    assert payload["formal_training_authorized"] is False
    assert payload["runtime"]["device"] == "cuda:0"


def test_live_cal_pair_uses_fixed_parent_orchestrator_after_cuda_guard(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    events: list[Any] = []

    monkeypatch.setattr(
        cli,
        "_configure_runtime",
        lambda *, require_cuda: events.append(("runtime", require_cuda))
        or _runtime(),
    )

    def run_pair(root: Path, **kwargs: Any) -> dict[str, Any]:
        events.append(("run_pair", root, kwargs))
        return {
            "path": str(tmp_path / "lambda.json"),
            "sha256": "c" * 64,
            "receipt_payload_sha256": "d" * 64,
            "authority_eligible": True,
            "final_authority_capability": object(),
            "formal_training_authorized": False,
        }

    pair = SimpleNamespace(run_live_cal_pair_and_freeze=run_pair)
    monkeypatch.setattr(
        cli,
        "_import_module",
        lambda name: events.append(("import", name)) or pair,
    )
    assert (
        cli.main(
            [
                "run-cal-pair",
                "--project-root",
                str(PROJECT_ROOT),
                "--bootstrap-receipt",
                "bootstrap.json",
                "--output-dir",
                "cal_pair",
                "--freeze-output",
                "lambda.json",
                "--final-output",
                "final.json",
            ]
        )
        == 0
    )
    assert events[0] == ("runtime", True)
    assert events[1] == ("import", "ibr1_experiment.cal_pair")
    assert events[2][0] == "run_pair"
    assert events[2][2]["final_output_path"] == "final.json"
    payload = json.loads(capsys.readouterr().out)
    assert payload["authority_eligible"] is True
    assert "final_authority_capability" not in payload
    assert payload["formal_training_authorized"] is False


def test_verify_assembly_does_not_require_cuda(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[Any] = []
    runtime = {**_runtime(), "cuda_available": False, "device": None}
    monkeypatch.setattr(
        cli,
        "_configure_runtime",
        lambda *, require_cuda: events.append(("runtime", require_cuda))
        or runtime,
    )
    authority = SimpleNamespace(
        verify_assembly_receipt=lambda root, receipt, *, required_phase: {
            "analysis_class": "ibr1_assembly_source_receipt",
            "phase": required_phase,
            "receipt_payload_sha256": "e" * 64,
        }
    )
    monkeypatch.setattr(cli, "_import_module", lambda _name: authority)
    assert (
        cli.main(
            [
                "verify-assembly",
                "--project-root",
                str(PROJECT_ROOT),
                "--receipt",
                "receipt.json",
                "--phase",
                "final",
            ]
        )
        == 0
    )
    assert events == [("runtime", False)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["phase"] == "final"
    assert payload["formal_training_authorized"] is False
