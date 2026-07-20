from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from ibr1_experiment import cal_worker


def test_worker_configures_determinism_before_cuda_device_query(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    events: list[str] = []
    state = {"initialized": False}

    fake_torch = ModuleType("torch")
    fake_torch.cuda = object()  # type: ignore[attr-defined]

    reproducibility = ModuleType("f2_experiment.reproducibility")

    def prepare() -> None:
        events.append("prepare")

    def configure(torch_module: Any) -> dict[str, Any]:
        assert torch_module is fake_torch
        assert state["initialized"] is False
        events.append("configure")
        return {"contract_id": "test"}

    reproducibility.prepare_cublas_workspace_config = prepare  # type: ignore[attr-defined]
    reproducibility.configure_cuda_reproducibility = configure  # type: ignore[attr-defined]

    calibration = ModuleType("ibr1_experiment.calibration")

    def run_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("cal")
        return {"ok": True, "args": len(args), "role": kwargs["role"]}

    calibration.run_ibr1_cal_audit_once = run_once  # type: ignore[attr-defined]

    def require_runtime(torch_module: Any) -> dict[str, Any]:
        assert torch_module is fake_torch
        events.append("official_cuda_query")
        state["initialized"] = True
        return {
            "python_executable": "python.exe",
            "torch_version": "2.6.0+cu124",
            "cuda_runtime": "12.4",
            "device": "cuda:0",
        }

    monkeypatch.setattr(cal_worker, "require_official_python", lambda: None)
    monkeypatch.setattr(cal_worker, "require_official_torch_cuda", require_runtime)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "f2_experiment.reproducibility", reproducibility
    )
    monkeypatch.setitem(sys.modules, "ibr1_experiment.calibration", calibration)

    exit_code = cal_worker.main(
        [
            "--project-root",
            str(tmp_path),
            "--role",
            "main",
            "--bootstrap-receipt",
            str(tmp_path / "bootstrap.json"),
            "--output-dir",
            str(tmp_path / "cal"),
            "--parent-challenge",
            "a" * 64,
            "--parent-pid",
            str(os.getpid() + 100_000),
        ]
    )

    assert exit_code == 0
    assert events == ["prepare", "configure", "official_cuda_query", "cal"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"]["device"] == "cuda:0"
    assert payload["calibration_result"]["ok"] is True
