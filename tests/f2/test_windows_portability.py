import os
from pathlib import Path
import subprocess

import pytest

from third_party.OpenTrackVLA.experiment_binding import (
    ExperimentBindingError,
    _relocate_manifest_path,
)
from third_party.OpenTrackVLA.local_weights import resolve_local_model_path
from f2_experiment.assembly_data import (
    F2AssemblyContractError,
    _ledger_from_relpaths,
)


def _make_model(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    return path


def _link_directory(destination: Path, source: Path) -> None:
    if os.name != "nt":
        destination.symlink_to(source, target_is_directory=True)
        return
    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(destination),
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"Windows junction unavailable: {result.stderr}")


def test_cache_relocation_rejects_junction_escape(tmp_path):
    relocated = tmp_path / "relocated"
    outside = tmp_path / "outside"
    relocated.mkdir()
    outside.mkdir()
    _link_directory(relocated / "escape", outside)
    with pytest.raises(ExperimentBindingError, match="escapes effective"):
        _relocate_manifest_path(
            "/producer/root/escape/file.pt",
            recorded_root="/producer/root",
            relocated_root=relocated,
        )


def test_token_ledger_rejects_junction_escape(tmp_path):
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    cache.mkdir()
    outside.mkdir()
    (outside / "cur.jpg_vfine.pt").write_bytes(b"fine")
    (outside / "cur.jpg_vcoarse.pt").write_bytes(b"coarse")
    _link_directory(cache / "eps", outside)
    with pytest.raises(F2AssemblyContractError, match="CACHE_PATH_ESCAPE"):
        _ledger_from_relpaths(("eps/cur.jpg",), cache)


def test_explicit_model_path_precedes_environment(tmp_path, monkeypatch):
    explicit = _make_model(tmp_path / "explicit")
    environment = _make_model(tmp_path / "environment")
    monkeypatch.setenv("MODEL_PATH", str(environment))
    resolved = resolve_local_model_path(
        label="model",
        repo_id="org/model",
        explicit=explicit,
        env_var="MODEL_PATH",
    )
    assert Path(resolved) == explicit.resolve()


def test_missing_explicit_model_path_fails_without_env_fallback(
    tmp_path, monkeypatch
):
    environment = _make_model(tmp_path / "environment")
    monkeypatch.setenv("MODEL_PATH", str(environment))
    with pytest.raises(FileNotFoundError, match="explicit local model path"):
        resolve_local_model_path(
            label="model",
            repo_id="org/model",
            explicit=tmp_path / "missing",
            env_var="MODEL_PATH",
        )
