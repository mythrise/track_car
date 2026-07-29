from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPEN_TRACK_ROOT = PROJECT_ROOT / "third_party" / "OpenTrackVLA"
sys.path.insert(0, str(OPEN_TRACK_ROOT))

from matched_smoke import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    MATCHED_OPTIMIZER_UPDATES,
    MATCHED_ROW_SHA256,
    MATCHED_ROWS,
    MatchedSmokeBinding,
    MatchedSmokeContractError,
    assert_matched_counters,
    build_scoped_subset,
    enforce_matched_args,
    load_matched_smoke_binding,
    prepare_matched_cli_environment,
)


SUPPORT_RECEIPT = (
    PROJECT_ROOT
    / "experiments"
    / "collected_v1_main"
    / "f2_smoke"
    / "support_receipt_v3.json"
)
TRAIN_JSON = PROJECT_ROOT / "data" / "collected_v1" / "datasets" / "train.jsonl"
TRAIN_SHA256 = "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36"


def _dataset_info():
    return {"sample_count": 13746, "data_hash": TRAIN_SHA256}


def _args(family):
    common = {
        "relocated_root": str(PROJECT_ROOT),
        "val_json": None,
        "val_cache_root": None,
        "seed": 0,
        "epochs": 1,
        "history": 31,
        "max_steps": 0,
        "max_optimizer_updates": 0,
        "batch_size": 2 if family == "B0" else 1,
    }
    if family == "B0":
        common.update({"lr": 2e-5, "balance_sampling": False})
    else:
        common.update(
            {
                "grad_accum_steps": 2,
                "base_lr": 2e-5,
                "head_lr": 3e-4,
                "variant": "polar_tim4",
                "state_mode": "rolling",
            }
        )
    return SimpleNamespace(**common)


def test_real_support_receipt_binds_original_train_and_matched_budget():
    binding = load_matched_smoke_binding(
        SUPPORT_RECEIPT,
        train_json=TRAIN_JSON,
        dataset_info=_dataset_info(),
        relocated_root=PROJECT_ROOT,
    )
    assert binding.train_path == TRAIN_JSON.resolve()
    assert binding.source_rows == 13746
    assert len(binding.row_indices) == MATCHED_ROWS
    assert binding.row_indices_sha256 == MATCHED_ROW_SHA256
    assert binding.expected_optimizer_updates == MATCHED_OPTIMIZER_UPDATES
    assert binding.row_indices[:3] == (900, 901, 902)
    assert binding.row_indices[-3:] == (12961, 12962, 12963)


def test_matched_binding_rejects_materialized_subset_file(tmp_path):
    copied_subset = tmp_path / "train.jsonl"
    copied_subset.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MatchedSmokeContractError, match="original full train"):
        load_matched_smoke_binding(
            SUPPORT_RECEIPT,
            train_json=copied_subset,
            dataset_info=_dataset_info(),
            relocated_root=PROJECT_ROOT,
        )


@pytest.mark.parametrize("family", ["B0", "B1"])
def test_matched_args_freeze_128_updates_without_changing_family_knobs(family):
    args = _args(family)
    enforce_matched_args(args, family=family)
    assert args.max_optimizer_updates == MATCHED_OPTIMIZER_UPDATES
    assert args.seed == 0
    if family == "B0":
        assert args.lr == pytest.approx(2e-5)
    else:
        assert args.base_lr == pytest.approx(2e-5)
        assert args.head_lr == pytest.approx(3e-4)
        assert args.state_mode == "rolling"


def test_matched_args_reject_validation_random_sampling_and_lr_changes():
    args = _args("B0")
    args.val_json = "val.jsonl"
    with pytest.raises(MatchedSmokeContractError, match="forbids validation"):
        enforce_matched_args(args, family="B0")

    args = _args("B0")
    args.balance_sampling = True
    with pytest.raises(MatchedSmokeContractError, match="random sampling"):
        enforce_matched_args(args, family="B0")

    args = _args("B1")
    args.head_lr = 1e-3
    with pytest.raises(MatchedSmokeContractError, match="learning rates"):
        enforce_matched_args(args, family="B1")


def test_scoped_subset_hashes_only_support_reachable_tokens(tmp_path):
    image = "data/collected_v1/episodes/train/example/frame.jpg"
    cache_root = tmp_path / "vision_cache"
    token_dir = cache_root / "data" / "collected_v1" / "episodes" / "train" / "example"
    token_dir.mkdir(parents=True)
    (token_dir / "frame.jpg_vfine.pt").write_bytes(b"fine")
    (token_dir / "frame.jpg_vcoarse.pt").write_bytes(b"coarse")

    class Dataset:
        base_root = tmp_path

        def __init__(self):
            self.requested = []

        def __len__(self):
            return 13746

        def get_example(self, index):
            self.requested.append(index)
            return {"current": image, "images": [image]}

        def __getitem__(self, index):
            return self.get_example(index)

    indices = tuple(range(MATCHED_ROWS))
    binding = MatchedSmokeBinding(
        receipt_path=tmp_path / "support.json",
        receipt_file_sha256="a" * 64,
        receipt_payload_sha256="b" * 64,
        relocated_root=tmp_path.resolve(),
        train_path=tmp_path / "data/collected_v1/datasets/train.jsonl",
        train_relative_path="data/collected_v1/datasets/train.jsonl",
        source_rows=13746,
        source_sha256=TRAIN_SHA256,
        row_indices=indices,
        row_indices_sha256="c" * 64,
    )
    dataset = Dataset()
    subset, ledger = build_scoped_subset(dataset, binding, cache_root=cache_root)
    assert dataset.requested == list(indices)
    assert list(subset.indices) == list(indices)
    assert len(subset) == MATCHED_ROWS
    assert ledger["scope"] == "SMK-TRAIN"
    assert ledger["token_files"] == 2
    assert len(ledger["ledger_sha256"]) == 64


def test_matched_counter_guard_is_exact():
    assert_matched_counters(MATCHED_ROWS, MATCHED_OPTIMIZER_UPDATES)
    with pytest.raises(MatchedSmokeContractError, match="wrong budget"):
        assert_matched_counters(MATCHED_ROWS - 1, MATCHED_OPTIMIZER_UPDATES)


def test_cli_bootstrap_sets_cublas_before_torch_import(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    assert prepare_matched_cli_environment(
        ["--matched_support_receipt", "support.json"]
    )
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(MatchedSmokeContractError, match="conflicts"):
        prepare_matched_cli_environment(
            ["--matched_128_support_receipt=support.json"]
        )


@pytest.mark.parametrize(
    "script_name", ("train_baseline.py", "train_trackvla_lite.py")
)
def test_direct_script_bootstrap_exposes_project_root_in_isolated_python(
    script_name, tmp_path
):
    script = OPEN_TRACK_ROOT / "scripts" / script_name
    probe = f"""
import importlib.util
import runpy
import sys

sys.argv = [r"{script}", "--matched_support_receipt", "support.json"]
sys.path.insert(0, r"{script.parent}")
runpy.run_path(r"{script}", run_name="matched_import_probe")
if importlib.util.find_spec("f2_experiment") is None:
    raise SystemExit("f2_experiment is not importable after direct script bootstrap")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
