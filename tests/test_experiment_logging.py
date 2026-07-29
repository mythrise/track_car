import json
import math

import pytest
import torch

from third_party.OpenTrackVLA.experiment_logging import (
    DeviceMemorySampler,
    JsonlMetricLogger,
    MetricLogError,
    canonical_json_sha256,
    sha256_file,
)


def _start(logger, **config_overrides):
    config = {"lr": 2e-5, "epochs": 1, **config_overrides}
    return logger.start_run(
        args=config,
        checkpoint_meta={
            "model_family": "test_family",
            "experiment_id": "T0",
            "seed": 0,
            "data_manifest_hash": "data-manifest",
            "data_jsonl_sha256": "data-jsonl",
            "vision_cache_manifest_sha256": "cache-manifest",
            "checkpoint_selection": {
                "metric": "validation_episode_macro_BCE@1",
                "mode": "min",
            },
        },
        total_params=20,
        trainable_params=10,
        install_exception_hook=False,
    )


def _records(logger):
    return [
        json.loads(line)
        for line in logger.path.read_text(encoding="utf-8").splitlines()
    ]


def test_metric_logger_requires_lifecycle_and_refuses_run_mixing(tmp_path):
    logger = JsonlMetricLogger(tmp_path)
    with pytest.raises(MetricLogError, match="run_start must be the first"):
        logger.log({"phase": "train", "loss": 1.25})

    start = _start(logger)
    assert start["phase"] == "run_start"
    assert start["sequence"] == 0
    assert start["config_sha256"] == canonical_json_sha256(start["config"])
    assert start["provenance"]["data_jsonl_sha256"] == "data-jsonl"
    assert start["parameters"] == {"total": 20, "trainable": 10}
    assert start["runtime"]["torch_version"]
    assert start["timestamp_utc"].endswith("+00:00")

    end = logger.end_run(status="completed", summary={"updates": 0})
    assert end["phase"] == "run_end"
    assert end["sequence"] == 1
    with pytest.raises(MetricLogError, match="after run_end"):
        logger.log({"phase": "train", "loss": 1.0})
    with pytest.raises(MetricLogError, match="refusing to mix"):
        JsonlMetricLogger(tmp_path)


def test_config_hash_is_canonical_and_rejects_nonfinite_values():
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )
    with pytest.raises(ValueError):
        canonical_json_sha256({"loss": float("nan")})


@pytest.mark.parametrize("microsteps", [2, 4])
def test_short_two_update_smoke_logs_first_and_true_last_loss(tmp_path, microsteps):
    logger = JsonlMetricLogger(tmp_path / f"run_{microsteps}")
    _start(logger)
    last_record = None
    for micro_step in range(1, microsteps + 1):
        last_record = {
            "phase": "train",
            "micro_step": micro_step,
            "processed_samples": micro_step,
            "loss": float(10 - micro_step),
        }
        logger.log_train_step(last_record)

    assert last_record is not None
    assert logger.log_train_step(last_record, final=True)
    assert not logger.log_train_step(last_record, final=True)
    logger.end_run(status="completed")

    train_records = [record for record in _records(logger) if record["phase"] == "train"]
    assert [record["micro_step"] for record in train_records] == [1, microsteps]
    assert [record["loss"] for record in train_records] == [
        9.0,
        float(10 - microsteps),
    ]
    assert all(record["samples_per_second"] > 0 for record in train_records)


def test_periodic_step_is_not_duplicated_when_it_is_epoch_final(tmp_path):
    logger = JsonlMetricLogger(tmp_path)
    _start(logger)
    record = {"phase": "train", "micro_step": 10, "loss": 1.0}
    assert logger.log_train_step(record)
    assert not logger.log_train_step(record, final=True)
    logger.end_run(status="completed")
    assert len([item for item in _records(logger) if item["phase"] == "train"]) == 1


def test_nonfinite_loss_logs_error_alert_and_never_writes_nan(tmp_path):
    logger = JsonlMetricLogger(tmp_path)
    _start(logger)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        logger.check_finite_losses(
            {"loss": torch.tensor(float("nan")), "component": 1.0},
            context={"epoch": 0, "micro_step": 1},
        )
    logger.end_run(status="failed", error=FloatingPointError("nonfinite"))
    raw = logger.path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    records = _records(logger)
    alert = next(record for record in records if record["phase"] == "alert")
    assert alert["severity"] == "error"
    assert alert["code"] == "nonfinite_loss"
    assert alert["context"]["fields"] == ["loss"]
    assert records[-1]["status"] == "failed"


def test_json_writer_strictly_rejects_nonfinite_records(tmp_path):
    logger = JsonlMetricLogger(tmp_path)
    _start(logger)
    with pytest.raises(ValueError):
        logger.log({"phase": "train", "micro_step": 1, "loss": float("inf")})
    logger.end_run(status="failed")
    assert all(math.isfinite(record["elapsed_s"]) for record in _records(logger))


def test_gradient_norm_is_recorded_and_nonfinite_gradient_is_fatal(tmp_path):
    logger = JsonlMetricLogger(tmp_path)
    _start(logger)
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    parameter.grad = torch.tensor([3.0])
    info = logger.clip_grad_norm_and_check(
        [parameter], 1.0, context={"optimizer_updates": 1}
    )
    assert info["grad_norm"] == pytest.approx(3.0)
    assert info["grad_clipped"] is True

    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(FloatingPointError, match="gradient norm"):
        logger.clip_grad_norm_and_check(
            [parameter], 1.0, context={"optimizer_updates": 2}
        )
    logger.end_run(status="failed")
    records = _records(logger)
    assert any(
        record.get("code") == "nonfinite_gradient_norm" for record in records
    )
    assert records[-1]["gradient_summary"]["clipped_updates"] == 1


def test_checkpoint_event_hashes_exact_artifact_and_run_end_references_it(tmp_path):
    logger = JsonlMetricLogger(tmp_path / "run")
    _start(logger)
    checkpoint = tmp_path / "run" / "best.pt"
    checkpoint.write_bytes(b"checkpoint-payload")
    artifact = logger.log_checkpoint(
        checkpoint,
        role="best_validation",
        epoch=2,
        optimizer_updates=10,
        selected_value=0.25,
        write_wall_time_s=0.5,
    )
    assert artifact["sha256"] == sha256_file(checkpoint)
    assert artifact["size_bytes"] == len(b"checkpoint-payload")
    logger.end_run(status="completed")
    records = _records(logger)
    assert records[-2]["phase"] == "checkpoint"
    assert records[-1]["checkpoints"][0]["sha256"] == artifact["sha256"]


def test_mps_memory_is_explicitly_sampled_not_exact(monkeypatch):
    current_values = iter((100, 150))
    driver_values = iter((200, 180))
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: next(current_values))
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: next(driver_values))
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1000)
    sampler = DeviceMemorySampler("mps")
    first = sampler.sample()
    second = sampler.sample()
    assert first["peak_kind"] == "sampled_high_water_not_backend_exact"
    assert second["sampled_peak_allocated_bytes"] == 150
    assert second["sampled_peak_driver_bytes"] == 200


def test_cuda_memory_uses_backend_peak_apis(monkeypatch):
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 10)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 20)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 40)
    sample = DeviceMemorySampler("cuda:0").sample()
    assert sample["peak_kind"] == "backend_exact"
    assert sample["peak_allocated_bytes"] == 30
    assert sample["peak_reserved_bytes"] == 40


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_run_end_supports_all_terminal_statuses(tmp_path, status):
    logger = JsonlMetricLogger(tmp_path / status)
    _start(logger)
    error = RuntimeError("boom") if status != "completed" else None
    logger.end_run(status=status, error=error)
    end = _records(logger)[-1]
    assert end["status"] == status
    assert ("error" in end) is (error is not None)
