"""Local-first, append-only telemetry shared by experiment trainers."""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import torch

try:  # ``resource`` is available on macOS/Linux, but not every Python target.
    import resource
except ImportError:  # pragma: no cover - exercised only on unsupported platforms.
    resource = None


TELEMETRY_SCHEMA_VERSION = 1
PROVENANCE_KEYS = (
    "data_manifest_hash",
    "data_jsonl_sha256",
    "base_model_sha256",
    "qwen_model_sha256",
    "vision_cache_manifest_sha256",
    "vision_cache_provenance_sha256",
    "vision_cache_token_payload_sha256",
    "dino_model_sha256",
    "siglip_model_sha256",
    "validation",
    "validation_vision_cache",
)


class MetricLogError(RuntimeError):
    pass


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_peak_rss_bytes():
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSDs exposed by CI commonly report KiB.
    return value if sys.platform == "darwin" else value * 1024


class DeviceMemorySampler:
    """Collect backend-aware memory without adding a monitoring dependency."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self._mps_peak_current = 0
        self._mps_peak_driver = 0
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except (RuntimeError, AssertionError):
                pass

    def sample(self) -> dict:
        result = {
            "backend": self.device.type,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
        }
        try:
            if self.device.type == "cuda":
                result.update(
                    {
                        "peak_kind": "backend_exact",
                        "current_allocated_bytes": int(
                            torch.cuda.memory_allocated(self.device)
                        ),
                        "current_reserved_bytes": int(
                            torch.cuda.memory_reserved(self.device)
                        ),
                        "peak_allocated_bytes": int(
                            torch.cuda.max_memory_allocated(self.device)
                        ),
                        "peak_reserved_bytes": int(
                            torch.cuda.max_memory_reserved(self.device)
                        ),
                    }
                )
            elif self.device.type == "mps":
                current = int(torch.mps.current_allocated_memory())
                driver = int(torch.mps.driver_allocated_memory())
                self._mps_peak_current = max(self._mps_peak_current, current)
                self._mps_peak_driver = max(self._mps_peak_driver, driver)
                result.update(
                    {
                        "peak_kind": "sampled_high_water_not_backend_exact",
                        "current_allocated_bytes": current,
                        "driver_allocated_bytes": driver,
                        "sampled_peak_allocated_bytes": self._mps_peak_current,
                        "sampled_peak_driver_bytes": self._mps_peak_driver,
                        "recommended_max_memory_bytes": int(
                            torch.mps.recommended_max_memory()
                        ),
                    }
                )
            else:
                result["peak_kind"] = "os_process_peak_rss"
            result["supported"] = True
        except (RuntimeError, AssertionError, AttributeError) as exc:
            result.update(
                {
                    "supported": False,
                    "peak_kind": "unavailable",
                    "error_type": type(exc).__name__,
                }
            )
        return result


def runtime_info(device: str | torch.device) -> dict:
    resolved = torch.device(device)
    info = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "device_type": resolved.type,
        "device": str(resolved),
        "torch_num_threads": int(torch.get_num_threads()),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        info.update(
            {
                "device_name": torch.cuda.get_device_name(resolved),
                "cuda_version": torch.version.cuda,
                "cuda_capability": list(torch.cuda.get_device_capability(resolved)),
            }
        )
    elif resolved.type == "mps":
        info["device_name"] = "Apple MPS"
    else:
        info["device_name"] = platform.processor() or platform.machine() or "CPU"
    return info


def _finite_scalar(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class JsonlMetricLogger:
    def __init__(
        self,
        output_dir: str | Path,
        filename: str = "metrics.jsonl",
        *,
        device: str | torch.device = "cpu",
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / filename
        if self.path.exists() and self.path.stat().st_size:
            raise MetricLogError(
                f"refusing to mix a new run into existing metric log: {self.path}"
            )
        self.run_id = uuid.uuid4().hex
        self.device = torch.device(device)
        self.memory = DeviceMemorySampler(self.device)
        self._started_monotonic = time.perf_counter()
        self._sequence = 0
        self._started = False
        self._ended = False
        self._last_train_micro_step = None
        self._alert_counts = {"info": 0, "warning": 0, "error": 0}
        self._gradient_updates = 0
        self._gradient_clipped_updates = 0
        self._gradient_spikes = 0
        self._gradient_spike_alerted = False
        self._checkpoint_artifacts = []
        self._previous_excepthook = None
        self._installed_excepthook = None
        self._atexit_callback = None

    def _elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._started_monotonic)

    @property
    def ended(self) -> bool:
        return self._ended

    def _write(self, record: dict, *, sync: bool) -> None:
        rendered = json.dumps(
            _json_ready(record),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
            handle.flush()
            if sync:
                os.fsync(handle.fileno())

    def log(self, record: dict) -> dict:
        if not isinstance(record, dict):
            raise MetricLogError("metric records must be objects")
        phase = record.get("phase")
        if not isinstance(phase, str) or not phase:
            raise MetricLogError("metric records require a non-empty phase")
        if self._ended:
            raise MetricLogError("cannot write after run_end")
        if not self._started and phase != "run_start":
            raise MetricLogError("run_start must be the first metric record")
        if self._started and phase == "run_start":
            raise MetricLogError("run_start may only be written once")

        elapsed = self._elapsed()
        payload = dict(record)
        payload.update(
            {
                "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
                "run_id": self.run_id,
                "sequence": int(self._sequence),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": float(elapsed),
            }
        )
        if phase in {
            "run_start",
            "train",
            "optimizer",
            "epoch",
            "validation",
            "checkpoint",
            "alert",
            "run_end",
        }:
            payload.setdefault("memory", self.memory.sample())
        processed_samples = payload.get("processed_samples")
        if processed_samples is not None and elapsed > 0:
            payload.setdefault(
                "samples_per_second",
                float(processed_samples) / elapsed,
            )
        critical = phase in {"run_start", "checkpoint", "run_end"} or (
            phase == "alert" and payload.get("severity") == "error"
        )
        self._write(payload, sync=critical)
        self._sequence += 1
        if phase == "run_start":
            self._started = True
        elif phase == "run_end":
            self._ended = True
            self._remove_exception_hooks()
        return payload

    def start_run(
        self,
        *,
        args: dict,
        checkpoint_meta: dict,
        total_params: int,
        trainable_params: int,
        install_exception_hook: bool = True,
    ) -> dict:
        config = _json_ready(dict(args))
        meta = _json_ready(dict(checkpoint_meta))
        provenance = {
            key: meta.get(key) for key in PROVENANCE_KEYS if key in meta
        }
        record = self.log(
            {
                "phase": "run_start",
                "config": config,
                "config_sha256": canonical_json_sha256(config),
                "checkpoint_meta": meta,
                "provenance": provenance,
                "runtime": runtime_info(self.device),
                "parameters": {
                    "total": int(total_params),
                    "trainable": int(trainable_params),
                },
                "checkpoint_selection": meta.get("checkpoint_selection"),
            }
        )
        if install_exception_hook:
            self.install_exception_hook()
        return record

    def log_alert(
        self,
        *,
        severity: str,
        code: str,
        message: str,
        context: dict | None = None,
    ) -> dict:
        severity = str(severity).lower()
        if severity not in self._alert_counts:
            raise MetricLogError("alert severity must be info, warning, or error")
        self._alert_counts[severity] += 1
        record = self.log(
            {
                "phase": "alert",
                "severity": severity,
                "code": str(code),
                "message": str(message),
                "context": _json_ready(context or {}),
            }
        )
        print(
            f"!!! [telemetry:{severity}] {code}: {message}",
            file=sys.stderr,
            flush=True,
        )
        return record

    def check_finite_losses(self, losses: dict, *, context: dict | None = None) -> None:
        nonfinite = [str(name) for name, value in losses.items() if not _finite_scalar(value)]
        if not nonfinite:
            return
        validation = (context or {}).get("phase") == "validation"
        self.log_alert(
            severity="error",
            code=("nonfinite_validation_metric" if validation else "nonfinite_loss"),
            message=(
                "non-finite validation metric detected"
                if validation
                else "non-finite loss detected before backward"
            ),
            context={**(context or {}), "fields": nonfinite},
        )
        raise FloatingPointError(
            (
                "non-finite validation metric detected: "
                if validation
                else "non-finite loss detected: "
            )
            + ", ".join(nonfinite)
        )

    def clip_grad_norm_and_check(
        self,
        parameters,
        max_norm: float,
        *,
        context: dict | None = None,
    ) -> dict:
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        with_grad = [parameter for parameter in parameters if parameter.grad is not None]
        if not with_grad:
            self.log_alert(
                severity="error",
                code="missing_gradients",
                message="no trainable parameter has a gradient before optimizer step",
                context=context,
            )
            raise FloatingPointError("optimizer step has no trainable gradients")
        norm = torch.nn.utils.clip_grad_norm_(with_grad, float(max_norm))
        norm_value = (
            float(norm.detach().cpu().item())
            if isinstance(norm, torch.Tensor)
            else float(norm)
        )
        self.memory.sample()  # Sample every optimizer update, especially for MPS.
        if not math.isfinite(norm_value):
            self.log_alert(
                severity="error",
                code="nonfinite_gradient_norm",
                message="non-finite gradient norm detected before optimizer step",
                context={**(context or {}), "grad_norm": str(norm_value)},
            )
            raise FloatingPointError("non-finite gradient norm detected")
        self._gradient_updates += 1
        clipped = norm_value > float(max_norm)
        if clipped:
            self._gradient_clipped_updates += 1
        spike = float(max_norm) > 0 and norm_value > 10.0 * float(max_norm)
        if spike:
            self._gradient_spikes += 1
            if not self._gradient_spike_alerted:
                self._gradient_spike_alerted = True
                self.log_alert(
                    severity="warning",
                    code="gradient_spike",
                    message="gradient norm exceeded 10x the clipping threshold",
                    context={
                        **(context or {}),
                        "grad_norm": norm_value,
                        "grad_clip": float(max_norm),
                    },
                )
        return {
            "grad_norm": norm_value,
            "grad_clip": float(max_norm),
            "grad_clipped": bool(clipped),
            "gradient_spike": bool(spike),
        }

    def log_train_step(
        self,
        record: dict,
        *,
        final: bool = False,
        interval: int = 10,
    ) -> bool:
        """Log step 1, periodic steps, and the true final step without duplicates."""

        if record.get("phase") != "train":
            raise MetricLogError("train-step records must use phase='train'")
        if interval <= 0:
            raise ValueError("interval must be positive")
        try:
            micro_step = int(record["micro_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MetricLogError("train-step records require an integer micro_step") from exc
        if micro_step <= 0:
            raise MetricLogError("train-step micro_step must be positive")

        should_log = final or micro_step == 1 or micro_step % interval == 0
        if not should_log or micro_step == self._last_train_micro_step:
            return False
        self.log(record)
        self._last_train_micro_step = micro_step
        return True

    def log_checkpoint(
        self,
        path: str | Path,
        *,
        role: str,
        epoch: int,
        optimizer_updates: int,
        selected_value=None,
        write_wall_time_s=None,
    ) -> dict:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        hash_started = time.perf_counter()
        digest = sha256_file(path)
        hash_wall_time = time.perf_counter() - hash_started
        artifact = {
            "role": str(role),
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": digest,
            "epoch": int(epoch),
            "optimizer_updates": int(optimizer_updates),
            "hash_wall_time_s": float(hash_wall_time),
        }
        if selected_value is not None:
            artifact["selected_value"] = float(selected_value)
        if write_wall_time_s is not None:
            artifact["write_wall_time_s"] = float(write_wall_time_s)
        self._checkpoint_artifacts.append(dict(artifact))
        self.log({"phase": "checkpoint", **artifact})
        return artifact

    def end_run(
        self,
        *,
        status: str,
        summary: dict | None = None,
        error: BaseException | None = None,
    ) -> dict:
        if status not in {"completed", "failed", "interrupted"}:
            raise MetricLogError("run_end status must be completed, failed, or interrupted")
        record = {
            "phase": "run_end",
            "status": status,
            "wall_time_s": self._elapsed(),
            "summary": _json_ready(summary or {}),
            "alert_counts": dict(self._alert_counts),
            "gradient_summary": {
                "optimizer_updates_observed": int(self._gradient_updates),
                "clipped_updates": int(self._gradient_clipped_updates),
                "gradient_spikes": int(self._gradient_spikes),
                "clip_fraction": (
                    self._gradient_clipped_updates / self._gradient_updates
                    if self._gradient_updates
                    else None
                ),
            },
            "checkpoints": list(self._checkpoint_artifacts),
        }
        if error is not None:
            record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        return self.log(record)

    def install_exception_hook(self) -> None:
        if self._installed_excepthook is not None:
            return
        self._previous_excepthook = sys.excepthook

        def telemetry_excepthook(exc_type, exc, traceback):
            try:
                if self._started and not self._ended:
                    self.end_run(
                        status=(
                            "interrupted"
                            if issubclass(exc_type, KeyboardInterrupt)
                            else "failed"
                        ),
                        error=exc,
                    )
            except Exception as telemetry_error:  # pragma: no cover - last-resort path.
                print(
                    f"!!! [telemetry:error] failed to record run_end: {telemetry_error}",
                    file=sys.stderr,
                )
            self._previous_excepthook(exc_type, exc, traceback)

        def finalize_unfinished_run():
            if self._started and not self._ended:
                try:
                    self.end_run(
                        status="failed",
                        error=RuntimeError("process exited without an explicit run_end"),
                    )
                except Exception:
                    pass

        self._installed_excepthook = telemetry_excepthook
        self._atexit_callback = finalize_unfinished_run
        sys.excepthook = telemetry_excepthook
        atexit.register(finalize_unfinished_run)

    def _remove_exception_hooks(self) -> None:
        if (
            self._installed_excepthook is not None
            and sys.excepthook is self._installed_excepthook
        ):
            sys.excepthook = self._previous_excepthook
        if self._atexit_callback is not None:
            try:
                atexit.unregister(self._atexit_callback)
            except Exception:  # pragma: no cover - interpreter shutdown edge case.
                pass
        self._installed_excepthook = None
        self._atexit_callback = None
