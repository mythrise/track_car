from __future__ import annotations

import copy
from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from f2_experiment.assembly import CalRowAudit
from f2_experiment.assembly_data import (
    COARSE_TOKEN_COUNT,
    FINE_TOKEN_COUNT,
    VISION_FEATURE_DIM,
    observation_packet_from_fields,
)
from f2_experiment.assembly_model import CalRowAuditor
from f2_experiment.model import AP2_HORIZON
from f2_experiment.runner import RunnerRow, checkpoint_init_sha256

from ibr1_experiment.assembly_model import (
    F2SealedInitEvidence,
    IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS,
    build_ibr1_package,
)
from ibr1_experiment.calibration import IBR1CalRowAudit
import ibr1_experiment.calibration_model as calibration_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DummyLLM(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mix = nn.Linear(d_model, d_model, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.mix.weight.dtype

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert attention_mask.shape == inputs_embeds.shape[:2]
        assert output_hidden_states is True
        assert use_cache is False
        mixed = self.mix(inputs_embeds)
        pooled = mixed.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=mixed + pooled)


class DummyOfficialBase(nn.Module):
    def __init__(
        self,
        input_dim: int = VISION_FEATURE_DIM,
        d_model: int = 8,
    ) -> None:
        super().__init__()
        self.D = d_model
        self.cfg = SimpleNamespace(use_angle_tvi=False)
        self.proj = nn.Linear(input_dim, d_model)
        self.llm = DummyLLM(d_model)
        self.act_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.text_token = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

    def _interleave_tvi(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        yaw_per_frame: torch.Tensor | None = None,
        use_angle: bool = False,
    ) -> torch.Tensor:
        assert t_idx.shape[:2] == tokens.shape[:2]
        assert kind_id in (0, 1)
        assert yaw_per_frame is None
        assert use_angle is False
        return tokens

    def _embed_text(
        self,
        instructions: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(instructions)
        return (
            self.text_token.expand(batch_size, -1, -1).to(device),
            torch.ones(batch_size, 2, dtype=torch.long, device=device),
        )


def _observation(seed: int):
    generator = torch.Generator().manual_seed(seed)
    return observation_packet_from_fields(
        {
            "coarse_tokens": torch.randn(
                COARSE_TOKEN_COUNT,
                VISION_FEATURE_DIM,
                generator=generator,
            ),
            "coarse_tidx": torch.zeros(COARSE_TOKEN_COUNT, dtype=torch.long),
            "fine_tokens": torch.randn(
                FINE_TOKEN_COUNT,
                VISION_FEATURE_DIM,
                generator=generator,
            ),
            "fine_tidx": torch.ones(FINE_TOKEN_COUNT, dtype=torch.long),
            "instruction": "follow the person",
        }
    )


def _aux_targets() -> dict[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {
        "polar_theta_idx": torch.zeros(1, dtype=torch.long),
        "polar_dist_idx": torch.zeros(1, dtype=torch.long),
        "polar_invalid": torch.zeros(1),
    }
    for horizon in (4, 8, 16):
        targets[f"fut_valid_{horizon}"] = torch.ones(1, dtype=torch.bool)
        targets[f"fut_vis_{horizon}"] = torch.ones(1)
        targets[f"fut_theta_idx_{horizon}"] = torch.zeros(1, dtype=torch.long)
        targets[f"fut_dist_idx_{horizon}"] = torch.zeros(1, dtype=torch.long)
    return targets


def _row(position: int) -> RunnerRow:
    return RunnerRow(
        original_row_index=1000 + position,
        sequence_id="sequence-a",
        frame_idx=position,
        mirrored=False,
        logged_prev_action=(0.2, 0.0, -0.1),
        target_actions=torch.tensor([[0.2, 0.0, -0.1]] * AP2_HORIZON),
        observation=_observation(position),
        aux_targets=_aux_targets(),
    )


def _dummy_init_sha(base: nn.Module) -> str:
    torch.manual_seed(0)
    arm = build_ibr1_package(
        copy.deepcopy(base),
        device="cpu",
        aux_coefficients=IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS,
    )
    return checkpoint_init_sha256(arm.full_state_dict())


def _evidence(init_sha: str) -> F2SealedInitEvidence:
    return F2SealedInitEvidence(
        primary_path="primary.json",
        primary_sha256="0" * 64,
        negative_adoption_path="adoption.json",
        negative_adoption_sha256="1" * 64,
        negative_seal_path="seal.json",
        negative_seal_sha256="2" * 64,
        smoke_summary_path="summary.json",
        smoke_summary_sha256="3" * 64,
        seed=0,
        checkpoint_init_sha256=init_sha,
    )


def _build_dummy_auditor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    base: nn.Module | None = None,
) -> tuple[calibration_model.IBR1ModelCalRowAuditor, F2SealedInitEvidence]:
    if base is None:
        torch.manual_seed(17)
        base = DummyOfficialBase()
    evidence = _evidence(_dummy_init_sha(base))
    monkeypatch.setattr(
        calibration_model,
        "read_sealed_f2_init_evidence",
        lambda _root: evidence,
    )
    auditor = calibration_model.build_ibr1_cal_row_auditor(
        PROJECT_ROOT,
        base=base,
        device="cpu",
    )
    return auditor, evidence


def test_factory_builds_disjoint_zero_optimizer_f2_and_ibr1_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor, evidence = _build_dummy_auditor(monkeypatch)
    assert isinstance(auditor.subordinate_auditor, CalRowAuditor)
    assert auditor.subordinate_auditor.arm is not auditor.ibr1_arm
    assert auditor.subordinate_auditor.arm.base is not auditor.ibr1_arm.base
    assert auditor.optimizer_objects == 0
    assert auditor.optimizer_updates == 0
    assert not hasattr(auditor.subordinate_auditor.arm, "optimizer")
    assert not hasattr(auditor.ibr1_arm, "optimizer")
    assert checkpoint_init_sha256(
        auditor.subordinate_auditor.arm.full_state_dict()
    ) == checkpoint_init_sha256(auditor.ibr1_arm.full_state_dict())
    assert auditor.context_receipt() == {
        "seed": 0,
        "device": "cpu",
        "package": "SA-Hstar",
        "probe_surface": "base.proj",
        "initialization": (
            "torch.manual_seed(seed) followed by build_package, the "
            "byte-identical smoke arm initialization path"
        ),
        "checkpoint_init_sha256": evidence.checkpoint_init_sha256,
    }


def test_real_runner_row_emits_frozen_subordinate_and_measured_ibr1_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor, evidence = _build_dummy_auditor(monkeypatch)
    f2_before = checkpoint_init_sha256(
        auditor.subordinate_auditor.arm.full_state_dict()
    )
    ibr1_before = checkpoint_init_sha256(auditor.ibr1_arm.full_state_dict())
    audit = auditor(_row(0), ("stream_first",), 0)
    assert isinstance(audit, IBR1CalRowAudit)
    assert isinstance(audit.subordinate_audit, CalRowAudit)
    assert audit.subordinate_audit.step0_parity is True
    assert audit.subordinate_audit.prev_free is True
    assert set(audit.subordinate_audit.aux_grad_norms) == {
        "L_cot",
        "L_future",
        "L_verify",
    }
    assert audit.subordinate_audit.track_grad_norm == 0.0
    assert audit.geometry_dtype == "torch.float32"
    assert audit.zero_init_persistence is True
    assert audit.post_decode_abs_max == pytest.approx(0.2)
    assert audit.controlled_tensor_shape == (8, 2)
    assert audit.controlled_cells == 16
    assert audit.realized_delta_reconstruction_error <= 1e-6
    assert audit.prev_free_observation_graph is True
    assert checkpoint_init_sha256(
        auditor.subordinate_auditor.arm.full_state_dict()
    ) == f2_before == evidence.checkpoint_init_sha256
    assert checkpoint_init_sha256(
        auditor.ibr1_arm.full_state_dict()
    ) == ibr1_before == evidence.checkpoint_init_sha256

    second = auditor(_row(1), (), 1)
    assert second.zero_init_persistence is True
    assert auditor.optimizer_updates == 0


def test_callback_surface_accepts_rows_not_caller_supplied_geometry_booleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor, _evidence_value = _build_dummy_auditor(monkeypatch)
    assert list(inspect.signature(auditor).parameters) == [
        "row",
        "reasons",
        "position",
    ]
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="RunnerRow",
    ):
        auditor(
            {"zero_init_persistence": True, "post_decode_range": True},
            ("stream_first",),
            0,
        )


def test_position_clock_and_zero_init_bytes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor, _evidence_value = _build_dummy_auditor(monkeypatch)
    auditor(_row(0), ("stream_first",), 0)
    with pytest.raises(Exception, match="discontinuity"):
        auditor(_row(2), (), 2)

    drifted, _evidence_value = _build_dummy_auditor(monkeypatch)
    with torch.no_grad():
        drifted.ibr1_arm.model.action_head.forward_branch.bias.add_(0.1)
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="initialization bytes",
    ):
        drifted(_row(0), ("stream_first",), 0)


def test_live_sealed_init_is_rechecked_by_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    base = DummyOfficialBase()
    evidence = _evidence(_dummy_init_sha(base))
    live = [evidence]
    monkeypatch.setattr(
        calibration_model,
        "read_sealed_f2_init_evidence",
        lambda _root: live[0],
    )
    auditor = calibration_model.build_ibr1_cal_row_auditor(
        PROJECT_ROOT,
        base=base,
        device="cpu",
    )
    live[0] = replace(evidence, primary_sha256="f" * 64)
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="live sealed F2 init evidence drifted",
    ):
        auditor.context_receipt()


def test_factory_rejects_sealed_init_mismatch_and_nonzero_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    base = DummyOfficialBase()
    monkeypatch.setattr(
        calibration_model,
        "read_sealed_f2_init_evidence",
        lambda _root: _evidence("f" * 64),
    )
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="initialization bytes",
    ):
        calibration_model.build_ibr1_cal_row_auditor(
            PROJECT_ROOT,
            base=base,
            device="cpu",
        )
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="seed is frozen at zero",
    ):
        calibration_model.build_ibr1_cal_row_auditor(
            PROJECT_ROOT,
            base=base,
            device="cpu",
            seed=1,
        )


def test_default_factory_forbids_cpu_fallback_before_loading_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = False

    def forbidden_load():
        nonlocal loaded
        loaded = True
        raise AssertionError("base loader must not run after CPU fallback")

    monkeypatch.setattr(calibration_model, "default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(calibration_model, "load_base_checkpoint", forbidden_load)
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="requires cuda:0",
    ):
        calibration_model.build_ibr1_cal_row_auditor(PROJECT_ROOT)
    assert loaded is False


def test_injected_base_requires_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(31)
    base = DummyOfficialBase()
    evidence = _evidence(_dummy_init_sha(base))
    monkeypatch.setattr(
        calibration_model,
        "read_sealed_f2_init_evidence",
        lambda _root: evidence,
    )
    with pytest.raises(
        calibration_model.IBR1CalibrationModelContractError,
        match="explicit device",
    ):
        calibration_model.build_ibr1_cal_row_auditor(
            PROJECT_ROOT,
            base=base,
        )
