import json
from pathlib import Path

import pytest
import torch

import f2_experiment.validation_diagnostics as diagnostics
from f2_experiment.assembly_data import ObservationPacket
from f2_experiment.controller import ActionFilterController
from f2_experiment.model import F2AP2Model
from f2_experiment.support import parse_train_jsonl
from f2_experiment.validation_diagnostics import (
    PerceptionStream,
    RolloutState,
    analyze_reasoning,
    derive_validation_reset_contract,
    derive_slices,
    intervention_alphas,
    reasoning_telemetry,
    resolve_validation_selection,
)


class _ResetRecordingAdapter:
    def __init__(self):
        self.reset_masks = []

    def init_state(self, batch_size, device):
        assert batch_size == 1
        return {"counter": torch.zeros(1, device=device)}

    def encode_step(
        self,
        coarse_tokens,
        coarse_tidx,
        fine_tokens,
        fine_tidx,
        instructions,
        previous_state,
        *,
        reset_mask,
        yaw_hist,
        yaw_curr,
    ):
        del coarse_tokens, coarse_tidx, fine_tokens, fine_tidx, instructions
        del yaw_hist, yaw_curr
        self.reset_masks.append(bool(reset_mask))
        counter = torch.zeros_like(previous_state["counter"]) if reset_mask else previous_state["counter"]
        return {"new_state": {"counter": counter + 1}}


class _Arm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)
        self.adapter = _ResetRecordingAdapter()


def _packet():
    return ObservationPacket(
        coarse_tokens=torch.zeros(4, 1536),
        coarse_tidx=torch.zeros(4, dtype=torch.long),
        fine_tokens=torch.zeros(64, 1536),
        fine_tidx=torch.ones(64, dtype=torch.long),
        instruction="follow",
    )


def test_recurrent_state_reset_forces_every_row_reset():
    reset_arm = _Arm()
    normal_arm = _Arm()
    reset_stream = PerceptionStream(reset_arm, reset_every_row=True)
    normal_stream = PerceptionStream(normal_arm, reset_every_row=False)

    for position in range(2):
        reset = position == 0
        reset_stream.encode(_packet(), reset=reset, position=position)
        normal_stream.encode(_packet(), reset=reset, position=position)

    assert reset_arm.adapter.reset_masks == [True, True]
    assert normal_arm.adapter.reset_masks == [True, False]


def test_direct_interventions_zero_alpha_not_feature():
    output = {
        "method_alphas": {
            "polar": torch.tensor([1.0]),
            "tim_q": torch.tensor([0.2]),
            "future": torch.tensor([0.3]),
            "event": torch.tensor([0.5]),
        }
    }

    polar = intervention_alphas(output, "polar_direct_off")
    future = intervention_alphas(output, "future_direct_off")
    combined = intervention_alphas(output, "reasoning_direct_off")

    assert polar["polar"].item() == 0.0
    assert polar["future"].item() == pytest.approx(0.3)
    assert future["future"].item() == 0.0
    assert future["polar"].item() == 1.0
    assert combined["polar"].item() == 0.0
    assert combined["future"].item() == 0.0
    assert output["method_alphas"]["polar"].item() == 1.0


def test_zero_alpha_removes_stream_even_with_projection_bias():
    torch.manual_seed(4)
    model = F2AP2Model(
        d_model=4,
        method_dims={"polar": 4, "tim_q": 4, "future": 4, "event": 4},
    )
    for scale in model.fusion.method_scales.values():
        scale.data.fill_(0.5)
    base = torch.ones(1, 4)
    features = {name: torch.zeros(1, 4) for name in model.fusion.method_dims}
    alphas = {
        "polar": torch.zeros(1),
        "tim_q": torch.full((1,), 0.2),
        "future": torch.full((1,), 0.3),
        "event": torch.full((1,), 0.5),
    }

    composition = model.fusion.compose_context(base, features, alphas)

    assert torch.count_nonzero(composition.method_streams["polar"]) == 0
    assert torch.linalg.vector_norm(composition.method_streams["future"]) > 0


def test_derive_slices_tracks_reacquisition_offsets():
    rows = [
        {
            "episode": "ep",
            "sequence_id": "seq",
            "frame_idx": frame,
            "mirrored": False,
            "polar_invalid": invalid,
            "transition_type": transition,
        }
        for frame, invalid, transition in (
            (1, 1.0, "other"),
            (2, 1.0, "other"),
            (3, 0.0, "turn_onset"),
            (4, 0.0, "sustained_turn"),
            (5, 0.0, "steady_forward"),
        )
    ]

    slices = derive_slices(rows)

    assert slices["current_invalid"] == [0, 1]
    assert slices["reacquisition_offset_0"] == [2]
    assert slices["reacquisition_offset_1"] == [3]
    assert slices["reacquisition_offset_2"] == [4]
    assert slices["turn"] == [2, 3]


def _probabilities(size, winner):
    values = [0.0] * size
    values[winner] = 1.0
    return values


def _telemetry(theta=30, distance=5, q=0.9):
    return {
        "current": {
            "theta_probability": _probabilities(60, theta),
            "distance_probability": _probabilities(30, distance),
            "invalid_probability": 0.01,
        },
        "future": {
            str(horizon): {
                "theta_probability": _probabilities(60, theta),
                "distance_probability": _probabilities(30, distance),
                "visibility_probability": 0.99,
            }
            for horizon in (4, 8, 16)
        },
        "q_write": q,
    }


def test_reasoning_analysis_reports_calibrated_perfect_synthetic_case():
    rows = []
    telemetry = []
    for frame in range(40):
        row = {
            "episode": "ep",
            "sequence_id": "seq",
            "frame_idx": frame,
            "polar_invalid": 0.0,
            "polar_theta_idx": 30,
            "polar_dist_idx": 5,
        }
        for horizon in (4, 8, 16):
            row[f"fut_valid_{horizon}"] = frame + horizon < 40
            row[f"fut_vis_{horizon}"] = 1.0
            row[f"fut_theta_idx_{horizon}"] = 30
            row[f"fut_dist_idx_{horizon}"] = 5
        rows.append(row)
        telemetry.append(_telemetry())

    result = analyze_reasoning(rows, telemetry)

    assert result["current"]["theta_accuracy"] == 1.0
    assert result["current"]["distance_accuracy"] == 1.0
    assert result["current"]["theta_nll"] == pytest.approx(0.0)
    assert result["q_calibration"]["brier"] == pytest.approx(0.01)
    assert result["future"]["4"]["theta_nll"] == pytest.approx(0.0)


def test_reasoning_q_calibration_includes_invalid_rows_as_negative_targets():
    rows = []
    telemetry = []
    for frame, invalid in enumerate((0.0, 1.0)):
        row = {
            "episode": "ep",
            "sequence_id": "seq",
            "frame_idx": frame,
            "polar_invalid": invalid,
            "polar_theta_idx": 30,
            "polar_dist_idx": 5,
        }
        for horizon in (4, 8, 16):
            row[f"fut_valid_{horizon}"] = False
            row[f"fut_vis_{horizon}"] = 0.0
            row[f"fut_theta_idx_{horizon}"] = 0
            row[f"fut_dist_idx_{horizon}"] = 0
        rows.append(row)
        telemetry.append(_telemetry(q=0.9))

    result = analyze_reasoning(rows, telemetry)

    assert result["q_calibration"]["rows"] == 2
    assert result["q_calibration"]["brier"] == pytest.approx((0.01 + 0.81) / 2)


def test_rollout_state_keeps_conditions_independent():
    first = RolloutState(controller=ActionFilterController())
    second = RolloutState(controller=ActionFilterController())
    logged = (0.5, 0.0, 0.0)
    first.previous(logged, reset=True, mode="self")
    second.previous(logged, reset=True, mode="self")

    first.advance([[1.0, 0.0, 1.0]] + [[0.0, 0.0, 0.0]] * 7, mode="self")

    assert first.prev_fy != second.prev_fy
    assert second.prev_fy == (0.5, 0.0)


def test_fixed_selection_contract_rejects_arbitrary_prefixes():
    probe = resolve_validation_selection(max_rows=None, determinism_probe=True)
    prefix = resolve_validation_selection(max_rows=512, determinism_probe=False)
    full = resolve_validation_selection(max_rows=None, determinism_probe=False)

    assert tuple(probe["original_indices"]) == diagnostics.VAL_DETERMINISM_PROBE_INDICES
    assert probe["original_index_sha256"] == diagnostics.VAL_DETERMINISM_PROBE_SHA256
    assert prefix["name"] == "prefix_512_engineering_smoke"
    assert full["abstract_claim_eligible"] is True
    with pytest.raises(diagnostics.F2ValidationDiagnosticError):
        resolve_validation_selection(max_rows=32, determinism_probe=False)
    with pytest.raises(diagnostics.F2ValidationDiagnosticError):
        resolve_validation_selection(max_rows=512, determinism_probe=True)


def test_public_val_reset_contract_matches_frozen_indices():
    root = Path(__file__).resolve().parents[2]
    rows = parse_train_jsonl(
        (root / "data/collected_v1/datasets/val.jsonl").read_bytes()
    )

    contract = derive_validation_reset_contract(rows)

    assert tuple(contract["base_indices"]) == diagnostics.VAL_BASE_RESET_INDICES
    assert tuple(contract["strafe_indices"]) == diagnostics.VAL_STRAFE_RESET_INDICES
    assert tuple(contract["combined_indices"]) == diagnostics.VAL_COMBINED_RESET_INDICES
    assert contract["combined_sha256"] == diagnostics.VAL_COMBINED_RESET_SHA256


def _head_output(theta_winner, distance_winner, q_write):
    theta = torch.zeros(1, 60)
    distance = torch.zeros(1, 30)
    theta[0, theta_winner] = 5.0
    distance[0, distance_winner] = 5.0
    return {
        "cot": {
            "theta_logits": theta,
            "dist_logits": distance,
            "invalid_logit": torch.tensor([-2.0]),
        },
        "cot_decoded": {
            "theta_idx": torch.tensor(theta_winner),
            "dist_idx": torch.tensor(distance_winner),
            "invalid_pred": torch.tensor(False),
            "confidence": torch.tensor(0.8),
        },
        "future": {
            horizon: {
                "theta_logits": theta.clone(),
                "dist_logits": distance.clone(),
                "vis_logit": torch.tensor([2.0]),
            }
            for horizon in (4, 8, 16)
        },
        "orchestrator": {
            "alpha_tim": torch.tensor([0.1]),
            "alpha_event": torch.tensor([0.2]),
            "alpha_future": torch.tensor([0.3]),
        },
        "q_write": torch.tensor([q_write]),
        "method_features": {"tim_q": torch.ones(1, 6)},
        "new_state": {"memory": torch.ones(1, 4)},
    }


def test_memory_reset_telemetry_seals_reasoning_probabilities():
    telemetry = reasoning_telemetry(
        _head_output(7, 3, 0.9),
        _head_output(11, 8, 0.2),
    )

    reset = telemetry["recurrent_reset_heads"]
    assert telemetry["current"]["theta_idx"] == 7
    assert reset["current"]["theta_idx"] == 11
    assert reset["current"]["distance_idx"] == 8
    assert reset["q_write"] == pytest.approx(0.2)
    assert reset["future"]["16"]["theta_idx"] == 11
    assert len(reset["future"]["4"]["theta_probability"]) == 60


def test_evaluator_receipt_allows_only_its_new_test(tmp_path, monkeypatch):
    tests_dir = tmp_path / "tests/f2"
    tests_dir.mkdir(parents=True)
    frozen_test = tests_dir / "test_frozen.py"
    evaluator_test = tests_dir / "test_validation_diagnostics.py"
    frozen_test.write_text("def test_frozen():\n    pass\n", encoding="utf-8")
    evaluator_test.write_text("def test_evaluator():\n    pass\n", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "tests_sha256": {
            "tests/f2/test_frozen.py": diagnostics.sha256_file(frozen_test)
        },
        "asset_binding": {
            "base_hf": {"path": str(tmp_path / "base")},
            "token_ledger_sha256": "a" * 64,
            "token_ledger_file_count": 1,
            "vision_cache": {"recorded_path_root": "legacy"},
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    original_test_bindings = diagnostics.assembly_core._test_bindings

    monkeypatch.setattr(
        diagnostics,
        "verify_frozen_assets",
        lambda *args, **kwargs: {
            "live": True,
            "vision_cache": {"recorded_path_root": "live"},
        },
    )

    def fake_verify(root, path, *, asset_verifier):
        assert diagnostics.assembly_core._test_bindings(root) == receipt["tests_sha256"]
        observed = asset_verifier(root)
        assert observed["live"] is True
        assert observed["token_ledger_sha256"] == "a" * 64
        assert observed["vision_cache"]["recorded_path_root"] == "legacy"
        return receipt

    monkeypatch.setattr(diagnostics.assembly_core, "verify_assembly_receipt", fake_verify)
    verified, trace = diagnostics.verify_evaluator_assembly_receipt(
        tmp_path, receipt_path
    )

    assert verified == receipt
    assert trace["recorded_tests_verified"] is True
    assert diagnostics.assembly_core._test_bindings is original_test_bindings

    frozen_test.write_text("def test_frozen():\n    assert False\n", encoding="utf-8")
    with pytest.raises(diagnostics.F2ValidationDiagnosticError):
        diagnostics.verify_evaluator_assembly_receipt(tmp_path, receipt_path)


def _baseline_record(index):
    sequence = "seq0" if index < 512 else "seq1"
    return {
        "step_actions": [[0.0, 0.0, 0.0]] * 8,
        "prev_action": [0.0, 0.0, 0.0],
        "valid_mask": [True] * 8,
        "transition_type": "other",
        "episode": sequence,
        "sequence_id": sequence,
        "chunk_id": sequence,
        "clip_id": sequence,
        "frame_idx": index,
        "mirrored": False,
        "command": "forward",
        "source_raw_dir": sequence,
    }


def test_probe_baselines_are_selected_by_global_index(tmp_path, monkeypatch):
    selected_indices = (0, 512)
    records = [_baseline_record(index) for index in selected_indices]
    baseline_paths = {}
    for name in diagnostics.BASELINE_NAMES:
        path = tmp_path / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in range(diagnostics.VAL_ROWS):
                row = _baseline_record(index)
                row["pred_step_actions"] = [[index / 1000.0, 0.0, 0.0]] * 8
                handle.write(json.dumps(row) + "\n")
        baseline_paths[name] = path

    def fake_evaluate(prediction, canonical, threshold):
        del canonical
        assert threshold == diagnostics.CONTROL_THRESHOLD
        return {
            "balanced_control_error_at1": {"value": 0.0},
            "selected_h1_forward": prediction[:, 0, 0].tolist(),
        }

    monkeypatch.setattr(diagnostics, "evaluate_predictions", fake_evaluate)
    monkeypatch.setattr(
        diagnostics,
        "action_slice_metrics",
        lambda *args, **kwargs: {"balanced_control_error_at1": 0.0},
    )
    zero_predictions = {
        condition: {
            mode: [[[0.0, 0.0, 0.0]] * 8 for _ in records]
            for mode in diagnostics.MODES
        }
        for condition in diagnostics.CONDITIONS
    }

    result = diagnostics.analyze_action_outputs(
        rows=records,
        records=records,
        predictions=zero_predictions,
        baseline_paths=baseline_paths,
        selected_indices=selected_indices,
    )

    for name in diagnostics.BASELINE_NAMES:
        assert result["methods"][name]["overall"]["selected_h1_forward"] == [
            0.0,
            0.512,
        ]


def test_preexisting_output_directory_is_not_failure_sealed(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        diagnostics,
        "run",
        lambda args: (_ for _ in ()).throw(FileExistsError("already exists")),
    )
    argv = [
        "--project-root", str(tmp_path),
        "--receipt", "receipt.json",
        "--checkpoint", "checkpoint.pt",
        "--preregistration", "prereg.json",
        "--output-dir", str(output),
    ]

    with pytest.raises(FileExistsError):
        diagnostics.main(argv)

    assert not (output / "failed.json").exists()


def test_owned_failed_run_gets_failure_seal(tmp_path, monkeypatch):
    output = tmp_path / "owned"

    def fake_run(args):
        output.mkdir()
        diagnostics._exclusive_write_json(
            output / "run_started.json",
            {
                "analysis_class": "f2_public_validation_memory_reasoning_run_owner",
                "owner_token": args._run_owner_token,
            },
        )
        raise RuntimeError("boom")

    monkeypatch.setattr(diagnostics, "run", fake_run)
    argv = [
        "--project-root", str(tmp_path),
        "--receipt", "receipt.json",
        "--checkpoint", "checkpoint.pt",
        "--preregistration", "prereg.json",
        "--output-dir", str(output),
    ]

    with pytest.raises(RuntimeError):
        diagnostics.main(argv)

    failure = json.loads((output / "failed.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED_ENGINEERING_BURNED_DIRECTORY"
    assert not (output / "complete.json").exists()
