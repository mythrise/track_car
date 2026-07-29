import math

import pytest
import torch

from scripts.analyze_adapter_scale_sweep import (
    DEFAULT_SCALES,
    aggregate_results,
    apply_action_filter_proxy,
    cap_residual_ratio,
    parse_named_checkpoints,
    parse_ratio_caps,
    parse_scales,
    residual_ratio,
    split_base_and_residual,
    summarize,
)


def test_parse_scales_uses_sorted_default_grid():
    assert parse_scales(None) == DEFAULT_SCALES
    assert parse_scales("1,-0.025,0") == (-0.025, 0.0, 1.0)


@pytest.mark.parametrize("value", ["", "0,,1", "0,nan", "0,inf", "0,0"])
def test_parse_scales_rejects_ambiguous_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        parse_scales(value)


def test_parse_ratio_caps_requires_unique_positive_finite_values():
    assert parse_ratio_caps(None) == ()
    assert parse_ratio_caps("") == ()
    assert parse_ratio_caps("1,0.25,0.5") == (0.25, 0.5, 1.0)
    for value in ("0", "-1,0.5", "nan", "inf", "0.5,0.5", "0.5,,1"):
        with pytest.raises(ValueError):
            parse_ratio_caps(value)


def test_parse_named_checkpoints_requires_unique_name_and_existing_file(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert parse_named_checkpoints([f"seed0={checkpoint}"]) == [
        ("seed0", checkpoint.resolve())
    ]
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_named_checkpoints([str(checkpoint)])
    with pytest.raises(ValueError, match="duplicated"):
        parse_named_checkpoints([f"seed0={checkpoint}", f"seed0={checkpoint}"])


def test_split_base_and_residual_round_trip_and_ratio():
    base = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    residual = torch.tensor([[0.0, 5.0], [0.0, 1.0]])
    planner_input = base + residual
    recovered = split_base_and_residual(planner_input, residual)
    torch.testing.assert_close(recovered, base)
    torch.testing.assert_close(
        residual_ratio(recovered, residual), torch.tensor([1.0, 0.5])
    )


def _aggregate_run(name, family, seed, value, *, include_cap=True):
    metric = {
        "raw": {"balanced_control_error_at1": {"value": value}},
        "post_action_filter_proxy": {
            "balanced_control_error_at1": {"value": value + 0.01}
        },
    }
    return {
        "name": name,
        "model_family": family,
        "seed": seed,
        "metrics_by_scale": {"0": metric},
        "metrics_by_ratio_cap": {"0.5": metric} if include_cap else {},
    }


def test_aggregate_rejects_duplicate_model_family_seed():
    runs = [
        _aggregate_run("first", "trackvla_pp_lite", 0, 0.3),
        _aggregate_run("duplicate", "trackvla_pp_lite", 0, 0.4),
    ]
    with pytest.raises(ValueError, match="duplicate model_family/seed"):
        aggregate_results(runs, (0.0,), (0.5,))


def test_aggregate_allows_same_seed_across_different_families():
    runs = [
        _aggregate_run("b1", "trackvla_pp_lite", 0, 0.3, include_cap=False),
        _aggregate_run("h0", "pfem_harness", 0, 0.4, include_cap=False),
    ]
    aggregate = aggregate_results(runs, (0.0,), ())
    assert aggregate["raw"]["scales"]["trackvla_pp_lite"][
        "best_diagnostic_mean"
    ] == pytest.approx(0.3)
    assert aggregate["raw"]["scales"]["pfem_harness"][
        "best_diagnostic_mean"
    ] == pytest.approx(0.4)


def test_aggregate_rejects_duplicate_identity_when_ratio_caps_are_empty():
    runs = [
        _aggregate_run("first", "pfem_harness", 2, 0.3, include_cap=False),
        _aggregate_run("duplicate", "pfem_harness", 2, 0.4, include_cap=False),
    ]
    with pytest.raises(ValueError, match="pfem_harness/seed2"):
        aggregate_results(runs, (0.0,), ())


@pytest.mark.parametrize("family", ["", None, 7])
def test_aggregate_rejects_empty_or_nonstring_model_family(family):
    run = _aggregate_run("invalid-family", "trackvla_pp_lite", 0, 0.3)
    run["model_family"] = family
    with pytest.raises(ValueError, match="invalid model_family"):
        aggregate_results([run], (0.0,), (0.5,))


@pytest.mark.parametrize("seed", [True, 0.5, "0", None])
def test_aggregate_rejects_bool_or_noninteger_seed(seed):
    run = _aggregate_run("invalid-seed", "trackvla_pp_lite", 0, 0.3)
    run["seed"] = seed
    with pytest.raises(ValueError, match="invalid seed"):
        aggregate_results([run], (0.0,), (0.5,))


def test_cap_residual_ratio_is_constructive_and_reports_applied_scale():
    base = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    residual = torch.tensor([[0.0, 10.0], [0.0, 0.5]])
    capped, factors = cap_residual_ratio(base, residual, 0.5)
    torch.testing.assert_close(factors, torch.tensor([0.25, 1.0]))
    ratios = residual_ratio(base, capped)
    assert bool((ratios <= 0.5 + 1e-7).all())
    with pytest.raises(ValueError, match="finite and > 0"):
        cap_residual_ratio(base, residual, 0.0)


def test_split_base_and_residual_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        split_base_and_residual(torch.zeros(1, 2), torch.zeros(1, 3))


def test_summarize_is_json_safe_and_rejects_nonfinite():
    assert summarize([]) == {
        "count": 0,
        "mean": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
    payload = summarize([1.0, 2.0, 3.0])
    assert payload["count"] == 3
    assert payload["mean"] == pytest.approx(2.0)
    assert payload["p50"] == pytest.approx(2.0)
    assert math.isfinite(payload["p95"])
    with pytest.raises(ValueError, match="finite"):
        summarize([float("nan")])


def test_action_filter_proxy_resets_and_recurs_only_inside_sequence():
    predictions = torch.tensor(
        [
            [[1.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0]],
            [[-1.0, 0.0, -1.0]],
        ],
        dtype=torch.float64,
    ).numpy()
    records = [
        {
            "episode": "ep0",
            "sequence_id": "seq0",
            "frame_idx": 0,
            "mirrored": False,
            "prev_action": [0.0, 0.0, 0.0],
        },
        {
            "episode": "ep0",
            "sequence_id": "seq0",
            "frame_idx": 1,
            "mirrored": False,
            "prev_action": [1.0, 0.0, 1.0],
        },
        {
            "episode": "ep1",
            "sequence_id": "seq1",
            "frame_idx": 10,
            "mirrored": False,
            "prev_action": [0.0, 0.0, 0.0],
        },
    ]
    filtered = apply_action_filter_proxy(predictions, records)
    torch.testing.assert_close(
        torch.from_numpy(filtered[:, 0, :]),
        torch.tensor(
            [[0.2, 0.0, 0.2], [0.4, 0.0, 0.4], [-0.2, 0.0, -0.2]],
            dtype=torch.float64,
        ),
    )
