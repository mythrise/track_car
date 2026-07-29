import copy
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from scripts.bootstrap_experiments import (
    BootstrapError,
    FROZEN_TEST_MANIFEST_SHA256,
    HEADLINE_EPISODE_IDS,
    VerifiedHeadlineResults,
    extract_bce_cells,
    load_verified_headline_results,
    main,
    paired_two_way_bootstrap,
)


def _sha(character):
    return character * 64


def _canonical_sha(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _render_frozen_json(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _verified_results(
    results,
    *,
    iterations=10,
    analysis_seed=20_260_715,
    baseline_experiment_id="B1",
    candidate_experiment_id="H0",
):
    run_receipts = {}
    for index, (run_name, payload) in enumerate(sorted(results.items())):
        provenance = payload["provenance"]
        run_receipts[run_name] = {
            "experiment_id": payload["metrics"]["experiment_id"],
            "seed": payload["metrics"]["seed"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "training_metrics_sha256": _sha(chr(ord("a") + index % 6)),
            "checkpoint_event_sha256": _sha(chr(ord("b") + index % 5)),
            "run_end_event_sha256": _sha(chr(ord("c") + index % 4)),
            "evaluation_results_sha256": _sha(chr(ord("d") + index % 3)),
            "predictions_sha256": (
                provenance.get("evaluation_predictions_sha256")
                if isinstance(provenance.get("evaluation_predictions_sha256"), str)
                and len(provenance["evaluation_predictions_sha256"]) == 64
                else _sha("0")
            ),
        }
    analysis_contract = {
        "schema_version": 1,
        "baseline_experiment_id": baseline_experiment_id,
        "candidate_experiment_id": candidate_experiment_id,
        "iterations": iterations,
        "analysis_seed": analysis_seed,
        "metric": "episode_macro_BCE@1",
        "seed_ids": [0, 1, 2],
        "episode_ids": list(HEADLINE_EPISODE_IDS),
    }
    shared_evaluation_contract = {
        "schema_version": 1,
        "loader": {
            "shuffle": False,
            "num_workers": 0,
            "ordered_record_validation": True,
        },
        "runtime": {
            "device_type": "cpu",
            "torch_default_dtype": "torch.float32",
            "parameter_dtypes": ["torch.float32"],
            "inference_mode": True,
            "autocast": False,
            "cache_payload_verified": True,
        },
    }
    trust_roots = {
        "schema_version": 1,
        "analysis_class": "headline_trust_roots",
        "status": "frozen_before_internal_test_synthetic_fixture",
        "experiment_registry_sha256": _sha("c"),
        "source_tree_sha256": _sha("9"),
        "evaluator_source_sha256": _canonical_sha(
            next(iter(results.values()))["provenance"]["evaluator_source"]
        ),
        "metric_contract_sha256": _canonical_sha(
            next(iter(results.values()))["provenance"]["metric_contract"]
        ),
        "builder_source_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "scripts" / "build_headline_cells.py").read_bytes()
        ).hexdigest(),
        "bootstrap_source_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "scripts" / "bootstrap_experiments.py").read_bytes()
        ).hexdigest(),
        "bootstrap_analysis_contract": analysis_contract,
        "bootstrap_analysis_contract_sha256": _canonical_sha(analysis_contract),
        "shared_evaluation_contract": shared_evaluation_contract,
        "shared_evaluation_contract_sha256": _canonical_sha(
            shared_evaluation_contract
        ),
        "test_manifest_sha256": FROZEN_TEST_MANIFEST_SHA256,
        "test_data_sha256": _sha("b"),
    }
    trust_roots_sha256 = hashlib.sha256(
        _render_frozen_json(trust_roots)
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "verification_status": "verified",
        "run_manifest_sha256": _sha("a"),
        "trust_roots_sha256": trust_roots_sha256,
        "trust_roots": trust_roots,
        "test_data_sha256": _sha("b"),
        "test_manifest_sha256": FROZEN_TEST_MANIFEST_SHA256,
        "registry_sha256": _sha("c"),
        "builder_source_sha256": trust_roots["builder_source_sha256"],
        "results_sha256": _canonical_sha(results),
        "metric_recomputation": {
            "passed": True,
            "metrics": [
                "balanced_control_error_at1",
                "smooth_l1",
                "turn_sign_accuracy",
                "chronological_transition",
                "saturation_rate",
            ],
            "ground_truth_source": "frozen_test_jsonl_not_predictions",
        },
        "runs": run_receipts,
    }
    document = {
        "schema_version": 1,
        "analysis_class": "verified_headline_results",
        "results": results,
        "receipt": receipt,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "verified.json"
        path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return load_verified_headline_results(
            path, expected_sha256, receipt["trust_roots_sha256"]
        )


def _headline_results(*, iterations=10, analysis_seed=20_260_715):
    results = {}
    evaluator_source = {
        "schema_version": 1,
        "files": [{"path": "scripts/eval_offline.py", "sha256": _sha("e")}],
    }
    metric_contract = {
        "schema_version": 1,
        "primary": "episode_macro_BCE@1",
        "transition_threshold": 0.2,
    }
    shared_provenance = {
        "state_mode": "rolling",
        "test_manifest_sha256": FROZEN_TEST_MANIFEST_SHA256,
        "train_manifest_sha256": _sha("0"),
        "train_data_sha256": _sha("1"),
        "validation_manifest_sha256": _sha("2"),
        "validation_data_sha256": _sha("3"),
        "base_model_sha256": _sha("4"),
        "qwen_model_sha256": _sha("5"),
        "vision_cache_manifest_sha256": _sha("2"),
        "vision_cache_provenance_sha256": _sha("6"),
        "vision_cache_token_payload_sha256": _sha("3"),
        "dino_model_sha256": _sha("7"),
        "siglip_model_sha256": _sha("8"),
        "source_tree_sha256": _sha("9"),
        "evaluator_source": evaluator_source,
        "evaluator_source_sha256": _canonical_sha(evaluator_source),
        "metric_contract": metric_contract,
        "metric_contract_sha256": _canonical_sha(metric_contract),
        "experiment_registry_sha256": _sha("a"),
        "fairness_contract_sha256": _sha("b"),
    }
    checkpoint_characters = {
        ("B1", 0): "a",
        ("B1", 1): "b",
        ("B1", 2): "c",
        ("H0", 0): "d",
        ("H0", 1): "e",
        ("H0", 2): "f",
    }
    prediction_characters = {
        ("B1", 0): "0",
        ("B1", 1): "1",
        ("B1", 2): "2",
        ("H0", 0): "3",
        ("H0", 1): "4",
        ("H0", 2): "5",
    }
    for experiment, offset in (("B1", 0.0), ("H0", -0.1)):
        for seed in (0, 1, 2):
            execution_contract = {
                "schema_version": 1,
                "experiment": experiment,
                "batch_size": 1,
                "state_mode": "rolling",
            }
            results[f"{experiment}_s{seed}"] = {
                "checkpoint": f"/checkpoints/{experiment}_s{seed}.pt",
                "provenance": {
                    **shared_provenance,
                    "checkpoint_sha256": _sha(
                        checkpoint_characters[(experiment, seed)]
                    ),
                    "method_contract_sha256": _sha(
                        "c" if experiment == "B1" else "d"
                    ),
                    "evaluation_execution_contract": execution_contract,
                    "evaluation_execution_contract_sha256": _canonical_sha(
                        execution_contract
                    ),
                    "evaluation_predictions_sha256": _sha(
                        prediction_characters[(experiment, seed)]
                    ),
                    "validation_selection_detail_sha256": _sha(
                        str(seed + 1)
                    ),
                    "checkpoint_role": "best_validation",
                    "selection_verified": True,
                    "selected_epoch": 0,
                    "selected_value": 0.25 + 0.01 * seed,
                    "checkpoint_seed": seed,
                    "evaluation_tier": "locked_final",
                    "evaluation_class": "headline",
                    "headline_eligible": True,
                    "state_mode_override": False,
                    "checkpoint_experiment_id": experiment,
                    "effective_experiment_id": experiment,
                },
                "metrics": {
                    "experiment_id": experiment,
                    "seed": seed,
                    "state_mode": "rolling",
                    "balanced_control_error_at1": {
                        "by_episode": {
                            episode: 1.0
                            + offset
                            + 0.01 * seed
                            + 0.001 * episode_index
                            for episode_index, episode in enumerate(
                                HEADLINE_EPISODE_IDS
                            )
                        }
                    },
                },
            }
    return _verified_results(
        results, iterations=iterations, analysis_seed=analysis_seed
    )


def test_headline_extract_and_paired_two_way_bootstrap():
    results = _headline_results(iterations=1000, analysis_seed=7)
    report = paired_two_way_bootstrap(
        extract_bce_cells(results, "B1"),
        extract_bce_cells(results, "H0"),
        iterations=1000,
        analysis_seed=7,
    )
    assert report["mean_delta"] == pytest.approx(-0.1)
    assert report["ci95"][1] < 0
    assert report["seed_ids"] == [0, 1, 2]
    assert report["episode_ids"] == list(HEADLINE_EPISODE_IDS)
    assert report["frozen_matrix"]["cells_per_method"] == 15
    assert report["paper_eligible"] is True
    assert report["publication_label"] == "FROZEN_HEADLINE"
    assert report["provenance_contract"]["shared"]["state_mode"] == "rolling"


def test_exploratory_mode_allows_small_matrix_but_marks_it_non_paper():
    baseline = {0: {"ep0": 1.0, "ep1": 0.8}, 1: {"ep0": 1.1, "ep1": 0.9}}
    candidate = {0: {"ep0": 0.9, "ep1": 0.7}, 1: {"ep0": 1.0, "ep1": 0.8}}
    report = paired_two_way_bootstrap(
        baseline,
        candidate,
        iterations=100,
        analysis_seed=3,
        analysis_mode="exploratory",
    )
    assert report["paper_eligible"] is False
    assert report["publication_label"] == "EXPLORATORY_ONLY_NOT_FOR_PAPER"
    assert "not eligible" in report["warning"]


def test_exploratory_cli_requires_opt_in_and_emits_non_paper_banner(
    tmp_path, capsys
):
    results = {}
    for experiment, offset in (("B1", 0.0), ("H0", -0.1)):
        for seed in (0, 1):
            results[f"{experiment}_s{seed}"] = {
                "metrics": {
                    "experiment_id": experiment,
                    "seed": seed,
                    "balanced_control_error_at1": {
                        "by_episode": {
                            "dev0": 1.0 + offset,
                            "dev1": 0.8 + offset,
                        }
                    },
                }
            }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    common_args = [
        "--results",
        str(results_path),
        "--baseline",
        "B1",
        "--candidate",
        "H0",
        "--iterations",
        "10",
    ]
    with pytest.raises(BootstrapError, match="requires --expected_results_sha256"):
        main(common_args)
    main(
        common_args + ["--analysis_mode", "exploratory"]
    )
    captured = capsys.readouterr()
    assert "EXPLORATORY_ONLY_NOT_FOR_PAPER" in captured.err
    assert json.loads(captured.out)["paper_eligible"] is False


def test_direct_headline_rejects_plain_evaluator_results():
    verified = _headline_results()
    raw = dict(verified)
    with pytest.raises(
        BootstrapError, match="loader-issued verified builder capability"
    ):
        paired_two_way_bootstrap(
            extract_bce_cells(raw, "B1"),
            extract_bce_cells(raw, "H0"),
            iterations=10,
        )


def test_headline_cli_rejects_plain_raw_json_even_with_matching_input_sha(tmp_path):
    raw = dict(_headline_results())
    results_path = tmp_path / "raw_evaluator.json"
    results_path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    with pytest.raises(BootstrapError, match="plain evaluator JSON"):
        main(
            [
                "--results",
                str(results_path),
                "--expected_results_sha256",
                hashlib.sha256(results_path.read_bytes()).hexdigest(),
                "--expected_trust_roots_sha256",
                _sha("d"),
                "--baseline",
                "B1",
                "--candidate",
                "H0",
                "--iterations",
                "10",
            ]
        )


def test_verified_builder_loader_rejects_external_input_sha_mismatch(tmp_path):
    results = dict(_headline_results())
    verified = _verified_results(results)
    document = {
        "schema_version": 1,
        "analysis_class": "verified_headline_results",
        "results": results,
        "receipt": verified.receipt,
    }
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    with pytest.raises(BootstrapError, match="document SHA-256 mismatch"):
        load_verified_headline_results(
            path, _sha("0"), verified.receipt["trust_roots_sha256"]
        )


def test_verified_builder_loader_rejects_wrong_preregistered_trust_root(tmp_path):
    results = dict(_headline_results())
    verified = _verified_results(results)
    document = {
        "schema_version": 1,
        "analysis_class": "verified_headline_results",
        "results": results,
        "receipt": verified.receipt,
    }
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    external_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(BootstrapError, match="trust-roots SHA-256 mismatch"):
        load_verified_headline_results(path, external_sha, _sha("e"))


def test_public_verified_headline_results_construction_cannot_bypass_loader():
    with pytest.raises(
        BootstrapError, match="must be created by load_verified_headline_results"
    ):
        VerifiedHeadlineResults(
            _headline_results(),
            receipt={"verification_status": "verified"},
            document_sha256=_sha("f"),
        )


def test_loaded_results_cannot_be_mutated_after_document_verification():
    verified = _headline_results()
    verified["H0_s0"]["metrics"]["balanced_control_error_at1"]["by_episode"][
        "test004"
    ] = 0.0
    with pytest.raises(BootstrapError, match="results_sha256 mismatch"):
        extract_bce_cells(verified, "H0")


def test_loaded_receipt_cannot_replace_external_trust_root_after_verification():
    verified = _headline_results()
    verified.receipt["trust_roots_sha256"] = _sha("e")
    with pytest.raises(BootstrapError, match="receipt changed after loader"):
        extract_bce_cells(verified, "H0")


def test_extracted_cells_cannot_be_mutated_before_headline_bootstrap():
    results = _headline_results()
    baseline = extract_bce_cells(results, "B1")
    candidate = extract_bce_cells(results, "H0")
    candidate[0]["test004"] = 0.0
    with pytest.raises(BootstrapError, match="cells changed after verified extraction"):
        paired_two_way_bootstrap(
            baseline, candidate, iterations=10, analysis_seed=20_260_715
        )


def test_verified_builder_loader_rejects_forged_builder_source_with_matching_document_sha(
    tmp_path,
):
    results = dict(_headline_results())
    verified = _verified_results(results)
    receipt = copy.deepcopy(verified.receipt)
    receipt["builder_source_sha256"] = _sha("0")
    document = {
        "schema_version": 1,
        "analysis_class": "verified_headline_results",
        "results": results,
        "receipt": receipt,
    }
    path = tmp_path / "forged_builder_source.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    external_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(BootstrapError, match="builder_source_sha256|builder source SHA"):
        load_verified_headline_results(
            path, external_sha, verified.receipt["trust_roots_sha256"]
        )


def test_headline_cli_accepts_verified_builder_document_with_external_sha(
    tmp_path, capsys
):
    results = dict(_headline_results())
    verified = _verified_results(results)
    document = {
        "schema_version": 1,
        "analysis_class": "verified_headline_results",
        "results": results,
        "receipt": verified.receipt,
    }
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    main(
        [
            "--results",
            str(path),
            "--expected_results_sha256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "--expected_trust_roots_sha256",
            verified.receipt["trust_roots_sha256"],
            "--baseline",
            "B1",
            "--candidate",
            "H0",
            "--iterations",
            "10",
        ]
    )
    assert json.loads(capsys.readouterr().out)["paper_eligible"] is True


def test_headline_rejects_missing_seed():
    results = _headline_results()
    del results["H0_s2"]
    results = _verified_results(dict(results))
    with pytest.raises(BootstrapError, match=r"requires seeds.*missing=\[2\]"):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


@pytest.mark.parametrize(
    ("iterations", "analysis_seed", "match"),
    (
        (11, 20_260_715, "iterations"),
        (10, 99, "analysis_seed"),
    ),
)
def test_headline_rejects_unregistered_bootstrap_invocation(
    iterations, analysis_seed, match
):
    results = _headline_results()
    with pytest.raises(BootstrapError, match=match):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=iterations,
            analysis_seed=analysis_seed,
        )


def test_headline_rejects_missing_frozen_episode():
    results = _headline_results()
    del results["H0_s1"]["metrics"]["balanced_control_error_at1"][
        "by_episode"
    ]["test017"]
    results = _verified_results(dict(results))
    with pytest.raises(
        BootstrapError, match=r"requires frozen episodes.*missing=\['test017'\]"
    ):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


def test_headline_rejects_paired_provenance_mismatch():
    results = _headline_results()
    results["H0_s1"]["provenance"]["vision_cache_manifest_sha256"] = _sha(
        "4"
    )
    results = _verified_results(dict(results))
    with pytest.raises(
        BootstrapError,
        match="paired provenance mismatch for vision_cache_manifest_sha256",
    ):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


@pytest.mark.parametrize(
    "field",
    (
        "train_manifest_sha256",
        "train_data_sha256",
        "validation_manifest_sha256",
        "validation_data_sha256",
        "base_model_sha256",
        "qwen_model_sha256",
        "vision_cache_provenance_sha256",
        "dino_model_sha256",
        "siglip_model_sha256",
        "source_tree_sha256",
        "evaluator_source_sha256",
        "metric_contract_sha256",
        "experiment_registry_sha256",
        "fairness_contract_sha256",
    ),
)
def test_headline_rejects_cross_invocation_provenance_drift(field):
    results = _headline_results()
    results["H0_s1"]["provenance"][field] = _sha("e")
    results = _verified_results(dict(results))
    with pytest.raises(BootstrapError, match=field):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


def test_headline_rejects_method_contract_drift_across_seeds():
    results = _headline_results()
    results["H0_s1"]["provenance"]["method_contract_sha256"] = _sha("e")
    results = _verified_results(dict(results))
    with pytest.raises(BootstrapError, match="method contract changes across seeds"):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


def test_headline_rejects_evaluation_execution_contract_drift_across_seeds():
    results = _headline_results()
    changed = copy.deepcopy(
        results["H0_s1"]["provenance"]["evaluation_execution_contract"]
    )
    changed["batch_size"] = 2
    results["H0_s1"]["provenance"].update(
        {
            "evaluation_execution_contract": changed,
            "evaluation_execution_contract_sha256": _canonical_sha(changed),
        }
    )
    results = _verified_results(dict(results))
    with pytest.raises(
        BootstrapError, match="evaluation execution contract changes across seeds"
    ):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


def test_headline_rejects_missing_evaluation_predictions_sha256():
    results = _headline_results()
    del results["H0_s1"]["provenance"]["evaluation_predictions_sha256"]
    with pytest.raises(BootstrapError, match="predictions SHA mismatch"):
        _verified_results(dict(results))


def test_headline_rejects_invalid_evaluation_predictions_sha256():
    results = _headline_results()
    results["H0_s1"]["provenance"]["evaluation_predictions_sha256"] = "invalid"
    with pytest.raises(BootstrapError, match="predictions SHA mismatch"):
        _verified_results(dict(results))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("checkpoint_role", "epoch", "not a locked final run"),
        ("selection_verified", False, "not a locked final run"),
        ("checkpoint_seed", 99, "seed disagrees"),
    ),
)
def test_headline_rejects_nonselected_or_misbound_checkpoint(field, value, match):
    results = _headline_results()
    results["H0_s1"]["provenance"][field] = value
    results = _verified_results(dict(results))
    with pytest.raises(BootstrapError, match=match):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


def test_extract_rejects_duplicate_seed_episode_cell():
    results = _headline_results()
    duplicate = copy.deepcopy(results["B1_s0"])
    duplicate["metrics"]["balanced_control_error_at1"]["by_episode"] = {
        "test004": 0.5
    }
    results["B1_s0_duplicate"] = duplicate
    results = _verified_results(dict(results))
    with pytest.raises(BootstrapError, match="duplicate seed/episode cell"):
        extract_bce_cells(results, "B1")


def test_headline_rejects_reused_checkpoint_provenance():
    results = _headline_results()
    results["H0_s2"]["provenance"]["checkpoint_sha256"] = results["H0_s1"][
        "provenance"
    ]["checkpoint_sha256"]
    results = _verified_results(dict(results))
    with pytest.raises(BootstrapError, match="checkpoint provenance is not unique"):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


def test_headline_rejects_sensitivity_run_even_with_complete_matrix():
    results = _headline_results()
    results["H0_s1"]["provenance"].update(
        {
            "evaluation_class": "sensitivity",
            "headline_eligible": False,
            "state_mode_override": True,
        }
    )
    results = _verified_results(dict(results))
    with pytest.raises(BootstrapError, match="not a locked final run"):
        paired_two_way_bootstrap(
            extract_bce_cells(results, "B1"),
            extract_bce_cells(results, "H0"),
            iterations=10,
        )


def test_bootstrap_rejects_unpaired_episode_support():
    baseline = {0: {"ep0": 1.0, "ep1": 1.0}, 1: {"ep0": 1.0, "ep1": 1.0}}
    candidate = {0: {"ep0": 0.9, "ep1": 0.9}, 1: {"ep0": 0.9}}
    with pytest.raises(BootstrapError, match="same episode"):
        paired_two_way_bootstrap(
            baseline,
            candidate,
            iterations=10,
            analysis_mode="exploratory",
        )
