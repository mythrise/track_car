import json
from pathlib import Path

from data_pipeline.dataset_stats import compute_dataset_stats


def test_dataset_stats_reports_required_distributions_and_delta_ranges(tmp_path):
    dataset = tmp_path / "tiny.jsonl"
    samples = [
        {
            "command": "forward",
            "transition_type": "steady_forward",
            "polar_invalid": 0.0,
            "polar_dist_idx": 29,
            "detection_source": "omdet",
            "delta_pos": [[0.1, 0.0, -0.2]],
            "delta_vel": [[0.0, 0.0, 0.5]],
        },
        {
            "command": "turn_left",
            "transition_type": "turn_onset",
            "polar_invalid": 1.0,
            "detection_source": "haar",
        },
    ]
    dataset.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    manifest = {
        "fps": 10,
        "dt": 0.1,
        "statistics": {"episode_reports": [{"episode": "ep", "fps": 10}]},
    }
    Path(str(dataset) + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    stats = compute_dataset_stats(dataset)
    assert stats["command_distribution"] == {"forward": 1, "turn_left": 1}
    assert stats["transition_type_distribution"] == {"steady_forward": 1, "turn_onset": 1}
    assert stats["polar"]["valid_rate"] == 0.5
    assert stats["detection_source_distribution"] == {"haar": 1, "omdet": 1}
    assert stats["polar_by_detection_source"]["omdet"]["max_distance_bin_rate"] == 1.0
    assert stats["polar_by_detection_source"]["haar"]["valid_rate"] == 0.0
    assert stats["fps"]["consistent"] is True
    assert stats["delta_ranges"]["delta_pos"]["min"] == -0.2
    assert stats["delta_ranges"]["delta_vel"]["max"] == 0.5
