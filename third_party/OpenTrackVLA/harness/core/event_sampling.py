"""Transition-event stratified sampling helpers."""

from __future__ import annotations

from collections import Counter


STEADY_TYPE = "steady_forward"


def compute_event_sampling_weights(
    transition_types,
    target_event_fraction: float = 0.4,
    max_weight: float = 10.0,
):
    """Return per-sample weights, balancing non-steady event strata.

    The steady stratum keeps weight 1.  When events are below the requested
    mass, each present non-steady transition type receives an equal share of
    the 40% target, with every individual event weight capped at 10x.
    """

    labels = [str(value) for value in transition_types]
    if not labels:
        return []
    counts = Counter(labels)
    steady_count = counts.get(STEADY_TYPE, 0)
    event_count = len(labels) - steady_count
    if event_count == 0 or event_count / len(labels) >= float(target_event_fraction):
        return [1.0] * len(labels)

    event_classes = sorted(label for label in counts if label != STEADY_TYPE)
    if not event_classes or steady_count == 0:
        return [1.0] * len(labels)
    steady_target = 1.0 - float(target_event_fraction)
    per_class_target = float(target_event_fraction) / len(event_classes)
    class_weights = {
        label: max(
            1.0,
            min(
                float(max_weight),
                (per_class_target / counts[label]) / (steady_target / steady_count),
            ),
        )
        for label in event_classes
    }
    return [1.0 if label == STEADY_TYPE else class_weights[label] for label in labels]


def weighted_event_fraction(transition_types, weights):
    denominator = sum(float(weight) for weight in weights)
    if denominator <= 0:
        return 0.0
    return sum(
        float(weight)
        for label, weight in zip(transition_types, weights)
        if str(label) != STEADY_TYPE
    ) / denominator
