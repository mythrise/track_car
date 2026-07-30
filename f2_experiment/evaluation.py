"""Fail-closed smoke gates for the isolated F2 experiment.

This module implements the independently decidable parts of Fable's F2
contract.  It consumes already-produced telemetry; it does not load data,
models, checkpoints, validation artifacts, or the sealed internal test.

The important clock convention is ``u_pre=optimizer_updates_completed``.  A
128-update smoke therefore has clocks 0 through 127, and G6's gradient window
is exactly 8 through 127 (120 points).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from statistics import median
from typing import Any

from .support import (
    ARCHITECTURE_LOCK,
    CHANGE_THRESHOLD,
    F2ContractError,
    TURN_TRANSITION_TYPES,
)


SMOKE_CLOCK = tuple(range(128))
G6_GRADIENT_CLOCK = tuple(range(8, 128))
G6_GRADIENT_POINTS = 120
G6_POSITIVE_PROJECTION_MIN = 108
G6_BSTAR_RATIO_MAX = 1.5
G6_FALLBACK_RATIO_MAX = 0.75
G6_COSINE_MEDIAN_MIN = 0.6
G6_AUX_REACHABLE_MIN = 127
G6_TRACK_REACHABLE_MIN = 120

G7_PER_STREAM_MAX = 0.5 + 1e-4
G7_TOTAL_MEDIAN_MAX = 1.0
G7_SATURATION_LEVEL = 0.99
G7_SATURATION_RATE_MAX = 0.05

G8_TAU = 1e-6
G8_LOGGED_CEILING_FACTOR = 1.10

G9_RANGE_VIOLATION_RATE_MAX = 0.05
G9_RECONSTRUCTION_ERROR_MAX = 1e-6
G9_DRIFT_RATIO_MAX = 2.0

STRATA = ("overall", "change", "turn", "other")
EVAL_MODES = ("logged", "self")


class GateContractError(F2ContractError):
    """Raised when gate telemetry is malformed, unsupported, or nonfinite."""


@dataclass(frozen=True)
class GateReceipt:
    """Machine-serializable result for one preregistered gate."""

    gate_id: str
    passed: bool
    checks: Mapping[str, Mapping[str, Any]]
    metrics: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    contract: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_class": "f2_preformal_smoke_gate",
            "architecture_lock": ARCHITECTURE_LOCK,
            "gate_id": self.gate_id,
            "valid_input": True,
            "passed": self.passed,
            "status": "PASS" if self.passed else "FAIL",
            "decision": "GO" if self.passed else "STOP",
            "checks": {name: dict(value) for name, value in self.checks.items()},
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "contract": dict(self.contract),
        }


@dataclass(frozen=True)
class StratifiedLossSummary:
    """Float64 loss means and support counts on the fixed evaluation support."""

    means: Mapping[str, float]
    counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.means) != set(STRATA) or set(self.counts) != set(STRATA):
            raise GateContractError(
                f"stratified loss keys must be exactly {list(STRATA)!r}"
            )
        for stratum in STRATA:
            count = _nonnegative_int(self.counts[stratum], f"counts.{stratum}")
            if count == 0:
                raise GateContractError(
                    f"G8_ZERO_SUPPORT: stratum {stratum!r} has zero rows"
                )
            loss = _finite_float(self.means[stratum], f"means.{stratum}")
            if loss < 0.0:
                raise GateContractError(f"means.{stratum} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accumulator": "IEEE-754 binary64 math.fsum",
            "means": {name: float(self.means[name]) for name in STRATA},
            "counts": {name: int(self.counts[name]) for name in STRATA},
        }


@dataclass(frozen=True)
class G6Update:
    """One pre-optimizer-step gradient telemetry point for S-CTRL."""

    u_pre: int
    aux_reachable: bool
    track_reachable: bool
    cosine_total_track: float | None = None
    signed_projection: float | None = None
    aux_track_ratio: float | None = None
    per_aux_ratios: Mapping[str, float] | None = None


@dataclass(frozen=True)
class G7Update:
    """One update's bounded-fusion telemetry.

    Values may be Python numerics, NumPy arrays, or detached tensor-like
    objects.  Per-method scale values must each be scalar because the
    preregistered saturation denominator is updates times streams.  The
    previous-stream ``abs_tanh_s_prev`` scale is one scalar per update with an
    all-or-none presence rule; its saturation denominator is updates.
    """

    u_pre: int
    per_method_over_base: Mapping[str, Any]
    total_method_over_base: Any
    abs_tanh_method_scales: Mapping[str, Any]
    r_prev: Any | None = None
    abs_tanh_s_prev: Any | None = None


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (RuntimeError, TypeError, ValueError) as error:
                raise GateContractError(f"{label} is not a scalar numeric") from error
        if isinstance(value, bool) or not isinstance(value, Real):
            raise GateContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GateContractError(f"{label} is nonfinite")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GateContractError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise GateContractError(f"{label} must be nonnegative")
    return result


def _flatten_numeric(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        return (_finite_float(value, label),)
    if hasattr(value, "detach") and hasattr(value, "numel"):
        try:
            detached = value.detach()
            flattened = detached.cpu().reshape(-1).tolist()
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise GateContractError(f"{label} tensor cannot be materialized") from error
        return _flatten_numeric(flattened, label)
    if hasattr(value, "tolist") and not isinstance(value, Mapping):
        try:
            materialized = value.tolist()
        except (RuntimeError, TypeError, ValueError) as error:
            raise GateContractError(f"{label} array cannot be materialized") from error
        if materialized is not value:
            return _flatten_numeric(materialized, label)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[float] = []
        for index, item in enumerate(value):
            result.extend(_flatten_numeric(item, f"{label}[{index}]"))
        if not result:
            raise GateContractError(f"{label} has zero numeric support")
        return tuple(result)
    raise GateContractError(f"{label} must be numeric or a numeric sequence")


def _scalar_numeric(value: Any, label: str) -> float:
    values = _flatten_numeric(value, label)
    if len(values) != 1:
        raise GateContractError(f"{label} must contain exactly one scalar")
    return values[0]


def _bool_mask(value: Any, length: int, label: str) -> tuple[bool, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise GateContractError(f"{label} must be a boolean sequence")
    if len(value) != length:
        raise GateContractError(
            f"{label} length {len(value)} does not match loss length {length}"
        )
    result: list[bool] = []
    for index, item in enumerate(value):
        if not isinstance(item, bool):
            raise GateContractError(f"{label}[{index}] must be boolean")
        result.append(item)
    return tuple(result)


def _mean64(values: Sequence[float], label: str) -> float:
    if not values:
        raise GateContractError(f"{label} has zero support")
    result = math.fsum(float(value) for value in values) / len(values)
    if not math.isfinite(result):
        raise GateContractError(f"{label} mean is nonfinite")
    return float(result)


def _median64(values: Sequence[float], label: str) -> float:
    if not values:
        raise GateContractError(f"{label} has zero support")
    result = float(median(float(value) for value in values))
    if not math.isfinite(result):
        raise GateContractError(f"{label} median is nonfinite")
    return result


def _check(
    passed: bool,
    *,
    observed: Any,
    comparator: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
    }


def _make_receipt(
    gate_id: str,
    *,
    checks: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> GateReceipt:
    return GateReceipt(
        gate_id=gate_id,
        passed=all(bool(check.get("passed")) for check in checks.values()),
        checks=dict(checks),
        metrics=dict(metrics),
        thresholds=dict(thresholds),
        contract=dict(contract),
    )


def _action3(value: Any, label: str) -> tuple[float, float, float]:
    values = _flatten_numeric(value, label)
    if len(values) != 3:
        raise GateContractError(f"{label} must contain exactly three axes")
    return (values[0], values[1], values[2])


def strata_masks_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[bool, ...]]:
    """Derive the exact corrigendum-2 change/turn/other memberships."""

    change: list[bool] = []
    turn: list[bool] = []
    other: list[bool] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GateContractError(f"row {index} must be a mapping")
        previous = _action3(row.get("prev_action"), f"row {index}.prev_action")
        horizon = row.get("step_actions")
        if not isinstance(horizon, Sequence) or isinstance(
            horizon, (str, bytes, bytearray)
        ):
            raise GateContractError(f"row {index}.step_actions must be a sequence")
        if not horizon:
            raise GateContractError(f"row {index}.step_actions is empty")
        first = _action3(horizon[0], f"row {index}.step_actions[0]")
        transition = row.get("transition_type")
        if not isinstance(transition, str):
            raise GateContractError(f"row {index}.transition_type must be a string")
        change.append(
            max(abs(first[0] - previous[0]), abs(first[2] - previous[2]))
            > CHANGE_THRESHOLD
        )
        turn.append(transition in TURN_TRANSITION_TYPES)
        other.append(transition == "other")
    return {
        "change": tuple(change),
        "turn": tuple(turn),
        "other": tuple(other),
    }


def aggregate_stratified_losses(
    losses: Any,
    *,
    change_mask: Sequence[bool],
    turn_mask: Sequence[bool],
    other_mask: Sequence[bool],
) -> StratifiedLossSummary:
    """Aggregate fixed-support losses with binary64 ``math.fsum`` means."""

    values = _flatten_numeric(losses, "losses")
    for index, value in enumerate(values):
        if value < 0.0:
            raise GateContractError(f"losses[{index}] must be nonnegative")
    masks = {
        "overall": tuple(True for _ in values),
        "change": _bool_mask(change_mask, len(values), "change_mask"),
        "turn": _bool_mask(turn_mask, len(values), "turn_mask"),
        "other": _bool_mask(other_mask, len(values), "other_mask"),
    }
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for stratum in STRATA:
        selected = [loss for loss, included in zip(values, masks[stratum]) if included]
        if not selected:
            raise GateContractError(
                f"G8_ZERO_SUPPORT: stratum {stratum!r} has zero rows"
            )
        counts[stratum] = len(selected)
        means[stratum] = _mean64(selected, f"losses.{stratum}")
    return StratifiedLossSummary(means=means, counts=counts)


def aggregate_row_losses(
    losses: Any,
    rows: Sequence[Mapping[str, Any]],
) -> StratifiedLossSummary:
    """Convenience wrapper using the frozen row-level stratum definitions."""

    masks = strata_masks_from_rows(rows)
    return aggregate_stratified_losses(
        losses,
        change_mask=masks["change"],
        turn_mask=masks["turn"],
        other_mask=masks["other"],
    )


def _coerce_g6_update(value: Any, index: int) -> G6Update:
    if isinstance(value, G6Update):
        return value
    if not isinstance(value, Mapping):
        raise GateContractError(f"G6 update {index} must be a mapping")
    try:
        return G6Update(
            u_pre=value["u_pre"],
            aux_reachable=value["aux_reachable"],
            track_reachable=value["track_reachable"],
            cosine_total_track=value.get("cosine_total_track"),
            signed_projection=value.get("signed_projection"),
            aux_track_ratio=value.get("aux_track_ratio"),
            per_aux_ratios=value.get("per_aux_ratios"),
        )
    except KeyError as error:
        raise GateContractError(
            f"G6 update {index} is missing {error.args[0]!r}"
        ) from error


def evaluate_g6(
    updates: Sequence[G6Update | Mapping[str, Any]],
    *,
    block_mode: str,
) -> GateReceipt:
    """Evaluate the corrected G6 clock, reachability, and gradient geometry."""

    if block_mode not in {"bstar", "per_aux"}:
        raise GateContractError("G6 block_mode must be 'bstar' or 'per_aux'")
    records = tuple(
        _coerce_g6_update(value, index) for index, value in enumerate(updates)
    )
    if len(records) != len(SMOKE_CLOCK):
        raise GateContractError(
            f"G6 requires exactly {len(SMOKE_CLOCK)} updates, got {len(records)}"
        )
    declared_clocks = tuple(
        _nonnegative_int(record.u_pre, f"G6[{index}].u_pre")
        for index, record in enumerate(records)
    )
    if declared_clocks != SMOKE_CLOCK:
        raise GateContractError(
            "G6 clock must be u_pre=0..127 in order; updates 8..128 is forbidden"
        )
    clocks: list[int] = []
    aux_reachable: list[bool] = []
    track_reachable: list[bool] = []
    window_ratios: list[float] = []
    window_per_aux: dict[str, list[float]] = {}
    window_cosines: list[float] = []
    window_projections: list[float] = []

    for index, record in enumerate(records):
        u_pre = declared_clocks[index]
        clocks.append(u_pre)
        if not isinstance(record.aux_reachable, bool):
            raise GateContractError(f"G6[{index}].aux_reachable must be boolean")
        if not isinstance(record.track_reachable, bool):
            raise GateContractError(f"G6[{index}].track_reachable must be boolean")
        aux_reachable.append(record.aux_reachable)
        track_reachable.append(record.track_reachable)

        supplied_ratio_values: list[tuple[str, float]] = []
        if record.aux_track_ratio is not None:
            supplied_ratio_values.append(
                (
                    "aux_track_ratio",
                    _finite_float(
                        record.aux_track_ratio, f"G6[{index}].aux_track_ratio"
                    ),
                )
            )
        if record.per_aux_ratios is not None:
            if (
                not isinstance(record.per_aux_ratios, Mapping)
                or not record.per_aux_ratios
            ):
                raise GateContractError(
                    f"G6[{index}].per_aux_ratios must be a nonempty mapping"
                )
            for name, raw_ratio in record.per_aux_ratios.items():
                if not isinstance(name, str) or not name:
                    raise GateContractError(
                        f"G6[{index}].per_aux_ratios has an invalid name"
                    )
                supplied_ratio_values.append(
                    (
                        f"per_aux_ratios.{name}",
                        _finite_float(raw_ratio, f"G6[{index}].per_aux_ratios.{name}"),
                    )
                )
        for name, ratio in supplied_ratio_values:
            if ratio < 0.0:
                raise GateContractError(f"G6[{index}].{name} must be nonnegative")

        cosine: float | None = None
        if record.cosine_total_track is not None:
            cosine = _finite_float(
                record.cosine_total_track,
                f"G6[{index}].cosine_total_track",
            )
            if not -1.0 <= cosine <= 1.0:
                raise GateContractError(
                    f"G6[{index}].cosine_total_track must be in [-1,1]"
                )
        projection: float | None = None
        if record.signed_projection is not None:
            projection = _finite_float(
                record.signed_projection,
                f"G6[{index}].signed_projection",
            )

        if u_pre in G6_GRADIENT_CLOCK:
            if cosine is None or projection is None:
                raise GateContractError(
                    f"G6[{index}] is missing gradient-window geometry"
                )
            window_cosines.append(cosine)
            window_projections.append(projection)
            if block_mode == "bstar":
                if record.aux_track_ratio is None:
                    raise GateContractError(
                        f"G6[{index}] is missing B* aux_track_ratio"
                    )
                if record.per_aux_ratios is not None:
                    raise GateContractError(f"G6[{index}] mixes B* and per-aux ratios")
                window_ratios.append(
                    _finite_float(
                        record.aux_track_ratio,
                        f"G6[{index}].aux_track_ratio",
                    )
                )
            else:
                if record.aux_track_ratio is not None:
                    raise GateContractError(f"G6[{index}] mixes B* and per-aux ratios")
                if record.per_aux_ratios is None:
                    raise GateContractError(
                        f"G6[{index}] is missing fallback per_aux_ratios"
                    )
                names = set(record.per_aux_ratios)
                if window_per_aux and names != set(window_per_aux):
                    raise GateContractError(
                        "G6 fallback aux names change within the gradient window"
                    )
                for name, raw_ratio in record.per_aux_ratios.items():
                    ratio = _finite_float(
                        raw_ratio, f"G6[{index}].per_aux_ratios.{name}"
                    )
                    window_per_aux.setdefault(name, []).append(ratio)

    if len(window_cosines) != G6_GRADIENT_POINTS:
        raise GateContractError("G6 gradient window is not exactly 120 points")

    aux_reachable_count = sum(aux_reachable)
    track_reachable_count = sum(track_reachable)
    late_unreachable = [
        clock
        for clock, aux_ok, track_ok in zip(clocks, aux_reachable, track_reachable)
        if clock >= 8 and (not aux_ok or not track_ok)
    ]
    cosine_median = _median64(window_cosines, "G6 cosine")
    positive_projection_count = sum(value > 0.0 for value in window_projections)

    checks: dict[str, Mapping[str, Any]] = {
        "aux_reachability": _check(
            aux_reachable_count >= G6_AUX_REACHABLE_MIN,
            observed=aux_reachable_count,
            comparator=">=",
            threshold=G6_AUX_REACHABLE_MIN,
        ),
        "track_reachability": _check(
            track_reachable_count >= G6_TRACK_REACHABLE_MIN,
            observed=track_reachable_count,
            comparator=">=",
            threshold=G6_TRACK_REACHABLE_MIN,
        ),
        "zero_grad_clock": _check(
            not late_unreachable,
            observed=late_unreachable,
            comparator="subset_of",
            threshold=list(range(8)),
        ),
        "cosine_median": _check(
            cosine_median >= G6_COSINE_MEDIAN_MIN,
            observed=cosine_median,
            comparator=">=",
            threshold=G6_COSINE_MEDIAN_MIN,
        ),
        "positive_projection": _check(
            positive_projection_count >= G6_POSITIVE_PROJECTION_MIN,
            observed=positive_projection_count,
            comparator=">=",
            threshold=G6_POSITIVE_PROJECTION_MIN,
        ),
    }

    ratio_metrics: Any
    if block_mode == "bstar":
        ratio_median = _median64(window_ratios, "G6 B* aux/track ratio")
        checks["aux_track_ratio_median"] = _check(
            ratio_median <= G6_BSTAR_RATIO_MAX,
            observed=ratio_median,
            comparator="<=",
            threshold=G6_BSTAR_RATIO_MAX,
        )
        ratio_metrics = ratio_median
    else:
        if not window_per_aux:
            raise GateContractError("G6 fallback has no aux gradient blocks")
        ratio_metrics = {
            name: _median64(values, f"G6 {name} ratio")
            for name, values in sorted(window_per_aux.items())
        }
        for name, ratio_median in ratio_metrics.items():
            checks[f"aux_track_ratio_median.{name}"] = _check(
                ratio_median <= G6_FALLBACK_RATIO_MAX,
                observed=ratio_median,
                comparator="<=",
                threshold=G6_FALLBACK_RATIO_MAX,
            )

    return _make_receipt(
        "G6",
        checks=checks,
        metrics={
            "block_mode": block_mode,
            "clock_first": clocks[0],
            "clock_last": clocks[-1],
            "gradient_window": [8, 127],
            "gradient_window_points": len(window_cosines),
            "aux_reachable_count": aux_reachable_count,
            "track_reachable_count": track_reachable_count,
            "late_unreachable_clocks": late_unreachable,
            "ratio_median": ratio_metrics,
            "cosine_median": cosine_median,
            "positive_projection_count": positive_projection_count,
        },
        thresholds={
            "bstar_ratio_max": G6_BSTAR_RATIO_MAX,
            "fallback_per_aux_ratio_max": G6_FALLBACK_RATIO_MAX,
            "cosine_median_min": G6_COSINE_MEDIAN_MIN,
            "positive_projection_min": G6_POSITIVE_PROJECTION_MIN,
            "aux_reachable_min": G6_AUX_REACHABLE_MIN,
            "track_reachable_min": G6_TRACK_REACHABLE_MIN,
        },
        contract={
            "arm": "S-CTRL",
            "clock": "u_pre=optimizer_updates_completed",
            "domain": [0, 127],
            "warmup": "u_pre<16",
            "gradient_window": [8, 127],
            "zero_grad_allowed_only": [0, 7],
            "source": "Fable implementation corrigendum + corrigendum-2 clock",
        },
    )


def _coerce_g7_update(value: Any, index: int) -> G7Update:
    if isinstance(value, G7Update):
        return value
    if not isinstance(value, Mapping):
        raise GateContractError(f"G7 update {index} must be a mapping")
    required = (
        "u_pre",
        "per_method_over_base",
        "total_method_over_base",
        "abs_tanh_method_scales",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise GateContractError(f"G7 update {index} is missing {missing!r}")
    return G7Update(
        u_pre=value["u_pre"],
        per_method_over_base=value["per_method_over_base"],
        total_method_over_base=value["total_method_over_base"],
        abs_tanh_method_scales=value["abs_tanh_method_scales"],
        r_prev=value.get("r_prev"),
        abs_tanh_s_prev=value.get("abs_tanh_s_prev"),
    )


def evaluate_g7(
    updates: Sequence[G7Update | Mapping[str, Any]],
) -> GateReceipt:
    """Evaluate constructive stream bounds and the exact saturation fraction."""

    records = tuple(
        _coerce_g7_update(value, index) for index, value in enumerate(updates)
    )
    if len(records) != len(SMOKE_CLOCK):
        raise GateContractError(
            f"G7 requires exactly {len(SMOKE_CLOCK)} updates, got {len(records)}"
        )
    declared_clocks = tuple(
        _nonnegative_int(record.u_pre, f"G7[{index}].u_pre")
        for index, record in enumerate(records)
    )
    if declared_clocks != SMOKE_CLOCK:
        raise GateContractError("G7 clock must be u_pre=0..127 in order")
    clocks: list[int] = []
    per_stream_values: dict[str, list[float]] = {}
    total_values: list[float] = []
    scale_values: dict[str, list[float]] = {}
    r_prev_values: list[float] = []
    r_prev_presence: list[bool] = []
    prev_scale_values: list[float] = []
    prev_scale_presence: list[bool] = []

    for index, record in enumerate(records):
        clocks.append(declared_clocks[index])
        if (
            not isinstance(record.per_method_over_base, Mapping)
            or not record.per_method_over_base
        ):
            raise GateContractError(
                f"G7[{index}].per_method_over_base must be a nonempty mapping"
            )
        if (
            not isinstance(record.abs_tanh_method_scales, Mapping)
            or not record.abs_tanh_method_scales
        ):
            raise GateContractError(
                f"G7[{index}].abs_tanh_method_scales must be a nonempty mapping"
            )
        ratio_names = set(record.per_method_over_base)
        scale_names = set(record.abs_tanh_method_scales)
        if ratio_names != scale_names:
            raise GateContractError(
                f"G7[{index}] method ratio/scale stream names differ"
            )
        if per_stream_values and ratio_names != set(per_stream_values):
            raise GateContractError("G7 method stream names change across updates")

        for name in sorted(ratio_names):
            if not isinstance(name, str) or not name:
                raise GateContractError(f"G7[{index}] has an invalid stream name")
            ratios = _flatten_numeric(
                record.per_method_over_base[name],
                f"G7[{index}].per_method_over_base.{name}",
            )
            if any(value < 0.0 for value in ratios):
                raise GateContractError(
                    f"G7[{index}].per_method_over_base.{name} must be nonnegative"
                )
            per_stream_values.setdefault(name, []).extend(ratios)

            scale = _scalar_numeric(
                record.abs_tanh_method_scales[name],
                f"G7[{index}].abs_tanh_method_scales.{name}",
            )
            if not 0.0 <= scale <= 1.0:
                raise GateContractError(
                    f"G7[{index}].abs_tanh_method_scales.{name} must be in [0,1]"
                )
            scale_values.setdefault(name, []).append(scale)

        totals = _flatten_numeric(
            record.total_method_over_base,
            f"G7[{index}].total_method_over_base",
        )
        if any(value < 0.0 for value in totals):
            raise GateContractError(
                f"G7[{index}].total_method_over_base must be nonnegative"
            )
        total_values.extend(totals)

        r_prev_presence.append(record.r_prev is not None)
        if record.r_prev is not None:
            values = _flatten_numeric(record.r_prev, f"G7[{index}].r_prev")
            if any(value < 0.0 for value in values):
                raise GateContractError(f"G7[{index}].r_prev must be nonnegative")
            r_prev_values.extend(values)

        prev_scale_presence.append(record.abs_tanh_s_prev is not None)
        if record.abs_tanh_s_prev is not None:
            prev_scale = _scalar_numeric(
                record.abs_tanh_s_prev, f"G7[{index}].abs_tanh_s_prev"
            )
            if not 0.0 <= prev_scale <= 1.0:
                raise GateContractError(
                    f"G7[{index}].abs_tanh_s_prev must be in [0,1]"
                )
            prev_scale_values.append(prev_scale)

    if any(r_prev_presence) and not all(r_prev_presence):
        raise GateContractError("G7 r_prev telemetry is present for only some updates")
    if any(prev_scale_presence) and not all(prev_scale_presence):
        raise GateContractError(
            "G7 abs_tanh_s_prev telemetry is present for only some updates"
        )

    per_stream_max = {
        name: max(values) for name, values in sorted(per_stream_values.items())
    }
    total_median = _median64(total_values, "G7 total method/base")
    saturation_count = sum(
        value >= G7_SATURATION_LEVEL
        for values in scale_values.values()
        for value in values
    )
    saturation_denominator = len(records) * len(scale_values)
    if saturation_denominator == 0:
        raise GateContractError("G7 saturation denominator is zero")
    saturation_rate = saturation_count / saturation_denominator

    checks: dict[str, Mapping[str, Any]] = {}
    for name, maximum in per_stream_max.items():
        checks[f"per_stream_bound.{name}"] = _check(
            maximum <= G7_PER_STREAM_MAX,
            observed=maximum,
            comparator="<=",
            threshold=G7_PER_STREAM_MAX,
        )
    checks["total_method_over_base_median"] = _check(
        total_median <= G7_TOTAL_MEDIAN_MAX,
        observed=total_median,
        comparator="<=",
        threshold=G7_TOTAL_MEDIAN_MAX,
    )
    checks["method_scale_saturation_rate"] = _check(
        saturation_rate < G7_SATURATION_RATE_MAX,
        observed=saturation_rate,
        comparator="<",
        threshold=G7_SATURATION_RATE_MAX,
    )

    r_prev_max: float | None = None
    if r_prev_values:
        r_prev_max = max(r_prev_values)
        checks["prev_stream_bound"] = _check(
            r_prev_max <= G7_PER_STREAM_MAX,
            observed=r_prev_max,
            comparator="<=",
            threshold=G7_PER_STREAM_MAX,
        )

    prev_scale_saturation_rate: float | None = None
    abs_tanh_s_prev_max: float | None = None
    if prev_scale_values:
        prev_scale_saturation_rate = sum(
            value >= G7_SATURATION_LEVEL for value in prev_scale_values
        ) / len(records)
        abs_tanh_s_prev_max = max(prev_scale_values)
        checks["prev_scale_saturation_rate"] = _check(
            prev_scale_saturation_rate < G7_SATURATION_RATE_MAX,
            observed=prev_scale_saturation_rate,
            comparator="<",
            threshold=G7_SATURATION_RATE_MAX,
        )

    return _make_receipt(
        "G7",
        checks=checks,
        metrics={
            "updates": len(records),
            "streams": sorted(per_stream_values),
            "per_stream_max": per_stream_max,
            "total_method_over_base_median": total_median,
            "method_scale_saturation_count": saturation_count,
            "method_scale_saturation_denominator": saturation_denominator,
            "method_scale_saturation_rate": saturation_rate,
            "r_prev_max": r_prev_max,
            "prev_scale_saturation_rate": prev_scale_saturation_rate,
            "abs_tanh_s_prev_max": abs_tanh_s_prev_max,
        },
        thresholds={
            "per_stream_max": G7_PER_STREAM_MAX,
            "total_method_over_base_median_max": G7_TOTAL_MEDIAN_MAX,
            "saturation_indicator": f"abs(tanh(s_m))>={G7_SATURATION_LEVEL}",
            "saturation_rate_max_exclusive": G7_SATURATION_RATE_MAX,
        },
        contract={
            "saturation_denominator": "updates*method_streams",
            "prev_saturation_denominator": "updates",
            "total_gate": "median over all emitted total ratios",
            "nonfinite": "hard_stop",
            "source": "Fable implementation corrigendum + corrigendum-2 prev bound",
        },
    )


def _coerce_summary(value: Any, label: str) -> StratifiedLossSummary:
    if isinstance(value, StratifiedLossSummary):
        return value
    if not isinstance(value, Mapping):
        raise GateContractError(f"{label} must be a StratifiedLossSummary")
    if "means" not in value or "counts" not in value:
        raise GateContractError(f"{label} is missing means/counts")
    means = value["means"]
    counts = value["counts"]
    if not isinstance(means, Mapping) or not isinstance(counts, Mapping):
        raise GateContractError(f"{label}.means/counts must be mappings")
    normalized_means = {
        stratum: _finite_float(means.get(stratum), f"{label}.means.{stratum}")
        for stratum in STRATA
    }
    normalized_counts = {
        stratum: _nonnegative_int(counts.get(stratum), f"{label}.counts.{stratum}")
        for stratum in STRATA
    }
    return StratifiedLossSummary(
        means=normalized_means,
        counts=normalized_counts,
    )


def _snapshot_mode(
    snapshot: Mapping[str, Any], mode: str, label: str
) -> StratifiedLossSummary:
    if not isinstance(snapshot, Mapping):
        raise GateContractError(f"{label} must be a mapping")
    if mode not in snapshot:
        raise GateContractError(f"{label} is missing mode {mode!r}")
    return _coerce_summary(snapshot[mode], f"{label}.{mode}")


def evaluate_g8(
    *,
    s_self_update0: Mapping[str, Any],
    s_self_update128: Mapping[str, Any],
    s_ctrl_update128: Mapping[str, Any],
) -> GateReceipt:
    """Evaluate fixed-support causal improvement, exposure gap, and ceiling."""

    summaries = {
        "s_self_update0_logged": _snapshot_mode(
            s_self_update0, "logged", "s_self_update0"
        ),
        "s_self_update0_self": _snapshot_mode(s_self_update0, "self", "s_self_update0"),
        "s_self_update128_logged": _snapshot_mode(
            s_self_update128, "logged", "s_self_update128"
        ),
        "s_self_update128_self": _snapshot_mode(
            s_self_update128, "self", "s_self_update128"
        ),
        "s_ctrl_update128_logged": _snapshot_mode(
            s_ctrl_update128, "logged", "s_ctrl_update128"
        ),
        "s_ctrl_update128_self": _snapshot_mode(
            s_ctrl_update128, "self", "s_ctrl_update128"
        ),
    }
    reference_counts = dict(next(iter(summaries.values())).counts)
    for label, summary in summaries.items():
        if dict(summary.counts) != reference_counts:
            raise GateContractError(
                f"G8 fixed-support counts differ at {label}: "
                f"{dict(summary.counts)!r} != {reference_counts!r}"
            )

    self0 = summaries["s_self_update0_self"]
    self128 = summaries["s_self_update128_self"]
    logged0 = summaries["s_self_update0_logged"]
    logged128 = summaries["s_self_update128_logged"]
    ctrl_self128 = summaries["s_ctrl_update128_self"]
    ctrl_logged128 = summaries["s_ctrl_update128_logged"]

    improvements = {
        stratum: float(self128.means[stratum] - self0.means[stratum])
        for stratum in STRATA
    }
    gaps = {
        "s_self_update0": float(self0.means["overall"] - logged0.means["overall"]),
        "s_self_update128": float(
            self128.means["overall"] - logged128.means["overall"]
        ),
        "s_ctrl_update128": float(
            ctrl_self128.means["overall"] - ctrl_logged128.means["overall"]
        ),
    }
    if any(
        not math.isfinite(value) for value in (*improvements.values(), *gaps.values())
    ):
        raise GateContractError("G8 derived improvement/gap is nonfinite")
    logged_ceiling = float(G8_LOGGED_CEILING_FACTOR * ctrl_logged128.means["overall"])
    if not math.isfinite(logged_ceiling):
        raise GateContractError("G8 logged ceiling is nonfinite")

    checks: dict[str, Mapping[str, Any]] = {}
    for stratum, delta in improvements.items():
        checks[f"self_improvement.{stratum}"] = _check(
            delta <= -G8_TAU,
            observed=delta,
            comparator="<=",
            threshold=-G8_TAU,
        )
    checks["gap_contraction"] = _check(
        gaps["s_self_update128"] <= gaps["s_self_update0"] - G8_TAU,
        observed=gaps["s_self_update128"],
        comparator="<=",
        threshold=gaps["s_self_update0"] - G8_TAU,
    )
    checks["gap_below_s_ctrl_update128"] = _check(
        gaps["s_self_update128"] <= gaps["s_ctrl_update128"] - G8_TAU,
        observed=gaps["s_self_update128"],
        comparator="<=",
        threshold=gaps["s_ctrl_update128"] - G8_TAU,
    )
    checks["logged_ceiling"] = _check(
        logged128.means["overall"] <= logged_ceiling,
        observed=logged128.means["overall"],
        comparator="<=",
        threshold=logged_ceiling,
    )

    per_stratum_gaps = {
        "s_self_update0": {
            stratum: float(self0.means[stratum] - logged0.means[stratum])
            for stratum in STRATA
        },
        "s_self_update128": {
            stratum: float(self128.means[stratum] - logged128.means[stratum])
            for stratum in STRATA
        },
        "s_ctrl_update128": {
            stratum: float(ctrl_self128.means[stratum] - ctrl_logged128.means[stratum])
            for stratum in STRATA
        },
    }

    return _make_receipt(
        "G8",
        checks=checks,
        metrics={
            "accumulator": "IEEE-754 binary64 math.fsum",
            "support_counts": reference_counts,
            "self_mode_improvement_delta": improvements,
            "overall_gaps_self_minus_logged": gaps,
            "diagnostic_per_stratum_gaps_self_minus_logged": per_stratum_gaps,
            "s_self_update128_logged_overall": logged128.means["overall"],
            "s_ctrl_update128_logged_overall": ctrl_logged128.means["overall"],
        },
        thresholds={
            "improvement_delta_max": -G8_TAU,
            "gap_strictness_tau": G8_TAU,
            "logged_ceiling_factor": G8_LOGGED_CEILING_FACTOR,
        },
        contract={
            "support": "EVAL-FIX",
            "gated_improvement_strata": list(STRATA),
            "gap_gate_stratum": "overall",
            "logged_ceiling_stratum": "overall",
            "zero_support": "hard_stop",
            "nonfinite": "hard_stop",
            "source": "Fable v5 implementation corrigendum",
        },
    )


def evaluate_g9(
    *,
    expected_static_resets: int,
    observed_static_resets: int,
    nonfinite_reset_count: int,
    range_violation_count: int,
    range_observation_count: int,
    reconstruction_errors: Any,
    first_quartile_self_errors: Any,
    last_quartile_self_errors: Any,
) -> GateReceipt:
    """Evaluate recurrence reset, range, reconstruction, and drift guards."""

    expected_resets = _nonnegative_int(
        expected_static_resets, "G9.expected_static_resets"
    )
    observed_resets = _nonnegative_int(
        observed_static_resets, "G9.observed_static_resets"
    )
    nonfinite_resets = _nonnegative_int(
        nonfinite_reset_count, "G9.nonfinite_reset_count"
    )
    violation_count = _nonnegative_int(
        range_violation_count, "G9.range_violation_count"
    )
    observation_count = _nonnegative_int(
        range_observation_count, "G9.range_observation_count"
    )
    if observation_count == 0:
        raise GateContractError("G9 range observation count is zero")
    if violation_count > observation_count:
        raise GateContractError("G9 range violations exceed observations")

    reconstruction = _flatten_numeric(reconstruction_errors, "G9.reconstruction_errors")
    first_errors = _flatten_numeric(
        first_quartile_self_errors, "G9.first_quartile_self_errors"
    )
    last_errors = _flatten_numeric(
        last_quartile_self_errors, "G9.last_quartile_self_errors"
    )
    for label, values in (
        ("first_quartile_self_errors", first_errors),
        ("last_quartile_self_errors", last_errors),
    ):
        if any(value < 0.0 for value in values):
            raise GateContractError(f"G9.{label} must be nonnegative")
    if len(first_errors) != len(last_errors):
        raise GateContractError(
            "G9 first/last quartile supports must have equal row counts"
        )

    range_rate = violation_count / observation_count
    reconstruction_max = max(abs(value) for value in reconstruction)
    first_mean = _mean64(first_errors, "G9 first-quartile self error")
    last_mean = _mean64(last_errors, "G9 last-quartile self error")
    if first_mean == 0.0:
        if last_mean == 0.0:
            drift_ratio = 1.0
        else:
            raise GateContractError(
                "G9 drift ratio is undefined: positive last quartile over zero first quartile"
            )
    else:
        drift_ratio = last_mean / first_mean
    if not math.isfinite(drift_ratio):
        raise GateContractError("G9 drift ratio is nonfinite")

    checks = {
        "static_reset_count": _check(
            observed_resets == expected_resets,
            observed=observed_resets,
            comparator="==",
            threshold=expected_resets,
        ),
        "nonfinite_reset_count": _check(
            nonfinite_resets == 0,
            observed=nonfinite_resets,
            comparator="==",
            threshold=0,
        ),
        "range_violation_rate": _check(
            range_rate < G9_RANGE_VIOLATION_RATE_MAX,
            observed=range_rate,
            comparator="<",
            threshold=G9_RANGE_VIOLATION_RATE_MAX,
        ),
        "reconstruction_error": _check(
            reconstruction_max <= G9_RECONSTRUCTION_ERROR_MAX,
            observed=reconstruction_max,
            comparator="<=",
            threshold=G9_RECONSTRUCTION_ERROR_MAX,
        ),
        "self_drift_ratio": _check(
            drift_ratio <= G9_DRIFT_RATIO_MAX,
            observed=drift_ratio,
            comparator="<=",
            threshold=G9_DRIFT_RATIO_MAX,
        ),
    }
    return _make_receipt(
        "G9",
        checks=checks,
        metrics={
            "expected_static_resets": expected_resets,
            "observed_static_resets": observed_resets,
            "nonfinite_reset_count": nonfinite_resets,
            "range_violation_count": violation_count,
            "range_observation_count": observation_count,
            "range_violation_rate": range_rate,
            "reconstruction_error_max": reconstruction_max,
            "first_quartile_self_error_mean": first_mean,
            "last_quartile_self_error_mean": last_mean,
            "self_drift_ratio": drift_ratio,
        },
        thresholds={
            "nonfinite_reset_count": 0,
            "range_violation_rate_max_exclusive": G9_RANGE_VIOLATION_RATE_MAX,
            "reconstruction_error_max": G9_RECONSTRUCTION_ERROR_MAX,
            "self_drift_ratio_max": G9_DRIFT_RATIO_MAX,
        },
        contract={
            "reset_predicate": (
                "chunk_start|new_sequence|not_frame_adjacent|"
                "mirror_chunk_start|nonfinite_trigger"
            ),
            "static_reset_accounting": "observed==frozen_expected",
            "nonfinite": "hard_stop",
            "source": "Fable v5 implementation corrigendum",
        },
    )


def build_smoke_gate_receipt(*receipts: GateReceipt) -> dict[str, Any]:
    """Combine G6-G9 results without weakening any individual STOP."""

    by_gate: dict[str, GateReceipt] = {}
    for receipt in receipts:
        if not isinstance(receipt, GateReceipt):
            raise GateContractError("combined smoke receipt contains an invalid gate")
        if receipt.gate_id in by_gate:
            raise GateContractError(f"duplicate smoke gate {receipt.gate_id}")
        by_gate[receipt.gate_id] = receipt
    required = {"G6", "G7", "G8", "G9"}
    if set(by_gate) != required:
        raise GateContractError(
            f"combined smoke receipt requires {sorted(required)!r}, "
            f"got {sorted(by_gate)!r}"
        )
    passed = all(by_gate[gate].passed for gate in sorted(required))
    return {
        "schema_version": 1,
        "analysis_class": "f2_preformal_smoke_gates",
        "architecture_lock": ARCHITECTURE_LOCK,
        "valid_input": True,
        "gate_order": ["G6", "G7", "G8", "G9"],
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "decision": "GO" if passed else "STOP",
        "formal_training_authorized": passed,
        "gates": {gate: by_gate[gate].to_dict() for gate in ("G6", "G7", "G8", "G9")},
    }
