"""Shared post-action-filter controller proxy for the isolated F2 experiment.

This is the exact arm-independent recurrence approved by Fable corrigendum-2:
finite check, clamp, rate limit, then EMA.  It is deliberately named a proxy;
it is not the perception-dependent deployment emergency-stop supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from .support import ARCHITECTURE_LOCK, F2ContractError, canonical_json_sha256


ACTION_FILTER_ESTIMAND = "fixed_logged_vision_post_action_filter_self_rollout_proxy"
CONTROLLED_AXES = (0, 2)
PARITY_NAME = "controlled_axis_raw_persistence"


class ControllerContractError(F2ContractError):
    """Raised when the shared action-filter proxy must fail closed."""


@dataclass(frozen=True)
class ActionFilterConfig:
    max_abs: float = 1.0
    max_action_rate: float = 4.0
    dt: float = 0.1
    ema_prev_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in ("max_abs", "max_action_rate", "dt", "ema_prev_weight"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ControllerContractError(
                    f"controller config {name} must be numeric"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ControllerContractError(f"controller config {name} is nonfinite")
        if self.max_abs <= 0:
            raise ControllerContractError("max_abs must be positive")
        if self.max_action_rate <= 0:
            raise ControllerContractError("max_action_rate must be positive")
        if self.dt <= 0:
            raise ControllerContractError("dt must be positive")
        if not 0.0 <= self.ema_prev_weight <= 1.0:
            raise ControllerContractError("ema_prev_weight must be in [0,1]")

    @property
    def max_step_delta(self) -> float:
        return self.max_action_rate * self.dt

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs": self.max_abs,
            "max_action_rate": self.max_action_rate,
            "dt": self.dt,
            "max_step_delta": self.max_step_delta,
            "ema_prev_weight": self.ema_prev_weight,
        }


DEFAULT_CONFIG = ActionFilterConfig()


@dataclass(frozen=True)
class ActionFilterState:
    prev_cmd: tuple[float, float, float]
    ticks: int = 0


@dataclass(frozen=True)
class ActionFilterTransition:
    raw_fy: tuple[float, float]
    scattered: tuple[float, float, float]
    bounded: tuple[float, float, float]
    rate_limited: tuple[float, float, float]
    filtered: tuple[float, float, float]
    prior_cmd: tuple[float, float, float]
    tick: int

    @property
    def sent_action(self) -> tuple[float, float, float]:
        return self.filtered

    @property
    def next_prev_fy(self) -> tuple[float, float]:
        return (self.filtered[0], self.filtered[2])

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_fy": list(self.raw_fy),
            "scattered": list(self.scattered),
            "bounded": list(self.bounded),
            "rate_limited": list(self.rate_limited),
            "filtered": list(self.filtered),
            "prior_cmd": list(self.prior_cmd),
            "tick": self.tick,
        }


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ControllerContractError(f"CTRL_NONFINITE: {label}")
    return result


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ControllerContractError(f"{label} must be a sequence")
    if len(value) != length:
        raise ControllerContractError(f"{label} must have length {length}")
    return tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value))


def scatter_controlled_action(raw_fy: Sequence[float]) -> tuple[float, float, float]:
    forward, yaw = _vector(raw_fy, 2, "raw_fy")
    return (forward, 0.0, yaw)


def controlled_axes(action_xyz: Sequence[float]) -> tuple[float, float]:
    forward, _strafe, yaw = _vector(action_xyz, 3, "action_xyz")
    return (forward, yaw)


def assert_controlled_axis_parity(
    logged_prev_action: Sequence[float], raw_fy: Sequence[float]
) -> None:
    expected = controlled_axes(logged_prev_action)
    actual = _vector(raw_fy, 2, "raw_fy")
    if actual != expected:
        raise ControllerContractError(
            f"{PARITY_NAME} mismatch: actual={actual!r}, expected={expected!r}"
        )


def clamp_stage(
    raw_fy: Sequence[float], max_abs: float = DEFAULT_CONFIG.max_abs
) -> tuple[float, float, float]:
    limit = _finite_float(max_abs, "max_abs")
    if limit <= 0:
        raise ControllerContractError("max_abs must be positive")
    scattered = scatter_controlled_action(raw_fy)
    return tuple(max(-limit, min(limit, value)) for value in scattered)


def controller_contract(config: ActionFilterConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_class": "f2_shared_action_filter_proxy",
        "architecture_lock": ARCHITECTURE_LOCK,
        "estimand_key": ACTION_FILTER_ESTIMAND,
        "not_deployment_sent": True,
        "input": "k0_raw_action_only",
        "controlled_axes": list(CONTROLLED_AXES),
        "controller_scatter": "[forward,0.0,yaw]",
        "stages": ["finite_check", "clamp", "rate_limit", "ema"],
        "state_init_on_reset": "scatter3(logged prev_action)",
        "config": config.to_dict(),
        "nonfinite": "CTRL_NONFINITE->rollout_abort->hard_stop_in_smoke",
        "shared_across": ["SA-B0", "SA-B1", "SA-H*"],
        "external_deployment_only": [
            "confidence_stop",
            "invalid_streak_stop",
            "waypoint_bounds_stop",
            "raw_action_bounds_stop",
        ],
    }


CONTROLLER_CONFIG_CONTRACT_SHA256 = canonical_json_sha256(controller_contract())


def bind_controller_identity(source_sha256: str) -> dict[str, str]:
    """Create the implementation-time source/config binding required by HS10."""

    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ControllerContractError("controller source SHA-256 is invalid")
    return {
        "controller_source_sha256": source_sha256,
        "controller_config_contract_sha256": CONTROLLER_CONFIG_CONTRACT_SHA256,
    }


class ActionFilterController:
    """Deterministic state machine shared byte-for-byte across all smoke arms."""

    def __init__(self, config: ActionFilterConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def reset(self, logged_prev_action: Sequence[float]) -> ActionFilterState:
        logged = _vector(logged_prev_action, 3, "logged_prev_action")
        if any(abs(value) > self.config.max_abs for value in logged):
            raise ControllerContractError(
                "logged_prev_action lies outside the frozen control domain"
            )
        return ActionFilterState(prev_cmd=(logged[0], 0.0, logged[2]), ticks=0)

    def step(
        self,
        state: ActionFilterState,
        raw_fy: Sequence[float],
    ) -> tuple[ActionFilterState, ActionFilterTransition]:
        if not isinstance(state, ActionFilterState):
            raise ControllerContractError("invalid action-filter state")
        if isinstance(state.ticks, bool) or not isinstance(state.ticks, int) or state.ticks < 0:
            raise ControllerContractError("invalid action-filter tick counter")
        prior = _vector(state.prev_cmd, 3, "state.prev_cmd")
        raw = _vector(raw_fy, 2, "raw_fy")
        scattered = scatter_controlled_action(raw)
        bounded = tuple(
            max(-self.config.max_abs, min(self.config.max_abs, value))
            for value in scattered
        )
        max_delta = self.config.max_step_delta
        rate_limited = tuple(
            previous + max(-max_delta, min(max_delta, target - previous))
            for previous, target in zip(prior, bounded)
        )
        previous_weight = self.config.ema_prev_weight
        current_weight = 1.0 - previous_weight
        filtered = tuple(
            previous_weight * previous + current_weight * current
            for previous, current in zip(prior, rate_limited)
        )
        if any(not math.isfinite(value) for value in filtered):
            raise ControllerContractError("CTRL_NONFINITE: filtered action")
        transition = ActionFilterTransition(
            raw_fy=(raw[0], raw[1]),
            scattered=(scattered[0], scattered[1], scattered[2]),
            bounded=(bounded[0], bounded[1], bounded[2]),
            rate_limited=(rate_limited[0], rate_limited[1], rate_limited[2]),
            filtered=(filtered[0], filtered[1], filtered[2]),
            prior_cmd=(prior[0], prior[1], prior[2]),
            tick=state.ticks,
        )
        next_state = ActionFilterState(
            prev_cmd=transition.filtered,
            ticks=state.ticks + 1,
        )
        return next_state, transition

    def clamp_horizon(
        self, raw_actions_fy: Iterable[Sequence[float]]
    ) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            clamp_stage(action, max_abs=self.config.max_abs)
            for action in raw_actions_fy
        )
