"""Previous-action-free OpenTrackVLA observation and auxiliary adapter for F2.

This module deliberately lives outside the vendored OpenTrackVLA tree.  It
reuses the official projector/LLM observation path and the existing
Polar-CoT, TIM, Cognitive Event Bank, and Orchestrator implementations while
enforcing the post-probe F2 corrigendum:

* observation encoding has no previous-action input;
* Future conditions only on ``h_act``, TIM, and event features;
* q predicts Polar self-correctness and is stop-gradient everywhere except
  its own auxiliary loss;
* the verifier action residual and ``alpha_verify`` do not exist; and
* the single perception-state stream is detached at every step and can be
  reset per batch element.

The adapter produces features for :class:`f2_experiment.model.F2AP2Model`; it
does not contain an action head, controller, trainer, or evaluator.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from third_party.OpenTrackVLA.harness.base_repro.polar_cot import (
    PolarCoTHead,
    polar_cot_loss,
)
from third_party.OpenTrackVLA.harness.base_repro.tim import (
    TIM,
    roi_pool_candidate,
)
from third_party.OpenTrackVLA.harness.core.event_bank import (
    EVENT_TYPES,
    CognitiveEventBank,
)
from third_party.OpenTrackVLA.harness.core.orchestrator import (
    META_FIELDS,
    Orchestrator,
)

from .support import ARCHITECTURE_LOCK, F2ContractError


FUTURE_HORIZONS = (4, 8, 16)
METHOD_FEATURE_DIMS = ("polar", "tim_q", "future", "event")


class F2ObservationContractError(F2ContractError):
    """Raised when the isolated F2 observation contract is violated."""


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise F2ObservationContractError(f"{label} must be a positive integer")
    return value


def _finite_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise F2ObservationContractError(f"{label} must be a torch.Tensor")
    if not value.is_floating_point():
        raise F2ObservationContractError(f"{label} must use a floating dtype")
    if bool((~torch.isfinite(value)).detach().any().to("cpu").item()):
        raise F2ObservationContractError(f"{label} contains nonfinite values")
    return value


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


def _state_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _state_tensors(item)


class DifferentiablePolarToken(nn.Module):
    """Expected embedding of the factorized Polar logits.

    Hard ``argmax`` remains reserved for ROI selection and event decisions;
    this soft token is the differentiable current-step action-credit path.
    """

    def __init__(self, d_model: int, n_theta: int, n_dist: int) -> None:
        super().__init__()
        self.theta_embedding = nn.Embedding(n_theta, d_model)
        self.distance_embedding = nn.Embedding(n_dist, d_model)
        self.invalid_embedding = nn.Parameter(torch.empty(d_model))
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.invalid_embedding, std=0.02)

    def forward(self, cot_output: Mapping[str, torch.Tensor]) -> torch.Tensor:
        theta_probability = torch.softmax(cot_output["theta_logits"], dim=-1)
        distance_probability = torch.softmax(cot_output["dist_logits"], dim=-1)
        invalid_probability = torch.sigmoid(
            cot_output["invalid_logit"]
        ).unsqueeze(-1)
        visible_token = (
            theta_probability @ self.theta_embedding.weight
            + distance_probability @ self.distance_embedding.weight
        )
        token = (
            (1.0 - invalid_probability) * visible_token
            + invalid_probability * self.invalid_embedding
        )
        return self.norm(token)


class PrevFreeFutureModule(nn.Module):
    """Hierarchical Future module with no action projection or action input."""

    def __init__(
        self,
        d_model: int,
        *,
        sig_dim: int = 8,
        n_theta: int = 60,
        n_dist: int = 30,
        horizons: Sequence[int] = FUTURE_HORIZONS,
        tokens_per_horizon: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = _positive_int(d_model, "d_model")
        self.sig_dim = _positive_int(sig_dim, "sig_dim")
        self.tokens_per_horizon = _positive_int(
            tokens_per_horizon, "tokens_per_horizon"
        )
        self.horizons = tuple(
            _positive_int(int(horizon), "future horizon") for horizon in horizons
        )
        if len(self.horizons) == 0 or len(set(self.horizons)) != len(self.horizons):
            raise F2ObservationContractError(
                "future horizons must be nonempty and unique"
            )

        # The legacy fourth input was act_proj(last_action).  F2 deletes it.
        self.shared = nn.Sequential(
            nn.LayerNorm(self.d_model * 3),
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
        )
        self.horizon_projections = nn.ModuleList()
        for index in range(len(self.horizons)):
            input_dim = self.d_model * (2 if index > 0 else 1)
            self.horizon_projections.append(
                nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(
                        input_dim,
                        self.tokens_per_horizon * self.d_model,
                    ),
                )
            )
        self.embedding_heads = nn.ModuleList(
            nn.Linear(self.d_model, self.sig_dim) for _ in self.horizons
        )
        self.theta_heads = nn.ModuleList(
            nn.Linear(self.d_model, n_theta) for _ in self.horizons
        )
        self.distance_heads = nn.ModuleList(
            nn.Linear(self.d_model, n_dist) for _ in self.horizons
        )
        self.visibility_heads = nn.ModuleList(
            nn.Linear(self.d_model, 1) for _ in self.horizons
        )

    def forward(
        self,
        h_act: torch.Tensor,
        tim_mean: torch.Tensor,
        event_token: torch.Tensor,
    ) -> dict[int | str, Any]:
        hidden = self.shared(torch.cat([h_act, tim_mean, event_token], dim=-1))
        output: dict[int | str, Any] = {}
        previous_representation = None
        all_tokens = []
        for index, horizon in enumerate(self.horizons):
            batch_size = hidden.shape[0]
            projection_input = (
                hidden
                if previous_representation is None
                else torch.cat(
                    [hidden, previous_representation.detach()], dim=-1
                )
            )
            tokens = self.horizon_projections[index](projection_input).view(
                batch_size,
                self.tokens_per_horizon,
                self.d_model,
            )
            representation = tokens.mean(dim=1)
            output[horizon] = {
                "tokens": tokens,
                "emb": F.normalize(
                    self.embedding_heads[index](representation), dim=-1
                ),
                "theta_logits": self.theta_heads[index](representation),
                "dist_logits": self.distance_heads[index](representation),
                "vis_logit": self.visibility_heads[index](
                    representation
                ).squeeze(-1),
            }
            all_tokens.append(tokens)
            previous_representation = representation
        output["all_tokens"] = torch.cat(all_tokens, dim=1)
        return output


class SelfCorrectnessHead(nn.Module):
    """Single-output q head; intentionally has no action-delta branch."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        self.q_head = nn.Linear(256, 1)

    def forward(
        self,
        candidate_mean: torch.Tensor,
        tim_mean: torch.Tensor,
        h_act: torch.Tensor,
        future_mean: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.trunk(
            torch.cat(
                [candidate_mean, tim_mean, h_act, future_mean], dim=-1
            )
        )
        return torch.sigmoid(self.q_head(hidden).squeeze(-1))


class OpenTrackVLAF2ObservationAdapter(nn.Module):
    """Wrap an official OpenTrackVLA base and emit prev-free F2 features."""

    architecture_lock = ARCHITECTURE_LOCK

    def __init__(
        self,
        base_model: nn.Module,
        *,
        sig_dim: int = 8,
        tim_tokens: int = 4,
        event_slots: int = 6,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise F2ObservationContractError("base_model must be an nn.Module")
        try:
            self.D = _positive_int(int(base_model.D), "base_model.D")
            use_angle_tvi = bool(base_model.cfg.use_angle_tvi)
        except (AttributeError, TypeError, ValueError) as exc:
            raise F2ObservationContractError(
                "base_model is missing the official OpenTrackVLA interface"
            ) from exc
        for attribute in (
            "proj",
            "llm",
            "act_token",
            "_interleave_tvi",
            "_embed_text",
        ):
            if not hasattr(base_model, attribute):
                raise F2ObservationContractError(
                    f"base_model is missing required attribute {attribute!r}"
                )

        self.base = base_model
        self.use_angle_tvi = use_angle_tvi
        self.cot = PolarCoTHead(self.D)
        self.tim = TIM(self.D, n_tokens=_positive_int(tim_tokens, "tim_tokens"))
        self.polar_token = DifferentiablePolarToken(
            self.D, self.cot.n_theta, self.cot.n_dist
        )
        self.future = PrevFreeFutureModule(self.D, sig_dim=sig_dim)
        self.self_correctness = SelfCorrectnessHead(self.D)
        self.events = CognitiveEventBank(
            self.D,
            n_types=len(EVENT_TYPES),
            L=_positive_int(event_slots, "event_slots"),
        )

        # Reuse the existing Orchestrator internals, but replace its obsolete
        # four-way alpha output before it can allocate probability mass to an
        # alpha_verify branch with no consumer.
        self.orchestrator = Orchestrator(self.D)
        old_alpha_head = self.orchestrator.alpha_mlp[-1]
        if not isinstance(old_alpha_head, nn.Linear):
            raise F2ObservationContractError(
                "unexpected Orchestrator alpha head implementation"
            )
        self.orchestrator.alpha_mlp[-1] = nn.Linear(
            old_alpha_head.in_features, 3
        )

        self.register_buffer(
            "expert_future_leak_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "self_state_expert_overwrite_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )

    @property
    def method_dims(self) -> dict[str, int]:
        return {
            "polar": self.D,
            "tim_q": self.D + 2,
            "future": self.D,
            "event": self.D,
        }

    def _counter_increment(self, counter: torch.Tensor, amount: int) -> None:
        count = _positive_int(amount, "counter increment")
        counter.add_(count)

    def record_expert_future_leak(self, amount: int = 1) -> None:
        """Runner hook for a forbidden future-label dataflow observation."""

        self._counter_increment(self.expert_future_leak_count, amount)

    def record_self_state_expert_overwrite(self, amount: int = 1) -> None:
        """Runner hook for a logged-state overwrite outside RESET."""

        self._counter_increment(self.self_state_expert_overwrite_count, amount)

    def audit_counters(self) -> dict[str, int]:
        return {
            "expert_future_leak_count": int(
                self.expert_future_leak_count.detach().to("cpu").item()
            ),
            "self_state_expert_overwrite_count": int(
                self.self_state_expert_overwrite_count.detach().to("cpu").item()
            ),
        }

    def assert_audit_counters_clean(self) -> None:
        dirty = {
            key: value for key, value in self.audit_counters().items() if value
        }
        if dirty:
            raise F2ObservationContractError(
                f"forbidden F2 dataflow counters are nonzero: {dirty}"
            )

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        batch = _positive_int(batch_size, "batch_size")
        tim_state = self.tim.init_state(batch, device)
        tim_state["last_gate"] = torch.zeros(batch, device=device)
        return {
            "tim": tim_state,
            "evt": self.events.init_state(batch, device),
            "pending_candidate": torch.zeros(
                batch,
                self.tim.n_tokens,
                self.D,
                device=device,
            ),
            "pending_confidence": torch.zeros(batch, device=device),
            "pending_q_write": torch.zeros(batch, device=device),
            "pending_invalid": torch.ones(
                batch, dtype=torch.bool, device=device
            ),
            "has_pending": torch.zeros(
                batch, dtype=torch.bool, device=device
            ),
        }

    def _normalize_reset_mask(
        self,
        reset_mask: bool | torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if reset_mask is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if isinstance(reset_mask, bool):
            return torch.full(
                (batch_size,), reset_mask, dtype=torch.bool, device=device
            )
        if not isinstance(reset_mask, torch.Tensor):
            raise F2ObservationContractError(
                "reset_mask must be bool, tensor, or None"
            )
        mask = reset_mask.to(device=device, dtype=torch.bool)
        if mask.shape != (batch_size,):
            raise F2ObservationContractError(
                f"reset_mask must have shape {(batch_size,)}"
            )
        return mask

    def _merge_reset_state(
        self,
        state: Mapping[str, Any],
        fresh: Mapping[str, Any],
        reset_mask: torch.Tensor,
    ) -> dict[str, Any]:
        if set(state) != set(fresh):
            raise F2ObservationContractError(
                "perception state keys do not match init_state"
            )
        merged: dict[str, Any] = {}
        for key in fresh:
            current = state[key]
            reset_value = fresh[key]
            if isinstance(reset_value, Mapping):
                if not isinstance(current, Mapping):
                    raise F2ObservationContractError(
                        f"perception state {key!r} must be a mapping"
                    )
                merged[key] = self._merge_reset_state(
                    current, reset_value, reset_mask
                )
                continue
            if not isinstance(current, torch.Tensor) or not isinstance(
                reset_value, torch.Tensor
            ):
                raise F2ObservationContractError(
                    f"perception state {key!r} must be a tensor"
                )
            if current.shape != reset_value.shape:
                raise F2ObservationContractError(
                    f"perception state {key!r} has an unexpected shape"
                )
            mask = reset_mask
            while mask.ndim < current.ndim:
                mask = mask.unsqueeze(-1)
            merged[key] = torch.where(mask, reset_value, current)
        return merged

    def _prepare_state(
        self,
        previous_state: Mapping[str, Any],
        *,
        batch_size: int,
        device: torch.device,
        reset_mask: bool | torch.Tensor | None,
    ) -> dict[str, Any]:
        if not isinstance(previous_state, Mapping):
            raise F2ObservationContractError(
                "previous_state must be an init_state-compatible mapping"
            )
        detached = _detach_tree(previous_state)
        fresh = self.init_state(batch_size, device)
        mask = self._normalize_reset_mask(
            reset_mask, batch_size=batch_size, device=device
        )
        prepared = self._merge_reset_state(detached, fresh, mask)
        for tensor in _state_tensors(prepared):
            if tensor.shape[0] != batch_size:
                raise F2ObservationContractError(
                    "all perception-state tensors must share the batch axis"
                )
        return prepared

    def _apply_pending_tim(self, state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        tim_state = state["tim"]
        has_pending = state["has_pending"]
        if not bool(has_pending.detach().any().to("cpu").item()):
            return dict(tim_state)
        invalid = state["pending_invalid"] | (~has_pending)
        updated = self.tim.update(
            tim_state,
            state["pending_candidate"],
            state["pending_confidence"],
            state["pending_q_write"],
            invalid_mask=invalid,
            count_invalid_in_average=True,
        )
        merged: dict[str, torch.Tensor] = {}
        for key in ("mem", "C_avg", "C_cnt", "initialized", "last_gate"):
            mask = has_pending
            while mask.ndim < updated[key].ndim:
                mask = mask.unsqueeze(-1)
            merged[key] = torch.where(mask, updated[key], tim_state[key])
        return merged

    def _llm_dtype(self) -> torch.dtype:
        dtype = getattr(self.base.llm, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        try:
            return next(self.base.llm.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _encode_official_base(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: Sequence[str],
        tim_state: Mapping[str, torch.Tensor],
        *,
        yaw_hist: torch.Tensor | None,
        yaw_curr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.base.act_token.device
        batch_size = coarse_tokens.shape[0]
        coarse_projected = self.base.proj(coarse_tokens.to(device))
        fine_projected = self.base.proj(fine_tokens.to(device))
        coarse_visual = self.base._interleave_tvi(
            coarse_projected,
            coarse_tidx.to(device),
            kind_id=0,
            yaw_per_frame=yaw_hist,
            use_angle=self.use_angle_tvi,
        )
        fine_visual = self.base._interleave_tvi(
            fine_projected,
            fine_tidx.to(device),
            kind_id=1,
            yaw_per_frame=yaw_curr,
            use_angle=self.use_angle_tvi,
        )
        text_embedding, text_mask = self.base._embed_text(instructions, device)
        llm_dtype = self._llm_dtype()
        tim_tokens = tim_state["mem"].to(device=device, dtype=llm_dtype)
        action_query = self.base.act_token.expand(batch_size, 1, -1)
        sequence = torch.cat(
            [text_embedding, tim_tokens, coarse_visual, fine_visual, action_query],
            dim=1,
        ).to(llm_dtype)
        attention = torch.cat(
            [
                text_mask.to(device),
                torch.ones(
                    batch_size,
                    tim_tokens.shape[1]
                    + coarse_visual.shape[1]
                    + fine_visual.shape[1]
                    + 1,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )
        llm_output = self.base.llm(
            inputs_embeds=sequence,
            attention_mask=attention,
            output_hidden_states=True,
            use_cache=False,
        )
        h_act = _finite_tensor(
            llm_output.last_hidden_state[:, -1, :].float(), "h_act"
        )
        return h_act, fine_projected.float()

    def _event_triggers(
        self,
        invalid_prediction: torch.Tensor,
        q_detached: torch.Tensor,
        distractor_rate: torch.Tensor,
    ) -> list[tuple[int, int]]:
        triggers: list[tuple[int, int]] = []
        for batch_index in range(invalid_prediction.shape[0]):
            if bool(invalid_prediction[batch_index].to("cpu").item()):
                triggers.append((batch_index, 1))
            if float(q_detached[batch_index].to("cpu").item()) < 0.3:
                triggers.append((batch_index, 3))
            if float(distractor_rate[batch_index].to("cpu").item()) > 0.5:
                triggers.append((batch_index, 2))
        return triggers

    def _orchestrate_three_way(
        self,
        confidence: torch.Tensor,
        distractor_rate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        metadata = {
            "identity_conf": self.orchestrator._bin(confidence, 5),
            "occlusion_risk": self.orchestrator._bin(1.0 - confidence, 5),
            "crowd_level": self.orchestrator._bin(distractor_rate, 3),
            "aggressiveness": torch.ones_like(confidence, dtype=torch.long),
            "reacq_patience": torch.ones_like(confidence, dtype=torch.long),
        }
        metadata_tokens = torch.stack(
            [
                self.orchestrator.meta_embs[field](metadata[field])
                for field in META_FIELDS
            ],
            dim=1,
        )
        flattened = metadata_tokens.flatten(1)
        mode_probability = torch.softmax(
            self.orchestrator.mode_mlp(flattened), dim=-1
        )
        alpha = torch.softmax(self.orchestrator.alpha_mlp(flattened), dim=-1)
        if alpha.shape[-1] != 3:
            raise F2ObservationContractError(
                "Orchestrator must emit exactly three F2 alpha values"
            )
        return {
            "mode": mode_probability.argmax(dim=-1),
            "mode_p": mode_probability,
            "mode_emb": mode_probability @ self.orchestrator.mode_table.weight,
            "meta_tokens": metadata_tokens,
            "alpha_tim": alpha[:, 0],
            "alpha_event": alpha[:, 1],
            "alpha_future": alpha[:, 2],
        }

    def encode_step(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: Sequence[str],
        previous_state: Mapping[str, Any],
        *,
        reset_mask: bool | torch.Tensor | None = None,
        distractor_rate: torch.Tensor | None = None,
        yaw_hist: torch.Tensor | None = None,
        yaw_curr: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Encode one observation without accepting action or expert inputs."""

        if coarse_tokens.ndim != 3 or fine_tokens.ndim != 3:
            raise F2ObservationContractError(
                "coarse_tokens and fine_tokens must have shape (B,N,D)"
            )
        batch_size = coarse_tokens.shape[0]
        if fine_tokens.shape[0] != batch_size or len(instructions) != batch_size:
            raise F2ObservationContractError(
                "observation inputs must share the batch axis"
            )
        device = self.base.act_token.device
        state = self._prepare_state(
            previous_state,
            batch_size=batch_size,
            device=device,
            reset_mask=reset_mask,
        )
        tim_state = self._apply_pending_tim(state)
        h_act, fine_projected = self._encode_official_base(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            instructions,
            tim_state,
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
        )

        cot_output = self.cot(h_act)
        cot_decoded = self.cot.decode(cot_output)
        confidence = cot_decoded["confidence"]
        invalid_prediction = cot_decoded["invalid_pred"]
        candidate = roi_pool_candidate(
            fine_projected,
            cot_decoded["theta_idx"],
            n_theta=self.cot.n_theta,
            n_tokens=self.tim.n_tokens,
        )
        tim_mean = tim_state["mem"].mean(dim=1).float()

        event_before_write = self.events.read(state["evt"], h_act)
        future_output = self.future(h_act, tim_mean, event_before_write)
        future_mean = future_output["all_tokens"].mean(dim=1)
        q_write = self.self_correctness(
            candidate.mean(dim=1), tim_mean, h_act, future_mean
        )
        q_detached = q_write.detach()

        if distractor_rate is None:
            distractor = torch.zeros_like(confidence)
        else:
            distractor = _finite_tensor(
                distractor_rate.to(device=device, dtype=confidence.dtype),
                "distractor_rate",
            )
            if distractor.shape != confidence.shape:
                raise F2ObservationContractError(
                    "distractor_rate must have shape (B,)"
                )
        triggers = self._event_triggers(
            invalid_prediction, q_detached, distractor
        )
        event_state = self.events.write(
            state["evt"], h_act, triggers, confidence
        )
        event_feature = self.events.read(event_state, h_act)
        orchestrator_output = self._orchestrate_three_way(
            confidence, distractor
        )

        method_features = {
            "polar": self.polar_token(cot_output),
            "tim_q": torch.cat(
                [
                    tim_mean,
                    confidence.unsqueeze(-1),
                    q_detached.unsqueeze(-1),
                ],
                dim=-1,
            ),
            "future": future_mean,
            "event": event_feature,
        }
        method_alphas = {
            "polar": torch.ones_like(confidence),
            "tim_q": orchestrator_output["alpha_tim"],
            "future": orchestrator_output["alpha_future"],
            "event": orchestrator_output["alpha_event"],
        }
        if tuple(method_features) != METHOD_FEATURE_DIMS:
            raise F2ObservationContractError(
                "F2 method feature order changed unexpectedly"
            )

        new_state = _detach_tree(
            {
                "tim": tim_state,
                "evt": event_state,
                "pending_candidate": candidate,
                "pending_confidence": confidence,
                "pending_q_write": q_detached,
                "pending_invalid": invalid_prediction,
                "has_pending": torch.ones(
                    batch_size, dtype=torch.bool, device=device
                ),
            }
        )
        if any(
            tensor.requires_grad or tensor.grad_fn is not None
            for tensor in _state_tensors(new_state)
        ):
            raise F2ObservationContractError(
                "perception state must be detached at every step"
            )
        return {
            "base_features": h_act,
            "h_act": h_act,
            "method_features": method_features,
            "method_alphas": method_alphas,
            "cot": cot_output,
            "cot_decoded": cot_decoded,
            "future": future_output,
            "q_write": q_write,
            "C": confidence,
            "orchestrator": orchestrator_output,
            "new_state": new_state,
            "audit_counters": self.audit_counters(),
        }

    @staticmethod
    def _target(
        targets: Mapping[str, torch.Tensor],
        primary: str,
        fallback: str | None,
    ) -> torch.Tensor:
        if primary in targets:
            value = targets[primary]
        elif fallback is not None and fallback in targets:
            value = targets[fallback]
        else:
            raise F2ObservationContractError(
                f"auxiliary targets are missing {primary!r}"
            )
        if not isinstance(value, torch.Tensor):
            raise F2ObservationContractError(
                f"auxiliary target {primary!r} must be a tensor"
            )
        return value.detach()

    def compute_aux_losses(
        self,
        output: Mapping[str, Any],
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return ``L_aux = L_cot + L_future + L_verify``.

        Future expert data enters only through this label-side function.  All
        labels are detached before loss construction and neither ``output`` nor
        its perception state is mutated.
        """

        if not isinstance(output, Mapping) or not isinstance(targets, Mapping):
            raise F2ObservationContractError(
                "output and targets must be mappings"
            )
        cot_output = output["cot"]
        device = output["h_act"].device
        theta_target = self._target(
            targets, "theta_idx", "polar_theta_idx"
        ).to(device=device, dtype=torch.long)
        distance_target = self._target(
            targets, "dist_idx", "polar_dist_idx"
        ).to(device=device, dtype=torch.long)
        invalid_target = self._target(
            targets, "invalid", "polar_invalid"
        ).to(device=device, dtype=torch.float32)
        cot_loss = polar_cot_loss(
            cot_output,
            theta_target,
            distance_target,
            invalid_target,
        )

        future_terms: list[torch.Tensor] = []
        for horizon in FUTURE_HORIZONS:
            prediction = output["future"][horizon]
            valid = self._target(
                targets, f"fut_valid_{horizon}", None
            ).to(device=device, dtype=torch.bool)
            visibility = self._target(
                targets, f"fut_vis_{horizon}", None
            ).to(device=device, dtype=prediction["vis_logit"].dtype)
            if bool(valid.detach().any().to("cpu").item()):
                future_terms.append(
                    F.binary_cross_entropy_with_logits(
                        prediction["vis_logit"][valid], visibility[valid]
                    )
                )
            visible = valid & (visibility > 0.5)
            if bool(visible.detach().any().to("cpu").item()):
                future_theta = self._target(
                    targets, f"fut_theta_idx_{horizon}", None
                ).to(device=device, dtype=torch.long)
                future_distance = self._target(
                    targets, f"fut_dist_idx_{horizon}", None
                ).to(device=device, dtype=torch.long)
                future_terms.extend(
                    [
                        F.cross_entropy(
                            prediction["theta_logits"][visible],
                            future_theta[visible].clamp_min(0),
                        ),
                        F.cross_entropy(
                            prediction["dist_logits"][visible],
                            future_distance[visible].clamp_min(0),
                        ),
                    ]
                )
        future_loss = (
            torch.stack(future_terms).mean()
            if future_terms
            else output["future"]["all_tokens"].sum() * 0.0
        )

        with torch.no_grad():
            self_correct = (
                (invalid_target < 0.5)
                & (cot_output["theta_logits"].argmax(dim=-1) == theta_target)
            ).to(dtype=output["q_write"].dtype)
        verify_loss = F.binary_cross_entropy(output["q_write"], self_correct)
        auxiliary_loss = cot_loss + future_loss + verify_loss
        return {
            "loss": auxiliary_loss,
            "L_aux": auxiliary_loss,
            "L_cot": cot_loss,
            "L_future": future_loss,
            "L_verify": verify_loss,
        }


__all__ = [
    "DifferentiablePolarToken",
    "F2ObservationContractError",
    "FUTURE_HORIZONS",
    "OpenTrackVLAF2ObservationAdapter",
    "PrevFreeFutureModule",
    "SelfCorrectnessHead",
]
