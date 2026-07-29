"""TrackVLA++-Lite on the local OpenTrackVLA-Qwen0.6B base.

This is an engineering adaptation of TrackVLA++ (arXiv:2510.07134), not a
claim of exact reproduction.  It keeps only the paper's two additions over the
navigation base: factorized Polar-CoT and a four-token confidence-gated Target
Identification Memory (TIM).  PFEM-only Future, Verifier, Event Bank and
Orchestrator modules are deliberately absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from harness.base_repro.polar_cot import PolarCoTHead, polar_cot_loss
from harness.base_repro.tim import TIM, roi_pool_candidate


class FactorizedPolarReasoningToken(nn.Module):
    """Turn factorized Polar-CoT probabilities into one soft feedback token.

    The paper uses one vocabulary token for a quantized angle-distance sector.
    The local lightweight implementation retains the existing 60-way angle,
    30-way distance and invalid heads, then feeds their differentiable expected
    embedding to the action context.  Checkpoint metadata records this
    approximation explicitly.
    """

    def __init__(self, d_model: int, n_theta: int = 60, n_dist: int = 30):
        super().__init__()
        self.theta_emb = nn.Embedding(n_theta, d_model)
        self.dist_emb = nn.Embedding(n_dist, d_model)
        self.invalid_emb = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.invalid_emb, std=0.02)

    def forward(self, cot_out: dict) -> torch.Tensor:
        theta_p = torch.softmax(cot_out["theta_logits"], dim=-1)
        dist_p = torch.softmax(cot_out["dist_logits"], dim=-1)
        invalid_p = torch.sigmoid(cot_out["invalid_logit"]).unsqueeze(-1)
        valid_token = theta_p @ self.theta_emb.weight + dist_p @ self.dist_emb.weight
        token = (1.0 - invalid_p) * valid_token + invalid_p * self.invalid_emb
        return self.norm(token)


class TrackVLAPlusPlusLite(nn.Module):
    """Paper-structured Lite baseline: history + Polar-CoT + four-token TIM.

    Differences from the paper are intentional and surfaced in checkpoint
    metadata by the trainer:

    * Qwen3-0.6B/OpenTrackVLA-MLP replaces NavFoM/Qwen2-7B.
    * Polar-CoT uses factorized classification heads instead of a joint special
      vocabulary token.
    * The soft reasoning token is fed through a lightweight action-context MLP
      rather than an autoregressive second language-model decoding pass.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        expected_history: int = 31,
        use_tim: bool = True,
        tim_tokens: int = 4,
    ):
        super().__init__()
        self.base = base_model
        self.D = int(base_model.D)
        self.expected_history = int(expected_history)
        self.use_tim = bool(use_tim)
        self.cot = PolarCoTHead(self.D, n_theta=60, n_dist=30)
        self.tim = TIM(self.D, n_tokens=int(tim_tokens)) if self.use_tim else None
        self.reason_token = FactorizedPolarReasoningToken(self.D, 60, 30)
        # Residual fusion preserves the official base planner at initialization:
        # the final layer starts at zero, so action_hidden == h_reason.
        self.action_delta = nn.Sequential(
            nn.LayerNorm(self.D * 4),
            nn.Linear(self.D * 4, self.D),
            nn.GELU(),
            nn.Linear(self.D, self.D),
        )
        nn.init.zeros_(self.action_delta[-1].weight)
        nn.init.zeros_(self.action_delta[-1].bias)

        # Match the PFEM initialization policy: frozen base, shared projector
        # trainable.  The native planner is the final action head and is also
        # trainable for this baseline.
        self.base.requires_grad_(False)
        self.base.proj.requires_grad_(True)
        self.base.planner.requires_grad_(True)

    def init_state(self, batch_size: int, device):
        if self.tim is None:
            return {"tim": None, "has_pending": torch.zeros(batch_size, dtype=torch.bool, device=device)}
        return {
            "tim": self.tim.init_state(batch_size, device),
            "pending_candidate": torch.zeros(
                batch_size, self.tim.n_tokens, self.D, device=device
            ),
            "pending_confidence": torch.zeros(batch_size, device=device),
            "pending_invalid": torch.ones(batch_size, dtype=torch.bool, device=device),
            "has_pending": torch.zeros(batch_size, dtype=torch.bool, device=device),
        }

    def _apply_pending_update(self, state: dict):
        """Build M_T from the candidate/confidence produced at T-1."""

        tim_state = state["tim"]
        if not self.use_tim or not bool(state["has_pending"].any().item()):
            return tim_state
        has_pending = state["has_pending"]
        invalid = state["pending_invalid"] | (~has_pending)
        updated = self.tim.update(
            tim_state,
            state["pending_candidate"],
            state["pending_confidence"],
            torch.ones_like(state["pending_confidence"]),
            invalid_mask=invalid,
            count_invalid_in_average=True,
        )
        merged = {}
        for key in ("mem", "C_avg", "C_cnt", "initialized"):
            mask = has_pending
            while mask.dim() < updated[key].dim():
                mask = mask.unsqueeze(-1)
            merged[key] = torch.where(mask, updated[key], tim_state[key])
        old_gate = tim_state.get(
            "last_gate", torch.zeros_like(updated["last_gate"])
        )
        merged["last_gate"] = torch.where(
            has_pending, updated["last_gate"], old_gate
        )
        return merged

    def _encode_observation(
        self,
        coarse_tokens,
        coarse_tidx,
        fine_tokens,
        fine_tidx,
        instructions,
        tim_state,
        yaw_hist=None,
        yaw_curr=None,
    ):
        device = next(self.parameters()).device
        batch_size = coarse_tokens.size(0)
        vis_c = self.base.proj(coarse_tokens.to(device))
        vis_f = self.base.proj(fine_tokens.to(device))
        vis_c = self.base._interleave_tvi(
            vis_c,
            coarse_tidx.to(device),
            kind_id=0,
            yaw_per_frame=yaw_hist,
            use_angle=self.base.cfg.use_angle_tvi,
        )
        vis_f = self.base._interleave_tvi(
            vis_f,
            fine_tidx.to(device),
            kind_id=1,
            yaw_per_frame=yaw_curr,
            use_angle=self.base.cfg.use_angle_tvi,
        )
        text_emb, text_mask = self.base._embed_text(instructions, device)
        reason_query = self.base.act_token.expand(batch_size, 1, -1)
        tim_tokens = (
            tim_state["mem"].to(self.base.llm.dtype)
            if self.use_tim
            else vis_c.new_zeros(batch_size, 0, self.D).to(self.base.llm.dtype)
        )
        sequence = torch.cat([text_emb, tim_tokens, vis_c, vis_f, reason_query], dim=1).to(
            self.base.llm.dtype
        )
        attention = torch.cat(
            [
                text_mask,
                torch.ones(
                    batch_size,
                    tim_tokens.size(1) + vis_c.size(1) + vis_f.size(1) + 1,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )
        output = self.base.llm(
            inputs_embeds=sequence,
            attention_mask=attention,
            output_hidden_states=True,
            use_cache=False,
        )
        return output.last_hidden_state[:, -1, :].float(), vis_f

    def forward_step(
        self,
        coarse_tokens,
        coarse_tidx,
        fine_tokens,
        fine_tidx,
        instructions,
        prev_state,
        distractor_rate=None,
        yaw_hist=None,
        yaw_curr=None,
        prev_action=None,
    ):
        del distractor_rate, prev_action
        tim_state = self._apply_pending_update(prev_state)
        h_reason, _vis_f_interleaved = self._encode_observation(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            instructions,
            tim_state,
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
        )

        cot_out = self.cot(h_reason)
        cot_decoded = self.cot.decode(cot_out)
        confidence = cot_decoded["confidence"]
        invalid = cot_decoded["invalid_pred"]

        # Candidate features must come from projected fine visual patches, not
        # from the interleaved time token representation.
        device = h_reason.device
        fine_projected = self.base.proj(fine_tokens.to(device)).float()
        if self.use_tim:
            pending_candidate = roi_pool_candidate(
                fine_projected,
                cot_decoded["theta_idx"],
                n_theta=self.cot.n_theta,
                n_tokens=self.tim.n_tokens,
            )

            # The current candidate is pending state for T+1.  It must not be
            # written into memory or influence the action at the same timestep.
            tim_mean = tim_state["mem"].mean(dim=1).float()
        else:
            pending_candidate = None
            tim_mean = torch.zeros_like(h_reason)

        reasoning_token = self.reason_token(cot_out)
        action_delta = self.action_delta(
            torch.cat(
                [
                    h_reason,
                    reasoning_token,
                    tim_mean,
                    fine_projected.mean(dim=1),
                ],
                dim=-1,
            )
        )
        action_hidden = h_reason + action_delta
        waypoints = self.base.planner(action_hidden) * self.base.alpha_task
        return {
            "waypoints": waypoints,
            "cot": cot_out,
            "cot_decoded": cot_decoded,
            "C": confidence,
            "new_state": (
                {
                    "tim": tim_state,
                    "pending_candidate": pending_candidate,
                    "pending_confidence": confidence,
                    "pending_invalid": invalid,
                    "has_pending": torch.ones_like(invalid, dtype=torch.bool),
                }
                if self.use_tim
                else {
                    "tim": None,
                    "has_pending": torch.zeros_like(invalid, dtype=torch.bool),
                }
            ),
        }

    def compute_losses(self, output: dict, ground_truth: dict) -> dict:
        trajectory_error = (output["waypoints"] - ground_truth["waypoints"]).pow(2)
        valid_mask = ground_truth.get("valid_mask")
        if valid_mask is not None:
            mask = valid_mask.to(
                device=trajectory_error.device, dtype=trajectory_error.dtype
            )
            while mask.dim() < trajectory_error.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(trajectory_error)
            denominator = mask.sum()
            trajectory_loss = (
                (trajectory_error * mask).sum() / denominator
                if denominator.item() > 0
                else trajectory_error.sum() * 0.0
            )
        else:
            trajectory_loss = trajectory_error.mean()
        reason_loss = polar_cot_loss(
            output["cot"],
            ground_truth["theta_idx"],
            ground_truth["dist_idx"],
            ground_truth["invalid"],
        )
        loss = trajectory_loss + 0.2 * reason_loss
        return {
            "loss": loss,
            "L_traj": trajectory_loss.item(),
            "L_reason": reason_loss.item(),
        }
