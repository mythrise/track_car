"""PFEMHarness — wraps OpenTrackVLA with the full PFEM module stack.

Usage:
    from model import OpenTrackVLA, ModelConfig
    base = OpenTrackVLA(ModelConfig(), vision_feat_dim=1536)
    harness = PFEMHarness(base)
    # harness.forward(...) runs the full pipeline
"""

from __future__ import annotations
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.base_repro.polar_cot import PolarCoTHead, polar_cot_loss
from harness.base_repro.tim import TIM, roi_pool_candidate
from harness.core.future_module import FutureModule
from harness.core.verifier import Verifier
from harness.core.event_bank import CognitiveEventBank, EVENT_TYPES
from harness.core.orchestrator import Orchestrator
from harness.core.step_planner import (
    DeltaVelocityHead,
    StepActionHead,
    step_action_track_loss,
)

from data_pipeline.kinematics import integrate_actions


class PFEMHarness(nn.Module):
    """Full PFEM-Harness wrapper around a frozen OpenTrackVLA base."""

    def __init__(self, base_model: nn.Module, sig_dim: int = 8, *,
                 label_mode: str = "absolute", dt: float = 0.1,
                 lambda_yaw: float = 2.0, aux_delta_vel: bool = False):
        super().__init__()
        if label_mode not in {"absolute", "step_action"}:
            raise ValueError(f"unsupported label_mode={label_mode!r}")
        if float(dt) <= 0:
            raise ValueError("dt must be > 0")
        self.base = base_model
        D = base_model.D
        self.D = D
        self.label_mode = label_mode
        self.dt = float(dt)
        self.lambda_yaw = float(lambda_yaw)
        self.aux_delta_vel = bool(aux_delta_vel)

        # Layer A
        self.cot = PolarCoTHead(D)
        self.tim = TIM(D, n_tokens=4)

        # Layer B
        self.future = FutureModule(D, sig_dim=sig_dim, action_dim=3)
        self.verifier = Verifier(D, n_waypoints=base_model.cfg.n_waypoints, action_dims=3)
        self.events = CognitiveEventBank(D, n_types=len(EVENT_TYPES), L=6)
        self.orch = Orchestrator(D)

        # Context projection: D * 7 components → D for action head input
        ctx_dim = D * 7
        if self.aux_delta_vel:
            self.prev_action_embed = nn.Linear(3, 64)
            ctx_dim += 64
        self.context_proj = nn.Linear(ctx_dim, D)
        if self.label_mode == "step_action":
            self.step_action_head = StepActionHead(D, base_model.cfg.n_waypoints)
        if self.aux_delta_vel:
            self.delta_vel_head = DeltaVelocityHead(D, base_model.cfg.n_waypoints)

        # Uncertainty weighting (Kendall et al.)
        self.log_sigma = nn.Parameter(torch.zeros(5))

        # Freeze the base model — harness modules train, base does not
        self.base.requires_grad_(False)
        # Re-enable projector training (it's shared between base and harness)
        self.base.proj.requires_grad_(True)

    def init_state(self, B: int, device):
        return {
            "tim": self.tim.init_state(B, device),
            "evt": self.events.init_state(B, device),
            "last_action": torch.zeros(B, 3, device=device),
        }

    def _condition_prev_action(self, prev_action):
        conditioning = prev_action.to(dtype=torch.float32)
        if self.training:
            conditioning = conditioning + 0.05 * torch.randn_like(conditioning)
        return self.prev_action_embed(conditioning)

    def _predict_tracking(self, ctx, q_write, orch_out, delta):
        if self.label_mode == "step_action":
            step_actions = self.step_action_head(ctx)
            return integrate_actions(step_actions, self.dt), step_actions

        wp = self.base.planner(ctx)
        tau_pred = wp * self.base.alpha_task
        if not self.training:
            gate = Verifier.runtime_gate(q_write, orch_out["mode"])
            tau_pred = tau_pred + gate.unsqueeze(-1).unsqueeze(-1) * orch_out[
                "alpha_verify"
            ].unsqueeze(-1).unsqueeze(-1) * delta
        return tau_pred, None

    def forward_step(self, coarse_tokens, coarse_tidx, fine_tokens, fine_tidx,
                     instructions, prev_state, distractor_rate=None,
                     yaw_hist=None, yaw_curr=None, prev_action=None):
        """Single timestep forward through the full harness."""
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)

        # ---- Base OpenTrackVLA encoding (reuse its projector + LLM) ----
        vis_c = self.base.proj(coarse_tokens.to(device))
        vis_f = self.base.proj(fine_tokens.to(device))
        vis_c = self.base._interleave_tvi(vis_c, coarse_tidx.to(device), kind_id=0,
                                           yaw_per_frame=yaw_hist,
                                           use_angle=self.base.cfg.use_angle_tvi)
        vis_f = self.base._interleave_tvi(vis_f, fine_tidx.to(device), kind_id=1,
                                           yaw_per_frame=yaw_curr,
                                           use_angle=self.base.cfg.use_angle_tvi)
        txt_emb, txt_mask = self.base._embed_text(instructions, device)

        # Build TIM tokens for LLM context
        tim_tokens = prev_state["tim"]["mem"].to(self.base.llm.dtype)

        act = self.base.act_token.expand(B, 1, -1)
        pieces = [txt_emb, tim_tokens, vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.base.llm.dtype)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, tim_tokens.size(1) + vis_c.size(1) + vis_f.size(1) + 1,
                       dtype=torch.long, device=device)
        ], dim=1)

        out = self.base.llm(inputs_embeds=seq, attention_mask=attn,
                            output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :].float()

        # ---- Polar-CoT ----
        cot_out = self.cot(h_act)
        cot_decoded = self.cot.decode(cot_out)
        C = cot_decoded["confidence"]
        invalid_pred = cot_decoded["invalid_pred"]
        theta_idx = cot_decoded["theta_idx"]

        # ---- Candidate feature (ROI pool from projected fine tokens) ----
        v_fine_proj = self.base.proj(fine_tokens.to(device)).float()
        candidate = roi_pool_candidate(v_fine_proj, theta_idx, n_theta=self.cot.n_theta,
                                       n_tokens=self.tim.n_tokens)

        tim_state = prev_state["tim"]
        tim_mean = tim_state["mem"].mean(dim=1).float()

        # ---- Event Bank read (from previous state) ----
        evt_tok = self.events.read(prev_state["evt"], h_act)

        # ---- Future Module ----
        last_action = prev_state["last_action"]
        fut_out = self.future(h_act, tim_mean, evt_tok, last_action)
        fut_mean = fut_out["all_tokens"].mean(dim=1)

        # ---- Verifier ----
        q_write, delta = self.verifier(candidate.mean(dim=1), tim_mean, h_act, fut_mean)

        # ---- TIM update ----
        tim_state = self.tim.update(tim_state, candidate, C, q_write, invalid_mask=invalid_pred)

        # ---- Event Bank write ----
        triggers = []
        for b in range(B):
            if invalid_pred[b]:
                triggers.append((b, 1))
            if q_write[b].item() < 0.3:
                triggers.append((b, 3))
        if distractor_rate is not None:
            for b in range(B):
                if distractor_rate[b] > 0.5:
                    triggers.append((b, 2))
        evt_state = self.events.write(prev_state["evt"], h_act, triggers, C)
        evt_tok_new = self.events.read(evt_state, h_act)

        # ---- Orchestrator ----
        invalid_streak = (1.0 - C).clamp(0, 1)
        dr = distractor_rate if distractor_rate is not None else torch.zeros_like(C)
        meta_bins = self.orch.compose_metadata(C, q_write, invalid_streak, dr)
        orch_out = self.orch(meta_bins)

        # ---- Action context assembly ----
        ctx_parts = [
            v_fine_proj.mean(dim=1),
            h_act,
            orch_out["alpha_tim"].unsqueeze(-1) * tim_state["mem"].mean(dim=1).float(),
            orch_out["alpha_evt"].unsqueeze(-1) * evt_tok_new,
            orch_out["alpha_fut"].unsqueeze(-1) * fut_mean,
            orch_out["mode_emb"],
            orch_out["meta_tokens"].mean(dim=1),
        ]
        conditioning_prev_action = prev_state["last_action"] if prev_action is None else prev_action
        conditioning_prev_action = conditioning_prev_action.to(device=device, dtype=torch.float32)
        if self.aux_delta_vel:
            ctx_parts.append(self._condition_prev_action(conditioning_prev_action))
        ctx = self.context_proj(torch.cat(ctx_parts, dim=-1))

        # ---- Action head ----
        tau_pred, step_actions = self._predict_tracking(ctx, q_write, orch_out, delta)

        delta_vel_pred = self.delta_vel_head(ctx) if self.aux_delta_vel else None

        new_state = {
            "tim": tim_state,
            "evt": evt_state,
            "last_action": (
                step_actions[:, 0].detach()
                if step_actions is not None
                else tau_pred[:, 0].detach()
            ),
        }
        result = {
            "waypoints": tau_pred,
            "cot": cot_out,
            "cot_decoded": cot_decoded,
            "future": fut_out,
            "q_write": q_write,
            "delta": delta,
            "C": C,
            "orch": orch_out,
            "new_state": new_state,
        }
        if step_actions is not None:
            result["step_actions"] = step_actions
        if delta_vel_pred is not None:
            result["delta_vel"] = delta_vel_pred
        return result

    def compute_losses(self, out: dict, gt: dict) -> dict:
        """Compute all PFEM losses with Uncertainty Weighting."""
        L_cot = polar_cot_loss(out["cot"], gt["theta_idx"], gt["dist_idx"], gt["invalid"])
        valid_mask = gt.get("valid_mask")
        if self.label_mode == "step_action":
            L_track, L_fwd, L_yaw, L_dvel = step_action_track_loss(
                out["step_actions"],
                gt["step_actions"],
                valid_mask,
                lambda_yaw=self.lambda_yaw,
                pred_delta_vel=out.get("delta_vel"),
                target_delta_vel=gt.get("delta_vel"),
            )
        else:
            track_error = F.mse_loss(out["waypoints"], gt["waypoints"], reduction="none")
            if valid_mask is None:
                L_track = track_error.mean()
            else:
                mask = valid_mask.to(device=track_error.device, dtype=track_error.dtype)
                while mask.dim() < track_error.dim():
                    mask = mask.unsqueeze(-1)
                mask = mask.expand_as(track_error)
                denominator = mask.sum()
                if denominator.item() <= 0:
                    L_track = track_error.sum() * 0.0
                else:
                    L_track = (track_error * mask).sum() / denominator
            L_fwd = L_track.new_zeros(())
            L_yaw = L_track.new_zeros(())
            L_dvel = L_track.new_zeros(())

        # Future loss (Δ=8 only for simplicity; expand for all horizons in full version)
        fut = out["future"]
        # Future loss — always compute to keep UW stable
        if 8 in fut and "theta_logits" in fut[8]:
            f8 = fut[8]
            L_future = F.binary_cross_entropy_with_logits(
                f8["vis_logit"], gt.get("fut_vis_8", torch.ones_like(out["C"])))
        else:
            # Small placeholder to keep log_sigma[2] from drifting
            L_future = torch.zeros(1, device=L_track.device, requires_grad=True).squeeze() * 0

        # Verifier q^write
        y_write = ((gt["invalid"] < 0.5) & (out["cot"]["theta_logits"].argmax(-1) == gt["theta_idx"])).float()
        L_verify = F.binary_cross_entropy(out["q_write"], y_write)

        # Uncertainty weighting (Kendall et al.): L = Σ 0.5*exp(-s_i)*L_i + 0.5*s_i
        losses = [L_track, L_cot, L_future, L_verify, out["delta"].pow(2).mean()]
        L = sum(0.5 * torch.exp(-self.log_sigma[i]) * losses[i] for i in range(5)) + 0.5 * self.log_sigma.sum()
        return {
            "loss": L, "L_track": L_track.item(), "L_cot": L_cot.item(),
            "L_fwd": L_fwd.item(), "L_yaw": L_yaw.item(), "L_dvel": L_dvel.item(),
            "L_future": L_future.item() if L_future.requires_grad else 0.0,
            "L_verify": L_verify.item(),
        }
