#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os, sys, json, math, argparse
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from cache_gridpool import (
    VisionFeatureCacher,
    VisionCacheConfig,
    adapt_siglip_grid,
    atomic_torch_save,
    grid_pool_tokens,
    legacy_token_cache_filename,
    token_cache_filename,
)
from local_weights import default_qwen_candidates, resolve_local_model_path

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / 'data_pipeline').is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.kinematics import integrate_actions
from data_pipeline.target_detector import load_omdet_components


# ----------------------- utils -----------------------

# Silence tokenizers fork warnings in dataloader workers
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_tokens_file(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise e
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                if t.dim() == 3 and t.size(0) == 1:
                    t = t[0]
                return t.float()
    raise ValueError(f"Unrecognized token file: {path}")


def token_cache_candidates(token_dir: Path, image_path: Path, level: str) -> List[Path]:
    """Prefer collision-safe v2 names, then fall back to legacy stem names."""

    current = token_dir / token_cache_filename(image_path, level)
    legacy = token_dir / legacy_token_cache_filename(image_path, level)
    return [current] if current == legacy else [current, legacy]


def load_first_cached_token(paths: List[Path]) -> torch.Tensor:
    last_error = None
    for path in paths:
        try:
            return load_tokens_file(str(path))
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def integrate_actions_to_waypoints(actions: np.ndarray, n_waypoints: int, dt: float = 0.2) -> np.ndarray:
    """Compatibility wrapper over the repository's shared kinematics."""

    trajectory = integrate_actions(actions, dt)
    if trajectory.shape[0] == 0:
        return trajectory
    if n_waypoints <= 1:
        return trajectory[-1:]
    if trajectory.shape[0] == n_waypoints:
        return trajectory
    indices = np.linspace(0, trajectory.shape[0] - 1, n_waypoints).round().astype(int)
    return trajectory[indices]


def dataset_manifest_path(dataset_path: Path) -> Path:
    return Path(str(dataset_path) + '.manifest.json')


def load_dataset_manifest(dataset_path: Path) -> Optional[Dict[str, Any]]:
    manifest_path = dataset_manifest_path(dataset_path)
    if not manifest_path.exists():
        return None
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {manifest_path}")
    if int(manifest.get('schema_version', -1)) != 1:
        raise ValueError(f"Unsupported dataset manifest schema_version: {manifest_path}")
    for field in ('path_root', 'fps', 'dt', 'action_semantics', 'label_mode'):
        if field not in manifest:
            raise ValueError(f"Dataset manifest missing {field!r}: {manifest_path}")
    if not isinstance(manifest['path_root'], str) or not manifest['path_root'].strip():
        raise ValueError(f"Invalid path_root in {manifest_path}: {manifest['path_root']!r}")
    if manifest['label_mode'] not in ('absolute', 'step_action'):
        raise ValueError(f"Unknown label_mode in {manifest_path}: {manifest['label_mode']!r}")
    semantics = manifest['action_semantics']
    semantics_values = semantics if isinstance(semantics, list) else [semantics]
    if not semantics_values or any(value not in ('spin_v1', 'arc_turn_v2') for value in semantics_values):
        raise ValueError(f"Unknown action_semantics in {manifest_path}: {semantics!r}")
    fps_values = manifest['fps'] if isinstance(manifest['fps'], list) else [manifest['fps']]
    dt_values = manifest['dt'] if isinstance(manifest['dt'], list) else [manifest['dt']]
    if not fps_values or any(float(value) <= 0 for value in fps_values):
        raise ValueError(f"Invalid fps in {manifest_path}: {manifest['fps']!r}")
    if not dt_values or any(float(value) <= 0 for value in dt_values):
        raise ValueError(f"Invalid dt in {manifest_path}: {manifest['dt']!r}")
    if len(fps_values) == len(dt_values):
        for fps, dt in zip(fps_values, dt_values):
            if not math.isclose(float(dt), 1.0 / float(fps), rel_tol=1e-4, abs_tol=1e-6):
                raise ValueError(
                    f"Inconsistent fps/dt in {manifest_path}: fps={fps!r}, dt={dt!r}"
                )
    elif len(fps_values) != 1 or len(dt_values) != 1:
        raise ValueError(f"fps/dt cardinality mismatch in {manifest_path}")
    return manifest


# Flexible loader for JSON/JSONL datasets and directories

def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            examples.append(json.loads(s))
    return examples


def load_examples_from_path(train_path: str) -> List[Dict[str, Any]]:
    p = Path(train_path)
    if p.is_dir():
        jsonl_files = sorted(p.rglob('*.jsonl'))
        if len(jsonl_files) == 0:
            raise FileNotFoundError(f"No .jsonl files found under directory: {train_path}")
        all_items: List[Dict[str, Any]] = []
        for fp in jsonl_files:
            all_items.extend(_read_jsonl_file(str(fp)))
        return all_items
    if p.is_file():
        if p.suffix.lower() == '.jsonl':
            return _read_jsonl_file(str(p))
        if p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list at top-level: {train_path}")
            return data
        raise ValueError(f"Unsupported file type: {train_path}")
    raise FileNotFoundError(f"Path does not exist: {train_path}")


# ----------------------- TVI + projector + planner -----------------------

class TVIEmbedder(nn.Module):
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        super().__init__()
        self.time_emb   = nn.Embedding(max_time, d_model)
        self.view_emb   = nn.Embedding(max_views, d_model)
        self.kind_emb   = nn.Embedding(2, d_model)
        self.angle_proj = nn.Linear(2, d_model)
        self.bbox_proj  = nn.Linear(4, d_model)

    def make_time_token(self, t_scalar: int, kind_id: int, view_id: int = 0,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        tok = self.time_emb.weight[t_scalar] + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok.to(device) if device is not None else tok

    def make_angle_token(self, theta: float, kind_id: int, view_id: int = 0,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        theta = (theta + math.pi) % (2*math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)],
                              dtype=self.angle_proj.weight.dtype,
                              device=device)
        ang = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        tok = ang + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok

    def make_bbox_token(self, bbox: torch.Tensor, kind_id: int, view_id: int = 1,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        bb = bbox.to(device=device, dtype=self.bbox_proj.weight.dtype)
        emb = F.linear(bb, self.bbox_proj.weight, self.bbox_proj.bias)
        emb = emb + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return emb.squeeze(0)


class MPSStableLayerNorm(nn.LayerNorm):
    """LayerNorm with an explicit float32 MPS path.

    PyTorch 2.11 on the current Apple-MPS stack can emit corrupt affine
    parameter gradients for ``nn.LayerNorm`` when its inputs do not require
    gradients.  The observed values were exact billion-scale sentinels while
    the analytical weight/bias gradients stayed below 0.1.  Expressing the
    normalization with ordinary reductions and elementwise operations avoids
    that backend kernel while preserving the standard state-dict contract.
    """

    def _manual_forward(self, value: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        value32 = value.float()
        normalized_dims = tuple(
            range(value32.dim() - len(self.normalized_shape), value32.dim())
        )
        mean = value32.mean(dim=normalized_dims, keepdim=True)
        centered = value32 - mean
        variance = centered.square().mean(dim=normalized_dims, keepdim=True)
        output = centered * torch.rsqrt(variance + self.eps)
        if self.elementwise_affine:
            output = output * self.weight.float()
            if self.bias is not None:
                output = output + self.bias.float()
        return output.to(original_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.device.type == "mps":
            return self._manual_forward(value)
        return super().forward(value)


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            MPSStableLayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


class PlannerHead3L(nn.Module):
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, out_dim)
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh
    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)
        return y.view(-1, self.nw, self.ad)


# ----------------------- Model -----------------------

@dataclass
class ModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    freeze_llm: bool = True
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = False
    # Action/target configuration
    use_tanh_actions: bool = True
    alpha_xy: Optional[float] = None
    default_dt: float = 0.1


class OpenTrackVLA(nn.Module):
    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.default_dt = float(cfg.default_dt)
        llm_path = resolve_local_model_path(
            label="Qwen/Qwen3-0.6B",
            repo_id="Qwen/Qwen3-0.6B",
            explicit=cfg.llm_name,
            env_var="QWEN_MODEL_PATH",
            candidates=default_qwen_candidates(),
        )
        self.cfg.llm_name = llm_path
        self.llm = AutoModel.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            local_files_only=True,
        )
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        self.D = self.llm.config.hidden_size
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        self.proj.requires_grad_(True)
        self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        action_dims = 3
        self.action_dims = action_dims
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        self.planner.requires_grad_(True)
        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False

        if cfg.alpha_xy is not None:
            vec = [1.0] * action_dims
            if action_dims >= 2:
                vec[0] = cfg.alpha_xy
                vec[1] = cfg.alpha_xy
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        else:
            alpha = torch.tensor((1.0, 1.0, 1.0), dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device):
        tok = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _interleave_tvi(self, tokens: torch.Tensor, t_idx: torch.Tensor, kind_id: int,
                        yaw_per_frame: Optional[torch.Tensor] = None, use_angle: bool = False) -> torch.Tensor:
        B, N, D = tokens.shape
        out_list = []
        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0
            fcount = 0
            while i < N:
                tcur = int(tb[i].item())
                j = i + 1
                while j < N and int(tb[j].item()) == tcur:
                    j += 1
                time_tok = self.tvi.make_time_token(tcur, kind_id, device=xb.device).unsqueeze(0)
                items.append(time_tok)
                if use_angle:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    angle_tok = self.tvi.make_angle_token(theta, kind_id, device=xb.device).unsqueeze(0)
                    items.append(angle_tok)
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def forward(self,
                coarse_tokens, coarse_tidx,
                fine_tokens, fine_tidx,
                instructions,
                yaw_hist: Optional[torch.Tensor] = None,
                yaw_curr: Optional[torch.Tensor] = None,
                bbox_feat: Optional[torch.Tensor] = None):
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        vis_c = self.proj(coarse_tokens.to(device))
        vis_f = self.proj(fine_tokens.to(device))
        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), kind_id=0,
            yaw_per_frame=yaw_hist, use_angle=self.cfg.use_angle_tvi
        )
        vis_f = self._interleave_tvi(
            vis_f, fine_tidx.to(device), kind_id=1,
            yaw_per_frame=yaw_curr, use_angle=self.cfg.use_angle_tvi
        )
        txt_emb, txt_mask = self._embed_text(instructions, device)
        extra = []
        act = self.act_token.expand(B, 1, -1)
        pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.llm.dtype)
        extra_len = (extra[0].size(1) if extra else 0)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, extra_len + vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)
        ], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :]
        h_act = h_act.float()
        a_hat = self.planner(h_act)
        tau_pred = a_hat * self.alpha_task
        return tau_pred


# ----------------------- Dataset -----------------------

@dataclass
class DataConfig:
    train_json: str
    n_waypoints: int = 8
    history: int = 31
    default_dt: float = 0.1
    cache_root: Optional[str] = None
    use_bbox_token: bool = False
    require_cached_tokens: bool = False


class JsonTrackingDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        p = Path(cfg.train_json)
        manifest_sources: List[Path] = []
        if p.is_file() and p.suffix.lower() == '.jsonl':
            manifest_sources = [p]
        elif p.is_dir():
            manifest_sources = sorted(p.rglob('*.jsonl'))
        manifests = [(source, load_dataset_manifest(source)) for source in manifest_sources]
        manifests = [(source, manifest) for source, manifest in manifests if manifest is not None]
        self.manifest = manifests[0][1] if len(manifests) == 1 else None
        self.default_dt = float(cfg.default_dt)
        if self.manifest is not None and not isinstance(self.manifest['dt'], list):
            self.default_dt = float(self.manifest['dt'])
        if manifests:
            roots = {
                str((source.parent / str(manifest['path_root'])).resolve())
                for source, manifest in manifests
            }
            if len(roots) != 1:
                raise ValueError(f"Dataset manifests disagree on path_root: {sorted(roots)}")
            self.base_root = Path(next(iter(roots)))
            if not self.base_root.exists():
                raise FileNotFoundError(f"Dataset manifest path_root does not exist: {self.base_root}")
            for source, manifest in manifests:
                if 'history' in manifest and int(manifest['history']) != int(cfg.history):
                    raise ValueError(
                        f"Dataset history mismatch for {source}: manifest={manifest['history']} cfg={cfg.history}"
                    )
                if 'n_waypoints' in manifest and int(manifest['n_waypoints']) != int(cfg.n_waypoints):
                    raise ValueError(
                        f"Dataset n_waypoints mismatch for {source}: "
                        f"manifest={manifest['n_waypoints']} cfg={cfg.n_waypoints}"
                    )
        else:
            candidate = p if p.is_dir() else p.parent
            max_up = 4
            while max_up >= 0 and not (candidate / 'frames').exists():
                if candidate.parent == candidate:
                    break
                candidate = candidate.parent
                max_up -= 1
            self.base_root = candidate
        self.cache_root = Path(cfg.cache_root) if cfg.cache_root is not None else (self.base_root / "vision_cache")
        self._online_encoder: Optional[VisionFeatureCacher] = None
        self._bbox_processor = None
        self._bbox_model = None
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None
        self.examples: Optional[List[Dict[str, Any]]] = None
        if p.is_file() and p.suffix.lower() == '.json':
            data = load_examples_from_path(cfg.train_json)
            assert isinstance(data, list) and len(data) > 0
            self.examples = data
        else:
            files: List[Path] = []
            if p.is_file() and p.suffix.lower() == '.jsonl':
                files = [p]
            elif p.is_dir():
                files = sorted(p.rglob('*.jsonl'))
            if len(files) == 0:
                raise FileNotFoundError(f"No .jsonl files found under: {cfg.train_json}")
            self._lazy = True
            self._index = []
            for fp in files:
                with open(fp, 'rb') as f:
                    pos = 0
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if line.strip():
                            self._index.append((str(fp), pos))
                        pos += len(line)
            if len(self._index) == 0:
                raise RuntimeError(f"No examples indexed from .jsonl sources under: {cfg.train_json}")
        H_target = int(self.cfg.history)
        self.coarse_frames = H_target

    def __len__(self):
        if self.examples is not None:
            return len(self.examples)
        if self._index is not None:
            return len(self._index)
        return 0

    def _load_tokens(self, path: str) -> torch.Tensor:
        return load_tokens_file(path)

    def _get_online_encoder(self) -> VisionFeatureCacher:
        if self._online_encoder is None:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            cfg = VisionCacheConfig(image_size=384, batch_size=8, device=('cuda' if use_cuda else 'cpu'))
            self._online_encoder = VisionFeatureCacher(cfg)
            self._online_encoder.eval()
        return self._online_encoder

    def _get_bbox_detector(self):
        if self._bbox_model is None or self._bbox_processor is None:
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            device = 'cuda' if use_cuda else 'cpu'
            try:
                self._bbox_processor, self._bbox_model = load_omdet_components(device)
            except Exception as _e:
                print(f"[bbox] failed to init omdet-turbo: {_e}")
                self._bbox_processor = None
                self._bbox_model = None
        return self._bbox_processor, self._bbox_model

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert('RGB')
        tok_dino, Hp, Wp = enc._encode_dino([pil])
        tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)
        return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        assert self._lazy and self._index is not None
        fp, off = self._index[idx]
        with open(fp, 'rb') as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode('utf-8'))

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._lazy:
            return self._read_indexed_example(idx)
        assert self.examples is not None
        return self.examples[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        H = self.coarse_frames

        curr_path = Path(ex['current'])
        abs_curr_img = curr_path if curr_path.is_absolute() else (self.base_root / curr_path)
        try:
            rel_curr = abs_curr_img.relative_to(self.base_root)
        except ValueError:
            rel_curr = abs_curr_img
        curr_token_dir = self.cache_root / rel_curr.parent
        curr_tok_path = token_cache_candidates(curr_token_dir, rel_curr, "fine")[0]
        try:
            fine_tokens = load_first_cached_token(
                token_cache_candidates(curr_token_dir, rel_curr, "fine")
            )
        except Exception:
            if self.cfg.require_cached_tokens:
                raise RuntimeError(
                    f"required fine vision token is missing or invalid: {curr_tok_path}"
                )
            curr_token_dir.mkdir(parents=True, exist_ok=True)
            vc, vf = self._encode_image_tokens(abs_curr_img)
            try:
                atomic_torch_save(vf.half(), curr_tok_path)
            except Exception:
                pass
            fine_tokens = vf
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=H, dtype=torch.long)

        imgs_src = ex.get('images', [])
        imgs_trim = imgs_src[-H:]
        missing = H - len(imgs_trim)
        coarse_list, coarse_tidx = [], []
        first_tok: Optional[torch.Tensor] = None
        current_vc: Optional[torch.Tensor] = None
        for t in range(H):
            if t < missing:
                tok = None
            else:
                img_p = imgs_trim[t - missing]
                rp = Path(img_p)
                abs_img = rp if rp.is_absolute() else (self.base_root / rp)
                try:
                    rel_img = abs_img.relative_to(self.base_root)
                except ValueError:
                    rel_img = abs_img
                token_dir = self.cache_root / rel_img.parent
                tok_path = token_cache_candidates(token_dir, rel_img, "coarse")[0]
                try:
                    tok = load_first_cached_token(
                        token_cache_candidates(token_dir, rel_img, "coarse")
                    )
                except Exception:
                    if self.cfg.require_cached_tokens:
                        raise RuntimeError(
                            f"required coarse vision token is missing or invalid: {tok_path}"
                        )
                    token_dir.mkdir(parents=True, exist_ok=True)
                    vc, vf = self._encode_image_tokens(abs_img)
                    try:
                        atomic_torch_save(vc.half(), tok_path)
                    except Exception:
                        pass
                    tok = vc
                if first_tok is None:
                    first_tok = tok
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    try:
                        if current_vc is None:
                            cur_coarse_path = token_cache_candidates(
                                curr_token_dir, rel_curr, "coarse"
                            )[0]
                            try:
                                current_vc = load_first_cached_token(
                                    token_cache_candidates(
                                        curr_token_dir, rel_curr, "coarse"
                                    )
                                )
                            except Exception:
                                if self.cfg.require_cached_tokens:
                                    raise RuntimeError(
                                        "required current-frame coarse token is missing or invalid: "
                                        f"{cur_coarse_path}"
                                    )
                                vc_tmp, _ = self._encode_image_tokens(abs_curr_img)
                                current_vc = vc_tmp
                        tok = current_vc
                    except Exception:
                        tok = torch.zeros(4, fine_tokens.size(1), dtype=torch.float32)
            coarse_list.append(tok)
            coarse_tidx.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))
        coarse_tokens = torch.cat(coarse_list, dim=0)
        coarse_tidx   = torch.cat(coarse_tidx, dim=0)

        yaw_hist = torch.tensor(ex.get('yaw_hist', [0.0]*H), dtype=torch.float32)
        yaw_curr = torch.tensor(ex.get('yaw_curr', 0.0), dtype=torch.float32).view(1)

        if 'waypoints' in ex:
            wp = torch.tensor(ex['waypoints'], dtype=torch.float32)
        else:
            assert 'actions' in ex, "JSON needs either 'waypoints' or 'actions'"
            dt = float(ex.get('dt', self.default_dt))
            traj = integrate_actions_to_waypoints(np.asarray(ex['actions'], dtype=np.float32), self.cfg.n_waypoints, dt)
            wp = torch.from_numpy(traj)

        if 'valid_mask' in ex:
            vm = torch.tensor(ex['valid_mask'], dtype=torch.bool)
        elif 'valid_idx' in ex:
            vm = torch.zeros(self.cfg.n_waypoints, dtype=torch.bool)
            vm[torch.tensor(ex['valid_idx'], dtype=torch.long)] = True
        else:
            vm = torch.ones(self.cfg.n_waypoints, dtype=torch.bool)

        raw_step_actions = ex.get('step_actions', ex.get('actions'))
        if raw_step_actions is None:
            raw_step_actions = np.zeros((self.cfg.n_waypoints, 3), dtype=np.float32)

        item: Dict[str, Any] = {
            'coarse_tokens': coarse_tokens,
            'coarse_tidx':   coarse_tidx,
            'fine_tokens':   fine_tokens,
            'fine_tidx':     fine_tidx,
            'yaw_hist':      yaw_hist,
            'yaw_curr':      yaw_curr,
            'waypoints':     wp,
            'step_actions':  torch.tensor(raw_step_actions, dtype=torch.float32),
            'delta_vel':     torch.tensor(ex.get('delta_vel', np.zeros((self.cfg.n_waypoints, 3))), dtype=torch.float32),
            'prev_action':   torch.tensor(ex.get('prev_action', [0.0, 0.0, 0.0]), dtype=torch.float32),
            'transition_type': str(ex.get('transition_type', 'other')),
            'command':          str(ex.get('command', 'unknown')),
            'source_raw_dir':   str(ex.get('source_raw_dir') or ex.get('episode', '')),
            'episode':        str(ex.get('episode', '')),
            'sequence_id':    str(ex.get('sequence_id', ex.get('chunk_id', ex.get('episode', '')))),
            'chunk_id':       str(ex.get('chunk_id', ex.get('episode', ''))),
            'clip_id':        str(ex.get('clip_id', ex.get('episode', ''))),
            'frame_idx':      int(ex.get('frame_idx', idx)),
            'mirrored':       bool(ex.get('mirrored', False)),
            'valid_mask':    vm,
            'instruction':   ex.get('instruction', 'follow the person'),
            'current_path':  str(abs_curr_img),
            'polar_theta_idx': torch.tensor(int(ex.get('polar_theta_idx', -1)), dtype=torch.long),
            'polar_dist_idx':  torch.tensor(int(ex.get('polar_dist_idx', -1)), dtype=torch.long),
            'polar_invalid':   torch.tensor(float(ex.get('polar_invalid', 1.0)), dtype=torch.float32),
        }
        for horizon in (4, 8, 16):
            item[f'fut_valid_{horizon}'] = torch.tensor(
                bool(ex.get(f'fut_valid_{horizon}', False)), dtype=torch.bool
            )
            item[f'fut_vis_{horizon}'] = torch.tensor(
                float(ex.get(f'fut_vis_{horizon}', 0.0)), dtype=torch.float32
            )
            item[f'fut_theta_idx_{horizon}'] = torch.tensor(
                int(ex.get(f'fut_theta_idx_{horizon}', -1)), dtype=torch.long
            )
            item[f'fut_dist_idx_{horizon}'] = torch.tensor(
                int(ex.get(f'fut_dist_idx_{horizon}', -1)), dtype=torch.long
            )
        if 'bbox' in ex:
            bb = np.asarray(ex['bbox'], dtype=np.float32)
            if bb.size == 4:
                x0,y0,x1,y1 = bb.tolist()
                w = max(1e-6, x1 - x0); h = max(1e-6, y1 - y0)
                cx = x0 + 0.5*w; cy = y0 + 0.5*h
                try:
                    with Image.open(str(abs_curr_img)) as _im:
                        W,H = _im.size
                    if max(cx,cy,w,h) > 1.5:
                        cx /= W; cy /= H; w /= W; h /= H
                except Exception:
                    pass
                item['bbox_feat'] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
        elif self.cfg.use_bbox_token:
            try:
                proc, det = self._get_bbox_detector()
                if proc is not None and det is not None:
                    with Image.open(str(abs_curr_img)).convert('RGB') as _im:
                        H, W = _im.height, _im.width
                        labels = ["person"]
                        inputs = proc(_im, text=labels, return_tensors='pt')
                        dev = next(det.parameters()).device
                        inputs = {k: v.to(dev) for k, v in inputs.items()}
                        with torch.inference_mode():
                            outputs = det(**inputs)
                        results = proc.post_process_grounded_object_detection(outputs,
                                                                             target_sizes=[(H, W)],
                                                                             text_labels=labels,
                                                                             threshold=0.25,
                                                                             nms_threshold=0.3)
                        res = results[0]
                        boxes = res.get('boxes', None)
                        scores = res.get('scores', None)
                        tlabels = res.get('text_labels', [])
                        if boxes is not None and scores is not None and len(boxes) > 0:
                            best = -1; best_s = -1.0
                            for i in range(len(boxes)):
                                lbl = tlabels[i] if i < len(tlabels) else ''
                                if lbl in ('person','human') and scores[i].item() > best_s:
                                    best_s = scores[i].item(); best = i
                            if best < 0:
                                best = int(torch.argmax(scores).item())
                            x0,y0,x1,y1 = boxes[best].tolist()
                            w = max(1e-6, x1 - x0); h = max(1e-6, y1 - y0)
                            cx = x0 + 0.5*w; cy = y0 + 0.5*h
                            cx /= W; cy /= H; w /= W; h /= H
                            item['bbox_feat'] = torch.tensor([cx, cy, w, h], dtype=torch.float32)
            except Exception:
                pass
        return item


# ----------------------- Collate -----------------------

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    instr = [b['instruction'] for b in batch]
    result = {
        'coarse_tokens': torch.stack([b['coarse_tokens'] for b in batch], dim=0),
        'coarse_tidx':   torch.stack([b['coarse_tidx']   for b in batch], dim=0),
        'fine_tokens':   torch.stack([b['fine_tokens']   for b in batch], dim=0),
        'fine_tidx':     torch.stack([b['fine_tidx']     for b in batch], dim=0),
        'yaw_hist':      torch.stack([b['yaw_hist']      for b in batch], dim=0),
        'yaw_curr':      torch.stack([b['yaw_curr']      for b in batch], dim=0),
        'waypoints':     torch.stack([b['waypoints']     for b in batch], dim=0),
        'step_actions':  torch.stack([b['step_actions']  for b in batch], dim=0),
        'delta_vel':     torch.stack([b['delta_vel']     for b in batch], dim=0),
        'prev_action':   torch.stack([b['prev_action']   for b in batch], dim=0),
        'transition_type': [b['transition_type'] for b in batch],
        'command':          [b.get('command', 'unknown') for b in batch],
        'source_raw_dir':   [b.get('source_raw_dir', b.get('episode', '')) for b in batch],
        'episode':        [b['episode'] for b in batch],
        'sequence_id':    [b['sequence_id'] for b in batch],
        'chunk_id':       [b['chunk_id'] for b in batch],
        'clip_id':        [b['clip_id'] for b in batch],
        'frame_idx':      torch.tensor([b['frame_idx'] for b in batch], dtype=torch.long),
        'mirrored':       torch.tensor([b['mirrored'] for b in batch], dtype=torch.bool),
        'valid_mask':    torch.stack([b['valid_mask']    for b in batch], dim=0),
        'instruction':   instr,
        'current_path':  [b['current_path'] for b in batch],
        'bbox_feat':     torch.stack([b['bbox_feat'] if 'bbox_feat' in b else torch.zeros(4, dtype=torch.float32) for b in batch], dim=0),
        'polar_theta_idx': torch.stack([b['polar_theta_idx'] for b in batch], dim=0),
        'polar_dist_idx':  torch.stack([b['polar_dist_idx'] for b in batch], dim=0),
        'polar_invalid':   torch.stack([b['polar_invalid'] for b in batch], dim=0),
    }
    for horizon in (4, 8, 16):
        for field in ('fut_valid', 'fut_vis', 'fut_theta_idx', 'fut_dist_idx'):
            key = f'{field}_{horizon}'
            result[key] = torch.stack([b[key] for b in batch], dim=0)
    return result


# ----------------------- Inference -----------------------

@torch.inference_mode()
def _run_inference(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = cfg.infer_ckpt
    try:
        if ckpt_path is None:
            from glob import glob
            pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
            ckpt_path = pts[-1] if pts else None
    except Exception:
        ckpt_path = None
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"No checkpoint found for inference (looked at --infer_ckpt and {cfg.out_dir})")

    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get('config', {})

    n_waypoints = int(ck_cfg.get('n_waypoints', getattr(cfg, 'n_waypoints', 8)))
    use_angle_tvi = bool(ck_cfg.get('use_angle_tvi', False))
    no_tanh_actions = bool(ck_cfg.get('no_tanh_actions', True))
    vision_feat_dim = int(ck_cfg.get('vision_feat_dim', getattr(cfg, 'vision_feat_dim', 1536)))
    alpha_xy        = ck_cfg.get('alpha_xy', getattr(cfg, 'alpha_xy', None))

    model = OpenTrackVLA(
        ModelConfig(
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get('beta_nav', 10.0)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()
    msd = ck.get('model_state', None)
    if msd:
        model.load_state_dict(msd, strict=False)

    if cfg.infer_json is None:
        raise ValueError('--infer_json is required for inference')
    vds = JsonTrackingDataset(DataConfig(train_json=cfg.infer_json, n_waypoints=n_waypoints, history=getattr(cfg, 'history', 31), cache_root=getattr(cfg, 'cache_root', None)))
    vdl = DataLoader(vds, batch_size=getattr(cfg, 'batch_size', 2), shuffle=False, num_workers=min(2, getattr(cfg, 'num_workers', 4)), pin_memory=True, collate_fn=collate_batch)

    os.makedirs(cfg.infer_out, exist_ok=True)
    vis_dir = os.path.join(cfg.infer_out, 'vis')
    npz_dir = os.path.join(cfg.infer_out, 'npz')
    if cfg.infer_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if cfg.infer_save_npz:
        os.makedirs(npz_dir, exist_ok=True)

    batches_limit = max(0, int(getattr(cfg, 'infer_batches', 0)))
    bdone = 0
    for bidx, batch in enumerate(vdl):
        coarse_tokens = batch['coarse_tokens'].to(device)
        coarse_tidx   = batch['coarse_tidx'].to(device)
        fine_tokens   = batch['fine_tokens'].to(device)
        fine_tidx     = batch['fine_tidx'].to(device)
        yaw_hist      = batch['yaw_hist'].to(device)
        yaw_curr      = batch['yaw_curr'].to(device)
        instr         = batch['instruction']
        bbox_feat     = batch.get('bbox_feat', None)

        pred = model(
            coarse_tokens, coarse_tidx,
            fine_tokens, fine_tidx,
            instr,
            yaw_hist=yaw_hist if use_angle_tvi else None,
            yaw_curr=yaw_curr if use_angle_tvi else None
        )

        if cfg.infer_save_npz:
            try:
                with torch.no_grad():
                    pred_np = pred.detach().float().cpu().numpy()
                Bcur = pred_np.shape[0]
                for bi in range(Bcur):
                    fpath = os.path.join(npz_dir, f"b{bidx:06d}_i{bi:03d}.npz")
                    np.savez_compressed(
                        fpath,
                        pred=pred_np[bi],
                        instruction=instr[bi],
                        current_path=(batch.get('current_path', [''])[bi] if isinstance(batch.get('current_path', []), list) else '')
                    )
            except Exception:
                pass

        if cfg.infer_vis:
            try:
                with torch.no_grad():
                    pred_draw = pred.detach().float()
                    try:
                        model_inspect = model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and pred_draw.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                            max_xy = pred_draw[..., 0:2].abs().max().item()
                            if max_xy <= 1.5:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6).to(pred_draw.device, pred_draw.dtype)
                                pred_draw = pred_draw.clone()
                                pred_draw[..., 0:2] = pred_draw[..., 0:2] * ax
                    except Exception:
                        pass
                    pred_np = pred_draw.cpu().numpy()
                cur_paths = batch.get('current_path', [])
                Bcur = pred_np.shape[0]
                for bi in range(min(Bcur, 4)):
                    cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                    if cur_path is None or (not os.path.exists(cur_path)):
                        continue
                    pil_img = Image.open(cur_path).convert('RGB')
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size
                    base_x = w // 2
                    base_y = int(h * 0.86)
                    def to_pxxy(traj):
                        pts = []
                        for i in range(min(traj.shape[0], 64)):
                            x, y = float(traj[i, 0]), float(traj[i, 1])
                            px = base_x - int(y * 120)
                            py = base_y - int(x * 120)
                            pts.append((px, py))
                        return pts
                    pts_pred = to_pxxy(pred_np[bi])
                    for i2 in range(1, len(pts_pred)):
                        draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                    if pts_pred:
                        r0 = 6
                        sx, sy = pts_pred[0]
                        draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                    out_path = os.path.join(vis_dir, f"b{bidx:06d}_i{bi:03d}.jpg")
                    pil_img.save(out_path)
            except Exception:
                pass

        bdone += 1
        if batches_limit and bdone >= batches_limit:
            break

    print(f"[INFER] Done. Outputs under {cfg.infer_out}")


# ----------------------- CLI -----------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infer_json', type=str, required=True, help='Run inference on this dataset (json/jsonl/dir)')
    ap.add_argument('--infer_ckpt', type=str, default=None, help='Checkpoint to load for inference (defaults to latest in out_dir)')
    ap.add_argument('--out_dir', type=str, default='./ckpts', help='Directory where checkpoints are stored (for default lookup)')
    ap.add_argument('--infer_out', type=str, default='./infer_out', help='Output directory for inference results')
    ap.add_argument('--infer_batches', type=int, default=0, help='Limit number of batches to run at inference (0 = all)')
    ap.add_argument('--infer_vis', action='store_true', help='Save visualization images during inference')
    ap.add_argument('--infer_save_npz', action='store_true', help='Save npz predictions during inference')
    ap.add_argument('--n_waypoints', type=int, default=8)
    ap.add_argument('--history', type=int, default=31)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--vision_feat_dim', type=int, default=1536)
    ap.add_argument('--cache_root', type=str, default=None)
    ap.add_argument('--alpha_xy', type=float, default=None)
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    cfg = parse_args()
    _run_inference(cfg)
