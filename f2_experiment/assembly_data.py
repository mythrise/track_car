"""Fail-closed data-side production assembly for the isolated F2 smoke.

This module owns only the data plane of the F2 production assembly
(blockers 6, 9, and 10 of the 2026-07-18 handoff):

* frozen-asset SHA bindings for the official base HF checkpoint, the
  vision cache (manifest/provenance/token payload), the DINOv3/SigLIP
  encoder provenance, the local Qwen weights, and the prompt
  normalization erratum;
* a cache-only observation loader that mirrors the vendored
  ``JsonTrackingDataset.__getitem__`` token path byte-for-byte on the
  success path while deleting every online-recompute and zero-fill
  fallback branch (missing or invalid tokens fail closed);
* a train-split per-file token hash ledger: the cache manifest carries
  only an aggregate ``token_payload_sha256`` (a sequential stream hash
  over all three splits, unverifiable per file and re-hashable only by
  reading the sealed internal-test subtree), so
  :func:`build_train_token_ledger` derives the token file set strictly
  from the frozen train JSONL and freezes one SHA-256 per token file;
  :func:`load_cached_observation` then verifies every token's bytes
  against the ledger before deserializing (``F2_CACHE_TAMPERED`` on
  mismatch) and never opens a file the ledger does not list;
* the machine-enforced :class:`ObservationPacket` allowlist that makes it
  structurally impossible for ``step_actions``, future labels, or expert
  actions to enter the observation seen by ``feature_forward``;
* the block-major JSONL to :class:`~f2_experiment.runner.RunnerRow`
  loader for the frozen CAL / SMK-TRAIN / EVAL-FIX supports, validated
  against the frozen support receipt row SHAs;
* the support reset plan derived from the receipt and cross-checked
  against the exact runner reset predicates; and
* the EVAL-FIX overall/change/turn/other stratum labels.

No model, optimizer, trainer, or evaluator is constructed here, and no
heavyweight third_party module (transformers/PIL) is imported: only the
stdlib-based ``experiment_binding`` and ``local_weights`` helpers.  The
sealed internal held-out test is never read.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch

from third_party.OpenTrackVLA.experiment_binding import (
    ExperimentBindingError,
    bind_hf_model_artifact,
    sha256_artifact,
    sha256_file,
    verify_vision_cache,
)
from third_party.OpenTrackVLA.local_weights import (
    default_qwen_candidates,
    resolve_local_model_path,
)

from .evaluation import strata_masks_from_rows
from .model import ACTION_MAX_ABS, AP2_HORIZON
from .runner import RunnerRow
from .support import (
    FROZEN_TRAIN_RELATIVE,
    FROZEN_TRAIN_ROWS,
    FROZEN_TRAIN_SHA256,
    SUPPORT_EXPECTATIONS,
    F2ContractError,
    FrozenSupportReceipt,
    canonical_json_sha256,
    continues_sequence,
    parse_train_jsonl,
)


class F2AssemblyContractError(F2ContractError):
    """Raised whenever the F2 data-side assembly must fail closed."""


# --------------------------------------------------------------------------
# Frozen asset identities (handoff section 6; live-verified 2026-07-18).
# --------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_BASE_HF_DIR_DEFAULT = (
    _PROJECT_ROOT.parent / "opentrackvla-qwen06b"
).resolve()
FROZEN_BASE_HF_ARTIFACT_SHA256 = (
    "ff1d31982271cb922c91a26f7767438124e12502e9341251041d8541f7d63a8f"
)
FROZEN_CACHE_ROOT_RELATIVE = PurePosixPath("data/collected_v1/vision_cache")
FROZEN_CACHE_RECORDED_PROJECT_ROOT = "/Users/mythrise/科研实习/track_car"
FROZEN_CACHE_MANIFEST_SHA256 = (
    "127bda80a3d748f704b01bcf456c1e2e7c6c5b607f7eebe848fb5dc0e7824009"
)
FROZEN_CACHE_PROVENANCE_SHA256 = (
    "5399927be976e13f7c180143514f0863ac8687faf800fd1612cb3b9c42640ba4"
)
FROZEN_TOKEN_PAYLOAD_SHA256 = (
    "f0016a2a25f8724ec45040eedb4ce73e54ca342ba1cf400a4c0a6ab0e1592744"
)
FROZEN_DINO_SHA256 = (
    "627c7bb4f39f79e15d5e3fdf61557172d11befbe0b42c6f4513bf3907f5fc7a1"
)
FROZEN_SIGLIP_SHA256 = (
    "e9549756bf15a3ff2064c8a32f1086e9391f374682ca16a05c30f91fcbb5a096"
)
FROZEN_QWEN_SHA256 = (
    "2f62d9a42d8cf3cd43a69155c345e024d0d5bd1590a701540c0f75aeae71162b"
)
FROZEN_QWEN_REPO_ID = "Qwen/Qwen3-0.6B"
FROZEN_PROMPT_ERRATUM_RELATIVE = Path(
    "data/collected_v1/audits/prompt_normalization_erratum_v4.json"
)
FROZEN_PROMPT_ERRATUM_SHA256 = (
    "baa9c322366e40377858cdedc9618dcc08e419df7991ae4bd3e7ca499facdbec"
)

FROZEN_TRAIN_IMAGE_PREFIXES = (
    "data/collected_v1/datasets/.train_mirrored_images",
    "data/collected_v1/episodes/train",
)
FROZEN_INTERNAL_TEST_IMAGE_PREFIX = "data/collected_v1/episodes/test"

HISTORY_FRAMES = 31
FINE_TOKEN_COUNT = 64
COARSE_TOKEN_COUNT = 4
VISION_FEATURE_DIM = 1536
POLAR_THETA_BINS = 60
POLAR_DIST_BINS = 30
AUX_FUTURE_HORIZONS = (4, 8, 16)
AUX_FUT_KEYS = tuple(
    f"fut_{kind}_{horizon}"
    for horizon in AUX_FUTURE_HORIZONS
    for kind in ("valid", "vis", "theta_idx", "dist_idx")
)
SUPPORT_NAMES = ("CAL", "SMK-TRAIN", "EVAL-FIX")

OBSERVATION_ALLOWED_KEYS = frozenset(
    {
        "coarse_tokens",
        "coarse_tidx",
        "fine_tokens",
        "fine_tidx",
        "instruction",
        "yaw_hist",
        "yaw_curr",
    }
)
OBSERVATION_OPTIONAL_KEYS = frozenset({"yaw_hist", "yaw_curr"})
OBSERVATION_FORBIDDEN_KEYS = frozenset(
    {
        "step_actions",
        "actions",
        "waypoints",
        "prev_action",
        "delta_vel",
        "delta_pos",
        "motors",
        "polar_theta_idx",
        "polar_dist_idx",
        "polar_invalid",
    }
) | frozenset(AUX_FUT_KEYS)


# --------------------------------------------------------------------------
# Small fail-closed validators.
# --------------------------------------------------------------------------


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise F2AssemblyContractError(f"{label} must be a mapping")
    return value


def _require_str(row: Mapping[str, Any], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise F2AssemblyContractError(f"{label} has invalid {key!r}")
    return value


def _require_int(row: Mapping[str, Any], key: str, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise F2AssemblyContractError(f"{label} has invalid {key!r}")
    return int(value)


def _require_bool(row: Mapping[str, Any], key: str, label: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise F2AssemblyContractError(f"{label} has invalid {key!r}")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise F2AssemblyContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise F2AssemblyContractError(f"{label} must be finite")
    return result


def _data_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise F2AssemblyContractError(f"{label} must be a torch.Tensor")
    if value.requires_grad:
        raise F2AssemblyContractError(
            f"{label} must not require grad inside a data packet"
        )
    return value


def _finite_float_tensor(value: Any, label: str) -> torch.Tensor:
    tensor = _data_tensor(value, label)
    if not tensor.is_floating_point():
        raise F2AssemblyContractError(f"{label} must be floating point")
    if bool((~torch.isfinite(tensor)).any().item()):
        raise F2AssemblyContractError(f"{label} is nonfinite")
    return tensor


def _long_tensor(value: Any, label: str) -> torch.Tensor:
    tensor = _data_tensor(value, label)
    if tensor.dtype != torch.long:
        raise F2AssemblyContractError(f"{label} must have dtype torch.long")
    return tensor


# --------------------------------------------------------------------------
# Blocker 10: machine-enforced observation allowlist.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationPacket:
    """The only value ``feature_forward`` may receive as observation.

    A frozen dataclass with exactly the seven allowlisted fields; the
    constructor validates shapes, dtypes, finiteness, and grad-freedom, so
    expert actions, ``step_actions``, and future labels cannot be smuggled
    in structurally.
    """

    coarse_tokens: torch.Tensor
    coarse_tidx: torch.Tensor
    fine_tokens: torch.Tensor
    fine_tidx: torch.Tensor
    instruction: str
    yaw_hist: torch.Tensor | None = None
    yaw_curr: torch.Tensor | None = None

    def __post_init__(self) -> None:
        fine_tokens = _finite_float_tensor(self.fine_tokens, "fine_tokens")
        if fine_tokens.shape != (FINE_TOKEN_COUNT, VISION_FEATURE_DIM):
            raise F2AssemblyContractError(
                "fine_tokens must have shape "
                f"({FINE_TOKEN_COUNT},{VISION_FEATURE_DIM})"
            )
        fine_tidx = _long_tensor(self.fine_tidx, "fine_tidx")
        if fine_tidx.shape != (FINE_TOKEN_COUNT,):
            raise F2AssemblyContractError(
                f"fine_tidx must have shape ({FINE_TOKEN_COUNT},)"
            )
        history_values = torch.unique(fine_tidx)
        if history_values.numel() != 1:
            raise F2AssemblyContractError("fine_tidx must be constant")
        history = int(history_values.item())
        if history < 1:
            raise F2AssemblyContractError("fine_tidx history must be >= 1")
        coarse_tokens = _finite_float_tensor(self.coarse_tokens, "coarse_tokens")
        expected_rows = history * COARSE_TOKEN_COUNT
        if coarse_tokens.shape != (expected_rows, VISION_FEATURE_DIM):
            raise F2AssemblyContractError(
                "coarse_tokens must have shape "
                f"({expected_rows},{VISION_FEATURE_DIM}) for history {history}"
            )
        coarse_tidx = _long_tensor(self.coarse_tidx, "coarse_tidx")
        expected_tidx = torch.arange(history, dtype=torch.long).repeat_interleave(
            COARSE_TOKEN_COUNT
        )
        if coarse_tidx.shape != expected_tidx.shape or not bool(
            torch.equal(coarse_tidx, expected_tidx)
        ):
            raise F2AssemblyContractError(
                "coarse_tidx must enumerate each history frame "
                f"{COARSE_TOKEN_COUNT} times in order"
            )
        if not isinstance(self.instruction, str) or not self.instruction:
            raise F2AssemblyContractError("instruction must be a nonempty string")
        if self.yaw_hist is not None:
            yaw_hist = _finite_float_tensor(self.yaw_hist, "yaw_hist")
            if yaw_hist.shape != (history,):
                raise F2AssemblyContractError(
                    f"yaw_hist must have shape ({history},)"
                )
        if self.yaw_curr is not None:
            yaw_curr = _finite_float_tensor(self.yaw_curr, "yaw_curr")
            if yaw_curr.numel() != 1:
                raise F2AssemblyContractError("yaw_curr must contain one scalar")

    @property
    def history_frames(self) -> int:
        return int(self.fine_tidx[0].item())


def observation_packet_from_fields(
    fields: Mapping[str, Any],
) -> ObservationPacket:
    """Build an :class:`ObservationPacket` from an allowlisted field mapping.

    Any forbidden key (``step_actions``, future labels, expert/previous
    actions, polar labels), any unknown key, and any missing required key
    raise :class:`F2AssemblyContractError`.  This is the only sanctioned
    constructor path, so whole-row dictionaries can never leak through.
    """

    mapping = _require_mapping(fields, "observation fields")
    keys = set()
    for key in mapping:
        if not isinstance(key, str):
            raise F2AssemblyContractError("observation field keys must be strings")
        keys.add(key)
    leaked = sorted(keys & OBSERVATION_FORBIDDEN_KEYS)
    if leaked:
        raise F2AssemblyContractError(
            f"OBSERVATION_LEAK: forbidden observation keys {leaked!r}"
        )
    unknown = sorted(keys - OBSERVATION_ALLOWED_KEYS)
    if unknown:
        raise F2AssemblyContractError(
            f"OBSERVATION_LEAK: unknown observation keys {unknown!r}"
        )
    missing = sorted((OBSERVATION_ALLOWED_KEYS - OBSERVATION_OPTIONAL_KEYS) - keys)
    if missing:
        raise F2AssemblyContractError(
            f"observation fields are missing {missing!r}"
        )
    return ObservationPacket(
        coarse_tokens=mapping["coarse_tokens"],
        coarse_tidx=mapping["coarse_tidx"],
        fine_tokens=mapping["fine_tokens"],
        fine_tidx=mapping["fine_tidx"],
        instruction=mapping["instruction"],
        yaw_hist=mapping.get("yaw_hist"),
        yaw_curr=mapping.get("yaw_curr"),
    )


def ensure_observation_packet(value: Any) -> ObservationPacket:
    """Type gate for callbacks: reject anything but an ObservationPacket."""

    if not isinstance(value, ObservationPacket):
        raise F2AssemblyContractError(
            "OBSERVATION_LEAK: feature_forward observation must be an "
            f"ObservationPacket, got {type(value).__name__}"
        )
    return value


@dataclass(frozen=True)
class AuxTargetPacket:
    """Label-side auxiliary targets, fully detached from any graph."""

    theta_idx: torch.Tensor
    dist_idx: torch.Tensor
    invalid: torch.Tensor
    fut: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        theta = _long_tensor(self.theta_idx, "theta_idx")
        dist = _long_tensor(self.dist_idx, "dist_idx")
        invalid = _finite_float_tensor(self.invalid, "invalid")
        for name, tensor in (
            ("theta_idx", theta),
            ("dist_idx", dist),
            ("invalid", invalid),
        ):
            if tensor.shape != (1,):
                raise F2AssemblyContractError(f"{name} must have shape (1,)")
        invalid_value = float(invalid.item())
        if invalid_value not in (0.0, 1.0):
            raise F2AssemblyContractError("invalid must be exactly 0.0 or 1.0")
        theta_value = int(theta.item())
        dist_value = int(dist.item())
        floor = -1 if invalid_value == 1.0 else 0
        if not floor <= theta_value < POLAR_THETA_BINS:
            raise F2AssemblyContractError("theta_idx lies outside the frozen bins")
        if not floor <= dist_value < POLAR_DIST_BINS:
            raise F2AssemblyContractError("dist_idx lies outside the frozen bins")
        fut = _require_mapping(self.fut, "fut")
        if set(fut) != set(AUX_FUT_KEYS):
            raise F2AssemblyContractError(
                f"fut must contain exactly the {len(AUX_FUT_KEYS)} frozen keys"
            )
        frozen: dict[str, torch.Tensor] = {}
        for horizon in AUX_FUTURE_HORIZONS:
            valid = _data_tensor(fut[f"fut_valid_{horizon}"], "fut_valid")
            if valid.dtype != torch.bool or valid.shape != (1,):
                raise F2AssemblyContractError(
                    f"fut_valid_{horizon} must be a bool tensor of shape (1,)"
                )
            vis = _finite_float_tensor(fut[f"fut_vis_{horizon}"], "fut_vis")
            if vis.shape != (1,):
                raise F2AssemblyContractError(
                    f"fut_vis_{horizon} must have shape (1,)"
                )
            vis_value = float(vis.item())
            if not 0.0 <= vis_value <= 1.0:
                raise F2AssemblyContractError(
                    f"fut_vis_{horizon} must lie in [0,1]"
                )
            future_theta = _long_tensor(
                fut[f"fut_theta_idx_{horizon}"], "fut_theta_idx"
            )
            future_dist = _long_tensor(
                fut[f"fut_dist_idx_{horizon}"], "fut_dist_idx"
            )
            for name, tensor, bins in (
                (f"fut_theta_idx_{horizon}", future_theta, POLAR_THETA_BINS),
                (f"fut_dist_idx_{horizon}", future_dist, POLAR_DIST_BINS),
            ):
                if tensor.shape != (1,):
                    raise F2AssemblyContractError(f"{name} must have shape (1,)")
                index_value = int(tensor.item())
                if not -1 <= index_value < bins:
                    raise F2AssemblyContractError(
                        f"{name} lies outside the frozen bins"
                    )
            if bool(valid.item()) and vis_value > 0.5:
                if int(future_theta.item()) < 0 or int(future_dist.item()) < 0:
                    raise F2AssemblyContractError(
                        f"fut horizon {horizon} is visible but has sentinel bins"
                    )
        for key in AUX_FUT_KEYS:
            frozen[key] = fut[key].detach()
        object.__setattr__(self, "theta_idx", theta.detach())
        object.__setattr__(self, "dist_idx", dist.detach())
        object.__setattr__(self, "invalid", invalid.detach())
        object.__setattr__(self, "fut", MappingProxyType(frozen))

    def as_targets(self) -> dict[str, torch.Tensor]:
        """Adapter-format targets for ``compute_aux_losses`` (batch of one)."""

        targets: dict[str, torch.Tensor] = {
            "polar_theta_idx": self.theta_idx,
            "polar_dist_idx": self.dist_idx,
            "polar_invalid": self.invalid,
        }
        targets.update(self.fut)
        return targets


def aux_target_packet_from_row(
    row: Mapping[str, Any], label: str
) -> AuxTargetPacket:
    """Build the detached auxiliary label packet for one frozen train row."""

    mapping = _require_mapping(row, label)
    theta = _require_int(mapping, "polar_theta_idx", label)
    dist = _require_int(mapping, "polar_dist_idx", label)
    invalid = _finite_float(mapping.get("polar_invalid"), f"{label}.polar_invalid")
    fut: dict[str, torch.Tensor] = {}
    for horizon in AUX_FUTURE_HORIZONS:
        valid = _require_bool(mapping, f"fut_valid_{horizon}", label)
        vis = _finite_float(
            mapping.get(f"fut_vis_{horizon}"), f"{label}.fut_vis_{horizon}"
        )
        fut[f"fut_valid_{horizon}"] = torch.tensor([valid], dtype=torch.bool)
        fut[f"fut_vis_{horizon}"] = torch.tensor([vis], dtype=torch.float32)
        fut[f"fut_theta_idx_{horizon}"] = torch.tensor(
            [_require_int(mapping, f"fut_theta_idx_{horizon}", label)],
            dtype=torch.long,
        )
        fut[f"fut_dist_idx_{horizon}"] = torch.tensor(
            [_require_int(mapping, f"fut_dist_idx_{horizon}", label)],
            dtype=torch.long,
        )
    return AuxTargetPacket(
        theta_idx=torch.tensor([theta], dtype=torch.long),
        dist_idx=torch.tensor([dist], dtype=torch.long),
        invalid=torch.tensor([invalid], dtype=torch.float32),
        fut=fut,
    )


# --------------------------------------------------------------------------
# Train-split per-file token hash ledger (P1-3 data side).
#
# The frozen cache manifest only binds an aggregate token payload SHA over
# all three splits; verifying it re-reads the sealed internal-test cache
# subtree, which is forbidden.  The ledger below freezes one SHA-256 per
# token file for exactly the token set reachable from the frozen train
# JSONL, so every production load can be byte-verified without ever
# touching the sealed subtree.
# --------------------------------------------------------------------------

_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class TokenHashLedger:
    """Frozen mapping of cache-relative token paths to byte SHA-256 values.

    An empty ledger is legal but rejects every load (fail-closed); the
    production ledger is built once by :func:`build_train_token_ledger`
    and its ``ledger_sha256`` belongs in assembly receipt v4.
    """

    entries: Mapping[str, str]

    def __post_init__(self) -> None:
        mapping = _require_mapping(self.entries, "token ledger entries")
        frozen: dict[str, str] = {}
        for key, value in mapping.items():
            if not isinstance(key, str) or not key:
                raise F2AssemblyContractError(
                    "token ledger keys must be nonempty strings"
                )
            key_path = PurePosixPath(key)
            if (
                "\\" in key
                or key_path.is_absolute()
                or PureWindowsPath(key).is_absolute()
                or key_path.as_posix() != key
                or any(part in ("..", ".") for part in key_path.parts)
            ):
                raise F2AssemblyContractError(
                    f"token ledger key must be a clean relative path: {key!r}"
                )
            if (
                not isinstance(value, str)
                or len(value) != 64
                or not set(value) <= _HEX_DIGITS
            ):
                raise F2AssemblyContractError(
                    f"token ledger value for {key!r} must be lowercase sha256 hex"
                )
            frozen[key] = value
        object.__setattr__(
            self, "entries", MappingProxyType(dict(sorted(frozen.items())))
        )

    @property
    def token_files(self) -> int:
        return len(self.entries)

    @property
    def ledger_sha256(self) -> str:
        return canonical_json_sha256(dict(self.entries))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_class": "f2_train_token_hash_ledger",
            "token_files": self.token_files,
            "ledger_sha256": self.ledger_sha256,
            "entries": dict(self.entries),
        }


def _require_ledger(value: Any) -> TokenHashLedger:
    if not isinstance(value, TokenHashLedger):
        raise F2AssemblyContractError(
            "token_ledger must be a TokenHashLedger; loading without byte "
            "verification is forbidden"
        )
    return value


def _resolved_cache_file(path: Path, cache_root: Path, label: str) -> Path:
    """Resolve a token path and reject junction/symlink escapes."""

    root = cache_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise F2AssemblyContractError(
            f"F2_CACHE_PATH_ESCAPE: {label} resolves outside the frozen "
            f"cache root: {path} -> {resolved}"
        ) from exc
    return resolved


def collect_image_relpaths(
    rows: Sequence[Mapping[str, Any]], base_root: str | Path
) -> tuple[str, ...]:
    """Unique sorted base-relative image paths referenced by ``rows``.

    Only ``current`` and ``images`` are consulted; every path must stay
    under ``base_root``.  This is the sole source of the token file set,
    so a ledger built from train rows structurally cannot name any
    internal-test file.
    """

    base = Path(base_root).expanduser().resolve()
    if not base.is_dir():
        raise F2AssemblyContractError(f"base_root does not exist: {base}")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise F2AssemblyContractError("rows must be an ordered sequence")
    raw_values: set[str] = set()
    for index, row in enumerate(rows):
        mapping = _require_mapping(row, f"row {index}")
        raw_values.add(_require_str(mapping, "current", f"row {index}"))
        images = mapping.get("images")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            raise F2AssemblyContractError(f"row {index} has invalid 'images'")
        for position, value in enumerate(images):
            if not isinstance(value, str) or not value:
                raise F2AssemblyContractError(
                    f"row {index}.images[{position}] must be a nonempty string"
                )
            raw_values.add(value)
    relpaths: set[str] = set()
    for value in raw_values:
        _absolute, relative = _resolve_image(base, value, "train image")
        relpaths.add(relative.as_posix())
    return tuple(sorted(relpaths))


def _assert_train_subtree(relpaths: Sequence[str]) -> None:
    for relpath in relpaths:
        if relpath == FROZEN_INTERNAL_TEST_IMAGE_PREFIX or relpath.startswith(
            FROZEN_INTERNAL_TEST_IMAGE_PREFIX + "/"
        ):
            raise F2AssemblyContractError(
                "INTERNAL_TEST_SEAL: train rows reference the sealed "
                f"internal-test subtree: {relpath}"
            )
        if not any(
            relpath == prefix or relpath.startswith(prefix + "/")
            for prefix in FROZEN_TRAIN_IMAGE_PREFIXES
        ):
            raise F2AssemblyContractError(
                "train image path lies outside the frozen train prefixes: "
                f"{relpath}"
            )


def _ledger_from_relpaths(
    relpaths: Sequence[str], cache_root: Path
) -> TokenHashLedger:
    if not relpaths:
        raise F2AssemblyContractError(
            "token ledger requires at least one referenced image"
        )
    entries: dict[str, str] = {}
    for relpath in relpaths:
        relative = Path(relpath)
        for level in ("fine", "coarse"):
            candidates = _token_candidates(
                cache_root / relative.parent, relative, level
            )
            chosen: tuple[Path, Path] | None = None
            for candidate in candidates:
                resolved = _resolved_cache_file(
                    candidate, cache_root, f"{level} token for {relpath}"
                )
                if resolved.is_file():
                    chosen = (candidate, resolved)
                    break
            if chosen is None:
                raise F2AssemblyContractError(
                    f"F2_CACHE_MISS: {level} token for image {relpath} is "
                    "missing from the frozen cache; online recomputation is "
                    "forbidden"
                )
            logical, resolved = chosen
            entries[logical.relative_to(cache_root).as_posix()] = sha256_file(
                resolved
            )
    return TokenHashLedger(entries=entries)


def build_token_ledger_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_root: str | Path,
    cache_root: str | Path,
) -> TokenHashLedger:
    """Hash the fine+coarse token files for every image ``rows`` reference."""

    base = Path(base_root).expanduser().resolve()
    cache = Path(cache_root).expanduser().resolve()
    if not cache.is_dir():
        raise F2AssemblyContractError(f"cache_root does not exist: {cache}")
    relpaths = collect_image_relpaths(rows, base)
    return _ledger_from_relpaths(relpaths, cache)


def build_train_token_ledger(project_root: str | Path) -> TokenHashLedger:
    """Build the production per-file token ledger from the frozen train.

    Fails closed on: missing train JSONL, train SHA drift (HS1), any
    image path outside the frozen train prefixes or inside the sealed
    internal-test subtree, and any missing token file.  Reads only token
    files reachable from train rows; the sealed subtree is never opened.
    """

    root = Path(project_root).expanduser().resolve()
    train_path = (root / FROZEN_TRAIN_RELATIVE).resolve()
    if not train_path.is_file():
        raise F2AssemblyContractError(
            f"frozen train JSONL is missing: {train_path}"
        )
    payload = train_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != FROZEN_TRAIN_SHA256:
        raise F2AssemblyContractError(
            "HS1_train_precondition: frozen train JSONL SHA mismatch"
        )
    rows = parse_train_jsonl(payload)
    base_root, cache_root = frozen_cache_roots(root)
    relpaths = collect_image_relpaths(rows, base_root)
    _assert_train_subtree(relpaths)
    return _ledger_from_relpaths(relpaths, cache_root)


# --------------------------------------------------------------------------
# Blocker 6b: cache-only observation loader (no online recompute, ever).
# --------------------------------------------------------------------------

_TOKEN_LEVEL_SUFFIX = {"fine": "vfine", "coarse": "vcoarse"}
_TOKEN_LEVEL_SHAPE = {
    "fine": (FINE_TOKEN_COUNT, VISION_FEATURE_DIM),
    "coarse": (COARSE_TOKEN_COUNT, VISION_FEATURE_DIM),
}
_TOKEN_PAYLOAD_KEYS = ("V", "Vfine", "Vcoarse", "tokens", "feat", "features")


def _token_candidates(
    token_dir: Path, image_relative: Path, level: str
) -> tuple[Path, ...]:
    """Mirror ``model.token_cache_candidates``: v2 name first, then legacy."""

    suffix = _TOKEN_LEVEL_SUFFIX.get(level)
    if suffix is None:
        raise F2AssemblyContractError(f"unknown vision token level: {level!r}")
    primary = token_dir / f"{image_relative.name}_{suffix}.pt"
    legacy = token_dir / f"{image_relative.stem}_{suffix}.pt"
    if primary == legacy:
        return (primary,)
    return (primary, legacy)


def _coerce_token_payload(payload: Any, path: Path) -> torch.Tensor:
    """Mirror ``model.load_tokens_file`` payload coercion (read-only)."""

    if isinstance(payload, torch.Tensor):
        return payload.float()
    if isinstance(payload, dict):
        for key in _TOKEN_PAYLOAD_KEYS:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                if value.dim() == 3 and value.size(0) == 1:
                    value = value[0]
                return value.float()
    raise F2AssemblyContractError(f"unrecognized vision token payload: {path}")


def _load_cached_token(
    candidates: Sequence[Path],
    *,
    level: str,
    label: str,
    cache_root: Path,
    ledger: TokenHashLedger,
) -> torch.Tensor:
    """Byte-verify and deserialize the first ledger-listed candidate.

    Membership is checked before any file is opened, so unlisted paths
    (including anything under the sealed internal-test subtree) are never
    read.  The bytes are read exactly once, hashed against the frozen
    ledger (mismatch is ``F2_CACHE_TAMPERED``), and deserialized from the
    same buffer, so there is no verify/load TOCTOU window.
    """

    last_error: Exception | None = None
    tensor: torch.Tensor | None = None
    for path in candidates:
        resolved_path = _resolved_cache_file(path, cache_root, label)
        try:
            key = path.relative_to(cache_root).as_posix()
        except ValueError as exc:
            raise F2AssemblyContractError(
                f"{label} {level} token path escapes the cache root: {path}"
            ) from exc
        expected_sha = ledger.entries.get(key)
        if expected_sha is None:
            continue  # unlisted candidates are never opened
        try:
            raw = resolved_path.read_bytes()
        except OSError as exc:
            last_error = exc
            continue
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise F2AssemblyContractError(
                f"F2_CACHE_TAMPERED: {level} token bytes for {label} ({key}) "
                "do not match the frozen token ledger"
            )
        try:
            payload = torch.load(io.BytesIO(raw), map_location="cpu")
            tensor = _coerce_token_payload(payload, path)
        except Exception as exc:  # noqa: BLE001 - any decode failure is final
            raise F2AssemblyContractError(
                f"F2_CACHE_MISS: {level} token for {label} ({key}) matches "
                "the frozen ledger but cannot be deserialized"
            ) from exc
        break
    if tensor is None:
        raise F2AssemblyContractError(
            f"F2_CACHE_MISS: required {level} vision token for {label} is "
            f"missing, unreadable, or not listed in the frozen token ledger "
            f"({candidates[0]}); online recomputation is forbidden"
        ) from last_error
    expected_shape = _TOKEN_LEVEL_SHAPE[level]
    if tensor.shape != expected_shape:
        raise F2AssemblyContractError(
            f"{label} {level} token has shape {tuple(tensor.shape)}; the "
            f"frozen layout requires {expected_shape}"
        )
    tensor = tensor.detach()
    if bool((~torch.isfinite(tensor)).any().item()):
        raise F2AssemblyContractError(f"{label} {level} token is nonfinite")
    return tensor


def _resolve_image(
    base_root: Path, value: Any, label: str
) -> tuple[Path, Path]:
    if not isinstance(value, str) or not value:
        raise F2AssemblyContractError(f"{label} must be a nonempty image path")
    portable = PurePosixPath(value)
    if (
        "\\" in value
        or portable.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or portable.as_posix() != value
        or any(part in (".", "..") for part in portable.parts)
    ):
        raise F2AssemblyContractError(
            f"{label} escapes the frozen image base root: {value}"
        )
    candidate = Path(*portable.parts)
    absolute = (base_root / candidate).resolve()
    try:
        relative = absolute.relative_to(base_root)
    except ValueError as exc:
        raise F2AssemblyContractError(
            f"{label} escapes the frozen image base root: {absolute}"
        ) from exc
    return absolute, relative


def load_cached_observation(
    row: Mapping[str, Any],
    *,
    base_root: str | Path,
    cache_root: str | Path,
    token_ledger: TokenHashLedger,
    history: int = HISTORY_FRAMES,
) -> ObservationPacket:
    """Load one row's observation strictly from the frozen vision cache.

    Mirrors the vendored ``JsonTrackingDataset.__getitem__`` token path
    (current fine token, ``history`` coarse frames trimmed/padded exactly
    the same way) with every online-recompute branch and the
    ``torch.zeros`` fallback deleted: any missing or invalid token raises
    :class:`F2AssemblyContractError` immediately.  Every token file's
    bytes are verified against the frozen ``token_ledger`` before
    deserialization; unlisted files are never opened.
    """

    mapping = _require_mapping(row, "observation row")
    ledger = _require_ledger(token_ledger)
    if isinstance(history, bool) or not isinstance(history, Integral) or history < 1:
        raise F2AssemblyContractError("history must be a positive integer")
    history = int(history)
    base = Path(base_root).expanduser().resolve()
    cache = Path(cache_root).expanduser().resolve()
    for name, directory in (("base_root", base), ("cache_root", cache)):
        if not directory.is_dir():
            raise F2AssemblyContractError(f"{name} does not exist: {directory}")

    current_value = _require_str(mapping, "current", "observation row")
    _abs_current, rel_current = _resolve_image(base, current_value, "current")
    current_dir = cache / rel_current.parent
    fine_tokens = _load_cached_token(
        _token_candidates(current_dir, rel_current, "fine"),
        level="fine",
        label="current frame",
        cache_root=cache,
        ledger=ledger,
    )
    fine_tidx = torch.full((FINE_TOKEN_COUNT,), history, dtype=torch.long)

    images = mapping.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise F2AssemblyContractError("observation row has invalid 'images'")
    trimmed = list(images[-history:])
    missing = history - len(trimmed)
    first_token: torch.Tensor | None = None
    current_coarse: torch.Tensor | None = None
    coarse_parts: list[torch.Tensor] = []
    coarse_tidx_parts: list[torch.Tensor] = []
    for frame in range(history):
        token: torch.Tensor | None = None
        if frame >= missing:
            _abs_image, rel_image = _resolve_image(
                base, trimmed[frame - missing], f"images[{frame - missing}]"
            )
            token = _load_cached_token(
                _token_candidates(cache / rel_image.parent, rel_image, "coarse"),
                level="coarse",
                label=f"history frame {frame}",
                cache_root=cache,
                ledger=ledger,
            )
            if first_token is None:
                first_token = token
        if token is None:
            if first_token is not None:
                token = first_token
            else:
                if current_coarse is None:
                    current_coarse = _load_cached_token(
                        _token_candidates(current_dir, rel_current, "coarse"),
                        level="coarse",
                        label="current frame",
                        cache_root=cache,
                        ledger=ledger,
                    )
                token = current_coarse
        coarse_parts.append(token)
        coarse_tidx_parts.append(
            torch.full((token.shape[0],), frame, dtype=torch.long)
        )
    coarse_tokens = torch.cat(coarse_parts, dim=0)
    coarse_tidx = torch.cat(coarse_tidx_parts, dim=0)

    fields: dict[str, Any] = {
        "coarse_tokens": coarse_tokens,
        "coarse_tidx": coarse_tidx,
        "fine_tokens": fine_tokens,
        "fine_tidx": fine_tidx,
        "instruction": _require_str(mapping, "instruction", "observation row"),
    }
    if "yaw_hist" in mapping:
        yaw_values = mapping["yaw_hist"]
        if not isinstance(yaw_values, Sequence) or isinstance(
            yaw_values, (str, bytes)
        ):
            raise F2AssemblyContractError("observation row has invalid 'yaw_hist'")
        if len(yaw_values) != history:
            raise F2AssemblyContractError(
                f"yaw_hist must have exactly {history} entries"
            )
        fields["yaw_hist"] = torch.tensor(
            [_finite_float(item, "yaw_hist entry") for item in yaw_values],
            dtype=torch.float32,
        )
    if "yaw_curr" in mapping:
        fields["yaw_curr"] = torch.tensor(
            [_finite_float(mapping["yaw_curr"], "yaw_curr")], dtype=torch.float32
        )
    return observation_packet_from_fields(fields)


# --------------------------------------------------------------------------
# Blocker 9: block-major allowlisted JSONL -> RunnerRow loader.
# --------------------------------------------------------------------------


def _support_components(
    receipt: FrozenSupportReceipt, support_name: str
) -> tuple[tuple[Any, ...], tuple[int, ...], str, Any]:
    if not isinstance(receipt, FrozenSupportReceipt):
        raise F2AssemblyContractError("receipt must be a FrozenSupportReceipt")
    if support_name not in SUPPORT_NAMES:
        raise F2AssemblyContractError(
            f"unknown support name {support_name!r}; expected one of "
            f"{SUPPORT_NAMES!r}"
        )
    try:
        blocks = tuple(receipt.supports[support_name])
        indices = tuple(receipt.row_indices[support_name])
        row_sha = receipt.row_sha256[support_name]
        coverage = receipt.coverage[support_name]
    except KeyError as exc:
        raise F2AssemblyContractError(
            f"support receipt is missing {support_name!r}"
        ) from exc
    return blocks, indices, row_sha, coverage


def _is_frozen_train(receipt: FrozenSupportReceipt) -> bool:
    return receipt.train_sha256 == FROZEN_TRAIN_SHA256


def ordered_support_rows(
    rows: Sequence[Mapping[str, Any]],
    receipt: FrozenSupportReceipt,
    support_name: str,
) -> tuple[tuple[int, Mapping[str, Any]], ...]:
    """Return the frozen block-major ``(original_index, row)`` sequence.

    The block order follows ``cli._ordered_rows`` semantics; the row
    identity is validated against the receipt's frozen row SHA, and, when
    the receipt binds the frozen train, additionally against
    ``SUPPORT_EXPECTATIONS``.  The order must be strictly increasing in
    ``original_row_index`` (live-verified for all three supports).
    """

    blocks, indices, row_sha, coverage = _support_components(
        receipt, support_name
    )
    ordered = [index for block in blocks for index in block.row_indices]
    if len(ordered) != len(set(ordered)):
        raise F2AssemblyContractError(
            f"{support_name} ordered rows are not unique"
        )
    if set(ordered) != set(indices):
        raise F2AssemblyContractError(
            f"{support_name} ordered rows differ from the frozen receipt"
        )
    actual_sha = canonical_json_sha256(sorted(ordered))
    if actual_sha != row_sha:
        raise F2AssemblyContractError(
            f"{support_name} row SHA does not match the frozen receipt"
        )
    if _is_frozen_train(receipt):
        expectation = SUPPORT_EXPECTATIONS[support_name]
        if actual_sha != expectation.sha256 or len(ordered) != expectation.rows:
            raise F2AssemblyContractError(
                f"{support_name} identity differs from the frozen "
                "SUPPORT_EXPECTATIONS for the frozen train"
            )
    if coverage.rows != len(ordered):
        raise F2AssemblyContractError(
            f"{support_name} coverage rows differ from the ordered rows"
        )
    if any(
        right <= left for left, right in zip(ordered, ordered[1:])
    ):
        raise F2AssemblyContractError(
            f"{support_name} block-major order is not strictly increasing"
        )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise F2AssemblyContractError("rows must be an ordered sequence")
    if len(rows) != receipt.train_rows:
        raise F2AssemblyContractError(
            "rows length differs from the receipt train row count"
        )
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for index in ordered:
        if not 0 <= index < len(rows):
            raise F2AssemblyContractError(
                f"{support_name} row index {index} is out of range"
            )
        selected.append((index, _require_mapping(rows[index], f"row {index}")))
    return tuple(selected)


def smoke_reset_sets(
    receipt: FrozenSupportReceipt, support_name: str
) -> tuple[frozenset[int], frozenset[int]]:
    """Return ``(strafe_reset_original_indices, expected_static_resets)``.

    Both sets are derived from the frozen receipt: STRAFE resets are the
    intersection of ``strafe.reset_boundary_12`` with the support rows
    (empty for SMK-TRAIN), and the expected static resets are the block
    first rows plus the base sequence resets inside the support, exactly
    the frozen ``static_resets`` accounting (12/28/30).
    """

    blocks, indices, _row_sha, coverage = _support_components(
        receipt, support_name
    )
    support_rows = frozenset(indices)
    block_firsts = frozenset(block.first_row_index for block in blocks)
    if not block_firsts <= support_rows:
        raise F2AssemblyContractError(
            f"{support_name} block first rows are not support rows"
        )
    strafe_in_support = frozenset(receipt.strafe.reset_boundary_12) & support_rows
    if not strafe_in_support <= block_firsts:
        raise F2AssemblyContractError(
            "HS4_support_strafe_intersection: STRAFE reset rows inside "
            f"{support_name} are not block-first rows"
        )
    expected = frozenset(
        (support_rows & frozenset(receipt.base_reset_rows)) | block_firsts
    )
    if len(expected) != coverage.static_resets:
        raise F2AssemblyContractError(
            f"{support_name} static reset count {len(expected)} differs from "
            f"the frozen coverage {coverage.static_resets}"
        )
    if _is_frozen_train(receipt):
        expectation = SUPPORT_EXPECTATIONS[support_name]
        if len(expected) != expectation.static_resets:
            raise F2AssemblyContractError(
                f"{support_name} static reset count differs from the frozen "
                "SUPPORT_EXPECTATIONS"
            )
    return strafe_in_support, expected


def support_reset_plan(
    rows: Sequence[Mapping[str, Any]],
    receipt: FrozenSupportReceipt,
    support_name: str,
) -> tuple[tuple[str, ...], ...]:
    """Replay the exact runner reset predicates over the block-major order.

    Reasons per position mirror ``runner._build_reset_plan``:
    ``stream_first`` at position 0, ``sequence_discontinuity`` whenever
    ``continues_sequence`` fails, and ``strafe_reset`` on frozen STRAFE
    boundary rows.  The derived reset set must equal the receipt-derived
    expectation from :func:`smoke_reset_sets`; any divergence fails closed.
    """

    ordered = ordered_support_rows(rows, receipt, support_name)
    strafe_in_support, expected = smoke_reset_sets(receipt, support_name)
    plan: list[tuple[str, ...]] = []
    previous: Mapping[str, Any] | None = None
    for index, row in ordered:
        reasons: list[str] = []
        if previous is None:
            reasons.append("stream_first")
        elif not continues_sequence(previous, row):
            reasons.append("sequence_discontinuity")
        if index in strafe_in_support:
            reasons.append("strafe_reset")
        plan.append(tuple(reasons))
        previous = row
    observed = frozenset(
        index for (index, _row), reasons in zip(ordered, plan) if reasons
    )
    if observed != expected:
        raise F2AssemblyContractError(
            f"RESET_PLAN_MISALIGNMENT: {support_name} runner reset predicates "
            "disagree with the frozen receipt expectation"
        )
    return tuple(plan)


def _target_actions_tensor(row: Mapping[str, Any], label: str) -> torch.Tensor:
    actions = row.get("step_actions")
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        raise F2AssemblyContractError(f"{label} has invalid step_actions")
    if len(actions) != AP2_HORIZON:
        raise F2AssemblyContractError(
            f"{label} step_actions must contain exactly {AP2_HORIZON} steps"
        )
    values: list[list[float]] = []
    for step, action in enumerate(actions):
        if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
            raise F2AssemblyContractError(
                f"{label}.step_actions[{step}] must be an action vector"
            )
        if len(action) < 3:
            raise F2AssemblyContractError(
                f"{label}.step_actions[{step}] must contain three axes"
            )
        axes = [
            _finite_float(action[axis], f"{label}.step_actions[{step}][{axis}]")
            for axis in range(3)
        ]
        if abs(axes[0]) > ACTION_MAX_ABS or abs(axes[2]) > ACTION_MAX_ABS:
            raise F2AssemblyContractError(
                f"{label}.step_actions[{step}] controlled axes lie outside "
                "the frozen action domain"
            )
        values.append(axes)
    return torch.tensor(values, dtype=torch.float32)


def _prev_action_tuple(
    row: Mapping[str, Any], label: str
) -> tuple[float, float, float]:
    previous = row.get("prev_action")
    if not isinstance(previous, Sequence) or isinstance(previous, (str, bytes)):
        raise F2AssemblyContractError(f"{label} has invalid prev_action")
    if len(previous) < 3:
        raise F2AssemblyContractError(
            f"{label} prev_action must contain three axes"
        )
    return (
        _finite_float(previous[0], f"{label}.prev_action[0]"),
        _finite_float(previous[1], f"{label}.prev_action[1]"),
        _finite_float(previous[2], f"{label}.prev_action[2]"),
    )


def build_runner_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    receipt: FrozenSupportReceipt,
    support_name: str,
    base_root: str | Path,
    cache_root: str | Path,
    token_ledger: TokenHashLedger,
) -> tuple[RunnerRow, ...]:
    """Build the frozen block-major :class:`RunnerRow` sequence for a support.

    Observations are loaded strictly from the frozen cache into
    :class:`ObservationPacket` (allowlist enforced, token bytes verified
    against the frozen ``token_ledger``), auxiliary labels into
    :class:`AuxTargetPacket`, and expert ``step_actions`` only into
    ``target_actions``.  The reset predicates are replayed and checked
    against the frozen expectation before any cache I/O happens.
    """

    ordered = ordered_support_rows(rows, receipt, support_name)
    support_reset_plan(rows, receipt, support_name)
    ledger = _require_ledger(token_ledger)
    runner_rows: list[RunnerRow] = []
    for index, row in ordered:
        label = f"{support_name} row {index}"
        observation = load_cached_observation(
            row,
            base_root=base_root,
            cache_root=cache_root,
            token_ledger=ledger,
        )
        aux_targets = aux_target_packet_from_row(row, label)
        runner_rows.append(
            RunnerRow(
                original_row_index=index,
                sequence_id=_require_str(row, "sequence_id", label),
                frame_idx=_require_int(row, "frame_idx", label),
                mirrored=_require_bool(row, "mirrored", label),
                logged_prev_action=_prev_action_tuple(row, label),
                target_actions=_target_actions_tensor(row, label),
                observation=observation,
                aux_targets=aux_targets,
            )
        )
    if len(runner_rows) != len(ordered):
        raise F2AssemblyContractError(
            f"{support_name} runner row count drifted during construction"
        )
    return tuple(runner_rows)


def eval_fix_strata(
    rows: Sequence[Mapping[str, Any]],
    receipt: FrozenSupportReceipt,
    support_name: str = "EVAL-FIX",
) -> dict[str, Any]:
    """Frozen overall/change/turn/other stratum labels in block-major order.

    ``change`` uses the corrigendum-2 first-step controlled-axis threshold
    and ``turn``/``other`` come from ``transition_type`` via the exact
    ``evaluation.strata_masks_from_rows`` contract; the resulting counts
    must match the frozen coverage receipt (EVAL-FIX: 69/154/211 on 512).
    """

    ordered = ordered_support_rows(rows, receipt, support_name)
    _blocks, _indices, _row_sha, coverage = _support_components(
        receipt, support_name
    )
    ordered_rows = [row for _index, row in ordered]
    masks = strata_masks_from_rows(ordered_rows)
    counts = {
        "overall": len(ordered_rows),
        "change": int(sum(masks["change"])),
        "turn": int(sum(masks["turn"])),
        "other": int(sum(masks["other"])),
    }
    expected_counts = {
        "overall": coverage.rows,
        "change": coverage.h1_change,
        "turn": coverage.turn,
        "other": coverage.other,
    }
    if counts != expected_counts:
        raise F2AssemblyContractError(
            f"{support_name} stratum counts {counts} differ from the frozen "
            f"coverage {expected_counts}"
        )
    if _is_frozen_train(receipt):
        expectation = SUPPORT_EXPECTATIONS[support_name]
        frozen_counts = {
            "overall": expectation.rows,
            "change": expectation.h1_change,
            "turn": expectation.turn,
            "other": expectation.other,
        }
        if counts != frozen_counts:
            raise F2AssemblyContractError(
                f"{support_name} stratum counts differ from the frozen "
                "SUPPORT_EXPECTATIONS"
            )
    return {
        "support": support_name,
        "strata": ("overall", "change", "turn", "other"),
        "row_order": tuple(index for index, _row in ordered),
        "masks": {name: tuple(mask) for name, mask in masks.items()},
        "counts": counts,
    }


# --------------------------------------------------------------------------
# Blocker 6: frozen checkpoint / cache / encoder / prompt bindings.
# --------------------------------------------------------------------------


def frozen_cache_roots(project_root: str | Path) -> tuple[Path, Path]:
    """Return ``(image_base_root, cache_root)`` for the frozen vision cache.

    The cache manifest's ``path_root`` must resolve to the project root;
    a drifted cache identity fails closed instead of silently resolving
    tokens against the wrong tree.
    """

    root = Path(project_root).expanduser().resolve()
    cache_root = (root / FROZEN_CACHE_ROOT_RELATIVE).resolve()
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise F2AssemblyContractError(
            f"vision cache manifest is missing: {manifest_path}"
        )
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise F2AssemblyContractError(
            f"vision cache manifest is unreadable: {manifest_path}"
        ) from exc
    path_root_value = manifest.get("path_root")
    if not isinstance(path_root_value, str) or not path_root_value:
        raise F2AssemblyContractError("vision cache manifest has no path_root")
    base_root = Path(path_root_value).expanduser().resolve()
    if base_root == root:
        return base_root, cache_root
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if (
        manifest_sha == FROZEN_CACHE_MANIFEST_SHA256
        and path_root_value == FROZEN_CACHE_RECORDED_PROJECT_ROOT
    ):
        # The manifest is a frozen content/provenance artifact whose producer
        # path was absolute.  Preserve its bytes and SHA, but explicitly map
        # that recorded root to this verified project root.  The returned asset
        # binding records both roots; arbitrary manifest drift still fails.
        return root, cache_root
    if base_root != root:
        raise F2AssemblyContractError(
            f"vision cache path_root {base_root} differs from the project "
            f"root {root}"
        )
    return root, cache_root


def resolve_frozen_base_hf_dir(
    base_hf_dir: str | Path | None = None,
) -> Path:
    """Resolve the official base checkpoint without a platform hard-code.

    An explicit argument is authoritative, followed by ``F2_BASE_HF_DIR``.
    With neither set, use the sibling asset layout shipped with this Windows
    workspace and then the vendored checkpoint layout.  An explicitly chosen
    missing path is returned so the artifact binder emits the precise error.
    """

    if base_hf_dir is not None:
        return Path(base_hf_dir).expanduser().resolve()
    env_value = os.getenv("F2_BASE_HF_DIR", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = (
        FROZEN_BASE_HF_DIR_DEFAULT,
        _PROJECT_ROOT
        / "third_party"
        / "OpenTrackVLA"
        / "ckpts_hf"
        / "opentrackvla-qwen06b",
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return resolved
    return FROZEN_BASE_HF_DIR_DEFAULT


def verify_frozen_assets(
    project_root: str | Path,
    *,
    base_hf_dir: str | Path | None = None,
    verify_token_payload: bool = True,
) -> dict[str, Any]:
    """Verify every frozen data-side asset SHA and return the binding doc.

    Checks, failing closed on the first mismatch: frozen train JSONL SHA,
    vision cache manifest/provenance (and optionally the full 52,820-token
    payload), DINOv3/SigLIP encoder provenance SHAs, the official base HF
    checkpoint artifact SHA, the local Qwen weights artifact SHA, and the
    prompt normalization erratum SHA.  The returned document is the
    data-side section of assembly receipt v4.
    """

    root = Path(project_root).expanduser().resolve()
    train_path = (root / FROZEN_TRAIN_RELATIVE).resolve()
    if not train_path.is_file():
        raise F2AssemblyContractError(
            f"frozen train JSONL is missing: {train_path}"
        )
    train_sha = sha256_file(train_path)
    if train_sha != FROZEN_TRAIN_SHA256:
        raise F2AssemblyContractError(
            "HS1_train_precondition: frozen train JSONL SHA mismatch"
        )

    base_root, cache_root = frozen_cache_roots(root)
    try:
        cache_binding = verify_vision_cache(
            cache_root,
            [str(train_path)],
            verify_payload=bool(verify_token_payload),
            relocated_root=root,
        )
    except ExperimentBindingError as exc:
        raise F2AssemblyContractError(
            f"vision cache verification failed: {exc}"
        ) from exc
    frozen_cache_values = {
        "cache_manifest_sha256": FROZEN_CACHE_MANIFEST_SHA256,
        "cache_provenance_sha256": FROZEN_CACHE_PROVENANCE_SHA256,
        "token_payload_sha256": FROZEN_TOKEN_PAYLOAD_SHA256,
        "dino_model_sha256": FROZEN_DINO_SHA256,
        "siglip_model_sha256": FROZEN_SIGLIP_SHA256,
    }
    for field, frozen_value in frozen_cache_values.items():
        if cache_binding.get(field) != frozen_value:
            raise F2AssemblyContractError(
                f"vision cache {field} differs from the frozen value"
            )

    hf_dir = resolve_frozen_base_hf_dir(base_hf_dir)
    try:
        hf_binding = bind_hf_model_artifact(hf_dir)
    except ExperimentBindingError as exc:
        raise F2AssemblyContractError(
            f"base HF checkpoint binding failed: {exc}"
        ) from exc
    if hf_binding.get("artifact_sha256") != FROZEN_BASE_HF_ARTIFACT_SHA256:
        raise F2AssemblyContractError(
            "base HF checkpoint artifact SHA differs from the frozen value"
        )

    try:
        qwen_path = resolve_local_model_path(
            label=FROZEN_QWEN_REPO_ID,
            repo_id=FROZEN_QWEN_REPO_ID,
            explicit=None,
            env_var="QWEN_MODEL_PATH",
            candidates=default_qwen_candidates(),
        )
    except FileNotFoundError as exc:
        raise F2AssemblyContractError(
            f"local Qwen weights were not found: {exc}"
        ) from exc
    try:
        qwen_sha = sha256_artifact(qwen_path)
    except ExperimentBindingError as exc:
        raise F2AssemblyContractError(
            f"local Qwen artifact hashing failed: {exc}"
        ) from exc
    if qwen_sha != FROZEN_QWEN_SHA256:
        raise F2AssemblyContractError(
            "local Qwen artifact SHA differs from the frozen value"
        )

    erratum_path = (root / FROZEN_PROMPT_ERRATUM_RELATIVE).resolve()
    if not erratum_path.is_file():
        raise F2AssemblyContractError(
            f"prompt normalization erratum is missing: {erratum_path}"
        )
    erratum_sha = sha256_file(erratum_path)
    if erratum_sha != FROZEN_PROMPT_ERRATUM_SHA256:
        raise F2AssemblyContractError(
            "prompt normalization erratum SHA differs from the frozen value"
        )

    return {
        "schema_version": 1,
        "analysis_class": "f2_assembly_frozen_asset_binding",
        "project_root": str(root),
        "train": {
            "relative_path": FROZEN_TRAIN_RELATIVE.as_posix(),
            "rows": FROZEN_TRAIN_ROWS,
            "sha256": train_sha,
        },
        "vision_cache": {
            "cache_root": str(cache_root),
            "image_base_root": str(base_root),
            "cache_manifest_sha256": cache_binding["cache_manifest_sha256"],
            "cache_provenance_sha256": cache_binding["cache_provenance_sha256"],
            "token_payload_sha256": cache_binding["token_payload_sha256"],
            "token_payload_verified": bool(verify_token_payload),
            "dino_model_sha256": cache_binding["dino_model_sha256"],
            "siglip_model_sha256": cache_binding["siglip_model_sha256"],
            "recorded_path_root": cache_binding["recorded_path_root"],
            "effective_path_root": cache_binding["effective_path_root"],
            "path_relocated": cache_binding["path_relocated"],
            "metadata_only": True,
        },
        "base_hf": {
            "path": str(hf_dir),
            **hf_binding,
        },
        "qwen": {
            "path": str(qwen_path),
            "artifact_sha256": qwen_sha,
        },
        "prompt_erratum": {
            "relative_path": FROZEN_PROMPT_ERRATUM_RELATIVE.as_posix(),
            "sha256": erratum_sha,
        },
        "internal_test_opened": False,
    }


__all__ = [
    "AUX_FUT_KEYS",
    "AUX_FUTURE_HORIZONS",
    "COARSE_TOKEN_COUNT",
    "FINE_TOKEN_COUNT",
    "FROZEN_BASE_HF_ARTIFACT_SHA256",
    "FROZEN_BASE_HF_DIR_DEFAULT",
    "FROZEN_CACHE_MANIFEST_SHA256",
    "FROZEN_CACHE_PROVENANCE_SHA256",
    "FROZEN_CACHE_RECORDED_PROJECT_ROOT",
    "FROZEN_CACHE_ROOT_RELATIVE",
    "FROZEN_DINO_SHA256",
    "FROZEN_PROMPT_ERRATUM_RELATIVE",
    "FROZEN_PROMPT_ERRATUM_SHA256",
    "FROZEN_QWEN_REPO_ID",
    "FROZEN_QWEN_SHA256",
    "FROZEN_SIGLIP_SHA256",
    "FROZEN_TOKEN_PAYLOAD_SHA256",
    "FROZEN_INTERNAL_TEST_IMAGE_PREFIX",
    "FROZEN_TRAIN_IMAGE_PREFIXES",
    "HISTORY_FRAMES",
    "OBSERVATION_ALLOWED_KEYS",
    "OBSERVATION_FORBIDDEN_KEYS",
    "OBSERVATION_OPTIONAL_KEYS",
    "POLAR_DIST_BINS",
    "POLAR_THETA_BINS",
    "SUPPORT_NAMES",
    "VISION_FEATURE_DIM",
    "AuxTargetPacket",
    "F2AssemblyContractError",
    "ObservationPacket",
    "TokenHashLedger",
    "aux_target_packet_from_row",
    "build_runner_rows",
    "build_token_ledger_for_rows",
    "build_train_token_ledger",
    "collect_image_relpaths",
    "ensure_observation_packet",
    "eval_fix_strata",
    "frozen_cache_roots",
    "load_cached_observation",
    "observation_packet_from_fields",
    "ordered_support_rows",
    "smoke_reset_sets",
    "support_reset_plan",
    "verify_frozen_assets",
    "resolve_frozen_base_hf_dir",
]
