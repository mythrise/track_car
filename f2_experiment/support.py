"""Frozen support construction for the isolated F2 causal smoke.

The implementation follows the Fable 5 architecture decision, implementation
corrigendum-2, and support-registry corrigendum-3.  It intentionally rebuilds
all support identities from the frozen training JSONL and fails closed on any
contract mismatch.  No validation or internal-test artifact is read here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARCHITECTURE_LOCK = "L1+D2+AP2+F2"
CANDIDATE_CAP = 1
FORMAL_RUNS = 9
FORMAL_UPDATES_PER_ARM = 6_873
SMOKE_UPDATES = 128
INTERNAL_TEST_POLICY = "sealed"

CONTRACT_COMPOSITION = {
    "base": "Fable primary L1+D2+AP2+F2 architecture",
    "corrigendum2": "apply every non-support_registry key",
    "corrigendum3": "replace support_registry in full",
    "void_corrigendum2_support_fields": ["is_mirrored", "expected_pool_n=236"],
}
HARD_STOPS = (
    "HS1_train_precondition",
    "HS2_reset_receipt",
    "HS3_support_build",
    "HS4_support_strafe_intersection",
    "HS5_strafe_fix_receipt_or_legacy_load",
    "HS6_step0_parity",
    "HS7_fusion_runtime",
    "HS8_g7_gates",
    "HS9_g6_gates",
    "HS10_controller_nonfinite_or_sha",
    "HS11_estimand_mislabel",
    "HS12_out_of_scope_diff",
)
AUTHORIZATION = {
    "granted": ["implement_patch", "build_receipts", "smoke_128_updates"],
    "formal_training": "gated_on_smoke_G6_G7_and_all_receipts",
}

FROZEN_TRAIN_RELATIVE = Path("data/collected_v1/datasets/train.jsonl")
FROZEN_TRAIN_ROWS = 13_746
FROZEN_TRAIN_SHA256 = (
    "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36"
)

ROWS_PER_BLOCK = 32
SELECTION_SLOTS = 40
SELECTION_N = 173
FORBIDDEN_SELECTION_N = frozenset({110, 236})
TURN_TRANSITION_TYPES = frozenset(
    {"turn_onset", "sustained_turn", "turn_exit"}
)
CHANGE_THRESHOLD = 0.2
EPS_STRAFE = 1e-6

BASE_RESET_COUNT = 290
BASE_RESET_SHA256 = (
    "8f3a49c1d9744b85fece88f604d3a90ee361a1bc8239f3a1fb05cb6aa7579086"
)
STRAFE_RESET_COUNT = 12
STRAFE_RESET_SHA256 = (
    "934d6e02b34e0197a0375b2a0f39b7a5f1bd79812fd7471066fdcc627b17c0ea"
)
COMBINED_RESET_COUNT = 302
COMBINED_RESET_SHA256 = (
    "eb80e8db40eb0d894f239e28193b6ce469afcd5f954e0d21929e2eaa21adeb94"
)
BUGGY_HELPER_RESET_COUNT = 4_175
BUGGY_HELPER_RESET_SHA256 = (
    "2d481ce6e6ac5b279ceaef96aada3df903c88cd8698571f38dd3042df332da09"
)


class F2ContractError(RuntimeError):
    """Raised whenever an approved F2 hard-stop condition is encountered."""


@dataclass(frozen=True)
class ApprovalBinding:
    binding_id: str
    relative_path: Path
    sha256: str
    role: str


APPROVAL_BINDINGS = (
    ApprovalBinding(
        binding_id="fable_architecture_primary_result",
        relative_path=Path(
            "experiments/collected_v1_main/external_reviews/"
            "20260717_fable5_harness_postprobe_architecture_corrigendum_raw.json"
        ),
        sha256="54300a1552605dc2c5122643e626a48d67b1235400f2b80ce5655801c22e994c",
        role="primary L1+D2+AP2+F2 architecture decision",
    ),
    ApprovalBinding(
        binding_id="fable_implementation_primary_result",
        relative_path=Path(
            "experiments/collected_v1_main/external_reviews/"
            "20260717_fable5_harness_postprobe_implementation_corrigendum_raw.json"
        ),
        sha256="e16be3d989940d13e8e8a4958a1c02fedfd98552b2d66226d555b9fe19c69453",
        role="primary implementation corrigendum",
    ),
    ApprovalBinding(
        binding_id="fable_corrigendum2_registry_patch",
        relative_path=Path(
            "experiments/collected_v1_main/external_reviews/"
            "20260717_fable5_harness_f2_implementation_corrigendum2_registry_patch.json"
        ),
        sha256="9732315640d04d28b5a82c79c6c724c076ff1e5bee7765c60c91175934dd0ba8",
        role="corrigendum-2 machine-readable registry patch",
    ),
    ApprovalBinding(
        binding_id="fable_corrigendum2_verdict",
        relative_path=Path(
            "experiments/collected_v1_main/external_reviews/"
            "20260717_fable5_harness_f2_implementation_corrigendum2_verdict.md"
        ),
        sha256="f19c7c888b35cc2b6ca1b7b0ca9e518b2646a972eed4a3bfc8bc4d8760f7c2d5",
        role="corrigendum-2 primary GO verdict",
    ),
    ApprovalBinding(
        binding_id="fable_corrigendum3_support_patch",
        relative_path=Path(
            "experiments/collected_v1_main/external_reviews/"
            "20260717_fable5_harness_f2_support_registry_corrigendum3_raw.md"
        ),
        sha256="77346c8c942ee3a77d96d2ed6078304c9b7e753d1cb412eecd91237a5bbd03c4",
        role="corrigendum-3 support-registry replacement",
    ),
    ApprovalBinding(
        binding_id="fable_corrigendum3_verdict",
        relative_path=Path(
            "experiments/collected_v1_main/external_reviews/"
            "20260717_fable5_harness_f2_support_registry_corrigendum3_verdict.md"
        ),
        sha256="4cf1ef8b668a91430029d69964aa497fb3e6029531cb1cac81878424420bd4aa",
        role="corrigendum-3 primary GO verdict",
    ),
)


@dataclass(frozen=True)
class SupportExpectation:
    blocks: int
    rows: int
    h1_change: int
    turn: int
    other: int
    unique_sequence: int
    episode: int
    mirror: int
    static_resets: int
    sha256: str


SUPPORT_EXPECTATIONS: Mapping[str, SupportExpectation] = {
    "CAL": SupportExpectation(
        blocks=16,
        rows=512,
        h1_change=90,
        turn=134,
        other=247,
        unique_sequence=16,
        episode=10,
        mirror=7,
        static_resets=30,
        sha256="ab719e31595a17e498d221f5459ce30467eedea39c22e17d4a6e1080d7b3c6f1",
    ),
    "SMK-TRAIN": SupportExpectation(
        blocks=8,
        rows=256,
        h1_change=28,
        turn=50,
        other=73,
        unique_sequence=8,
        episode=8,
        mirror=2,
        static_resets=12,
        sha256="7073a02c866913903a67438673ec7cf6898574bd7fa9bac891ad6142b563818f",
    ),
    "EVAL-FIX": SupportExpectation(
        blocks=16,
        rows=512,
        h1_change=69,
        turn=154,
        other=211,
        unique_sequence=16,
        episode=10,
        mirror=7,
        static_resets=28,
        sha256="5123a14dc526dfcef96e73ee838e33b265dee0bff0efe66e36e806540e1922ec",
    ),
}
UNION_ROWS = 1_280
UNION_SHA256 = "906f990a34ed9bcc6c852f7295293b467628ab56c465d41450d4ae9715aa19be"

SUPPORT_ASSIGNMENT: Mapping[int, str] = {
    0: "CAL",
    1: "EVAL-FIX",
    2: "SMK-TRAIN",
    3: "CAL",
    4: "EVAL-FIX",
}


@dataclass(frozen=True)
class SupportBlock:
    key: tuple[str, str, str]
    source_raw_dir: str
    sequence_id: str
    clip_id: str
    mirrored: bool
    row_indices: tuple[int, ...]

    @property
    def first_row_index(self) -> int:
        return self.row_indices[0]


@dataclass(frozen=True)
class SupportCoverage:
    blocks: int
    rows: int
    h1_change: int
    turn: int
    other: int
    unique_sequence: int
    episode: int
    mirror: int
    static_resets: int

    def to_dict(self) -> dict[str, int]:
        return {
            "blocks": self.blocks,
            "rows": self.rows,
            "h1_change": self.h1_change,
            "turn": self.turn,
            "other": self.other,
            "unique_sequence": self.unique_sequence,
            "episode": self.episode,
            "mirror": self.mirror,
            "static_resets": self.static_resets,
        }


@dataclass(frozen=True)
class StrafeLedger:
    audit_24: tuple[int, ...]
    unsupported_10: tuple[int, ...]
    reset_boundary_12: tuple[int, ...]
    diagnostic_superset_26: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_24": _row_array_receipt(self.audit_24),
            "unsupported_10": _row_array_receipt(self.unsupported_10),
            "reset_boundary_12": _row_array_receipt(self.reset_boundary_12),
            "diagnostic_superset_26": _row_array_receipt(
                self.diagnostic_superset_26
            ),
        }


@dataclass(frozen=True)
class FrozenSupportReceipt:
    train_sha256: str
    train_rows: int
    eligible_pool_total: int
    eligible_pool_nonmirrored: int
    eligible_pool_mirrored: int
    supports: Mapping[str, tuple[SupportBlock, ...]]
    row_indices: Mapping[str, tuple[int, ...]]
    row_sha256: Mapping[str, str]
    coverage: Mapping[str, SupportCoverage]
    union_row_indices: tuple[int, ...]
    union_sha256: str
    base_reset_rows: tuple[int, ...]
    combined_reset_rows: tuple[int, ...]
    strafe: StrafeLedger

    def to_dict(self) -> dict[str, Any]:
        support_documents: dict[str, Any] = {}
        for name in ("CAL", "SMK-TRAIN", "EVAL-FIX"):
            support_documents[name] = {
                "blocks": [
                    {
                        "key": list(block.key),
                        "mirrored": block.mirrored,
                        "row_indices": list(block.row_indices),
                    }
                    for block in self.supports[name]
                ],
                "row_indices": list(self.row_indices[name]),
                "row_sha256": self.row_sha256[name],
                "coverage": self.coverage[name].to_dict(),
            }
        return {
            "schema_version": 1,
            "analysis_class": "f2_versioned_calibration_support",
            "architecture_lock": ARCHITECTURE_LOCK,
            "contract_composition": CONTRACT_COMPOSITION,
            "support_registry": "corrigendum3",
            "train": {"rows": self.train_rows, "sha256": self.train_sha256},
            "eligible_pool": {
                "total": self.eligible_pool_total,
                "nonmirrored": self.eligible_pool_nonmirrored,
                "mirrored": self.eligible_pool_mirrored,
            },
            "supports": support_documents,
            "union": {
                "row_indices": list(self.union_row_indices),
                "rows": len(self.union_row_indices),
                "sha256": self.union_sha256,
                "pairwise_disjoint": True,
            },
            "recurrence_reset": {
                "base": _row_array_receipt(self.base_reset_rows),
                "combined": _row_array_receipt(self.combined_reset_rows),
            },
            "strafe_fix": self.strafe.to_dict(),
            "candidate_cap": CANDIDATE_CAP,
            "formal_runs": FORMAL_RUNS,
            "formal_updates_per_arm": FORMAL_UPDATES_PER_ARM,
            "internal_test": INTERNAL_TEST_POLICY,
            "hard_stops": list(HARD_STOPS),
            "authorization": AUTHORIZATION,
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise F2ContractError("value is not canonical-JSON serializable") from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _row_array_receipt(indices: Iterable[int]) -> dict[str, Any]:
    rows = tuple(sorted(int(index) for index in indices))
    return {
        "count": len(rows),
        "sha256": canonical_json_sha256(list(rows)),
        "rows": list(rows),
    }


def verify_approval_files(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).expanduser().resolve()
    verified: dict[str, str] = {}
    for binding in APPROVAL_BINDINGS:
        path = (root / binding.relative_path).resolve()
        if not path.is_file():
            raise F2ContractError(f"missing Fable approval artifact: {path}")
        actual = sha256_bytes(path.read_bytes())
        if actual != binding.sha256:
            raise F2ContractError(
                f"Fable approval artifact SHA mismatch: {binding.binding_id}"
            )
        verified[binding.binding_id] = actual
    return verified


def parse_train_jsonl(payload: bytes) -> tuple[Mapping[str, Any], ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise F2ContractError("frozen train JSONL is not UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise F2ContractError(f"blank train JSONL line: {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise F2ContractError(
                f"invalid train JSONL at line {line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise F2ContractError(f"train row {line_number} must be an object")
        rows.append(value)
    return tuple(rows)


def _required_string(row: Mapping[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise F2ContractError(f"row {index} has invalid {field}")
    return value


def _required_bool(row: Mapping[str, Any], field: str, index: int) -> bool:
    if field not in row or not isinstance(row[field], bool):
        raise F2ContractError(f"row {index} has invalid {field}")
    return bool(row[field])


def _required_int(row: Mapping[str, Any], field: str, index: int) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise F2ContractError(f"row {index} has invalid {field}")
    return int(value)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise F2ContractError(f"{label} must be finite")
    return result


def _action_vector(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise F2ContractError(f"{label} must be an action vector")
    if len(value) < 3:
        raise F2ContractError(f"{label} must contain three axes")
    return (
        _finite_number(value[0], f"{label}[0]"),
        _finite_number(value[1], f"{label}[1]"),
        _finite_number(value[2], f"{label}[2]"),
    )


def _horizon_actions(row: Mapping[str, Any], index: int) -> tuple[tuple[float, float, float], ...]:
    value = row.get("step_actions")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise F2ContractError(f"row {index} has invalid step_actions")
    return tuple(
        _action_vector(action, f"row {index}.step_actions[{horizon}]")
        for horizon, action in enumerate(value)
    )


def continues_sequence(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> bool:
    """Corrigendum-2 continuity: mirror identity must match, not be false."""

    if previous is None:
        return False
    previous_sequence = _required_string(previous, "sequence_id", -1)
    current_sequence = _required_string(current, "sequence_id", -1)
    previous_frame = _required_int(previous, "frame_idx", -1)
    current_frame = _required_int(current, "frame_idx", -1)
    previous_mirrored = _required_bool(previous, "mirrored", -1)
    current_mirrored = _required_bool(current, "mirrored", -1)
    return (
        previous_sequence == current_sequence
        and current_frame == previous_frame + 1
        and previous_mirrored == current_mirrored
    )


def derive_base_reset_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    reset_rows: list[int] = []
    previous: Mapping[str, Any] | None = None
    for index, row in enumerate(rows):
        if previous is None or not continues_sequence(previous, row):
            reset_rows.append(index)
        previous = row
    return tuple(reset_rows)


def _with_immediate_successors(
    indices: Iterable[int], row_count: int
) -> tuple[int, ...]:
    expanded: set[int] = set()
    for index in indices:
        value = int(index)
        expanded.add(value)
        if value + 1 < row_count:
            expanded.add(value + 1)
    return tuple(sorted(expanded))


def derive_strafe_ledger(
    rows: Sequence[Mapping[str, Any]], eps_strafe: float = EPS_STRAFE
) -> StrafeLedger:
    if not math.isfinite(eps_strafe) or eps_strafe < 0:
        raise F2ContractError("eps_strafe must be finite and non-negative")
    audit: list[int] = []
    unsupported: list[int] = []
    for index, row in enumerate(rows):
        previous = _action_vector(row.get("prev_action"), f"row {index}.prev_action")
        horizon = _horizon_actions(row, index)
        if abs(previous[1]) > eps_strafe or any(
            abs(action[1]) > eps_strafe for action in horizon
        ):
            audit.append(index)
        if abs(previous[1]) > eps_strafe or abs(horizon[0][1]) > eps_strafe:
            unsupported.append(index)
    return StrafeLedger(
        audit_24=tuple(audit),
        unsupported_10=tuple(unsupported),
        reset_boundary_12=_with_immediate_successors(unsupported, len(rows)),
        diagnostic_superset_26=_with_immediate_successors(audit, len(rows)),
    )


def build_eligible_blocks(
    rows: Sequence[Mapping[str, Any]],
    rows_per_block: int = ROWS_PER_BLOCK,
) -> tuple[SupportBlock, ...]:
    if rows_per_block <= 0:
        raise F2ContractError("rows_per_block must be positive")
    groups: dict[tuple[str, str, str], list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(rows):
        key = (
            _required_string(row, "source_raw_dir", index),
            _required_string(row, "sequence_id", index),
            _required_string(row, "clip_id", index),
        )
        _required_bool(row, "mirrored", index)
        _required_int(row, "frame_idx", index)
        groups.setdefault(key, []).append((index, row))

    blocks: list[SupportBlock] = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) < rows_per_block:
            continue
        mirror_values = {
            _required_bool(row, "mirrored", index) for index, row in group
        }
        if len(mirror_values) != 1:
            raise F2ContractError(f"eligible group has mixed mirrored values: {key}")
        ordered = sorted(
            group,
            key=lambda pair: (
                _required_int(pair[1], "frame_idx", pair[0]),
                pair[0],
            ),
        )
        selected = ordered[:rows_per_block]
        blocks.append(
            SupportBlock(
                key=key,
                source_raw_dir=key[0],
                sequence_id=key[1],
                clip_id=key[2],
                mirrored=next(iter(mirror_values)),
                row_indices=tuple(index for index, _row in selected),
            )
        )
    return tuple(blocks)


def midpoint_stride_indices(total: int, slots: int) -> tuple[int, ...]:
    if total <= 0 or slots <= 0 or slots > total:
        raise F2ContractError("invalid midpoint-stride capacity")
    return tuple((((2 * slot + 1) * total) // (2 * slots)) % total for slot in range(slots))


def select_support_blocks(
    blocks: Sequence[SupportBlock],
    *,
    slots: int = SELECTION_SLOTS,
    assignment: Mapping[int, str] = SUPPORT_ASSIGNMENT,
) -> dict[str, tuple[SupportBlock, ...]]:
    if not blocks:
        raise F2ContractError("support pool is empty")
    if set(assignment) != {0, 1, 2, 3, 4}:
        raise F2ContractError("support assignment must cover j mod 5")
    starts = midpoint_stride_indices(len(blocks), slots)
    selected: dict[str, list[SupportBlock]] = {
        "CAL": [],
        "SMK-TRAIN": [],
        "EVAL-FIX": [],
    }
    used_blocks: set[tuple[str, str, str]] = set()
    used_sequences: dict[str, set[str]] = {name: set() for name in selected}
    for slot, start in enumerate(starts):
        support_name = assignment[slot % 5]
        chosen: SupportBlock | None = None
        for advance in range(len(blocks)):
            candidate = blocks[(start + advance) % len(blocks)]
            if candidate.key in used_blocks:
                continue
            if candidate.sequence_id in used_sequences[support_name]:
                continue
            chosen = candidate
            break
        if chosen is None:
            raise F2ContractError("SUPPORT_BUILD_EXHAUSTED")
        selected[support_name].append(chosen)
        used_blocks.add(chosen.key)
        used_sequences[support_name].add(chosen.sequence_id)
    return {name: tuple(values) for name, values in selected.items()}


def support_row_indices(blocks: Sequence[SupportBlock]) -> tuple[int, ...]:
    return tuple(sorted(index for block in blocks for index in block.row_indices))


def _is_h1_change(row: Mapping[str, Any], index: int) -> bool:
    previous = _action_vector(row.get("prev_action"), f"row {index}.prev_action")
    first = _horizon_actions(row, index)[0]
    return max(abs(first[0] - previous[0]), abs(first[2] - previous[2])) > CHANGE_THRESHOLD


def measure_support_coverage(
    rows: Sequence[Mapping[str, Any]],
    blocks: Sequence[SupportBlock],
    base_reset_rows: Iterable[int],
) -> SupportCoverage:
    indices = support_row_indices(blocks)
    reset_set = set(int(index) for index in base_reset_rows)
    # Every selected block is replayed as its own support stream.  Its first
    # row therefore satisfies the frozen ``stream_first`` reset predicate even
    # when that row was contiguous in the original full-train ordering.
    support_reset_rows = (set(indices) & reset_set) | {
        block.first_row_index for block in blocks
    }
    h1_change = 0
    turn = 0
    other = 0
    for index in indices:
        row = rows[index]
        h1_change += int(_is_h1_change(row, index))
        transition = row.get("transition_type")
        if not isinstance(transition, str):
            raise F2ContractError(f"row {index} has invalid transition_type")
        turn += int(transition in TURN_TRANSITION_TYPES)
        other += int(transition == "other")
    return SupportCoverage(
        blocks=len(blocks),
        rows=len(indices),
        h1_change=h1_change,
        turn=turn,
        other=other,
        unique_sequence=len({block.sequence_id for block in blocks}),
        episode=len({block.source_raw_dir for block in blocks}),
        mirror=sum(int(block.mirrored) for block in blocks),
        static_resets=len(support_reset_rows),
    )


def _validate_strafe_ledger(ledger: StrafeLedger) -> None:
    expected_counts = {
        "audit_24": 24,
        "unsupported_10": 10,
        "reset_boundary_12": 12,
        "diagnostic_superset_26": 26,
    }
    for field, expected in expected_counts.items():
        actual = len(getattr(ledger, field))
        if actual != expected:
            raise F2ContractError(
                f"STRAFE-FIX {field} count mismatch: {actual} != {expected}"
            )
    if not set(ledger.unsupported_10).issubset(ledger.audit_24):
        raise F2ContractError("STRAFE-FIX unsupported_10 is not a subset of audit_24")
    if not set(ledger.reset_boundary_12).issubset(ledger.diagnostic_superset_26):
        raise F2ContractError(
            "STRAFE-FIX reset_boundary_12 is not a subset of diagnostic_superset_26"
        )
    if canonical_json_sha256(list(ledger.reset_boundary_12)) != STRAFE_RESET_SHA256:
        raise F2ContractError("STRAFE-FIX reset-boundary SHA mismatch")
    if (
        canonical_json_sha256(list(ledger.diagnostic_superset_26))
        != "74a2672080343199269d774cb6e92e13c8495c10ce1a0fc08b06da4efdee95c5"
    ):
        raise F2ContractError("STRAFE-FIX diagnostic-superset SHA mismatch")


def _validate_support(
    name: str,
    blocks: Sequence[SupportBlock],
    indices: tuple[int, ...],
    coverage: SupportCoverage,
) -> str:
    expected = SUPPORT_EXPECTATIONS[name]
    actual_sha = canonical_json_sha256(list(indices))
    if coverage != SupportCoverage(
        blocks=expected.blocks,
        rows=expected.rows,
        h1_change=expected.h1_change,
        turn=expected.turn,
        other=expected.other,
        unique_sequence=expected.unique_sequence,
        episode=expected.episode,
        mirror=expected.mirror,
        static_resets=expected.static_resets,
    ):
        raise F2ContractError(f"{name} coverage receipt mismatch")
    if len(blocks) != expected.blocks or len(indices) != expected.rows:
        raise F2ContractError(f"{name} support size mismatch")
    if actual_sha != expected.sha256:
        raise F2ContractError(f"{name} row SHA mismatch")
    return actual_sha


def build_frozen_support_from_rows(
    rows: Sequence[Mapping[str, Any]], train_sha256: str
) -> FrozenSupportReceipt:
    if train_sha256 != FROZEN_TRAIN_SHA256:
        raise F2ContractError("HS1_train_precondition: train SHA mismatch")
    if len(rows) != FROZEN_TRAIN_ROWS:
        raise F2ContractError("HS1_train_precondition: train row count mismatch")

    base_resets = derive_base_reset_rows(rows)
    if len(base_resets) != BASE_RESET_COUNT:
        raise F2ContractError("HS2_reset_receipt: base reset count mismatch")
    if canonical_json_sha256(list(base_resets)) != BASE_RESET_SHA256:
        raise F2ContractError("HS2_reset_receipt: base reset SHA mismatch")

    strafe = derive_strafe_ledger(rows)
    _validate_strafe_ledger(strafe)
    if set(base_resets) & set(strafe.reset_boundary_12):
        raise F2ContractError("HS2_reset_receipt: base/strafe reset overlap")
    combined_resets = tuple(sorted(set(base_resets) | set(strafe.reset_boundary_12)))
    if len(combined_resets) != COMBINED_RESET_COUNT:
        raise F2ContractError("HS2_reset_receipt: combined reset count mismatch")
    if canonical_json_sha256(list(combined_resets)) != COMBINED_RESET_SHA256:
        raise F2ContractError("HS2_reset_receipt: combined reset SHA mismatch")

    blocks = build_eligible_blocks(rows)
    total = len(blocks)
    mirrored = sum(int(block.mirrored) for block in blocks)
    nonmirrored = total - mirrored
    if total != SELECTION_N or nonmirrored != 110 or mirrored != 63:
        raise F2ContractError("HS3_support_build: eligible pool mismatch")
    if total in FORBIDDEN_SELECTION_N:
        raise F2ContractError("HS3_support_build: forbidden selection capacity")

    supports = select_support_blocks(blocks)
    row_indices: dict[str, tuple[int, ...]] = {}
    row_sha256: dict[str, str] = {}
    coverage: dict[str, SupportCoverage] = {}
    for name in ("CAL", "SMK-TRAIN", "EVAL-FIX"):
        indices = support_row_indices(supports[name])
        measured = measure_support_coverage(rows, supports[name], base_resets)
        row_indices[name] = indices
        coverage[name] = measured
        row_sha256[name] = _validate_support(
            name, supports[name], indices, measured
        )

    sets = [set(row_indices[name]) for name in ("CAL", "SMK-TRAIN", "EVAL-FIX")]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise F2ContractError("HS3_support_build: supports are not disjoint")
    union = tuple(sorted(set().union(*sets)))
    if len(union) != UNION_ROWS or canonical_json_sha256(list(union)) != UNION_SHA256:
        raise F2ContractError("HS3_support_build: union receipt mismatch")

    support_rows = set(union)
    block_first_rows = {
        block.first_row_index for values in supports.values() for block in values
    }
    if not (support_rows & set(strafe.reset_boundary_12)).issubset(block_first_rows):
        raise F2ContractError("HS4_support_strafe_intersection")

    return FrozenSupportReceipt(
        train_sha256=train_sha256,
        train_rows=len(rows),
        eligible_pool_total=total,
        eligible_pool_nonmirrored=nonmirrored,
        eligible_pool_mirrored=mirrored,
        supports=supports,
        row_indices=row_indices,
        row_sha256=row_sha256,
        coverage=coverage,
        union_row_indices=union,
        union_sha256=UNION_SHA256,
        base_reset_rows=base_resets,
        combined_reset_rows=combined_resets,
        strafe=strafe,
    )


def build_frozen_support_from_payload(payload: bytes) -> FrozenSupportReceipt:
    actual_sha256 = sha256_bytes(payload)
    rows = parse_train_jsonl(payload)
    return build_frozen_support_from_rows(rows, actual_sha256)


def build_frozen_support(path: str | Path) -> FrozenSupportReceipt:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise F2ContractError(f"frozen train JSONL is missing: {source}")
    return build_frozen_support_from_payload(source.read_bytes())
