"""Fail-closed binding for the B0/B1 matched-128 SMK-TRAIN run.

The source dataset remains the frozen full ``train.jsonl``.  The 256-row
training view is a :class:`torch.utils.data.Subset` created in memory from the
indices frozen in F2 ``support_receipt_v3.json``.  Cache verification is scoped
to token files reachable from those rows; the aggregate cache payload is never
walked because it also contains the sealed internal-test split.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


MATCHED_ANALYSIS_CLASS = "b0_b1_matched_f2_smk_train_v1"
MATCHED_SUPPORT = "SMK-TRAIN"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
MATCHED_ROWS = 256
MATCHED_OPTIMIZER_UPDATES = 128
MATCHED_SOURCE_ROWS = 13746
MATCHED_SOURCE_SHA256 = (
    "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36"
)
MATCHED_TRAIN_RELATIVE_PATH = "data/collected_v1/datasets/train.jsonl"
MATCHED_SUPPORT_RECEIPT_PAYLOAD_SHA256 = (
    "df4801315743c58a0267cbc958587e559250c416b713941a73166970d76f9d0a"
)
MATCHED_ROW_SHA256 = (
    "7073a02c866913903a67438673ec7cf6898574bd7fa9bac891ad6142b563818f"
)


class MatchedSmokeContractError(RuntimeError):
    """Raised when a matched-128 run would leave the frozen contract."""


def prepare_matched_cli_environment(argv: Sequence[str]) -> bool:
    """Set cuBLAS determinism before torch-heavy CLI imports when requested."""

    flags = ("--matched_support_receipt", "--matched_128_support_receipt")
    requested = any(
        argument == flag or argument.startswith(flag + "=")
        for argument in argv
        for flag in flags
    )
    if not requested:
        return False
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise MatchedSmokeContractError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the matched Windows CUDA "
            f"contract: {existing!r} != {CUBLAS_WORKSPACE_CONFIG!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return True


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MatchedSmokeContractError(
            "matched support value is not canonical-JSON serializable"
        ) from exc


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchedSmokeContractError(f"{label} must be an object")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatchedSmokeContractError(f"{label} must be an integer")
    return int(value)


def _indices(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MatchedSmokeContractError(f"{label} must be an integer array")
    normalized = tuple(_integer(item, f"{label}[{position}]") for position, item in enumerate(value))
    if any(index < 0 for index in normalized):
        raise MatchedSmokeContractError(f"{label} contains a negative row index")
    if len(set(normalized)) != len(normalized):
        raise MatchedSmokeContractError(f"{label} contains duplicate row indices")
    if tuple(sorted(normalized)) != normalized:
        raise MatchedSmokeContractError(f"{label} must preserve frozen ascending order")
    return normalized


@dataclass(frozen=True)
class MatchedSmokeBinding:
    receipt_path: Path
    receipt_file_sha256: str
    receipt_payload_sha256: str
    relocated_root: Path
    train_path: Path
    train_relative_path: str
    source_rows: int
    source_sha256: str
    row_indices: tuple[int, ...]
    row_indices_sha256: str

    @property
    def expected_processed_samples(self) -> int:
        return MATCHED_ROWS

    @property
    def expected_optimizer_updates(self) -> int:
        return MATCHED_OPTIMIZER_UPDATES

    def metadata(
        self,
        *,
        token_ledger: Mapping[str, Any],
        cuda_reproducibility: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_class": MATCHED_ANALYSIS_CLASS,
            "support": MATCHED_SUPPORT,
            "support_receipt_path": str(self.receipt_path),
            "support_receipt_file_sha256": self.receipt_file_sha256,
            "support_receipt_payload_sha256": self.receipt_payload_sha256,
            "relocated_root": str(self.relocated_root),
            "source_dataset": {
                "path": str(self.train_path),
                "relative_path": self.train_relative_path,
                "rows": self.source_rows,
                "sha256": self.source_sha256,
            },
            "subset_materialization": "torch.utils.data.Subset_in_memory",
            "ordered_row_indices": list(self.row_indices),
            "ordered_row_indices_sha256": self.row_indices_sha256,
            "processed_samples": MATCHED_ROWS,
            "optimizer_updates": MATCHED_OPTIMIZER_UPDATES,
            "cache_verification": (
                "manifest_metadata_plus_smk_train_reachable_token_ledger;"
                "no_full_payload_walk"
            ),
            "scoped_token_ledger": dict(token_ledger),
            "cuda_reproducibility": dict(cuda_reproducibility),
            "internal_test": "sealed",
            "internal_test_opened": False,
        }


def enforce_matched_args(args: Any, *, family: str) -> None:
    """Freeze knobs that define the matched 256-sample/128-update run."""

    if family not in {"B0", "B1"}:
        raise MatchedSmokeContractError(f"unsupported matched family: {family}")
    if getattr(args, "relocated_root", None) in (None, ""):
        raise MatchedSmokeContractError(
            "--relocated_root is required with --matched_support_receipt"
        )
    if getattr(args, "val_json", None) not in (None, "") or getattr(
        args, "val_cache_root", None
    ) not in (None, ""):
        raise MatchedSmokeContractError(
            "matched-128 training forbids validation/cache access; evaluate "
            "EVAL-FIX in the separate frozen evaluator"
        )
    if int(args.seed) != 0:
        raise MatchedSmokeContractError("matched-128 seed is frozen to 0")
    if int(args.epochs) != 1:
        raise MatchedSmokeContractError("matched-128 epochs is frozen to 1")
    if int(args.history) != 31:
        raise MatchedSmokeContractError("matched-128 history is frozen to 31")
    if int(args.max_steps) != 0:
        raise MatchedSmokeContractError(
            "matched-128 uses optimizer updates, not --max_steps"
        )
    requested_updates = int(args.max_optimizer_updates)
    if requested_updates not in (0, MATCHED_OPTIMIZER_UPDATES):
        raise MatchedSmokeContractError(
            "matched-128 max_optimizer_updates must be 128 (or omitted)"
        )
    args.max_optimizer_updates = MATCHED_OPTIMIZER_UPDATES

    if family == "B0":
        if int(args.batch_size) != 2:
            raise MatchedSmokeContractError("B0 matched-128 batch_size is frozen to 2")
        if float(args.lr) != 2e-5:
            raise MatchedSmokeContractError("B0 matched-128 lr is frozen to 2e-5")
        if bool(args.balance_sampling):
            raise MatchedSmokeContractError(
                "B0 matched-128 forbids weighted/random sampling"
            )
    else:
        if int(args.batch_size) != 1 or int(args.grad_accum_steps) != 2:
            raise MatchedSmokeContractError(
                "B1 matched-128 requires batch_size=1 and grad_accum_steps=2"
            )
        if float(args.base_lr) != 2e-5 or float(args.head_lr) != 3e-4:
            raise MatchedSmokeContractError(
                "B1 matched-128 learning rates are frozen to base=2e-5, head=3e-4"
            )
        if args.variant != "polar_tim4":
            raise MatchedSmokeContractError(
                "B1 matched-128 variant is frozen to polar_tim4"
            )
        if args.state_mode != "rolling":
            raise MatchedSmokeContractError(
                "B1 matched-128 recurrence/reset mode is frozen to rolling"
            )


def configure_matched_cuda(torch_module: Any) -> dict[str, Any]:
    """Apply and return the frozen Windows CUDA deterministic receipt."""

    if os.name != "nt":
        raise MatchedSmokeContractError(
            "matched-128 production training is frozen to Windows CUDA"
        )
    from f2_experiment.reproducibility import configure_cuda_reproducibility

    receipt = configure_cuda_reproducibility(torch_module)
    if not bool(torch_module.cuda.is_available()):
        raise MatchedSmokeContractError(
            "matched-128 production training requires an available CUDA device"
        )
    return receipt


def load_matched_smoke_binding(
    receipt_path: str | Path,
    *,
    train_json: str | Path,
    dataset_info: Mapping[str, Any],
    relocated_root: str | Path,
) -> MatchedSmokeBinding:
    """Validate the frozen receipt without rebuilding or opening other splits."""

    path = Path(receipt_path).expanduser().resolve()
    if not path.is_file():
        raise MatchedSmokeContractError(f"support receipt is missing: {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatchedSmokeContractError("support receipt is not valid UTF-8 JSON") from exc
    receipt = _mapping(document, "support receipt")
    recorded_payload = receipt.get("receipt_payload_sha256")
    if not isinstance(recorded_payload, str):
        raise MatchedSmokeContractError("support receipt payload SHA is missing")
    payload_without_sha = dict(receipt)
    payload_without_sha.pop("receipt_payload_sha256", None)
    if _canonical_json_sha256(payload_without_sha) != recorded_payload:
        raise MatchedSmokeContractError("support receipt payload SHA mismatch")
    if recorded_payload != MATCHED_SUPPORT_RECEIPT_PAYLOAD_SHA256:
        raise MatchedSmokeContractError(
            "support receipt is not the frozen support_receipt_v3 payload"
        )
    if receipt.get("analysis_class") != "f2_preformal_support_receipt":
        raise MatchedSmokeContractError("unexpected support receipt analysis class")
    if receipt.get("internal_test") != "sealed" or receipt.get(
        "internal_test_opened"
    ) is not False:
        raise MatchedSmokeContractError("support receipt violates internal-test seal")

    root = Path(relocated_root).expanduser().resolve()
    if not root.is_dir():
        raise MatchedSmokeContractError(f"relocated_root does not exist: {root}")
    relative_text = receipt.get("train_relative_path")
    if not isinstance(relative_text, str) or not relative_text:
        raise MatchedSmokeContractError("receipt train_relative_path is missing")
    if relative_text != MATCHED_TRAIN_RELATIVE_PATH:
        raise MatchedSmokeContractError("receipt is not bound to the frozen train path")
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise MatchedSmokeContractError("receipt train_relative_path is not clean")
    expected_train = root.joinpath(*relative.parts).resolve()
    actual_train = Path(train_json).expanduser().resolve()
    if actual_train != expected_train:
        raise MatchedSmokeContractError(
            "matched-128 must bind the original full train.jsonl under relocated_root"
        )

    support = _mapping(receipt.get("support"), "support")
    support_train = _mapping(support.get("train"), "support.train")
    source_rows = _integer(support_train.get("rows"), "support.train.rows")
    source_sha = support_train.get("sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise MatchedSmokeContractError("support.train.sha256 is invalid")
    if source_rows != int(dataset_info.get("sample_count", -1)):
        raise MatchedSmokeContractError("source train row count differs from receipt")
    if source_sha != dataset_info.get("data_hash"):
        raise MatchedSmokeContractError("source train SHA differs from receipt")
    if source_rows != MATCHED_SOURCE_ROWS or source_sha != MATCHED_SOURCE_SHA256:
        raise MatchedSmokeContractError("source train identity differs from frozen binding")

    supports = _mapping(support.get("supports"), "support.supports")
    smk_support = _mapping(supports.get(MATCHED_SUPPORT), "SMK-TRAIN support")
    support_indices = _indices(smk_support.get("row_indices"), "SMK-TRAIN row_indices")
    support_sha = smk_support.get("row_sha256")

    plan = _mapping(receipt.get("smoke_plan"), "smoke_plan")
    smoke = _mapping(plan.get("smoke"), "smoke_plan.smoke")
    plan_indices = _indices(
        smoke.get("ordered_row_indices"), "smoke_plan.smoke.ordered_row_indices"
    )
    if support_indices != plan_indices:
        raise MatchedSmokeContractError("SMK-TRAIN support/plan row order differs")
    if len(support_indices) != MATCHED_ROWS or _integer(
        smoke.get("rows"), "smoke_plan.smoke.rows"
    ) != MATCHED_ROWS:
        raise MatchedSmokeContractError("matched support must contain 256 rows")
    if support_indices[-1] >= source_rows:
        raise MatchedSmokeContractError("SMK-TRAIN row index exceeds source dataset")
    computed_row_sha = _canonical_json_sha256(list(support_indices))
    if (
        computed_row_sha != MATCHED_ROW_SHA256
        or support_sha != computed_row_sha
        or smoke.get("ordered_row_indices_sha256") != computed_row_sha
    ):
        raise MatchedSmokeContractError("SMK-TRAIN ordered row SHA mismatch")

    if _integer(smoke.get("optimizer_updates"), "smoke optimizer_updates") != MATCHED_OPTIMIZER_UPDATES:
        raise MatchedSmokeContractError("SMK-TRAIN optimizer update budget differs")
    if _integer(smoke.get("gradient_accumulation_rows"), "smoke gradient_accumulation_rows") != 2:
        raise MatchedSmokeContractError("SMK-TRAIN accumulation budget differs")
    pairs = smoke.get("update_pairs")
    expected_pairs = [list(support_indices[offset : offset + 2]) for offset in range(0, MATCHED_ROWS, 2)]
    if pairs != expected_pairs or len(expected_pairs) != MATCHED_OPTIMIZER_UPDATES:
        raise MatchedSmokeContractError("SMK-TRAIN update-pair schedule differs")
    if smoke.get("update_pairs_sha256") != _canonical_json_sha256(expected_pairs):
        raise MatchedSmokeContractError("SMK-TRAIN update-pair SHA mismatch")

    budget = _mapping(plan.get("budget"), "smoke_plan.budget")
    per_arm = _mapping(budget.get("per_arm"), "smoke_plan.budget.per_arm")
    if budget.get("arm_identical") is not True:
        raise MatchedSmokeContractError("matched smoke budget is not arm-identical")
    if _integer(per_arm.get("rows"), "budget rows") != MATCHED_ROWS or _integer(
        per_arm.get("optimizer_steps"), "budget optimizer_steps"
    ) != MATCHED_OPTIMIZER_UPDATES:
        raise MatchedSmokeContractError("matched smoke per-arm budget differs")

    return MatchedSmokeBinding(
        receipt_path=path,
        receipt_file_sha256=hashlib.sha256(raw).hexdigest(),
        receipt_payload_sha256=recorded_payload,
        relocated_root=root,
        train_path=actual_train,
        train_relative_path=relative.as_posix(),
        source_rows=source_rows,
        source_sha256=source_sha,
        row_indices=support_indices,
        row_indices_sha256=computed_row_sha,
    )


def build_scoped_subset(
    dataset: Any,
    binding: MatchedSmokeBinding,
    *,
    cache_root: str | Path,
):
    """Return the in-memory Subset and its support-reachable token ledger."""

    from torch.utils.data import Subset
    from f2_experiment.assembly_data import build_token_ledger_for_rows

    if len(dataset) != binding.source_rows:
        raise MatchedSmokeContractError("dataset length differs from source binding")
    base_root = Path(getattr(dataset, "base_root", "")).expanduser().resolve()
    if base_root != binding.relocated_root:
        raise MatchedSmokeContractError(
            "dataset manifest base_root differs from --relocated_root"
        )
    rows = [dataset.get_example(index) for index in binding.row_indices]
    ledger = build_token_ledger_for_rows(
        rows,
        base_root=base_root,
        cache_root=cache_root,
    )
    if int(ledger.token_files) <= 0:
        raise MatchedSmokeContractError("scoped token ledger is empty")
    ledger_binding = {
        "schema_version": 1,
        "scope": MATCHED_SUPPORT,
        "token_files": int(ledger.token_files),
        "ledger_sha256": str(ledger.ledger_sha256),
    }
    return Subset(dataset, list(binding.row_indices)), ledger_binding


def assert_matched_loader(loader: Any, *, family: str) -> None:
    expected_batches = MATCHED_OPTIMIZER_UPDATES if family == "B0" else MATCHED_ROWS
    if len(loader.dataset) != MATCHED_ROWS or len(loader) != expected_batches:
        raise MatchedSmokeContractError(
            f"{family} matched loader budget differs: dataset={len(loader.dataset)}, "
            f"batches={len(loader)}"
        )


def assert_matched_counters(processed_samples: int, optimizer_updates: int) -> None:
    if int(processed_samples) != MATCHED_ROWS or int(
        optimizer_updates
    ) != MATCHED_OPTIMIZER_UPDATES:
        raise MatchedSmokeContractError(
            "matched-128 completed with the wrong budget: "
            f"processed_samples={processed_samples}, optimizer_updates={optimizer_updates}"
        )


__all__ = [
    "MATCHED_ANALYSIS_CLASS",
    "CUBLAS_WORKSPACE_CONFIG",
    "MATCHED_OPTIMIZER_UPDATES",
    "MATCHED_ROWS",
    "MATCHED_ROW_SHA256",
    "MATCHED_SUPPORT",
    "MATCHED_SOURCE_ROWS",
    "MATCHED_SOURCE_SHA256",
    "MATCHED_SUPPORT_RECEIPT_PAYLOAD_SHA256",
    "MATCHED_TRAIN_RELATIVE_PATH",
    "MatchedSmokeBinding",
    "MatchedSmokeContractError",
    "assert_matched_counters",
    "assert_matched_loader",
    "build_scoped_subset",
    "configure_matched_cuda",
    "enforce_matched_args",
    "load_matched_smoke_binding",
    "prepare_matched_cli_environment",
]
