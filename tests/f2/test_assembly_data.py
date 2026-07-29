"""Tests for the fail-closed data-side F2 production assembly.

Tiny synthetic fixtures exercise the loader mechanics; the frozen
identities (asset SHAs, block order, 12/28/30 static resets, EVAL-FIX
strata 69/154/211) are asserted against the real frozen train JSONL and
the real on-disk assets.  The sealed internal held-out test is never
read.
"""

import builtins
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import pytest
import torch

from f2_experiment.assembly_data import (
    AUX_FUT_KEYS,
    FROZEN_BASE_HF_ARTIFACT_SHA256,
    FROZEN_BASE_HF_DIR_DEFAULT,
    FROZEN_CACHE_MANIFEST_SHA256,
    FROZEN_CACHE_PROVENANCE_SHA256,
    FROZEN_CACHE_ROOT_RELATIVE,
    FROZEN_DINO_SHA256,
    FROZEN_INTERNAL_TEST_IMAGE_PREFIX,
    FROZEN_PROMPT_ERRATUM_SHA256,
    FROZEN_QWEN_SHA256,
    FROZEN_SIGLIP_SHA256,
    FROZEN_TOKEN_PAYLOAD_SHA256,
    FROZEN_TRAIN_IMAGE_PREFIXES,
    OBSERVATION_ALLOWED_KEYS,
    OBSERVATION_FORBIDDEN_KEYS,
    AuxTargetPacket,
    F2AssemblyContractError,
    ObservationPacket,
    TokenHashLedger,
    aux_target_packet_from_row,
    build_runner_rows,
    build_token_ledger_for_rows,
    build_train_token_ledger,
    collect_image_relpaths,
    ensure_observation_packet,
    eval_fix_strata,
    frozen_cache_roots,
    load_cached_observation,
    observation_packet_from_fields,
    ordered_support_rows,
    smoke_reset_sets,
    support_reset_plan,
    verify_frozen_assets,
)
from f2_experiment.runner import RunnerRow
from f2_experiment.support import (
    SUPPORT_EXPECTATIONS,
    FrozenSupportReceipt,
    StrafeLedger,
    SupportBlock,
    build_frozen_support_from_rows,
    canonical_json_sha256,
    derive_base_reset_rows,
    measure_support_coverage,
    parse_train_jsonl,
    sha256_bytes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data/collected_v1/datasets/train.jsonl"
SUPPORT_NAMES = ("CAL", "SMK-TRAIN", "EVAL-FIX")


# --------------------------------------------------------------------------
# Tiny fixture helpers.
# --------------------------------------------------------------------------


def _tiny_row(
    *,
    source="ep0",
    sequence="seq0",
    clip="clip0",
    frame=0,
    mirrored=False,
    current="eps/cur.jpg",
    images=(),
    prev=(0.1, 0.0, -0.2),
    first=(0.5, 0.0, 0.3),
    transition="other",
    theta=12,
    dist=5,
    invalid=0.0,
):
    row = {
        "source_raw_dir": source,
        "sequence_id": sequence,
        "clip_id": clip,
        "frame_idx": frame,
        "mirrored": mirrored,
        "current": current,
        "images": list(images),
        "instruction": "Follow the person wearing black pants.",
        "prev_action": list(prev),
        "step_actions": [list(first) for _ in range(8)],
        "transition_type": transition,
        "polar_theta_idx": theta,
        "polar_dist_idx": dist,
        "polar_invalid": invalid,
    }
    for horizon in (4, 8, 16):
        row[f"fut_valid_{horizon}"] = True
        row[f"fut_vis_{horizon}"] = 1.0
        row[f"fut_theta_idx_{horizon}"] = 3
        row[f"fut_dist_idx_{horizon}"] = 4
    return row


def _write_token(cache_root: Path, rel_image: str, level: str, tensor=None):
    suffix = {"fine": "vfine", "coarse": "vcoarse"}[level]
    shape = (64, 1536) if level == "fine" else (4, 1536)
    if tensor is None:
        tensor = torch.randn(*shape, dtype=torch.float32).half()
    destination = cache_root / Path(rel_image).parent
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{Path(rel_image).name}_{suffix}.pt"
    torch.save(tensor, path)
    return path


def _ledger_for_cache(cache_root: Path) -> TokenHashLedger:
    """Hash every token file under a tiny fixture cache into a ledger."""

    entries = {
        path.relative_to(cache_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(cache_root.rglob("*.pt"))
    }
    return TokenHashLedger(entries=entries)


def _tiny_receipt(rows, blocks, *, train_sha="7" * 64, strafe_rows=()):
    base_resets = derive_base_reset_rows(rows)
    indices = tuple(sorted(i for block in blocks for i in block.row_indices))
    coverage = measure_support_coverage(rows, blocks, base_resets)
    row_sha = canonical_json_sha256(list(indices))
    names = ("CAL", "SMK-TRAIN", "EVAL-FIX")
    return FrozenSupportReceipt(
        train_sha256=train_sha,
        train_rows=len(rows),
        eligible_pool_total=len(blocks),
        eligible_pool_nonmirrored=len(blocks),
        eligible_pool_mirrored=0,
        supports={name: tuple(blocks) for name in names},
        row_indices={name: indices for name in names},
        row_sha256={name: row_sha for name in names},
        coverage={name: coverage for name in names},
        union_row_indices=indices,
        union_sha256=canonical_json_sha256(list(indices)),
        base_reset_rows=base_resets,
        combined_reset_rows=tuple(sorted(set(base_resets) | set(strafe_rows))),
        strafe=StrafeLedger(
            audit_24=tuple(strafe_rows),
            unsupported_10=tuple(strafe_rows),
            reset_boundary_12=tuple(strafe_rows),
            diagnostic_superset_26=tuple(strafe_rows),
        ),
    )


def _block(rows, indices, *, key=None):
    first = rows[indices[0]]
    key = key or (
        first["source_raw_dir"],
        first["sequence_id"],
        first["clip_id"],
    )
    return SupportBlock(
        key=key,
        source_raw_dir=key[0],
        sequence_id=key[1],
        clip_id=key[2],
        mirrored=first["mirrored"],
        row_indices=tuple(indices),
    )


@pytest.fixture()
def tiny_assembly(tmp_path):
    rows = [
        _tiny_row(sequence="seq0", clip="c0", frame=0),
        _tiny_row(sequence="seq0", clip="c0", frame=1),
        _tiny_row(sequence="seq0", clip="c0", frame=2, transition="turn_onset"),
        _tiny_row(sequence="seq1", clip="c1", frame=5, first=(0.9, 0.0, -0.7)),
        _tiny_row(sequence="seq1", clip="c1", frame=6),
        # frame gap 6 -> 8: a mid-block sequence discontinuity.
        _tiny_row(sequence="seq1", clip="c1", frame=8),
    ]
    blocks = (_block(rows, (0, 1, 2)), _block(rows, (3, 4, 5)))
    receipt = _tiny_receipt(rows, blocks)
    base_root = tmp_path
    cache_root = tmp_path / "vision_cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    _write_token(cache_root, "eps/cur.jpg", "coarse")
    ledger = _ledger_for_cache(cache_root)
    return rows, receipt, base_root, cache_root, ledger


@pytest.fixture(scope="module")
def real_support():
    payload = TRAIN_PATH.read_bytes()
    rows = parse_train_jsonl(payload)
    receipt = build_frozen_support_from_rows(rows, sha256_bytes(payload))
    return rows, receipt


# --------------------------------------------------------------------------
# Frozen identity constants.
# --------------------------------------------------------------------------


def test_frozen_constants_match_adjudicated_values():
    assert FROZEN_BASE_HF_ARTIFACT_SHA256 == (
        "ff1d31982271cb922c91a26f7767438124e12502e9341251041d8541f7d63a8f"
    )
    assert FROZEN_CACHE_MANIFEST_SHA256 == (
        "127bda80a3d748f704b01bcf456c1e2e7c6c5b607f7eebe848fb5dc0e7824009"
    )
    assert FROZEN_CACHE_PROVENANCE_SHA256 == (
        "5399927be976e13f7c180143514f0863ac8687faf800fd1612cb3b9c42640ba4"
    )
    assert FROZEN_TOKEN_PAYLOAD_SHA256 == (
        "f0016a2a25f8724ec45040eedb4ce73e54ca342ba1cf400a4c0a6ab0e1592744"
    )
    assert FROZEN_DINO_SHA256 == (
        "627c7bb4f39f79e15d5e3fdf61557172d11befbe0b42c6f4513bf3907f5fc7a1"
    )
    assert FROZEN_SIGLIP_SHA256 == (
        "e9549756bf15a3ff2064c8a32f1086e9391f374682ca16a05c30f91fcbb5a096"
    )
    assert FROZEN_QWEN_SHA256 == (
        "2f62d9a42d8cf3cd43a69155c345e024d0d5bd1590a701540c0f75aeae71162b"
    )
    assert FROZEN_PROMPT_ERRATUM_SHA256 == (
        "baa9c322366e40377858cdedc9618dcc08e419df7991ae4bd3e7ca499facdbec"
    )
    assert str(FROZEN_CACHE_ROOT_RELATIVE) == "data/collected_v1/vision_cache"
    assert FROZEN_BASE_HF_DIR_DEFAULT.name == "opentrackvla-qwen06b"


def test_frozen_static_reset_and_strata_expectations():
    assert SUPPORT_EXPECTATIONS["SMK-TRAIN"].static_resets == 12
    assert SUPPORT_EXPECTATIONS["EVAL-FIX"].static_resets == 28
    assert SUPPORT_EXPECTATIONS["CAL"].static_resets == 30
    assert SUPPORT_EXPECTATIONS["EVAL-FIX"].h1_change == 69
    assert SUPPORT_EXPECTATIONS["EVAL-FIX"].turn == 154
    assert SUPPORT_EXPECTATIONS["EVAL-FIX"].other == 211


def test_observation_key_partition_is_machine_checkable():
    assert not OBSERVATION_ALLOWED_KEYS & OBSERVATION_FORBIDDEN_KEYS
    for key in (
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
    ):
        assert key in OBSERVATION_FORBIDDEN_KEYS
    for kind in ("valid", "vis", "theta_idx", "dist_idx"):
        for horizon in (4, 8, 16):
            assert f"fut_{kind}_{horizon}" in OBSERVATION_FORBIDDEN_KEYS
    assert len(AUX_FUT_KEYS) == 12


# --------------------------------------------------------------------------
# Frozen asset bindings against the real on-disk assets.
# --------------------------------------------------------------------------


def test_verify_frozen_assets_real_bindings():
    document = verify_frozen_assets(PROJECT_ROOT, verify_token_payload=False)
    assert document["train"]["sha256"] == (
        "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36"
    )
    cache = document["vision_cache"]
    assert cache["cache_manifest_sha256"] == FROZEN_CACHE_MANIFEST_SHA256
    assert cache["cache_provenance_sha256"] == FROZEN_CACHE_PROVENANCE_SHA256
    assert cache["token_payload_sha256"] == FROZEN_TOKEN_PAYLOAD_SHA256
    assert cache["dino_model_sha256"] == FROZEN_DINO_SHA256
    assert cache["siglip_model_sha256"] == FROZEN_SIGLIP_SHA256
    assert cache["token_payload_verified"] is False
    assert cache["image_base_root"] == str(PROJECT_ROOT)
    assert document["base_hf"]["artifact_sha256"] == (
        FROZEN_BASE_HF_ARTIFACT_SHA256
    )
    assert document["qwen"]["artifact_sha256"] == FROZEN_QWEN_SHA256
    assert document["prompt_erratum"]["sha256"] == FROZEN_PROMPT_ERRATUM_SHA256
    assert document["internal_test_opened"] is False


def test_verify_frozen_assets_rejects_unknown_base_checkpoint(tmp_path):
    fake = tmp_path / "fake_hf"
    fake.mkdir()
    (fake / "config.json").write_text("{}", encoding="utf-8")
    (fake / "model.safetensors").write_bytes(b"not the frozen checkpoint")
    with pytest.raises(F2AssemblyContractError, match="base HF checkpoint"):
        verify_frozen_assets(
            PROJECT_ROOT, base_hf_dir=fake, verify_token_payload=False
        )


def test_verify_frozen_assets_missing_train_fails(tmp_path):
    with pytest.raises(F2AssemblyContractError, match="frozen train JSONL"):
        verify_frozen_assets(tmp_path, verify_token_payload=False)


def test_frozen_cache_roots_real_identity():
    base_root, cache_root = frozen_cache_roots(PROJECT_ROOT)
    assert base_root == PROJECT_ROOT.resolve()
    assert cache_root == (PROJECT_ROOT / FROZEN_CACHE_ROOT_RELATIVE).resolve()


def test_frozen_cache_roots_rejects_foreign_path_root(tmp_path):
    cache_dir = tmp_path / FROZEN_CACHE_ROOT_RELATIVE
    cache_dir.mkdir(parents=True)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps({"path_root": str(tmp_path / "elsewhere")}),
        encoding="utf-8",
    )
    with pytest.raises(F2AssemblyContractError, match="path_root"):
        frozen_cache_roots(tmp_path)


# --------------------------------------------------------------------------
# Real frozen supports: block order, resets, strata.
# --------------------------------------------------------------------------


def test_real_block_major_order_is_frozen(real_support):
    rows, receipt = real_support
    lengths = {"CAL": 512, "SMK-TRAIN": 256, "EVAL-FIX": 512}
    for name in SUPPORT_NAMES:
        ordered = ordered_support_rows(rows, receipt, name)
        assert len(ordered) == lengths[name]
        indices = [index for index, _row in ordered]
        assert all(b > a for a, b in zip(indices, indices[1:]))
    smk = ordered_support_rows(rows, receipt, "SMK-TRAIN")
    assert [index for index, _row in smk[:5]] == [900, 901, 902, 903, 904]


def test_real_smoke_reset_sets_match_frozen_counts(real_support):
    _rows, receipt = real_support
    strafe_smk, expected_smk = smoke_reset_sets(receipt, "SMK-TRAIN")
    assert strafe_smk == frozenset()
    assert len(expected_smk) == 12
    strafe_eval, expected_eval = smoke_reset_sets(receipt, "EVAL-FIX")
    assert len(expected_eval) == 28
    assert strafe_eval <= expected_eval
    strafe_cal, expected_cal = smoke_reset_sets(receipt, "CAL")
    assert len(expected_cal) == 30
    assert strafe_cal <= expected_cal


def test_real_reset_plans_align_with_runner_predicates(real_support):
    rows, receipt = real_support
    expectations = {"CAL": 30, "SMK-TRAIN": 12, "EVAL-FIX": 28}
    for name, resets in expectations.items():
        plan = support_reset_plan(rows, receipt, name)
        assert len(plan) == SUPPORT_EXPECTATIONS[name].rows
        assert plan[0] == ("stream_first",)
        assert sum(1 for reasons in plan if reasons) == resets


def test_real_eval_fix_strata_counts(real_support):
    rows, receipt = real_support
    strata = eval_fix_strata(rows, receipt, "EVAL-FIX")
    assert strata["counts"] == {
        "overall": 512,
        "change": 69,
        "turn": 154,
        "other": 211,
    }
    assert strata["strata"] == ("overall", "change", "turn", "other")
    assert len(strata["masks"]["turn"]) == 512
    cal = eval_fix_strata(rows, receipt, "CAL")
    assert cal["counts"] == {
        "overall": 512,
        "change": 90,
        "turn": 134,
        "other": 247,
    }


# --------------------------------------------------------------------------
# Observation allowlist (blocker 10).
# --------------------------------------------------------------------------


def _packet_fields(history=2):
    return {
        "coarse_tokens": torch.zeros(history * 4, 1536),
        "coarse_tidx": torch.arange(history, dtype=torch.long)
        .repeat_interleave(4),
        "fine_tokens": torch.zeros(64, 1536),
        "fine_tidx": torch.full((64,), history, dtype=torch.long),
        "instruction": "Follow the person wearing black pants.",
    }


def test_observation_packet_from_fields_happy():
    packet = observation_packet_from_fields(_packet_fields())
    assert isinstance(packet, ObservationPacket)
    assert packet.history_frames == 2
    assert packet.yaw_hist is None and packet.yaw_curr is None
    assert ensure_observation_packet(packet) is packet


def test_observation_packet_rejects_forbidden_keys():
    for leak in ("step_actions", "prev_action", "fut_valid_4", "waypoints"):
        fields = _packet_fields()
        fields[leak] = [0.0, 0.0, 0.0]
        with pytest.raises(F2AssemblyContractError, match="OBSERVATION_LEAK"):
            observation_packet_from_fields(fields)


def test_observation_packet_rejects_unknown_and_missing_keys():
    fields = _packet_fields()
    fields["bonus_feature"] = torch.zeros(1)
    with pytest.raises(F2AssemblyContractError, match="unknown"):
        observation_packet_from_fields(fields)
    fields = _packet_fields()
    del fields["instruction"]
    with pytest.raises(F2AssemblyContractError, match="missing"):
        observation_packet_from_fields(fields)


def test_observation_packet_validates_tensor_contract():
    fields = _packet_fields()
    fields["fine_tokens"] = torch.zeros(63, 1536)
    with pytest.raises(F2AssemblyContractError, match="fine_tokens"):
        observation_packet_from_fields(fields)
    fields = _packet_fields()
    fields["coarse_tokens"] = torch.full((8, 1536), float("nan"))
    with pytest.raises(F2AssemblyContractError, match="nonfinite"):
        observation_packet_from_fields(fields)
    fields = _packet_fields()
    fields["coarse_tidx"] = torch.zeros(8, dtype=torch.long)
    with pytest.raises(F2AssemblyContractError, match="coarse_tidx"):
        observation_packet_from_fields(fields)
    fields = _packet_fields()
    fields["coarse_tokens"] = torch.zeros(8, 1536, requires_grad=True)
    with pytest.raises(F2AssemblyContractError, match="grad"):
        observation_packet_from_fields(fields)
    fields = _packet_fields()
    fields["instruction"] = ""
    with pytest.raises(F2AssemblyContractError, match="instruction"):
        observation_packet_from_fields(fields)


def test_ensure_observation_packet_rejects_raw_rows():
    with pytest.raises(F2AssemblyContractError, match="OBSERVATION_LEAK"):
        ensure_observation_packet({"step_actions": [[1.0, 0.0, 0.0]]})


# --------------------------------------------------------------------------
# Auxiliary target packet.
# --------------------------------------------------------------------------


def test_aux_target_packet_roundtrip_and_immutability():
    packet = aux_target_packet_from_row(_tiny_row(), "row 0")
    assert isinstance(packet, AuxTargetPacket)
    assert packet.theta_idx.dtype == torch.long
    assert packet.theta_idx.shape == (1,)
    assert packet.invalid.dtype == torch.float32
    targets = packet.as_targets()
    assert set(targets) == {
        "polar_theta_idx",
        "polar_dist_idx",
        "polar_invalid",
    } | set(AUX_FUT_KEYS)
    assert all(not tensor.requires_grad for tensor in targets.values())
    with pytest.raises(TypeError):
        packet.fut["fut_vis_4"] = torch.zeros(1)


def test_aux_target_packet_fail_closed_paths():
    row = _tiny_row(invalid=0.5)
    with pytest.raises(F2AssemblyContractError, match="invalid"):
        aux_target_packet_from_row(row, "row 0")
    row = _tiny_row(theta=60)
    with pytest.raises(F2AssemblyContractError, match="theta_idx"):
        aux_target_packet_from_row(row, "row 0")
    row = _tiny_row(theta=-1, dist=-1)
    with pytest.raises(F2AssemblyContractError, match="theta_idx"):
        aux_target_packet_from_row(row, "row 0")
    row = _tiny_row()
    row["fut_theta_idx_8"] = -1  # visible horizon with a sentinel bin
    with pytest.raises(F2AssemblyContractError, match="visible"):
        aux_target_packet_from_row(row, "row 0")
    row = _tiny_row()
    del row["fut_vis_16"]
    with pytest.raises(F2AssemblyContractError, match="fut_vis_16"):
        aux_target_packet_from_row(row, "row 0")


def test_aux_target_packet_allows_invalid_sentinels():
    packet = aux_target_packet_from_row(
        _tiny_row(theta=-1, dist=-1, invalid=1.0), "row 0"
    )
    assert int(packet.theta_idx.item()) == -1
    assert float(packet.invalid.item()) == 1.0


# --------------------------------------------------------------------------
# Cache-only observation loader (blocker 6b).
# --------------------------------------------------------------------------


def test_load_cached_observation_pads_missing_history(tmp_path):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    coarse_path = _write_token(cache_root, "eps/cur.jpg", "coarse")
    row = _tiny_row(images=())
    packet = load_cached_observation(
        row,
        base_root=tmp_path,
        cache_root=cache_root,
        token_ledger=_ledger_for_cache(cache_root),
        history=3,
    )
    assert packet.history_frames == 3
    assert packet.coarse_tokens.shape == (12, 1536)
    current_coarse = torch.load(coarse_path, map_location="cpu").float()
    for frame in range(3):
        assert torch.equal(
            packet.coarse_tokens[frame * 4 : (frame + 1) * 4], current_coarse
        )
    assert packet.fine_tokens.shape == (64, 1536)
    assert bool((packet.fine_tidx == 3).all())
    assert packet.yaw_hist is None and packet.yaw_curr is None


def test_load_cached_observation_mixed_history_and_yaw(tmp_path):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    cur_coarse = _write_token(cache_root, "eps/cur.jpg", "coarse")
    img0 = _write_token(cache_root, "eps/h0.jpg", "coarse")
    img1 = _write_token(cache_root, "eps/h1.jpg", "coarse")
    row = _tiny_row(images=("eps/h0.jpg", "eps/h1.jpg"))
    row["yaw_hist"] = [0.0, 0.1, -0.1, 0.2]
    row["yaw_curr"] = 0.25
    packet = load_cached_observation(
        row,
        base_root=tmp_path,
        cache_root=cache_root,
        token_ledger=_ledger_for_cache(cache_root),
        history=4,
    )
    expected_current = torch.load(cur_coarse, map_location="cpu").float()
    expected_h0 = torch.load(img0, map_location="cpu").float()
    expected_h1 = torch.load(img1, map_location="cpu").float()
    assert torch.equal(packet.coarse_tokens[0:4], expected_current)
    assert torch.equal(packet.coarse_tokens[4:8], expected_current)
    assert torch.equal(packet.coarse_tokens[8:12], expected_h0)
    assert torch.equal(packet.coarse_tokens[12:16], expected_h1)
    assert packet.yaw_hist.shape == (4,)
    assert float(packet.yaw_curr.item()) == pytest.approx(0.25)


def test_load_cached_observation_missing_tokens_fail_closed(tmp_path):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    _write_token(cache_root, "eps/cur.jpg", "coarse")
    ledger = _ledger_for_cache(cache_root)
    row = _tiny_row(images=("eps/absent.jpg",))
    before = sorted(str(p) for p in cache_root.rglob("*"))
    with pytest.raises(F2AssemblyContractError, match="F2_CACHE_MISS"):
        load_cached_observation(
            row,
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=ledger,
            history=1,
        )
    after = sorted(str(p) for p in cache_root.rglob("*"))
    assert before == after  # never recomputes or writes tokens online
    row = _tiny_row(current="eps/nofine.jpg", images=())
    with pytest.raises(F2AssemblyContractError, match="fine vision token"):
        load_cached_observation(
            row,
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=ledger,
            history=1,
        )


def test_load_cached_observation_rejects_bad_token_payloads(tmp_path):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    _write_token(
        cache_root, "eps/cur.jpg", "coarse", tensor=torch.zeros(4, 7)
    )
    row = _tiny_row(images=())
    with pytest.raises(F2AssemblyContractError, match="shape"):
        load_cached_observation(
            row,
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=_ledger_for_cache(cache_root),
            history=1,
        )
    _write_token(
        cache_root,
        "eps/cur.jpg",
        "coarse",
        tensor=torch.full((4, 1536), float("nan")),
    )
    with pytest.raises(F2AssemblyContractError, match="nonfinite"):
        load_cached_observation(
            row,
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=_ledger_for_cache(cache_root),
            history=1,
        )


def test_load_cached_observation_accepts_legacy_token_names(tmp_path):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    legacy_dir = cache_root / "eps"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        torch.randn(4, 1536, dtype=torch.float32).half(),
        legacy_dir / "cur_vcoarse.pt",
    )
    row = _tiny_row(images=())
    packet = load_cached_observation(
        row,
        base_root=tmp_path,
        cache_root=cache_root,
        token_ledger=_ledger_for_cache(cache_root),
        history=1,
    )
    assert packet.coarse_tokens.shape == (4, 1536)


def test_load_cached_observation_rejects_escaping_paths(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    row = _tiny_row(current="../outside.jpg", images=())
    with pytest.raises(F2AssemblyContractError, match="escapes"):
        load_cached_observation(
            row,
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=TokenHashLedger(entries={}),
            history=1,
        )


# --------------------------------------------------------------------------
# Block-major RunnerRow loader (blocker 9).
# --------------------------------------------------------------------------


def test_build_runner_rows_tiny_happy_path(tiny_assembly):
    rows, receipt, base_root, cache_root, ledger = tiny_assembly
    runner_rows = build_runner_rows(
        rows=rows,
        receipt=receipt,
        support_name="SMK-TRAIN",
        base_root=base_root,
        cache_root=cache_root,
        token_ledger=ledger,
    )
    assert len(runner_rows) == 6
    indices = [row.original_row_index for row in runner_rows]
    assert indices == [0, 1, 2, 3, 4, 5]
    for runner_row, (index, source) in zip(
        runner_rows, [(i, rows[i]) for i in indices]
    ):
        assert isinstance(runner_row, RunnerRow)
        assert isinstance(runner_row.observation, ObservationPacket)
        assert isinstance(runner_row.aux_targets, AuxTargetPacket)
        assert runner_row.sequence_id == source["sequence_id"]
        assert runner_row.frame_idx == source["frame_idx"]
        assert runner_row.target_actions.shape == (8, 3)
        assert torch.equal(
            runner_row.target_actions,
            torch.tensor(source["step_actions"], dtype=torch.float32),
        )
        assert runner_row.logged_prev_action == tuple(source["prev_action"])
        assert runner_row.observation.history_frames == 31


def test_tiny_reset_plan_matches_runner_predicates(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    strafe, expected = smoke_reset_sets(receipt, "SMK-TRAIN")
    assert strafe == frozenset()
    assert expected == frozenset({0, 3, 5})
    plan = support_reset_plan(rows, receipt, "SMK-TRAIN")
    assert plan[0] == ("stream_first",)
    assert plan[3] == ("sequence_discontinuity",)
    assert plan[5] == ("sequence_discontinuity",)
    assert all(not reasons for position, reasons in enumerate(plan)
               if position not in (0, 3, 5))


def test_tiny_strafe_reset_on_block_first(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    blocks = receipt.supports["SMK-TRAIN"]
    strafed = _tiny_receipt(rows, blocks, strafe_rows=(0,))
    strafe, expected = smoke_reset_sets(strafed, "SMK-TRAIN")
    assert strafe == frozenset({0})
    assert expected == frozenset({0, 3, 5})
    plan = support_reset_plan(rows, strafed, "SMK-TRAIN")
    assert plan[0] == ("stream_first", "strafe_reset")


def test_smoke_reset_sets_rejects_non_block_first_strafe(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    blocks = receipt.supports["SMK-TRAIN"]
    tampered = _tiny_receipt(rows, blocks, strafe_rows=(1,))
    with pytest.raises(F2AssemblyContractError, match="HS4"):
        smoke_reset_sets(tampered, "SMK-TRAIN")


def test_smoke_reset_sets_rejects_coverage_mismatch(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    coverage = receipt.coverage["SMK-TRAIN"]
    tampered_cov = replace(coverage, static_resets=coverage.static_resets + 1)
    tampered = replace(
        receipt,
        coverage={name: tampered_cov for name in receipt.coverage},
    )
    with pytest.raises(F2AssemblyContractError, match="static reset"):
        smoke_reset_sets(tampered, "SMK-TRAIN")


def test_support_reset_plan_detects_predicate_misalignment(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    coverage = receipt.coverage["SMK-TRAIN"]
    tampered = replace(
        receipt,
        base_reset_rows=tuple(sorted(set(receipt.base_reset_rows) | {1})),
        coverage={
            name: replace(coverage, static_resets=coverage.static_resets + 1)
            for name in receipt.coverage
        },
    )
    with pytest.raises(
        F2AssemblyContractError, match="RESET_PLAN_MISALIGNMENT"
    ):
        support_reset_plan(rows, tampered, "SMK-TRAIN")


def test_ordered_support_rows_fail_closed_paths(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    tampered = replace(
        receipt,
        row_sha256={name: "0" * 64 for name in receipt.row_sha256},
    )
    with pytest.raises(F2AssemblyContractError, match="row SHA"):
        ordered_support_rows(rows, tampered, "SMK-TRAIN")
    blocks = tuple(receipt.supports["SMK-TRAIN"])
    reordered = replace(
        receipt,
        supports={name: (blocks[1], blocks[0]) for name in receipt.supports},
    )
    with pytest.raises(F2AssemblyContractError, match="strictly increasing"):
        ordered_support_rows(rows, reordered, "SMK-TRAIN")
    with pytest.raises(F2AssemblyContractError, match="train row count"):
        ordered_support_rows(rows[:-1], receipt, "SMK-TRAIN")
    with pytest.raises(F2AssemblyContractError, match="unknown support"):
        ordered_support_rows(rows, receipt, "SMOKE")


def test_frozen_train_sha_hard_binds_support_expectations(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    impostor = replace(
        receipt,
        train_sha256=(
            "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36"
        ),
    )
    with pytest.raises(
        F2AssemblyContractError, match="SUPPORT_EXPECTATIONS"
    ):
        ordered_support_rows(rows, impostor, "SMK-TRAIN")


def test_build_runner_rows_rejects_bad_step_actions(tiny_assembly):
    rows, receipt, base_root, cache_root, ledger = tiny_assembly
    rows[0]["step_actions"] = rows[0]["step_actions"][:7]
    with pytest.raises(F2AssemblyContractError, match="exactly 8"):
        build_runner_rows(
            rows=rows,
            receipt=receipt,
            support_name="SMK-TRAIN",
            base_root=base_root,
            cache_root=cache_root,
            token_ledger=ledger,
        )
    rows[0]["step_actions"] = [[1.5, 0.0, 0.0]] * 8
    with pytest.raises(F2AssemblyContractError, match="action domain"):
        build_runner_rows(
            rows=rows,
            receipt=receipt,
            support_name="SMK-TRAIN",
            base_root=base_root,
            cache_root=cache_root,
            token_ledger=ledger,
        )


def test_eval_fix_strata_tiny_validates_against_coverage(tiny_assembly):
    rows, receipt, _base_root, _cache_root, _ledger = tiny_assembly
    strata = eval_fix_strata(rows, receipt, "EVAL-FIX")
    assert strata["counts"]["overall"] == 6
    assert strata["counts"]["turn"] == 1
    assert strata["counts"]["other"] == 5
    assert strata["row_order"] == (0, 1, 2, 3, 4, 5)
    coverage = receipt.coverage["EVAL-FIX"]
    tampered = replace(
        receipt,
        coverage={
            name: replace(coverage, h1_change=coverage.h1_change + 1)
            for name in receipt.coverage
        },
    )
    with pytest.raises(F2AssemblyContractError, match="stratum counts"):
        eval_fix_strata(rows, tampered, "EVAL-FIX")


# --------------------------------------------------------------------------
# Train-split token hash ledger (P1-3): byte verification + seal guarantee.
# --------------------------------------------------------------------------


def test_token_byte_tamper_rejected(tmp_path):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    _write_token(
        cache_root, "eps/cur.jpg", "coarse", tensor=torch.ones(4, 1536).half()
    )
    ledger = _ledger_for_cache(cache_root)
    # Same shape, different bytes: shape checks alone would accept this.
    _write_token(
        cache_root, "eps/cur.jpg", "coarse", tensor=torch.zeros(4, 1536).half()
    )
    with pytest.raises(F2AssemblyContractError, match="F2_CACHE_TAMPERED"):
        load_cached_observation(
            _tiny_row(images=()),
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=ledger,
            history=1,
        )
    ledger = _ledger_for_cache(cache_root)
    _write_token(
        cache_root, "eps/cur.jpg", "fine", tensor=torch.ones(64, 1536).half()
    )
    with pytest.raises(F2AssemblyContractError, match="F2_CACHE_TAMPERED"):
        load_cached_observation(
            _tiny_row(images=()),
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=ledger,
            history=1,
        )


def test_unlisted_tokens_are_never_opened(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    _write_token(cache_root, "eps/cur.jpg", "coarse")
    ledger = _ledger_for_cache(cache_root)
    canary = _write_token(
        cache_root, "data/collected_v1/episodes/test/leak.jpg", "coarse"
    )
    ghost = _write_token(cache_root, "eps/ghost.jpg", "coarse")
    opened: list[str] = []
    real_io_open = io.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    packet = load_cached_observation(
        _tiny_row(images=()),
        base_root=tmp_path,
        cache_root=cache_root,
        token_ledger=ledger,
        history=2,
    )
    assert packet.history_frames == 2
    with pytest.raises(F2AssemblyContractError, match="ledger"):
        load_cached_observation(
            _tiny_row(images=("eps/ghost.jpg",)),
            base_root=tmp_path,
            cache_root=cache_root,
            token_ledger=ledger,
            history=1,
        )
    assert str(canary) not in opened
    assert str(ghost) not in opened
    opened_tokens = {path for path in opened if path.endswith(".pt")}
    listed = {str(cache_root / key) for key in ledger.entries}
    assert opened_tokens <= listed


def test_build_token_ledger_for_rows(tmp_path):
    cache_root = tmp_path / "cache"
    _write_token(cache_root, "eps/cur.jpg", "fine")
    _write_token(cache_root, "eps/cur.jpg", "coarse")
    _write_token(cache_root, "eps/h0.jpg", "fine")
    _write_token(cache_root, "eps/h0.jpg", "coarse")
    rows = [_tiny_row(images=("eps/h0.jpg",))]
    ledger = build_token_ledger_for_rows(
        rows, base_root=tmp_path, cache_root=cache_root
    )
    assert ledger.token_files == 4
    assert set(ledger.entries) == {
        "eps/cur.jpg_vfine.pt",
        "eps/cur.jpg_vcoarse.pt",
        "eps/h0.jpg_vfine.pt",
        "eps/h0.jpg_vcoarse.pt",
    }
    assert ledger.ledger_sha256 == canonical_json_sha256(dict(ledger.entries))
    document = ledger.to_dict()
    assert document["analysis_class"] == "f2_train_token_hash_ledger"
    assert document["token_files"] == 4
    assert document["ledger_sha256"] == ledger.ledger_sha256
    (cache_root / "eps/h0.jpg_vfine.pt").unlink()
    with pytest.raises(F2AssemblyContractError, match="F2_CACHE_MISS"):
        build_token_ledger_for_rows(
            rows, base_root=tmp_path, cache_root=cache_root
        )


def test_token_ledger_validation_and_immutability():
    ledger = TokenHashLedger(entries={"a/b.pt": "0" * 64})
    with pytest.raises(TypeError):
        ledger.entries["x.pt"] = "0" * 64
    with pytest.raises(F2AssemblyContractError, match="sha256 hex"):
        TokenHashLedger(entries={"a.pt": "Z" * 64})
    with pytest.raises(F2AssemblyContractError, match="relative"):
        TokenHashLedger(entries={"../a.pt": "0" * 64})
    with pytest.raises(F2AssemblyContractError, match="relative"):
        TokenHashLedger(entries={"/abs/a.pt": "0" * 64})
    with pytest.raises(F2AssemblyContractError, match="TokenHashLedger"):
        load_cached_observation(
            _tiny_row(images=()),
            base_root=".",
            cache_root=".",
            token_ledger={"a.pt": "0" * 64},
            history=1,
        )


def test_build_train_token_ledger_fail_closed(tmp_path):
    with pytest.raises(F2AssemblyContractError, match="frozen train JSONL"):
        build_train_token_ledger(tmp_path)
    train = tmp_path / "data/collected_v1/datasets/train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text('{"not": "the frozen train"}\n', encoding="utf-8")
    with pytest.raises(F2AssemblyContractError, match="HS1"):
        build_train_token_ledger(tmp_path)


def test_real_train_images_never_touch_internal_test_subtree(real_support):
    rows, _receipt = real_support
    relpaths = collect_image_relpaths(rows, PROJECT_ROOT)
    assert len(relpaths) == 18_473
    for relpath in relpaths:
        assert not relpath.startswith(FROZEN_INTERNAL_TEST_IMAGE_PREFIX)
        assert any(
            relpath.startswith(prefix + "/")
            for prefix in FROZEN_TRAIN_IMAGE_PREFIXES
        )
