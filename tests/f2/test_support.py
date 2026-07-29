import hashlib
import json

import pytest

from f2_experiment.support import (
    APPROVAL_BINDINGS,
    ARCHITECTURE_LOCK,
    BASE_RESET_COUNT,
    CANDIDATE_CAP,
    CONTRACT_COMPOSITION,
    FROZEN_TRAIN_ROWS,
    FROZEN_TRAIN_SHA256,
    FORMAL_RUNS,
    FORMAL_UPDATES_PER_ARM,
    SELECTION_N,
    SUPPORT_EXPECTATIONS,
    UNION_ROWS,
    UNION_SHA256,
    F2ContractError,
    SupportBlock,
    build_eligible_blocks,
    canonical_json_bytes,
    canonical_json_sha256,
    continues_sequence,
    derive_base_reset_rows,
    derive_strafe_ledger,
    measure_support_coverage,
    midpoint_stride_indices,
    select_support_blocks,
    support_row_indices,
)


def _row(
    *,
    source="episode0",
    sequence="sequence0",
    clip="clip0",
    frame=0,
    mirrored=False,
    prev=(0.0, 0.0, 0.0),
    first=(0.0, 0.0, 0.0),
    later=None,
    transition="steady_forward",
):
    horizon = [list(first) for _ in range(8)]
    if later is not None:
        horizon[-1] = list(later)
    return {
        "source_raw_dir": source,
        "sequence_id": sequence,
        "clip_id": clip,
        "frame_idx": frame,
        "mirrored": mirrored,
        "prev_action": list(prev),
        "step_actions": horizon,
        "transition_type": transition,
    }


def _block(index, sequence=None):
    sequence_id = sequence or f"sequence{index}"
    return SupportBlock(
        key=("episode", sequence_id, f"clip{index}"),
        source_raw_dir="episode",
        sequence_id=sequence_id,
        clip_id=f"clip{index}",
        mirrored=bool(index % 2),
        row_indices=(index * 32,),
    )


def test_fable_contract_bindings_are_explicit_and_primary():
    assert ARCHITECTURE_LOCK == "L1+D2+AP2+F2"
    assert CONTRACT_COMPOSITION["corrigendum2"] == (
        "apply every non-support_registry key"
    )
    assert CONTRACT_COMPOSITION["corrigendum3"] == (
        "replace support_registry in full"
    )
    assert CONTRACT_COMPOSITION["void_corrigendum2_support_fields"] == [
        "is_mirrored",
        "expected_pool_n=236",
    ]
    assert CANDIDATE_CAP == 1
    assert FORMAL_RUNS == 9
    assert FORMAL_UPDATES_PER_ARM == 6_873
    assert FROZEN_TRAIN_ROWS == 13_746
    assert FROZEN_TRAIN_SHA256 == (
        "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36"
    )
    binding_ids = {binding.binding_id for binding in APPROVAL_BINDINGS}
    assert {
        "fable_architecture_primary_result",
        "fable_implementation_primary_result",
        "fable_corrigendum2_registry_patch",
        "fable_corrigendum2_verdict",
        "fable_corrigendum3_support_patch",
        "fable_corrigendum3_verdict",
    } == binding_ids
    assert all(len(binding.sha256) == 64 for binding in APPROVAL_BINDINGS)


def test_canonical_json_uses_compact_sorted_utf8_bytes():
    value = {"z": [3, 2, 1], "a": "辅助"}
    expected = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert canonical_json_bytes(value) == expected
    assert canonical_json_sha256(value) == hashlib.sha256(expected).hexdigest()


def test_corrected_continuity_allows_mirrored_internal_steps_only():
    mirrored_0 = _row(sequence="seq__mirror", frame=0, mirrored=True)
    mirrored_1 = _row(sequence="seq__mirror", frame=1, mirrored=True)
    assert continues_sequence(mirrored_0, mirrored_1)
    assert not continues_sequence(
        mirrored_0,
        _row(sequence="seq__mirror", frame=1, mirrored=False),
    )
    assert not continues_sequence(
        mirrored_0,
        _row(sequence="seq__mirror", frame=2, mirrored=True),
    )
    assert not continues_sequence(
        mirrored_0,
        _row(sequence="other", frame=1, mirrored=True),
    )


def test_base_reset_rows_use_original_stream_order_and_mirror_identity():
    rows = [
        _row(sequence="a", frame=0, mirrored=False),
        _row(sequence="a", frame=1, mirrored=False),
        _row(sequence="a__mirror", frame=0, mirrored=True),
        _row(sequence="a__mirror", frame=1, mirrored=True),
        _row(sequence="a__mirror", frame=3, mirrored=True),
    ]
    assert derive_base_reset_rows(rows) == (0, 2, 4)


def test_eligible_blocks_use_corrigendum3_group_key_and_original_row_tiebreak():
    rows = [
        _row(frame=31 - index, mirrored=True)
        for index in range(32)
    ]
    blocks = build_eligible_blocks(rows)
    assert len(blocks) == 1
    assert blocks[0].key == ("episode0", "sequence0", "clip0")
    assert blocks[0].mirrored is True
    assert blocks[0].row_indices == tuple(reversed(range(32)))


def test_eligible_group_with_mixed_mirrored_values_fails_closed():
    rows = [
        _row(frame=index, mirrored=(index == 31))
        for index in range(32)
    ]
    with pytest.raises(F2ContractError, match="mixed mirrored"):
        build_eligible_blocks(rows)


def test_missing_exact_mirrored_field_fails_closed():
    rows = [_row(frame=index) for index in range(32)]
    for row in rows:
        row["is_mirrored"] = row.pop("mirrored")
    with pytest.raises(F2ContractError, match="invalid mirrored"):
        build_eligible_blocks(rows)


def test_eligibility_boundary_is_32_and_33_takes_only_first_32():
    rows_31 = [_row(frame=index) for index in range(31)]
    assert build_eligible_blocks(rows_31) == ()
    rows_33 = [_row(frame=index) for index in range(33)]
    block = build_eligible_blocks(rows_33)[0]
    assert block.row_indices == tuple(range(32))


def test_midpoint_stride_is_integer_exact_and_binds_n173():
    indices = midpoint_stride_indices(SELECTION_N, 40)
    assert indices[:4] == (2, 6, 10, 15)
    assert indices[-1] == 170
    assert len(indices) == len(set(indices)) == 40


def test_raw_mirror_sequence_suffix_is_not_normalized_for_dedup():
    blocks = (
        _block(0, "seq"),
        _block(1, "seq__mirror"),
        _block(2, "other2"),
        _block(3, "other3"),
        _block(4, "other4"),
    )
    selected = select_support_blocks(blocks, slots=5)
    sequences = {block.sequence_id for values in selected.values() for block in values}
    assert "seq" in sequences
    assert "seq__mirror" in sequences


def test_selection_is_globally_disjoint_and_sequence_unique_within_support():
    blocks = (
        _block(0, "shared"),
        _block(1, "unique1"),
        _block(2, "shared"),
        _block(3, "unique3"),
        _block(4, "unique4"),
        _block(5, "unique5"),
        _block(6, "unique6"),
        _block(7, "unique7"),
    )
    selected = select_support_blocks(blocks, slots=5)
    all_blocks = [block for values in selected.values() for block in values]
    assert len({block.key for block in all_blocks}) == 5
    for values in selected.values():
        assert len({block.sequence_id for block in values}) == len(values)


def test_strafe_ledger_separates_loss_audit_from_recurrence_unsupported():
    rows = [
        _row(frame=0, prev=(0.0, 0.1, 0.0)),
        _row(frame=1),
        _row(frame=2, later=(0.0, 0.2, 0.0)),
        _row(frame=3),
    ]
    ledger = derive_strafe_ledger(rows)
    assert ledger.audit_24 == (0, 2)
    assert ledger.unsupported_10 == (0,)
    assert ledger.reset_boundary_12 == (0, 1)
    assert ledger.diagnostic_superset_26 == (0, 1, 2, 3)


def test_support_coverage_uses_frozen_change_turn_other_predicates():
    rows = [
        _row(frame=0, first=(0.3, 0.0, 0.0), transition="turn_onset"),
        _row(frame=1, transition="other"),
    ]
    block = SupportBlock(
        key=("episode0", "sequence0", "clip0"),
        source_raw_dir="episode0",
        sequence_id="sequence0",
        clip_id="clip0",
        mirrored=False,
        row_indices=(0, 1),
    )
    coverage = measure_support_coverage(rows, (block,), base_reset_rows=(0,))
    assert coverage.to_dict() == {
        "blocks": 1,
        "rows": 2,
        "h1_change": 1,
        "turn": 1,
        "other": 1,
        "unique_sequence": 1,
        "episode": 1,
        "mirror": 0,
        "static_resets": 1,
    }
    assert support_row_indices((block,)) == (0, 1)


def test_support_coverage_counts_each_block_first_row_as_stream_reset():
    rows = [
        _row(frame=0),
        _row(frame=1),
    ]
    block = SupportBlock(
        key=("episode0", "sequence0", "clip0"),
        source_raw_dir="episode0",
        sequence_id="sequence0",
        clip_id="clip0",
        mirrored=False,
        row_indices=(1,),
    )

    coverage = measure_support_coverage(rows, (block,), base_reset_rows=(0,))

    assert coverage.static_resets == 1


def test_h1_change_is_strict_and_ignores_strafe_axis():
    rows = [
        _row(frame=0, first=(0.2, 0.9, 0.0), transition="turn"),
        _row(frame=1, first=(0.0, 0.0, 0.200001), transition="turn_exit"),
    ]
    block = SupportBlock(
        key=("episode0", "sequence0", "clip0"),
        source_raw_dir="episode0",
        sequence_id="sequence0",
        clip_id="clip0",
        mirrored=False,
        row_indices=(0, 1),
    )
    coverage = measure_support_coverage(rows, (block,), base_reset_rows=(0,))
    assert coverage.h1_change == 1
    assert coverage.turn == 1


def test_frozen_support_receipt_constants_match_corrigendum3():
    assert BASE_RESET_COUNT == 290
    assert SELECTION_N == 173
    assert {name: value.blocks for name, value in SUPPORT_EXPECTATIONS.items()} == {
        "CAL": 16,
        "SMK-TRAIN": 8,
        "EVAL-FIX": 16,
    }
    assert {name: value.rows for name, value in SUPPORT_EXPECTATIONS.items()} == {
        "CAL": 512,
        "SMK-TRAIN": 256,
        "EVAL-FIX": 512,
    }
    assert UNION_ROWS == 1_280
    assert UNION_SHA256 == (
        "906f990a34ed9bcc6c852f7295293b467628ab56c465d41450d4ae9715aa19be"
    )
