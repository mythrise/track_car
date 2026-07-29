import json
from pathlib import Path

import pytest

from scripts.materialize_hf_xsym_cache import (
    MaterializationError,
    materialize_snapshot,
)


def _fake_cache(tmp_path: Path) -> tuple[Path, str, dict[str, bytes]]:
    cache = tmp_path / "models--org--model"
    snapshot = "abc123"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text(snapshot, encoding="utf-8")
    snapshot_root = cache / "snapshots" / snapshot
    blobs = cache / "blobs"
    snapshot_root.mkdir(parents=True)
    blobs.mkdir()
    payloads = {
        "config.json": b'{"model_type":"fake"}\n',
        "model.safetensors": b"fake-safetensors",
    }
    for index, (name, payload) in enumerate(payloads.items()):
        blob = blobs / f"blob{index}"
        blob.write_bytes(payload)
        target = f"../../blobs/{blob.name}"
        (snapshot_root / name).write_text(
            f"XSym\n0000\nplaceholder\n{target}\n", encoding="utf-8"
        )
    return cache, snapshot, payloads


def test_materialize_xsym_snapshot(tmp_path):
    cache, snapshot, payloads = _fake_cache(tmp_path)
    output = tmp_path / "resolved"
    receipt = tmp_path / "receipt.json"
    result = materialize_snapshot(
        cache, output, receipt, snapshot=snapshot, mode="copy"
    )
    for name, payload in payloads.items():
        assert (output / name).read_bytes() == payload
    assert result["analysis_class"] == "windows_hf_xsym_materialization"
    assert result["internal_test_opened"] is False
    document = json.loads(receipt.read_text(encoding="utf-8"))
    assert document["artifact_sha256"] == result["artifact_sha256"]
    assert {item["source_kind"] for item in document["files"]} == {"xsym"}


def test_materialize_refuses_existing_destination(tmp_path):
    cache, snapshot, _payloads = _fake_cache(tmp_path)
    output = tmp_path / "resolved"
    output.mkdir()
    with pytest.raises(MaterializationError, match="already exists"):
        materialize_snapshot(
            cache, output, tmp_path / "receipt.json", snapshot=snapshot
        )


def test_materialize_rejects_xsym_target_outside_blobs(tmp_path):
    cache, snapshot, _payloads = _fake_cache(tmp_path)
    entry = cache / "snapshots" / snapshot / "config.json"
    outside = tmp_path / "outside"
    outside.write_bytes(b"bad")
    entry.write_text(
        "XSym\n0000\nplaceholder\n../../blobs/../../outside\n",
        encoding="utf-8",
    )
    with pytest.raises(MaterializationError, match="XSym target escapes"):
        materialize_snapshot(
            cache,
            tmp_path / "resolved",
            tmp_path / "receipt.json",
            snapshot=snapshot,
        )


def test_materialize_rejects_receipt_inside_output(tmp_path):
    cache, snapshot, _payloads = _fake_cache(tmp_path)
    output = tmp_path / "resolved"
    with pytest.raises(MaterializationError, match="receipt must be outside"):
        materialize_snapshot(
            cache,
            output,
            output / "receipt.json",
            snapshot=snapshot,
        )
