from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_import(program: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_package_import_does_not_import_torch() -> None:
    result = _run_isolated_import(
        "import json, sys; import ibr1_experiment; "
        "print(json.dumps({'torch_loaded': 'torch' in sys.modules, "
        "'model_loaded': 'ibr1_experiment.model' in sys.modules}))"
    )

    assert result == {"torch_loaded": False, "model_loaded": False}


def test_public_model_symbol_is_loaded_lazily_and_cached() -> None:
    result = _run_isolated_import(
        "import json, sys; import ibr1_experiment as package; "
        "before = 'torch' in sys.modules; "
        "first = package.IBR1_FAMILY_ID; second = package.IBR1_FAMILY_ID; "
        "print(json.dumps({'before': before, 'after': 'torch' in sys.modules, "
        "'value': first, 'same_object': first is second}))"
    )

    assert result == {
        "before": False,
        "after": True,
        "value": "IBR1",
        "same_object": True,
    }
