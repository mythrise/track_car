#!/usr/bin/env python3
"""Per-wheel PWM trim: pure storage + math, no hardware/curses dependency.

This is the single source of truth for wheel-trim calibration, used by both
`car_hardware.py` (to apply the correction automatically on every real motor
command) and `wheel_trim_tuner.py` (the interactive calibration UI). It has
no import of `car_hardware` so that `car_hardware.py` can import from here
without a circular import.

The vendor PWM neutral/pulse-range constants below intentionally mirror
`car_hardware.NEUTRAL`/`MIN_PULSE`/`MAX_PULSE` (1500 / 500-2500us) — real
callers always pass `base=car_hardware.NEUTRAL` explicitly; the local
defaults just let this module be used standalone (e.g. in tests).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

NEUTRAL = 1500
MIN_PULSE = 500
MAX_PULSE = 2500

WHEEL_KEYS = ("l1", "r1", "l2", "r2")
TRIM_MIN = 0.50
TRIM_MAX = 1.50
DEFAULT_TRIM_PATH = Path(__file__).with_name("wheel_trim.json")


def default_trim() -> Dict[str, float]:
    return {key: 1.0 for key in WHEEL_KEYS}


def sanitize_trim(trim: dict | None) -> Dict[str, float]:
    """Coerce an arbitrary dict into four valid, clamped wheel-trim floats."""
    result = default_trim()
    if not trim:
        return result
    for key in WHEEL_KEYS:
        try:
            value = float(trim[key])
        except (KeyError, TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        result[key] = max(TRIM_MIN, min(TRIM_MAX, value))
    return result


def load_trim(path: str | Path = DEFAULT_TRIM_PATH) -> Dict[str, float]:
    """Load wheel trim multipliers, defaulting to 1.0 for any missing/invalid file."""
    path = Path(path)
    if not path.exists():
        return default_trim()
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return default_trim()
    if not isinstance(raw, dict):
        return default_trim()
    return sanitize_trim(raw)


def save_trim(trim: dict, path: str | Path = DEFAULT_TRIM_PATH) -> None:
    """Persist wheel trim multipliers as JSON, clamped and sorted for readability."""
    path = Path(path)
    payload: dict = dict(sanitize_trim(trim))
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _clamp_pulse(value: float, min_pulse: int = MIN_PULSE, max_pulse: int = MAX_PULSE) -> int:
    return max(min_pulse, min(max_pulse, int(round(value))))


def apply_trim(
    motors: Sequence[int],
    trim: dict | None = None,
    path: str | Path = DEFAULT_TRIM_PATH,
    base: int = NEUTRAL,
) -> List[int]:
    """Scale a final 4-channel PWM pulse command `[l1, r1, l2, r2]` by per-wheel trim.

    A trim of 1.0 is a no-op; >1.0 makes that wheel spin faster/harder, <1.0
    makes it spin slower/weaker, symmetrically in both forward and reverse
    (it scales the pulse's offset from neutral, not the raw pulse itself).
    """
    if trim is None:
        trim = load_trim(path)
    else:
        trim = sanitize_trim(trim)
    motors = [int(v) for v in motors]
    if len(motors) != 4:
        raise ValueError(f"expected 4 motor pulses [l1, r1, l2, r2], got {len(motors)}")
    return [_clamp_pulse(base + (motors[i] - base) * trim[key]) for i, key in enumerate(WHEEL_KEYS)]
