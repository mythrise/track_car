#!/usr/bin/env python3
"""Interactive per-wheel trim tuner for straight-line drift correction.

Background: `car_hardware.py` commands all four wheels at the same speed for
a straight drive (`move_forward`, `action_to_wheel_speeds` with
forward=1/strafe=0/yaw=0, `waypoint_to_motor`, ...). On the real car this
still drifts left or right because the four motors are not perfectly
matched.

The correction lives in two places:

  1. `wheel_trim.py` -- pure per-wheel trim storage (`wheel_trim.json`) and
     math (`apply_trim`). `CarHardware.run_raw`/`run_speeds` in
     `car_hardware.py` call `apply_trim()` automatically on every send, so
     every script in this repo (`pi_client.py`, `move_test.py`,
     `speed_sweep.py`, `data_pipeline/collect_data.py`, and anything driven
     by `inference_pipeline/mac_server.py`) gets the correction for free --
     no per-script changes needed.
  2. This file -- an interactive curses terminal UI (works fine over plain
     SSH to a headless Raspberry Pi, no X11/display needed) to find those
     per-wheel trim values in the first place, and to re-tune them later.

Run (dry-run, safe, prints what would be sent but never touches UART):
    python3 car_runtime/wheel_trim_tuner.py

Run for real (moves the car -- lift it or clear the floor first):
    python3 car_runtime/wheel_trim_tuner.py --execute

Key bindings (shown in the UI footer too):
    Up/Down     select a row (4 wheels + base test speed)
    Left/Right  fine step  (trim +-0.01, speed +-10)
    PgUp/PgDn   coarse step (trim +-0.05, speed +-50)
    f / b       drive all four wheels forward / backward for a short test
    t           spin ONLY the selected wheel (diagnostic + isolated trim check)
    space / x   stop immediately
    r           reset all trims to 1.0 (not saved until you press w)
    w           save current trims to the trim file
    l           reload trims from the trim file (discard unsaved edits)
    q           quit (asks for confirmation if there are unsaved edits)

While tuning, test drives (f/b/t) push the live, possibly-unsaved slider
values into the `CarHardware` instance via `set_trim()` before each send, so
you feel the effect of a slider move immediately, before deciding to save.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path
from typing import List, Sequence

try:
    import curses
except ImportError:  # pragma: no cover - curses is unavailable on some platforms (e.g. Windows)
    curses = None

try:
    from car_hardware import DEFAULT_BASE_SPEED, MAX_SPEED, NEUTRAL, CarHardware, speed_to_pwm
    from process_cleanup import cleanup_named_processes
    from wheel_trim import DEFAULT_TRIM_PATH, TRIM_MIN, TRIM_MAX, WHEEL_KEYS, apply_trim, default_trim, load_trim, save_trim
except ImportError:
    from car_runtime.car_hardware import DEFAULT_BASE_SPEED, MAX_SPEED, NEUTRAL, CarHardware, speed_to_pwm
    from car_runtime.process_cleanup import cleanup_named_processes
    from car_runtime.wheel_trim import (
        DEFAULT_TRIM_PATH,
        TRIM_MIN,
        TRIM_MAX,
        WHEEL_KEYS,
        apply_trim,
        default_trim,
        load_trim,
        save_trim,
    )


WHEEL_LABELS = {
    "l1": "L1 (ch6, best-guess front-left)",
    "r1": "R1 (ch7, best-guess front-right)",
    "l2": "L2 (ch8, best-guess rear-left)",
    "r2": "R2 (ch9, best-guess rear-right)",
}
ROWS = WHEEL_KEYS + ("speed",)


def clamp_test_speed(value: float) -> int:
    return max(0, min(MAX_SPEED, int(round(value))))


# --------------------------------------------------------------------------
# Interactive curses tuner (requires real hardware access; run on the Pi)
# --------------------------------------------------------------------------


class TunerState:
    def __init__(self, trim_path: Path, speed: int, step: float, coarse_step: float,
                 speed_step: int, speed_coarse_step: int):
        self.trim_path = trim_path
        self.trim = load_trim(trim_path)
        self.saved_trim = dict(self.trim)
        self.speed = clamp_test_speed(speed)
        self.step = step
        self.coarse_step = coarse_step
        self.speed_step = speed_step
        self.speed_coarse_step = speed_coarse_step
        self.selected = 0
        self.log = "ready"

    @property
    def dirty(self) -> bool:
        return self.trim != self.saved_trim

    def move_selection(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(ROWS)

    def adjust(self, coarse: bool, sign: int) -> None:
        row = ROWS[self.selected]
        if row == "speed":
            step = self.speed_coarse_step if coarse else self.speed_step
            self.speed = clamp_test_speed(self.speed + sign * step)
        else:
            step = self.coarse_step if coarse else self.step
            value = self.trim[row] + sign * step
            self.trim[row] = max(TRIM_MIN, min(TRIM_MAX, value))

    def reset(self) -> None:
        self.trim = default_trim()
        self.log = "reset all trims to 1.0 (not saved yet, press w to save)"

    def save(self) -> None:
        save_trim(self.trim, self.trim_path)
        self.saved_trim = dict(self.trim)
        self.log = f"saved to {self.trim_path}"

    def reload(self) -> None:
        self.trim = load_trim(self.trim_path)
        self.saved_trim = dict(self.trim)
        self.log = f"reloaded from {self.trim_path}"

    def forward_raw_pulses(self) -> List[int]:
        return speed_to_pwm(self.speed, self.speed, self.speed, self.speed)

    def backward_raw_pulses(self) -> List[int]:
        return speed_to_pwm(-self.speed, -self.speed, -self.speed, -self.speed)

    def isolated_raw_pulses(self, key: str) -> List[int]:
        pulses = [NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL]
        idx = WHEEL_KEYS.index(key)
        pulses[idx] = self.forward_raw_pulses()[idx]
        return pulses

    def forward_pulses(self) -> List[int]:
        """Trimmed forward pulses, for on-screen preview only."""
        return apply_trim(self.forward_raw_pulses(), trim=self.trim, base=NEUTRAL)


def _capture_stdout(fn) -> str:
    """Run fn() with stdout redirected, since CarHardware/UartTransport print
    directly to stdout in dry-run mode -- writing to the real terminal while
    curses owns the screen would corrupt the display."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue().strip()


def run_test_drive(hardware: "CarHardware | None", raw_pulses: Sequence[int], duration: float,
                    label: str, state: TunerState) -> None:
    """Send `raw_pulses` (untrimmed) through `hardware`, which applies the
    live `state.trim` itself via `set_trim()` -- trim must be applied exactly
    once, by CarHardware, or a pre-trimmed array here would be trimmed twice."""
    trimmed_preview = apply_trim(raw_pulses, trim=state.trim, base=NEUTRAL)
    prefix = f"{label} test: raw={list(raw_pulses)} trimmed={trimmed_preview} duration={duration:.2f}s"
    if hardware is None:
        state.log = f"{prefix} (no hardware)"
        return

    def _do() -> None:
        hardware.set_trim(state.trim)
        hardware.run_raw(*raw_pulses)
        time.sleep(duration)
        hardware.stop()

    captured = _capture_stdout(_do)
    last_line = captured.splitlines()[-1] if captured else ""
    state.log = f"{prefix}{' -- ' + last_line if last_line else ''}"


def _clip(text: str, width: int) -> str:
    return text[: max(0, width)]


def draw(stdscr, state: TunerState, args) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    def put(y: int, x: int, text: str, attr: int = 0) -> None:
        if 0 <= y < height and x < width:
            try:
                stdscr.addstr(y, x, _clip(text, width - x - 1), attr)
            except curses.error:
                pass

    mode = "EXECUTE -- REAL UART" if args.execute else "dry-run (no hardware output)"
    dirty_mark = "*" if state.dirty else ""
    put(0, 0, f"Wheel Trim Tuner -- {mode}", curses.A_BOLD)
    put(1, 0, f"uart={args.uart_port or 'auto'}  trim_file={state.trim_path}{dirty_mark}")

    y = 3
    for i, key in enumerate(WHEEL_KEYS):
        value = state.trim[key]
        bar_width = 24
        filled = int(round((value - TRIM_MIN) / (TRIM_MAX - TRIM_MIN) * bar_width))
        filled = max(0, min(bar_width, filled))
        bar = "#" * filled + "-" * (bar_width - filled)
        marker = ">" if state.selected == i else " "
        attr = curses.A_REVERSE if state.selected == i else 0
        put(y, 0, f"{marker} {WHEEL_LABELS[key]:<34} [{bar}] {value:5.3f} ({value * 100:5.1f}%)", attr)
        y += 1

    y += 1
    speed_row = len(WHEEL_KEYS)
    marker = ">" if state.selected == speed_row else " "
    attr = curses.A_REVERSE if state.selected == speed_row else 0
    put(y, 0, f"{marker} base test speed{'':<27} {state.speed:>5d} / {MAX_SPEED}", attr)

    y += 2
    put(y, 0, f"forward preview (trimmed):   {state.forward_pulses()}")
    y += 1
    put(y, 0, f"forward preview (untrimmed): {state.forward_raw_pulses()}")

    y += 2
    put(y, 0, "Up/Down select    Left/Right +-step (trim 0.01 / speed 10)")
    y += 1
    put(y, 0, "PgUp/PgDn +-coarse (trim 0.05 / speed 50)")
    y += 1
    put(y, 0, "f forward-test   b backward-test   t spin-selected-wheel-only   space/x stop")
    y += 1
    put(y, 0, "r reset-all   w save   l reload   q quit")

    y += 2
    put(y, 0, f"log: {state.log}")
    stdscr.refresh()


def curses_main(stdscr, args, hardware: "CarHardware | None") -> None:
    curses.curs_set(0)
    stdscr.keypad(True)

    state = TunerState(
        Path(args.trim_file),
        args.speed,
        args.step,
        args.coarse_step,
        args.speed_step,
        args.speed_coarse_step,
    )
    duration = max(0.1, min(args.test_duration, 3.0))
    quit_confirm_pending = False

    while True:
        draw(stdscr, state, args)
        ch = stdscr.getch()

        if quit_confirm_pending:
            quit_confirm_pending = False
            if ch in (ord("q"), ord("Q")):
                return
            state.log = "quit cancelled"
            continue

        if ch in (curses.KEY_UP,):
            state.move_selection(-1)
        elif ch in (curses.KEY_DOWN,):
            state.move_selection(1)
        elif ch == curses.KEY_LEFT:
            state.adjust(coarse=False, sign=-1)
        elif ch == curses.KEY_RIGHT:
            state.adjust(coarse=False, sign=1)
        elif ch == curses.KEY_PPAGE:
            state.adjust(coarse=True, sign=1)
        elif ch == curses.KEY_NPAGE:
            state.adjust(coarse=True, sign=-1)
        elif ch in (ord("r"), ord("R")):
            state.reset()
        elif ch in (ord("w"), ord("W")):
            state.save()
        elif ch in (ord("l"), ord("L")):
            state.reload()
        elif ch in (ord("f"), ord("F")):
            run_test_drive(hardware, state.forward_raw_pulses(), duration, "forward", state)
        elif ch in (ord("b"), ord("B")):
            run_test_drive(hardware, state.backward_raw_pulses(), duration, "backward", state)
        elif ch in (ord("t"), ord("T")):
            row = ROWS[state.selected]
            if row == "speed":
                state.log = "select a wheel row first ('t' spins only the selected wheel)"
            else:
                run_test_drive(hardware, state.isolated_raw_pulses(row), duration, f"isolated {row}", state)
        elif ch in (ord(" "), ord("x"), ord("X")):
            if hardware is not None:
                captured = _capture_stdout(hardware.stop)
                state.log = "stopped" + (f" -- {captured.splitlines()[-1]}" if captured else "")
            else:
                state.log = "stopped (no hardware)"
        elif ch in (ord("q"), ord("Q")):
            if state.dirty:
                state.log = "unsaved changes -- press w to save, or q again to quit without saving"
                quit_confirm_pending = True
            else:
                return


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uart_port", default=None, help="UART device, for example /dev/ttyAMA0 or /dev/serial0.")
    ap.add_argument("--trim_file", default=str(DEFAULT_TRIM_PATH), help="Path to the wheel trim JSON file.")
    ap.add_argument("--speed", type=int, default=DEFAULT_BASE_SPEED, help="Initial base test wheel speed.")
    ap.add_argument("--step", type=float, default=0.01, help="Fine trim adjustment step (Left/Right).")
    ap.add_argument("--coarse_step", type=float, default=0.05, help="Coarse trim adjustment step (PgUp/PgDn).")
    ap.add_argument("--speed_step", type=int, default=10, help="Fine base-speed adjustment step (Left/Right).")
    ap.add_argument("--speed_coarse_step", type=int, default=50, help="Coarse base-speed step (PgUp/PgDn).")
    ap.add_argument("--test_duration", type=float, default=0.6,
                     help="Seconds to run each f/b/t test drive, clamped to 3.0.")
    ap.add_argument("--execute", action="store_true", help="Actually send UART commands. Default is dry-run.")
    ap.add_argument("--no_cleanup_processes", action="store_true",
                     help="Do not kill vendor camera/main processes before opening hardware.")
    ap.add_argument("--cleanup_dry_run", action="store_true",
                     help="Print cleanup targets without killing them.")
    return ap.parse_args()


def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    args = parse_args()

    if not args.no_cleanup_processes:
        cleanup_named_processes(["mjpg", "z_main"], dry_run=args.cleanup_dry_run)
        if args.cleanup_dry_run:
            return

    if curses is None:
        raise SystemExit(
            "The curses module is unavailable on this platform. Run this tool on the "
            "Raspberry Pi (Linux) over SSH, or any POSIX terminal."
        )

    if not sys.stdin.isatty():
        raise SystemExit("wheel_trim_tuner needs an interactive terminal (stdin is not a tty).")

    if args.execute and not confirm("REAL HARDWARE: lift the car or clear the floor. Continue?"):
        print("[wheel_trim_tuner] aborted")
        return

    hardware = CarHardware(uart_port=args.uart_port, dry_run=not args.execute, trim_path=args.trim_file)
    try:
        curses.wrapper(curses_main, args, hardware)
    finally:
        hardware.close()  # close() stops the motors first
        print(f"[wheel_trim_tuner] stopped. trim file: {args.trim_file}")


if __name__ == "__main__":
    main()
