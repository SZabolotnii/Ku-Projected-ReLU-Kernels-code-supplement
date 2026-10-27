"""Timing session 2 (BRIEF A6, S1): the gate cells of spec.py again, on another day.

Same panels, seeds, ladders and rules; accuracy is recomputed, not copied. Run only
after session 1 finished, in a fresh process on a different day.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec import CONTROLS, cell  # noqa: E402,F401
import spec as _s1  # noqa: E402

RUN_ID = "kernel-gate-2026-10-25-s2"
GRID = [p for p in _s1.GRID if p["part"] == "gate"]
