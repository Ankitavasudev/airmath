"""Headless app smoke test: run_over_frame with a fake hand state."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from airmath.app import Session, run_over_frame
from airmath.hand_tracker import HandState


def _frame(w=640, h=360):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_and_solve_pipeline():
    s = Session(640, 360)

    # 'pen down' drag right (writes a '-')
    run_over_frame(_frame(), s, HandState(x=100, y=100, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=200, y=100, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=210, y=100, pen=False, wipe=False))  # pen up

    # two horizontal strokes = '='
    run_over_frame(_frame(), s, HandState(x=300, y=80, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=380, y=80, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=385, y=80, pen=False, wipe=False))
    run_over_frame(_frame(), s, HandState(x=300, y=110, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=380, y=110, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=385, y=110, pen=False, wipe=False))

    # '=' triggers a flush: "-" alone is not a solvable expression, so the
    # session records the error and empties its token buffer
    assert s.result_kind == "error"
    assert s.tokens == []


def test_wipe_clears_everything():
    s = Session(640, 360)
    run_over_frame(_frame(), s, HandState(x=100, y=100, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=110, y=100, pen=True, wipe=False))
    run_over_frame(_frame(), s, HandState(x=0, y=0, pen=False, wipe=True))
    assert s.canvas.image.sum() == 0
    assert s.tokens == []


def test_no_hand_is_noop():
    s = Session(640, 360)
    out = run_over_frame(_frame(), s, None)
    assert out.shape == (360, 640, 3)
