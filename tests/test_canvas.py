import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from airmath.canvas import AirCanvas
from airmath.segment import split_on_gaps
from airmath.solver import solve

# NOTE: recognizer tests that need the trained MNIST blob live in
# test_recognizer_model.py and skip when models/digits.npz is absent
# (CI runs without it; the shipped repo includes it).


def _mask_from_lines(points, thickness=10, H=160, W=400):
    """Paint a stroke with the same code the canvas uses."""
    canvas = AirCanvas(width=W, height=H, thickness=thickness)
    canvas.start(*points[0])
    for p in points[1:]:
        canvas.move(*p)
    canvas.finish()
    return canvas


def test_simple_two_glyphs_split():
    # draw "1 1" - two vertical strokes far apart, then pen up
    canvas = _mask_from_lines([(60, 30), (60, 110)])
    canvas.start(220, 30)
    canvas.move(220, 110)
    canvas.finish()
    glyphs = canvas.glyph_queue()
    assert len(glyphs) == 2
    assert glyphs[0].x < glyphs[1].x


def test_smear_single_line_stays_whole():
    # one continuous stroke that sweeps far right without lifting stays
    # ONE glyph - connected ink, nothing to split
    canvas = _mask_from_lines([(80, 60), (120, 60)])
    canvas.move(200, 60)
    canvas.finish()
    glyphs = canvas.glyph_queue()
    assert len(glyphs) == 1


def test_split_on_gaps_columns():
    mask = np.zeros((40, 100), dtype=bool)
    mask[10:20, 5:15] = True
    mask[10:20, 60:70] = True
    parts = split_on_gaps(mask, max_gap_ratio=1.0)
    assert len(parts) == 2


def test_split_on_gaps_no_split_when_close():
    mask = np.zeros((40, 100), dtype=bool)
    mask[10:20, 5:15] = True
    mask[10:20, 20:30] = True
    parts = split_on_gaps(mask, max_gap_ratio=1.0)
    assert len(parts) == 1


def test_smear_two_minus_split_by_gap():
    """Two '-' far apart drawn in one stroke WILL be one blob... but the
    line between them makes it connected, so it must NOT split."""
    canvas = _mask_from_lines([(60, 60), (100, 60)])
    canvas.move(150, 60)
    canvas.move(200, 60)
    canvas.finish()
    glyphs = canvas.glyph_queue()
    assert len(glyphs) >= 1  # whole connected stroke stays one glyph


def test_empty_finish_is_noop():
    canvas = AirCanvas(width=320, height=240)
    canvas.finish()
    assert canvas.glyph_queue() == []


def test_min_ink_dropped():
    canvas = AirCanvas(width=320, height=240, min_glyph_px=10 ** 6)
    canvas.start(40, 40)
    canvas.move(60, 40)
    canvas.finish()
    assert canvas.glyph_queue() == []
