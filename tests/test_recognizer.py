"""Operator + digit classification tests. Operator rules run with no model
installed; digit tests need scripts/train_recognizer.py output."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from airmath.recognizer import Glyph, available, classify, predict_operator
from airmath.solver import solve


def _mask(draw):
    """draw: function(y, x) -> ink bool on a 40x60 grid."""
    ys, xs = np.mgrid[0:40, 0:60]
    return draw(ys, xs)


def _minus():
    m = np.zeros((40, 60), dtype=bool)
    m[18:22, 8:52] = True
    return Glyph(ink=m, x=0)


def _plus():
    m = np.zeros((44, 60), dtype=bool)
    m[4:40, 28:32] = True
    m[20:24, 6:54] = True
    return Glyph(ink=m, x=0)


def _equals():
    m = np.zeros((40, 60), dtype=bool)
    m[8:12, 10:50] = True
    m[26:30, 10:50] = True
    return Glyph(ink=m, x=0)


def _slash():
    m = np.zeros((40, 40), dtype=bool)
    for i in range(40):
        j = 39 - i  # corner-to-corner: '/' as a hand would write it
        m[i, max(0, j - 1) : min(40, j + 2)] = True
    return Glyph(ink=m, x=0)


def test_minus():
    assert predict_operator(_minus()) == "-"


def test_equals():
    assert predict_operator(_equals()) == "="


def test_plus():
    assert predict_operator(_plus()) == "+"


def test_slash_div():
    assert predict_operator(_slash()) == "/"


def test_digit_detected_when_model_missing_or_present():
    m = np.zeros((40, 60), dtype=bool)
    m[6:34, 24:36] = True  # fat vertical bar -> closest MNIST digit "1"
    tok = classify(Glyph(ink=m[6:34, 24:36], x=0))
    if not available():
        assert tok is None
    else:
        assert tok is not None


def test_end_to_end_written_glyphs():
    """Canvas -> glyphs -> Session (merge + flush) - the real app path."""
    from airmath.canvas import AirCanvas
    from airmath.app import Session

    c = AirCanvas(width=800, height=200, thickness=12)
    # '-' as a horizontal stroke
    c.start(60, 80)
    c.move(120, 80)
    c.finish()
    # '=' as two horizontal strokes at the same x, second lands later
    c.start(200, 70)
    c.move(260, 70)
    c.finish()
    c.start(200, 100)
    c.move(260, 100)
    c.finish()
    glyphs = c.glyph_queue()
    assert glyphs, "canvas produced no glyphs from straight-line strokes"

    s = Session(800, 200)
    s.handle_glyphs(sorted(glyphs, key=lambda g: g.x))
    toks = [t for t, _ in s.tokens]
    # '=' pinch fired a flush with a lone dash -> error recorded, buffer cleared
    assert s.result_kind == "error"
    assert "-" in s.result
    assert toks == []
