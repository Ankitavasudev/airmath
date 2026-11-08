"""Digit-recognition sanity tests - run only when models/digits.npz exists."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402

from airmath.recognizer import Glyph, available, predict_digit  # noqa: E402

pytestmark = pytest.mark.skipif(not available(), reason="models/digits.npz not trained yet")


def _blob_digit(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[:, w // 2] = True
    return m


def test_vertical_bar_is_one():
    assert predict_digit(Glyph(ink=_blob_digit(30, 16), x=0)) == "1"


def test_two_bars_is_one_not_eleven():
    tok = predict_digit(Glyph(ink=_blob_digit(30, 16), x=0))
    assert tok.isdigit()
