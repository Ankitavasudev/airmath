"""Stroke bookkeeping: pen-down points in, glyphs out.

An "air-written" character is whatever happened on the canvas between two
pen-up events.  Very long glyphs (whole word smears) get split on x-axis
ink gaps with :mod:`airmath.segment`, which keeps hurried air-writing
usable.  The display image keeps every stroke drawn so far - clearing is
a separate explicit action (channel: raise an open palm).
"""

from __future__ import annotations

import numpy as np

from .recognizer import Glyph
from .segment import split_on_gaps

INK = 255


class AirCanvas:
    """Accumulates ink points and turns completed strokes into glyphs."""

    def __init__(self, width: int, height: int, thickness: int = 14, min_glyph_px: int = 40,
                 max_gap_to_split: float = 2.0):
        self.width = width
        self.height = height
        self.thickness = thickness
        self.min_glyph_px = min_glyph_px
        self.max_gap_to_split = max_gap_to_split
        self.clear()

    def clear(self) -> None:
        self.image = np.zeros((self.height, self.width), dtype=np.uint8)
        self.glyphs: list[tuple[np.ndarray, int, int]] = []  # (ink_mask, x, y)
        self._current: list[tuple[int, int]] = []

    # -- live drawing ------------------------------------------------------

    def start(self, x: int, y: int) -> None:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < self.width and 0 <= y < self.height:
            self._current = [(x, y)]

    def move(self, x: int, y: int) -> None:
        x, y = int(round(x)), int(round(y))
        if not self._current:
            self.start(x, y)
            return
        if 0 <= x < self.width and 0 <= y < self.height:
            px, py = self._current[-1]
            self._current.append((x, y))
            _line(self.image, px, py, x, y, self.thickness)

    def finish(self) -> None:
        """Pen up: everything stroked since the last finish becomes glyphs."""
        if not self._current:
            return
        xs = [p[0] for p in self._current]
        ys = [p[1] for p in self._current]
        self._current = []
        if not xs:
            return
        pad = self.thickness + 2
        x0 = max(0, min(xs) - pad)
        x1 = min(self.width - 1, max(xs) + pad)
        y0 = max(0, min(ys) - pad)
        y1 = min(self.height - 1, max(ys) + pad)
        if int(self.image[y0 : y1 + 1, x0 : x1 + 1].sum()) < self.min_glyph_px:
            return  # touched without drawing - drop it
        region = self.image[y0 : y1 + 1, x0 : x1 + 1]
        for sub in split_on_gaps(region.astype(bool), self.max_gap_to_split):
            if int(sub.sum()) >= self.min_glyph_px:
                self._stage(sub, x0, y0)

    def _stage(self, ink: np.ndarray, dx: int, dy: int) -> None:
        """Register a glyph from a boolean ink mask found at offset (dx,dy)."""
        mask = ink.astype(bool)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        self.glyphs.append((mask[y0 : y1 + 1, x0 : x1 + 1], x0 + dx, y0 + dy))

    # -- API for the app ----------------------------------------------------

    def glyph_queue(self) -> list[Glyph]:
        out = [Glyph(ink=ink, x=x, y=y) for ink, x, y in self.glyphs]
        self.glyphs = []
        return out


def _line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int) -> None:
    """One antialiased line segment; cv2 is imported lazily so the recognizer
    and solver stay testable without opencv installed."""
    import cv2

    cv2.line(img, (x0, y0), (x1, y1), INK, thickness, cv2.LINE_AA)
