"""Render docs/banner.png - what the AirMath overlay looks like.

Replays real strokes through AirCanvas (same path a finger produces),
colourizes the ink, and composes a promo frame.  The classifier is also
run and reported to stdout, but the banner shows the ink trail, not a
webcam capture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

from airmath.canvas import AirCanvas
from airmath.recognizer import classify


def stroke(canvas: AirCanvas, pts):
    canvas.start(*pts[0])
    for p in pts[1:]:
        canvas.move(*p)
    canvas.finish()


def line_pt(x0, y0, x1, y1, n=34):
    return list(zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)))


def arc_dx(x0, x1, y0, amp=30, y1=None):
    """Digit-ish wiggle: two humps - good enough ink for the banner."""
    xs = np.linspace(x0, x1, 40)
    t = np.linspace(0, 1, 40)
    below = y1 if y1 is not None else y0
    ys = y0 + amp * np.sin(2 * np.pi * t) + (below - y0) * t
    return list(zip(xs, ys))


def main() -> int:
    W, H = 1100, 640
    canvas = AirCanvas(width=W, height=H, thickness=15)

    # "12+8=" left of x=760, then the answer chip rendered by hand
    stroke(canvas, line_pt(110, 420, 110, 300))            # '1'
    stroke(canvas, arc_dx(170, 260, 296, amp=24, y1=420))   # '2'
    stroke(canvas, line_pt(330, 355, 410, 355))             # '+' horizontal
    stroke(canvas, line_pt(370, 315, 370, 395))             # '+' vertical
    stroke(canvas, arc_dx(470, 540, 330, amp=0, y1=420))    # '8' lean vertical
    stroke(canvas, line_pt(600, 350, 680, 350))             # '=' bar 1
    stroke(canvas, line_pt(600, 390, 680, 390))             # '=' bar 2

    glyphs = [c for c in _drain(canvas)]
    tokens = [classify(g) for g in glyphs]
    print("banner glyphs classified as:", tokens)

    frame = np.full((H, W, 3), 10, dtype=np.uint8)
    frame[:, :, 0] = 14
    ink = canvas.image
    frame[ink > 0] = (0, 229, 160)  # mint
    frame = _glowify(frame, ink)
    frame = _charger(frame)  # answer chip
    frame = _chrome(frame)

    out = Path(__file__).resolve().parent.parent / "docs" / "banner.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), frame)
    print("wrote", out)
    return 0


def _drain(canvas: AirCanvas):
    got = canvas.glyph_queue()
    return got


def _glowify(frame, ink):
    ys, xs = np.nonzero(ink)
    layer = frame.copy()
    cv2.GaussianBlur(layer, (0, 0), 9, dst=layer)
    mask = np.zeros_like(ink, dtype=np.uint8)
    mask[ink > 0] = 255
    mask = cv2.dilate(mask, np.ones((23, 23), np.uint8))
    frame[mask > 0] = layer[mask > 0]
    return frame


def _charger(frame):
    import cv2 as _c

    _c.rectangle(frame, (770, 300), (1030, 430), (40, 70, 40), -1, _c.LINE_AA)
    _c.rectangle(frame, (770, 300), (1030, 430), (0, 229, 160), 2, _c.LINE_AA)
    _c.putText(frame, "= 20", (800, 390), _c.FONT_HERSHEY_SIMPLEX, 2.2, (0, 229, 160), 6, _c.LINE_AA)
    return frame


def _chrome(frame):
    import cv2 as _c

    _c.putText(frame, "AirMath", (60, 90), _c.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, _c.LINE_AA)
    _c.putText(frame, "write it in the air. the camera solves it.",
               (60, 135), _c.FONT_HERSHEY_SIMPLEX, 0.85, (150, 150, 150), 2, _c.LINE_AA)
    _c.putText(frame, "index=draw   open palm=clear   q=quit",
               (60, 600), _c.FONT_HERSHEY_SIMPLEX, 0.7, (110, 110, 110), 1, _c.LINE_AA)
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
