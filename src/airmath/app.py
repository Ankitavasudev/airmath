"""Main AirMath loop: camera in, drawing + recognition + solution out.

The pipeline is one function, run_over_frame(frame, state), so headless
tests (no webcam, no mediapipe) can drive the exact same code path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import fx
from .canvas import AirCanvas
from .hand_tracker import HandState, HandTracker
from .recognizer import classify
from .solver import ExpressionError, solve

RESULT_FONT_COLORS = {
    "ok": (0, 229, 160),      # mint
    "error": (0, 80, 255),    # red-ish
}
UI_COLOR = (120, 184, 255)
INK_COLOR = (60, 180, 255)  # drawn trail on the frame - bgr


class Session:
    """State threaded between frames: tokens collected, last result, FX."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.canvas = AirCanvas(width=width, height=height)
        self.tokens: list[tuple[str, int]] = []  # (token, x)
        self.result: str = ""
        self.result_kind: str = "ok"
        self.history: list[tuple[str, str]] = []  # (expr, answer)
        # fx state
        self.tick = 0
        self.trail: list[tuple[int, int]] = []
        self.chips: list[fx.FloatChip] = []
        self.particles: list[fx.Particle] = []
        self.scans: list[fx.Scan] = []
        self.mood: str = "idle"
        self.mood_until = 0

    def _celebrate(self, ok: bool) -> None:
        y = self.height - 70
        if ok:
            fx.star_burst(self.particles, x=210, y=y, count=34)
            self.mood, self.mood_until = "ok", self.tick + 170
        else:
            fx.float_chip(self.chips, "oops!", 210, y, color=fx.RED)
            self.mood, self.mood_until = "err", self.tick + 90
        if self.canvas.image.any():
            ys, xs = np.nonzero(self.canvas.image)
            self.scans.append(fx.Scan(int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()), tick0=self.tick))

    def handle_glyphs(self, glyphs) -> None:
        for g in glyphs:
            token = classify(g)
            if token is None:
                continue
            self.tokens.append((token, g.x))
            self.tokens.sort(key=lambda t: t[1])
            self._merge_equals_pair()
            if self.tokens and self.tokens[-1][0] == "=":
                self._flush()
                self.tokens = []

    def _merge_equals_pair(self) -> None:
        """Air-drawn glyphs come out as strokes: '=' is two '-' at the same
        x, and '+' is a '-' followed by a vertical bar ('1') at the same x.
        Pinch each pattern into the operator it represents."""
        if len(self.tokens) < 2:
            return
        (t1, x1), (t2, x2) = self.tokens[-2], self.tokens[-1]
        if t1 == "-" and t2 == "-" and abs(x1 - x2) <= 60:
            self.tokens[-2:] = [("=", max(x1, x2))]
        elif t1 == "-" and t2 == "1" and -30 <= (x2 - x1) <= 25:
            self.tokens[-2:] = [("+", min(x1, x2))]

    def _flush(self) -> None:
        if not self.tokens:
            return
        tok = [t for t, _ in self.tokens]
        try:
            value = solve(tok)
            self.result = "%s = %d" % (" ".join(tok[:-1]), value)
            self.result_kind = "ok"
            self.history.append((" ".join(tok[:-1]), str(value)))
            self._celebrate(ok=True)
        except ExpressionError as exc:
            self.result = " ".join(tok[:-1]) + " -> " + str(exc)
            self.result_kind = "error"
            self._celebrate(ok=False)

    def wipe(self) -> None:
        self.canvas.clear()
        self.tokens = []
        self.result = ""


def run_over_frame(frame_bgr: np.ndarray, session: Session, hand: HandState | None) -> np.ndarray:
    """Draw/display one frame. Returns the annotated frame, FX advanced."""
    session.tick += 1
    if hand is not None:
        if hand.wipe:
            session.canvas.clear()
            session.tokens = []
            session.result = ""
        elif hand.pen:
            session.canvas.move(hand.x, hand.y)
        else:
            session.canvas.finish()
        session.trail.append((hand.x, hand.y))
        if len(session.trail) > 20:
            del session.trail[:-18]
    else:
        session.canvas.finish()

    glyphs = session.canvas.glyph_queue()
    if glyphs:
        session.handle_glyphs(glyphs)

    fx.step(session.chips, session.particles)
    fx.step_scans(session.scans, session.tick)
    if session.mood_until < session.tick:
        session.mood = "idle"

    frame = frame_bgr.copy()
    ink = session.canvas.image
    frame[ink > 0] = INK_COLOR

    for scan in session.scans:
        fx.draw_scan(frame, scan, session.tick)
    if hand is not None:
        fx.draw_trail(frame, session.trail[-18:], hand.pen)

    if glyphs:
        for g in glyphs:
            tok = classify(g)
            if tok:
                fx.float_chip(session.chips, tok, g.x + 12, g.y - 8)

    fx.draw_chips(frame, session.chips)
    fx.draw_particles(frame, session.particles)
    if hand is not None:
        fx.draw_cursor(frame, hand.x, hand.y, hand.pen, session.tick)
        fx.draw_mascot(frame, frame.shape[1] - 70, 56, session.mood, session.tick)

    _hud(frame, session, hand)
    return frame


def _hud(frame: np.ndarray, session: Session, hand: HandState | None) -> None:
    import cv2

    h = frame.shape[0]
    mode = "PEN DOWN" if (hand and hand.pen) else ("WIPE" if (hand and hand.wipe) else "pen up")
    cv2.putText(frame, mode, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_COLOR, 1, cv2.LINE_AA)
    written = " ".join(t for t, _ in session.tokens)
    if written:
        cv2.putText(frame, "you wrote: " + written, (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_COLOR, 1, cv2.LINE_AA)
    if session.result:
        color = RESULT_FONT_COLORS[session.result_kind]
        cv2.putText(frame, session.result, (14, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, "index=draw  |  open palm=clear  |  q=quit",
                (14, h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="airmath", description="Write math in the air, watch it get solved.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args(argv)

    try:
        import mediapipe  # noqa: F401 - fail early, before the camera opens
    except ImportError:
        print("AirMath needs mediapipe: pip install mediapipe")
        return 2

    try:
        import cv2
    except ImportError:
        print("AirMath needs opencv-python: pip install opencv-python")
        return 2

    session = Session(width=args.width, height=args.height)
    tracker = HandTracker()
    # Windows: DirectShow first - the default MSMF backend can fail to open
    # built-in cameras even when they are healthy.
    if sys.platform == "win32":
        cam = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam = cv2.VideoCapture(args.camera)  # fall back to the default backend
    else:
        cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        print("could not open camera %d" % args.camera)
        return 2
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("AirMath - index finger to draw, open palm to clear, q to quit.")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, so drawing feels natural
            hand = tracker.process(frame)
            out = run_over_frame(frame, session, hand)
            cv2.imshow("AirMath - write an equation with your finger", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
