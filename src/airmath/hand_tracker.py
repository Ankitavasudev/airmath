"""Hand tracker: index-fingertip position + pen state off a webcam frame.

MediaPipe imports lazily - the recognizer, solver and canvas all work
without it, which keeps the CI test run headless and fast.

Pen conventions:
  * pen down  - index finger extended, middle finger folded
  * pen up    - index folded
  * wipe      - open palm (all four fingertips extended)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandState:
    x: int
    y: int
    pen: bool   # drawing right now
    wipe: bool  # open palm -> clear canvas


class HandTracker:
    """One hand at a time. process() returns None when no hand is visible."""

    def __init__(self, max_hands: int = 1):
        self._hands = None
        self._max_hands = max_hands

    def _ensure(self):
        if self._hands is None:
            import mediapipe as mp  # lazy import keeps headless tests clean

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=0.6,
            )
        return self._hands

    def process(self, frame_bgr: np.ndarray) -> HandState | None:
        h, w = frame_bgr.shape[:2]
        results = self._ensure().process(frame_bgr)
        if not results.multi_hand_landmarks:
            return None
        lm = results.multi_hand_landmarks[0].landmark

        tip = lm[8]  # index fingertip
        x, y = int(tip.x * w), int(tip.y * h)

        def extended(tip_idx: int) -> bool:
            pip_idx = tip_idx - 2
            # a straight finger reaches further from the wrist than its PIP
            return _dist(lm, tip_idx) > _dist(lm, pip_idx)

        index = extended(8)
        middle = extended(12)
        ring = extended(16)
        pinky = extended(20)

        wipe = index and middle and ring and pinky
        pen = index and not middle
        return HandState(x=x, y=y, pen=pen and not wipe, wipe=wipe)

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None


def _dist(lm, i: int) -> float:
    return (lm[i].x - lm[0].x) ** 2 + (lm[i].y - lm[0].y) ** 2
