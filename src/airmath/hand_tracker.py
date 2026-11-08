"""Hand tracker: index-fingertip position + pen state off a webcam frame.

Supports both MediaPipe generations:
  * mediapipe >= 1.0 - the new tasks API (HandLandmarker)
  * mediapipe 0.10.x - the legacy solutions API

The hand-landmarker model ships in models/hand_landmarker.task; if it is
missing the tracker downloads it once. MediaPipe itself imports lazily -
the recognizer, solver and canvas all work without it, which keeps the
CI test run headless and fast.

Pen conventions:
  * pen down  - index finger extended, middle finger folded
  * pen up    - index folded
  * wipe      - open palm (all four fingertips extended)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class HandState:
    x: int
    y: int
    pen: bool   # drawing right now
    wipe: bool  # open palm -> clear canvas


class HandTracker:
    """One hand at a time. process() returns None when no hand is visible."""

    def __init__(self, max_hands: int = 1):
        self._mode: str | None = None  # "tasks" | "legacy"
        self._landmarker = None
        self._hands = None
        self._max_hands = max_hands
        self._ts_ms = 0

    # --- setup ------------------------------------------------------------

    def _ensure(self):
        if self._landmarker is None and self._hands is None:
            import mediapipe as mp  # lazy: not needed by headless tests

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self._mode = "legacy"
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self._max_hands,
                    min_detection_confidence=0.6,
                )
            else:
                self._mode = "tasks"
                self._landmarker = self._create_tasks(mp)
        return self._landmarker or self._hands

    def _create_tasks(self, mp):
        from pathlib import Path

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request

            print("downloading hand_landmarker.task (one-time)...")
            urllib.request.urlretrieve(MODEL_URL, model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._max_hands,
            min_hand_detection_confidence=0.6,
        )
        return vision.HandLandmarker.create_from_options(options)

    # --- per-frame --------------------------------------------------------

    def process(self, frame_bgr: np.ndarray) -> HandState | None:
        h, w = frame_bgr.shape[:2]
        self._ts_ms += 33
        self._ensure()
        if self._mode == "tasks":
            lm = self._detect_tasks(frame_bgr)
        else:
            lm = self._detect_legacy(frame_bgr)
        if lm is None:
            return None
        return _state_from(lm, w, h)

    def _detect_tasks(self, frame_bgr: np.ndarray):
        import mediapipe as mp

        rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._ts_ms)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def _detect_legacy(self, frame_bgr: np.ndarray):
        result = self._hands.process(frame_bgr)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0].landmark

    def close(self) -> None:
        self._landmarker = None
        self._hands = None
        self._mode = None


def _state_from(lm, w: int, h: int) -> HandState:
    tip = lm[8]  # index fingertip
    x, y = int(tip.x * w), int(tip.y * h)

    def extended(i: int) -> bool:
        return _dist(lm, i) > _dist(lm, i - 2)  # tip further from wrist than PIP

    index = extended(8)
    middle = extended(12)
    ring = extended(16)
    pinky = extended(20)

    wipe = index and middle and ring and pinky
    pen = index and not middle
    return HandState(x=x, y=y, pen=pen and not wipe, wipe=wipe)


def _dist(lm, i: int) -> float:
    return (lm[i].x - lm[0].x) ** 2 + (lm[i].y - lm[0].y) ** 2
