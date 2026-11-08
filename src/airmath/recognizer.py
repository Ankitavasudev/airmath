"""Classify one air-written glyph.

Digits 0-9 come from a tiny 1-NN classifier over MNIST data shipped in
models/digits.npz.  Operators (+ - * / =) are classified by cheap shape
rules on the ink mask - they have distinctive geometry and a digit model
would never learn them from MNIST anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"

OPERATOR_TOKENS = ("+", "-", "*", "/", "=")


@dataclass
class Glyph:
    """One air-written character: ink pixels plus its place on the canvas."""

    ink: np.ndarray  # boolean HxW mask, True = ink
    x: int  # left edge on the canvas


@lru_cache(maxsize=1)
def _load_digits() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(MODELS_DIR / "digits.npz")
    return data["images"], data["labels"]


def available() -> bool:
    return (MODELS_DIR / "digits.npz").exists()


def predict_digit(glyph: Glyph) -> str:
    """Return "0".."9" for a digit-shaped glyph using 1-NN over MNIST."""
    x28 = _to_28x28(glyph.ink)
    images, labels = _load_digits()
    diff = images - x28.ravel()[None, :]
    best = int(np.argmin((diff * diff).sum(axis=1)))
    return str(int(labels[best]))


# --- operator rules ------------------------------------------------------


def predict_operator(glyph: Glyph) -> str | None:
    """Try to explain the glyph as +, -, *, / or =. None means "not an
    operator - probably a digit"."""
    mask = _crop(glyph.ink)
    h, w = mask.shape
    ink = float(mask.sum())
    if ink == 0:
        return None
    aspect = w / max(h, 1)

    labels = _row_profile_groups(mask)

    # horizontal check: "=" (two bands), "-" (one flat band)
    horiz = _horizontalness(mask)
    vert = _verticalness(mask)
    diag_both = _diagonalness(mask)          # both diagonals -> "*"
    diag_single = _diagonalness_single(mask)  # one diagonal -> "/"

    if aspect > 1.1 and horiz > 0.6 and labels == 2 and h >= 4:
        return "="
    # "-" also works for a thick brush-drawn bar; the narrow-tall shapes it
    # could collide with (digits "1", "/") fail the aspect>2 gate instead
    if labels == 1 and horiz > 0.7 and aspect > 2.0:
        return "-"
    if diag_both > 0.6 and horiz < 0.5 and vert < 0.5:
        return "*"
    if vert > 0.55 and horiz > 0.55 and _spine_columns(mask) < 0.35 and _spine_rows(mask) < 0.35:
        return "+"
    if diag_single > 0.75 and aspect < 1.2 and abs(_rel_drift(mask)) > 0.15:
        return "/"
    return None


def _row_profile_groups(mask: np.ndarray) -> int:
    """Count separate horizontal ink bands: rows with ink -> run-length 1s."""
    rows = mask.any(axis=1)
    groups, prev = 0, False
    for r in rows:
        if r and not prev:
            groups += 1
        prev = r
    return groups


def _crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask.astype(bool)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _spine_columns(mask: np.ndarray, frac: float = 0.7) -> float:
    """Fraction of columns with a near-full-height vertical run. A '+' has a
    thin spine (few columns); a fat brush bar fills nearly every column."""
    h, w = mask.shape
    if h < 2 or w < 2 or not mask.any():
        return 1.0
    need = frac * h
    tall = 0
    for c in range(w):
        ys = np.nonzero(mask[:, c])[0]
        if len(ys) and np.ptp(ys) + 1 >= need:
            tall += 1
    return tall / w


def _spine_rows(mask: np.ndarray, frac: float = 0.7) -> float:
    """Fraction of rows with a near-full-width horizontal run."""
    h, w = mask.shape
    if h < 2 or w < 2 or not mask.any():
        return 1.0
    need = frac * w
    wide = 0
    for r in range(h):
        xs = np.nonzero(mask[r])[0]
        if len(xs) and np.ptp(xs) + 1 >= need:
            wide += 1
    return wide / h


def _rel_drift(mask: np.ndarray) -> float:
    """(bottom_x - top_x) / height of the ink's centre line. ~0 for a
    vertical bar, strongly negative for '/', strongly positive for '\\'."""
    ys, xs = np.nonzero(mask)
    h = ys.max() - ys.min() + 1
    if h < 5:
        return 0.0
    third = max(1, len(ys) // 4)
    order = np.argsort(ys)
    top_x = float(np.mean(xs[order[:third]])) - xs.min()
    bot_x = float(np.mean(xs[order[-third:]])) - xs.min()
    return (bot_x - top_x) / h


def _horizontalness(mask: np.ndarray) -> float:
    """Widest horizontal ink run relative to width - any row, not just centre."""
    h, w = mask.shape
    if w < 2 or not mask.any():
        return 0.0
    best = 0
    for r in range(h):
        xs = np.nonzero(mask[r])[0]
        if len(xs):
            best = max(best, int(np.ptp(xs)) + 1)
    return best / w


def _verticalness(mask: np.ndarray) -> float:
    """Tallest vertical ink run relative to height - any column."""
    h, w = mask.shape
    if h < 2 or not mask.any():
        return 0.0
    best = 0
    for c in range(w):
        ys = np.nonzero(mask[:, c])[0]
        if len(ys):
            best = max(best, int(np.ptp(ys)) + 1)
    return best / h


def _main_diagonal_score(mask: np.ndarray, direction: int, tol: int = 1) -> float:
    """Fraction of diagonal cells with ink within +-tol of the ideal line."""
    h, w = mask.shape
    if h < 3 or w < 3:
        return 0.0
    hit = 0
    cells = 0
    for i in range(h):
        t = i * (w - 1) / (h - 1)
        j = int(round(t)) if direction > 0 else (w - 1) - int(round(t))
        lo, hi = max(0, j - tol), min(w, j + tol + 1)
        cells += 1
        if mask[i, lo:hi].any():
            hit += 1
    return hit / cells


def _diagonalness(mask: np.ndarray) -> float:
    # cross: BOTH diagonals present
    return min(_main_diagonal_score(mask, +1), _main_diagonal_score(mask, -1))


def _diagonalness_single(mask: np.ndarray) -> float:
    return max(_main_diagonal_score(mask, +1), _main_diagonal_score(mask, -1))


# --- classification entry point -----------------------------------------


def _to_28x28(mask: np.ndarray) -> np.ndarray:
    """Matcher pad: crop, scale the longest side to 20 px, centre in a
    28x28 field - same convention MNIST itself uses."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(784, dtype=np.float32)
    ink = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    h, w = ink.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    # nearest-neighbour resize without cv2, so tests run headless
    row_idx = np.clip((np.arange(nh) / scale).astype(int), 0, h - 1)
    col_idx = np.clip((np.arange(nw) / scale).astype(int), 0, w - 1)
    small = ink[row_idx][:, col_idx]
    out = np.zeros((28, 28), dtype=np.float32)
    y0, x0 = (28 - nh) // 2, (28 - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = small
    # shift by centre of gravity, MNIST-style (digits in MNIST are CoG-centred)
    s = out.sum()
    if s > 0:
        yy, xx = np.mgrid[0:28, 0:28]
        cy, cx = (out * yy).sum() / s, (out * xx).sum() / s
        dy, dx = int(round(14 - cy)), int(round(14 - cx))
        shifted = np.zeros_like(out)
        src_y = np.clip(np.arange(28) - dy, 0, 27)
        src_x = np.clip(np.arange(28) - dx, 0, 27)
        shifted = out[src_y][:, src_x]
        out = shifted
    return out.ravel().astype(np.float32)


def classify(glyph: Glyph) -> str | None:
    """One glyph -> one token. Operators first (rules), then MNIST digits."""
    if not available():
        return None
    op = predict_operator(glyph)
def classify(glyph: Glyph) -> str | None:
    """One glyph -> one token. Operators first (shape rules, no model
    needed), then MNIST digits (skipped when the blob is not shipped)."""
    op = predict_operator(glyph)
    if op is not None:
        return op
    if not available():
        return None
    return predict_digit(glyph)


def model_info() -> str:
    images, labels = _load_digits()
    return json.dumps({"samples": int(images.shape[0]), "shape": list(images.shape[1:])})
