"""Cute overlay effects: fingertip trail, floating token chips, star
bursts, a blinking mascot, and the solve-time scanline sweep.

Everything is frame-count driven (no wall clock), so headless tests can
replay frames deterministically. cv2 is imported lazily inside draw
calls only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MINT = (0, 229, 160)    # bgr
BLUE = (120, 184, 255)  # bgr
GOLD = (0, 190, 255)    # bgr
RED = (60, 60, 255)     # bgr
WHITE = (255, 255, 255)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    age: int = 0
    color: tuple = GOLD
    star: bool = False
    size: int = 3


@dataclass
class FloatChip:
    token: str
    x: float
    y: float
    age: int = 0
    ttl: int = 42
    color: tuple = MINT


@dataclass
class Scan:
    """A one-shot scanline sweep across a region when solving."""
    x0: int
    x1: int
    y0: int
    y1: int
    tick0: int
    ttl: int = 26


def star_burst(particles: list[Particle], x: float, y: float, count: int = 30) -> None:
    """Radial celebration: mint + gold stars and dots."""
    import random

    rng = random.Random(int(x * 31 + y * 17))
    for i in range(count):
        ang = rng.uniform(0, 2 * math.pi)
        sp = rng.uniform(2.0, 6.5)
        particles.append(
            Particle(
                x=x,
                y=y,
                vx=math.cos(ang) * sp,
                vy=math.sin(ang) * sp - 1.8,
                life=rng.randint(26, 52),
                color=MINT if rng.random() < 0.55 else GOLD,
                star=rng.random() < 0.45,
                size=rng.randint(2, 4),
            )
        )


def float_chip(chips: list[FloatChip], token: str, x: float, y: float, color: tuple = MINT) -> None:
    chips.append(FloatChip(token=token, x=x, y=y, color=color))


def step(chips: list[FloatChip], particles: list[Particle]) -> None:
    """Advance one tick; prune expired chips and dead particles."""
    for c in chips:
        c.age += 1
        c.y -= 0.55
    chips[:] = [c for c in chips if c.age < c.ttl]
    for p in particles:
        p.age += 1
        p.x += p.vx
        p.y += p.vy
        p.vy += 0.07
    particles[:] = [p for p in particles if p.age < p.life]


def step_scans(scans: list[Scan], tick: int) -> None:
    scans[:] = [s for s in scans if tick - s.tick0 < s.ttl]


# --- draw calls ----------------------------------------------------------


def draw_chips(frame, chips: list[FloatChip]) -> None:
    import cv2

    h, w = frame.shape[:2]
    for c in chips:
        a = max(0.0, 1.0 - c.age / c.ttl)
        col = tuple(int(v * a) for v in c.color)
        (tw, th), _ = cv2.getTextSize(c.token, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x, y = int(c.x - tw / 2), int(c.y)
        x = max(4, min(w - tw - 12, x))
        cv2.rectangle(frame, (x - 8, y - th - 6), (x + tw + 8, y + 8), col, 1, cv2.LINE_AA)
        cv2.putText(frame, c.token, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)


def draw_particles(frame, particles: list[Particle]) -> None:
    import cv2

    for p in particles:
        a = max(0.0, 1.0 - p.age / p.life)
        col = tuple(int(v * a) for v in p.color)
        if p.star:
            r = int(4 + 9 * a)
            cv2.drawMarker(frame, (int(p.x), int(p.y)), col, cv2.MARKER_TILTED_CROSS, r, max(1, int(2 * a)), cv2.LINE_AA)
        else:
            cv2.circle(frame, (int(p.x), int(p.y)), max(1, int(p.size * a)) + 1, col, -1, cv2.LINE_AA)


def draw_scan(frame, scan: Scan, tick: int) -> None:
    import cv2

    t = (tick - scan.tick0) / scan.ttl
    y = int(scan.y0 + (scan.y1 - scan.y0) * t)
    fade = math.sin(math.pi * min(1.0, t))  # ramp in, ramp out
    col = tuple(int(v * fade) for v in MINT)
    cv2.line(frame, (scan.x0, y), (scan.x1, y), col, 3, cv2.LINE_AA)


def draw_cursor(frame, x: int, y: int, pen: bool, tick: int) -> None:
    import cv2

    pulse = 2 * math.sin(tick * 0.22)
    r = 12 + int(pulse)
    col = MINT if pen else BLUE
    cv2.circle(frame, (x, y), r, col, 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 4, col, -1, cv2.LINE_AA)
    if pen:
        cv2.circle(frame, (x, y), r + 6, (110, 245, 190), 1, cv2.LINE_AA)


def draw_trail(frame, trail: list[tuple[int, int]], pen: bool) -> None:
    """Fading comet tail behind the fingertip (last ~18 points)."""
    import cv2

    if not trail:
        return
    col = MINT if pen else BLUE
    n = len(trail)
    for i, (x, y) in enumerate(trail):
        a = (i + 1) / n
        col_i = tuple(int(v * 0.55 * a) for v in col)
        rad = max(1, int(7 * a))
        cv2.circle(frame, (x, y), rad, col_i, -1, cv2.LINE_AA)


def draw_mascot(frame, x: int, y: int, mood: str, tick: int) -> None:
    """A tiny friendly face: dot eyes, curvy mouth. mood: ok|err|idle."""
    import cv2

    blink = (tick % 150) < 7
    eye_col = MINT if mood == "ok" else (RED if mood == "err" else BLUE)
    r_eye = 3 if blink else 5 if mood == "ok" else 4
    for dx in (-11, 11):
        cv2.circle(frame, (x + dx, y), r_eye, eye_col, -1, cv2.LINE_AA)
    cv2.circle(frame, (x - 11, y), 5, eye_col, 1, cv2.LINE_AA) if blink else None
    mouth_col = eye_col
    if mood == "ok":
        cv2.ellipse(frame, (x, y + 12), (12, 7), 0, 20, 160, mouth_col, 2, cv2.LINE_AA)
    elif mood == "err":
        cv2.ellipse(frame, (x, y + 22), (12, 7), 0, 200, 340, mouth_col, 2, cv2.LINE_AA)
    else:
        cv2.line(frame, (x - 9, y + 14), (x + 9, y + 14), mouth_col, 2, cv2.LINE_AA)
