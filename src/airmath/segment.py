"""Split wide ink blobs into glyphs on x-axis gaps.

If the writer smears several characters into one stroke (common when the
pen-up gesture is missed), the columns of the blob show ink valleys at
character boundaries.  Wide glyphs with a big valley get cut there;
otherwise the blob stays whole.
"""

from __future__ import annotations

import numpy as np


def split_on_gaps(mask: np.ndarray, max_gap_ratio: float) -> list[np.ndarray]:
    """Yield sub-masks.  A blob is split when a vertical ink gap wider than
    max_gap_ratio * height is found; each cut leaves both sides wide enough."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    blob = mask[y0 : y1 + 1, x0 : x1 + 1]
    h, w = blob.shape

    gap_threshold = max_gap_ratio * h
    col_ink = blob.any(axis=0)
    cuts: list[int] = []
    run_start = None
    for cx in range(w):
        if not col_ink[cx]:
            if run_start is None:
                run_start = cx
        else:
            if run_start is not None and (cx - run_start) > gap_threshold and run_start > 4 and cx < w - 4:
                cuts.append((run_start + cx - 1) // 2)
            run_start = None

    if not cuts:
        return [blob] if int(blob.sum()) > 0 else []

    edges = [0] + cuts + [w]
    out = []
    for a, b in zip(edges, edges[1:]):
        sub = blob[:, a:b]
        subys, subxs = np.nonzero(sub)
        if len(subxs) == 0:
            continue
        sy0, sy1 = int(subys.min()), int(subys.max())
        out.append(sub[sy0 : sy1 + 1, :])
    return out
