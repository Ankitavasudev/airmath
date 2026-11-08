"""Train and ship the digit recogniser.

Downloads MNIST via scikit-learn (fetch_openml), subsamples it, and saves
an npz blob the 1-NN classifier loads at runtime.  The resulting file is
committed so users never need to train anything.

    py -3.13 scripts/train_recognizer.py --samples 12000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "src" / "airmath" / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the AirMath digit blob (MNIST 1-NN).")
    parser.add_argument("--samples", type=int, default=12000, help="training subset size (~5 MB at 12k)")
    args = parser.parse_args()

    from sklearn.datasets import fetch_openml

    print("downloading MNIST (one-time, ~15 MB)...")
    t0 = time.time()
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = data.data.astype(np.uint8)
    y = data.target.astype(np.int64)
    print(f"  {X.shape[0]} images in {time.time() - t0:.0f}s")

    rng = np.random.default_rng(2026)
    idx = rng.permutation(len(X))[: args.samples]
    Xs, ys = X[idx], y[idx]

    # quick sanity: hold out 2 percent and check 1-NN accuracy (batched, no explosion)
    cut = max(1, len(Xs) // 50)
    train_x, train_y = Xs[cut:].astype(np.float32), ys[cut:]
    test_x, test_y = Xs[:cut].astype(np.float32), ys[:cut]
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b avoids building the full diff tensor
    d2 = (
        (train_x ** 2).sum(axis=1)[:, None]
        + (test_x ** 2).sum(axis=1)[None, :]
        - 2.0 * (train_x @ test_x.T)
    )
    preds = np.array([train_y[d2[:, j].argmin()] for j in range(len(test_y))])
    acc = float((preds == test_y).mean())
    print(f"held-out 1-NN accuracy on {cut} samples: {acc:.3f}")
    if acc < 0.85:
        print("accuracy too low - not shipping this model")
        return 1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "digits.npz"
    np.savez_compressed(out, images=Xs.reshape(-1, 784), labels=ys)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {len(Xs)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
