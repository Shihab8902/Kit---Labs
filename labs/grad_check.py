"""Chunk 1.4 — gradient check: fast math vs tiny manual nudge."""
import numpy as np


def loss_fn(W, X, Y):
    logits = X @ W
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    P = e / e.sum(axis=1, keepdims=True)
    return float(-np.log(P[np.arange(len(X)), Y] + 1e-12).mean()), P


def main():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (4, 3))
    Y = np.array([0, 1, 0, 1])
    W = rng.normal(0, 0.5, (3, 2))

    loss, P = loss_fn(W, X, Y)
    Yoh = np.zeros_like(P)
    Yoh[np.arange(len(X)), Y] = 1.0
    analytic = X.T @ (P - Yoh) / len(X)  # known formula for this setup

    i, j, e = 1, 1, 1e-5
    Wp = W.copy(); Wp[i, j] += e
    Wm = W.copy(); Wm[i, j] -= e
    numeric = (loss_fn(Wp, X, Y)[0] - loss_fn(Wm, X, Y)[0]) / (2 * e)

    a, n = float(analytic[i, j]), float(numeric)
    rel = abs(a - n) / (abs(a) + abs(n) + 1e-12)
    print(f"analytic={a:.6f} numeric={n:.6f} rel_error={rel:.2e}")
    print(f"check_pass={rel < 1e-4}")


if __name__ == "__main__":
    main()
