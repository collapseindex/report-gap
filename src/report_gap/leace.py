"""LEACE: closed-form linear concept erasure (Belrose et al., NeurIPS 2023, arXiv:2306.03819).

`RESULTS_prompt_erase.md` reports that iterative nullspace projection failed to erase a
prompt-induced valence property: a refit probe still read it at cv 1.000 after 128 directions were
removed, at n=60 and again at n=1800. This module is the named fix, and it is a different operation
rather than more of the same one.

The difference that matters. INLP removes directions in RAW residual space, one fitted classifier at
a time, so in 2048 dimensions with a redundantly-encoded property there is always another direction
and the loop never converges. LEACE WHITENS first and removes the concept's component in whitened
space, where for a binary label the required projection is RANK ONE. It carries a guarantee INLP
does not: after erasure, the class-conditional means coincide, so **no linear classifier can recover
the label better than chance**.

That guarantee is checkable, and `tests/test_leace.py` checks it rather than trusting the citation.
"""

from __future__ import annotations

import numpy as np


class LeaceEraser:
    """An affine map that removes a binary concept from a representation.

    Attributes:
        proj: (hidden, hidden) the linear part, applied to centered activations.
        bias: (hidden,) the mean that is added back, so the eraser is affine and preserves the
            overall mean rather than shifting everything to the origin.
    """

    def __init__(self, proj: np.ndarray, bias: np.ndarray):
        self.proj, self.bias = proj, bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Erase the concept from `x`, shape (..., hidden)."""
        return (x - self.bias) @ self.proj.T + self.bias

    @property
    def rank(self) -> int:
        """Dimensions actually removed. For a binary label this is 1, which is the point."""
        return int(round(self.proj.shape[0] - np.trace(self.proj)))


def fit_leace(x: np.ndarray, z: np.ndarray, shrinkage: float = 1e-6) -> LeaceEraser:
    """Fit the LEACE eraser for a binary concept.

    Args:
        x: (n, hidden) activations.
        z: (n,) binary labels, any two distinct values.
        shrinkage: Ridge added to the covariance diagonal before whitening. Necessary, not
            cosmetic: with n < hidden the sample covariance is singular, and an unregularized
            inverse would silently produce an eraser that erases nothing.

    Returns:
        A `LeaceEraser`.

    Raises:
        ValueError: If `z` is not binary, or if either class is empty, where the concept direction
            is undefined and a silent no-op would look like a clean erasure.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z)
    classes = np.unique(z)
    if classes.size != 2:
        raise ValueError("LEACE here is binary; got %d classes" % classes.size)
    zb = (z == classes[1]).astype(np.float64)
    if zb.sum() == 0 or zb.sum() == zb.size:
        raise ValueError("one class is empty; the concept direction is undefined")

    mu = x.mean(axis=0)
    xc = x - mu
    zc = zb - zb.mean()

    # Cross-covariance between representation and label. For a binary label this is a single
    # vector, which is why the erasure is rank one.
    sigma_xz = xc.T @ zc / (len(x) - 1)

    sigma = np.cov(xc, rowvar=False)
    sigma = np.atleast_2d(sigma)
    sigma.flat[:: sigma.shape[0] + 1] += shrinkage * np.trace(sigma) / sigma.shape[0]

    # Whitening transform W and its pseudo-inverse, via the symmetric square root.
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.clip(evals, 1e-12, None)
    w = evecs @ np.diag(evals ** -0.5) @ evecs.T
    w_inv = evecs @ np.diag(evals ** 0.5) @ evecs.T

    # Project out the concept in WHITENED space, then map back. This is the whole method.
    u = w @ sigma_xz
    nrm = np.linalg.norm(u)
    if nrm < 1e-12:
        raise ValueError("the concept has no linear component to erase at this layer")
    u = u / nrm

    proj = np.eye(x.shape[1]) - w_inv @ np.outer(u, u) @ w
    return LeaceEraser(proj=proj, bias=mu)


def class_mean_gap(x: np.ndarray, z: np.ndarray) -> float:
    """Distance between class-conditional means, the quantity LEACE drives to zero.

    A linear classifier's achievable advantage is bounded by this, so it is the direct readout of
    whether the erasure worked, independent of any probe anyone chooses to fit afterwards.
    """
    classes = np.unique(z)
    a = x[z == classes[0]].mean(axis=0)
    b = x[z == classes[1]].mean(axis=0)
    return float(np.linalg.norm(a - b))
