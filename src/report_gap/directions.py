"""Fitting state directions from contrastive stimuli.

The direction is the discriminative one: the logistic-regression coefficient vector, fit on
standardized activations and then mapped back into raw residual space so it can be added to the
stream directly. A difference-of-means direction is provided too, because `recipient-probe` found
that the two behave differently under steering (the difference of means at the peak-probe layer did
not steer, the discriminative direction at a later layer did), and that is worth measuring again
rather than assuming.

Scoring uses leave-one-group-out, matching the bag-of-words guard in `validate_stimuli.py`, so that
probe accuracy and the guard are answering the same question over the same folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Direction:
    """A fitted direction and the evidence that it is real.

    Attributes:
        vector: Unit-norm direction in raw residual space, shape (hidden_size,).
        cv_accuracy: Leave-one-group-out accuracy of the probe that produced it.
        n: Number of stimulus rows it was fit on.
        layer: The layer the activations came from.
        method: Either "discriminative" or "diffmeans".
    """

    vector: np.ndarray
    cv_accuracy: float
    n: int
    layer: int
    method: str


@torch.no_grad()
def collect_activations(model, tokenizer, texts: list[str], layer: int) -> np.ndarray:
    """Last-token residual activations at `layer` for each text.

    Args:
        model: A causal language model in eval mode.
        tokenizer: Its tokenizer, with a chat template.
        texts: Stimulus strings.
        layer: Decoder layer index.

    Returns:
        Array of shape (len(texts), hidden_size), float32.
    """
    device = next(model.parameters()).device
    out = []
    for text in texts:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(device)
        hidden = model(**inputs, output_hidden_states=True).hidden_states[layer + 1]
        out.append(hidden[0, -1, :].float().cpu().numpy())
    return np.asarray(out, dtype=np.float32)


def _logistic_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=1.0, max_iter=3000).fit(x, y).coef_[0]


def fit_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    layer: int,
    method: str = "discriminative",
) -> Direction:
    """Fit a unit-norm direction in raw residual space, with a held-out accuracy attached.

    Standardization matters here: a coefficient vector fit on standardized features lives in a
    rescaled space, so it is divided by the feature standard deviations before being used as a
    direction to add to raw activations. Skipping that step gives a vector that points somewhere
    plausible and steers badly.

    Args:
        activations: Shape (n, hidden_size).
        labels: Binary labels, shape (n,).
        groups: Frame-group ids, shape (n,), held out as units.
        layer: Layer the activations came from, recorded on the result.
        method: "discriminative" for the logistic coefficient, "diffmeans" for the class-mean
            difference.

    Returns:
        A `Direction`.

    Raises:
        ValueError: On an unknown method, or fewer than two groups.
    """
    if len(set(groups.tolist())) < 2:
        raise ValueError("need at least two frame groups to score leave-one-group-out")

    mu = activations.mean(axis=0)
    sigma = activations.std(axis=0)
    sigma[sigma == 0] = 1.0
    standardized = (activations - mu) / sigma

    if method == "discriminative":
        weights = _logistic_fit(standardized, labels)
        vector = weights / sigma
    elif method == "diffmeans":
        vector = activations[labels == 1].mean(axis=0) - activations[labels == 0].mean(axis=0)
    else:
        raise ValueError("unknown method: %r" % method)

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("fitted direction has zero norm")
    vector = vector / norm

    correct = total = 0
    for held in sorted(set(groups.tolist())):
        train, test = groups != held, groups == held
        if len(set(labels[train].tolist())) < 2:
            continue
        m, s = activations[train].mean(axis=0), activations[train].std(axis=0)
        s[s == 0] = 1.0
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1.0, max_iter=3000).fit((activations[train] - m) / s, labels[train])
        correct += int((clf.predict((activations[test] - m) / s) == labels[test]).sum())
        total += int(test.sum())

    return Direction(
        vector=vector.astype(np.float32),
        cv_accuracy=round(correct / total, 3) if total else float("nan"),
        n=len(labels),
        layer=layer,
        method=method,
    )


def random_direction(hidden_size: int, seed: int) -> np.ndarray:
    """A seeded random unit direction, for the norm-matched control.

    Args:
        hidden_size: Dimensionality of the residual stream.
        seed: Seed, so the control is reproducible.

    Returns:
        Unit-norm array of shape (hidden_size,), float32.
    """
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(hidden_size)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity between two directions.

    Absolute because a direction's sign is a labelling convention, not a fact about the geometry.

    Args:
        a: First direction.
        b: Second direction.

    Returns:
        Absolute cosine similarity in [0, 1].
    """
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.dot(a, b) / denominator)) if denominator else 0.0


def random_cosine_floor(hidden_size: int, n: int = 64, seed: int = 0) -> tuple[float, float]:
    """The cosine you get between unrelated directions, which is not zero in finite dimensions.

    Any reported cosine between two fitted directions is meaningless without this floor. In 2048
    dimensions two random vectors sit around 0.02, so a reported 0.09 is near-orthogonal and a
    reported 0.6 is not.

    Args:
        hidden_size: Dimensionality.
        n: Number of random pairs to sample.
        seed: Base seed.

    Returns:
        (mean, max) absolute cosine over `n` random pairs.
    """
    values = [
        cosine(random_direction(hidden_size, seed + 2 * i), random_direction(hidden_size, seed + 2 * i + 1))
        for i in range(n)
    ]
    return float(np.mean(values)), float(np.max(values))
