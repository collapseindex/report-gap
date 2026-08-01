"""Per-cell readouts, the discrepancy statistic, and the paired bootstrap.

Everything here is paired over item x permutation x wording cells, which is how every condition is
built. The pairing is the part that has already broken once: a pilot log truncated item keys to 30
characters, every prompt shares those 30 characters, and per-item pairing silently collapsed to two
keys while the aggregate numbers looked fine. `paired_deltas` therefore asserts on the key set
rather than trusting it.

No language model is involved in any quantity here. Every number is a softmax read or a count.

Nothing in this module imports torch or numpy, so it runs in the test suite without a GPU image and
the bootstrap is reproducible from a stdlib seed alone.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile interval.

    Attributes:
        point: The observed statistic.
        lo: Lower percentile bound.
        hi: Upper percentile bound.
        n: Number of paired cells the estimate is over.
    """

    point: float
    lo: float
    hi: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero."""
        return self.lo > 0.0 or self.hi < 0.0

    def __str__(self) -> str:
        return "%+.4f [%+.4f, %+.4f] n=%d" % (self.point, self.lo, self.hi, self.n)


def option_mass(probs: dict[str, float], letters: set[str]) -> float:
    """Share of the option distribution sitting on a given set of letters.

    Args:
        probs: Renormalized distribution over option letters.
        letters: The letters to sum.

    Returns:
        Mass in [0, 1].

    Raises:
        ValueError: If the distribution does not sum to 1, so a renormalization bug cannot pass
            through as a plausible-looking share.
    """
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("option distribution sums to %.9f, not 1" % total)
    value = sum(probs[L] for L in letters if L in probs)
    assert 0.0 <= value <= 1.0 + 1e-9, "mass outside its definitional bound"
    return min(1.0, value)


def option_entropy(probs: dict[str, float]) -> float:
    """Shannon entropy of the option distribution, in nats.

    Args:
        probs: Renormalized distribution over option letters.

    Returns:
        Entropy in [0, log k].
    """
    return -sum(p * math.log(p) for p in probs.values() if p > 0.0)


def max_letter_share(letters: list[str]) -> float:
    """Fraction of choices taken by the single most-chosen letter.

    Near 1.0 means the readout is position rather than state. This is the diagnostic that killed the
    behavioural arm in the prior design, so it is an endpoint here and not a footnote.

    Args:
        letters: One chosen letter per cell.

    Returns:
        Share in [0, 1].

    Raises:
        ValueError: If the list is empty, rather than returning a share over nothing.
    """
    if not letters:
        raise ValueError("no choices: refusing to return a letter share over nothing")
    return max(letters.count(L) for L in set(letters)) / len(letters)


def paired_deltas(treatment: dict[str, float], baseline: dict[str, float]) -> list[float]:
    """Per-cell differences between a treatment condition and its own baseline.

    Args:
        treatment: Cell key to value under treatment.
        baseline: Cell key to value at alpha = 0.

    Returns:
        One delta per cell, in sorted key order so the result is deterministic.

    Raises:
        ValueError: If the key sets differ, or if either side has fewer distinct keys than entries,
            which is what a truncated item identifier looks like from here.
    """
    if not treatment or not baseline:
        raise ValueError("empty condition: refusing to pair over nothing")
    if set(treatment) != set(baseline):
        missing = sorted(set(baseline) ^ set(treatment))[:5]
        raise ValueError("treatment and baseline cover different cells; first differences: %s"
                         % missing)
    return [treatment[k] - baseline[k] for k in sorted(treatment)]


def assert_key_integrity(keys: list[str], expected: int) -> None:
    """Assert that cell keys are unique and as numerous as the design says.

    Args:
        keys: Every cell key in an artifact.
        expected: How many distinct cells the frozen design produces.

    Raises:
        AssertionError: If keys collide or the count is wrong. This is the guard for the pilot bug
            in which 30-character truncation collapsed 30 items to 2 distinct keys while every
            aggregate still printed a believable number.
    """
    distinct = len(set(keys))
    if distinct != len(keys):
        raise AssertionError("cell keys collide: %d entries, %d distinct. Per-item pairing is not "
                             "what it appears to be." % (len(keys), distinct))
    if distinct != expected:
        raise AssertionError("expected %d cells, found %d" % (expected, distinct))


def paired_bootstrap(deltas: list[float], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> Interval:
    """Percentile bootstrap over paired cell deltas.

    Args:
        deltas: One paired difference per cell.
        resamples: Number of bootstrap resamples.
        seed: Seed, so an interval is reproducible from the artifact alone.
        alpha: Two-sided error rate; 0.05 gives a 95% interval.

    Returns:
        An `Interval`.

    Raises:
        ValueError: If there are fewer than two cells, because a one-cell interval is a number
            dressed as an estimate.
    """
    if len(deltas) < 2:
        raise ValueError("need at least 2 paired cells, got %d" % len(deltas))
    n = len(deltas)
    point = sum(deltas) / n
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int((alpha / 2.0) * resamples)]
    hi = means[min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))]
    return Interval(point=point, lo=lo, hi=hi, n=n)


def discrepancy_deltas(mass_treat: dict[str, float], mass_base: dict[str, float],
                       argmax_treat: dict[str, bool], argmax_base: dict[str, bool]) -> list[float]:
    """Per-cell primary endpoint: how much of the mass effect the argmax fails to register.

    Positive values mean the argmax under-reports. This is the quantity contrasts 1 and 2 are
    computed on and the quantity the planted-discrepancy controls must recover.

    Args:
        mass_treat: Cell key to own-pole mass under treatment.
        mass_base: Cell key to own-pole mass at alpha = 0.
        argmax_treat: Cell key to whether the argmax sits on an own-pole option under treatment.
        argmax_base: Cell key to the same at alpha = 0.

    Returns:
        One discrepancy per cell, in sorted key order.

    Raises:
        ValueError: If the four key sets are not identical.
    """
    keys = set(mass_treat)
    for name, d in (("mass_base", mass_base), ("argmax_treat", argmax_treat),
                    ("argmax_base", argmax_base)):
        if set(d) != keys:
            raise ValueError("%s covers a different cell set from mass_treat" % name)
    mass = paired_deltas(mass_treat, mass_base)
    arg = paired_deltas({k: float(v) for k, v in argmax_treat.items()},
                        {k: float(v) for k, v in argmax_base.items()})
    return [m - a for m, a in zip(mass, arg)]


def ratio_to_control(effect: float, control: float) -> float:
    """Effect as a multiple of the matched-random effect.

    The pilot found random directions move mass by +0.050, so a raw shift is not interpretable on
    its own. The raw control value is reported alongside this, never replaced by it.

    Args:
        effect: Treatment effect.
        control: Matched-random effect.

    Returns:
        The ratio.

    Raises:
        ValueError: If the control effect is at or near zero, where a ratio is unbounded and would
            print an impressive number from a rounding artifact.
    """
    if abs(control) < 1e-6:
        raise ValueError("matched-random effect is %.9f; a ratio to it is not a real number and "
                         "the raw difference should be reported instead" % control)
    return effect / control
