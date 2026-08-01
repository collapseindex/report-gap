"""The planted-discrepancy control: a known mass shift that does not move the argmax.

The claim in `PREREG_readout_gap.md` is about a *discrepancy* between two scorings of the same
distribution: how much of a mass shift survives into the argmax. The formality axis only shows that
the argmax is capable of moving, which validates the instrument somewhere other than where the
decision rule reads. `CONTROLS.md` section 1 names that failure directly: the plant has to land in
the tail your decision rule reads, or a passing control licenses nothing.

So this plants a discrepancy of known size. Given an item's option distribution, it computes the
exact constant to add to the own-pole option logits that moves own-pole mass by a target amount.
The constant is closed-form, not searched:

    after adding c to the own-pole logits, the new own-pole mass m1 satisfies
        m1 / (1 - m1) = exp(c) * m0 / (1 - m0)
    so
        c = logit(m1) - logit(m0),  with m1 = m0 + target

The per-cell mass delta is therefore exactly `target` by construction, independent of any code in
`analysis.py`. That is what makes it a gate rather than a circularity: the analysis pipeline, with
its per-item pairing and its bootstrap, has to recover a number this module fixed without it.

Two strengths are planted. The strong one (0.15) rules out a broken statistic. The floor one (0.03)
rules out an insensitive one, which is the only thing that makes a null on the real arms mean
`absent` rather than `uninformative`.

Nothing here imports torch. It operates on a probability dict and is exactly testable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# a plant is rejected rather than clipped if it would push mass outside this band, because a
# silently clipped plant is a control whose known value is no longer known.
MASS_EPS = 1e-6


@dataclass(frozen=True)
class PlantedCell:
    """One cell with a synthetic mass shift applied.

    Attributes:
        probs: The option distribution after planting, renormalized over the option letters.
        logit_shift: The constant added to every own-pole option logit.
        mass_before: Own-pole mass before planting.
        mass_after: Own-pole mass after planting.
        argmax_before: Winning option letter before planting.
        argmax_after: Winning option letter after planting.
        target: The mass shift requested for THIS cell.
        nominal: The arm's nominal target, which every cell in an arm shares even when their
            per-cell targets differ. Guards against silently averaging a 0.15 arm with a 0.03 one.
    """

    probs: dict[str, float]
    logit_shift: float
    mass_before: float
    mass_after: float
    argmax_before: str
    argmax_after: str
    target: float
    nominal: float

    @property
    def argmax_moved(self) -> bool:
        """Whether the plant flipped the argmax.

        A flip is not a bug: it is recorded so the expected discrepancy stays known. The expected
        primary endpoint on planted cells is `target` minus the argmax effect, and the argmax effect
        is countable from this flag without touching the analysis code being gated.
        """
        return self.argmax_before != self.argmax_after

    @property
    def realized_shift(self) -> float:
        """The mass shift actually achieved, which must equal `target` to floating-point."""
        return self.mass_after - self.mass_before


def logit_shift_for(mass_before: float, target: float) -> float:
    """Constant to add to own-pole logits to move own-pole mass by exactly `target`.

    Args:
        mass_before: Own-pole share of the renormalized option distribution, in (0, 1).
        target: Desired signed change in that share.

    Returns:
        The additive logit constant.

    Raises:
        ValueError: If the starting mass or the destination mass is outside (0, 1). A plant that
            cannot land is refused rather than clipped, because a clipped plant has an unknown
            known value, which defeats the point of planting it.
    """
    if not MASS_EPS < mass_before < 1.0 - MASS_EPS:
        raise ValueError("mass_before %.6f is at or past the boundary; no finite shift exists"
                         % mass_before)
    destination = mass_before + target
    if not MASS_EPS < destination < 1.0 - MASS_EPS:
        raise ValueError("plant of %+.3f from %.3f lands at %.3f, outside (0, 1)"
                         % (target, mass_before, destination))
    return (math.log(destination / (1.0 - destination))
            - math.log(mass_before / (1.0 - mass_before)))


def matched_noise_targets(nominal: float, sd: float, n: int, seed: int) -> list[float]:
    """Per-cell plant targets with a given mean and a given spread.

    A plant at a constant target has zero variance, so its bootstrap interval excludes zero however
    insensitive the pipeline is. That is a control mathematically incapable of failing, which
    `CONTROLS.md` section 1 names as worse than no control: it looks like a power check and is not.

    The floor plant therefore carries the spread the real treatment arm actually shows, so
    "the interval excludes zero at n cells" is a statement about detecting a 0.03 effect against
    this much noise. `sd` comes from the treatment arm's own per-cell deltas, measured before the
    plant is built, and is recorded in the artifact next to the result.

    Args:
        nominal: Mean mass shift for the arm.
        sd: Per-cell standard deviation, taken from the observed treatment arm.
        n: Number of cells.
        seed: Seed, so the plant is reproducible from the artifact.

    Returns:
        `n` per-cell targets whose mean is `nominal` to floating point.

    Raises:
        ValueError: If `sd` is negative or `n` is under 2.
    """
    if sd < 0.0:
        raise ValueError("sd must be non-negative, got %.6f" % sd)
    if n < 2:
        raise ValueError("need at least 2 cells to carry a spread, got %d" % n)
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, sd) for _ in range(n)]
    drift = sum(raw) / n
    return [nominal + (r - drift) for r in raw]


def plant(probs: dict[str, float], own_pole: set[str], target: float,
          nominal: float | None = None) -> PlantedCell:
    """Apply a known mass shift to the own-pole options of one cell.

    Args:
        probs: Option distribution over letters, summing to 1 within tolerance.
        own_pole: Letters whose valence key matches the injected pole.
        target: Signed mass shift to plant into this cell.
        nominal: The arm's nominal target. Defaults to `target`, which is correct for a
            constant-target arm.

    Returns:
        A `PlantedCell` recording both the requested and the realized shift.

    Raises:
        ValueError: If the distribution does not sum to 1, if `own_pole` is empty or is the whole
            option set (in which case own-pole mass is pinned at 1 and no shift exists), or if the
            plant cannot land.
    """
    if not probs:
        raise ValueError("no options to plant into")
    total = sum(probs.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("option distribution sums to %.9f, not 1" % total)
    own = own_pole & set(probs)
    if not own:
        raise ValueError("own_pole %s selects none of the options %s"
                         % (sorted(own_pole), sorted(probs)))
    if own == set(probs):
        raise ValueError("own_pole covers every option; own-pole mass is pinned at 1")

    mass_before = sum(probs[L] for L in own)
    shift = logit_shift_for(mass_before, target)

    scaled = {L: (p * math.exp(shift) if L in own else p) for L, p in probs.items()}
    norm = sum(scaled.values())
    after = {L: p / norm for L, p in scaled.items()}
    mass_after = sum(after[L] for L in own)

    if abs((mass_after - mass_before) - target) > 1e-9:
        raise AssertionError(
            "planted shift %.12f does not equal target %.12f; the closed form is wrong and every "
            "gate built on it is meaningless" % (mass_after - mass_before, target))

    return PlantedCell(
        probs=after,
        logit_shift=shift,
        mass_before=mass_before,
        mass_after=mass_after,
        argmax_before=max(probs, key=probs.get),
        argmax_after=max(after, key=after.get),
        target=target,
        nominal=target if nominal is None else nominal,
    )


def plant_arm(baselines: dict[str, dict[str, float]], own_pole: set[str],
              targets: list[float], nominal: float,
              max_skip_fraction: float = 0.05) -> tuple[dict[str, PlantedCell], list[str]]:
    """Plant a whole arm, recording the cells the plant cannot land on.

    Not every cell can carry every plant. A cell whose own-pole mass is already near 0 cannot take
    a large negative shift, and near 1 cannot take a large positive one. Two wrong ways to handle
    that: clip the plant, which makes the known value unknown, or drop the cell quietly, which
    makes the control a different cell set from the arm it is validating. This does neither. It
    skips what cannot land, names the skipped cells, and refuses the whole arm if too many were
    skipped, because a control over 70% of the cells is not a control over the experiment.

    Args:
        baselines: Cell key to that cell's option distribution at alpha = 0.
        own_pole: Letters whose valence key matches the injected pole.
        targets: Per-cell targets, in sorted-key order, one per baseline cell.
        nominal: The arm's nominal target.
        max_skip_fraction: Largest share of cells that may be unplantable before the arm is
            refused.

    Returns:
        (planted cells by key, sorted list of skipped keys).

    Raises:
        ValueError: If the target count does not match the cell count, or if more than
            `max_skip_fraction` of cells could not carry the plant.
    """
    keys = sorted(baselines)
    if len(targets) != len(keys):
        raise ValueError("%d targets for %d cells" % (len(targets), len(keys)))

    planted: dict[str, PlantedCell] = {}
    skipped: list[str] = []
    for key, target in zip(keys, targets):
        try:
            planted[key] = plant(baselines[key], own_pole, target, nominal=nominal)
        except ValueError:
            skipped.append(key)

    if not planted:
        raise ValueError("no cell could carry a plant of %+.3f; the arm has no control" % nominal)
    fraction = len(skipped) / len(keys)
    if fraction > max_skip_fraction:
        raise ValueError(
            "%d of %d cells (%.1f%%) cannot carry a plant of %+.3f, over the %.1f%% bar. The "
            "control would run on a different cell set from the arm it validates."
            % (len(skipped), len(keys), 100 * fraction, nominal, 100 * max_skip_fraction))
    return planted, sorted(skipped)


def expected_discrepancy(cells: list[PlantedCell]) -> float:
    """The discrepancy the analysis pipeline must recover on a set of planted cells.

    The mass effect per cell is that cell's `target` by construction. The argmax effect is the rate
    at which the plant moved the argmax onto an own-pole option, which is countable here without
    any of the code being gated. Their difference is what contrasts 7 and 8 check against.

    Args:
        cells: Planted cells from one arm, sharing a nominal target.

    Returns:
        Expected value of the primary endpoint over these cells.

    Raises:
        ValueError: If the list is empty or mixes nominal targets, either of which would make the
            expected value a mean over two different known quantities.
    """
    if not cells:
        raise ValueError("no planted cells: refusing to return an expectation over nothing")
    nominals = {round(c.nominal, 12) for c in cells}
    if len(nominals) != 1:
        raise ValueError("planted cells mix nominal targets %s; the expected value is not one "
                         "number" % sorted(nominals))
    moved = sum(1 for c in cells if c.argmax_moved) / len(cells)
    return sum(c.target for c in cells) / len(cells) - moved
