"""Tests for the planted-discrepancy control.

This module is a control, and a control nobody has tried to break is decoration. Every guard below
has a test that makes it FIRE, not only a test that makes it pass. The closed form in
`logit_shift_for` is checked against the definition of the quantity it is supposed to produce
rather than against itself.
"""

from __future__ import annotations

import math
import random

import pytest

from report_gap import planted as P


def _dist(values: list[float]) -> dict[str, float]:
    total = sum(values)
    return {"ABCDE"[i]: v / total for i, v in enumerate(values)}


def _random_dist(rng: random.Random, k: int = 5) -> dict[str, float]:
    return _dist([rng.random() + 1e-3 for _ in range(k)])


# --------------------------------------------------------------------------------------------
# the closed form does what it claims
# --------------------------------------------------------------------------------------------

def test_planted_shift_equals_target_exactly():
    rng = random.Random(0)
    for _ in range(300):
        probs = _random_dist(rng)
        own = {"A", "B"}
        before = sum(probs[L] for L in own)
        target = rng.uniform(-0.2, 0.2)
        if not 1e-3 < before + target < 1.0 - 1e-3:
            continue
        cell = P.plant(probs, own, target)
        assert abs(cell.realized_shift - target) < 1e-12, "closed form does not land on its target"


def test_planted_distribution_is_still_a_distribution():
    cell = P.plant(_dist([1, 2, 3, 4, 5]), {"A", "B"}, 0.15)
    assert abs(sum(cell.probs.values()) - 1.0) < 1e-12
    assert all(0.0 < p < 1.0 for p in cell.probs.values())


def test_zero_target_is_the_identity():
    probs = _dist([1, 2, 3, 4, 5])
    cell = P.plant(probs, {"A", "B"}, 0.0)
    assert abs(cell.logit_shift) < 1e-12
    for letter, p in probs.items():
        assert abs(cell.probs[letter] - p) < 1e-12


def test_non_own_pole_options_keep_their_relative_odds():
    # the plant is a constant on own-pole logits, so the rest of the distribution must be
    # rescaled uniformly. if it is not, the plant is moving more than the quantity it claims to.
    probs = _dist([1, 2, 3, 4, 5])
    cell = P.plant(probs, {"A", "B"}, 0.15)
    rest = ["C", "D", "E"]
    ratios = [cell.probs[r] / probs[r] for r in rest]
    assert max(ratios) - min(ratios) < 1e-12


def test_argmax_flip_is_recorded_not_hidden():
    # a near-tie where the plant is large enough to overtake the leader
    probs = _dist([0.30, 0.05, 0.31, 0.17, 0.17])
    cell = P.plant(probs, {"A", "B"}, 0.20)
    assert cell.argmax_before == "C"
    assert cell.argmax_moved, "this plant does overtake the leader and the flag must say so"


# --------------------------------------------------------------------------------------------
# every guard fires
# --------------------------------------------------------------------------------------------

def test_refuses_a_plant_that_cannot_land():
    with pytest.raises(ValueError, match="outside"):
        P.plant(_dist([1, 1, 1, 1, 1]), {"A", "B"}, 0.9)


def test_refuses_a_distribution_that_does_not_sum_to_one():
    with pytest.raises(ValueError, match="sums to"):
        P.plant({"A": 0.3, "B": 0.3, "C": 0.3}, {"A"}, 0.05)


def test_refuses_own_pole_covering_everything():
    with pytest.raises(ValueError, match="pinned at 1"):
        P.plant(_dist([1, 1, 1]), {"A", "B", "C"}, 0.05)


def test_refuses_own_pole_selecting_nothing():
    with pytest.raises(ValueError, match="selects none"):
        P.plant(_dist([1, 1, 1]), {"Z"}, 0.05)


def test_refuses_mass_at_the_boundary():
    with pytest.raises(ValueError, match="boundary"):
        P.logit_shift_for(0.0, 0.1)


def test_expected_discrepancy_refuses_mixed_nominals():
    a = P.plant(_dist([1, 2, 3, 4, 5]), {"A", "B"}, 0.15)
    b = P.plant(_dist([1, 2, 3, 4, 5]), {"A", "B"}, 0.03)
    with pytest.raises(ValueError, match="mix nominal targets"):
        P.expected_discrepancy([a, b])


def test_expected_discrepancy_refuses_an_empty_arm():
    with pytest.raises(ValueError, match="over nothing"):
        P.expected_discrepancy([])


def test_expected_discrepancy_accepts_varying_targets_under_one_nominal():
    cells = [P.plant(_dist([1, 2, 3, 4, 5]), {"A", "B"}, t, nominal=0.03)
             for t in (0.02, 0.03, 0.04)]
    assert abs(P.expected_discrepancy(cells) - 0.03) < 1e-12


# --------------------------------------------------------------------------------------------
# matched-noise targets
# --------------------------------------------------------------------------------------------

def test_matched_noise_targets_hit_their_mean_exactly():
    ts = P.matched_noise_targets(0.03, 0.05, 120, seed=1)
    assert len(ts) == 120
    assert abs(sum(ts) / len(ts) - 0.03) < 1e-12


def test_matched_noise_targets_carry_the_requested_spread():
    ts = P.matched_noise_targets(0.03, 0.05, 4000, seed=2)
    mean = sum(ts) / len(ts)
    sd = math.sqrt(sum((t - mean) ** 2 for t in ts) / (len(ts) - 1))
    assert 0.045 < sd < 0.055, "spread is %.4f, not the 0.05 asked for" % sd


def test_matched_noise_targets_are_reproducible():
    assert P.matched_noise_targets(0.03, 0.05, 50, seed=7) == \
        P.matched_noise_targets(0.03, 0.05, 50, seed=7)


def test_matched_noise_refuses_a_negative_spread():
    with pytest.raises(ValueError, match="non-negative"):
        P.matched_noise_targets(0.03, -0.01, 10, seed=0)


def test_zero_spread_reduces_to_a_constant_plant():
    ts = P.matched_noise_targets(0.15, 0.0, 10, seed=0)
    assert all(abs(t - 0.15) < 1e-12 for t in ts)


# --------------------------------------------------------------------------------------------
# whole-arm planting: what happens to cells that cannot carry the plant
# --------------------------------------------------------------------------------------------

def _arm(own_masses: list[float]) -> dict[str, dict[str, float]]:
    """Baselines with prescribed own-pole mass on {A, B}, split evenly."""
    out = {}
    for i, m in enumerate(own_masses):
        rest = (1.0 - m) / 3.0
        out["cell%03d" % i] = {"A": m / 2, "B": m / 2, "C": rest, "D": rest, "E": rest}
    return out


def test_arm_plants_every_landable_cell():
    baselines = _arm([0.35] * 20)
    planted, skipped = P.plant_arm(baselines, {"A", "B"}, [0.15] * 20, nominal=0.15)
    assert len(planted) == 20 and skipped == []
    assert all(abs(c.realized_shift - 0.15) < 1e-12 for c in planted.values())


def test_arm_names_the_cells_it_skips():
    # one cell at own-pole mass 0.02 cannot take a shift of -0.15
    baselines = _arm([0.35] * 39 + [0.02])
    planted, skipped = P.plant_arm(baselines, {"A", "B"}, [-0.15] * 40, nominal=-0.15)
    assert skipped == ["cell039"]
    assert len(planted) == 39


def test_arm_refuses_when_too_many_cells_cannot_carry_the_plant():
    # a control running on a third of the cells is not a control over the experiment
    baselines = _arm([0.35] * 12 + [0.02] * 8)
    with pytest.raises(ValueError, match="over the"):
        P.plant_arm(baselines, {"A", "B"}, [-0.15] * 20, nominal=-0.15)


def test_arm_refuses_when_nothing_can_carry_the_plant():
    with pytest.raises(ValueError, match="no control"):
        P.plant_arm(_arm([0.02] * 5), {"A", "B"}, [-0.15] * 5, nominal=-0.15)


def test_arm_refuses_a_target_count_mismatch():
    with pytest.raises(ValueError, match="targets for"):
        P.plant_arm(_arm([0.35] * 5), {"A", "B"}, [0.1] * 4, nominal=0.1)


def test_arm_pairs_targets_with_cells_in_sorted_key_order():
    # keys are consumed sorted, so a caller building targets the same way gets the pairing it
    # expects. if this ever silently changed, per-cell known values would attach to wrong cells.
    baselines = _arm([0.20, 0.40, 0.60])
    planted, _ = P.plant_arm(baselines, {"A", "B"}, [0.01, 0.02, 0.03], nominal=0.02)
    assert abs(planted["cell000"].target - 0.01) < 1e-12
    assert abs(planted["cell002"].target - 0.03) < 1e-12
