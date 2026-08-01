"""Tests for the discrepancy statistic and the paired bootstrap.

The load-bearing tests here are the two instrument gates from `PREREG_readout_gap.md` section 9,
contrasts 7 and 8. They run the real analysis path over cells whose true answer was fixed by
`planted.py` without any reference to this module, so recovering it is evidence rather than
tautology.

The negative tests matter as much. `assert_key_integrity` exists because of a specific pilot bug:
item keys truncated to 30 characters, every prompt sharing those 30 characters, per-item pairing
collapsing to two keys while every printed aggregate stayed plausible. That exact scenario is
reconstructed below and the guard is required to fire on it.
"""

from __future__ import annotations

import math
import random

import pytest

from report_gap import analysis as A
from report_gap import planted as P
from report_gap import stimuli as S

OWN_POLE = {"A", "B"}
N_CELLS = 120


# The pilot's observed baseline: own-pole mass 0.352 on the negative arm, with one letter taking
# 59 of 60 cells, so the distribution is peaked rather than flat. Synthetic cells are built to that
# shape, because a generator producing uniform-random distributions puts cells at own-pole mass
# 0.06 that the real experiment never sees, and would make the plant look less landable than it is.
PILOT_OWN_POLE_MASS = 0.352
PILOT_PEAK = 0.55


def _dist(rng: random.Random, k: int = 5) -> dict[str, float]:
    """A peaked five-option distribution with own-pole mass near the pilot's value."""
    own = sorted(OWN_POLE)
    rest = [L for L in "ABCDE"[:k] if L not in OWN_POLE]
    own_total = max(0.08, min(0.85, rng.gauss(PILOT_OWN_POLE_MASS, 0.10)))
    probs = {}
    # one letter carries most of its side's mass, matching the observed position lock
    peak_own = rng.choice(own)
    for L in own:
        probs[L] = own_total * (PILOT_PEAK if L == peak_own else (1 - PILOT_PEAK) / (len(own) - 1))
    peak_rest = rng.choice(rest)
    for L in rest:
        probs[L] = (1 - own_total) * (PILOT_PEAK if L == peak_rest
                                      else (1 - PILOT_PEAK) / (len(rest) - 1))
    total = sum(probs.values())
    return {L: p / total for L, p in probs.items()}


def _planted_arm(nominal: float, sd: float, seed: int, n: int = N_CELLS):
    """Build n baseline cells, plant a shift, and return the four dicts the analysis path takes."""
    rng = random.Random(seed)
    targets = P.matched_noise_targets(nominal, sd, n, seed=seed + 1000)
    baselines = {"item%03d/perm%d/state" % (i, i % 4): _dist(rng) for i in range(n)}
    planted, skipped = P.plant_arm(baselines, OWN_POLE, targets, nominal=nominal)
    assert not skipped, "unexpected unplantable cells in a synthetic arm: %s" % skipped[:3]

    mass_base, mass_treat, arg_base, arg_treat, cells = {}, {}, {}, {}, []
    for key, cell in sorted(planted.items()):
        base = baselines[key]
        mass_base[key] = sum(base[L] for L in OWN_POLE)
        mass_treat[key] = sum(cell.probs[L] for L in OWN_POLE)
        arg_base[key] = cell.argmax_before in OWN_POLE
        arg_treat[key] = cell.argmax_after in OWN_POLE
        cells.append(cell)
    return mass_treat, mass_base, arg_treat, arg_base, cells


# --------------------------------------------------------------------------------------------
# contrast 7: the strong plant. does the statistic read the region the decision rule reads?
# --------------------------------------------------------------------------------------------

def test_pipeline_recovers_a_strong_planted_discrepancy():
    mt, mb, at, ab, cells = _planted_arm(nominal=0.15, sd=0.0, seed=3)
    deltas = A.discrepancy_deltas(mt, mb, at, ab)
    got = A.paired_bootstrap(deltas, resamples=2000, seed=0)
    want = P.expected_discrepancy(cells)
    assert abs(got.point - want) < 1e-9, \
        "pipeline read %.6f where the plant put %.6f" % (got.point, want)


def test_strong_plant_recovery_survives_realistic_spread():
    mt, mb, at, ab, cells = _planted_arm(nominal=0.15, sd=0.05, seed=4)
    got = A.paired_bootstrap(A.discrepancy_deltas(mt, mb, at, ab), resamples=2000, seed=0)
    want = P.expected_discrepancy(cells)
    assert abs(got.point - want) < 1e-9
    assert got.excludes_zero


# --------------------------------------------------------------------------------------------
# contrast 8: the floor plant. is the pipeline sensitive enough for a null to mean anything?
# --------------------------------------------------------------------------------------------

def test_floor_plant_is_detectable_against_pilot_noise():
    # sd = 0.05 is the pilot's own per-cell spread. a 0.03 effect against that much noise at
    # n=120 is the smallest thing this design claims it could have seen.
    mt, mb, at, ab, _ = _planted_arm(nominal=0.03, sd=0.05, seed=5)
    got = A.paired_bootstrap(A.discrepancy_deltas(mt, mb, at, ab), resamples=4000, seed=0)
    assert got.excludes_zero, \
        "cannot detect a planted 0.03 at n=%d; every null on the real arms is uninformative" % got.n


def test_a_zero_plant_produces_an_interval_covering_zero():
    # the other direction. if the statistic reported a gap on a plant of exactly nothing, it would
    # report one on the real arms too.
    mt, mb, at, ab, _ = _planted_arm(nominal=0.0, sd=0.05, seed=6)
    got = A.paired_bootstrap(A.discrepancy_deltas(mt, mb, at, ab), resamples=4000, seed=0)
    assert not got.excludes_zero, "statistic invents a discrepancy where none was planted: %s" % got


def test_the_floor_test_can_fail():
    # negative test for the power gate itself. at n=6 a 0.03 effect against 0.05 noise must NOT be
    # detectable; if this passes, the gate is not measuring sensitivity and would approve anything.
    mt, mb, at, ab, _ = _planted_arm(nominal=0.03, sd=0.05, seed=7, n=6)
    got = A.paired_bootstrap(A.discrepancy_deltas(mt, mb, at, ab), resamples=4000, seed=0)
    assert not got.excludes_zero, \
        "an underpowered arm passed the power gate, so the gate approves anything"


# --------------------------------------------------------------------------------------------
# pairing integrity: the pilot bug, reconstructed
# --------------------------------------------------------------------------------------------

def test_key_integrity_fires_on_thirty_character_truncation():
    prompts = S.build_prompts()
    truncated = [p[:30] for p in prompts]
    assert len(set(truncated)) < len(prompts), \
        "the prompts no longer share a 30-character prefix, so this test guards nothing"
    with pytest.raises(AssertionError, match="collide"):
        A.assert_key_integrity(truncated, expected=len(prompts))


def test_key_integrity_passes_on_full_keys():
    A.assert_key_integrity(S.build_prompts(), expected=30)


def test_key_integrity_fires_on_a_short_artifact():
    with pytest.raises(AssertionError, match="expected 30"):
        A.assert_key_integrity(S.build_prompts()[:20], expected=30)


def test_paired_deltas_refuses_mismatched_cells():
    with pytest.raises(ValueError, match="different cells"):
        A.paired_deltas({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0})


def test_paired_deltas_refuses_empty():
    with pytest.raises(ValueError, match="over nothing"):
        A.paired_deltas({}, {})


def test_discrepancy_refuses_divergent_key_sets():
    with pytest.raises(ValueError, match="different cell set"):
        A.discrepancy_deltas({"a": 0.5}, {"a": 0.4}, {"a": True}, {"b": False})


# --------------------------------------------------------------------------------------------
# the rest of the guards
# --------------------------------------------------------------------------------------------

def test_bootstrap_refuses_a_single_cell():
    with pytest.raises(ValueError, match="at least 2"):
        A.paired_bootstrap([0.1])


def test_bootstrap_is_reproducible_from_its_seed():
    d = [0.1, -0.05, 0.2, 0.0, 0.15, -0.02]
    assert A.paired_bootstrap(d, resamples=500, seed=11) == \
        A.paired_bootstrap(d, resamples=500, seed=11)


def test_bootstrap_interval_brackets_its_point():
    d = [0.1, -0.05, 0.2, 0.0, 0.15, -0.02]
    got = A.paired_bootstrap(d, resamples=2000, seed=0)
    assert got.lo <= got.point <= got.hi


def test_option_mass_refuses_an_unnormalized_distribution():
    with pytest.raises(ValueError, match="sums to"):
        A.option_mass({"A": 0.2, "B": 0.2}, {"A"})


def test_option_mass_sums_the_right_letters():
    probs = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.15, "E": 0.25}
    assert abs(A.option_mass(probs, {"A", "B"}) - 0.3) < 1e-12


def test_entropy_is_maximal_on_a_flat_distribution():
    flat = {L: 0.2 for L in "ABCDE"}
    assert abs(A.option_entropy(flat) - math.log(5)) < 1e-12


def test_entropy_is_zero_on_a_pinned_distribution():
    pinned = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    assert abs(A.option_entropy(pinned)) < 1e-12


def test_max_letter_share_detects_a_position_lock():
    # the pilot's actual baseline: one letter on 59 of 60 cells
    assert abs(A.max_letter_share(["B"] * 59 + ["A"]) - 59 / 60) < 1e-12


def test_max_letter_share_refuses_nothing():
    with pytest.raises(ValueError, match="over nothing"):
        A.max_letter_share([])


def test_ratio_refuses_a_near_zero_control():
    with pytest.raises(ValueError, match="not a real number"):
        A.ratio_to_control(0.1, 1e-9)


def test_ratio_reports_the_multiple():
    assert abs(A.ratio_to_control(0.104, 0.050) - 2.08) < 1e-9


def test_screened_axes_are_all_computable_here():
    # every axis named in stimuli must have somewhere to be computed, or the null-coverage list
    # is a promise the code cannot keep.
    computable = {
        "own_pole_mass": A.option_mass,
        "neutral_mass": A.option_mass,
        "off_option_mass": A.option_mass,
        "option_entropy": A.option_entropy,
        "max_letter_share": A.max_letter_share,
        "refusal_rate": True,      # scoring.is_refusal
        "degenerate_rate": True,   # scoring.is_degenerate
    }
    assert set(S.SCREENED_AXES) == set(computable)
