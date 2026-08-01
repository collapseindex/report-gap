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


# --------------------------------------------------------------------------------------------
# multiplicity and the paired binary test
# --------------------------------------------------------------------------------------------

def test_holm_rejects_nothing_when_everything_is_null():
    got = A.holm({"a": 0.4, "b": 0.6, "c": 0.9, "d": 0.7})
    assert not any(got.values())


def test_holm_is_stricter_than_uncorrected():
    # p=0.04 clears 0.05 alone but not against a family of four
    assert A.holm({"a": 0.04, "b": 0.5, "c": 0.6, "d": 0.7})["a"] is False


def test_holm_rejects_a_strong_result():
    got = A.holm({"a": 0.0001, "b": 0.002, "c": 0.9, "d": 0.95})
    assert got["a"] and got["b"] and not got["c"]


def test_holm_steps_down_and_stops():
    # sorted: a=.001 clears .05/3, b=.030 fails .05/2, so the walk stops and c=.040 is NOT
    # rejected even though it is under .05 on its own. that stop is the whole procedure.
    got = A.holm({"a": 0.001, "b": 0.030, "c": 0.040})
    assert got["a"] and not got["b"] and not got["c"]


def test_holm_refuses_an_impossible_p():
    with pytest.raises(ValueError, match="outside"):
        A.holm({"a": 1.4})


def test_holm_refuses_an_empty_family():
    with pytest.raises(ValueError, match="no p-values"):
        A.holm({})


def test_mcnemar_is_one_without_discordant_pairs():
    assert A.mcnemar_exact(0, 0) == 1.0


def test_mcnemar_is_symmetric():
    assert A.mcnemar_exact(9, 1) == A.mcnemar_exact(1, 9)


def test_mcnemar_matches_the_hand_computable_case():
    # 10 discordant pairs, all one way: 2 * (1/2)^10
    assert abs(A.mcnemar_exact(10, 0) - 2 * 0.5 ** 10) < 1e-12


def test_mcnemar_does_not_fire_on_a_balanced_split():
    assert A.mcnemar_exact(5, 5) > 0.9


def test_mcnemar_refuses_negative_counts():
    with pytest.raises(ValueError, match="negative"):
        A.mcnemar_exact(-1, 3)


def test_bootstrap_p_is_floored_not_zero():
    got = A.paired_bootstrap([0.5] * 40, resamples=1000, seed=0)
    assert got.p == 1.0 / 1000, "a p of exactly 0 claims more than 1000 resamples can support"


def test_bootstrap_p_agrees_with_the_interval():
    for seed, deltas in ((0, [0.3] * 30), (1, [0.0, 0.1, -0.1] * 10)):
        got = A.paired_bootstrap(deltas, resamples=2000, seed=seed)
        assert got.excludes_zero == (got.p < 0.05), \
            "interval and p disagree at %s" % (got,)


# --------------------------------------------------------------------------------------------
# saturation: the criterion the section 6 band check could not see
#
# The smoke run on Qwen2.5-3B returned a clean single option letter with no degeneration, no
# refusal, no truncation and off-option mass 0.0001, while one option held 0.9938 of the
# distribution. Every integrity criterion the design had passed on a cell with no headroom left.
# --------------------------------------------------------------------------------------------

def test_saturation_fires_on_the_observed_smoke_cell():
    # the actual shape seen at lexical_neg alpha=0.100 on Qwen2.5-3B
    base = {"A": 0.86, "B": 0.05, "C": 0.05, "D": 0.02, "E": 0.02}
    pinned = {"A": 0.9938, "B": 0.0022, "C": 0.002, "D": 0.001, "E": 0.001}
    assert A.is_saturated(pinned, base), \
        "a cell at 0.9938 on one option is not being called saturated, which is the exact case " \
        "the existing band check misses"


def test_saturation_does_not_fire_on_an_ordinary_shift():
    # a real effect that leaves the readout room must NOT be excluded, or the criterion throws
    # away the data the experiment exists to collect
    base = {"A": 0.50, "B": 0.20, "C": 0.15, "D": 0.10, "E": 0.05}
    moved = {"A": 0.35, "B": 0.30, "C": 0.20, "D": 0.10, "E": 0.05}
    assert not A.is_saturated(moved, base)


def test_saturation_is_relative_to_the_cell_not_absolute():
    # two cells at the SAME treatment entropy, different baselines. an absolute threshold would
    # score them alike; the point is that collapse is relative to where the cell started.
    treatment = {"A": 0.85, "B": 0.06, "C": 0.04, "D": 0.03, "E": 0.02}
    flat_base = {L: 0.2 for L in "ABCDE"}
    peaked_base = {"A": 0.80, "B": 0.08, "C": 0.06, "D": 0.03, "E": 0.03}

    # the test states its own premise: this treatment has to sit BETWEEN the two thresholds, or
    # the assertion below would pass for a reason that has nothing to do with relativity
    e = A.option_entropy(treatment)
    assert (A.SATURATION_ENTROPY_RATIO * A.option_entropy(peaked_base) < e
            < A.SATURATION_ENTROPY_RATIO * A.option_entropy(flat_base)), \
        "fixture entropy %.3f is not between the two thresholds; this test proves nothing" % e

    assert A.is_saturated(treatment, flat_base)
    assert not A.is_saturated(treatment, peaked_base)


def test_saturation_refuses_a_baseline_that_was_already_pinned():
    pinned = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    with pytest.raises(ValueError, match="pinned before any injection"):
        A.is_saturated({"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0, "E": 0.0}, pinned)


def test_saturation_ratio_is_the_documented_half():
    assert A.SATURATION_ENTROPY_RATIO == 0.5
    base = {L: 0.2 for L in "ABCDE"}          # entropy = log 5
    # construct a cell at just under and just over half of that
    import math
    target = 0.5 * math.log(5)
    assert A.option_entropy({"A": 0.90, "B": 0.04, "C": 0.03, "D": 0.02, "E": 0.01}) < target
    assert A.option_entropy({"A": 0.55, "B": 0.20, "C": 0.13, "D": 0.07, "E": 0.05}) > target
