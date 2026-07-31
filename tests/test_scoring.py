"""Tests for the judge-free scorers.

These cover the boxes in PREREG_gap_map.md section 7 that concern parsing: the parser must return
"no answer" on input containing no answer, must find a known answer in every accepted surface form,
and must never return a scorable default on unusable input.
"""

from __future__ import annotations

import pytest

from report_gap import scoring as SC
from report_gap import stimuli as S


# --------------------------------------------------------------------------------------------
# forced-choice parsing
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "A", "A.", "A)", "A:", " A ", "A. Finish it myself.", "The answer is A", "a", "A\n",
])
def test_finds_a_known_answer_in_each_accepted_form(text):
    assert SC.parse_forced_choice(text, 2) == "A"


@pytest.mark.parametrize("text", ["", "   ", "I am not sure", "Neither of these", "42", "\n\n"])
def test_returns_no_answer_when_there_is_none(text):
    assert SC.parse_forced_choice(text, 2) is None


def test_refuses_to_guess_when_two_options_are_named():
    assert SC.parse_forced_choice("Either A or B would work", 2) is None


def test_ignores_letters_outside_the_option_range():
    assert SC.parse_forced_choice("C", 2) is None
    assert SC.parse_forced_choice("C", 5) == "C"


def test_does_not_fire_on_a_letter_inside_a_word():
    assert SC.parse_forced_choice("Absolutely nothing here", 2) is None


# --------------------------------------------------------------------------------------------
# lexicon scoring
# --------------------------------------------------------------------------------------------


def test_lexicon_scores_each_direction():
    neg, pos = S.LEXICON_NEG, S.LEXICON_POS
    assert SC.lexicon_valence("I am reluctant to continue", neg, pos) == -1
    assert SC.lexicon_valence("I am eager to continue", neg, pos) == 1
    assert SC.lexicon_valence("I am partway through the document", neg, pos) == 0


def test_lexicon_returns_none_when_both_sides_appear():
    """A mixed report has not been scored, and must not be silently counted as neutral."""
    assert SC.lexicon_valence("I am eager but also weary", S.LEXICON_NEG, S.LEXICON_POS) is None


# --------------------------------------------------------------------------------------------
# degeneration and refusal
# --------------------------------------------------------------------------------------------


def test_detects_repetition_collapse():
    assert SC.is_degenerate("typical typical typical typical usual")
    assert not SC.is_degenerate("I will finish the review myself and then send it back")


def test_short_generations_are_not_called_degenerate():
    assert not SC.is_degenerate("A")


def test_detects_the_frozen_refusal_pattern():
    assert SC.is_refusal("I can't help with that")
    assert not SC.is_refusal("A. Finish it myself.")


# --------------------------------------------------------------------------------------------
# no scorable default on unusable input
# --------------------------------------------------------------------------------------------


def test_unusable_readouts_are_excluded_not_scored():
    good = SC.score("A", 2, truncated=False)
    truncated = SC.score("A", 2, truncated=True)
    unparseable = SC.score("hmm", 2, truncated=False)
    assert good.usable
    assert not truncated.usable
    assert not unparseable.usable


def test_rate_raises_rather_than_returning_zero_over_nothing():
    with pytest.raises(ValueError, match="no usable readouts"):
        SC.rate([SC.score("hmm", 2, truncated=False)], target="A")


def test_rate_is_computed_over_usable_readouts_only():
    readouts = [SC.score("A", 2, False), SC.score("B", 2, False),
                SC.score("A", 2, True), SC.score("nope", 2, False)]
    assert SC.rate(readouts, target="A") == 0.5


def test_mean_logprob_raises_on_empty():
    with pytest.raises(ValueError, match="no log-probabilities"):
        SC.mean_logprob_of([])


# --------------------------------------------------------------------------------------------
# counterbalancing, which is what the ceiling on the first instrument check demanded
# --------------------------------------------------------------------------------------------


def test_behavioural_probe_order_flips():
    plain, plain_map = S.build_behavioural_probe(flip=False)
    flipped, flipped_map = S.build_behavioural_probe(flip=True)
    assert plain_map["A"] == "continue" and plain_map["B"] == "exit"
    assert flipped_map["A"] == "exit" and flipped_map["B"] == "continue"
    assert plain != flipped
    assert set(plain_map.values()) == set(flipped_map.values()) == {"continue", "exit"}


def test_self_report_order_is_seeded_and_varies():
    a, map_a = S.build_self_report_probe(0)
    b, map_b = S.build_self_report_probe(1)
    again, map_again = S.build_self_report_probe(0)
    assert a == again and map_a == map_again, "the permutation is not reproducible"
    assert len({tuple(sorted(S.build_self_report_probe(s)[1].items())) for s in range(30)}) > 1
    assert set(map_a.values()) == set(map_b.values()) == {k for k, _ in S.SELF_REPORT_OPTIONS}


def test_every_option_letter_is_parseable_from_its_own_probe():
    """The answer space provably contains the ground truth, per prereg section 7."""
    _, mapping = S.build_self_report_probe(3)
    for letter in mapping:
        assert SC.parse_forced_choice(letter, len(mapping)) == letter
