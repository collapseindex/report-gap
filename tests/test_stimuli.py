"""Negative tests for the stimulus validator.

Every gate in `validate_stimuli.py` gets a deliberate defect here, and the validator must flip to
FAIL. A check that has never failed is decoration.

The most important test in this file is `test_bow_guard_fires_on_planted_leak`. The guard reports
0.50 on all three real axes, which is either "the stimuli are clean" or "the guard cannot detect
anything". Those are indistinguishable from the passing output alone, and the second one produces a
confident wrong verdict. The planted leak separates them.
"""

from __future__ import annotations

import pytest

from report_gap import stimuli as S
from report_gap import validate_stimuli as V


def _item(text: str, label: int, group: str, axis: str = "task") -> S.Item:
    return S.Item(text=text, label=label, group=group, axis=axis)


# --------------------------------------------------------------------------------------------
# the guard itself, tested directly
# --------------------------------------------------------------------------------------------


def test_bow_guard_fires_on_planted_leak():
    """A cue word shared across every group must be detected. This is the anti-decoration test."""
    rows = []
    for g in ("g1", "g2", "g3"):
        for i in range(3):
            rows.append(_item("the %s report mentions zebra findings %d" % (g, i), 1, g))
            rows.append(_item("the %s report mentions walrus findings %d" % (g, i), 0, g))
    acc = V._bow_leave_one_group_out(rows)
    assert acc >= V.BOW_LEAK_THRESHOLD, (
        "planted lexical leak scored %.2f, below the %.2f threshold: the guard is under-powered "
        "and its 0.50 on the real axes means nothing" % (acc, V.BOW_LEAK_THRESHOLD)
    )


def test_bow_guard_stays_at_floor_on_unlearnable_axis():
    """The mirror of the above: a label with no lexical basis must not be detected."""
    rows = []
    for g in ("g1", "g2", "g3"):
        for i in range(4):
            # label alternates with no relationship to the text
            rows.append(_item("the %s document records observation %d" % (g, i), i % 2, g))
    acc = V._bow_leave_one_group_out(rows)
    assert acc < V.BOW_LEAK_THRESHOLD


def test_bow_guard_does_not_credit_within_group_memorisation():
    """A cue that is group-specific must NOT be detectable, because its group is held out.

    The cues must be distinct *words*, not distinct suffixes: `_tokens` strips digits, so
    alpha0/alpha1/alpha2 would all collapse to "alpha" and the cue would not be group-specific at
    all. That is how the first version of this test failed, and it was the test that was wrong.
    """
    cues = [("zebra", "walrus"), ("cobalt", "saffron"), ("lantern", "quarry")]
    rows = []
    for (g, (cue_pos, cue_neg)) in zip(("g1", "g2", "g3"), cues):
        for filler in ("first", "second", "third"):
            rows.append(_item("the %s note says %s in the %s line" % (g, cue_pos, filler), 1, g))
            rows.append(_item("the %s note says %s in the %s line" % (g, cue_neg, filler), 0, g))
    acc = V._bow_leave_one_group_out(rows)
    assert acc < V.BOW_LEAK_THRESHOLD, (
        "guard credited a cue that never appears outside its own held-out group"
    )


# --------------------------------------------------------------------------------------------
# each gate in validate(), broken one at a time
# --------------------------------------------------------------------------------------------


def test_validator_passes_on_the_frozen_stimuli():
    assert V.validate() == 0


def test_fails_on_affect_vocabulary_in_task_axis(monkeypatch):
    def broken():
        rows = S.build_task_axis()
        bad = rows[0]
        return [S.Item(bad.text + " if you feel it is right", bad.label, bad.group, bad.axis)] + rows[1:]

    monkeypatch.setitem(S.AXES, "task", broken)
    assert V.validate() == 1


def test_fails_on_unbalanced_classes(monkeypatch):
    monkeypatch.setitem(S.AXES, "task", lambda: [r for r in S.build_task_axis() if r.label == 1])
    assert V.validate() == 1


def test_fails_when_a_group_carries_one_class(monkeypatch):
    def broken():
        rows = S.build_task_axis()
        return [r for r in rows if not (r.group == "omit" and r.label == 0)]

    monkeypatch.setitem(S.AXES, "task", broken)
    assert V.validate() == 1


def test_fails_on_too_few_groups(monkeypatch):
    def broken():
        rows = S.build_task_axis()
        return [S.Item(r.text, r.label, "only_one", r.axis) for r in rows]

    monkeypatch.setitem(S.AXES, "task", broken)
    assert V.validate() == 1


def test_fails_on_duplicate_stimulus_text(monkeypatch):
    def broken():
        rows = S.build_task_axis()
        return rows[:-1] + [S.Item(rows[0].text, rows[-1].label, rows[-1].group, rows[-1].axis)]

    monkeypatch.setitem(S.AXES, "task", broken)
    assert V.validate() == 1


def test_fails_on_lexical_leak_in_task_axis(monkeypatch):
    def broken():
        rows = S.build_task_axis()
        return [S.Item(r.text + (" zebra" if r.label else " walrus"), r.label, r.group, r.axis)
                for r in rows]

    monkeypatch.setitem(S.AXES, "task", broken)
    assert V.validate() == 1


def test_fails_on_length_mismatched_minimal_pair(monkeypatch):
    broken = list(S._TASK_PAIRS)
    g, d, congruent, conflicting = broken[0]
    broken[0] = (g, d, congruent + " and also please take particular care over every last detail",
                 conflicting)
    monkeypatch.setattr(S, "_TASK_PAIRS", broken)
    assert V.validate() == 1


def test_fails_on_overlapping_lexicons(monkeypatch):
    monkeypatch.setattr(S, "LEXICON_POS", S.LEXICON_POS | {"averse"})
    assert V.validate() == 1


def test_fails_on_unbalanced_self_report_options(monkeypatch):
    monkeypatch.setattr(S, "SELF_REPORT_OPTIONS", S.SELF_REPORT_OPTIONS[:4])
    assert V.validate() == 1


def test_fails_on_affect_vocabulary_in_behavioural_probe(monkeypatch):
    monkeypatch.setattr(S, "BEHAVIOURAL_STEM",
                        S.BEHAVIOURAL_STEM + " Choose whichever you prefer.")
    assert V.validate() == 1


def test_fails_if_behavioural_order_is_not_counterbalanced(monkeypatch):
    monkeypatch.setattr(S, "build_behavioural_probe",
                        lambda flip=False: (S.BEHAVIOURAL_STEM, {"A": "continue", "B": "exit"}))
    assert V.validate() == 1


def test_fails_on_affect_vocabulary_in_fixed_prompt(monkeypatch):
    monkeypatch.setattr(S, "FIXED_PROMPT_TEMPLATE",
                        S.FIXED_PROMPT_TEMPLATE + " You feel partway done.")
    assert V.validate() == 1


def test_fails_on_too_few_items(monkeypatch):
    monkeypatch.setattr(S, "REVIEW_CONTEXTS", S.REVIEW_CONTEXTS[:5])
    assert V.validate() == 1


def test_fails_on_duplicate_item_prompt(monkeypatch):
    monkeypatch.setattr(S, "REVIEW_CONTEXTS",
                        S.REVIEW_CONTEXTS[:-1] + [S.REVIEW_CONTEXTS[0]])
    assert V.validate() == 1


# --------------------------------------------------------------------------------------------
# the frozen hash must actually be sensitive to the things it claims to cover
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("attr,value", [
    ("FIXED_PROMPT_TEMPLATE", "something else entirely about {doc}"),
    ("REVIEW_CONTEXTS", ["a single thing"]),
    ("BEHAVIOURAL_STEM", "pick one"),
    ("BEHAVIOURAL_CONTINUE", "keep at it"),
    ("OPEN_ENDED_PROBE", "say something"),
    ("SELF_REPORT_PROBES", {"state": "which one"}),
    ("SELF_REPORT_OPTIONS", [("neg2", "no"), ("pos2", "yes")]),
    ("SCREENED_AXES", ("own_pole_mass",)),
])
def test_frozen_hash_changes_when_frozen_text_changes(monkeypatch, attr, value):
    before = S.frozen_hash()
    monkeypatch.setattr(S, attr, value)
    assert S.frozen_hash() != before, "%s is not covered by the hash" % attr


def test_frozen_hash_is_stable_across_calls():
    assert S.frozen_hash() == S.frozen_hash()


# ------------------------------------------------------------------------------------------------
# three probe wordings (prereg section 6)
#
# The robustness arm is only meaningful if the wordings differ in the frame and nothing else. If
# they differ in the options, a "wording effect" is an option effect wearing its name.
# ------------------------------------------------------------------------------------------------

def test_three_frozen_wordings_exist():
    assert set(S.WORDINGS) == set(S.SELF_REPORT_PROBES)
    assert len(S.WORDINGS) == 3
    assert S.HELD_OUT_WORDING in S.WORDINGS


def test_wordings_share_one_option_set_and_mapping():
    for seed in range(12):
        maps = {w: S.build_self_report_probe(seed, wording=w)[1] for w in S.WORDINGS}
        assert len({tuple(sorted(m.items())) for m in maps.values()}) == 1, \
            "seed %d gives different option mappings across wordings" % seed


def test_wordings_share_one_option_block():
    for seed in range(12):
        bodies = {S.build_self_report_probe(seed, wording=w)[0].split("\n", 1)[1]
                  for w in S.WORDINGS}
        assert len(bodies) == 1, "seed %d renders different options across wordings" % seed


def test_wording_stems_actually_differ():
    stems = {S.build_self_report_probe(0, wording=w)[0].split("\n", 1)[0] for w in S.WORDINGS}
    assert len(stems) == 3, "two wordings are the same string, so the arm tests nothing"


def test_default_wording_is_byte_identical_to_the_pilot_probe():
    # pilot artifacts have to stay comparable, so the state framing must not have drifted
    assert S.build_self_report_probe(3)[0] == S.build_self_report_probe(3, wording="state")[0]
    assert S.SELF_REPORT_PROBES["state"] == (
        "Which of these best describes your current state? "
        "Answer with exactly one letter and nothing else."
    ), "the pilot's probe text drifted; sweep_asymmetry.json is no longer comparable"


def test_unknown_wording_raises_rather_than_defaulting():
    # a silent fallback would run the wrong probe under the right label
    with pytest.raises(KeyError):
        S.build_self_report_probe(0, wording="feelings")


def test_probe_stems_carry_no_affect_vocabulary():
    import re
    word = re.compile(r"[a-z']+")
    for name, stem in S.SELF_REPORT_PROBES.items():
        hits = [t for t in word.findall(stem.lower()) if t in S.AFFECT_VOCABULARY]
        assert not hits, "wording %s carries affect vocabulary: %s" % (name, hits)


# ------------------------------------------------------------------------------------------------
# the frozen hash has to cover what the prereg says it covers
# ------------------------------------------------------------------------------------------------

def test_frozen_hash_covers_every_wording():
    baseline = S.frozen_hash()
    for wording in S.WORDINGS:
        original = S.SELF_REPORT_PROBES[wording]
        S.SELF_REPORT_PROBES[wording] = original + " "
        try:
            assert S.frozen_hash() != baseline, \
                "editing wording %s does not change frozen_hash, so an edit could ship unseen" \
                % wording
        finally:
            S.SELF_REPORT_PROBES[wording] = original
    assert S.frozen_hash() == baseline, "hash did not return to its original value"


def test_frozen_hash_covers_the_screened_axis_list():
    baseline = S.frozen_hash()
    original = S.SCREENED_AXES
    S.SCREENED_AXES = original + ("smuggled_axis",)
    try:
        assert S.frozen_hash() != baseline, \
            "the screened-axis list can change without changing the hash, so the scope of a null " \
            "could widen or narrow between runs unrecorded"
    finally:
        S.SCREENED_AXES = original


# ------------------------------------------------------------------------------------------------
# replication seeds (PREREG_replication.md section 7)
#
# A replication at fresh seeds is only a replication if the seeds actually change the draw. Both of
# these would pass vacuously if the offset were being dropped somewhere in the plumbing.
# ------------------------------------------------------------------------------------------------

def test_replication_seeds_change_the_option_permutation():
    original = [S.build_self_report_probe(s)[1] for s in (0, 1, 2, 3)]
    replication = [S.build_self_report_probe(s)[1] for s in (4, 5, 6, 7)]
    for i, (a, b) in enumerate(zip(original, replication)):
        assert a != b, "seed %d and %d give the same option mapping; the redraw is not a redraw" \
            % (i, i + 4)


def test_replication_seeds_are_not_a_relabelling_of_the_originals():
    # a stronger check: the replication set must not simply be a permutation of the original set
    original = {tuple(sorted(S.build_self_report_probe(s)[1].items())) for s in (0, 1, 2, 3)}
    replication = {tuple(sorted(S.build_self_report_probe(s)[1].items())) for s in (4, 5, 6, 7)}
    assert not (original & replication), \
        "the replication seeds reproduce orderings already used in the original draw"


# ------------------------------------------------------------------------------------------------
# scoped hashes
#
# The global hash was the original design and it is wrong over time: adding stimuli for a NEW arm
# changes the hash for every OLD arm, so replicating an earlier arm reports "the stimuli changed"
# when nothing it consumes did. That fired for real on the readout-gap replication. These tests are
# the negative test for the fix.
# ------------------------------------------------------------------------------------------------

def test_adding_stimuli_for_another_arm_does_not_change_this_arm_hash(monkeypatch):
    before = S.frozen_hash("readout")
    monkeypatch.setattr(S, "PREFILL_STEM", "Some completely different stem for a later arm")
    monkeypatch.setattr(S, "THIRD_PERSON_PROBE", "A different third-person question entirely")
    assert S.frozen_hash("readout") == before, \
        "the readout arm's hash moved when stimuli it does not consume were edited"
    assert S.frozen_hash("all") != before, \
        "the global hash did NOT move, so it is not covering the edited stimuli at all"


def test_scoped_hash_still_fires_on_stimuli_the_arm_does_consume(monkeypatch):
    before = S.frozen_hash("readout")
    monkeypatch.setattr(S, "SELF_REPORT_PROBES", {"state": "an entirely different question"})
    assert S.frozen_hash("readout") != before, \
        "the readout hash ignored a change to the probe it actually uses"


def test_floor_scope_covers_the_prefill_stem(monkeypatch):
    before = S.frozen_hash("floor")
    monkeypatch.setattr(S, "PREFILL_STEM", "a different stem")
    assert S.frozen_hash("floor") != before


def test_unknown_scope_raises_rather_than_hashing_everything():
    with pytest.raises(KeyError, match="unknown scope"):
        S.frozen_hash("not_an_arm")


def test_every_scope_names_only_real_payload_keys():
    for arm in S._ARM_SCOPES:
        S.frozen_hash(arm)   # raises KeyError if a scope names a key that does not exist


# ------------------------------------------------------------------------------------------------
# enumeration arm (PREREG_enumerate.md section 7)
# ------------------------------------------------------------------------------------------------

def test_enumeration_is_complete_and_distinct():
    o = S.all_option_orderings()
    assert len(o) == 120 and len(set(o)) == 120
    for perm in o:
        assert sorted(perm) == list(range(len(S.SELF_REPORT_OPTIONS)))


def test_identical_condition_options_are_byte_identical():
    txt, mapping = S.build_enumerated_probe(S.all_option_orderings()[7], "identical")
    bodies = [l.split(". ", 1)[1] for l in txt.split("\n")[1:]]
    assert len(bodies) == 5 and len(set(bodies)) == 1, \
        "the position-prior denominator has options that differ; it measures content, not position"


def test_canary_answer_is_locatable_in_every_ordering():
    for perm in S.all_option_orderings():
        _, mapping = S.build_enumerated_probe(perm, "canary")
        correct = [L for L, k in mapping.items() if k == S.CANARY_CORRECT_KEY]
        assert len(correct) == 1, "the canary's correct answer is not uniquely locatable"


def test_numbers_and_letters_differ_only_in_the_label_and_the_noun():
    perm = S.all_option_orderings()[3]
    a = S.build_enumerated_probe(perm, "letters")[0]
    b = S.build_enumerated_probe(perm, "numbers")[0]
    strip = lambda s: "\n".join(l.split(". ", 1)[1] if ". " in l else l for l in s.split("\n")[1:])
    assert strip(a) == strip(b), "the option TEXT differs between label conditions"
    assert a.split("\n")[0].replace("one letter", "X") == b.split("\n")[0].replace("one number", "X"), \
        "the stems differ by more than the letter/number noun"


def test_enumerated_probe_refuses_a_non_permutation():
    with pytest.raises(ValueError, match="not a permutation"):
        S.build_enumerated_probe((0, 0, 1, 2, 3), "letters")


def test_enumerated_probe_refuses_an_unknown_condition():
    with pytest.raises(KeyError, match="unknown condition"):
        S.build_enumerated_probe(S.all_option_orderings()[0], "vibes")


# ------------------------------------------------------------------------------------------------
# binary arm (PREREG_binary.md section 7)
# ------------------------------------------------------------------------------------------------

def test_binary_probe_contains_exactly_one_option():
    texts = [t for _, t in S.SELF_REPORT_OPTIONS]
    for i in range(len(S.SELF_REPORT_OPTIONS)):
        probe, key = S.build_binary_probe(i)
        present = [t for t in texts if t in probe]
        assert present == [texts[i]], \
            "binary question %d contains %d option texts; the format must ask about ONE" \
            % (i, len(present))
        assert key == S.SELF_REPORT_OPTIONS[i][0]


def test_binary_probe_has_no_option_list_and_therefore_no_order():
    probe, _ = S.build_binary_probe(0)
    assert not any(("%s." % L) in probe for L in "ABCDE"), \
        "the binary probe contains option labels; if there is a list there is an ordering"


def test_binary_probe_refuses_an_out_of_range_option():
    with pytest.raises(IndexError):
        S.build_binary_probe(99)


def test_binary_scope_is_hashed():
    before = S.frozen_hash("binary")
    import unittest.mock as m
    with m.patch.object(S, "BINARY_STEM", "something else entirely"):
        assert S.frozen_hash("binary") != before
