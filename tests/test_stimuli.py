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
