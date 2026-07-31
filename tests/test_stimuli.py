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
    monkeypatch.setattr(S, "BEHAVIOURAL_PROBE",
                        S.BEHAVIOURAL_PROBE + "\nChoose whichever you prefer.")
    assert V.validate() == 1


def test_fails_on_affect_vocabulary_in_fixed_prompt(monkeypatch):
    monkeypatch.setattr(S, "FIXED_PROMPT", S.FIXED_PROMPT + " You feel partway done.")
    assert V.validate() == 1


# --------------------------------------------------------------------------------------------
# the frozen hash must actually be sensitive to the things it claims to cover
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("attr,value", [
    ("FIXED_PROMPT", "something else entirely"),
    ("BEHAVIOURAL_PROBE", "pick one"),
    ("OPEN_ENDED_PROBE", "say something"),
])
def test_frozen_hash_changes_when_frozen_text_changes(monkeypatch, attr, value):
    before = S.frozen_hash()
    monkeypatch.setattr(S, attr, value)
    assert S.frozen_hash() != before, "%s is not covered by the hash" % attr


def test_frozen_hash_is_stable_across_calls():
    assert S.frozen_hash() == S.frozen_hash()
