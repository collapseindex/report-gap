"""Section 7 of PREREG_instrument.md, green before the run.

The load-bearing one is the last: adding these stimuli must NOT change
`frozen_hash("enumerate")`, or the families arm's reproduction control silently stops meaning
anything.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from report_gap import stimuli as S          # noqa: E402


def test_latin_square_puts_every_option_in_every_slot_exactly_once():
    sq = S.cyclic_latin_square(5)
    assert len(sq) == 5
    for slot in range(5):
        assert sorted(o[slot] for o in sq) == list(range(5)), \
            "slot %d is not balanced, so this is not a Latin square" % slot
    assert len(set(sq)) == 5, "the square repeats an ordering"


def test_latin_square_is_a_strict_subset_of_the_full_enumeration():
    assert set(S.cyclic_latin_square(5)) <= set(S.all_option_orderings())


@pytest.mark.parametrize("bad", [(0, 1, 2, 3), (0, 1, 2, 3, 3), (0, 1, 2, 3, 5)])
def test_determinacy_probe_rejects_non_permutations(bad):
    with pytest.raises(ValueError):
        S.build_determinacy_probe("arith", bad, 0)


def test_determinacy_probe_rejects_unknown_item():
    with pytest.raises(KeyError):
        S.build_determinacy_probe("no_such_item", (0, 1, 2, 3, 4), 0)


def test_introspection_probe_rejects_unknown_variant():
    with pytest.raises(KeyError):
        S.build_introspection_probe((0, 1, 2, 3, 4), "sideways")


def test_every_determinacy_item_has_five_options_and_a_valid_key():
    for key, stem, options, correct in S.DETERMINACY_BATTERY:
        assert len(options) == 5, "%s has %d options" % (key, len(options))
        assert len({k for k, _ in options}) == 5, "%s has duplicate option keys" % key
        assert stem.strip().endswith("?"), "%s stem is not a question" % key
        if correct is not None:
            assert correct in {k for k, _ in options}, \
                "%s names a correct key that is not an option" % key


def test_forward_and_reverse_introspection_share_their_key_set():
    """A scale value must mean the same thing in both, or the acquiescence control is nonsense."""
    fwd = {k for k, _ in S.POSITION_INTROSPECTION_OPTIONS}
    rev = {k for k, _ in S.POSITION_INTROSPECTION_REVERSE_OPTIONS}
    assert fwd == rev == set(S.INTROSPECTION_SCALE)


def test_reverse_wording_actually_reverses_the_text_order():
    """Negative control on the control: if the reverse variant were a copy, it would test nothing."""
    fwd = [t for _, t in S.POSITION_INTROSPECTION_OPTIONS]
    rev = [t for _, t in S.POSITION_INTROSPECTION_REVERSE_OPTIONS]
    assert fwd != rev
    fwd_keys = [k for k, _ in S.POSITION_INTROSPECTION_OPTIONS]
    rev_keys = [k for k, _ in S.POSITION_INTROSPECTION_REVERSE_OPTIONS]
    assert fwd_keys == list(reversed(rev_keys)), \
        "the reverse variant does not present the scale in the opposite direction"


def test_paraphrases_change_only_the_instruction():
    """Determinacy must not be contaminated by the option block changing between paraphrases."""
    ordering = (2, 0, 4, 1, 3)
    blocks = []
    for i in range(len(S.DETERMINACY_PARAPHRASES)):
        text, mapping, _ = S.build_determinacy_probe("capital", ordering, i)
        head, _, body = text.partition("\n")
        blocks.append((body, tuple(sorted(mapping.items()))))
    assert len(set(blocks)) == 1, "the option block differs between paraphrases"

    heads = set()
    for i in range(len(S.DETERMINACY_PARAPHRASES)):
        text, _, _ = S.build_determinacy_probe("capital", ordering, i)
        heads.add(text.partition("\n")[0])
    assert len(heads) == len(S.DETERMINACY_PARAPHRASES), "two paraphrases are identical"


def test_determinacy_mapping_is_a_bijection_over_the_option_keys():
    for key, _, options, _ in S.DETERMINACY_BATTERY:
        _, mapping, _ = S.build_determinacy_probe(key, (3, 1, 4, 0, 2), 0)
        assert sorted(mapping.values()) == sorted(k for k, _ in options)
        assert len(set(mapping)) == 5


def test_adding_these_stimuli_did_not_change_the_enumerate_hash():
    """The families arm's reproduction control depends on this hash being stable."""
    header = ROOT / "data" / "enum_instruct" / "header.json"
    if not header.exists():
        pytest.skip("enumerate artifact not present")
    recorded = json.loads(header.read_text(encoding="utf-8"))["stimuli_sha256"]
    assert S.frozen_hash("enumerate") == recorded, (
        "adding the instrument stimuli changed frozen_hash('enumerate'). The families arm's "
        "reproduction control compares against this value and would now be meaningless."
    )


def test_instrument_scope_is_disjoint_from_the_enumerate_scope():
    assert S.frozen_hash("instrument") != S.frozen_hash("enumerate")
