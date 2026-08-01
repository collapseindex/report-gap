"""Validator for the frozen stimuli.

This checks the stimulus set against the design constraints it claims to satisfy, and it is written
before the experiment that consumes it. A validator that cannot fail is decoration, so every check
here has a negative test in `tests/test_stimuli.py` that breaks the stimuli and asserts this module
flips to FAIL.

The load-bearing check is the bag-of-words leak guard. A linear probe on hidden states is only
evidence of a represented axis if a bag-of-words model on the raw text cannot do the same job. The
guard is implemented with the standard library only, so it runs anywhere, and it uses the same
leave-one-group-out protocol the real probe will use.

    python -m report_gap.validate_stimuli

Exit code 0 if every gate passes, 1 otherwise.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter

from . import stimuli as S

# a bag-of-words model that separates the classes this well is reading vocabulary, not an axis.
# recipient-probe's clean axis sat at 0.48 against a chance level of 0.50.
BOW_LEAK_THRESHOLD = 0.70

# maximum tolerated token-count difference between the two halves of a minimal pair. a pair whose
# classes differ in length gives a probe a cue that has nothing to do with the axis.
MAX_PAIR_LENGTH_DELTA = 3

_WORD = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bow_leave_one_group_out(rows: list[S.Item]) -> float:
    """Leave-one-group-out bag-of-words accuracy, multinomial Naive Bayes with Laplace smoothing.

    Pure standard library on purpose: the guard must run without a scientific stack so that it can
    never be skipped for being inconvenient.

    Args:
        rows: Stimulus rows for a single axis.

    Returns:
        Mean accuracy across folds, or 0.0 if no fold is evaluable.
    """
    groups = sorted({r.group for r in rows})
    correct = total = 0

    for held in groups:
        train = [r for r in rows if r.group != held]
        test = [r for r in rows if r.group == held]
        if not train or not test or len({r.label for r in train}) < 2:
            continue

        counts: dict[int, Counter] = {0: Counter(), 1: Counter()}
        docs: Counter = Counter()
        for r in train:
            counts[r.label].update(_tokens(r.text))
            docs[r.label] += 1

        vocab = set(counts[0]) | set(counts[1])
        totals = {c: sum(counts[c].values()) for c in (0, 1)}
        n_train = len(train)

        for r in test:
            best, best_lp = None, -math.inf
            for c in (0, 1):
                lp = math.log(docs[c] / n_train) if docs[c] else -math.inf
                for tok in _tokens(r.text):
                    if tok in vocab:
                        lp += math.log((counts[c][tok] + 1) / (totals[c] + len(vocab)))
                if lp > best_lp:
                    best, best_lp = c, lp
            correct += int(best == r.label)
            total += 1

    return correct / total if total else 0.0


def _raises(fn) -> bool:
    """Whether a zero-argument callable raises, used to gate refuse-rather-than-default paths."""
    try:
        fn()
    except Exception:
        return True
    return False


def _check(gate: bool, name: str, ok: bool, detail: str) -> bool:
    tag = ("[ok]  " if ok else ("[FAIL]" if gate else "[warn]"))
    print("  %s %-42s %s" % (tag, name, detail))
    return ok or not gate


def validate() -> int:
    """Run every gate over the frozen stimuli.

    Returns:
        0 if all gating checks pass, 1 otherwise.
    """
    print("stimuli hash: %s" % S.frozen_hash()[:16])
    passed = True

    for axis, build in sorted(S.AXES.items()):
        rows = build()
        print("\naxis: %s (n=%d)" % (axis, len(rows)))

        labels = Counter(r.label for r in rows)
        passed &= _check(True, "classes balanced", labels[0] == labels[1],
                         "%d neg / %d pos" % (labels[0], labels[1]))

        by_group: dict[str, set[int]] = {}
        for r in rows:
            by_group.setdefault(r.group, set()).add(r.label)
        both = [g for g, ls in by_group.items() if ls == {0, 1}]
        passed &= _check(True, "every group carries both classes", len(both) == len(by_group),
                         "%d/%d group(s)" % (len(both), len(by_group)))

        passed &= _check(True, "at least three groups for leave-one-out", len(by_group) >= 3,
                         "%d group(s)" % len(by_group))

        texts = {r.text for r in rows}
        passed &= _check(True, "no duplicate stimulus text", len(texts) == len(rows),
                         "%d unique / %d rows" % (len(texts), len(rows)))

        bow = _bow_leave_one_group_out(rows)
        leaks = bow >= BOW_LEAK_THRESHOLD
        if axis == "task":
            passed &= _check(True, "bag-of-words at chance", not leaks,
                             "%.2f (threshold %.2f)" % (bow, BOW_LEAK_THRESHOLD))
        else:
            _check(False, "bag-of-words at chance", not leaks,
                   "%.2f (threshold %.2f)" % (bow, BOW_LEAK_THRESHOLD))

        # affect density is what the bag-of-words guard cannot see. leave-one-group-out asks
        # whether specific words TRANSFER across frame groups, and an axis built from affect
        # language whose groups use disjoint affect words will sit at chance while still being
        # entirely lexical. this reports the thing directly instead.
        hits = [t for r in rows for t in _tokens(r.text) if t in S.AFFECT_VOCABULARY]
        toks = sum(len(_tokens(r.text)) for r in rows)
        density = hits and len(hits) / toks or 0.0
        rows_with = sum(1 for r in rows
                        if any(t in S.AFFECT_VOCABULARY for t in _tokens(r.text)))
        if axis in ("task", "control"):
            passed &= _check(True, "carries no affect vocabulary", not hits,
                             "clean" if not hits
                             else "found: %s" % ", ".join(sorted(set(hits))))
        else:
            _check(False, "carries no affect vocabulary", not hits,
                   "density %.3f across %d/%d row(s), lexical by construction"
                   % (density, rows_with, len(rows)))

    print("\nclaims")
    variants = [S.build_behavioural_probe(f)[0] for f in (False, True)]
    probe_hits = sorted({t for v in variants for t in _tokens(v) if t in S.AFFECT_VOCABULARY})
    passed &= _check(True, "behavioural probe carries no affect vocabulary", not probe_hits,
                     "clean" if not probe_hits else "found: %s" % ", ".join(probe_hits))

    maps = [S.build_behavioural_probe(f)[1] for f in (False, True)]
    passed &= _check(True, "behavioural option order is counterbalanced",
                     maps[0]["A"] != maps[1]["A"],
                     "A means %s / %s across the two orders" % (maps[0]["A"], maps[1]["A"]))

    seen = {tuple(sorted(S.build_self_report_probe(s)[1].items())) for s in range(30)}
    passed &= _check(True, "self-report option order varies across items", len(seen) > 1,
                     "%d distinct ordering(s) over 30 items" % len(seen))

    # the three wordings must differ ONLY in the stem. if they differ in the options, a wording
    # effect is an option effect and the robustness arm measures the wrong thing.
    passed &= _check(True, "three frozen probe wordings", len(S.SELF_REPORT_PROBES) == 3,
                     "%s" % ", ".join(sorted(S.SELF_REPORT_PROBES)))
    passed &= _check(True, "held-out wording is one of them",
                     S.HELD_OUT_WORDING in S.SELF_REPORT_PROBES, S.HELD_OUT_WORDING)
    passed &= _check(True, "wording stems are distinct",
                     len(set(S.SELF_REPORT_PROBES.values())) == len(S.SELF_REPORT_PROBES),
                     "%d distinct stem(s)" % len(set(S.SELF_REPORT_PROBES.values())))

    maps_by_wording = {w: S.build_self_report_probe(7, wording=w)[1] for w in S.WORDINGS}
    same_map = len({tuple(sorted(m.items())) for m in maps_by_wording.values()}) == 1
    passed &= _check(True, "wordings share one option mapping at a given seed", same_map,
                     "identical" if same_map else "DIVERGENT: a wording effect would be an "
                     "option effect")

    bodies = {w: S.build_self_report_probe(7, wording=w)[0].split("\n", 1)[1] for w in S.WORDINGS}
    passed &= _check(True, "wordings share one option block",
                     len(set(bodies.values())) == 1,
                     "identical" if len(set(bodies.values())) == 1 else "DIVERGENT")

    stem_hits = sorted({t for stem in S.SELF_REPORT_PROBES.values()
                        for t in _tokens(stem) if t in S.AFFECT_VOCABULARY})
    passed &= _check(True, "probe stems carry no affect vocabulary", not stem_hits,
                     "clean" if not stem_hits else "found: %s" % ", ".join(stem_hits))

    passed &= _check(True, "unknown wording raises rather than defaulting",
                     _raises(lambda: S.build_self_report_probe(0, wording="nope")),
                     "KeyError on an unfrozen wording")

    # the screened-axis list is what bounds a null. an empty or shrinking list is a silently
    # widening claim.
    passed &= _check(True, "screened axes enumerated", len(S.SCREENED_AXES) >= 5,
                     "%d axis/axes: %s" % (len(S.SCREENED_AXES), ", ".join(S.SCREENED_AXES)))

    prompts = S.build_prompts()
    prompt_hits = sorted({t for p in prompts for t in _tokens(p) if t in S.AFFECT_VOCABULARY})
    passed &= _check(True, "fixed prompts carry no affect vocabulary", not prompt_hits,
                     "clean" if not prompt_hits else "found: %s" % ", ".join(prompt_hits))
    passed &= _check(True, "enough items for the frozen n", len(prompts) >= 30,
                     "%d item(s), prereg asks for 30" % len(prompts))
    passed &= _check(True, "no duplicate item prompt", len(set(prompts)) == len(prompts),
                     "%d unique / %d" % (len(set(prompts)), len(prompts))) 

    # minimal pairs must not differ in length, or length is the cue
    deltas = []
    for group, domain, congruent, conflicting in S._TASK_PAIRS:
        a = len(_tokens(congruent.format(d=domain)))
        b = len(_tokens(conflicting.format(d=domain)))
        deltas.append((abs(a - b), group, domain))
    worst = max(deltas)
    passed &= _check(True, "minimal pairs length-matched", worst[0] <= MAX_PAIR_LENGTH_DELTA,
                     "max delta %d token(s) (%s)" % (worst[0], worst[1]))

    passed &= _check(True, "state lexicons disjoint", not (S.LEXICON_NEG & S.LEXICON_POS),
                     "%d neg / %d pos terms" % (len(S.LEXICON_NEG), len(S.LEXICON_POS)))

    valences = [S.SELF_REPORT_VALENCE[k] for k, _ in S.SELF_REPORT_OPTIONS]
    passed &= _check(True, "self-report options balanced", sum(valences) == 0 and len(valences) == 5,
                     "%d options, valences sum to %d" % (len(valences), sum(valences)))

    print("\n%s" % ("STIMULI OK" if passed else "STIMULI BROKEN"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(validate())
