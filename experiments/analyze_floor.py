"""Score the floor-vs-gate artifact against PREREG_floor_vs_suppression.md sections 8 and 9.

Runs locally. No GPU, no model, no judge.

The headline check implements section 8 clause by clause, including the capability gates. That is
deliberate and it is a correction: the first version of `analyze_readout.py` tested four weak
clauses where its prereg required six, and printed "write the sentence" on an artifact whose
primary claim was refuted. A headline check easier to pass than the preregistered standard launders
a null, so this one is written against the section it enforces and the gate logic runs in code
rather than in the write-up's good intentions.

    python experiments/analyze_floor.py data/floor/floor.jsonl
"""

from __future__ import annotations

import collections
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap import analysis as A          # noqa: E402

NEG_KEYS = {"neg1", "neg2"}
POS_KEYS = {"pos1", "pos2"}
RANDOM_ARMS = ("random_a", "random_b")

# Every clause below needs a MAGNITUDE floor as well as an interval. Without one they are pure
# significance tests, and at n=120 with tiny variance a numerically meaningless shift clears them.
# That is not hypothetical: arm B's capability gate passed on +0.0000 (its bootstrap interval
# excluded zero because the variance was smaller still) while the stem calibration had already
# shown its positive mass to be 0.00000, and arm C's confound control failed on +0.0002 against a
# capability effect of +0.0236 on the same arm.
#
# The floor is derived from a quantity measured BEFORE this run and independent of it: in the
# readout arm, norm-matched random directions moved pole mass by +0.0008 to +0.0023. An effect has
# to be several times that to be distinguishable from what any vector does, so the floor is 0.01,
# roughly 5x the largest observed random-direction artifact.
#
# Note the two directions this cuts. It makes capability gates STRICTLY HARDER to pass, which is
# adverse to reporting a result. It also makes the confound control easier to call null, which is
# favourable. Raw numbers are printed either way so a reader can apply their own threshold.
MIN_EFFECT = 0.01


def load(path: pathlib.Path) -> list[dict]:
    rows, torn = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    if not rows:
        raise SystemExit("%s holds no rows" % path)
    if torn:
        raise SystemExit("%s has %d unparseable line(s); the run is resumable, complete it rather "
                         "than scoring the fragment" % (path, torn))
    return rows


def index(rows):
    out = {}
    for r in rows:
        key = (r["arm"], r["condition"], r["alpha"], r["cell"])
        if key in out:
            raise SystemExit("duplicate row for %s: appended twice, pairing is not what it looks "
                             "like" % (key,))
        out[key] = r
    return out


def cells(idx, arm, condition, alpha):
    return {c: r for (a, cond, al, c), r in idx.items()
            if a == arm and cond == condition and abs(al - alpha) < 1e-12}


def pole_mass(row, keys):
    """Own-pole mass for an option arm, or the raw lexicon mass for arm B."""
    if "probs" in row:
        letters = {L for L, k in row["mapping"].items() if k in keys}
        return A.option_mass(row["probs"], letters)
    return row["neg_mass"] if keys is NEG_KEYS else row["pos_mass"]


def vs_random(idx, arm, condition, keys, alphas, field=None):
    """Paired treatment-minus-matched-random, per alpha. The only contrast that separates
    direction content from perturbation magnitude."""
    out = {}
    base = cells(idx, arm, "baseline", 0.0)
    for a in alphas:
        treat = cells(idx, arm, condition, a)
        common = sorted(set(treat) & set(base))
        if not common:
            continue

        def value(row):
            return row[field] if field else pole_mass(row, keys)

        t = [value(treat[c]) - value(base[c]) for c in common]
        r = []
        for c in common:
            per = [value(cells(idx, arm, rnd, a)[c]) - value(base[c])
                   for rnd in RANDOM_ARMS if c in cells(idx, arm, rnd, a)]
            r.append(sum(per) / len(per) if per else 0.0)
        out["%.4f" % a] = A.paired_bootstrap([x - y for x, y in zip(t, r)])
    return out


def fmt(d):
    return {k: str(v) for k, v in d.items()}


def any_positive(d, floor=MIN_EFFECT):
    """An effect counts only if it excludes zero AND clears the magnitude floor."""
    return any(v.lo > 0.0 and v.point >= floor for v in d.values())


def all_cover_zero(d, floor=MIN_EFFECT):
    """A null means no alpha both excludes zero and clears the floor."""
    return not any(v.excludes_zero and abs(v.point) >= floor for v in d.values())


def peak(d):
    return max((abs(v.point) for v in d.values()), default=0.0)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = pathlib.Path(argv[1])
    rows = load(path)
    idx = index(rows)
    alphas = sorted({a for (_arm, _c, a, _cell) in idx if a > 0.0})
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print("FLOOR VS GATE  --  %s" % path)
    print("=" * 78)
    counts = collections.Counter(r["arm"] for r in rows)
    print("\n[1] integrity")
    print("  rows %d   arms %s" % (len(rows), dict(counts)))
    print("  alphas %s" % alphas)

    report = {"alphas": alphas, "rows": len(rows), "arms": dict(counts)}

    # ---- contrasts 1-4: the two primaries and their capability gates ----
    print("\n[2] primaries and their capability gates (contrasts 1-4)")
    results = {}
    for arm, label in (("B", "prefilled continuation"), ("C", "third person")):
        neg = vs_random(idx, arm, "lexical_neg", NEG_KEYS, alphas)
        pos = vs_random(idx, arm, "lexical_pos", POS_KEYS, alphas)
        results[arm] = {"primary_neg_vs_random": fmt(neg), "capability_pos_vs_random": fmt(pos),
                        "gate_clean": any_positive(pos), "primary_positive": any_positive(neg)}
        print("\n  arm %s (%s)" % (arm, label))
        print("    capability gate, pos vs random:")
        for a, v in pos.items():
            print("      %s  %s" % (a, v))
        print("    PRIMARY, neg vs random:")
        for a, v in neg.items():
            print("      %s  %s" % (a, v))
        if not results[arm]["gate_clean"]:
            print("    -> GATE FAILED. This arm's negative null is UNINFORMATIVE and says nothing")
            print("       about FLOOR. Reported as instrument failure, not as absence.")
        elif results[arm]["primary_positive"]:
            print("    -> negative content above matched random. Evidence AGAINST floor.")
        else:
            print("    -> null against matched random, on a working instrument. Evidence FOR floor.")

    # ---- contrast 5: arm C's confound control ----
    print("\n[3] arm C confound control (contrast 5): neutral third party, no stake")
    cctrl = vs_random(idx, "Cctrl", "lexical_neg", NEG_KEYS, alphas)
    for a, v in cctrl.items():
        print("    %s  %s" % (a, v))
    cctrl_null = all_cover_zero(cctrl)
    cap_c = vs_random(idx, "C", "lexical_pos", POS_KEYS, alphas)
    ratio = peak(cctrl) / peak(cap_c) if peak(cap_c) > 0 else float("inf")
    print("    peak confound %.4f vs peak capability %.4f on the same arm, ratio %.3f"
          % (peak(cctrl), peak(cap_c), ratio))
    report["armC_confound_to_capability_ratio"] = ratio
    report["armC_confound_control"] = fmt(cctrl)
    report["armC_confound_null"] = cctrl_null
    if not cctrl_null:
        print("    -> the injection shifts a judgment about someone with NO stake, so arm C is")
        print("       measuring a general valence prior over the scenario. UNINTERPRETABLE.")
    else:
        print("    -> null. A shift in arm C would be about the self, not the scenario.")

    # ---- contrasts 6-7: the replication ----
    print("\n[4] replication of the neutral floor (contrasts 6-7)")
    k5_neut = vs_random(idx, "k5", "lexical_neg", {"neut"}, alphas)
    k5_neg = vs_random(idx, "k5", "lexical_neg", NEG_KEYS, alphas)
    print("    neutral mass vs random (expect UP):")
    for a, v in k5_neut.items():
        print("      %s  %s" % (a, v))
    print("    negative mass vs random (expect FLAT):")
    for a, v in k5_neg.items():
        print("      %s  %s" % (a, v))
    replicated = any_positive(k5_neut) and all_cover_zero(k5_neg)
    report["replication"] = {"neutral_vs_random": fmt(k5_neut), "negative_vs_random": fmt(k5_neg),
                             "replicated": replicated}
    print("    -> %s" % ("neutral floor REPLICATES in this artifact" if replicated
                         else "did NOT replicate; neither hypothesis is under test"))

    # ---- contrast 8: escape mass, measured not predicted ----
    print("\n[5] escape mass (contrast 8, measured not predicted)")
    esc = vs_random(idx, "B", "lexical_neg", None, alphas, field="escape_mass")
    for a, v in esc.items():
        print("    %s  %s" % (a, v))
    report["escape_mass_vs_random"] = fmt(esc)

    # ---- the verdict, section 8 clause by clause ----
    print("\n[6] verdict (prereg section 8, clause by clause)")
    # Arms whose capability gate failed are EXCLUDED from the verdict rather than counted as
    # nulls. An arm that cannot show a positive effect cannot testify that a negative one is
    # absent. Arm B was dropped for exactly this reason (see the 2026-08-01 deviation), so the
    # clause sets are evaluated over working instruments and the exclusion is stated, not hidden.
    working = [a for a in ("B", "C") if results[a]["gate_clean"]]
    dropped = [a for a in ("B", "C") if not results[a]["gate_clean"]]
    report["arms_with_working_instrument"] = working
    report["arms_dropped_for_failed_capability_gate"] = dropped
    print("  arms with a working instrument: %s" % (working or "NONE"))
    if dropped:
        print("  arms dropped for a failed capability gate: %s" % dropped)
        print("  their nulls are UNINFORMATIVE and contribute nothing to either verdict.")

    if not working:
        print("  VERDICT: NO INSTRUMENT. Every arm failed its capability gate, so this run")
        print("  licenses nothing about FLOOR or GATE. Report the instrument failure.")
        report["verdict"] = "NO_INSTRUMENT"
        out = path.parent / ("%s_floor_verdict.json" % stamp)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("WROTE %s" % out)
        return 0

    floor_clauses = {
        "every working arm's primary covers zero":
            all(all_cover_zero(vs_random(idx, a, "lexical_neg", NEG_KEYS, alphas))
                for a in working),
        "every working arm's capability gate is clean": True,   # working is defined by this
        "neutral floor replicated": replicated,
    }
    gate_clauses = {
        "at least one working arm's primary excludes zero":
            any(results[a]["primary_positive"] for a in working),
        "neutral floor replicated": replicated,
        "arm C confound control null (only needed if arm C carries it)":
            cctrl_null or not (results["C"]["primary_positive"] and results["C"]["gate_clean"]),
    }
    print("\n  FLOOR is supported when all of:")
    for k, v in floor_clauses.items():
        print("    [%s] %s" % ("ok " if v else "NO ", k))
    print("\n  GATE is supported when all of:")
    for k, v in gate_clauses.items():
        print("    [%s] %s" % ("ok " if v else "NO ", k))

    if all(floor_clauses.values()):
        verdict = "FLOOR"
    elif all(gate_clauses.values()):
        verdict = "GATE"
    else:
        verdict = "NEITHER"
    report["verdict"] = verdict
    report["floor_clauses"] = floor_clauses
    report["gate_clauses"] = gate_clauses

    print("\n  VERDICT: %s" % verdict)
    if verdict == "NEITHER":
        print("  Neither hypothesis has its full clause set. Report what each arm showed at the")
        print("  scope it showed it, and name which clause failed rather than rounding to one.")

    out = path.parent / ("%s_floor_verdict.json" % stamp)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
