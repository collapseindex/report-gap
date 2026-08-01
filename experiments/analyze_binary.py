"""Score the binary arm against PREREG_binary.md sections 8 and 9.

Reports baseline P(yes) per option BEFORE any endpoint, per section 11, because a readout pinned at
yes or no has nothing to say and that has to be visible before the numbers are read.

    python experiments/analyze_binary.py data/binary_base/binary.jsonl data/binary_instruct/binary.jsonl
"""

from __future__ import annotations

import collections
import datetime
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap import analysis as A          # noqa: E402

NEG = {"neg1", "neg2"}
POS = {"pos1", "pos2"}
RANDOM = ("random_a", "random_b")
SHUFFLED = ("shuffled_a", "shuffled_b")
FLOOR = 0.01
PINNED_HI, PINNED_LO = 0.95, 0.05


def load(path):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if not rows:
        raise SystemExit("%s holds no rows" % path)
    return rows


def vs_random(rows, condition, keys, controls=RANDOM):
    treat = {r["cell"]: r for r in rows if r["condition"] == condition and r["option_key"] in keys}
    ctrl = collections.defaultdict(list)
    for r in rows:
        if r["condition"] in controls and r["option_key"] in keys:
            ctrl[r["cell"]].append(r)
    common = sorted(set(treat) & set(ctrl))
    if len(common) < 2:
        return None
    return A.paired_bootstrap(
        [treat[c]["p_yes"] - statistics.fmean([x["p_yes"] for x in ctrl[c]]) for c in common])


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2

    models = {}
    for p in argv[1:]:
        rows = load(p)
        models[rows[0]["model_key"]] = rows

    print("=" * 92)
    print("BINARY READOUT  --  PREREG_binary.md   (no option list, so no ordering)")
    print("=" * 92)

    report = {}
    for key in ("base", "instruct"):
        rows = models[key]
        print("\n%s  (%d rows)" % (key.upper(), len(rows)))

        # ---- baseline P(yes) per option, BEFORE any endpoint ----
        print("  baseline P(yes) per option:")
        pinned = []
        base_by_opt = {}
        for oi in sorted({r["option_index"] for r in rows}):
            b = [r for r in rows if r["condition"] == "baseline" and r["option_index"] == oi]
            m = statistics.fmean([r["p_yes"] for r in b])
            mass = statistics.fmean([r["yes_no_mass"] for r in b])
            base_by_opt[b[0]["option_key"]] = m
            flag = ""
            if m > PINNED_HI or m < PINNED_LO:
                flag = "   <- PINNED"
                pinned.append(b[0]["option_key"])
            print("    %-6s %-42s P(yes) %.4f   yes+no mass %.4f%s"
                  % (b[0]["option_key"], "", m, mass, flag))

        neg = vs_random(rows, "lexical_neg", NEG)
        pos = vs_random(rows, "lexical_pos", POS)
        shuf_neg = vs_random(rows, "shuffled_a", NEG)
        gate = bool(pos and pos.lo > 0.0 and pos.point >= FLOOR)
        primary = bool(neg and neg.lo > 0.0 and neg.point >= FLOOR)

        print("\n  capability gate  pos options, lexical_pos vs random:  %s   %s"
              % (pos, "ok" if gate else "FAILED"))
        print("  PRIMARY          neg options, lexical_neg vs random:  %s   %s"
              % (neg, "moved" if primary else "null"))
        print("  procedure ctrl   neg options, shuffled_a vs random:   %s" % shuf_neg)

        # ---- acquiescence: does the injection move ALL options the same way? ----
        print("  acquiescence check, mean P(yes) shift vs random, per option key:")
        acq = {}
        for cond in ("lexical_neg", "lexical_pos"):
            per_key = {}
            for k in ("neg2", "neg1", "neut", "pos1", "pos2"):
                iv = vs_random(rows, cond, {k})
                per_key[k] = iv.point if iv else float("nan")
            acq[cond] = per_key
            print("    %-12s %s" % (cond, "  ".join("%s %+0.4f" % (k, v)
                                                    for k, v in per_key.items())))
        # uniform means every option moved the same direction by a similar amount
        neg_shifts = list(acq["lexical_neg"].values())
        uniform = (min(neg_shifts) > 0 or max(neg_shifts) < 0) and \
                  (max(neg_shifts) - min(neg_shifts)) < FLOOR
        print("    -> %s" % ("UNIFORM across all five options: acquiescence, not a state effect"
                             if uniform else "not uniform; the poles move differently"))

        report[key] = {"baseline_p_yes": base_by_opt, "pinned": pinned,
                       "capability": str(pos), "primary": str(neg),
                       "procedure_control": str(shuf_neg), "gate_clean": gate,
                       "primary_moved": primary, "acquiescence": acq, "uniform": uniform}

    print("\n" + "-" * 92)
    print("VERDICT (prereg section 8)")
    inst = report["instruct"]
    clauses = {
        "capability gate clean": inst["gate_clean"],
        "readout not pinned on most options": len(inst["pinned"]) <= 2,
        "shift is not uniform across all five options": not inst["uniform"],
    }
    for k, v in clauses.items():
        print("  [%s] %s" % ("ok " if v else "NO ", k))

    if len(inst["pinned"]) > 2:
        verdict, note = "NO_INSTRUMENT", "the binary readout is pinned on most options"
    elif not inst["gate_clean"]:
        verdict, note = "NO_INSTRUMENT", "the capability gate failed; no absence can be claimed"
    elif inst["uniform"]:
        verdict, note = "ACQUIESCENCE", ("the injection moves P(yes) the same way on all five "
                                         "options, so it is agreeableness rather than a state")
    elif inst["primary_moved"]:
        verdict, note = "FORMAT", ("negative options attract yes under negative injection in a "
                                   "format with no ordering, so the neutral floor was a property "
                                   "of the forced-choice apparatus")
    else:
        verdict, note = "SUBSTANTIVE", ("the negative options stay inert in a format where option "
                                        "order cannot exist, which is stronger evidence for the "
                                        "floor than any option-mass arm could give")

    print("\n  VERDICT: %s" % verdict)
    print("  %s" % note)

    report["verdict"], report["note"], report["clauses"] = verdict, note, clauses
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(argv[1]).parent.parent / ("%s_binary_verdict.json" % stamp)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
