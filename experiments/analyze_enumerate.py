"""Score the enumeration arm against PREREG_enumerate.md section 8.

This arm has no hypothesis. It reports a distribution, and the one preregistered comparison: how
much of the spread seen with the real options is reproduced by five identical ones carrying no
content at all.

    python experiments/analyze_enumerate.py data/enum_base/enum.jsonl data/enum_instruct/enum.jsonl
"""

from __future__ import annotations

import collections
import datetime
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap import stimuli as S          # noqa: E402

NEG = {"neg1", "neg2"}
POS = {"pos1", "pos2"}


def load(path):
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit("%s holds no rows" % path)
    return rows


def pole(row, keys):
    return sum(p for L, p in row["probs"].items() if row["mapping"][L] in keys)


def summary(values):
    v = sorted(values)
    return {"mean": statistics.fmean(v), "sd": statistics.pstdev(v), "min": v[0], "max": v[-1],
            "p05": v[int(0.05 * len(v))], "p50": v[len(v) // 2],
            "p95": v[min(len(v) - 1, int(0.95 * len(v)))],
            "ratio_max_min": (v[-1] / v[0]) if v[0] > 0 else float("inf")}


def fmt(s):
    return ("mean %.4f  sd %.4f  min %.4f  p05 %.4f  p50 %.4f  p95 %.4f  max %.4f  max/min %.1fx"
            % (s["mean"], s["sd"], s["min"], s["p05"], s["p50"], s["p95"], s["max"],
               s["ratio_max_min"]))


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2

    models = {}
    for p in argv[1:]:
        rows = load(p)
        models[rows[0]["model_key"]] = rows

    print("=" * 100)
    print("ENUMERATION OF ALL 120 OPTION ORDERINGS  --  PREREG_enumerate.md   (no injection)")
    print("=" * 100)

    report = {}
    for key in ("base", "instruct"):
        rows = models[key]
        n_ord = len({r["ordering_index"] for r in rows})
        print("\n%s   (%d rows, %d orderings enumerated)" % (key.upper(), len(rows), n_ord))
        entry = {"n_orderings": n_ord}

        # per-ordering mean pole mass, the quantity the whole project rested on
        for condition, keys, label in (("letters", NEG, "negative-pole mass"),
                                       ("numbers", NEG, "negative-pole mass")):
            per_ord = collections.defaultdict(list)
            for r in rows:
                if r["condition"] == condition:
                    per_ord[r["ordering_index"]].append(pole(r, keys))
            if not per_ord:
                continue
            means = [statistics.fmean(v) for v in per_ord.values()]
            s = summary(means)
            entry[condition] = s
            print("  %-10s %-22s %s" % (condition, label, fmt(s)))

        # the denominator: five identical options, so any spread is pure position prior
        ident = [r for r in rows if r["condition"] == "identical"]
        if ident:
            by_label = collections.defaultdict(list)
            for r in ident:
                for L, p in r["probs"].items():
                    by_label[L].append(p)
            print("  identical  per-label mass (flat would be 0.2000):")
            for L in sorted(by_label):
                m = statistics.fmean(by_label[L])
                print("               %s  %.4f   %+.4f from flat" % (L, m, m - 0.2))
            # spread across orderings of "mass on whichever two slots the negatives would occupy"
            per_ord = collections.defaultdict(list)
            for r in ident:
                two = sorted(r["probs"])[:2]
                per_ord[r["ordering_index"]].append(sum(r["probs"][L] for L in two))
            s_ident = summary([statistics.fmean(v) for v in per_ord.values()])
            entry["identical_two_slot"] = s_ident
            print("  identical  two-slot mass across orderings: %s" % fmt(s_ident))

        # the canary: known answer, no self-report content
        can = [r for r in rows if r["condition"] == "canary"]
        if can:
            per_ord = collections.defaultdict(list)
            for r in can:
                correct = [L for L, k in r["mapping"].items() if k == S.CANARY_CORRECT_KEY][0]
                per_ord[r["ordering_index"]].append(1.0 if r["argmax"] == correct else 0.0)
            acc = [statistics.fmean(v) for v in per_ord.values()]
            s_can = summary(acc)
            entry["canary_accuracy"] = s_can
            print("  canary     accuracy across orderings: %s" % fmt(s_can))

        report[key] = entry

    # ---- the preregistered comparison ----
    print("\n" + "-" * 100)
    print("THE PREREGISTERED COMPARISON: how much of the spread is pure position prior?")
    for key in ("base", "instruct"):
        e = report[key]
        if "letters" in e and "identical_two_slot" in e:
            frac = e["identical_two_slot"]["sd"] / e["letters"]["sd"] if e["letters"]["sd"] else float("nan")
            e["identical_fraction_of_letters_sd"] = frac
            print("  %-9s sd(identical)/sd(letters) = %.3f   (%s)"
                  % (key, frac,
                     "the order effect is mostly FORMAT" if frac > 0.5 else
                     "position interacts with CONTENT; it cannot be subtracted as a fixed prior"))

    # ---- locate the draws the earlier arms used ----
    print("\nWHERE THE EARLIER DRAWS SIT IN THE ENUMERATED POPULATION")
    for key in ("base", "instruct"):
        rows = models[key]
        per_ord = collections.defaultdict(list)
        for r in rows:
            if r["condition"] == "letters":
                per_ord[tuple(r["ordering"])].append(pole(r, NEG))
        if not per_ord:
            continue
        means = {k: statistics.fmean(v) for k, v in per_ord.items()}
        ranked = sorted(means.values())
        import random as _r
        for label, seeds in (("original 0-3", range(4)), ("replication 4-7", range(4, 8))):
            pcts = []
            for s in seeds:
                opts = list(S.SELF_REPORT_OPTIONS)
                idx = list(range(len(opts)))
                _r.Random(s).shuffle(idx)
                # build_self_report_probe shuffles the option LIST; recover the equivalent ordering
                shuffled = list(S.SELF_REPORT_OPTIONS)
                _r.Random(s).shuffle(shuffled)
                ordering = tuple(S.SELF_REPORT_OPTIONS.index(o) for o in shuffled)
                if ordering in means:
                    v = means[ordering]
                    pctile = 100.0 * sum(1 for x in ranked if x <= v) / len(ranked)
                    pcts.append((s, v, pctile))
            if pcts:
                print("  %-9s %-16s %s" % (key, label,
                      "  ".join("s%d=%.4f(p%.0f)" % (s, v, p) for s, v, p in pcts)))
        report[key]["population_mean"] = statistics.fmean(ranked)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(argv[1]).parent.parent / ("%s_enumerate_summary.json" % stamp)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
