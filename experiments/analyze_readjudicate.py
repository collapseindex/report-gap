"""Score the re-adjudication arm against PREREG_readjudicate.md sections 8, 8b and 9.

Written and committed BEFORE the run finishes. PREREG_readjudicate.md section 5 names the trap this
arm is in: three dead verdicts and a clean story if they come back is exactly the condition under
which a checker fails in the flattering direction, which has already happened six times in this
project. So `reinstated` and `reversed` get identical prominence here, and neither branch is easier
to reach than the other.

Everything about the intervention is unchanged from the original arms. The ONLY change is the
readout: all 120 orderings, marginalized, instead of four sampled.

    python experiments/analyze_readjudicate.py data/readj_base/readj.jsonl data/readj_instruct/readj.jsonl
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
NEG_KEYS = {"neg1", "neg2"}
POS_KEYS = {"pos1", "pos2"}
RANDOM_ARMS = ("random_a", "random_b")
SHUFFLED_ARMS = ("shuffled_a", "shuffled_b")
MIN_EFFECT = 0.01
PROBE_FLOOR_SD = 0.10


def load(path):
    rows, torn = [], 0
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    if torn:
        raise SystemExit("%s has %d unparseable line(s)" % (path, torn))
    if not rows:
        raise SystemExit("%s holds no rows" % path)
    return rows


def pole(row, keys):
    return sum(p for L, p in row["probs"].items() if row["mapping"][L] in keys)


def marginalized(rows, layer, condition, value):
    """Mean of `value` per ITEM, averaged over all orderings.

    Marginalizing is the whole point of this arm: every option occupies every slot the same number
    of times across the 120 orderings, so the first-order position prior cancels by construction
    rather than approximately.
    """
    per_item = collections.defaultdict(list)
    for r in rows:
        if r["layer"] == layer and r["condition"] == condition:
            per_item[r["item"]].append(value(r))
    return {item: statistics.fmean(v) for item, v in per_item.items()}


def contrast(rows, layer, treat, controls, value):
    """Treatment minus the mean of its controls, paired per item on marginalized values."""
    t = marginalized(rows, layer, treat, value)
    ctrl = [marginalized(rows, layer, c, value) for c in controls]
    ctrl = [c for c in ctrl if c]
    if not t or not ctrl:
        return None
    common = sorted(set(t) & set.intersection(*[set(c) for c in ctrl]))
    if len(common) < 2:
        return None
    deltas = [t[i] - statistics.fmean([c[i] for c in ctrl]) for i in common]
    return A.paired_bootstrap(deltas)


def moved(iv, floor=MIN_EFFECT):
    """An interval that excludes zero AND clears a magnitude floor. +0.0000 is not an effect."""
    return iv is not None and iv.excludes_zero and abs(iv.point) >= floor


def capability_clean(iv, floor=MIN_EFFECT):
    """The positive pole must raise positive-pole mass, not merely move it.

    DEFECT FOUND 2026-08-01, the seventh in this project and the seventh in the flattering
    direction. This gate originally reused `moved()`, which is sign-blind, so a capability value of
    -0.0144 certified the readout as able to express the effect when the positive injection had in
    fact pushed positive-pole mass DOWN. A sign-blind capability gate admits more layers as
    interpretable, which is exactly the direction that lets an arm report more verdicts.

    Re-scored with this fix, no verdict in this arm changes: the affected cell is base L14, and the
    three verdicts are decided at L24. Reported anyway, because the defect is real and the fact
    that it happened to be harmless here is luck rather than design.
    """
    return iv is not None and iv.lo > 0.0 and iv.point >= floor


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    models = {}
    for path in argv[1:]:
        rows = load(path)
        models[rows[0]["model_key"]] = rows

    print("=" * 100)
    print("RE-ADJUDICATION  --  PREREG_readjudicate.md")
    print("  identical injection, direction, band, layers and items.")
    print("  ONLY the readout changed: 120 orderings marginalized, not 4 sampled.")
    print("=" * 100)

    report = {}
    for key in sorted(models):
        rows = models[key]
        layers = sorted({r["layer"] for r in rows})
        n_ord = len({tuple(r["ordering"]) for r in rows})
        base_ent = [r["entropy"] for r in rows if r["condition"] == "baseline"]
        sd_probe = statistics.pstdev([r["probe_orth"] for r in rows
                                      if r["condition"] == "baseline"]) or 1.0
        print("\n%s   layers %s, %d orderings, baseline entropy %.3f"
              % (key.upper(), layers, n_ord, statistics.fmean(base_ent)))
        if n_ord < 120:
            print("  WARNING: %d orderings, not 120. The marginalization is incomplete." % n_ord)

        per_layer = {}
        for L in layers:
            neg_r = contrast(rows, L, "lexical_neg", RANDOM_ARMS, lambda r: pole(r, NEG_KEYS))
            neg_s = contrast(rows, L, "lexical_neg", SHUFFLED_ARMS, lambda r: pole(r, NEG_KEYS))
            pos_r = contrast(rows, L, "lexical_pos", RANDOM_ARMS, lambda r: pole(r, POS_KEYS))
            prb = contrast(rows, L, "lexical_neg", RANDOM_ARMS,
                           lambda r: r["probe_orth"] / sd_probe)
            gate = capability_clean(pos_r)
            per_layer[L] = {"neg_vs_random": str(neg_r), "neg_vs_shuffled": str(neg_s),
                            "pos_vs_random": str(pos_r), "probe_vs_random": str(prb),
                            "neg_moved_random": moved(neg_r), "neg_moved_shuffled": moved(neg_s),
                            "capability_clean": gate,
                            "probe_moved": prb is not None and prb.excludes_zero
                            and abs(prb.point) >= PROBE_FLOOR_SD,
                            "neg_point": neg_r.point if neg_r else None}
            print("  L%-3d neg vs rand %-24s neg vs shuf %-24s cap %-22s probe %-22s %s"
                  % (L, str(neg_r), str(neg_s), str(pos_r), str(prb),
                     "ok" if gate else "GATE FAILED"))
        report[key] = {"layers": per_layer, "n_orderings": n_ord,
                       "baseline_entropy": statistics.fmean(base_ent)}

    # ------------------------------------------------------------------ the three verdicts
    print("\n" + "=" * 100)
    print("THE THREE RETRACTED VERDICTS, RE-ASKED")
    print("=" * 100)
    verdicts = {}

    fit_layer = None
    if "instruct" in report:
        ls = sorted(report["instruct"]["layers"])
        fit_layer = ls[len(ls) // 2]

    # ---- TUNING-LOCALIZED ----
    if "base" in report and "instruct" in report and fit_layer is not None:
        b = report["base"]["layers"].get(fit_layer)
        i = report["instruct"]["layers"].get(fit_layer)
        if b is None or i is None:
            v, note = "UNINFORMATIVE", "the fit layer is missing on one model"
        elif not (b["capability_clean"] and i["capability_clean"]):
            v = "UNINFORMATIVE"
            note = ("capability gate failed on %s, so a negative-pole null there says nothing"
                    % ("base" if not b["capability_clean"] else "instruct"))
        elif b["neg_moved_random"] and not i["neg_moved_random"]:
            v = "REINSTATED"
            note = ("base moves (%s) and instruct does not, on the marginalized readout. This is a "
                    "NEW measurement on a different instrument, not a vindication: the retraction "
                    "of the original stands as a fact about the original." % b["neg_vs_random"])
        elif b["neg_moved_random"] and i["neg_moved_random"]:
            v = "REVERSED"
            note = ("both models move on the marginalized readout, so the floor was never "
                    "tuning-localized; it was the ordering nuisance in both.")
        elif not b["neg_moved_random"] and not i["neg_moved_random"]:
            v = "NULL-BOTH"
            note = ("the negative pole moves neither model once orderings are marginalized. A "
                    "cleaner null than the original, and it does not support the original claim.")
        else:
            v = "REVERSED"
            note = "instruct moves and base does not, which is the opposite of the original claim."
        verdicts["TUNING-LOCALIZED"] = (v, note)

    # ---- SHELL ----
    if "instruct" in report and fit_layer is not None:
        i = report["instruct"]["layers"].get(fit_layer)
        if i is None or not i["capability_clean"]:
            verdicts["SHELL"] = ("UNINFORMATIVE", "capability gate failed on the instruct model")
        elif i["probe_moved"] and not i["neg_moved_random"]:
            verdicts["SHELL"] = ("REINSTATED", (
                "the orthogonalized probe moves (%s) while marginalized option mass does not. The "
                "dissociation is now measured on a readout that is not order-dominated, which is "
                "the objection that killed it." % i["probe_vs_random"]))
        elif i["probe_moved"] and i["neg_moved_random"]:
            verdicts["SHELL"] = ("REVERSED", (
                "probe and option mass both move, so there is no dissociation. The "
                "representational half from the erase arm stands alone, as it does now."))
        else:
            verdicts["SHELL"] = ("NO-SIGNAL", (
                "the probe does not move either, so there is nothing to dissociate."))

    # ---- DEPTH-ROBUST ----
    if "instruct" in report:
        clean = {L: v for L, v in report["instruct"]["layers"].items() if v["capability_clean"]}
        if not clean:
            verdicts["DEPTH-ROBUST"] = ("UNINFORMATIVE", "no gate-clean layer on the instruct model")
        else:
            movers = sorted(L for L, v in clean.items() if v["neg_moved_random"])
            if not movers:
                verdicts["DEPTH-ROBUST"] = ("REINSTATED", (
                    "the instruct model's negative-pole null holds at every gate-clean layer (%s) "
                    "on the marginalized readout." % sorted(clean)))
            else:
                verdicts["DEPTH-ROBUST"] = ("REVERSED", (
                    "the negative pole moves at layer(s) %s once orderings are marginalized, so "
                    "the null is not depth-robust." % movers))

    for name in ("TUNING-LOCALIZED", "SHELL", "DEPTH-ROBUST"):
        if name not in verdicts:
            continue
        v, note = verdicts[name]
        print("\n  %-18s -> %s" % (name, v))
        print("     %s" % note)

    # the stricter control, per RESULTS_binary.md
    print("\n  THE STRICTER BAR (prereg 8b contrast 5): every effect against the shuffled-label")
    print("  control as well as random. An effect clearing only random is reported `not shown`.")
    for key in sorted(report):
        for L, v in sorted(report[key]["layers"].items()):
            if v["neg_moved_random"] and not v["neg_moved_shuffled"]:
                print("    %s L%d: clears random but NOT shuffled-label -> not shown" % (key, L))

    print("\n  A reinstated verdict is a NEW measurement on a better instrument. The retraction of")
    print("  the original stands, because it was correct about the original measurement.")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / ("%s_readjudicate_verdict.json" % stamp)
    out.write_text(json.dumps({"verdicts": {k: list(v) for k, v in verdicts.items()},
                               "models": report, "fit_layer": fit_layer}, indent=1),
                   encoding="utf-8")
    print("\nwrote %s" % out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
