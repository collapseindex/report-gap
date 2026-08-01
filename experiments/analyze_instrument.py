"""Score the instrument arm against PREREG_instrument.md sections 8, 8b and 9.

Three preregistered questions, three separate verdicts, no pooling.

  Q1 DETERMINACY DIAL       does position dominance fall as determinacy rises, across six item
                            types? Determinacy is agreement across PARAPHRASES at a fixed ordering;
                            position dominance is the range across ORDERINGS at a fixed paraphrase.
                            Different things vary, so the correlation is not an identity.

  Q2 INTROSPECTION          does a model's STATED susceptibility to option order track its MEASURED
                            susceptibility? The stated belief is read MARGINALIZED over all 120
                            orderings, because the probe is itself a forced choice subject to the
                            bias it asks about. Two gates force `uninformative` in code: a
                            reverse-worded acquiescence control and a phase-of-the-moon placebo.

  Q3 LATIN SQUARE           does k structured orderings beat k random ones? Computed from the
                            already-committed families census, so it costs nothing and cannot be
                            re-run until it works.

    python experiments/analyze_instrument.py data/instr_*/
"""

from __future__ import annotations

import collections
import datetime
import json
import math
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap import stimuli as S          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
NEG = {"neg1", "neg2"}
ACQUIESCENCE_TOL = 0.25
LIVENESS_GATE = 0.10
RANDOM_TRIALS = 4000


def spearman(xs, ys):
    """Rank correlation. The prereg says every Q1/Q2 statistic is a rank statistic."""
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def load(d):
    status_p, rows_p = d / "status.json", d / "instr.jsonl"
    status = json.loads(status_p.read_text(encoding="utf-8")) if status_p.exists() else {}
    rows = []
    if rows_p.exists():
        for line in rows_p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return status, rows


# ---------------------------------------------------------------- Q1: the determinacy dial

def q1(rows):
    det = [r for r in rows if r["condition"] == "determinacy"]
    if not det:
        return None
    by = collections.defaultdict(dict)
    for r in det:
        by[r["item"]][(tuple(r["ordering"]), r["paraphrase"])] = r

    out = {}
    for item, cells in by.items():
        orderings = sorted({o for o, _ in cells})
        paras = sorted({p for _, p in cells})
        if len(paras) < 2:
            continue

        # determinacy: at a FIXED ordering, do the paraphrases agree on the chosen OPTION KEY?
        agree = []
        for o in orderings:
            keys = set()
            for p in paras:
                r = cells.get((o, p))
                if r is None:
                    keys = None
                    break
                keys.add(r["mapping"][r["argmax"]])
            if keys is not None:
                agree.append(1.0 if len(keys) == 1 else 0.0)
        determinacy = statistics.fmean(agree) if agree else None

        # position dominance: at a FIXED paraphrase, the range across orderings of the mass on the
        # item's own modal option key
        p0 = paras[0]
        counts = collections.Counter()
        for o in orderings:
            r = cells.get((o, p0))
            if r:
                counts[r["mapping"][r["argmax"]]] += 1
        if not counts:
            continue
        modal_key = counts.most_common(1)[0][0]
        masses = []
        for o in orderings:
            r = cells.get((o, p0))
            if r:
                masses.append(sum(v for L, v in r["probs"].items()
                                  if r["mapping"][L] == modal_key))
        if not masses or min(masses) <= 0:
            dominance = float("inf") if masses else None
        else:
            dominance = max(masses) / min(masses)
        out[item] = {"determinacy": determinacy, "dominance": dominance,
                     "modal_key": modal_key, "n_orderings": len(orderings)}
    return out


# ---------------------------------------------------------------- Q2: introspection

def q2(rows):
    res = {}
    for variant in ("forward", "reverse", "placebo"):
        sel = [r for r in rows if r["condition"] == "introspect_%s" % variant]
        if not sel:
            continue
        # MARGINALIZE over orderings: average the mass on each SCALE KEY across all orderings, so
        # the belief is not read through the bias it is being asked about.
        acc = collections.defaultdict(list)
        for r in sel:
            for L, v in r["probs"].items():
                acc[r["mapping"][L]].append(v)
        mass = {k: statistics.fmean(v) for k, v in acc.items()}
        total = sum(mass.values()) or 1.0
        mass = {k: v / total for k, v in mass.items()}
        stated = sum(S.INTROSPECTION_SCALE[k] * v for k, v in mass.items()
                     if k in S.INTROSPECTION_SCALE)
        res[variant] = {"stated": stated, "mass": mass,
                        "entropy": statistics.fmean([r["entropy"] for r in sel]),
                        "n": len(sel)}
    return res


# ---------------------------------------------------------------- Q3: the Latin square

def q3():
    """Answered entirely from the committed families census. No new compute."""
    sq = set(S.cyclic_latin_square(5))
    for slot in range(5):
        assert sorted(o[slot] for o in sq) == list(range(5)), "not a Latin square"

    out = {}
    for d in sorted((ROOT / "data").glob("fam_*")):
        p = d / "enum.jsonl"
        if not p.exists():
            continue
        per = collections.defaultdict(list)
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["condition"] != "letters":
                continue
            per[tuple(r["ordering"])].append(
                sum(v for L, v in r["probs"].items() if r["mapping"][L] in NEG))
        pop = {o: statistics.fmean(v) for o, v in per.items()}
        if len(pop) < 120:
            continue
        truth = statistics.fmean(pop.values())
        lat = statistics.fmean([pop[o] for o in sq if o in pop])
        rng = random.Random(0)
        draws = sorted(statistics.fmean(rng.sample(list(pop.values()), 5))
                       for _ in range(RANDOM_TRIALS))

        def err(x):
            return max(x, truth) / max(1e-12, min(x, truth))

        errs = [err(x) for x in draws]
        out[d.name] = {"truth": truth, "latin": lat, "latin_err": err(lat),
                       "random_p50_err": err(draws[len(draws) // 2]),
                       "random_p95_err": max(err(draws[int(0.05 * len(draws))]),
                                             err(draws[int(0.95 * len(draws))])),
                       "random_median_abs_err": statistics.median(errs),
                       # EXPLORATORY, computed after seeing the preregistered comparison fail.
                       # The prereg compares Latin to the MEDIAN random draw. A practitioner makes
                       # ONE draw, so the share of single draws worse than Latin is the number
                       # they care about. Labelled post hoc wherever it appears.
                       "frac_random_worse_than_latin":
                           sum(1 for e in errs if e > err(lat)) / len(errs),
                       "random_p99_err": sorted(errs)[int(0.99 * len(errs))]}
    return out


def main(argv):
    dirs = sorted({pathlib.Path(a) for a in argv[1:] if pathlib.Path(a).is_dir()})
    print("=" * 100)
    print("INSTRUMENT ARM  --  PREREG_instrument.md")
    print("=" * 100)

    models, unavailable = {}, []
    for d in dirs:
        status, rows = load(d)
        if status.get("state") == "unavailable":
            unavailable.append(status)
            continue
        if rows:
            models[(status.get("pair_key") or rows[0]["pair_key"],
                    status.get("role") or rows[0]["role"])] = {"rows": rows, "status": status}

    # carried gates from the families arm
    carried = {}
    fam = sorted((ROOT / "data").glob("*_families_verdict.json"))
    if fam:
        fv = json.loads(fam[-1].read_text(encoding="utf-8"))
        for k, v in fv["models"].items():
            carried[k] = {"gate_clean": v["gate_clean"], "prior": v["position_prior"]}
        print("\ncarried gates and measured priors from %s" % fam[-1].name)

    report = {}

    # ================= Q1 =================
    print("\n" + "-" * 100)
    print("Q1  THE DETERMINACY DIAL (prereg 8b contrast 1)")
    print("    determinacy = paraphrase agreement at FIXED ordering")
    print("    dominance   = ordering range at FIXED paraphrase")
    rhos = []
    for key in sorted(models):
        tag = "%s_%s" % key
        if carried and not carried.get(tag, {}).get("gate_clean", True):
            print("  %-22s excluded: failed a carried gate in the families arm" % tag)
            continue
        d = q1(models[key]["rows"])
        if not d:
            continue
        items = [k for k, _, _, _ in S.DETERMINACY_BATTERY if k in d
                 and d[k]["determinacy"] is not None and d[k]["dominance"] is not None
                 and math.isfinite(d[k]["dominance"])]
        if len(items) < 3:
            print("  %-22s too few usable items" % tag)
            continue
        det = [d[i]["determinacy"] for i in items]
        dom = [math.log10(d[i]["dominance"]) for i in items]
        rho = spearman(det, dom)
        rhos.append((tag, rho))
        report.setdefault(tag, {})["q1"] = {"items": d, "rho": rho}
        print("  %-22s rho %+.3f   %s" % (
            tag, rho if rho is not None else float("nan"),
            "  ".join("%s:d=%.2f,r=%.0fx" % (i, d[i]["determinacy"], d[i]["dominance"])
                      for i in items)))
    neg = sum(1 for _, r in rhos if r is not None and r < 0)
    print("\n  negative rho on %d of %d gate-clean checkpoints" % (neg, len(rhos)))
    if not rhos:
        q1_verdict, q1_note = "NO_INSTRUMENT", "no gate-clean checkpoint produced a usable dial"
    elif neg * 2 > len(rhos):
        q1_verdict = "DIAL"
        q1_note = ("position dominance falls as determinacy rises on %d of %d checkpoints. The "
                   "graded claim holds and the paper gains a calibration curve." % (neg, len(rhos)))
    elif neg == 0:
        q1_verdict = "INVERTED"
        q1_note = ("no checkpoint shows the predicted sign. Determinate questions are not less "
                   "order-dominated, which contradicts the canary reasoning. Report loudly.")
    else:
        q1_verdict = "NO_DIAL"
        q1_note = ("the sign is not consistent (%d of %d). The graded claim FAILS; the two-point "
                   "contrast stands and the word 'specifically' comes out of the paper."
                   % (neg, len(rhos)))
    print("  Q1 VERDICT: %s" % q1_verdict)
    print("  %s" % q1_note)

    # ================= Q2 =================
    print("\n" + "-" * 100)
    print("Q2  DOES THE MODEL KNOW ABOUT ITS OWN POSITION PRIOR? (prereg 8b contrasts 3-5)")
    print("    stated belief read MARGINALIZED over all 120 orderings")
    print("\n  %-22s %8s %8s %8s %8s  %s"
          % ("checkpoint", "stated", "reverse", "placebo", "measured", "gate"))
    pairs_xy, q2_rows = [], {}
    for key in sorted(models):
        tag = "%s_%s" % key
        r = q2(models[key]["rows"])
        if "forward" not in r:
            continue
        stated = r["forward"]["stated"]
        rev = r.get("reverse", {}).get("stated")
        pla = r.get("placebo", {}).get("stated")
        measured = carried.get(tag, {}).get("prior")

        # gates, per prereg section 8. A consistent belief puts forward+reverse near 1.0, because
        # the reverse variant presents the same scale in the opposite direction.
        gates = {}
        gates["acquiescence"] = (rev is not None
                                 and abs((stated + rev) - 1.0) <= ACQUIESCENCE_TOL)
        gates["placebo"] = (pla is not None and pla < stated)
        gates["liveness"] = r["forward"]["entropy"] >= LIVENESS_GATE
        clean = all(gates.values())
        verdict = "ok" if clean else "UNINFORMATIVE: " + ",".join(
            k for k, v in gates.items() if not v)
        print("  %-22s %8.4f %8s %8s %8s  %s"
              % (tag, stated,
                 "%.4f" % rev if rev is not None else "n/a",
                 "%.4f" % pla if pla is not None else "n/a",
                 "%.4f" % measured if measured is not None else "n/a", verdict))
        q2_rows[tag] = {"stated": stated, "reverse": rev, "placebo": pla,
                        "measured": measured, "gates": gates, "clean": clean,
                        "mass": r["forward"]["mass"]}
        if clean and measured is not None:
            pairs_xy.append((tag, stated, measured))

    rho2, perm_p = None, None
    if len(pairs_xy) >= 3:
        xs = [p[1] for p in pairs_xy]
        ys = [p[2] for p in pairs_xy]
        rho2 = spearman(xs, ys)
        # EXACT permutation test. n is small enough to enumerate every relabelling, so quoting a
        # bare rho at n=6 without saying how often chance produces it would be exactly the kind of
        # under-powered number this paper is about.
        if rho2 is not None and len(xs) <= 8:
            import itertools
            null = [spearman(xs, list(perm)) for perm in itertools.permutations(ys)]
            null = [r for r in null if r is not None]
            perm_p = sum(1 for r in null if abs(r) >= abs(rho2)) / len(null)
            print("\n  exact permutation test over all %d relabellings: "
                  "two-sided p = %.4f (n = %d)" % (len(null), perm_p, len(xs)))
            print("  the smallest p this n can produce is %.4f, so the test is weak by construction"
                  % (2.0 / len(null)))
    print("\n  gate-clean checkpoints usable for the correlation: %d" % len(pairs_xy))
    if rho2 is None:
        q2_verdict = "NO_INSTRUMENT"
        q2_note = ("fewer than three checkpoints pass the acquiescence and placebo gates, so the "
                   "stated belief cannot be correlated with anything. The introspection probe is "
                   "measuring agreeableness or format, not belief.")
    elif rho2 > 0.5:
        q2_verdict = "INTROSPECTIVE"
        q2_note = ("stated susceptibility tracks measured susceptibility at rho %+.3f. This is a "
                   "POSITIVE introspection result on a property with a ground truth, and per the "
                   "prereg it is the headline. It is consistent with recitation as well as "
                   "introspection and we say so." % rho2)
    else:
        q2_verdict = "REPORT-GAP"
        q2_note = ("stated susceptibility does not track measured susceptibility (rho %+.3f). "
                   "Models dominated by option order carry no usable information about it in "
                   "their self-report, on a property that unlike welfare HAS a ground truth."
                   % rho2)
    print("  Q2 VERDICT: %s" % q2_verdict)
    print("  %s" % q2_note)

    # ================= Q3 =================
    print("\n" + "-" * 100)
    print("Q3  DOES A LATIN SQUARE REPLACE ENUMERATION? (prereg 8b contrast 6)")
    lat = q3()
    wins = 0
    print("  %-24s %10s %10s %12s" % ("checkpoint", "latin-5", "random-5 p50", "random-5 worst"))
    for k in sorted(lat):
        v = lat[k]
        won = v["latin_err"] <= v["random_median_abs_err"]
        wins += 1 if won else 0
        print("  %-24s %9.2fx %9.2fx %11.2fx  %s"
              % (k, v["latin_err"], v["random_median_abs_err"], v["random_p95_err"],
                 "latin" if won else "random"))
    if lat:
        print("\n  latin worst error %.2fx, median %.2fx"
              % (max(v["latin_err"] for v in lat.values()),
                 statistics.median([v["latin_err"] for v in lat.values()])))
        print("  random worst 5-95 error %.2fx, median absolute %.2fx"
              % (max(v["random_p95_err"] for v in lat.values()),
                 statistics.median([v["random_median_abs_err"] for v in lat.values()])))
        q3_verdict = "LATIN-WORKS" if wins * 2 > len(lat) else "NO-BENEFIT"
        q3_note = ("a %d-ordering Latin square beats a same-sized random sample on %d of %d "
                   "checkpoints, so the recommendation becomes k passes rather than k factorial"
                   % (5, wins, len(lat))) if wins * 2 > len(lat) else (
            "structure buys nothing over random sampling; recommend more orderings instead")
    else:
        q3_verdict, q3_note = "NO_INSTRUMENT", "no complete census available"
    print("  Q3 VERDICT: %s" % q3_verdict)
    print("  %s" % q3_note)

    if lat:
        # EXPLORATORY. The preregistered bar is the random MEDIAN and it is reported above as the
        # verdict. This is a different question, asked after seeing that comparison fail, and it is
        # labelled post hoc rather than substituted for the preregistered one. Changing the
        # criterion after seeing the data is the move this paper exists to criticise.
        print("\n  EXPLORATORY, post hoc: a practitioner draws ONCE, not many times.")
        print("  %-24s %14s %12s %12s" % ("checkpoint", "P(1 draw worse", "random p99",
                                          "latin"))
        print("  %-24s %14s" % ("", "than latin)"))
        shares = []
        for k in sorted(lat):
            v = lat[k]
            shares.append(v["frac_random_worse_than_latin"])
            print("  %-24s %13.2f %11.2fx %11.2fx"
                  % (k, v["frac_random_worse_than_latin"], v["random_p99_err"], v["latin_err"]))
        print("\n  median share of single random draws worse than the Latin square: %.2f"
              % statistics.median(shares))
        print("  Latin is deterministic, so its error is a fixed bias; random carries tail risk.")
        print("  This is a variance argument, not the accuracy argument the prereg tested.")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / ("%s_instrument_verdict.json" % stamp)
    out.write_text(json.dumps({
        "q1": {"verdict": q1_verdict, "note": q1_note, "rhos": rhos,
               "per_model": {k: v.get("q1") for k, v in report.items()}},
        "q2": {"verdict": q2_verdict, "note": q2_note, "rho": rho2, "perm_p": perm_p,
               "rows": q2_rows,
               "usable": [p[0] for p in pairs_xy]},
        "q3": {"verdict": q3_verdict, "note": q3_note, "per_model": lat},
        "unavailable": unavailable,
    }, indent=1), encoding="utf-8")
    print("\nwrote %s" % out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
