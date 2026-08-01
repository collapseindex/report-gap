"""Score the families arm against PREREG_families.md sections 8, 8b and 9.

The claim under test: the collapse of a five-option self-report readout into a position prior is a
property of PREFERENCE TUNING and appears across architecture families, not only in
Qwen2.5-3B-Instruct.

Three gates run in code before any endpoint is read, per prereg section 8:

  reproduction   the Qwen2.5-3B rows must match the committed enumerate artifact. If they do not,
                 the arm is VOID and the discrepancy is the finding.
  canary         mean accuracy across orderings >= 0.50, or the model cannot do the format and its
                 position prior says nothing about self-report.
  liveness       mean baseline option entropy >= 0.10 nats, or the readout is pinned and has no
                 room to show an ordering effect.

Qwen2.5-3B is EXCLUDED from the majority count: the hypothesis came from it. Families vote, not
pairs, so one lineage cannot outvote the field by being cheap to run.

    python experiments/analyze_families.py data/fam_*/
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

from report_gap import analysis as A          # noqa: E402

NEG_KEYS = {"neg1", "neg2"}
CANARY_GATE = 0.50
LIVENESS_GATE = 0.10
REPRO_TOL = 5e-4
REPRO_PAIR = "qwen3b"
SUBSAMPLE_K = (2, 4, 8, 16, 32, 64)
SUBSAMPLE_TRIALS = 4000
ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_dir(d: pathlib.Path):
    status_p, rows_p = d / "status.json", d / "enum.jsonl"
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


def pole(row, keys):
    return sum(p for L, p in row["probs"].items() if row["mapping"][L] in keys)


def per_ordering_pole(rows, condition="letters"):
    per = collections.defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            per[tuple(r["ordering"])].append(pole(r, NEG_KEYS))
    return {o: statistics.fmean(v) for o, v in per.items()}


def summarize(vals):
    if not vals:
        return None
    s = sorted(vals)
    def q(p):
        return s[min(len(s) - 1, int(p * len(s)))]
    lo, hi = s[0], s[-1]
    return {"n": len(s), "min": lo, "p05": q(0.05), "p50": q(0.50), "p95": q(0.95), "max": hi,
            "ratio": (hi / lo) if lo > 0 else float("inf"), "mean": statistics.fmean(s)}


def position_prior(rows):
    """Max per-label mass with five IDENTICAL options. Flat would be 0.2000."""
    ident = [r for r in rows if r["condition"] == "identical"]
    if not ident:
        return None, {}
    labels = sorted(ident[0]["probs"])
    per = {L: statistics.fmean([r["probs"][L] for r in ident]) for L in labels}
    return max(per.values()), per


def canary_accuracy(rows):
    per = collections.defaultdict(list)
    for r in rows:
        if r["condition"] == "canary":
            correct = [L for L, k in r["mapping"].items() if k == "four"]
            if correct:
                per[r["ordering_index"]].append(1.0 if r["argmax"] == correct[0] else 0.0)
    if not per:
        return None, None
    means = [statistics.fmean(v) for v in per.values()]
    return statistics.fmean(means), (statistics.pstdev(means) if len(means) > 1 else 0.0)


def liveness(rows):
    ent = [r["entropy"] for r in rows if r["condition"] == "letters"]
    return statistics.fmean(ent) if ent else None


def subsample_recovery(vals, seed=0):
    """Prereg contrast 6: what does a study that samples k orderings actually see?"""
    if len(vals) < max(SUBSAMPLE_K):
        return {}
    rng = random.Random(seed)
    out = {}
    truth_ratio = max(vals) / min(vals) if min(vals) > 0 else float("inf")
    for k in SUBSAMPLE_K:
        means, ratios = [], []
        for _ in range(SUBSAMPLE_TRIALS):
            s = rng.sample(vals, k)
            means.append(statistics.fmean(s))
            ratios.append(max(s) / min(s) if min(s) > 0 else float("inf"))
        means.sort()
        ratios.sort()
        out[k] = {
            "mean_p05": means[int(0.05 * len(means))],
            "mean_p50": means[len(means) // 2],
            "mean_p95": means[int(0.95 * (len(means) - 1))],
            "ratio_p50": ratios[len(ratios) // 2],
            "ratio_frac_under_10x": sum(1 for r in ratios if r < 10) / len(ratios),
            "ratio_as_frac_of_truth": (ratios[len(ratios) // 2] / truth_ratio)
            if math.isfinite(truth_ratio) and truth_ratio else float("nan"),
        }
    return out


def check_reproduction(rows):
    """The Qwen2.5-3B rows must match the committed enumerate artifact, or the arm is void."""
    problems, checked = [], 0
    for role in ("base", "instruct"):
        old_p = ROOT / "data" / ("enum_%s" % role) / "enum.jsonl"
        new_rows = [r for r in rows if r.get("role") == role]
        if not old_p.exists() or not new_rows:
            continue
        old = [json.loads(l) for l in old_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        old_ord = per_ordering_pole(old)
        new_ord = per_ordering_pole(new_rows)
        common = sorted(set(old_ord) & set(new_ord))
        if not common:
            problems.append("%s: no orderings in common" % role)
            continue
        worst = max(abs(old_ord[o] - new_ord[o]) for o in common)
        checked += len(common)
        print("  reproduction %-9s %3d orderings in common, worst |delta| %.6f  %s"
              % (role, len(common), worst, "ok" if worst < REPRO_TOL else "MISMATCH"))
        if worst >= REPRO_TOL:
            problems.append("%s: worst delta %.6f exceeds %.0e" % (role, worst, REPRO_TOL))
    return problems, checked


def main(argv):
    dirs = [pathlib.Path(a) for a in argv[1:]]
    dirs = sorted({d for d in dirs if d.is_dir()})
    if not dirs:
        print(__doc__)
        return 2

    print("=" * 100)
    print("FAMILIES ARM  --  PREREG_families.md")
    print("=" * 100)

    models, unavailable = {}, []
    for d in dirs:
        status, rows = load_dir(d)
        if status.get("state") == "unavailable":
            unavailable.append(status)
            continue
        if not rows:
            continue
        key = (status.get("pair_key") or rows[0]["pair_key"],
               status.get("role") or rows[0]["role"])
        models[key] = {"status": status, "rows": rows,
                       "family": status.get("family") or rows[0]["family"],
                       "model": status.get("model", "?")}

    if unavailable:
        print("\nUNAVAILABLE (recorded, not dropped):")
        for s in unavailable:
            print("  %-12s %-9s %-34s %s"
                  % (s["pair_key"], s["role"], s["model"], str(s.get("error"))[:70]))

    # ---- gate 0: reproduction. Checked FIRST; a failure voids the arm. ----
    print("\nREPRODUCTION CONTROL (prereg 7): the %s rows must match the enumerate artifact"
          % REPRO_PAIR)
    repro_rows = [r for k, v in models.items() if k[0] == REPRO_PAIR for r in v["rows"]]
    if not repro_rows:
        print("  %s not present in this run; reproduction control NOT checked" % REPRO_PAIR)
        repro_problems = []
    else:
        repro_problems, _ = check_reproduction(repro_rows)
    if repro_problems:
        print("\n" + "!" * 100)
        print("ARM VOID: the reproduction control failed. %s" % "; ".join(repro_problems))
        print("Investigate the code before interpreting anything else in this run.")
        print("!" * 100)
        return 1

    # ---- per-model endpoints and gates ----
    print("\n%-12s %-9s %-11s %8s %8s %8s %9s %7s  %s"
          % ("pair", "role", "family", "prior", "ratio", "canary", "can.sd", "entropy", "gate"))
    report = {}
    for key in sorted(models):
        m = models[key]
        rows = m["rows"]
        prior, per_label = position_prior(rows)
        ord_pole = per_ordering_pole(rows)
        summ = summarize(list(ord_pole.values()))
        can, can_sd = canary_accuracy(rows)
        ent = liveness(rows)

        gates = {
            "canary": can is not None and can >= CANARY_GATE,
            "liveness": ent is not None and ent >= LIVENESS_GATE,
        }
        verdict = "ok" if all(gates.values()) else "GATE FAILED: " + ",".join(
            k for k, v in gates.items() if not v)
        print("%-12s %-9s %-11s %8s %8s %8s %9s %7s  %s"
              % (key[0], key[1], m["family"],
                 "%.4f" % prior if prior is not None else "n/a",
                 ("%.1f" % summ["ratio"]) if summ else "n/a",
                 "%.4f" % can if can is not None else "n/a",
                 "%.4f" % can_sd if can_sd is not None else "n/a",
                 "%.3f" % ent if ent is not None else "n/a",
                 verdict))
        report["%s_%s" % key] = {
            "family": m["family"], "model": m["model"], "position_prior": prior,
            "per_label": per_label, "ordering_summary": summ,
            "canary_mean": can, "canary_sd": can_sd, "baseline_entropy": ent,
            "gates": gates, "gate_clean": all(gates.values()),
        }

    # ---- per-pair primary, then the family vote ----
    print("\nPRIMARY (prereg 8b contrast 1): position_prior(instruct) - position_prior(base)")
    pairs = {}
    for pair_key in sorted({k[0] for k in models}):
        b, i = models.get((pair_key, "base")), models.get((pair_key, "instruct"))
        if not b or not i:
            print("  %-12s incomplete pair, excluded" % pair_key)
            continue
        rb, ri = report["%s_base" % pair_key], report["%s_instruct" % pair_key]
        if not (rb["gate_clean"] and ri["gate_clean"]):
            print("  %-12s %-11s gate failed on %s, excluded from the primary"
                  % (pair_key, rb["family"],
                     "base" if not rb["gate_clean"] else "instruct"))
            pairs[pair_key] = {"family": rb["family"], "gate_clean": False}
            continue
        delta = ri["position_prior"] - rb["position_prior"]
        lr = None
        if rb["ordering_summary"] and ri["ordering_summary"]:
            rr, rrb = ri["ordering_summary"]["ratio"], rb["ordering_summary"]["ratio"]
            if math.isfinite(rr) and math.isfinite(rrb) and rrb > 0:
                lr = math.log10(rr) - math.log10(rrb)
        excluded = pair_key == REPRO_PAIR
        print("  %-12s %-11s base %.4f  instruct %.4f  delta %+.4f   log10 range delta %s%s"
              % (pair_key, rb["family"], rb["position_prior"], ri["position_prior"], delta,
                 ("%+.2f" % lr) if lr is not None else "n/a",
                 "   [EXCLUDED: hypothesis came from this pair]" if excluded else ""))
        pairs[pair_key] = {"family": rb["family"], "gate_clean": True, "delta": delta,
                           "log10_range_delta": lr, "excluded": excluded}

    votes = collections.defaultdict(list)
    for pk, v in pairs.items():
        if v.get("gate_clean") and not v.get("excluded"):
            votes[v["family"]].append(v["delta"])

    print("\nFAMILY VOTE (prereg 8b contrast 2). Families vote, not pairs. %s excluded."
          % REPRO_PAIR)
    positive, total = 0, 0
    for fam in sorted(votes):
        ds = votes[fam]
        up = sum(1 for d in ds if d > 0)
        direction = "instruct HIGHER" if up * 2 > len(ds) else (
            "base HIGHER" if up * 2 < len(ds) else "TIED")
        positive += 1 if up * 2 > len(ds) else 0
        total += 1
        print("  %-11s %d pair(s), %d with instruct higher  ->  %s   deltas %s"
              % (fam, len(ds), up, direction, ["%+.4f" % d for d in ds]))

    # ---- subsample recovery, prereg contrast 6 ----
    print("\nSUBSAMPLE RECOVERY (prereg 8b contrast 6): what a study sampling k orderings sees")
    subs = {}
    for key in sorted(models):
        if not report["%s_%s" % key]["gate_clean"]:
            continue
        vals = list(per_ordering_pole(models[key]["rows"]).values())
        rec = subsample_recovery(vals)
        if not rec:
            continue
        subs["%s_%s" % key] = rec
        truth = max(vals) / min(vals) if min(vals) > 0 else float("inf")
        cells = "  ".join("k=%d:%.0f%%" % (k, 100 * rec[k]["ratio_as_frac_of_truth"])
                          for k in SUBSAMPLE_K)
        print("  %-12s %-9s true range %8.1fx | median range SEEN as %% of truth:  %s"
              % (key[0], key[1], truth, cells))

    # ---- verdict, prereg section 9 ----
    print("\n" + "-" * 100)
    print("VERDICT (prereg section 9)")
    if total == 0:
        verdict = "NO_INSTRUMENT"
        note = ("no gate-clean pair outside %s. The arm cannot address the claim; report how many "
                "models failed which gate." % REPRO_PAIR)
    elif positive * 2 > total:
        verdict = "TUNING-GENERAL"
        note = ("the instruct checkpoint shows the larger position prior in %d of %d families. The "
                "readout collapse is a property of preference tuning and generalizes beyond Qwen."
                % (positive, total))
    elif positive == 0:
        verdict = "QWEN-SPECIFIC"
        note = ("no family outside %s shows the direction. The 986x is a property of one lineage. "
                "Per prereg section 0 the paper's title and abstract must be rewritten to name the "
                "model, and the general claim is withdrawn." % REPRO_PAIR)
    else:
        verdict = "MIXED"
        note = ("the direction holds in %d of %d families. Not a majority, so the claim in section "
                "0 is REFUTED as stated; report the split and narrow the paper to the families "
                "where it holds." % (positive, total))
    print("  VERDICT: %s" % verdict)
    print("  %s" % note)

    n_failed = sum(1 for k, v in report.items() if not v["gate_clean"])
    print("\n  gate failures: %d of %d checkpoints; unavailable: %d"
          % (n_failed, len(report), len(unavailable)))
    print("  These counts belong in the RESULTS abstract, not buried (prereg section 8).")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / ("%s_families_verdict.json" % stamp)
    out.write_text(json.dumps({
        "verdict": verdict, "note": note, "models": report, "pairs": pairs,
        "family_votes": {k: v for k, v in votes.items()},
        "families_positive": positive, "families_total": total,
        "subsample_recovery": subs,
        "unavailable": unavailable, "gate_failures": n_failed,
    }, indent=1), encoding="utf-8")
    print("\nwrote %s" % out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
