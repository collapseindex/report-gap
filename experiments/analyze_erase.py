"""Score the erase arm against PREREG_erase.md sections 8 and 9.

Two gates run before any primary is read, both in code:

  erase-artifact gate   projecting a direction out is itself a perturbation. If `erase_only` moves
                        the probe with no injection present, that erase layer is invalidated and
                        its primary is `uninformative` whatever it shows.
  capability gate       the positive pole under the same erase must move the probe, or the arm
                        cannot testify that the negative pole does not.

Option mass is printed and carries NO verdict, per the prereg: that channel is dominated by
option-ordering noise at the permutation counts this project can afford.

    python experiments/analyze_erase.py data/erase_base/erase.jsonl data/erase_instruct/erase.jsonl
"""

from __future__ import annotations

import datetime
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap import analysis as A          # noqa: E402

NEG_KEYS = {"neg1", "neg2"}
RANDOM_ARMS = ("random_a_erase", "random_b_erase")
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
    if not rows:
        raise SystemExit("%s holds no rows" % path)
    if torn:
        raise SystemExit("%s has %d unparseable line(s)" % (path, torn))
    return rows


def paired(rows, layer, cond_a, cond_b, value, sd):
    """cond_a minus cond_b, paired per cell, standardized by the baseline probe SD."""
    at = [r for r in rows if r["erase_layer"] == layer or r["condition"] == "baseline"]
    A_ = {r["cell"]: r for r in rows if r["condition"] == cond_a and r["erase_layer"] == layer}
    if cond_b == "baseline":
        B_ = {r["cell"]: r for r in rows if r["condition"] == "baseline"}
    else:
        B_ = {r["cell"]: r for r in rows if r["condition"] == cond_b and r["erase_layer"] == layer}
    common = sorted(set(A_) & set(B_))
    if not common:
        return None
    return A.paired_bootstrap([(value(A_[c]) - value(B_[c])) / sd for c in common])


def vs_random(rows, layer, cond, value, sd):
    A_ = {r["cell"]: r for r in rows if r["condition"] == cond and r["erase_layer"] == layer}
    common = sorted(A_)
    deltas = []
    for c in common:
        per = [value(r) for rnd in RANDOM_ARMS for r in rows
               if r["condition"] == rnd and r["erase_layer"] == layer and r["cell"] == c]
        if per:
            deltas.append((value(A_[c]) - sum(per) / len(per)) / sd)
    return A.paired_bootstrap(deltas) if len(deltas) > 1 else None


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2

    models = {}
    for path in argv[1:]:
        rows = load(path)
        models[rows[0]["model_key"]] = (rows, pathlib.Path(path).parent)
    if set(models) != {"base", "instruct"}:
        raise SystemExit("need one base and one instruct artifact")

    print("=" * 96)
    print("ERASE ARM  --  PREREG_erase.md")
    print("=" * 96)

    report = {}
    for key in ("base", "instruct"):
        rows, folder = models[key]
        var = json.loads((folder / "ordering_variance.json").read_text(encoding="utf-8"))
        base_scores = [r["probe_orth"] for r in rows if r["condition"] == "baseline"]
        sd = statistics.pstdev(base_scores) or 1.0

        def probe(r):
            return r["probe_orth"]

        print("\n%s" % key.upper())
        print("  baseline probe SD %.4f | SD across orderings %.4f (%.0f%% of within-ordering SD)"
              % (sd, var["sd_across_orderings"],
                 100 * var["sd_across_orderings"] / max(1e-9, sd)))

        no_erase = vs_random(rows, -1, "neg", probe, sd)
        neg_plain = paired(rows, -1, "neg", "baseline", probe, sd)
        print("  neg without erase, vs baseline: %s" % neg_plain)

        layers = sorted({r["erase_layer"] for r in rows if r["erase_layer"] > 0})
        per_layer, pvals = {}, {}
        print("\n  %-7s %26s %26s %26s" % ("erase L", "erase_only vs baseline",
                                           "capability (pos vs rand)", "PRIMARY (neg vs rand)"))
        for L in layers:
            artifact = paired(rows, L, "erase_only", "baseline", probe, sd)
            cap = vs_random(rows, L, "pos_erase", probe, sd)
            prim = vs_random(rows, L, "neg_erase", probe, sd)
            art_clean = artifact is not None and (
                not artifact.excludes_zero or abs(artifact.point) < PROBE_FLOOR_SD)
            cap_clean = cap is not None and cap.lo > 0.0 and cap.point >= PROBE_FLOOR_SD
            survived = prim is not None and prim.hi < 0.0 and abs(prim.point) >= PROBE_FLOOR_SD
            print("  %-7d %26s %26s %26s  %s" % (
                L, str(artifact), str(cap), str(prim),
                "ok" if (art_clean and cap_clean) else "GATE FAILED"))
            per_layer[L] = {"erase_artifact": str(artifact), "capability": str(cap),
                            "primary": str(prim), "primary_point": prim.point if prim else None,
                            "artifact_clean": art_clean, "capability_clean": cap_clean,
                            "survived": survived}
            if art_clean and cap_clean and prim is not None:
                pvals[str(L)] = prim.p

        rejected = A.holm(pvals) if pvals else {}
        clean = [L for L, v in per_layer.items() if v["artifact_clean"] and v["capability_clean"]]
        survived = [L for L in clean if per_layer[L]["survived"] and rejected.get(str(L))]
        print("  gate-clean erase layers: %s" % (clean or "NONE"))
        print("  layers where the probe SURVIVED the erase (Holm): %s" % (survived or "NONE"))

        # The profile may only use GATE-CLEAN layers. The first version compared E=30 to E=25, and
        # E=25 fails its erase-artifact gate on both models: its primary is `uninformative` per the
        # prereg, so it cannot carry a profile either.
        profile = None
        if len(clean) >= 2:
            early, late = min(clean), max(clean)
            profile = abs(per_layer[late]["primary_point"]) - abs(per_layer[early]["primary_point"])
            print("  profile over GATE-CLEAN layers only, E=%d to E=%d: %+.4f  (%s)"
                  % (early, late, profile,
                     "later erase leaves more" if profile > 0 else "flat or inverted"))
        else:
            print("  profile: not computable, fewer than two gate-clean layers")

        # The confound the profile cannot escape on its own: erasing earlier perturbs MORE, so a
        # growing primary is partly just a shrinking perturbation. Report the erase_only artifact
        # beside it. If the profile were pure perturbation, this ratio would be FLAT.
        # Exploratory, computed after seeing the data.
        print("  %-7s %12s %12s %10s" % ("erase L", "|primary|", "erase_only", "ratio"))
        ratios = {}
        for L in sorted(per_layer):
            art = per_layer[L]["erase_artifact"]
            art_pt = abs(float(art.split()[0])) if art and art != "None" else float("nan")
            pri = abs(per_layer[L]["primary_point"] or float("nan"))
            ratios[L] = pri / art_pt if art_pt else float("nan")
            print("  %-7d %12.4f %12.4f %10.1f%s"
                  % (L, pri, art_pt, ratios[L], "" if L in clean else "   (gate failed)"))

        report[key] = {"layers": per_layer, "gate_clean": clean, "survived": survived,
                       "primary_to_artifact_ratio": ratios,
                       "profile_late_minus_early": profile, "baseline_probe_sd": sd,
                       "sd_across_orderings": var["sd_across_orderings"],
                       "neg_no_erase_vs_baseline": str(neg_plain),
                       "neg_no_erase_vs_random": str(no_erase)}

    # ---- verdict on the instruct model, per prereg section 8 ----
    print("\n" + "-" * 96)
    print("VERDICT (prereg section 8, clause by clause)")
    inst = report["instruct"]
    prof = inst["profile_late_minus_early"]
    clauses = {
        "at least two gate-clean erase layers": len(inst["gate_clean"]) >= 2,
        "probe survives the erase at one or more gate-clean layers": bool(inst["survived"]),
        "profile: later erase leaves more than an immediate one": bool(prof and prof > 0),
    }
    for k, v in clauses.items():
        print("  [%s] %s" % ("ok " if v else "NO ", k))

    if len(inst["gate_clean"]) < 2:
        verdict, note = "NO_INSTRUMENT", ("fewer than two erase layers pass both gates, so no "
                                          "null is interpretable")
    elif inst["survived"] and clauses["profile: later erase leaves more than an immediate one"]:
        verdict, note = "TRANSFORMED", ("the probe reads the state after the direction is projected "
                                        "out, and more so the later the projection, which is what "
                                        "a downstream transformation looks like")
    elif inst["survived"]:
        verdict, note = "FLAT-SURVIVAL", (
            "signal survives at every erase point with no temporal profile. The likely reading is "
            "that p_orth and d_hat share a subspace the erase does not reach, both being fit from "
            "the same lexical axis. This is an instrument confound, NOT support for TRANSFORMED")
    else:
        verdict, note = "WAKE", ("the probe signal dies with the erase at every gate-clean layer. "
                                 "Nothing survived that was not the injected vector persisting, and "
                                 "RESULTS_shell.md's SHELL reading is retracted rather than "
                                 "qualified")

    print("\n  VERDICT: %s" % verdict)
    print("  %s" % note)

    report["verdict"], report["note"], report["clauses"] = verdict, note, clauses
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(argv[1]).parent.parent / ("%s_erase_verdict.json" % stamp)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
