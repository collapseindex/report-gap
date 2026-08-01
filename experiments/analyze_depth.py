"""Score the depth sweep against PREREG_depth.md sections 8 and 9.

Implements the section 8 clause set in full, including Holm across layers and the three-layer
minimum, because this run exists to break our own result and a lenient check would let it survive
by accident.

    python experiments/analyze_depth.py data/depth_base/depth.jsonl data/depth_instruct/depth.jsonl
"""

from __future__ import annotations

import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap import analysis as A          # noqa: E402

NEG_KEYS = {"neg1", "neg2"}
POS_KEYS = {"pos1", "pos2"}
RANDOM_ARMS = ("random_a", "random_b")
MIN_EFFECT = 0.01
MIN_GATE_CLEAN_LAYERS = 3
# the band Venkatesh (arXiv:2605.05653) reports negative valence concentrating in
VENKATESH_BAND = (0.14, 0.27)


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


def vs_random(rows, layer, condition, keys):
    """Paired treatment-minus-matched-random at one layer, per alpha."""
    at = [r for r in rows if r["layer"] == layer]
    alphas = sorted({r["alpha"] for r in at if r["alpha"] > 0.0})
    base = {r["cell"]: r for r in at if r["condition"] == "baseline"}
    out = {}

    def m(row):
        letters = {L for L, k in row["mapping"].items() if k in keys}
        return A.option_mass(row["probs"], letters)

    for a in alphas:
        treat = {r["cell"]: r for r in at if r["condition"] == condition and r["alpha"] == a}
        common = sorted(set(treat) & set(base))
        if not common:
            continue
        t = [m(treat[c]) - m(base[c]) for c in common]
        r = []
        for c in common:
            per = []
            for rnd in RANDOM_ARMS:
                cell = next((x for x in at if x["condition"] == rnd and x["alpha"] == a
                             and x["cell"] == c), None)
                if cell is not None:
                    per.append(m(cell) - m(base[c]))
            r.append(sum(per) / len(per) if per else 0.0)
        out["%.4f" % a] = A.paired_bootstrap([x - y for x, y in zip(t, r)])
    return out


def best(d):
    """The alpha with the largest point estimate, which is what Holm is corrected over."""
    if not d:
        return None
    return max(d.values(), key=lambda v: v.point)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2

    models = {}
    for path in argv[1:]:
        rows = load(path)
        models[rows[0]["model_key"]] = rows
    if set(models) != {"base", "instruct"}:
        raise SystemExit("need one base and one instruct artifact, got %s" % sorted(models))

    print("=" * 96)
    print("DEPTH SWEEP  --  PREREG_depth.md")
    print("=" * 96)

    report = {}
    for key in ("base", "instruct"):
        rows = models[key]
        layers = sorted({(r["layer"], r["frac"]) for r in rows})
        print("\n%s  (%d rows, %d layers)" % (key.upper(), len(rows), len(layers)))
        print("  %-6s %-7s %28s %28s %s"
              % ("layer", "depth", "capability (pos vs rand)", "PRIMARY (neg vs rand)", "gate"))

        per_layer, pvals = {}, {}
        for layer, frac in layers:
            pos = vs_random(rows, layer, "lexical_pos", POS_KEYS)
            neg = vs_random(rows, layer, "lexical_neg", NEG_KEYS)
            bp, bn = best(pos), best(neg)
            gate = bool(bp and bp.lo > 0.0 and bp.point >= MIN_EFFECT)
            per_layer[layer] = {
                "frac": frac, "gate_clean": gate,
                "capability": str(bp) if bp else "none",
                "primary": str(bn) if bn else "none",
                "primary_point": bn.point if bn else float("nan"),
                "in_venkatesh_band": VENKATESH_BAND[0] <= frac <= VENKATESH_BAND[1],
            }
            if gate and bn:
                pvals[str(layer)] = bn.p
            print("  %-6d %-7.2f %28s %28s %s"
                  % (layer, frac, str(bp) if bp else "none", str(bn) if bn else "none",
                     "ok" if gate else "FAILED"))

        rejected = A.holm(pvals) if pvals else {}
        moved = [L for L, info in per_layer.items()
                 if info["gate_clean"] and rejected.get(str(L))
                 and info["primary_point"] >= MIN_EFFECT]
        clean = [L for L, info in per_layer.items() if info["gate_clean"]]
        in_band = [L for L in clean if per_layer[L]["in_venkatesh_band"]]
        print("  gate-clean layers: %s" % (clean or "NONE"))
        print("  of those, inside the 0.14-0.27 band: %s" % (in_band or "NONE"))
        print("  layers where negative mass moved (Holm-corrected, >= %.2f): %s"
              % (MIN_EFFECT, moved or "NONE"))
        report[key] = {"layers": per_layer, "holm": rejected, "moved": moved,
                       "gate_clean": clean, "in_band": in_band}

    # ---- the branch, per prereg section 8 ----
    print("\n" + "-" * 96)
    print("VERDICT (prereg section 8, clause by clause)")
    inst = report["instruct"]
    clauses = {
        "at least %d instruct layers have a clean capability gate" % MIN_GATE_CLEAN_LAYERS:
            len(inst["gate_clean"]) >= MIN_GATE_CLEAN_LAYERS,
        "at least one gate-clean instruct layer lies in the 0.14-0.27 band":
            bool(inst["in_band"]),
        "no gate-clean instruct layer shows negative mass moving": not inst["moved"],
    }
    for k, v in clauses.items():
        print("  [%s] %s" % ("ok " if v else "NO ", k))

    if len(inst["gate_clean"]) < MIN_GATE_CLEAN_LAYERS:
        verdict = "NO_INSTRUMENT"
        note = ("fewer than %d instruct layers have a working instrument, so this sweep licenses "
                "nothing about either branch" % MIN_GATE_CLEAN_LAYERS)
    elif inst["moved"]:
        verdict = "DEPTH-ARTIFACT"
        note = ("negative mass moves on gate-clean instruct layer(s) %s. The previous nulls were "
                "measured at the wrong depth for that pole. RESULTS_pair.md and RESULTS_floor.md "
                "need corrections in the same commit as this result." % inst["moved"])
    elif not inst["in_band"]:
        verdict = "INCONCLUSIVE"
        note = ("no gate-clean layer falls inside the band where the effect was predicted, so the "
                "directed attempt to break the null did not actually reach the target region")
    else:
        verdict = "DEPTH-ROBUST"
        note = ("negative mass is null at every gate-clean depth including inside the predicted "
                "band. The tuning claim survives a directed attempt to break it and should be "
                "stated as null across %d depths." % len(inst["gate_clean"]))

    print("\n  VERDICT: %s" % verdict)
    print("  %s" % note)

    report["verdict"] = verdict
    report["note"] = note
    report["clauses"] = clauses
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(argv[1]).parent.parent / ("%s_depth_verdict.json" % stamp)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
