"""Score the base/instruct pair against PREREG_base_pair.md sections 5 and 6.

Every clause requires an interval that excludes zero AND a point estimate clearing MIN_EFFECT.
Pure significance tests certify nonsense at this n: in the previous arm a capability gate passed on
+0.0000 and a confound control failed on +0.0002. The floor is carried over from the readout arm's
measured random-direction artifacts (+0.0008 to +0.0023), not chosen here.

    python experiments/analyze_pair.py data/pair_base/pair.jsonl data/pair_instruct/pair.jsonl
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
        raise SystemExit("%s has %d unparseable line(s); complete the run rather than scoring the "
                         "fragment" % (path, torn))
    return rows


def index(rows):
    out = {}
    for r in rows:
        key = (r["condition"], r["alpha"], r["cell"])
        if key in out:
            raise SystemExit("duplicate row for %s; pairing is not what it looks like" % (key,))
        out[key] = r
    return out


def cells(idx, condition, alpha):
    return {c: r for (cond, a, c), r in idx.items()
            if cond == condition and abs(a - alpha) < 1e-12}


def mass(row, keys):
    letters = {L for L, k in row["mapping"].items() if k in keys}
    return A.option_mass(row["probs"], letters)


def vs_random(idx, condition, keys, alphas):
    out = {}
    base = cells(idx, "baseline", 0.0)
    for a in alphas:
        treat = cells(idx, condition, a)
        common = sorted(set(treat) & set(base))
        if not common:
            continue
        t = [mass(treat[c], keys) - mass(base[c], keys) for c in common]
        r = []
        for c in common:
            per = [mass(cells(idx, rnd, a)[c], keys) - mass(base[c], keys)
                   for rnd in RANDOM_ARMS if c in cells(idx, rnd, a)]
            r.append(sum(per) / len(per) if per else 0.0)
        out["%.4f" % a] = A.paired_bootstrap([x - y for x, y in zip(t, r)])
    return out


def positive(d):
    return any(v.lo > 0.0 and v.point >= MIN_EFFECT for v in d.values())


def null(d):
    return not any(v.excludes_zero and abs(v.point) >= MIN_EFFECT for v in d.values())


def show(label, d):
    print("    %s" % label)
    for a, v in d.items():
        print("      %s  %s" % (a, v))


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2

    models = {}
    for path in argv[1:]:
        rows = load(path)
        key = rows[0]["model_key"]
        idx = index(rows)
        alphas = sorted({a for (_c, a, _cell) in idx if a > 0.0})
        models[key] = {"idx": idx, "alphas": alphas, "n": len(rows), "path": path}

    if set(models) != {"base", "instruct"}:
        raise SystemExit("need one base and one instruct artifact, got %s" % sorted(models))

    print("=" * 78)
    print("BASE / INSTRUCT PAIR  --  PREREG_base_pair.md")
    print("=" * 78)

    report, results = {}, {}
    for key in ("base", "instruct"):
        m = models[key]
        idx, alphas = m["idx"], m["alphas"]
        neg = vs_random(idx, "lexical_neg", NEG_KEYS, alphas)
        pos = vs_random(idx, "lexical_pos", POS_KEYS, alphas)
        neut = vs_random(idx, "lexical_neg", {"neut"}, alphas)
        results[key] = {"gate_clean": positive(pos), "primary_positive": positive(neg),
                        "primary_null": null(neg), "neutral_up": positive(neut),
                        "neg_flat": null(neg)}
        print("\n%s  (%d rows, alphas %s)" % (key.upper(), m["n"], alphas))
        show("capability gate, positive mass vs random:", pos)
        show("PRIMARY, negative mass vs random:", neg)
        show("neutral mass vs random:", neut)
        if not results[key]["gate_clean"]:
            print("    -> GATE FAILED. This model's null is UNINFORMATIVE.")
        elif results[key]["primary_positive"]:
            print("    -> negative content above matched random.")
        else:
            print("    -> no negative content, on a working instrument.")
        report[key] = {"capability": {k: str(v) for k, v in pos.items()},
                       "primary": {k: str(v) for k, v in neg.items()},
                       "neutral": {k: str(v) for k, v in neut.items()},
                       **results[key]}

    # ---- the branch, per prereg section 5 ----
    print("\n" + "-" * 78)
    print("VERDICT (prereg section 5, clause by clause)")

    both_gates = results["base"]["gate_clean"] and results["instruct"]["gate_clean"]
    replication = results["instruct"]["neutral_up"] and results["instruct"]["neg_flat"]

    clauses = {
        "base capability gate clean": results["base"]["gate_clean"],
        "instruct capability gate clean": results["instruct"]["gate_clean"],
        "instruct reproduces the neutral floor in this format": replication,
    }
    for k, v in clauses.items():
        print("  [%s] %s" % ("ok " if v else "NO ", k))

    if not both_gates:
        verdict = "NO_INSTRUMENT"
        note = ("at least one capability gate failed, so no branch is selected and the nulls "
                "license nothing")
    elif not replication:
        verdict = "FORMAT-DEPENDENT"
        note = ("the instruct model does not reproduce the neutral floor in plain-completion "
                "format, so the earlier finding was format-specific. This RETRACTS rather than "
                "extends RESULTS_floor.md and must be reported as a retraction")
    elif results["base"]["primary_positive"] and results["instruct"]["primary_null"]:
        verdict = "TUNING-LOCALIZED"
        note = ("the direction adds negative content to the base model and not to the tuned one, "
                "at matched norm, format and architecture")
    elif results["base"]["primary_null"] and results["instruct"]["primary_null"]:
        verdict = "DIRECTION-LIMITED"
        note = ("the direction adds no negative content to either member of the pair, so the "
                "neutral floor is a property of the direction rather than of preference tuning")
    else:
        verdict = "NEITHER"
        note = ("no branch has its full clause set; report each model at the scope it showed and "
                "name the clause that failed")

    print("\n  VERDICT: %s" % verdict)
    print("  %s" % note)

    report["verdict"] = verdict
    report["note"] = note
    report["clauses"] = clauses
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(argv[1]).parent.parent / ("%s_pair_verdict.json" % stamp)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
