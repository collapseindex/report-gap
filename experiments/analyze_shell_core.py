"""Score the Shell-vs-Core artifact against PREREG_shell_core.md sections 8 and 9.

Every table prints the ORTHOGONALIZED probe effect next to the UN-orthogonalized one, because the
difference between them is how much of the apparent representation was just the injected vector
persisting. SHELL requires the orthogonalized effect to be at least a third of the raw one.

    python experiments/analyze_shell_core.py data/shell_base/shell.jsonl data/shell_instruct/shell.jsonl
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
POS_KEYS = {"pos1", "pos2"}
RANDOM_ARMS = ("random_a", "random_b")
MASS_FLOOR = 0.01          # carried over from the readout arm
PROBE_FLOOR_SD = 0.10      # in units of the baseline probe score's SD, fixed in the prereg
MIN_ORTH_FRACTION = 1.0 / 3.0


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


def mass(row, keys):
    letters = {L for L, k in row["mapping"].items() if k in keys}
    return A.option_mass(row["probs"], letters)


def vs_random(rows, layer, condition, value):
    """Paired treatment-minus-matched-random at one injection layer, per alpha."""
    at = [r for r in rows if r["inject_layer"] == layer]
    alphas = sorted({r["alpha"] for r in at if r["alpha"] > 0.0})
    base = {r["cell"]: r for r in at if r["condition"] == "baseline"}
    out = {}
    for a in alphas:
        treat = {r["cell"]: r for r in at if r["condition"] == condition and r["alpha"] == a}
        common = sorted(set(treat) & set(base))
        if not common:
            continue
        t = [value(treat[c]) - value(base[c]) for c in common]
        r = []
        for c in common:
            per = []
            for rnd in RANDOM_ARMS:
                cell = next((x for x in at if x["condition"] == rnd and x["alpha"] == a
                             and x["cell"] == c), None)
                if cell is not None:
                    per.append(value(cell) - value(base[c]))
            r.append(sum(per) / len(per) if per else 0.0)
        out["%.4f" % a] = A.paired_bootstrap([x - y for x, y in zip(t, r)])
    return out


def best(d):
    return max(d.values(), key=lambda v: v.point) if d else None


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

    print("=" * 100)
    print("SHELL VS CORE  --  PREREG_shell_core.md")
    print("=" * 100)

    report = {}
    for key in ("base", "instruct"):
        rows = models[key]
        layers = sorted({r["inject_layer"] for r in rows}, reverse=True)
        print("\n%s  (%d rows, probe layer %d, cos(p,d)=%+.4f)"
              % (key.upper(), len(rows), rows[0]["probe_layer"], rows[0]["cos_p_d"]))

        per_layer = {}
        for layer in layers:
            at = [r for r in rows if r["inject_layer"] == layer]
            base_scores = [r["probe_orth"] for r in at if r["condition"] == "baseline"]
            sd = statistics.pstdev(base_scores) or 1.0

            def orth(r):
                return r["probe_orth"] / sd

            def raw(r):
                return r["probe_raw"] / sd

            probe_neg = vs_random(rows, layer, "lexical_neg", orth)
            probe_pos = vs_random(rows, layer, "lexical_pos", orth)
            probe_neg_raw = vs_random(rows, layer, "lexical_neg", raw)
            mass_neg = vs_random(rows, layer, "lexical_neg", lambda r: mass(r, NEG_KEYS))
            mass_pos = vs_random(rows, layer, "lexical_pos", lambda r: mass(r, POS_KEYS))

            bn, bp, bnr = best(probe_neg), best(probe_pos), best(probe_neg_raw)
            bmn, bmp = best(mass_neg), best(mass_pos)
            gate = bool(bp and bp.lo > 0.0 and bp.point >= PROBE_FLOOR_SD)
            moved = bool(bn and bn.lo > 0.0 and bn.point >= PROBE_FLOOR_SD)
            mass_moved = bool(bmn and bmn.lo > 0.0 and bmn.point >= MASS_FLOOR)
            frac = (abs(bn.point) / abs(bnr.point)) if (bn and bnr and abs(bnr.point) > 1e-9) \
                else float("nan")

            print("\n  inject layer %d   (baseline probe SD %.4f)" % (layer, sd))
            print("    probe gate  (pos, orth)  %s   %s" % (bp, "ok" if gate else "FAILED"))
            print("    PROBE       (neg, orth)  %s   %s" % (bn, "moved" if moved else "null"))
            print("    probe       (neg, RAW)   %s" % bnr)
            print("    orthogonalized / raw     %.3f%s" % (frac,
                  "" if frac >= MIN_ORTH_FRACTION else "   <- below the 1/3 bar"))
            print("    option mass (neg)        %s   %s" % (bmn, "moved" if mass_moved else "null"))
            print("    option mass (pos)        %s" % bmp)

            per_layer[layer] = {
                "probe_gate_clean": gate, "probe_moved": moved, "mass_moved": mass_moved,
                "orth_fraction": frac,
                "probe_pos": str(bp), "probe_neg": str(bn), "probe_neg_raw": str(bnr),
                "mass_neg": str(bmn), "mass_pos": str(bmp), "baseline_probe_sd": sd,
            }
        report[key] = {"layers": per_layer, "probe_layer": rows[0]["probe_layer"],
                       "cos_p_d": rows[0]["cos_p_d"]}

    # ---- end-to-end validation: the probe must see what reaches the options in the BASE model ----
    print("\n" + "-" * 100)
    base_valid = any(v["probe_moved"] and v["mass_moved"] for v in report["base"]["layers"].values())
    print("[validation] base model: probe and option mass both move at some layer: %s"
          % ("YES" if base_valid else "NO"))
    if not base_valid:
        print("  The probe cannot see the state in the one model where it demonstrably reaches the")
        print("  options. No instruct result is interpretable. Contrast 5 exists to catch this.")

    inst = report["instruct"]["layers"]
    shell_layers = [L for L, v in inst.items()
                    if v["probe_gate_clean"] and v["probe_moved"] and not v["mass_moved"]
                    and v["orth_fraction"] >= MIN_ORTH_FRACTION]
    gate_clean = [L for L, v in inst.items() if v["probe_gate_clean"]]

    print("\nVERDICT (prereg section 8, clause by clause)")
    clauses = {
        "base validation: probe sees what reaches the options": base_valid,
        "at least one instruct layer has a clean probe gate": bool(gate_clean),
        "instruct: probe moves while option mass does not, orth >= 1/3 of raw":
            bool(shell_layers),
    }
    for k, v in clauses.items():
        print("  [%s] %s" % ("ok " if v else "NO ", k))

    if not base_valid:
        verdict = "NO_INSTRUMENT"
        note = "the probe failed its end-to-end validation on the base model"
    elif not gate_clean:
        verdict = "NO_INSTRUMENT"
        note = "no instruct layer has a clean probe gate, so no null is interpretable"
    elif shell_layers:
        verdict = "SHELL"
        note = ("a decodable correlate of the injected negative state persists downstream at "
                "layer(s) %s while the option readout does not express it" % shell_layers)
    else:
        verdict = "CORE-ABSENT"
        note = ("the negative state is not decodable downstream on any gate-clean layer either, so "
                "tuning changed what the negative pole does to the representation and not merely "
                "what reaches the options")

    print("\n  VERDICT: %s" % verdict)
    print("  %s" % note)

    report["verdict"] = verdict
    report["note"] = note
    report["clauses"] = clauses
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(argv[1]).parent.parent / ("%s_shell_verdict.json" % stamp)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nWROTE %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
