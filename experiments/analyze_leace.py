"""Score the LEACE arm against PREREG_leace.md sections 8 and 9.

Gate order is the whole design. The primary ("is the state decodable again at layer 32?") is only
interpretable if the state was actually removed at layer E, and that is measured on HELD-OUT items
with an eraser fit on train, because the previous arm's version of this check could confirm itself.

    python experiments/analyze_leace.py data/leace_base/leace.jsonl data/leace_instruct/leace.jsonl
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
INDUCTION_GATE = 0.10       # SD
ERASURE_GATE = 0.60         # held-out decodability
GAP_GATE = 1e-3             # held-out class-mean gap, relative to clean


def load(path):
    return [json.loads(l) for l in
            pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def decodability(acc):
    """A probe at 0.067 is a probe at 0.933 sign-flipped. Distance from chance, never accuracy."""
    return max(acc, 1.0 - acc)


def sep_by_item(rows, layer, kind):
    """Aversive minus pleasant, paired per held-out item."""
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("what") == "probe32" and r["erase_layer"] == layer and r["kind"] == kind:
            by[r["item_index"]][r["framing"]] = r["read"]
    return [v["aversive"] - v["pleasant"] for v in by.values()
            if "aversive" in v and "pleasant" in v]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    print("=" * 100)
    print("LEACE ARM  --  PREREG_leace.md")
    print("  eraser fit on TRAIN topics; every endpoint read on HELD-OUT topics")
    print("=" * 100)

    report, verdicts = {}, {}
    for path in argv[1:]:
        rows = load(path)
        key = rows[0]["model_key"]
        layers = sorted({r["erase_layer"] for r in rows})
        checks = {(r["erase_layer"], r["kind"]): r for r in rows
                  if r.get("what") == "erasure_check"}

        clean = sep_by_item(rows, layers[0], "clean")
        sd = statistics.pstdev(clean) or 1.0
        clean_iv = A.paired_bootstrap([c / sd for c in clean]) if len(clean) > 1 else None
        induction_ok = clean_iv is not None and abs(clean_iv.point) >= INDUCTION_GATE
        print("\n%s   held-out n=%d   clean layer-32 separation %s SD   induction: %s"
              % (key.upper(), len(clean), clean_iv, "ok" if induction_ok else "FAILED"))

        per = {}
        for L in layers:
            c_clean = checks.get((L, "clean"))
            c_leace = checks.get((L, "leace"))
            c_rand = checks.get((L, "random"))
            if not (c_clean and c_leace):
                continue

            dec_clean = decodability(c_clean["refit_cv"])
            dec_leace = decodability(c_leace["refit_cv"])
            gap_ratio = c_leace["gap_heldout"] / max(1e-12, c_clean["gap_heldout"])
            erased_ok = dec_leace <= ERASURE_GATE

            s_clean = sep_by_item(rows, L, "clean")
            s_leace = sep_by_item(rows, L, "leace")
            s_rand = sep_by_item(rows, L, "random")
            iv_l = A.paired_bootstrap([v / sd for v in s_leace]) if len(s_leace) > 1 else None
            iv_r = A.paired_bootstrap([v / sd for v in s_rand]) if len(s_rand) > 1 else None
            base = statistics.fmean(s_clean) if s_clean else 0.0
            surv_l = statistics.fmean(s_leace) / base if base and s_leace else None
            surv_r = statistics.fmean(s_rand) / base if base and s_rand else None
            artifact_ok = (surv_r is not None and surv_l is not None and surv_r > surv_l)

            print("\n  layer %d" % L)
            print("    erasure check (HELD-OUT): clean decodability %.3f -> LEACE %.3f   %s"
                  % (dec_clean, dec_leace, "PASSES" if erased_ok else "GATE FAILED"))
            print("    held-out class-mean gap: %.3e of clean (%.3e -> %.3e)"
                  % (gap_ratio, c_clean["gap_heldout"], c_leace["gap_heldout"]))
            print("    layer-32 separation, held-out:")
            print("      clean  %s   (%.0f%%)" % (clean_iv, 100.0))
            print("      LEACE  %s   (%.0f%% survives)" % (iv_l, 100 * (surv_l or 0)))
            print("      random %s   (%.0f%% survives)  rank-matched  %s"
                  % (iv_r, 100 * (surv_r or 0), "ok" if artifact_ok else "ARTIFACT GATE FAILED"))
            per[L] = {"dec_clean": dec_clean, "dec_leace": dec_leace, "gap_ratio": gap_ratio,
                      "erased_ok": erased_ok, "artifact_ok": artifact_ok,
                      "survive_leace": surv_l, "survive_random": surv_r,
                      "leace_iv": str(iv_l), "random_iv": str(iv_r), "clean_iv": str(clean_iv)}

        report[key] = {"induction_ok": induction_ok, "clean_sd": sd, "n_heldout": len(clean),
                       "layers": {str(k): v for k, v in per.items()}}

        # ---- verdict per model, prereg section 9 ----
        if not induction_ok:
            v, note = "NO_INSTRUMENT", "the framings do not separate on held-out topics"
        else:
            clean_layers = [L for L, d in per.items() if d["erased_ok"] and d["artifact_ok"]]
            if not clean_layers:
                if not any(d["erased_ok"] for d in per.values()):
                    v, note = "ERASURE_UNINFORMATIVE", (
                        "LEACE did not erase on held-out real activations despite the synthetic "
                        "guarantee. Report and stop; the guarantee's assumptions do not hold here, "
                        "and that is itself worth stating.")
                else:
                    v, note = "ARTIFACT", (
                        "the rank-matched random eraser reduces layer-32 separation as much as "
                        "LEACE, so the reduction is about perturbation, not the concept")
            else:
                surv = [per[L]["survive_leace"] for L in clean_layers]
                if max(surv) >= 0.10:
                    v, note = "RE-ENCODED", (
                        "the state is provably gone at layer E on held-out items and readable "
                        "again at layer 32 (%s of the clean separation). This is the claim the "
                        "previous two arms could not reach."
                        % ", ".join("%.0f%%" % (100 * s) for s in surv))
                else:
                    v, note = "ERASED", (
                        "the state does not survive a guaranteed linear erasure (%s remaining). "
                        "The erase arm's narrowing tightens further."
                        % ", ".join("%.0f%%" % (100 * s) for s in surv))
        verdicts[key] = (v, note)
        print("\n  VERDICT (%s): %s\n    %s" % (key, v, note))

    print("\n" + "-" * 100)
    print("A failed erasure gate is not a null. An unexamined layer is not a clean one.")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / ("%s_leace_verdict.json" % stamp)
    out.write_text(json.dumps({"verdicts": {k: list(v) for k, v in verdicts.items()},
                               "models": report}, indent=2), encoding="utf-8")
    print("wrote %s" % out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
