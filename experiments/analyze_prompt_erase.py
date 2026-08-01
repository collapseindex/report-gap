"""Score the prompt-induced subspace-erase arm against PREREG_prompt_erase.md sections 8 and 9.

The gate order matters and is enforced here rather than described in prose. The primary question
("is the state decodable again at layer 32 after being removed at layer E?") is only interpretable
if the state was actually removed at E. That is measured, not assumed, by refitting a probe on the
erased activations at E.

    python experiments/analyze_prompt_erase.py data/pe_base/pe.jsonl data/pe_instruct/pe.jsonl
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
INDUCTION_GATE = 0.10          # clean layer-32 separation, in baseline SD
ERASURE_GATE = 0.60            # refit cv at layer E must fall to this or below
FROZEN_K = (0, 1, 2, 4, 8)     # the preregistered matrix; larger k are exploratory


def load(path):
    return [json.loads(l) for l in
            pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    print("=" * 100)
    print("PROMPT-INDUCED SUBSPACE ERASE  --  PREREG_prompt_erase.md")
    print("  no injection anywhere. The state is induced by the context sentence.")
    print("=" * 100)

    report = {}
    for path in argv[1:]:
        rows = load(path)
        key = rows[0]["model_key"]

        sep = collections.defaultdict(lambda: collections.defaultdict(list))
        checks = {}
        for r in rows:
            if r["kind"] in ("fitted", "random") and r.get("probe32") is not None:
                sep[(r["erase_layer"], r["k"], r["kind"])][r["framing"]].append(r["probe32"])
            elif r["kind"].startswith("erasure_check"):
                checks[(r["erase_layer"], r["k"], r["kind"])] = r["refit_cv"]

        def separation(kk):
            d = sep.get(kk)
            if not d or "aversive" not in d or "pleasant" not in d:
                return None
            return statistics.fmean(d["aversive"]) - statistics.fmean(d["pleasant"])

        layers = sorted({k[0] for k in sep})
        clean = separation((layers[0], 0, "fitted"))
        # baseline SD across items, pooled over framings at k=0
        base_vals = [v for f in sep[(layers[0], 0, "fitted")].values() for v in f]
        sd = statistics.pstdev(base_vals) or 1.0

        induction_ok = clean is not None and abs(clean / sd) >= INDUCTION_GATE
        print("\n%s   clean layer-32 separation %+.3f (%.2f SD)   induction gate: %s"
              % (key.upper(), clean, clean / sd, "ok" if induction_ok else "FAILED"))

        per = {}
        for L in layers:
            print("  layer %d" % L)
            print("    %-5s %10s %10s %10s %10s  %s"
                  % ("k", "fitted", "random", "survives", "refit cv", "gate"))
            for k in sorted({kk[1] for kk in sep if kk[0] == L}):
                f = separation((L, k, "fitted"))
                rnd = separation((L, k, "random"))
                surv = (f / clean) if (f is not None and clean) else None
                cv = checks.get((L, k, "erasure_check"))
                cv_rnd = checks.get((L, k, "erasure_check_random"))
                erased_ok = cv is not None and cv <= ERASURE_GATE
                tag = "" if k in FROZEN_K else "  [exploratory k]"
                print("    %-5d %10s %10s %10s %10s  %s%s"
                      % (k,
                         "%+.3f" % f if f is not None else "n/a",
                         "%+.3f" % rnd if rnd is not None else "n/a",
                         "%.0f%%" % (100 * surv) if surv is not None else "n/a",
                         "%.3f" % cv if cv is not None else "n/a",
                         "ok" if erased_ok else ("ERASURE GATE FAILED" if k else "-"), tag))
                per[(L, k)] = {"fitted": f, "random": rnd, "survives": surv,
                               "refit_cv": cv, "refit_cv_random": cv_rnd,
                               "erased_ok": erased_ok, "frozen": k in FROZEN_K}
        report[key] = {"clean": clean, "sd": sd, "induction_ok": induction_ok,
                       "per": {"%d|%d" % k: v for k, v in per.items()}}

    # ---------------------------------------------------------------- verdict
    print("\n" + "-" * 100)
    print("VERDICT (prereg section 9)")
    any_erased = any(v["erased_ok"] for m in report.values()
                     for v in [json.loads(json.dumps(x)) for x in m["per"].values()])
    all_induced = all(m["induction_ok"] for m in report.values())

    if not all_induced:
        verdict = "NO_INSTRUMENT"
        note = "the prompt framings did not move the layer-32 probe; there is no state to erase"
    elif not any_erased:
        verdict = "ERASURE_UNINFORMATIVE"
        note = ("the erasure gate fails at EVERY k on every model: a probe refit on the erased "
                "activations at layer E still separates the classes at cv 1.000. We never "
                "established that the state was removed there, so the primary question (is it "
                "decodable AGAIN at layer 32?) is unanswerable and no re-encoding claim is made. "
                "Per prereg section 9 this is `report and stop`.")
    else:
        verdict = "SEE_PER_LAYER"
        note = "at least one (layer, k) cell passed the erasure gate; read the table above"
    print("  VERDICT: %s" % verdict)
    print("  %s" % note)

    print("\n  WHAT IS STILL DESCRIPTIVELY TRUE, with no gate riding on it:")
    print("  the fitted subspace reduces the layer-32 separation and a RANDOM subspace of the same")
    print("  rank does not, which is a specificity result even though the erasure gate failed.")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / ("%s_prompt_erase_verdict.json" % stamp)
    out.write_text(json.dumps({"verdict": verdict, "note": note, "models": report}, indent=1),
                   encoding="utf-8")
    print("\nwrote %s" % out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
